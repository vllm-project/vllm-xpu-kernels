# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""
Benchmark `torch.ops._xpu_C.kda_attention` (Kimi Delta Attention).

The op fuses the three causal conv1d streams with the KDA recurrence and
updates both conv_state and recurrent_state.

Workload modes are shared with benchmark_gdn_attn.py:
    - prefill : every sequence is a prefill chunk
    - decode  : every sequence is exactly one new token
    - mix     : a batch containing prefill and decode sequences
    - spec    : MTP / speculative decode

Output format matches the other GDN benchmarks
(triton.testing.perf_report -> nightly picks up the generated CSV).
"""
# isort: off
import gc
import random
from dataclasses import dataclass

import torch
import triton
import triton.testing

from utils import bootstrap_benchmark_env, ensure_save_path_exists

bootstrap_benchmark_env(__file__)

import vllm_xpu_kernels._xpu_C  # noqa: F401
from benchmark.presets import get_hardware_preset
from tests.utils import parse_args, seed_everything
from benchmark_gdn_attn import WORKLOADS, _bpe
# isort: on

DEVICE = "xpu"


def clear_xpu_cache():
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    gc.collect()


# ----------------------------------------------------------------------------
# Model shape presets
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class KdaShape:
    name: str
    num_heads: int
    head_dim: int
    conv_width: int = 4
    tp_size: int = 1


MODEL_SHAPES = [
    KdaShape("Kimi-Linear_tp1", 32, 128, 4, tp_size=1),
    KdaShape("Kimi-Linear_tp2", 32, 128, 4, tp_size=2),
    KdaShape("Kimi-Linear_tp4", 32, 128, 4, tp_size=4),
    KdaShape("Kimi-Linear_tp8", 32, 128, 4, tp_size=8),
    # Synthetic shapes exercise every supported recurrent K bucket.
    KdaShape("Synthetic_8x32", 8, 32),
    KdaShape("Synthetic_8x64", 8, 64),
    KdaShape("Synthetic_8x256", 8, 256),
]


# ----------------------------------------------------------------------------
# Input construction
# ----------------------------------------------------------------------------
def _query_start(lengths):
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, dtype=torch.int32, device=DEVICE)


def _make_non_spec_metadata(workload, cache_batch_size):
    if workload.mode == "decode":
        num_prefills = 0
        num_decodes = workload.batch_size
        lengths = [1] * workload.batch_size
    elif workload.mode == "prefill":
        num_prefills = workload.batch_size
        num_decodes = 0
        lengths = [workload.seqlen] * workload.batch_size
    elif workload.mode == "mix":
        num_decodes = max(
            1, int(workload.batch_size * workload.decode_frac)
        )
        num_prefills = workload.batch_size - num_decodes
        lengths = [1] * num_decodes + [workload.seqlen] * num_prefills
    else:
        raise ValueError(f"Unsupported non-spec mode {workload.mode}")

    state_indices = torch.tensor(
        random.sample(range(cache_batch_size), workload.batch_size),
        dtype=torch.int32,
        device=DEVICE,
    )
    return dict(
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        num_spec_decodes=0,
        num_actual_tokens=sum(lengths),
        has_initial_state=(
            torch.rand(workload.batch_size, device=DEVICE) > 0.5
        ),
        non_spec_query_start_loc=_query_start(lengths),
        non_spec_token_indx=None,
        non_spec_state_indices=state_indices,
        spec_query_start_loc=None,
        spec_token_indx=None,
        spec_state_indices=None,
        num_accepted_tokens=None,
    )


def _make_spec_metadata(workload, cache_batch_size):
    tokens_per_sequence = workload.num_spec_tokens + 1
    num_actual_tokens = workload.batch_size * tokens_per_sequence
    state_indices = torch.tensor(
        random.sample(range(cache_batch_size), num_actual_tokens),
        dtype=torch.int32,
        device=DEVICE,
    ).reshape(workload.batch_size, tokens_per_sequence)
    return dict(
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=workload.batch_size,
        num_actual_tokens=num_actual_tokens,
        has_initial_state=None,
        non_spec_query_start_loc=None,
        non_spec_token_indx=None,
        non_spec_state_indices=None,
        spec_query_start_loc=torch.arange(
            0,
            num_actual_tokens + 1,
            tokens_per_sequence,
            dtype=torch.int32,
            device=DEVICE,
        ),
        spec_token_indx=torch.arange(
            num_actual_tokens, dtype=torch.int32, device=DEVICE
        ),
        spec_state_indices=state_indices,
        num_accepted_tokens=torch.randint(
            1,
            tokens_per_sequence + 1,
            (workload.batch_size,),
            dtype=torch.int32,
            device=DEVICE,
        ),
    )


def make_inputs(shape, workload, dtype):
    local_num_heads = shape.num_heads // shape.tp_size
    hidden_dim = local_num_heads * shape.head_dim
    tokens_per_spec_sequence = workload.num_spec_tokens + 1
    required_slots = (
        workload.batch_size * tokens_per_spec_sequence
        if workload.mode == "spec"
        else workload.batch_size
    )
    cache_batch_size = max(256, 2 * required_slots)

    if workload.mode == "spec":
        metadata = _make_spec_metadata(workload, cache_batch_size)
    else:
        metadata = _make_non_spec_metadata(workload, cache_batch_size)
    num_actual_tokens = metadata["num_actual_tokens"]

    projections = tuple(
        (torch.randn(num_actual_tokens, hidden_dim, device=DEVICE) * 0.2).to(
            dtype
        )
        for _ in range(3)
    )
    conv_weights = tuple(
        (
            torch.randn(
                hidden_dim,
                shape.conv_width,
                dtype=torch.float32,
                device=DEVICE,
            )
            * 0.1
        )
        for _ in range(3)
    )

    return dict(
        core_attn_out=torch.empty(
            1,
            num_actual_tokens,
            local_num_heads,
            shape.head_dim,
            dtype=dtype,
            device=DEVICE,
        ),
        q_proj=projections[0],
        k_proj=projections[1],
        v_proj=projections[2],
        raw_gate=(
            torch.randn(
                1,
                num_actual_tokens,
                local_num_heads,
                shape.head_dim,
                device=DEVICE,
            )
            * 0.2
        ).to(dtype),
        beta=torch.randn(
            1,
            num_actual_tokens,
            local_num_heads,
            dtype=torch.float32,
            device=DEVICE,
        ).sigmoid(),
        conv_state=(
            torch.randn(
                cache_batch_size,
                3 * hidden_dim,
                shape.conv_width - 1,
                device=DEVICE,
            )
            * 0.1
        ).to(dtype),
        recurrent_state=(
            torch.randn(
                cache_batch_size,
                local_num_heads,
                shape.head_dim,
                shape.head_dim,
                dtype=torch.float32,
                device=DEVICE,
            )
            * 0.05
        ),
        q_conv_weight=conv_weights[0],
        k_conv_weight=conv_weights[1],
        v_conv_weight=conv_weights[2],
        a_log=(
            torch.randn(
                1,
                1,
                local_num_heads,
                1,
                dtype=torch.float32,
                device=DEVICE,
            )
            * 0.1
        ),
        dt_bias=(
            torch.randn(
                hidden_dim, dtype=torch.float32, device=DEVICE
            )
            * 0.1
        ),
        num_heads=local_num_heads,
        head_dim=shape.head_dim,
        conv_width=shape.conv_width,
        cache_batch_size=cache_batch_size,
        **metadata,
    )


def _run_kda(kwargs):
    torch.ops._xpu_C.kda_attention(
        kwargs["core_attn_out"],
        kwargs["q_proj"],
        kwargs["k_proj"],
        kwargs["v_proj"],
        kwargs["raw_gate"],
        kwargs["beta"],
        kwargs["conv_state"],
        kwargs["recurrent_state"],
        kwargs["q_conv_weight"],
        kwargs["k_conv_weight"],
        kwargs["v_conv_weight"],
        kwargs["a_log"],
        kwargs["dt_bias"],
        kwargs["num_prefills"],
        kwargs["num_decodes"],
        kwargs["num_spec_decodes"],
        kwargs["has_initial_state"],
        kwargs["non_spec_query_start_loc"],
        kwargs["non_spec_token_indx"],
        kwargs["non_spec_state_indices"],
        kwargs["spec_query_start_loc"],
        kwargs["spec_token_indx"],
        kwargs["spec_state_indices"],
        kwargs["num_accepted_tokens"],
        kwargs["num_actual_tokens"],
    )


# ----------------------------------------------------------------------------
# Memory / FLOPs models
# ----------------------------------------------------------------------------
def estimate_bytes_moved(kwargs):
    """Estimate logical bytes moved by the fused conv and recurrence."""
    n_tok = kwargs["num_actual_tokens"]
    num_heads = kwargs["num_heads"]
    head_dim = kwargs["head_dim"]
    hidden_dim = num_heads * head_dim
    width = kwargs["conv_width"]
    bpe_activation = _bpe(kwargs["q_proj"].dtype)

    batch_size = max(
        1,
        kwargs["num_prefills"]
        + kwargs["num_decodes"]
        + kwargs["num_spec_decodes"],
    )
    bytes_inputs = n_tok * (
        4 * hidden_dim * bpe_activation + num_heads * 4
    )
    bytes_output = n_tok * hidden_dim * bpe_activation
    # The conv writes q/k/v intermediates and recurrence reads them back.
    bytes_intermediates = 6 * n_tok * hidden_dim * bpe_activation
    bytes_conv_state = (
        batch_size
        * 3
        * hidden_dim
        * (width - 1)
        * bpe_activation
        * 2
    )
    bytes_recurrent_state = (
        batch_size * num_heads * head_dim * head_dim * 4 * 2
    )
    bytes_weights = (
        3 * hidden_dim * width * 4
        + num_heads * 4
        + hidden_dim * 4
    )
    return (
        bytes_inputs
        + bytes_output
        + bytes_intermediates
        + bytes_conv_state
        + bytes_recurrent_state
        + bytes_weights
    )


def estimate_flops(kwargs):
    """Estimate dominant conv and recurrent KDA FLOPs."""
    n_tok = kwargs["num_actual_tokens"]
    num_heads = kwargs["num_heads"]
    head_dim = kwargs["head_dim"]
    hidden_dim = num_heads * head_dim
    width = kwargs["conv_width"]

    conv_flops = 2 * n_tok * 3 * hidden_dim * width
    recurrent_flops = n_tok * num_heads * (
        7 * head_dim * head_dim + 2 * head_dim
    )
    return conv_flops + recurrent_flops


# ----------------------------------------------------------------------------
# Benchmark driver
# ----------------------------------------------------------------------------
def benchmark_kda(shape_name, workload_name, dtype_str, provider, iterations):
    shape = next(s for s in MODEL_SHAPES if s.name == shape_name)
    workload = next(w for w in WORKLOADS if w.name == workload_name)
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16

    print(
        f"Running config: shape={shape.name}, workload={workload.name}, "
        f"dtype={dtype_str}, Provider: {provider}",
        flush=True,
    )
    assert iterations > 5, (
        "Number of iterations should be greater than 5 to account for warmup"
    )

    kwargs = make_inputs(shape, workload, dtype)
    for _ in range(5):
        _run_kda(kwargs)
    torch.xpu.synchronize()

    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    start_event.record()
    for _ in range(5, iterations):
        _run_kda(kwargs)
    end_event.record()
    torch.xpu.synchronize()
    ms = start_event.elapsed_time(end_event) / (iterations - 5)

    if provider == "kda":
        clear_xpu_cache()
        return 1000 * ms
    if provider == "kda_memBandwidth":
        bytes_moved = estimate_bytes_moved(kwargs)
        clear_xpu_cache()
        return (bytes_moved / 1e9) / (ms / 1000)
    if provider == "kda_MBU":
        hardware_presets = get_hardware_preset(torch.xpu.get_device_name())
        if hardware_presets is None:
            clear_xpu_cache()
            return float("nan")
        peak_bw = hardware_presets["memory_bandwidth_GBs"]
        bytes_moved = estimate_bytes_moved(kwargs)
        bandwidth = (bytes_moved / 1e9) / (ms / 1000)
        clear_xpu_cache()
        return (bandwidth / peak_bw) * 100
    if provider == "kda_TFLOPS":
        flops = estimate_flops(kwargs)
        clear_xpu_cache()
        return flops / (ms / 1000) / 1e12
    raise ValueError(f"Unknown provider {provider}")


def get_benchmark(configs, iterations=20):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["shape_name", "workload_name", "dtype_str"],
            x_vals=[tuple(c) for c in configs],
            line_arg="provider",
            line_vals=[
                "kda",
                "kda_memBandwidth",
                "kda_MBU",
                "kda_TFLOPS",
            ],
            line_names=[
                "KDA(us)",
                "KDA_memBandwidth(GB/s)",
                "KDA_MBU (%)",
                "KDA_TFLOPS",
            ],
            styles=[
                ("blue", "-"),
                ("purple", "-"),
                ("red", "-"),
                ("green", "-"),
            ],
            ylabel="Latency (us)",
            plot_name="kda-attn",
            args={},
        )
    )
    def benchmark(shape_name, workload_name, dtype_str, provider):
        return benchmark_kda(
            shape_name=shape_name,
            workload_name=workload_name,
            dtype_str=dtype_str,
            provider=provider,
            iterations=iterations,
        )

    return benchmark


def gen_perf_configs(dtype_str="bf16"):
    return [
        (shape.name, workload.name, dtype_str)
        for shape in MODEL_SHAPES
        for workload in WORKLOADS
    ]


if __name__ == "__main__":
    args = parse_args()
    seed_everything(1234)
    iterations = 30
    torch.set_default_device("xpu")
    torch.xpu.set_device("xpu:0")

    configs = gen_perf_configs("bf16")
    benchmark = get_benchmark(configs, iterations=iterations)
    save_path = ensure_save_path_exists(args.save_path)
    benchmark.run(print_data=True, save_path=save_path)
