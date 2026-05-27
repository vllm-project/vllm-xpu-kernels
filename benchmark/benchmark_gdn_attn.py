# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""
Benchmark `torch.ops._xpu_C.gdn_attention` (Gated DeltaNet linear attention,
used by Qwen3-Next).

Algorithm path is described in csrc/xpu/gdn_attn/gdn_attn_interface.cpp:
    causal_conv1d(qkv) + gated_delta_rule(SSM recurrence).

Workload modes:
    - prefill : every sequence is a prefill chunk (varlen)  -> XE2 chunked path
    - decode  : every sequence is exactly 1 new token       -> native path
    - mix     : mix of prefill + decode                     -> XE2 chunked path
    - spec    : MTP / speculative decode (drafts per seq)   -> native path

Output format matches benchmark_cutlass_flash_attn_decode.py
(triton.testing.perf_report -> nightly picks up the auto-generated CSV).
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
# isort: on

DEVICE = "xpu"


def clear_xpu_cache():
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    gc.collect()


# ----------------------------------------------------------------------------
# Model shape presets
# ----------------------------------------------------------------------------
# Qwen3-Next is the canonical GDN model. The two published variants
# (80B-A3B-Instruct / Thinking) share the same linear-attention shape:
# linear_num_key_heads=16, linear_num_value_heads=32,
# linear_key_head_dim=128, linear_value_head_dim=128, conv_kernel_size=4.
@dataclass(frozen=True)
class GdnShape:
    name: str
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_width: int = 4
    tp_size: int = 1


MODEL_SHAPES = [
    GdnShape("Qwen3-Next-80B_tp1", 16, 32, 128, 128, 4, tp_size=1),
    GdnShape("Qwen3-Next-80B_tp2", 16, 32, 128, 128, 4, tp_size=2),
    GdnShape("Qwen3-Next-80B_tp4", 16, 32, 128, 128, 4, tp_size=4),
    GdnShape("Qwen3-Next-80B_tp8", 16, 32, 128, 128, 4, tp_size=8),
    # Synthetic shapes to stress wider / larger configurations.
    GdnShape("Synthetic_MHA_16x16",  16, 16, 128, 128, 4),
    GdnShape("Synthetic_16x64x128",  16, 64, 128, 128, 4),
    GdnShape("Synthetic_16x32x256",  16, 32, 256, 256, 4),
]


# ----------------------------------------------------------------------------
# Workload presets
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Workload:
    name: str
    mode: str            # "prefill" | "decode" | "mix" | "spec"
    batch_size: int
    seqlen: int          # per-seq length for prefill / spec; ignored for decode
    decode_frac: float = 0.0  # only for "mix"
    num_spec_tokens: int = 0  # only for "spec"  (drafts per seq)


WORKLOADS = [
    # ---- pure decode (native path) ----
    Workload("decode_b1",         "decode",     1,    1),
    Workload("decode_b8",         "decode",     8,    1),
    Workload("decode_b32",        "decode",    32,    1),
    Workload("decode_b128",       "decode",   128,    1),
    Workload("decode_b256",       "decode",   256,    1),
    # ---- pure prefill (XE2 chunked path) ----
    Workload("prefill_b1_1k",     "prefill",    1, 1024),
    Workload("prefill_b1_4k",     "prefill",    1, 4096),
    Workload("prefill_b1_8k",     "prefill",    1, 8192),
    Workload("prefill_b4_2k",     "prefill",    4, 2048),
    Workload("prefill_b8_1k",     "prefill",    8, 1024),
    Workload("prefill_b16_512",   "prefill",   16,  512),
    Workload("prefill_b32_256",   "prefill",   32,  256),
    # ---- mixed prefill + decode (XE2 chunked path) ----
    Workload("mix_b32_1k_d050",   "mix",       32, 1024, decode_frac=0.50),
    Workload("mix_b64_512_d075",  "mix",       64,  512, decode_frac=0.75),
    Workload("mix_b16_2k_d025",   "mix",       16, 2048, decode_frac=0.25),
    # ---- speculative decode (MTP, native path) ----
    Workload("spec_b16_mtp1",     "spec",      16,    1, num_spec_tokens=1),
    Workload("spec_b32_mtp1",     "spec",      32,    1, num_spec_tokens=1),
    Workload("spec_b64_mtp1",     "spec",      64,    1, num_spec_tokens=1),
    Workload("spec_b16_mtp3",     "spec",      16,    1, num_spec_tokens=3),
]


# ----------------------------------------------------------------------------
# Input construction (mirrors tests/gdn_attn/test_gdn_attn.py)
# ----------------------------------------------------------------------------
def _simple_random_distribute(num_tokens: int, batch_size: int):
    dist = torch.ones([batch_size], dtype=torch.int64)
    for _ in range(num_tokens - batch_size):
        dist[random.randint(0, batch_size - 1)] += 1
    return dist


def _make_non_spec_inputs(shape: GdnShape, workload: Workload, dtype):
    """Build inputs for the non-spec (prefill+decode) path."""
    num_k_heads = shape.num_k_heads
    num_v_heads = shape.num_v_heads
    head_k_dim = shape.head_k_dim
    head_v_dim = shape.head_v_dim
    tp_size = shape.tp_size
    width = shape.conv_width

    if workload.mode == "decode":
        num_decodes = workload.batch_size
        num_prefills = 0
        num_actual_tokens = num_decodes
        per_seq = torch.ones(workload.batch_size, dtype=torch.int64)
    elif workload.mode == "prefill":
        num_decodes = 0
        num_prefills = workload.batch_size
        num_actual_tokens = workload.batch_size * workload.seqlen
        per_seq = torch.full((workload.batch_size, ), workload.seqlen,
                             dtype=torch.int64)
    elif workload.mode == "mix":
        num_decodes = max(1, int(workload.batch_size * workload.decode_frac))
        num_prefills = workload.batch_size - num_decodes
        prefill_tokens = num_prefills * workload.seqlen
        num_actual_tokens = num_decodes + prefill_tokens
        prefill_dist = _simple_random_distribute(prefill_tokens, num_prefills)
        per_seq = torch.cat([torch.ones(num_decodes, dtype=torch.int64),
                             prefill_dist])
    else:
        raise ValueError(f"Unsupported non-spec mode {workload.mode}")

    cache_batch_size = max(256, workload.batch_size * 2)

    mixed_qkvz_size = (num_k_heads // tp_size *
                       (2 * head_k_dim +
                        2 * head_v_dim * num_v_heads // num_k_heads))
    mixed_ba_size = num_k_heads // tp_size * (2 * num_v_heads // num_k_heads)
    mixed_qkv_size = (num_k_heads // tp_size *
                      (2 * head_k_dim +
                       head_v_dim * num_v_heads // num_k_heads))

    projected_states_qkvz = torch.randn(
        (num_actual_tokens, mixed_qkvz_size), dtype=dtype, device=DEVICE)
    projected_states_ba = torch.randn(
        (num_actual_tokens, mixed_ba_size), dtype=dtype, device=DEVICE)
    conv_state = torch.randn(
        (cache_batch_size, width - 1, mixed_qkv_size),
        dtype=dtype, device=DEVICE)
    ssm_state = torch.randn(
        (cache_batch_size, num_v_heads // tp_size, head_v_dim, head_k_dim),
        dtype=dtype, device=DEVICE)
    conv_weights = torch.randn(
        (mixed_qkv_size, width), dtype=dtype, device=DEVICE)
    conv_bias = torch.randn((mixed_qkv_size, ), dtype=dtype, device=DEVICE)
    A_log = torch.randn(
        (num_v_heads // tp_size, ), dtype=torch.float32, device=DEVICE)
    dt_bias = torch.randn(
        (num_v_heads // tp_size, ), dtype=dtype, device=DEVICE)

    non_spec_query_start_loc = torch.cat([
        torch.zeros(1, dtype=torch.int64),
        torch.cumsum(per_seq, dim=0)
    ]).to(torch.int32).to(DEVICE)
    has_initial_state = (
        torch.rand(workload.batch_size, device=DEVICE) > 0.5)
    non_spec_state_indices_tensor = torch.tensor(
        random.sample(range(cache_batch_size), workload.batch_size),
        device=DEVICE, dtype=torch.int32)

    core_attn_out = torch.zeros(
        (num_actual_tokens, num_v_heads // tp_size, head_v_dim),
        dtype=dtype, device=DEVICE)
    z = torch.empty_like(core_attn_out)

    return dict(
        core_attn_out=core_attn_out,
        z=z,
        projected_states_qkvz=projected_states_qkvz,
        projected_states_ba=projected_states_ba,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_state=conv_state,
        ssm_state=ssm_state,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        activation="silu",
        A_log=A_log,
        dt_bias=dt_bias,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        num_spec_decodes=0,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        spec_query_start_loc=None,
        spec_token_indx=None,
        spec_state_indices_tensor=None,
        num_accepted_tokens=None,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size,
        reorder_input=False,
    )


def _make_spec_inputs(shape: GdnShape, workload: Workload, dtype):
    """Build inputs for the speculative-decode (MTP) path."""
    num_k_heads = shape.num_k_heads
    num_v_heads = shape.num_v_heads
    head_k_dim = shape.head_k_dim
    head_v_dim = shape.head_v_dim
    tp_size = shape.tp_size
    width = shape.conv_width

    num_spec_decodes = workload.batch_size
    num_spec_tokens = workload.num_spec_tokens
    tokens_per_seq = num_spec_tokens + 1
    spec_token = num_spec_decodes * tokens_per_seq
    num_actual_tokens = spec_token
    cache_batch_size = max(256, workload.batch_size * 2)

    mixed_qkvz_size = (num_k_heads // tp_size *
                       (2 * head_k_dim +
                        2 * head_v_dim * num_v_heads // num_k_heads))
    mixed_ba_size = num_k_heads // tp_size * (2 * num_v_heads // num_k_heads)
    mixed_qkv_size = (num_k_heads // tp_size *
                      (2 * head_k_dim +
                       head_v_dim * num_v_heads // num_k_heads))

    projected_states_qkvz = torch.randn(
        (num_actual_tokens, mixed_qkvz_size), dtype=dtype, device=DEVICE)
    projected_states_ba = torch.randn(
        (num_actual_tokens, mixed_ba_size), dtype=dtype, device=DEVICE)
    conv_state = torch.randn(
        (cache_batch_size, width - 1, mixed_qkv_size),
        dtype=dtype, device=DEVICE)
    ssm_state = torch.randn(
        (cache_batch_size, num_v_heads // tp_size, head_v_dim, head_k_dim),
        dtype=dtype, device=DEVICE)
    conv_weights = torch.randn(
        (mixed_qkv_size, width), dtype=dtype, device=DEVICE)
    conv_bias = torch.randn((mixed_qkv_size, ), dtype=dtype, device=DEVICE)
    A_log = torch.randn(
        (num_v_heads // tp_size, ), dtype=torch.float32, device=DEVICE)
    dt_bias = torch.randn(
        (num_v_heads // tp_size, ), dtype=dtype, device=DEVICE)

    spec_query_start_loc = torch.arange(
        0, (num_spec_decodes + 1) * tokens_per_seq, tokens_per_seq,
        dtype=torch.int32, device=DEVICE)
    spec_token_indx = torch.arange(
        0, spec_token, dtype=torch.int32, device=DEVICE)
    spec_state_indices_tensor = torch.tensor(
        random.sample(range(cache_batch_size),
                      num_spec_decodes * tokens_per_seq),
        dtype=torch.int32, device=DEVICE,
    ).reshape(num_spec_decodes, tokens_per_seq)
    num_accepted_tokens = torch.randint(
        1, tokens_per_seq + 1, (num_spec_decodes, ),
        dtype=torch.int32, device=DEVICE)

    core_attn_out = torch.zeros(
        (num_actual_tokens, num_v_heads // tp_size, head_v_dim),
        dtype=dtype, device=DEVICE)
    z = torch.empty_like(core_attn_out)

    return dict(
        core_attn_out=core_attn_out,
        z=z,
        projected_states_qkvz=projected_states_qkvz,
        projected_states_ba=projected_states_ba,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_state=conv_state,
        ssm_state=ssm_state,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        activation="silu",
        A_log=A_log,
        dt_bias=dt_bias,
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=num_spec_decodes,
        has_initial_state=None,
        non_spec_query_start_loc=None,
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=None,
        spec_query_start_loc=spec_query_start_loc,
        spec_token_indx=spec_token_indx,
        spec_state_indices_tensor=spec_state_indices_tensor,
        num_accepted_tokens=num_accepted_tokens,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size,
        reorder_input=False,
    )


def make_inputs(shape, workload, dtype):
    if workload.mode == "spec":
        return _make_spec_inputs(shape, workload, dtype)
    return _make_non_spec_inputs(shape, workload, dtype)


# ----------------------------------------------------------------------------
# Memory / FLOPs models
# ----------------------------------------------------------------------------
def _bpe(dtype):
    return torch.tensor([], dtype=dtype).element_size()


def estimate_bytes_moved(kwargs):
    """Lower-bound bytes moved by gdn_attention for BW/MBU reporting."""
    bpe_in = _bpe(kwargs["projected_states_qkvz"].dtype)
    bpe_out = _bpe(kwargs["core_attn_out"].dtype)
    bpe_ssm = _bpe(kwargs["ssm_state"].dtype)

    n_tok = kwargs["num_actual_tokens"]
    nk = kwargs["num_k_heads"] // kwargs["tp_size"]
    nv = kwargs["num_v_heads"] // kwargs["tp_size"]
    hk = kwargs["head_k_dim"]
    hv = kwargs["head_v_dim"]
    width = kwargs["conv_weights"].shape[-1]

    qkvz_per_tok = nk * (2 * hk + 2 * hv * nv // nk)
    ba_per_tok = nk * (2 * nv // nk)
    qkv_per_tok = nk * (2 * hk + hv * nv // nk)
    out_per_tok = nv * hv

    batch = max(1,
                kwargs["num_prefills"] + kwargs["num_decodes"]
                + kwargs["num_spec_decodes"])

    bytes_in = n_tok * (qkvz_per_tok + ba_per_tok) * bpe_in
    bytes_out = n_tok * (out_per_tok * 2) * bpe_out     # core_attn_out + z
    bytes_conv_state = batch * (width - 1) * qkv_per_tok * bpe_in * 2  # RW
    bytes_ssm_state = batch * nv * hk * hv * bpe_ssm * 2              # RW
    bytes_weights = qkv_per_tok * width * bpe_in + qkv_per_tok * bpe_in

    return (bytes_in + bytes_out + bytes_conv_state
            + bytes_ssm_state + bytes_weights)


def estimate_flops(kwargs):
    """Lower-bound FLOPs for the GDN op (matmuls dominate).

    Conv1d:  2 * n_tok * mixed_qkv_size * width  (per-token depthwise MAC)
    Gated delta rule per token (dominant terms after GQA expansion):
        S *= g                              :     nv*hv*hk    (mul)
        kv_mem = S @ k                      : 2 * nv*hv*hk
        delta = (v - kv_mem) * beta         : 2 * nv*hv
        S    += outer(delta, k)             : 2 * nv*hv*hk
        out  = S @ q                        : 2 * nv*hv*hk
    -> ~7 * nv * hv * hk + 2*nv*hv per token, dominant ~7*nv*hv*hk.
    """
    n_tok = kwargs["num_actual_tokens"]
    nk = kwargs["num_k_heads"] // kwargs["tp_size"]
    nv = kwargs["num_v_heads"] // kwargs["tp_size"]
    hk = kwargs["head_k_dim"]
    hv = kwargs["head_v_dim"]
    width = kwargs["conv_weights"].shape[-1]

    qkv_per_tok = nk * (2 * hk + hv * nv // nk)
    conv_flops = 2 * n_tok * qkv_per_tok * width
    sdr_flops = n_tok * (7 * nv * hv * hk + 2 * nv * hv)
    return conv_flops + sdr_flops


# ----------------------------------------------------------------------------
# Benchmark driver (mirrors flash_attn_decode pattern)
# ----------------------------------------------------------------------------
def benchmark_gdn(shape_name, workload_name, dtype_str, provider, iterations):
    shape = next(s for s in MODEL_SHAPES if s.name == shape_name)
    workload = next(w for w in WORKLOADS if w.name == workload_name)
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16

    print(f"Running config: shape={shape.name}, workload={workload.name}, "
          f"dtype={dtype_str}, Provider: {provider}", flush=True)
    assert iterations > 5, \
        "Number of iterations should be greater than 5 to account for warmup"

    kwargs = make_inputs(shape, workload, dtype)

    def _run():
        torch.ops._xpu_C.gdn_attention(**kwargs)

    # warmup
    for _ in range(5):
        _run()
    torch.xpu.synchronize()

    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    start_event.record()
    for _ in range(5, iterations):
        _run()
    end_event.record()
    torch.xpu.synchronize()
    ms = start_event.elapsed_time(end_event) / (iterations - 5)

    if provider == "gdn":
        clear_xpu_cache()
        return 1000 * ms  # us
    if provider == "gdn_memBandwidth":
        bytes_moved = estimate_bytes_moved(kwargs)
        clear_xpu_cache()
        return (bytes_moved / 1e9) / (ms / 1000)  # GB/s
    if provider == "gdn_MBU":
        hardware_presets = get_hardware_preset(torch.xpu.get_device_name())
        if hardware_presets is None:
            clear_xpu_cache()
            return float("nan")
        peak_bw = hardware_presets["memory_bandwidth_GBs"]
        bytes_moved = estimate_bytes_moved(kwargs)
        bw = (bytes_moved / 1e9) / (ms / 1000)
        clear_xpu_cache()
        return (bw / peak_bw) * 100
    if provider == "gdn_TFLOPS":
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
            line_vals=["gdn", "gdn_memBandwidth", "gdn_MBU", "gdn_TFLOPS"],
            line_names=[
                "GDN(us)",
                "GDN_memBandwidth(GB/s)",
                "GDN_MBU (%)",
                "GDN_TFLOPS",
            ],
            styles=[("blue", "-"), ("purple", "-"), ("red", "-"),
                    ("green", "-")],
            ylabel="Latency (us)",
            plot_name="gdn-attn",
            args={},
        ))
    def benchmark(shape_name, workload_name, dtype_str, provider):
        return benchmark_gdn(shape_name=shape_name,
                             workload_name=workload_name,
                             dtype_str=dtype_str,
                             provider=provider,
                             iterations=iterations)

    return benchmark


def gen_perf_configs(dtype_str="bf16"):
    return [(s.name, w.name, dtype_str)
            for s in MODEL_SHAPES for w in WORKLOADS]


if __name__ == "__main__":
    args = parse_args()
    seed = 1234
    seed_everything(seed)
    iterations = 30
    torch.set_default_device("xpu")
    torch.xpu.set_device("xpu:0")

    configs = gen_perf_configs("bf16")
    benchmark = get_benchmark(configs, iterations=iterations)
    save_path = ensure_save_path_exists(args.save_path)
    benchmark.run(print_data=True, save_path=save_path)
