# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""Benchmark the recurrent stage of Kimi Delta Attention."""
# isort: off
import torch
import triton
import triton.testing

from utils import bootstrap_benchmark_env, ensure_save_path_exists

bootstrap_benchmark_env(__file__)

import vllm_xpu_kernels._xpu_C  # noqa: F401
from benchmark.presets import get_hardware_preset
from tests.utils import parse_args, seed_everything
from benchmark_kda import (
    MODEL_SHAPES,
    WORKLOADS,
    _bpe,
    clear_xpu_cache,
    make_inputs,
)
# isort: on


def _run_conv(kwargs):
    return torch.ops._xpu_C.kda_causal_conv1d(
        kwargs["q_proj"],
        kwargs["k_proj"],
        kwargs["v_proj"],
        kwargs["conv_state"],
        kwargs["q_conv_weight"],
        kwargs["k_conv_weight"],
        kwargs["v_conv_weight"],
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


def _run(kwargs, intermediates):
    torch.ops._xpu_C.kda_gated_delta_rule(
        kwargs["core_attn_out"],
        *intermediates,
        kwargs["raw_gate"],
        kwargs["beta"],
        kwargs["recurrent_state"],
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


def estimate_bytes_moved(kwargs):
    n_tok = kwargs["num_actual_tokens"]
    num_heads = kwargs["num_heads"]
    head_dim = kwargs["head_dim"]
    bpe = _bpe(kwargs["q_proj"].dtype)
    batch_size = max(
        1,
        kwargs["num_prefills"]
        + kwargs["num_decodes"]
        + kwargs["num_spec_decodes"],
    )
    activations = n_tok * num_heads * (5 * head_dim * bpe + 4)
    state = batch_size * num_heads * head_dim * head_dim * 4 * 2
    parameters = num_heads * 4 + num_heads * head_dim * 4
    return activations + state + parameters


def estimate_flops(kwargs):
    return (
        kwargs["num_actual_tokens"]
        * kwargs["num_heads"]
        * (
            7 * kwargs["head_dim"] * kwargs["head_dim"]
            + 2 * kwargs["head_dim"]
        )
    )


def benchmark_kda_gated_delta_rule(
    shape_name, workload_name, dtype_str, provider, iterations
):
    shape = next(s for s in MODEL_SHAPES if s.name == shape_name)
    workload = next(w for w in WORKLOADS if w.name == workload_name)
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16
    assert iterations > 5
    kwargs = make_inputs(shape, workload, dtype)
    intermediates = _run_conv(kwargs)

    for _ in range(5):
        _run(kwargs, intermediates)
    torch.xpu.synchronize()
    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    start_event.record()
    for _ in range(5, iterations):
        _run(kwargs, intermediates)
    end_event.record()
    torch.xpu.synchronize()
    ms = start_event.elapsed_time(end_event) / (iterations - 5)

    if provider == "delta":
        result = 1000 * ms
    elif provider == "delta_memBandwidth":
        result = (estimate_bytes_moved(kwargs) / 1e9) / (ms / 1000)
    elif provider == "delta_MBU":
        preset = get_hardware_preset(torch.xpu.get_device_name())
        result = (
            float("nan")
            if preset is None
            else (
                (estimate_bytes_moved(kwargs) / 1e9)
                / (ms / 1000)
                / preset["memory_bandwidth_GBs"]
                * 100
            )
        )
    elif provider == "delta_TFLOPS":
        result = estimate_flops(kwargs) / (ms / 1000) / 1e12
    else:
        raise ValueError(f"Unknown provider {provider}")
    clear_xpu_cache()
    return result


def get_benchmark(configs, iterations=20):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["shape_name", "workload_name", "dtype_str"],
            x_vals=[tuple(config) for config in configs],
            line_arg="provider",
            line_vals=[
                "delta",
                "delta_memBandwidth",
                "delta_MBU",
                "delta_TFLOPS",
            ],
            line_names=[
                "KDA GatedDeltaRule(us)",
                "KDA GatedDeltaRule_memBandwidth(GB/s)",
                "KDA GatedDeltaRule_MBU(%)",
                "KDA GatedDeltaRule_TFLOPS",
            ],
            styles=[
                ("blue", "-"),
                ("purple", "-"),
                ("red", "-"),
                ("green", "-"),
            ],
            ylabel="Latency (us)",
            plot_name="kda-gated-delta-rule",
            args={},
        )
    )
    def benchmark(shape_name, workload_name, dtype_str, provider):
        return benchmark_kda_gated_delta_rule(
            shape_name,
            workload_name,
            dtype_str,
            provider,
            iterations,
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
    torch.set_default_device("xpu")
    torch.xpu.set_device("xpu:0")
    benchmark = get_benchmark(gen_perf_configs(), iterations=30)
    benchmark.run(
        print_data=True,
        save_path=ensure_save_path_exists(args.save_path),
    )
