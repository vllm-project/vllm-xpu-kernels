# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""
Benchmark `torch.ops._xpu_C.gated_delta_rule` (the SSM recurrence stage of
Gated DeltaNet linear attention, used by Qwen3-Next).

gated_delta_rule is the second half of the split gdn_attention op (see
csrc/xpu/gdn_attn/gdn_attn_interface.cpp). It consumes the intermediate
{q, k, v, b, a} tensors produced by causal_conv1d, advances ssm_state, and
writes core_attn_out. The conv stage is run once (outside the timing loop) to
produce the intermediates, then only gated_delta_rule is timed.

Workload modes (shared with benchmark_gdn_attn.py):
    - prefill : every sequence is a prefill chunk (varlen)  -> XE2 chunked path
    - decode  : every sequence is exactly 1 new token       -> native path
    - mix     : mix of prefill + decode                     -> XE2 chunked path
    - spec    : MTP / speculative decode (drafts per seq)   -> native path

Output format matches benchmark_gdn_attn.py
(triton.testing.perf_report -> nightly picks up the auto-generated CSV).
"""
# isort: off
import gc

import torch
import triton
import triton.testing

from utils import bootstrap_benchmark_env, ensure_save_path_exists

bootstrap_benchmark_env(__file__)

import vllm_xpu_kernels._xpu_C  # noqa: F401
from benchmark.presets import get_hardware_preset
from tests.utils import parse_args, seed_everything
# Reuse the model shapes, workloads and input construction from the fused
# gdn_attention benchmark so the two stay in lockstep.
from benchmark_gdn_attn import (
    MODEL_SHAPES,
    WORKLOADS,
    _bpe,
    make_inputs,
)
# isort: on

DEVICE = "xpu"


def clear_xpu_cache():
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    gc.collect()


def _run_conv(kwargs):
    """Run the conv stage once to produce the intermediates consumed by
    gated_delta_rule."""
    return torch.ops._xpu_C.causal_conv1d(
        kwargs["z"],
        kwargs["projected_states_qkvz"],
        kwargs["projected_states_ba"],
        kwargs["num_k_heads"],
        kwargs["num_v_heads"],
        kwargs["head_k_dim"],
        kwargs["head_v_dim"],
        conv_state=kwargs["conv_state"],
        conv_weights=kwargs["conv_weights"],
        conv_bias=kwargs["conv_bias"],
        activation=kwargs["activation"],
        num_prefills=kwargs["num_prefills"],
        num_decodes=kwargs["num_decodes"],
        num_spec_decodes=kwargs["num_spec_decodes"],
        has_initial_state=kwargs["has_initial_state"],
        non_spec_query_start_loc=kwargs["non_spec_query_start_loc"],
        non_spec_token_indx=kwargs["non_spec_token_indx"],
        non_spec_state_indices_tensor=kwargs["non_spec_state_indices_tensor"],
        spec_query_start_loc=kwargs["spec_query_start_loc"],
        spec_token_indx=kwargs["spec_token_indx"],
        spec_state_indices_tensor=kwargs["spec_state_indices_tensor"],
        num_accepted_tokens=kwargs["num_accepted_tokens"],
        num_actual_tokens=kwargs["num_actual_tokens"],
        tp_size=kwargs["tp_size"],
        reorder_input=kwargs["reorder_input"])


# ----------------------------------------------------------------------------
# Memory / FLOPs models (delta stage only)
# ----------------------------------------------------------------------------
def estimate_bytes_moved(kwargs, intermediates):
    """Lower-bound bytes moved by gated_delta_rule for BW/MBU reporting.

    The delta stage reads the {q,k,v,b,a} intermediates, reads A_log/dt_bias,
    reads/writes ssm_state, and writes core_attn_out.
    """
    bpe_out = _bpe(kwargs["core_attn_out"].dtype)
    bpe_ssm = _bpe(kwargs["ssm_state"].dtype)

    n_tok = kwargs["num_actual_tokens"]
    nv = kwargs["num_v_heads"] // kwargs["tp_size"]
    hk = kwargs["head_k_dim"]
    hv = kwargs["head_v_dim"]

    out_per_tok = nv * hv

    batch = max(1,
                kwargs["num_prefills"] + kwargs["num_decodes"]
                + kwargs["num_spec_decodes"])

    bytes_inter = sum(t.numel() * t.element_size() for t in intermediates)
    bytes_out = n_tok * out_per_tok * bpe_out                 # core_attn_out
    bytes_ssm_state = batch * nv * hk * hv * bpe_ssm * 2      # RW

    return bytes_inter + bytes_out + bytes_ssm_state


def estimate_flops(kwargs):
    """Lower-bound FLOPs for the delta stage (matmuls dominate).

    Gated delta rule per token (dominant terms after GQA expansion):
        S *= g                              :     nv*hv*hk    (mul)
        kv_mem = S @ k                      : 2 * nv*hv*hk
        delta = (v - kv_mem) * beta         : 2 * nv*hv
        S    += outer(delta, k)             : 2 * nv*hv*hk
        out  = S @ q                        : 2 * nv*hv*hk
    -> ~7 * nv * hv * hk + 2*nv*hv per token, dominant ~7*nv*hv*hk.
    """
    n_tok = kwargs["num_actual_tokens"]
    nv = kwargs["num_v_heads"] // kwargs["tp_size"]
    hk = kwargs["head_k_dim"]
    hv = kwargs["head_v_dim"]
    return n_tok * (7 * nv * hv * hk + 2 * nv * hv)


# ----------------------------------------------------------------------------
# Benchmark driver (mirrors benchmark_gdn_attn.py)
# ----------------------------------------------------------------------------
def benchmark_gated_delta_rule(shape_name, workload_name, dtype_str, provider,
                               iterations):
    shape = next(s for s in MODEL_SHAPES if s.name == shape_name)
    workload = next(w for w in WORKLOADS if w.name == workload_name)
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16

    print(f"Running config: shape={shape.name}, workload={workload.name}, "
          f"dtype={dtype_str}, Provider: {provider}", flush=True)
    assert iterations > 5, \
        "Number of iterations should be greater than 5 to account for warmup"

    kwargs = make_inputs(shape, workload, dtype)
    # Produce the intermediates once; only the delta stage is timed.
    intermediates = _run_conv(kwargs)

    def _run():
        torch.ops._xpu_C.gated_delta_rule(
            kwargs["core_attn_out"],
            intermediates,
            kwargs["num_v_heads"],
            kwargs["head_v_dim"],
            A_log=kwargs["A_log"],
            dt_bias=kwargs["dt_bias"],
            ssm_state=kwargs["ssm_state"],
            num_prefills=kwargs["num_prefills"],
            num_decodes=kwargs["num_decodes"],
            num_spec_decodes=kwargs["num_spec_decodes"],
            has_initial_state=kwargs["has_initial_state"],
            non_spec_query_start_loc=kwargs["non_spec_query_start_loc"],
            non_spec_token_indx=kwargs["non_spec_token_indx"],
            non_spec_state_indices_tensor=kwargs[
                "non_spec_state_indices_tensor"],
            spec_query_start_loc=kwargs["spec_query_start_loc"],
            spec_token_indx=kwargs["spec_token_indx"],
            spec_state_indices_tensor=kwargs["spec_state_indices_tensor"],
            num_accepted_tokens=kwargs["num_accepted_tokens"],
            num_actual_tokens=kwargs["num_actual_tokens"],
            tp_size=kwargs["tp_size"])

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

    if provider == "delta":
        clear_xpu_cache()
        return 1000 * ms  # us
    if provider == "delta_memBandwidth":
        bytes_moved = estimate_bytes_moved(kwargs, intermediates)
        clear_xpu_cache()
        return (bytes_moved / 1e9) / (ms / 1000)  # GB/s
    if provider == "delta_MBU":
        hardware_presets = get_hardware_preset(torch.xpu.get_device_name())
        if hardware_presets is None:
            clear_xpu_cache()
            return float("nan")
        peak_bw = hardware_presets["memory_bandwidth_GBs"]
        bytes_moved = estimate_bytes_moved(kwargs, intermediates)
        bw = (bytes_moved / 1e9) / (ms / 1000)
        clear_xpu_cache()
        return (bw / peak_bw) * 100
    if provider == "delta_TFLOPS":
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
                "delta",
                "delta_memBandwidth",
                "delta_MBU",
                "delta_TFLOPS",
            ],
            line_names=[
                "GatedDeltaRule(us)",
                "GatedDeltaRule_memBandwidth(GB/s)",
                "GatedDeltaRule_MBU (%)",
                "GatedDeltaRule_TFLOPS",
            ],
            styles=[("blue", "-"), ("purple", "-"), ("red", "-"),
                    ("green", "-")],
            ylabel="Latency (us)",
            plot_name="gated-delta-rule",
            args={},
        ))
    def benchmark(shape_name, workload_name, dtype_str, provider):
        return benchmark_gated_delta_rule(shape_name=shape_name,
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
