# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

# isort: off
import gc
from pathlib import Path

import torch
import triton
import triton.testing

from utils import bootstrap_benchmark_env, ensure_save_path_exists

bootstrap_benchmark_env(__file__)
from benchmark.src.fused_moe_interface_ import xpu_fused_moe_CalKernelTime
from benchmark.src.get_model_config import (
    gen_cutlass_fused_moe_correctness_configs as gen_correctness_config)
from benchmark.src.get_model_config import (gen_cutlass_fused_moe_perf_configs
                                            as gen_perf_configs)
from tests.fused_moe.test_fused_moe import ref_fused_moe
from tests.ops.fp8_quant_op import scaled_fp8_quant
from tests.utils import parse_args, seed_everything
from vllm_xpu_kernels.fused_moe_interface import xpu_fused_moe
# isort: on

DEVICE = "xpu"
DEFAULT_PEAK_FP8_TFLOPS = 197.0
DEFAULT_PEAK_BF16_TFLOPS = 98.5
DEFAULT_PEAK_BW_GBS = 456.0

def clear_xpu_cache():
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    gc.collect()


def calculate_flops(m, n, k):
    return 2 * m * n * k


def calculate_memory_usage(m, n, k, num_experts, x_dtype, w_dtype=None):
    io_memory = (m * k + m * n) * torch.tensor(
        [], dtype=x_dtype).element_size() / (1000**3)  # in GB
    weight_memory = num_experts * k * n * torch.tensor(
        [], dtype=x_dtype if w_dtype is None else w_dtype).element_size() / (
            1000**3)  # in GB
    return io_memory + weight_memory


def _row_peak_tflops(row):
    return (
        DEFAULT_PEAK_FP8_TFLOPS
        if row.get("w_dtype") is not None
        else DEFAULT_PEAK_BF16_TFLOPS
    )


def _resolve_metric_col(df, preferred, fallback_prefix):
    if preferred in df.columns:
        return preferred
    if fallback_prefix in df.columns:
        return fallback_prefix
    matches = [
        col for col in df.columns
        if str(col).startswith(fallback_prefix)
    ]
    return matches[0] if matches else None


def _append_roofline_metrics(df, tflops_col, bw_col, suffix):
    peak_tflops_col = f"peak_tflops_{suffix}"
    df[peak_tflops_col] = df.apply(_row_peak_tflops, axis=1)
    df[f"%peak_tflops_{suffix}"] = df[tflops_col] / df[peak_tflops_col] * 100.0
    df[f"%peak_bw_{suffix}"] = df[bw_col] / DEFAULT_PEAK_BW_GBS * 100.0
    df[f"AI(F/B)_{suffix}"] = df[tflops_col] * 1000.0 / df[bw_col]
    df[f"roofline_tflops_{suffix}"] = (
        df[f"AI(F/B)_{suffix}"] * DEFAULT_PEAK_BW_GBS / 1000.0
    ).clip(upper=df[peak_tflops_col])
    df[f"eff_vs_roofline(%)_{suffix}"] = (
        df[tflops_col] / df[f"roofline_tflops_{suffix}"] * 100.0
    )

    norm_ratio = (
        df[f"%peak_tflops_{suffix}"]
        / df[f"%peak_bw_{suffix}"].clip(lower=1e-6)
    )
    df[f"bound_hint_{suffix}"] = "mixed"
    df.loc[norm_ratio > 1.25, f"bound_hint_{suffix}"] = "compute-leaning"
    df.loc[norm_ratio < 0.75, f"bound_hint_{suffix}"] = "memory-leaning"


def print_roofline_analysis(df):
    analysis_df = df.copy()
    for col in ["m", "n", "k", "num_experts", "topk"]:
        if col in analysis_df.columns:
            analysis_df[col] = analysis_df[col].astype(int)

    gemm1_tflops_col = _resolve_metric_col(
        analysis_df,
        "vllm_kernel_gemm1_tflops (value)",
        "vllm_kernel_gemm1_tflops",
    )
    gemm2_tflops_col = _resolve_metric_col(
        analysis_df,
        "vllm_kernel_gemm2_tflops (value)",
        "vllm_kernel_gemm2_tflops",
    )
    gemm1_bw_col = _resolve_metric_col(
        analysis_df,
        "vllm_kernel_gemm1_memory(GB/s) (value)",
        "vllm_kernel_gemm1_memory(GB/s)",
    )
    gemm2_bw_col = _resolve_metric_col(
        analysis_df,
        "vllm_kernel_gemm2_memory(GB/s) (value)",
        "vllm_kernel_gemm2_memory(GB/s)",
    )

    required_cols = [
        gemm1_tflops_col,
        gemm2_tflops_col,
        gemm1_bw_col,
        gemm2_bw_col,
    ]
    if any(col is None for col in required_cols):
        print(
            "Warning: skip roofline metrics because required columns are "
            f"missing. columns={list(analysis_df.columns)}"
        )
        return analysis_df

    _append_roofline_metrics(analysis_df, gemm1_tflops_col, gemm1_bw_col,
                             "gemm1")
    _append_roofline_metrics(analysis_df, gemm2_tflops_col, gemm2_bw_col,
                             "gemm2")

    show_cols = [
        "m",
        "n",
        "k",
        "num_experts",
        "topk",
        "x_dtype",
        "w_dtype",
        gemm1_tflops_col,
        gemm1_bw_col,
        "%peak_tflops_gemm1",
        "%peak_bw_gemm1",
        "AI(F/B)_gemm1",
        "roofline_tflops_gemm1",
        "eff_vs_roofline(%)_gemm1",
        "bound_hint_gemm1",
        gemm2_tflops_col,
        gemm2_bw_col,
        "%peak_tflops_gemm2",
        "%peak_bw_gemm2",
        "AI(F/B)_gemm2",
        "roofline_tflops_gemm2",
        "eff_vs_roofline(%)_gemm2",
        "bound_hint_gemm2",
    ]
    show_cols = [c for c in show_cols if c in analysis_df.columns]
    print("fused_moe_roofline_analysis:")
    print(
        analysis_df[show_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    return analysis_df


def make_fused_moe_input(config):
    mnk, e, topk, x_dtype, w_dtype, has_bias = config
    m, n, k = mnk
    input_len = m
    hidden_size = k
    intermediate_size = n
    num_experts = e

    a = torch.randn(
        (input_len, hidden_size), device=DEVICE, dtype=x_dtype) / 16
    w13 = torch.randn((num_experts, 2 * intermediate_size, hidden_size),
                      device=DEVICE,
                      dtype=x_dtype) / 16
    w2 = torch.randn((num_experts, hidden_size, intermediate_size),
                     device=DEVICE,
                     dtype=x_dtype) / 16
    ref_a = a.clone()

    if has_bias:
        w13_bias = torch.randn(
            (num_experts, 2 * intermediate_size), device=DEVICE,
            dtype=x_dtype) / 16
        w2_bias = torch.randn(
            (num_experts, hidden_size), device=DEVICE, dtype=x_dtype) / 16
    else:
        w13_bias = None
        w2_bias = None
    # moe gate
    scores = torch.randn((input_len, num_experts),
                         device=DEVICE,
                         dtype=torch.float32)
    expert_scores, expert_indices = torch.topk(scores,
                                               k=topk,
                                               dim=-1,
                                               sorted=False)

    flat_expert_indices = expert_indices.view(-1)
    flat_expert_weights = expert_scores.view(-1, 1)

    if w_dtype is not None:
        w13_fp8 = torch.empty_like(w13, dtype=w_dtype)
        w2_fp8 = torch.empty_like(w2, dtype=w_dtype)

        # scale
        random_exponents = torch.randint(-3, 4, (num_experts, ), device=DEVICE)
        w13_scales = torch.pow(2.0, random_exponents.float())
        random_exponents = torch.randint(-3, 4, (num_experts, ), device=DEVICE)
        w2_scales = torch.pow(2.0, random_exponents.float())

        for i in range(num_experts):
            w13_fp8[i], _ = scaled_fp8_quant(w13[i],
                                             w13_scales[i].to(torch.float32),
                                             False,
                                             False,
                                             fp8_dtype=w_dtype)
            w2_fp8[i], _ = scaled_fp8_quant(w2[i],
                                            w2_scales[i].to(torch.float32),
                                            False,
                                            False,
                                            fp8_dtype=w_dtype)
        w13 = w13_fp8
        w2 = w2_fp8

        ref_w13 = torch.empty_like(w13_fp8, dtype=x_dtype)
        ref_w2 = torch.empty_like(w2_fp8, dtype=x_dtype)
        for i in range(num_experts):
            ref_w13[i] = w13_fp8[i].to(x_dtype) * w13_scales[i]
            ref_w2[i] = w2_fp8[i].to(x_dtype) * w2_scales[i]
    else:
        w13_scales = None
        w2_scales = None
        ref_w13 = w13
        ref_w2 = w2

    w13 = w13.transpose(-1, -2).contiguous()
    w2 = w2.transpose(-1, -2).contiguous()
    return (ref_a, ref_w13, w13_bias, ref_w2, w2_bias, flat_expert_weights,
            flat_expert_indices, a, w13, w13_scales, w2, w2_scales,
            expert_scores, expert_indices)


def calculate_diff(config):
    _, e, topk, x_dtype, w_dtype, _ = config
    ref_a, ref_w13, w13_bias, ref_w2, w2_bias, flat_expert_weights, \
        flat_expert_indices, a, w13, w13_scales, w2, w2_scales, \
            expert_scores, expert_indices = make_fused_moe_input(config)

    ref_out = ref_fused_moe(ref_a, ref_w13, w13_bias, ref_w2, w2_bias,
                            flat_expert_weights, flat_expert_indices, topk,
                            "silu", e)

    output = xpu_fused_moe(hidden_states=a,
                           w13=w13,
                           w13_scales=w13_scales,
                           w13_bias=w13_bias,
                           w2=w2,
                           w2_scales=w2_scales,
                           w2_bias=w2_bias,
                           topk_weights=expert_scores,
                           topk_ids=expert_indices,
                           n_experts_per_token=topk,
                           activation="silu",
                           num_experts=e,
                           is_fp8=(w_dtype is not None))
    if x_dtype == torch.float16:
        rtol = 1e-2
        atol = 1e-2
    else:
        rtol = 2e-2
        atol = 2e-2

    try:
        torch.testing.assert_close(output, ref_out, rtol=rtol, atol=atol)
        print("✅ All implementations match, ", config)
    except AssertionError as e:
        print("❌ Implementations differ, ", config, " error: ", e)


def get_benchmark(iterations):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=[
                "m", "n", "k", "num_experts", "topk", "x_dtype", "w_dtype",
                "has_bias"
            ],
            x_vals=[(*tuple(c)[0], *tuple(c)[1:]) for c in configs],
            line_arg="provider",
            line_vals=[
                "vllm", "vllm_kernel_remap", "vllm_kernel_gemm1",
                "vllm_kernel_gemm2", "vllm_kernel_gather",
                "vllm_kernel_gemm1_tflops", "vllm_kernel_gemm2_tflops",
                "vllm_kernel_gemm1_memory", "vllm_kernel_gemm2_memory"
            ],
            line_names=[
                "vllm(us)", "vllm_kernel_remap(us)",
                "vllm_kernel_gemm1(us)", "vllm_kernel_gemm2(us)",
                "vllm_kernel_gather(us)", "vllm_kernel_gemm1_tflops",
                "vllm_kernel_gemm2_tflops",
                "vllm_kernel_gemm1_memory(GB/s)",
                "vllm_kernel_gemm2_memory(GB/s)"
            ],
            styles=[
                ("blue", "-"), ("red", "-"), ("green", "-"),
                ("orange", "-"), ("purple", "-"), ("green", "--"),
                ("orange", "--"), ("green", ":"), ("orange", ":")],
            ylabel="Latency (us)",
            plot_name="fused_moe-cutlass",
            args={},
        ))
    def benchmark(m,
                  n,
                  k,
                  num_experts,
                  topk,
                  x_dtype,
                  w_dtype,
                  has_bias,
                  provider,
                  iterations=iterations):
        print(f"Running config: {m, n, k, num_experts, topk, \
                                x_dtype, w_dtype, \
                                has_bias}, Provider: {provider}",
              flush=True)
        total_latency = 0.0
        ms = 0.0
        assert iterations > 5, \
            "Iterations should be greater than " \
            "5 to have enough warmup iterations."

        _, _, w13_bias, _, w2_bias, _, \
            _, a, w13, w13_scales, w2, w2_scales, \
                expert_scores, expert_indices = make_fused_moe_input(
                    config=((m, n, k), num_experts,
                            topk, x_dtype, w_dtype, has_bias))

        if provider == "vllm":
            start_event = torch.xpu.Event(enable_timing=True)
            end_event = torch.xpu.Event(enable_timing=True)
            for index in range(5):
                xpu_fused_moe(hidden_states=a,
                              w13=w13,
                              w13_scales=w13_scales,
                              w13_bias=w13_bias,
                              w2=w2,
                              w2_scales=w2_scales,
                              w2_bias=w2_bias,
                              topk_weights=expert_scores,
                              topk_ids=expert_indices,
                              n_experts_per_token=topk,
                              activation="silu",
                              num_experts=num_experts,
                              is_fp8=(w_dtype is not None))
            start_event.record()
            for index in range(5, iterations):
                xpu_fused_moe(hidden_states=a,
                              w13=w13,
                              w13_scales=w13_scales,
                              w13_bias=w13_bias,
                              w2=w2,
                              w2_scales=w2_scales,
                              w2_bias=w2_bias,
                              topk_weights=expert_scores,
                              topk_ids=expert_indices,
                              n_experts_per_token=topk,
                              activation="silu",
                              num_experts=num_experts,
                              is_fp8=(w_dtype is not None))
            end_event.record()
            torch.xpu.synchronize()
            total_latency = start_event.elapsed_time(end_event)
        else:
            n_measured = iterations - 5
            remap_se = [
                torch.xpu.Event(enable_timing=True) for _ in range(n_measured)
            ]
            remap_ee = [
                torch.xpu.Event(enable_timing=True) for _ in range(n_measured)
            ]
            gemm1_se = [
                torch.xpu.Event(enable_timing=True) for _ in range(n_measured)
            ]
            gemm1_ee = [
                torch.xpu.Event(enable_timing=True) for _ in range(n_measured)
            ]
            gemm2_se = [
                torch.xpu.Event(enable_timing=True) for _ in range(n_measured)
            ]
            gemm2_ee = [
                torch.xpu.Event(enable_timing=True) for _ in range(n_measured)
            ]
            gather_se = [
                torch.xpu.Event(enable_timing=True) for _ in range(n_measured)
            ]
            gather_ee = [
                torch.xpu.Event(enable_timing=True) for _ in range(n_measured)
            ]
            gemm1_info = gemm2_info = None
            for index in range(iterations):
                i = index - 5 if index >= 5 else None
                cur_gemm1_info, cur_gemm2_info = xpu_fused_moe_CalKernelTime(
                    hidden_states=a,
                    w13=w13,
                    w13_scales=w13_scales,
                    w13_bias=w13_bias,
                    w2=w2,
                    w2_scales=w2_scales,
                    w2_bias=w2_bias,
                    topk_weights=expert_scores,
                    topk_ids=expert_indices,
                    n_experts_per_token=topk,
                    activation="silu",
                    num_experts=num_experts,
                    is_fp8=(w_dtype is not None),
                    start_event_remap=remap_se[i] if i is not None else None,
                    end_event_remap=remap_ee[i] if i is not None else None,
                    start_event_gemm1=gemm1_se[i] if i is not None else None,
                    end_event_gemm1=gemm1_ee[i] if i is not None else None,
                    start_event_gemm2=gemm2_se[i] if i is not None else None,
                    end_event_gemm2=gemm2_ee[i] if i is not None else None,
                    start_event_gather=gather_se[i] if i is not None else None,
                    end_event_gather=gather_ee[i] if i is not None else None,
                )
                if index == 5:
                    gemm1_info = cur_gemm1_info
                    gemm2_info = cur_gemm2_info
            torch.xpu.synchronize()
            remap_latency = sum(
                remap_se[i].elapsed_time(remap_ee[i])
                for i in range(n_measured)
            )
            gemm1_latency = sum(
                gemm1_se[i].elapsed_time(gemm1_ee[i])
                for i in range(n_measured)
            )
            gemm2_latency = sum(
                gemm2_se[i].elapsed_time(gemm2_ee[i])
                for i in range(n_measured)
            )
            gather_latency = sum(
                gather_se[i].elapsed_time(gather_ee[i])
                for i in range(n_measured)
            )
            gemm1_m, gemm1_n, gemm1_k, gemm1_expert = gemm1_info
            gemm2_m, gemm2_n, gemm2_k, gemm2_expert = gemm2_info
            if provider == "vllm_kernel_remap":
                total_latency = remap_latency
            elif provider in ("vllm_kernel_gemm1", "vllm_kernel_gemm1_tflops",
                              "vllm_kernel_gemm1_memory"):
                total_latency = gemm1_latency
                m, n, k, active_experts = (gemm1_m, gemm1_n, gemm1_k,
                                           gemm1_expert)
            elif provider in ("vllm_kernel_gemm2", "vllm_kernel_gemm2_tflops",
                              "vllm_kernel_gemm2_memory"):
                total_latency = gemm2_latency
                m, n, k, active_experts = (gemm2_m, gemm2_n, gemm2_k,
                                           gemm2_expert)
            elif provider == "vllm_kernel_gather":
                total_latency = gather_latency
            if provider in ("vllm_kernel_gemm1_tflops",
                            "vllm_kernel_gemm2_tflops"):
                ms = total_latency / n_measured
                clear_xpu_cache()
                flops = calculate_flops(m, n, k)
                return flops / (ms / 1000) / 1e12
            if provider in ("vllm_kernel_gemm1_memory",
                            "vllm_kernel_gemm2_memory"):
                ms = total_latency / n_measured
                clear_xpu_cache()
                memory_usage_GB = calculate_memory_usage(
                    m, n, k, active_experts, x_dtype, w_dtype)
                return memory_usage_GB / (ms / 1000)  # GB/s

        torch.xpu.synchronize()
        ms = total_latency / (iterations - 5)
        clear_xpu_cache()
        return 1000 * ms

    return benchmark


if __name__ == "__main__":

    args = parse_args()
    seed = 1234
    seed_everything(seed)
    iterations = 20

    configs = gen_correctness_config()

    for config in configs:
        try:
            calculate_diff(config)
        except Exception as e:
            print("Error in config: ", config, " error: ", e)
        clear_xpu_cache()

    configs = gen_perf_configs()
    benchmark = get_benchmark(iterations=iterations)
    save_path = ensure_save_path_exists(args.save_path)
    # Run performance benchmark
    perf_df = benchmark.run(
        print_data=False,
        save_path=save_path,
        return_df=True,
    )
    analysis_df = print_roofline_analysis(perf_df)
    bench_cfg = getattr(benchmark, "benchmarks", None)
    bench_plot_name = getattr(bench_cfg, "plot_name", None)
    if bench_plot_name is not None:
        csv_path = Path(save_path) / f"{bench_plot_name}.csv"
        analysis_df.to_csv(csv_path, index=False, float_format="%.6f")
