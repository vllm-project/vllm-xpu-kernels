# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

# isort: off
import gc
import statistics

import torch
import triton
import triton.testing

from utils import bootstrap_benchmark_env, ensure_save_path_exists

bootstrap_benchmark_env(__file__)
from benchmark.src.fused_moe_interface_ import XpuFusedMoe_CalKernelTime
from benchmark.src.get_model_config import (
    gen_cutlass_fused_moe_correctness_configs as gen_correctness_config)
from benchmark.src.get_model_config import (gen_cutlass_fused_moe_perf_configs
                                            as gen_perf_configs)
from tests.fused_moe.test_fused_moe import ref_fused_moe
from tests.ops.fp8_quant_op import scaled_fp8_quant
from tests.utils import parse_args, seed_everything
from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe
# isort: on

DEVICE = "xpu"

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

    moe = XpuFusedMoe(w13=w13,
                      w13_scales=w13_scales,
                      w13_bias=w13_bias,
                      w2=w2,
                      w2_scales=w2_scales,
                      w2_bias=w2_bias,
                      n_experts_per_token=topk,
                      activation="silu",
                      num_experts=e)
    output = torch.empty_like(a)
    moe.apply(output=output,
              hidden_states=a,
              topk_weights=expert_scores,
              topk_ids=expert_indices)
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


def get_benchmark():

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
                  provider):
        print(f"Running config: {m, n, k, num_experts, topk, \
                                x_dtype, w_dtype, \
                                has_bias}, Provider: {provider}",
              flush=True)
        total_latency = 0.0
        ms = 0.0

        _, _, w13_bias, _, w2_bias, _, \
            _, a, w13, w13_scales, w2, w2_scales, \
                expert_scores, expert_indices = make_fused_moe_input(
                    config=((m, n, k), num_experts,
                            topk, x_dtype, w_dtype, has_bias))

        if provider == "vllm":
            moe = XpuFusedMoe(w13=w13,
                              w13_scales=w13_scales,
                              w13_bias=w13_bias,
                              w2=w2,
                              w2_scales=w2_scales,
                              w2_bias=w2_bias,
                              n_experts_per_token=topk,
                              activation="silu",
                              num_experts=num_experts)
            output = torch.empty_like(a)

            def run_vllm_moe():
                moe.apply(output=output,
                          hidden_states=a,
                          topk_weights=expert_scores,
                          topk_ids=expert_indices)

            # Use triton do_bench instead of a single start/end event span:
            # it auto-warms up, flushes the L2 cache between reps and reports
            # the median. A single event span also captures host-side launch
            # overhead / queue bubbles between kernels, which is a large and
            # noisy fraction of the time for small MoE configs.
            ms, _, _ = triton.testing.do_bench(
                run_vllm_moe,
                quantiles=[0.5, 0.2, 0.8],
            )
            clear_xpu_cache()
            return 1000 * ms
        else:
            moe = XpuFusedMoe_CalKernelTime(w13=w13,
                            w13_scales=w13_scales,
                            w13_bias=w13_bias,
                            w2=w2,
                            w2_scales=w2_scales,
                            w2_bias=w2_bias,
                            n_experts_per_token=topk,
                            activation="silu",
                            num_experts=num_experts)
            output = torch.empty_like(a)
            # L2 cache flush buffer, mirroring triton.testing.do_bench, so
            # the per-sub-kernel timings are not biased by warm caches.
            cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=DEVICE)

            # Size the measurement loop by a time budget like
            # triton.testing.do_bench: first estimate the per-call time, then
            # pick warmup / repeat counts so we warm up ~25ms and measure
            # ~100ms worth of iterations (auto-scales with kernel size).
            warmup_ms = 25
            rep_ms = 100
            for _ in range(5):
                moe.apply(output=output,
                    hidden_states=a,
                    topk_weights=expert_scores,
                    topk_ids=expert_indices)
            torch.xpu.synchronize()
            est_start = torch.xpu.Event(enable_timing=True)
            est_end = torch.xpu.Event(enable_timing=True)
            est_start.record()
            for _ in range(5):
                moe.apply(output=output,
                    hidden_states=a,
                    topk_weights=expert_scores,
                    topk_ids=expert_indices)
            est_end.record()
            torch.xpu.synchronize()
            estimate_ms = est_start.elapsed_time(est_end) / 5
            n_warmup = max(1, int(warmup_ms / estimate_ms))
            n_measured = max(1, int(rep_ms / estimate_ms))

            # per-sub-kernel events, sized to the measured loop
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
            # extra warm up before the measured loop
            for _ in range(n_warmup):
                moe.apply(output=output,
                    hidden_states=a,
                    topk_weights=expert_scores,
                    topk_ids=expert_indices)
            gemm1_info = gemm2_info = None
            for i in range(n_measured):
                cache.zero_()  # flush L2 before each measured iteration
                cur_gemm1_info, cur_gemm2_info = \
                    moe.apply(output=output,
                        hidden_states=a,
                        topk_weights=expert_scores,
                        topk_ids=expert_indices,
                        start_event_remap=remap_se[i],
                        end_event_remap=remap_ee[i],
                        start_event_gemm1=gemm1_se[i],
                        end_event_gemm1=gemm1_ee[i],
                        start_event_gemm2=gemm2_se[i],
                        end_event_gemm2=gemm2_ee[i],
                        start_event_gather=gather_se[i],
                        end_event_gather=gather_ee[i],)
                if i == 0:
                    gemm1_info = cur_gemm1_info
                    gemm2_info = cur_gemm2_info
            torch.xpu.synchronize()
            # Take the per-iteration median (like do_bench) instead of the
            # mean, then scale by n_measured so the downstream
            # `total_latency / n_measured` still yields the median.
            remap_latency = statistics.median([
                remap_se[i].elapsed_time(remap_ee[i])
                for i in range(n_measured)
            ]) * n_measured
            gemm1_latency = statistics.median([
                gemm1_se[i].elapsed_time(gemm1_ee[i])
                for i in range(n_measured)
            ]) * n_measured
            gemm2_latency = statistics.median([
                gemm2_se[i].elapsed_time(gemm2_ee[i])
                for i in range(n_measured)
            ]) * n_measured
            gather_latency = statistics.median([
                gather_se[i].elapsed_time(gather_ee[i])
                for i in range(n_measured)
            ]) * n_measured
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
        ms = total_latency / n_measured
        clear_xpu_cache()
        return 1000 * ms

    return benchmark


if __name__ == "__main__":

    args = parse_args()
    seed = 1234
    seed_everything(seed)

    configs = gen_correctness_config()

    for config in configs:
        try:
            calculate_diff(config)
        except Exception as e:
            print("Error in config: ", config, " error: ", e)
        clear_xpu_cache()

    configs = gen_perf_configs()
    benchmark = get_benchmark()
    save_path = ensure_save_path_exists(args.save_path)
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=save_path)
