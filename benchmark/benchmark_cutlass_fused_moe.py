# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

# isort: off
import gc

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

def clear_xpu_cache():
    torch.xpu.empty_cache()
    gc.collect()
    torch.xpu.synchronize()


def calculate_flops(m, n, k):
    return 2 * m * n * k


def calculate_memory_usage(m, n, k, num_experts, x_dtype, w_dtype=None):
    io_memory = (m * k + m * n) * torch.tensor(
        [], dtype=x_dtype).element_size() / (1024**3)  # in GB
    weight_memory = num_experts * k * n * torch.tensor(
        [], dtype=x_dtype if w_dtype is None else w_dtype).element_size() / (
            1024**3)  # in GB
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
                "vllm", "vllm_kernel_gemm1", "vllm_kernel_gemm2",
                "vllm_kernel_gather", "vllm_kernel_gemm1_tflops",
                "vllm_kernel_gemm2_tflops", "vllm_kernel_gemm1_memory",
                "vllm_kernel_gemm2_memory"
            ],
            line_names=[
                "vllm(us)", "vllm_kernel_gemm1(us)", "vllm_kernel_gemm2(us)",
                "vllm_kernel_gather(us)", "vllm_kernel_gemm1_tflops",
                "vllm_kernel_gemm2_tflops", "vllm_kernel_gemm1_memory(GB/s)",
                "vllm_kernel_gemm2_memory(GB/s)"
            ],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-"),
                    ("purple", "-"), ("green", "--"), ("orange", "--"),
                    ("green", ":"), ("orange", ":")],
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
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
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
            for index in range(iterations):
                start.record()
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
                end.record()
                end.synchronize()
                if index >= 5:  # skip the first 5 iterations for warmup
                    total_latency += start.elapsed_time(end)
        else:
            for index in range(iterations):
                gemm1, gemm2, gather, (
                    gemm1_m, gemm1_n, gemm1_k, gemm1_expert), (
                        gemm2_m, gemm2_n, gemm2_k,
                        gemm2_expert) = xpu_fused_moe_CalKernelTime(
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
                            start_event=start,
                            end_event=end)
                if index >= 5:  # skip the first 5 iterations for warmup
                    if provider == "vllm_kernel_gemm1" or \
                    provider == "vllm_kernel_gemm1_tflops" or \
                    provider == "vllm_kernel_gemm1_memory":
                        total_latency += gemm1
                        m, n, k, active_experts = gemm1_m, \
                        gemm1_n, gemm1_k, gemm1_expert
                    elif provider == "vllm_kernel_gemm2" or \
                    provider == "vllm_kernel_gemm2_tflops" or \
                    provider == "vllm_kernel_gemm2_memory":
                        total_latency += gemm2
                        m, n, k, active_experts = gemm2_m, \
                        gemm2_n, gemm2_k, gemm2_expert
                    elif provider == "vllm_kernel_gather":
                        total_latency += gather
            if provider == "vllm_kernel_gemm1_tflops" or \
            provider == "vllm_kernel_gemm2_tflops":
                torch.xpu.synchronize()
                ms = total_latency / (iterations - 5)
                clear_xpu_cache()
                flops = calculate_flops(m, n, k)
                return flops / (ms / 1000) / 1e12
            elif provider == "vllm_kernel_gemm1_memory" or \
                provider == "vllm_kernel_gemm2_memory":
                torch.xpu.synchronize()
                ms = total_latency / (iterations - 5)
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
    benchmark.run(print_data=True, save_path=save_path)
