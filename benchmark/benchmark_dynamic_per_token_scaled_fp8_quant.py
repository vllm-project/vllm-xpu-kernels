# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import torch
import triton
from tests.utils import parse_args
from tests.test_fp8_quant import ref_dynamic_per_token_quant, assert_close_percentage
from tests.ops.fp8_quant_op import scaled_fp8_quant


DEVICE = "xpu"


def calculate_diff(config):
    num_tokens, hidden_size, dtype, scale_ub, fp8_dtype = config
    x = (torch.rand(num_tokens, hidden_size, dtype=dtype, device=DEVICE) + 1e-6
         )  # avoid nans
    scale_ub = torch.mean(x).to(dtype=torch.float32,
        device=DEVICE) if scale_ub else None
    ref_out, ref_scales = ref_dynamic_per_token_quant(x, fp8_dtype, scale_ub)

    ops_out, ops_scales = scaled_fp8_quant(x,
                                           scale_ub=scale_ub,
                                           use_per_token_if_dynamic=True,
                                           fp8_dtype=fp8_dtype)

    try:
        torch.testing.assert_close(ref_scales, ops_scales)
        assert_close_percentage(
            ref_out.to(dtype=torch.float32),
            ops_out.to(dtype=torch.float32),
            mismatch_threshold=0.005,
        )  # 0.5% mismatch allowed
        print("✅ All implementations match, ", config)
    except AssertionError as e:
        print("❌ Implementations differ, ", config)
        print(e)


def benchmark_quantization(
    num_tokens, hidden_size, dtype, scale_ub, fp8_dtype, provider
):
    x = (torch.rand(num_tokens, hidden_size, dtype=dtype, device=DEVICE) + 1e-6
         )  # avoid nans
    scale_ub = torch.mean(x).to(dtype=torch.float32,
        device=DEVICE) if scale_ub else None
    quantiles = [0.5, 0.2, 0.8]

    if provider == "native":
        x_clone = x.clone()
        fn = lambda: ref_dynamic_per_token_quant(x_clone, fp8_dtype, scale_ub)
    elif provider == "vllm":
        x_clone = x.clone()
        fn = lambda: scaled_fp8_quant(x_clone, scale_ub=scale_ub, use_per_token_if_dynamic=True, fp8_dtype=fp8_dtype)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


def get_benchmark():
    benchmark = triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens", "hidden_size", "dtype", "scale_ub", "fp8_dtype"],
            x_vals=[tuple(i) for i in configs],
            line_arg="provider",
            line_vals=["native", "vllm"],
            line_names=["Native", "vLLM"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="us",
            plot_name="Dynamic Per-Token Scaled FP8 Quantization Performance",
            args={},
        )
    )(benchmark_quantization)
    return benchmark


if __name__ == "__main__":

    args = parse_args()
    seed = 1234
    torch.manual_seed(seed)

    num_tokens = [1, 7, 83, 4096]
    HIDDEN_SIZES = [
        1,
        2,
        3,
        4,
        16,
        67,
        768,
        2048,
        5120,
        5137,
        8192,
        8193,
    ]  # Arbitrary values for testing
    HIDDEN_SIZES += list(range(1024, 1033))  # vectorized conversion edge cases
    hidden_size = HIDDEN_SIZES
    dtype = [torch.half, torch.bfloat16, torch.float]
    scale_ub = [True, False]
    fp8_dtype = [torch.float8_e5m2, torch.float8_e4m3fn]
    print("Final configuration:")
    print("  num_tokens: ", num_tokens)
    print("  hidden_size: ", hidden_size)
    print("  dtype: ", dtype)
    print("  scale_ub: ", scale_ub)
    print("  fp8_dtype: ", fp8_dtype)

    configs = list(
        itertools.product(num_tokens, hidden_size, dtype, scale_ub, fp8_dtype))

    for config in configs:
        calculate_diff(config)

    benchmark = get_benchmark()
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)
