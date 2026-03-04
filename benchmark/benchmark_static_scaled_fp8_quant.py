# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import torch
import triton
import random
import numpy as np

from tests.utils import parse_args
from tests.ops.fp8_quant_op import scaled_fp8_quant, scaled_quantize


DEVICE = "xpu"
def seed_everything(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def calculate_diff(config):
    num_tokens, hidden_size, dtype, fp8_dtype, group_shape = config
    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device=DEVICE)
    ref_out, scale_2d = scaled_quantize(x,
                                        group_shape,
                                        fp8_dtype,
                                        compute_dtype=torch.float32)

    # Flatten scale to 1D for testing 1D scale path
    scale_1d = scale_2d.flatten()
    ops_out, ops_scale = scaled_fp8_quant(x,
                                          scale=scale_1d,
                                          fp8_dtype=fp8_dtype,
                                          group_shape=group_shape)

    if torch.allclose(scale_1d, ops_scale, atol=0, rtol=0):
        print("✅ All implementations match, ", config)
    else:
        print("❌ Implementations differ, ", config)

    if torch.allclose(ref_out.float(), ops_out.float(), atol=0.0, rtol=0.12):
        print("✅ All implementations match, ", config)
    else:
        print("❌ Implementations differ, ", config)


def get_benchmark():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens", "hidden_size", "dtype", "fp8_dtype", "group_shape"],
            x_vals=[tuple(i) for i in configs],
            line_arg="provider",
            line_vals=["native", "vllm"],
            line_names=["Native", "vLLM"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="us",
            plot_name="static_scaled_fp8_quant-perf",
            args={},
        )
    )
    def benchmark(num_tokens, hidden_size, dtype, fp8_dtype, group_shape, provider):
        x = torch.rand(num_tokens, hidden_size, dtype=dtype, device=DEVICE)
        quantiles = [0.5, 0.2, 0.8]

        if provider == "native":
            x_clone = x.clone()
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: scaled_quantize(x_clone,
                group_shape,
                fp8_dtype,
                compute_dtype=torch.float32),
                quantiles=quantiles,
            )
        else:
            x_clone = x.clone()
            _, scale_2d = scaled_quantize(x_clone,
                group_shape,
                fp8_dtype,
                compute_dtype=torch.float32)
            scale_1d = scale_2d.flatten()
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: scaled_fp8_quant(x_clone,
                scale=scale_1d,
                fp8_dtype=fp8_dtype,
                group_shape=group_shape),
                quantiles=quantiles,
            )
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":

    args = parse_args()
    seed_everything(0)

    num_toknes = [1, 7, 83, 4096]
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
    fp8_dtype = [torch.float8_e4m3fn]
    group_shape = [(1, -1), (-1, 1)]
    print("Final configuration:")
    print(f"  num_tokens: {num_toknes}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  dtype: {dtype}")
    print(f"  fp8_dtype: {fp8_dtype}")
    print(f"  group_shape: {group_shape}")

    configs = list(
        itertools.product(num_toknes, hidden_size, dtype, fp8_dtype, group_shape))

    for config in configs:
        calculate_diff(config)

    benchmark = get_benchmark()
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)
