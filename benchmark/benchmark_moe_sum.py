# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import torch
import triton

from tests.register_ops import moe_sum
from tests.utils import parse_args, opcheck


DEVICE = "xpu"


def moe_sum_native(input):
    return input.sum(dim=1)


def moe_sum_vllm(m, k, input, dtype):
    output = torch.empty((m, k), device=DEVICE, dtype=dtype)
    moe_sum(input, output)
    return output


def calculate_diff(config):
    m, topk, k, dtype = config
    input = torch.randn((m, topk, k), device=DEVICE, dtype=dtype)
    output_native = moe_sum_native(input.clone())
    output_vllm = moe_sum_vllm(m, k, input.clone(), dtype)

    if torch.allclose(output_native, output_vllm, atol=2e-2, rtol=0):
        print("✅ All implementations match, ", config)
    else:
        print("❌ Implementations differ, ", config)

    opcheck(torch.ops._moe_C.moe_sum, (input, output_vllm))


def get_benchmark():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "topk", "k", "dtype"],
            x_vals=[tuple(i) for i in configs],
            line_arg="provider",
            line_vals=["native", "vllm"],
            line_names=["Native", "vLLM"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="us",
            plot_name="fp8-gemm-perf",
            args={},
        )
    )
    def benchmark(m, topk, k, dtype, provider):
        input = torch.randn((m, topk, k), device=DEVICE, dtype=dtype)

        quantiles = [0.5, 0.2, 0.8]

        if provider == "native":
            input_clone = input.clone()
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: moe_sum_native(input_clone),
                quantiles=quantiles,
            )
        else:
            input_clone = input.clone()
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: moe_sum_vllm(m, k, input_clone, dtype),
                quantiles=quantiles,
            )
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":

    args = parse_args()
    seed = 1234
    torch.manual_seed(seed)

    m = [1, 33, 64, 222]
    topk = [2, 6]
    k = [128, 511, 1024]
    dtype = [torch.float32, torch.float16, torch.bfloat16]
    print("Final configuration:")
    print(f"  m: {m}")
    print(f"  topk: {topk}")
    print(f"  k: {k}")
    print(f"  dtype: {dtype}")

    configs = list(
        itertools.product(m, topk, k, dtype))
    # fp8_numerical_sanity_check
    for config in configs:
        calculate_diff(config)

    benchmark = get_benchmark()
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)
