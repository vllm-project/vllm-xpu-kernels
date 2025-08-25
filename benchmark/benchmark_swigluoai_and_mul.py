# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from argparse import ArgumentParser

import torch
import triton

from tests.ops.swigluoai_and_mul_op import SwigluOAIAndMul


@torch.compile
def swigluoai_and_mul_compile(x: torch.Tensor,
                              alpha: float = 1.702,
                              limit: float = 7.0) -> torch.Tensor:
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu
    return gated_output


num_tokens_range = [7, 83, 2048]  # Arbitrary values for testing
d_range = [512, 13824]  # Arbitrary values for testing
dtypes_range = [torch.half, torch.bfloat16, torch.float]
configs = list(itertools.product(
    num_tokens_range,
    d_range,
    dtypes_range,
))


def get_benchmark():

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens", "d", "dtype"],
            x_vals=[tuple(_) for _ in configs],
            line_arg="provider",
            line_vals=["vllm", "native", "compile"],
            line_names=["vllm", "native", "compile"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-"),
                    ("red", "-")],
            ylabel="us",
            plot_name="swigluoai_and_mul-perf",
            args={},
        ))
    def benchmark(
        num_tokens: int,
        d: int,
        dtype: torch.dtype,
        provider: str = "vllm",
    ):
        torch.set_default_device("xpu")

        x = torch.randn(num_tokens, 2 * d, dtype=dtype)

        layer = SwigluOAIAndMul()

        quantiles = [0.5, 0.2, 0.8]

        if provider == "vllm":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: layer(x),
                quantiles=quantiles,
            )
        elif provider == "native":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: layer.forward_native(x),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: swigluoai_and_mul_compile(x),
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Benchmark the swigluoai_and_mul kernel.")
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/swigluoai_and_mul/",
        help="Path to save swigluoai_and_mul benchmark results",
    )

    args = parser.parse_args()

    benchmark = get_benchmark()
    benchmark.run(print_data=True, save_path=args.save_path)
