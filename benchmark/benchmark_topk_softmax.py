# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from argparse import ArgumentParser
from typing import Optional

import torch
import triton

from tests.ops.topk_softmax_op import fused_topk, topk_softmax


@torch.compile
def topk_softmax_compile(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    routing_weights = torch.softmax(gating_output, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


n_token_range = [1, 64, 256]
n_expert_range = [16, 192, 512, 1024]
topk_range = [2, 4]
renormalize_range = [True, False]
dtype_range = [torch.float16, torch.bfloat16, torch.float32]
configs = list(
    itertools.product(
        n_token_range,
        n_expert_range,
        topk_range,
        renormalize_range,
        dtype_range,
    ))


def get_benchmark():

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=[
                "n_token",
                "n_expert",
                "topk",
                "renormalize",
                "dtype",
            ],
            x_vals=[tuple(_) for _ in configs],
            line_arg="provider",
            line_vals=["vllm", "native", "compile"],
            line_names=["vllm", "native", "compile"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-"),
                    ("red", "-")],
            ylabel="us",
            plot_name="topk_softmax-perf",
            args={},
        ))
    def benchmark(
        n_token: int,
        n_expert: int,
        topk: int,
        renormalize: int,
        dtype: torch.dtype,
        provider: str = "vllm",
    ):
        n_hidden = 1024
        hidden_states = torch.randn((n_token, n_hidden),
                                    dtype=dtype,
                                    device="xpu")
        gating_output = torch.randn((n_token, n_expert),
                                    dtype=dtype,
                                    device="xpu")

        quantiles = [0.5, 0.2, 0.8]

        if provider == "vllm":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fused_topk(hidden_states=hidden_states,
                                   gating_output=gating_output,
                                   topk=topk,
                                   renormalize=renormalize),
                quantiles=quantiles,
            )
        elif provider == "native":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: topk_softmax(hidden_states=hidden_states,
                                     gating_output=gating_output,
                                     topk=topk,
                                     renormalize=renormalize),
                quantiles=quantiles,
            )
        elif provider == "compile":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: topk_softmax_compile(hidden_states=hidden_states,
                                             gating_output=gating_output,
                                             topk=topk,
                                             renormalize=renormalize),
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the topk_softmax kernel.")
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/topk_softmax/",
        help="Path to save topk_softmax benchmark results",
    )

    args = parser.parse_args()

    benchmark = get_benchmark()
    benchmark.run(print_data=True, save_path=args.save_path)
