# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from argparse import ArgumentParser
from typing import Optional

import torch
import triton

from tests.ops.topk_op import (fused_topk_sigmoid, fused_topk_softmax,
                              topk_sigmoid, topk_softmax)


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


@torch.compile
def topk_sigmoid_compile(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    routing_weights = torch.sigmoid(gating_output).to(torch.float32)
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


def get_benchmark(scoring_func: str):
    if scoring_func == "softmax":
        fused_fn = fused_topk_softmax
        native_fn = topk_softmax
        compile_fn = topk_softmax_compile
        plot_name = "topk_softmax-perf"
    elif scoring_func == "sigmoid":
        fused_fn = fused_topk_sigmoid
        native_fn = topk_sigmoid
        compile_fn = topk_sigmoid_compile
        plot_name = "topk_sigmoid-perf"
    else:
        raise ValueError(f"Unsupported scoring_func: {scoring_func}")

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
            plot_name=plot_name,
            args={},
        ))
    def benchmark(
        n_token: int,
        n_expert: int,
        topk: int,
        renormalize: bool,
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
                lambda: fused_fn(hidden_states=hidden_states,
                                 gating_output=gating_output,
                                 topk=topk,
                                 renormalize=renormalize),
                quantiles=quantiles,
            )
        elif provider == "native":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: native_fn(hidden_states=hidden_states,
                                  gating_output=gating_output,
                                  topk=topk,
                                  renormalize=renormalize),
                quantiles=quantiles,
            )
        elif provider == "compile":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: compile_fn(hidden_states=hidden_states,
                                   gating_output=gating_output,
                                   topk=topk,
                                   renormalize=renormalize),
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the topk kernel.")
    parser.add_argument(
        "--scoring-func",
        choices=["softmax", "sigmoid"],
        default="softmax",
        help="Scoring function to benchmark",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/topk/",
        help="Path to save topk benchmark results",
    )

    args = parser.parse_args()

    benchmark = get_benchmark(args.scoring_func)
    save_path = f"{args.save_path.rstrip('/')}/{args.scoring_func}"
    benchmark.run(print_data=True, save_path=save_path)
