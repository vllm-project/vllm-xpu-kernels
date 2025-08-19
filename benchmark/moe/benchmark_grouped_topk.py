# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from argparse import ArgumentParser
from typing import Optional

import torch
import triton

from tests.moe.ops.grouped_topk import grouped_topk, grouped_topk_native

dpcpp_device = torch.device("xpu")


@torch.compile
def grouped_topk_compile(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert hidden_states.size(0) == gating_output.size(0), (
        "Number of tokens mismatch")
    if scoring_func == "softmax":
        gating_output = gating_output.to(torch.float32)
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.shape[0]
    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        e_score_correction_bias = e_score_correction_bias.to(torch.float32)
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (scores.view(num_token, num_expert_group,
                                    -1).topk(2, dim=-1)[0].sum(dim=-1))
    else:
        group_scores = (scores.view(num_token, num_expert_group,
                                    -1).max(dim=-1).values)  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=True)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.shape[-1] // num_expert_group).reshape(num_token, -1))  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(),
                                    float("-inf"))  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=True)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores,
                                            k=topk,
                                            dim=-1,
                                            sorted=True)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


num_tokens_range = [1, 2, 64, 256]
num_experts_range = [64, 128, 256]
topk_range = [4, 6, 8]
renormalize_range = [True, False]
num_expert_group_range = [8]
topk_group_range = [4, 6, 8]
scoring_func_range = ["sigmoid", "softmax"]
has_bias_range = [True, False]
configs = list(
    itertools.product(
        num_tokens_range,
        num_experts_range,
        topk_range,
        renormalize_range,
        num_expert_group_range,
        topk_group_range,
        scoring_func_range,
        has_bias_range,
    ))


def get_benchmark():

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=[
                "num_tokens", "num_experts", "topk", "renormalize",
                "num_expert_group", "topk_group", "scoring_func", "has_bias"
            ],
            x_vals=[tuple(_) for _ in configs],
            line_arg="provider",
            line_vals=["vllm", "native", "compile"],
            line_names=["vllm", "native", "compile"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-"),
                    ("red", "-")],
            ylabel="us",
            plot_name="grouped_topk-perf",
            args={},
        ))
    def benchmark(
        num_tokens: int,
        num_experts: int,
        topk: int,
        renormalize: bool,
        num_expert_group: int,
        topk_group: int,
        scoring_func: str = "softmax",
        has_bias: bool = False,
        provider: str = "vllm",
    ):
        dtype = torch.float16
        torch.set_default_device("xpu")

        gating_output = torch.randn(num_tokens,
                                    num_experts,
                                    device=dpcpp_device).to(dtype)
        hidden_states = torch.zeros(num_tokens,
                                    num_experts,
                                    device=dpcpp_device).to(dtype)

        bias = None
        if has_bias:
            if has_bias and scoring_func == "sigmoid" \
                and dtype is not torch.float32:
                # using a bias of bigger number to avoid Low-precision
                bias = torch.arange(1,
                                    num_experts + 1).to(dpcpp_device).to(dtype)
            else:
                bias = torch.randn(num_experts, device=dpcpp_device).to(dtype)

        quantiles = [0.5, 0.2, 0.8]

        if provider == "vllm":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: grouped_topk(
                    hidden_states,
                    gating_output,
                    topk,
                    renormalize,
                    num_expert_group,
                    topk_group,
                    scoring_func=scoring_func,
                    e_score_correction_bias=bias,
                ),
                quantiles=quantiles,
            )
        elif provider == "native":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: grouped_topk_native(
                    hidden_states,
                    gating_output,
                    topk,
                    renormalize,
                    num_expert_group,
                    topk_group,
                    scoring_func=scoring_func,
                    e_score_correction_bias=bias,
                ),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: grouped_topk_compile(
                    hidden_states,
                    gating_output,
                    topk,
                    renormalize,
                    num_expert_group,
                    topk_group,
                    scoring_func=scoring_func,
                    e_score_correction_bias=bias,
                ),
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the grouped topk kernel.")
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/grouped_topk/",
        help="Path to save grouped_topk benchmark results",
    )

    args = parser.parse_args()

    benchmark = get_benchmark()
    benchmark.run(print_data=True, save_path=args.save_path)
