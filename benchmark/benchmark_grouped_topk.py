# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from argparse import ArgumentParser
from typing import Optional

import torch
import triton

from tests.ops.grouped_topk_op import fused_grouped_topk, grouped_topk


@torch.compile
def grouped_topk_compile(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    assert hidden_states.size(0) == gating_output.size(0), (
        "Number of tokens mismatch")
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")
    num_token = scores.size(0)
    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (scores.view(num_token, num_expert_group,
                                    -1).topk(2, dim=-1)[0].sum(dim=-1))
    else:
        group_scores = scores.view(num_token, num_expert_group,
                                   -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.size(-1) // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(),
                                    float("-inf"))  # [n, e]
    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores,
                                            k=topk,
                                            dim=-1,
                                            sorted=False)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


n_token_range = [1, 64, 256]
n_expert_range = [16, 64, 128]
topk_range = [2, 4]
topk_group_range = [4, 8]
scoring_func_range = ["sigmoid", "softmax"]
dtype_range = [torch.float16]
configs = list(
    itertools.product(
        n_token_range,
        n_expert_range,
        topk_range,
        topk_group_range,
        scoring_func_range,
        dtype_range,
    ))


def get_benchmark():

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=[
                "n_token", "n_expert", "topk", "topk_group", "scoring_func",
                "dtype"
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
        n_token: int,
        n_expert: int,
        topk: int,
        topk_group: int,
        scoring_func: str,
        dtype: torch.dtype,
        provider: str = "vllm",
    ):
        n_hidden = 1024
        routed_scaling_factor = 1.0
        num_expert_group = 8
        renormalize = True
        hidden_states = torch.randn((n_token, n_hidden),
                                    dtype=dtype,
                                    device="xpu")
        gating_output = torch.randn((n_token, n_expert),
                                    dtype=dtype,
                                    device="xpu")
        e_score_correction_bias = torch.randn((n_expert, ),
                                              dtype=torch.float32,
                                              device="xpu")

        quantiles = [0.5, 0.2, 0.8]

        if provider == "vllm":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fused_grouped_topk(
                    hidden_states=hidden_states,
                    gating_output=gating_output,
                    topk=topk,
                    renormalize=renormalize,
                    num_expert_group=num_expert_group,
                    topk_group=topk_group,
                    scoring_func=scoring_func,
                    routed_scaling_factor=routed_scaling_factor,
                    e_score_correction_bias=e_score_correction_bias),
                quantiles=quantiles,
            )
        elif provider == "native":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: grouped_topk(
                    hidden_states=hidden_states,
                    gating_output=gating_output,
                    topk=topk,
                    renormalize=renormalize,
                    num_expert_group=num_expert_group,
                    topk_group=topk_group,
                    scoring_func=scoring_func,
                    routed_scaling_factor=routed_scaling_factor,
                    e_score_correction_bias=e_score_correction_bias),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: grouped_topk_compile(
                    hidden_states=hidden_states,
                    gating_output=gating_output,
                    topk=topk,
                    renormalize=renormalize,
                    num_expert_group=num_expert_group,
                    topk_group=topk_group,
                    scoring_func=scoring_func,
                    routed_scaling_factor=routed_scaling_factor,
                    e_score_correction_bias=e_score_correction_bias),
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
