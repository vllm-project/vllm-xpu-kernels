# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch


def grouped_topk_native(
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


def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    import tests.register_ops as ops
    return ops.grouped_topk(hidden_states, gating_output, topk, renormalize,
                            num_expert_group, topk_group, scoring_func,
                            e_score_correction_bias)
