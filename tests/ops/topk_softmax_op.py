# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

import tests.register_ops as ops


def topk_softmax(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    bias: Optional[torch.Tensor] = None,
    indices_type: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    routing_weights = torch.softmax(gating_output, dim=-1, dtype=torch.float32)
    if bias is not None:
        routing_weights_with_bias = routing_weights + bias.unsqueeze(0)
        _, topk_ids = torch.topk(routing_weights_with_bias, topk, dim=-1)
        topk_weights = routing_weights.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    bias: Optional[torch.Tensor] = None,
    indices_type: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert hidden_states.size(0) == gating_output.size(0), (
        "Number of tokens mismatch")

    M, _ = hidden_states.size()

    topk_weights = torch.empty(M,
                               topk,
                               dtype=torch.float32,
                               device=hidden_states.device)
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=hidden_states.device)
    token_expert_indices = torch.empty(M,
                                       topk,
                                       dtype=torch.int32,
                                       device=hidden_states.device)

    ops.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        bias,
    )

    return topk_weights, topk_ids
