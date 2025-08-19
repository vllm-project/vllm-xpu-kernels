# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Optional

import torch
import vllm_xpu_kernels._moe_C  # noqa: F401

if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake


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
    return torch.ops._moe_C.grouped_topk(hidden_states, gating_output, topk,
                                         renormalize, num_expert_group,
                                         topk_group, scoring_func,
                                         e_score_correction_bias)


if hasattr(torch.ops._moe_C, "grouped_topk"):

    @register_fake("_moe_C::grouped_topk")
    def _grouped_topk_fake(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        num_expert_group: int,
        topk_group: int,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_weights = torch.empty((gating_output.size(0), topk),
                                   dtype=torch.float32,
                                   device=hidden_states.device)
        topk_indices = torch.empty((gating_output.size(0), topk),
                                   dtype=torch.int32,
                                   device=hidden_states.device)
        return topk_weights, topk_indices
