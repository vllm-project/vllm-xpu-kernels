# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import tests.register_ops as ops  # noqa: F401
from tests.ops.custom_ops import CustomOp


class ReduceMax(CustomOp):
    """Row-wise argmax over the last dimension of a 2D tensor.

    Shapes:
        x: (batch, dim)
        return: (values, indices), each (batch, 1); indices are int32.
    """

    def __init__(self):
        super().__init__()
        self.op = torch.ops._xpu_C.reduce_max_row_wise

    def forward_native(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        values, indices = torch.max(x, dim=-1, keepdim=True)
        return values, indices.to(torch.int32)

    def forward_xpu(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = x.shape[0]
        values = torch.empty(num_tokens, 1, dtype=x.dtype, device=x.device)
        indices = torch.empty(
            num_tokens, 1, dtype=torch.int32, device=x.device)
        self.op(values, indices, x)
        return values, indices


class ReduceMin(CustomOp):
    """Row-wise argmin over the last dimension of a 2D tensor.

    Shapes:
        x: (batch, dim)
        return: (values, indices), each (batch, 1); indices are int32.
    """

    def __init__(self):
        super().__init__()
        self.op = torch.ops._xpu_C.reduce_min_row_wise

    def forward_native(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        values, indices = torch.min(x, dim=-1, keepdim=True)
        return values, indices.to(torch.int32)

    def forward_xpu(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = x.shape[0]
        values = torch.empty(num_tokens, 1, dtype=x.dtype, device=x.device)
        indices = torch.empty(
            num_tokens, 1, dtype=torch.int32, device=x.device)
        self.op(values, indices, x)
        return values, indices
