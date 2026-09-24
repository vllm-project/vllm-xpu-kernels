# SPDX-License-Identifier: Apache-2.0
import torch

import tests.register_ops as ops  # noqa: F401
from tests.ops.custom_ops import CustomOp


class Softmax(CustomOp):
    """Row-wise softmax over the last dimension of a 2D tensor.

    The function computes x -> softmax(x, dim=-1).

    Shapes:
        x: (batch, dim)
        return: same shape as x
    """

    def __init__(self):
        super().__init__()
        self.op = torch.ops._xpu_C.softmax

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return torch.softmax(x, dim=-1)

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        self.op(out, x)
        return out
