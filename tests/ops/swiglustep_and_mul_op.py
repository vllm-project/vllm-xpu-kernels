# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F

from tests.ops.custom_ops import CustomOp


def swiglustep_and_mul(
    out: torch.Tensor,
    input: torch.Tensor,
    limit: float = 7.0,
) -> None:
    import tests.register_ops as ops
    ops.swiglustep_and_mul(out, input, limit)


class SwigluStepAndMul(CustomOp):

    def __init__(self, limit: float = 7.0):
        super().__init__()
        self.limit = limit

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        gate, up = x.chunk(2, dim=-1)
        gate = F.silu(gate)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return gate * up

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        swiglustep_and_mul(out, x, self.limit)
        return out

    def extra_repr(self) -> str:
        return f"limit={repr(self.limit)}"
