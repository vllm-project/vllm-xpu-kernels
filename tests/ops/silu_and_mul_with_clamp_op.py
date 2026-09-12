# SPDX-License-Identifier: Apache-2.0
import torch

from tests.ops.custom_ops import CustomOp


def silu_and_mul_with_clamp(
    out: torch.Tensor,
    input: torch.Tensor,
    limit: float,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> None:
    import tests.register_ops as ops
    ops.silu_and_mul_with_clamp(out, input, limit, alpha, beta)


class SiluAndMulWithClamp(CustomOp):

    def __init__(self, limit: float = 7.0, alpha: float = 1.0,
                 beta: float = 0.0):
        super().__init__()
        self.limit = limit
        self.alpha = alpha
        self.beta = beta

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        gate, up = x.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return gate * torch.sigmoid(self.alpha * gate) * (up + self.beta)

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        silu_and_mul_with_clamp(out, x, self.limit, self.alpha, self.beta)
        return out

    def extra_repr(self) -> str:
        return (f"limit={repr(self.limit)}, alpha={repr(self.alpha)}, "
                f"beta={repr(self.beta)}")
