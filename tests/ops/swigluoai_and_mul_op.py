# SPDX-License-Identifier: Apache-2.0
import torch

from tests.ops.custom_ops import CustomOp


def swigluoai_and_mul(
    out: torch.Tensor,
    input: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> None:
    import tests.register_ops as ops
    ops.swigluoai_and_mul(out, input, alpha, limit)


# @CustomOp.register("swigluoai_and_mul")
class SwigluOAIAndMul(CustomOp):
    # https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L106-L110
    def __init__(self, alpha: float = 1.702, limit: float = 7.0):
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""

        gate, up = x[..., ::2], x[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        gated_output = (up + 1) * glu
        return gated_output

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        swigluoai_and_mul(out, x, self.alpha, self.limit)
        return out

    def extra_repr(self) -> str:
        return f"alpha={repr(self.alpha)}, limit={repr(self.limit)}"
