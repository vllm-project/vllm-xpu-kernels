# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F

import tests.register_ops as ops
from tests.ops.custom_ops import CustomOp


class SiluAndMul(CustomOp):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    def __init__(self):
        super().__init__()
        self.op = ops.silu_and_mul

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out

class FastGELU(CustomOp):

    def __init__(self):
        super().__init__()
        self.op = torch.ops._C.gelu_fast

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 *
                                           (1.0 + 0.044715 * x * x)))

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        self.op(out, x)
        return out

class NewGELU(CustomOp):

    def __init__(self):
        super().__init__()
        self.op = torch.ops._C.gelu_new

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        import math
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c *
                                           (x + 0.044715 * torch.pow(x, 3.0))))

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        self.op(out, x)
        return out
