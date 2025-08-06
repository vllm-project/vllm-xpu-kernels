import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.ops.custom_ops import CustomOp


class SiluAndMul(CustomOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward_xpu(self, x):
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        import tests.register_ops as ops
        ops.silu_and_mul(out, x)
        return out

    def naive_forward(self, x: torch.Tensor):
        dtype = x.dtype
        d = x.shape[-1] // 2
        x = x.to(torch.float32)
        ret = F.silu(x[..., :d]) * x[..., d:]
        return ret.to(dtype)
