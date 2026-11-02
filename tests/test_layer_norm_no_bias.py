# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for torch.ops._xpu_C.layer_norm: weighted LayerNorm without bias,
float32 weight, norm_dim specialized at compile time. Not to be confused with
torch.ops._C.layer_norm (optional weight/bias, any hidden size), which is
covered by tests/test_layernorm.py."""
import pytest
import torch
from torch import nn

import tests.register_ops as ops  # noqa: F401
from tests.ops.custom_ops import CustomOp
from tests.utils import REF_DEVICE, opcheck

DEVICE = REF_DEVICE
KERNEL_DEVICE = "xpu"

NUM_TOKENS = [1, 7, 83, 4096]
# layer_norm requires norm_dim to be 7168 or a power of two in
# [128, 131072]; this list exercises every kernel size specialization
# (small/medium/large).
NORM_DIMS = [
    128, 256, 512, 1024, 2048, 4096, 7168, 8192, 16384, 32768, 65536, 131072
]
DTYPES = [torch.float, torch.half, torch.bfloat16]
SEEDS = [0]
XPU_DEVICES = ["xpu:0"]
OPCHECK_MAX_NUMEL = 1 << 24

# override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [7],
        "norm_dim": [128],
        "dtype": [torch.bfloat16],
    },
}


def _to_kernel(x):
    return None if x is None else x.to(KERNEL_DEVICE)


class LayerNormNoBias(CustomOp):
    """Weighted LayerNorm (no bias) over the last dimension of a 2D tensor.

    Computes x -> (x - mean) / sqrt(var + eps) * weight, where weight is
    always float32 regardless of the input dtype. norm_dim must be 7168 or
    a power of two in [128, 131072].
    """

    def __init__(self, norm_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm_dim = norm_dim
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(norm_dim, dtype=torch.float32))
        self.op = torch.ops._xpu_C.layer_norm

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """Mirrors the kernel's actual math rather than torch's more
        numerically stable layer_norm: accumulate mean/E[x^2] in float32
        (regardless of input dtype, like the kernel does), derive variance
        as the one-pass E[x^2] - E[x]^2 (not a two-pass/Welford reduction),
        then normalize and cast back to x's dtype."""
        orig_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        mean = x_f32.mean(dim=-1, keepdim=True)
        mean_sq = x_f32.pow(2).mean(dim=-1, keepdim=True)
        var = torch.clamp(mean_sq - mean.pow(2), min=0.0)
        inv_std = torch.rsqrt(var + self.variance_epsilon)
        out = (x_f32 - mean) * inv_std * self.weight.to(torch.float32)
        return out.to(orig_dtype)

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        self.op(out, x, self.weight, self.variance_epsilon)
        return out


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("norm_dim", NORM_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_layer_norm_no_bias(
    num_tokens: int,
    norm_dim: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.manual_seed(seed)
    torch.xpu.set_device(device)
    layer = LayerNormNoBias(norm_dim).to(DEVICE)
    layer.weight.data.uniform_(-5.0, 5.0)
    x = (torch.randn(num_tokens, norm_dim, dtype=dtype, device=DEVICE) * 10.0 -
         5.0)

    ref_out = layer.forward_native(x)
    layer_k = layer.to(KERNEL_DEVICE)
    out = layer_k(_to_kernel(x))
    out = out.to(DEVICE)
    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)

    # opcheck clones its inputs several times; skip it for the largest
    # shapes (e.g. 4096x131072) so it fits on 24 GB cards. The schema is
    # already covered by the smaller shapes.
    if num_tokens * norm_dim > OPCHECK_MAX_NUMEL:
        return
    out_k = torch.empty_like(x).to(KERNEL_DEVICE)
    opcheck(
        torch.ops._xpu_C.layer_norm,
        (out_k, _to_kernel(x), layer_k.weight.data, layer_k.variance_epsilon),
    )
