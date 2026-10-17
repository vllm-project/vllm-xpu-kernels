# SPDX-License-Identifier: Apache-2.0
# RMSNorm-prologue + fp8 linear ops: fused GEMV for single rows, unfused
# norm + fp8_gemm_w8a16 otherwise; both must match the fp32 reference.
import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

DEVICE = "xpu"
EPS = 1e-6


def _w(n, k):
    return (torch.randn(n, k, device=DEVICE) * 0.05).to(torch.float8_e4m3fn)


def _lin(y, w, s):
    return (y.float() @ (w.float() * s).t()).to(y.dtype)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("h,d", [(16, 128), (8, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gated_rmsnorm_fp8_gemm(m, h, d, dtype):
    torch.manual_seed(0)
    x = torch.randn(m, h, d, device=DEVICE, dtype=dtype)
    z = torch.randn(m, h, d, device=DEVICE, dtype=dtype)
    nw = torch.randn(d, device=DEVICE, dtype=dtype) * 0.1 + 1
    w, s = _w(2048, h * d), torch.tensor([0.02], device=DEVICE)
    out = torch.ops._xpu_C.gated_rmsnorm_fp8_gemm(x, z, nw, EPS, w.t(), s)
    xf = x.float()
    y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) *
         nw.float() * torch.nn.functional.silu(z.float())).to(dtype)
    torch.testing.assert_close(out,
                               _lin(y.reshape(m, -1), w, s),
                               atol=3e-3,
                               rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_unfused_norm_strided_inputs(dtype):
    # M > 1 takes the standalone norm kernel: a row-strided view of a wider
    # buffer, [M, H*D] gated layout.
    torch.manual_seed(0)
    m, h, d = 5, 16, 128
    xb = torch.randn(m, 2 * h * d, device=DEVICE, dtype=dtype)
    zb = torch.randn(m, 3 * h * d, device=DEVICE, dtype=dtype)
    x, z = xb[:, h * d:], zb[:, :h * d]
    nw = torch.randn(d, device=DEVICE, dtype=dtype) * 0.1 + 1
    w, s = _w(2048, h * d), torch.tensor([0.02], device=DEVICE)
    out = torch.ops._xpu_C.gated_rmsnorm_fp8_gemm(x, z, nw, EPS, w.t(), s)
    xf = x.float().reshape(m, h, d)
    y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) *
         nw.float() * torch.nn.functional.silu(z.float().reshape(m, h, d)))
    torch.testing.assert_close(out,
                               _lin(y.to(dtype).reshape(m, -1), w, s),
                               atol=3e-3,
                               rtol=2e-2)
