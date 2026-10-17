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


def _resadd_ref(x, r, nw):
    t = x.float() + r.float()
    y = (t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + EPS) *
         (nw.float() + 1)).to(x.dtype)
    return y, t.to(x.dtype)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_resadd_rmsnorm_fp8_gemm(m, dtype):
    torch.manual_seed(0)
    x = torch.randn(m, 2048, device=DEVICE, dtype=dtype)
    r = torch.randn(m, 2048, device=DEVICE, dtype=dtype)
    nw = torch.randn(2048, device=DEVICE, dtype=dtype) * 0.1
    w, s = _w(4608, 2048), torch.tensor([0.02], device=DEVICE)
    out, r_out = torch.ops._xpu_C.resadd_rmsnorm_fp8_gemm(
        x, r, nw, EPS, w.t(), s)
    y, t = _resadd_ref(x, r, nw)
    torch.testing.assert_close(r_out, t, atol=0, rtol=0)
    torch.testing.assert_close(out, _lin(y, w, s), atol=3e-3, rtol=2e-2)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_resadd_rmsnorm_fp8_gemm_pair(m, dtype):
    torch.manual_seed(0)
    x = torch.randn(m, 2048, device=DEVICE, dtype=dtype)
    r = torch.randn(m, 2048, device=DEVICE, dtype=dtype)
    nw = torch.randn(2048, device=DEVICE, dtype=dtype) * 0.1
    w1, w2 = _w(6144, 2048), _w(32, 2048)
    s1, s2 = torch.tensor([0.02], device=DEVICE), torch.tensor([0.03],
                                                               device=DEVICE)
    o1, o2, r_out = torch.ops._xpu_C.resadd_rmsnorm_fp8_gemm_pair(
        x, r, nw, EPS, w1.t(), s1, w2.t(), s2)
    y, t = _resadd_ref(x, r, nw)
    torch.testing.assert_close(r_out, t, atol=0, rtol=0)
    torch.testing.assert_close(o1, _lin(y, w1, s1), atol=3e-3, rtol=2e-2)
    torch.testing.assert_close(o2, _lin(y, w2, s2), atol=3e-3, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_unfused_norm_strided_inputs(dtype):
    # M > 1 takes the standalone norm kernels: row-strided views of wider
    # buffers, [M, H*D] gated layout.
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

    xb = torch.randn(m, 4096, device=DEVICE, dtype=dtype)
    rb = torch.randn(m, 6144, device=DEVICE, dtype=dtype)
    x, r = xb[:, 2048:], rb[:, :2048]
    nw = torch.randn(2048, device=DEVICE, dtype=dtype) * 0.1
    w1, w2 = _w(6144, 2048), _w(32, 2048)
    s1, s2 = torch.tensor([0.02], device=DEVICE), torch.tensor([0.03],
                                                               device=DEVICE)
    o1, o2, r_out = torch.ops._xpu_C.resadd_rmsnorm_fp8_gemm_pair(
        x, r, nw, EPS, w1.t(), s1, w2.t(), s2)
    y, t = _resadd_ref(x, r, nw)
    torch.testing.assert_close(r_out, t, atol=0, rtol=0)
    torch.testing.assert_close(o1, _lin(y, w1, s1), atol=3e-3, rtol=2e-2)
    torch.testing.assert_close(o2, _lin(y, w2, s2), atol=3e-3, rtol=2e-2)


@pytest.mark.parametrize("m", [1, 4])
def test_bf16_resadd_norm_keeps_large_residual(m):
    x = torch.full((m, 2048), 2.0**17, device=DEVICE, dtype=torch.bfloat16)
    residual = torch.zeros_like(x)
    norm_weight = torch.zeros(2048, device=DEVICE, dtype=x.dtype)
    w = _w(32, 2048)
    scale = torch.tensor([0.02], device=DEVICE)
    out, residual_out = torch.ops._xpu_C.resadd_rmsnorm_fp8_gemm(
        x, residual, norm_weight, EPS, w.t(), scale)
    y, expected_residual = _resadd_ref(x, residual, norm_weight)
    assert out.dtype == residual_out.dtype == torch.bfloat16
    torch.testing.assert_close(residual_out, expected_residual, atol=0, rtol=0)
    torch.testing.assert_close(out, _lin(y, w, scale), atol=3e-3, rtol=2e-2)
