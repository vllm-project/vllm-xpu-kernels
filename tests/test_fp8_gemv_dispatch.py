# SPDX-License-Identifier: Apache-2.0
# fp8_gemm_w8a16 single-row (decode) inputs take the SYCL GEMV path on BMG;
# the result must match the fp32 reference like the oneDNN path does.
import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

DEVICE = "xpu"


def _reference(x, w, scale):
    return (x.float() @ (w.float() * scale).t()).to(x.dtype)


@pytest.mark.parametrize("n,k", [(4608, 2048), (2048, 2048), (6144, 2048),
                                 (32, 2048), (1000, 320), (8, 64)])
@pytest.mark.parametrize("m", [1, 2, 8])
@pytest.mark.parametrize("x_3d", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp8_gemm_w8a16_small_m(n, k, m, x_3d, dtype):
    torch.manual_seed(0)
    w = (torch.randn(n, k, device=DEVICE) * 0.05).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.02], device=DEVICE)
    x = torch.randn(m, k, device=DEVICE, dtype=dtype)
    if x_3d:
        x = x.view(m, 1, k)
    out = torch.ops._xpu_C.fp8_gemm_w8a16(x, w.t(), scale, None)
    ref = _reference(x.reshape(m, k), w, scale).reshape(*x.shape[:-1], n)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp8_gemm_w8a16_strided_row(dtype):
    torch.manual_seed(0)
    w = (torch.randn(2048, 2048, device=DEVICE) * 0.05).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.02], device=DEVICE)
    base = torch.randn(1, 4096, device=DEVICE, dtype=dtype)
    x = base[:, 2048:]
    out = torch.ops._xpu_C.fp8_gemm_w8a16(x, w.t(), scale, None)
    torch.testing.assert_close(out, _reference(x.contiguous(), w, scale),
                               atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("value", [2.0**17, -(2.0**17), 2.0**-100])
def test_bf16_gemv_preserves_values_outside_fp16_range(value):
    """BF16 activations must not overflow or underflow through FP16."""
    x = torch.zeros(1, 64, device=DEVICE, dtype=torch.bfloat16)
    x[:, 0] = value
    w = torch.zeros(16, 64, device=DEVICE)
    w[:, 0] = 1
    w = w.to(torch.float8_e4m3fn)
    scale = torch.ones(1, device=DEVICE)
    out = torch.ops._xpu_C.fp8_gemm_w8a16(x, w.t(), scale, None)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, x[:, :1].expand(1, 16), atol=0, rtol=0)
