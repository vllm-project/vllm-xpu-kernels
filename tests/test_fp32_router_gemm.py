# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for the XPU SYCL ``fp32_router_gemm`` kernel.

out[m, n] = sum_k mat_a[m, k] * mat_b[n, k], computed in fp32. mat_a is bf16 or
fp32; mat_b is fp32. Validated against a fp32 ``torch.matmul`` reference.
"""

import pytest
import torch

from tests.register_ops import fp32_router_gemm

DEVICE = torch.device("xpu")


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 32])
@pytest.mark.parametrize("num_experts", [256])
@pytest.mark.parametrize("hidden_dim", [3072])
@pytest.mark.parametrize("act_dtype", [torch.bfloat16, torch.float32])
def test_fp32_router_gemm(num_tokens, num_experts, hidden_dim, act_dtype):
    torch.manual_seed(0)
    mat_a = torch.randn(num_tokens, hidden_dim, device=DEVICE, dtype=act_dtype)
    mat_b = torch.randn(num_experts, hidden_dim, device=DEVICE, dtype=torch.float32)

    out = fp32_router_gemm(mat_a, mat_b)
    torch.xpu.synchronize()

    assert out.shape == (num_tokens, num_experts)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()

    ref = torch.matmul(mat_a.float(), mat_b.t())
    # bf16 activation loses precision relative to the fp32 reference.
    atol = 1e-1 if act_dtype == torch.bfloat16 else 1e-3
    rtol = 1e-2 if act_dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


def test_fp32_router_gemm_rejects_too_many_tokens():
    mat_a = torch.randn(64, 128, device=DEVICE, dtype=torch.float32)
    mat_b = torch.randn(8, 128, device=DEVICE, dtype=torch.float32)
    with pytest.raises(Exception):
        fp32_router_gemm(mat_a, mat_b)
