# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# UT for the dynamic per-token symmetric int8 activation quant kernel.
# The reference is an eager copy of vLLM's dynamic_per_token_int8_quant_ref;
# the kernel is expected to match it exactly (codes and scales). 16-bit random
# inputs of these sizes contain exact x / scale == q + 0.5 ties (about 1 code in
# 8000 at 512x5120 bf16), which is what the half-even handling is checked on.

import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401 – registers torch.ops._C.*

DEVICE = "xpu"

MINI_PYTEST_PARAMS = {
    "default": {
        "m,k": [(7, 100)],
    },
}


def ref_dynamic_per_token_int8_quant(x: torch.Tensor):
    x2 = x.reshape(-1, x.shape[-1])
    mn = torch.min(x2, dim=-1)[0].to(torch.float32).unsqueeze(-1)
    mx = torch.max(x2, dim=-1)[0].to(torch.float32).unsqueeze(-1)
    s = (torch.maximum(mn.abs(), mx.abs()) / 127).clamp(min=1e-5).to(x.dtype)
    q = torch.clamp(torch.round(x2 / s.to(torch.float32)), -128, 127)
    return q.to(torch.int8).view(x.shape), s.view(x.shape[:-1] + (1, ))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m,k", [(1, 5120), (7, 17408), (512, 5120),
                                 (33, 100), (3, 8)])
@pytest.mark.parametrize("scale_f32", [True, False])
@torch.inference_mode()
def test_int8_quant_exact(dtype, m, k, scale_f32):
    torch.manual_seed(0)
    x = (torch.randn(m, k, device=DEVICE) * 3).to(dtype)
    x[0, :4] = 0
    if m > 2:
        x[2] = 0  # all-zero row -> scale clamped to 1e-5
    q_ref, s_ref = ref_dynamic_per_token_int8_quant(x)
    q = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty(m,
                    1,
                    device=DEVICE,
                    dtype=torch.float32 if scale_f32 else dtype)
    torch.ops._C.dynamic_per_token_scaled_int8_quant(q, x, s)
    assert torch.equal(q, q_ref)
    assert torch.equal(s.to(torch.float32), s_ref.to(torch.float32))
