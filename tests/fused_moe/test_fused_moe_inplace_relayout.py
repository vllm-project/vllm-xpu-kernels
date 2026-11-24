# SPDX-License-Identifier: Apache-2.0
"""XpuFusedMoe must relayout weights inside their existing storage.
"""
import pytest
import torch

from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe, _assign_inplace


@pytest.mark.parametrize("e,n,k", [(4, 256, 128)])
def test_relayout_keeps_storage_for_existing_views(e, n, k):
    torch.manual_seed(0)
    w13 = torch.nn.Parameter(
        torch.randn(e, 2 * n, k, dtype=torch.bfloat16, device="xpu") / 10,
        requires_grad=False)
    w2 = torch.nn.Parameter(
        torch.randn(e, k, n, dtype=torch.bfloat16, device="xpu") / 10,
        requires_grad=False)
    # Views held by an external owner, as EPLB does.
    w13_view, w2_view = w13.view(e, -1), w2.view(e, -1)
    ptrs = (w13.untyped_storage().data_ptr(), w2.untyped_storage().data_ptr())

    torch.xpu.synchronize()
    mem_before = torch.xpu.memory_allocated()
    XpuFusedMoe(w13=w13, w13_scales=None, w13_bias=None,
                w2=w2, w2_scales=None, w2_bias=None,
                n_experts_per_token=2, activation="silu", num_experts=e)
    torch.xpu.synchronize()

    assert (w13.untyped_storage().data_ptr(),
            w2.untyped_storage().data_ptr()) == ptrs
    assert torch.equal(w13_view, w13.data.reshape(e, -1))
    assert torch.equal(w2_view, w2.data.reshape(e, -1))
    assert torch.xpu.memory_allocated() == mem_before


def test_assign_inplace_reinterprets_same_size_dtype():
    p = torch.nn.Parameter(
        torch.arange(16, dtype=torch.uint8, device="xpu").view(2, 8),
        requires_grad=False)
    ptr = p.untyped_storage().data_ptr()
    new = torch.full((2, 8), -3, dtype=torch.int8, device="xpu")
    _assign_inplace(p, new)
    assert p.dtype == torch.int8
    assert p.untyped_storage().data_ptr() == ptr
    assert torch.equal(p.data, new)
