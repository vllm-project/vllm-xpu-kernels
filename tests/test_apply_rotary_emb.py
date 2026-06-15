# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for apply_rotary_emb SYCL kernel and Python wrapper.

Validates correctness against a pure-PyTorch reference implementation
for both GPT-J (interleaved) and GPT-NeoX styles.
"""

import pytest
import torch

try:
    import tests.register_ops as ops  # noqa: F401
except (ImportError, ModuleNotFoundError):
    import vllm_xpu_kernels._xpu_C  # noqa: F401

from vllm_xpu_kernels.rotary import apply_rotary_emb

# Override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [16],
        "num_heads": [8],
        "head_dim": [64],
    },
}


def _ref_rotary_emb_interleaved(x, cos, sin):
    """Reference: GPT-J style (is_neox=False), pairs are (2i, 2i+1)."""
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def _ref_rotary_emb_neox(x, cos, sin):
    """Reference: GPT-NeoX style (is_neox=True), first half + second half."""
    half = x.shape[-1] // 2
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1 = x[..., :half]
    x2 = x[..., half:]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


@pytest.mark.parametrize("device", ["xpu"])
@pytest.mark.parametrize("is_neox", [False, True], ids=["interleaved", "neox"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "num_tokens,num_heads,head_dim",
    [
        (1, 1, 64),
        (16, 8, 64),
        (16, 8, 128),
        (128, 40, 128),
        (75600, 40, 128),
    ],
    ids=["1x1x64", "16x8x64", "16x8x128", "128x40x128", "75600x40x128"],
)
def test_apply_rotary_emb_wrapper(
    device, is_neox, dtype, num_tokens, num_heads, head_dim
):
    """Test the Python wrapper (vllm_xpu_kernels.rotary.apply_rotary_emb)."""
    torch.manual_seed(42)
    half_rot = head_dim // 2

    x = torch.randn(num_tokens, num_heads, head_dim, device=device, dtype=dtype)
    cos = torch.randn(num_tokens, half_rot, device=device, dtype=dtype)
    sin = torch.randn(num_tokens, half_rot, device=device, dtype=dtype)

    # Kernel
    out = apply_rotary_emb(x, cos, sin, is_neox=is_neox)

    # Reference
    if is_neox:
        ref = _ref_rotary_emb_neox(x, cos, sin)
    else:
        ref = _ref_rotary_emb_interleaved(x, cos, sin)

    atol = 1e-5 if dtype == torch.float32 else 1e-2
    rtol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", ["xpu"])
@pytest.mark.parametrize("is_neox", [False, True], ids=["interleaved", "neox"])
def test_apply_rotary_emb_4d_input(device, is_neox):
    """Test wrapper handles 4D input [batch, seq, heads, dim]."""
    torch.manual_seed(123)
    batch, seq, heads, dim = 2, 100, 8, 64
    half_rot = dim // 2

    x = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.float32)
    cos = torch.randn(seq, half_rot, device=device, dtype=torch.float32)
    sin = torch.randn(seq, half_rot, device=device, dtype=torch.float32)

    out = apply_rotary_emb(x, cos, sin, is_neox=is_neox)
    assert out.shape == x.shape

    # Verify against per-batch reference
    for b in range(batch):
        x_b = x[b]  # [seq, heads, dim]
        out_b = apply_rotary_emb(x_b, cos, sin, is_neox=is_neox)
        torch.testing.assert_close(out[b], out_b, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", ["xpu"])
def test_apply_rotary_emb_cos_sin_broadcast(device):
    """Test wrapper broadcasts cos/sin from [seq, D/2] to [batch*seq, D/2]."""
    torch.manual_seed(456)
    batch, seq, heads, dim = 3, 50, 4, 64
    half_rot = dim // 2

    x = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.float32)
    cos = torch.randn(seq, half_rot, device=device, dtype=torch.float32)
    sin = torch.randn(seq, half_rot, device=device, dtype=torch.float32)

    out = apply_rotary_emb(x, cos, sin, is_neox=False)

    # Manual: expand cos/sin then compute reference
    cos_expanded = cos.unsqueeze(-2)
    sin_expanded = sin.unsqueeze(-2)
    for b in range(batch):
        x_b = x[b]
        x1 = x_b[..., ::2]
        x2 = x_b[..., 1::2]
        ref_b = torch.stack(
            (x1 * cos_expanded - x2 * sin_expanded,
             x2 * cos_expanded + x1 * sin_expanded),
            dim=-1,
        ).flatten(-2)
        torch.testing.assert_close(out[b], ref_b, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", ["xpu"])
@pytest.mark.parametrize("is_neox", [False, True], ids=["interleaved", "neox"])
def test_apply_rotary_emb_kernel_direct(device, is_neox):
    """Test the raw _C kernel directly with 3D tensors."""
    torch.manual_seed(789)
    num_tokens, num_heads, head_dim = 32, 8, 64
    half_rot = head_dim // 2

    x = torch.randn(
        num_tokens, num_heads, head_dim, device=device, dtype=torch.float32
    ).contiguous()
    cos = torch.randn(
        num_tokens, half_rot, device=device, dtype=torch.float32
    ).contiguous()
    sin = torch.randn(
        num_tokens, half_rot, device=device, dtype=torch.float32
    ).contiguous()
    out = torch.empty_like(x)

    torch.ops._xpu_C.apply_rotary_emb(out, x, cos, sin, is_neox)

    if is_neox:
        ref = _ref_rotary_emb_neox(x, cos, sin)
    else:
        ref = _ref_rotary_emb_interleaved(x, cos, sin)

    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
