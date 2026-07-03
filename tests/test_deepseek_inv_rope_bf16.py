# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for deepseek_inv_rope_bf16 SYCL kernel.

Tests the fused inverse GPT-J RoPE + group-major layout transform kernel
against a pure-PyTorch reference implementation.

Run:
    pytest tests/test_deepseek_inv_rope_bf16.py -v
"""
import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

DEVICE = torch.device("xpu")

# DeepSeek-V4 default dimensions
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
HALF_ROPE = ROPE_DIM // 2


def reference_inv_rope_bf16(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = NOPE_DIM,
    rope_dim: int = ROPE_DIM,
) -> torch.Tensor:
    """Pure PyTorch reference for inverse GPT-J RoPE.

    Input:  o [T, H, D] bf16
    Output: [T, G, hpg*D] bf16
    """
    T, H, D = o.shape
    half_rope = rope_dim // 2
    d = heads_per_group * D

    # Reshape: [T, H, D] -> [T, G, hpg, D]
    o_4d = o.view(T, n_groups, heads_per_group, D).float()

    # Lookup cos/sin in fp32
    cos_sin = cos_sin_cache[positions]  # [T, rope_dim]
    cos = cos_sin[:, :half_rope]  # [T, half_rope]
    sin = cos_sin[:, half_rope:]  # [T, half_rope]

    # Apply inverse GPT-J RoPE on the last rope_dim elements
    # GPT-J interleaved: pairs at (even, odd) positions
    rope_region = o_4d[..., nope_dim:]  # [T, G, hpg, rope_dim]

    x_even = rope_region[..., ::2]   # [T, G, hpg, half_rope]
    x_odd = rope_region[..., 1::2]   # [T, G, hpg, half_rope]

    # Broadcast cos/sin: [T, 1, 1, half_rope]
    cos_b = cos[:, None, None, :]
    sin_b = sin[:, None, None, :]

    # Inverse rotation
    new_even = x_even * cos_b + x_odd * sin_b
    new_odd = x_odd * cos_b - x_even * sin_b

    # Interleave back
    out_rope = torch.stack([new_even, new_odd], dim=-1).flatten(-2)

    # Combine NoPE + rotated RoPE
    result = torch.cat([o_4d[..., :nope_dim], out_rope], dim=-1)

    # Reshape to [T, G, hpg*D] and cast back to bf16
    return result.reshape(T, n_groups, d).to(torch.bfloat16)


@pytest.mark.parametrize("num_tokens", [1, 4, 8, 32, 128, 512])
@pytest.mark.parametrize(
    "num_heads,n_groups",
    [(64, 8), (32, 4), (128, 8)],
    ids=["H64_G8", "H32_G4", "H128_G8"],
)
@pytest.mark.parametrize("seed", [0, 42])
def test_correctness(num_tokens, num_heads, n_groups, seed):
    """Compare SYCL kernel output against PyTorch reference."""
    torch.manual_seed(seed)
    heads_per_group = num_heads // n_groups
    max_pos = 4096

    o = torch.randn(num_tokens, num_heads, HEAD_DIM,
                    dtype=torch.bfloat16, device=DEVICE)
    positions = torch.randint(0, max_pos, (num_tokens,),
                              dtype=torch.int64, device=DEVICE)
    cos_sin_cache = torch.randn(max_pos, ROPE_DIM,
                                dtype=torch.float32, device=DEVICE)

    # Reference
    ref = reference_inv_rope_bf16(o, positions, cos_sin_cache,
                                  n_groups, heads_per_group)

    # SYCL kernel
    out = torch.ops._xpu_C.deepseek_inv_rope_bf16(
        o, positions, cos_sin_cache,
        n_groups, heads_per_group, NOPE_DIM, ROPE_DIM)

    # Check shapes
    d = heads_per_group * HEAD_DIM
    assert out.shape == (num_tokens, n_groups, d)
    assert ref.shape == (num_tokens, n_groups, d)

    # bf16 has ~0.004 relative error; kernel does fp32 rotation then casts
    # back so max error should be within bf16 representable range
    max_diff = (out.float() - ref.float()).abs().max().item()
    mean_diff = (out.float() - ref.float()).abs().mean().item()

    # NoPE region should be exact (just a copy)
    nope_out = out.view(num_tokens, n_groups, heads_per_group, HEAD_DIM)
    nope_ref = ref.view(num_tokens, n_groups, heads_per_group, HEAD_DIM)
    nope_max_diff = (nope_out[..., :NOPE_DIM].float() -
                     nope_ref[..., :NOPE_DIM].float()).abs().max().item()
    assert nope_max_diff == 0.0, (
        f"NoPE region should be exact copy, got max_diff={nope_max_diff}")

    # RoPE region: allow bf16 rounding. The kernel computes x*cos + y*sin in
    # fp32 then casts to bf16. The reference does the same but PyTorch may use
    # FMA which causes up to 2 ULP difference at rounding boundaries.
    # bf16 ULP at magnitude 1.0 = 2^{-7} = 0.0078125; allow up to 4 ULP.
    assert max_diff <= 0.035, (
        f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")


@pytest.mark.parametrize("num_tokens", [1, 4, 64, 8192])
def test_output_layout(num_tokens):
    """Verify output is contiguous [T, G, D] tensor."""
    num_heads, n_groups = 64, 8
    heads_per_group = num_heads // n_groups
    max_pos = 4096

    o = torch.randn(num_tokens, num_heads, HEAD_DIM,
                    dtype=torch.bfloat16, device=DEVICE)
    positions = torch.randint(0, max_pos, (num_tokens,),
                              dtype=torch.int64, device=DEVICE)
    cos_sin_cache = torch.randn(max_pos, ROPE_DIM,
                                dtype=torch.float32, device=DEVICE)

    out = torch.ops._xpu_C.deepseek_inv_rope_bf16(
        o, positions, cos_sin_cache,
        n_groups, heads_per_group, NOPE_DIM, ROPE_DIM)

    d = heads_per_group * HEAD_DIM
    assert out.shape == (num_tokens, n_groups, d)
    assert out.is_contiguous(), (
        f"Output should be contiguous, stride={out.stride()}")
    assert out.dtype == torch.bfloat16


def test_zero_tokens():
    """Edge case: zero tokens should return empty tensor."""
    num_heads, n_groups = 64, 8
    heads_per_group = num_heads // n_groups
    max_pos = 4096

    o = torch.empty(0, num_heads, HEAD_DIM,
                    dtype=torch.bfloat16, device=DEVICE)
    positions = torch.empty(0, dtype=torch.int64, device=DEVICE)
    cos_sin_cache = torch.randn(max_pos, ROPE_DIM,
                                dtype=torch.float32, device=DEVICE)

    out = torch.ops._xpu_C.deepseek_inv_rope_bf16(
        o, positions, cos_sin_cache,
        n_groups, heads_per_group, NOPE_DIM, ROPE_DIM)

    d = heads_per_group * HEAD_DIM
    assert out.shape == (0, n_groups, d)


def test_positions_independence():
    """Different positions produce different RoPE results but same NoPE."""
    num_tokens = 4
    num_heads, n_groups = 64, 8
    heads_per_group = num_heads // n_groups
    max_pos = 4096

    o = torch.randn(num_tokens, num_heads, HEAD_DIM,
                    dtype=torch.bfloat16, device=DEVICE)
    cos_sin_cache = torch.randn(max_pos, ROPE_DIM,
                                dtype=torch.float32, device=DEVICE)

    pos1 = torch.zeros(num_tokens, dtype=torch.int64, device=DEVICE)
    pos2 = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE) + 100

    out1 = torch.ops._xpu_C.deepseek_inv_rope_bf16(
        o, pos1, cos_sin_cache,
        n_groups, heads_per_group, NOPE_DIM, ROPE_DIM)
    out2 = torch.ops._xpu_C.deepseek_inv_rope_bf16(
        o, pos2, cos_sin_cache,
        n_groups, heads_per_group, NOPE_DIM, ROPE_DIM)

    # NoPE regions should be identical (position-independent)
    out1_4d = out1.view(num_tokens, n_groups, heads_per_group, HEAD_DIM)
    out2_4d = out2.view(num_tokens, n_groups, heads_per_group, HEAD_DIM)
    assert torch.equal(out1_4d[..., :NOPE_DIM], out2_4d[..., :NOPE_DIM])

    # RoPE regions should differ
    rope_diff = (out1_4d[..., NOPE_DIM:].float() -
                 out2_4d[..., NOPE_DIM:].float()).abs().max().item()
    assert rope_diff > 0.0, "RoPE region should differ with different positions"
