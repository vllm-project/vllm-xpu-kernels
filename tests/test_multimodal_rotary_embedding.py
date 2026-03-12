# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Multi-Modal Rotary Embedding (M-RoPE) SYCL kernel.

M-RoPE (used by e.g. Qwen2-VL) partitions the rotation dimensions into
sections so that each section can encode a different positional axis
(e.g. temporal / height / width for video/image tokens, or a single text
position for text tokens).

Tensor shapes
─────────────
positions      : [num_mrope_sections, num_tokens]  int64
query / key    : [num_tokens, num_heads * head_size]  or
                 [num_tokens, num_heads, head_size]
cos_sin_cache  : [max_position, rot_dim]  float
mrope_section  : [num_mrope_sections]  int32, on device;
                 values in embed_dim = rot_dim/2 units, must sum to embed_dim
"""

from typing import Optional

import pytest
import torch

import sys
import tests.register_ops as ops  # noqa: F401 – ensure custom ops are loaded


# ─── pure-Python reference ───────────────────────────────────────────────────

def _apply_rotary_emb_torch(x, cos, sin, is_neox_style):
    """Apply RoPE to a single head slice x[..., rot_dim]."""
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def _compute_cos_sin_cache(max_position: int, rot_dim: int,
                            base: float = 10000.0) -> torch.Tensor:
    inv_freq = 1.0 / (base ** (torch.arange(0, rot_dim, 2, dtype=torch.float)
                                / rot_dim))
    t = torch.arange(max_position, dtype=torch.float)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_pos, rot_dim//2]
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)  # [max_pos, rot_dim]


def _ref_multimodal_rotary_embedding(
    positions: torch.Tensor,         # [num_sections, num_tokens]  CPU int64
    query: torch.Tensor,             # [num_tokens, num_heads, head_size]  CPU
    key: Optional[torch.Tensor],     # same or None
    cos_sin_cache: torch.Tensor,     # [max_position, rot_dim]  CPU float
    is_neox_style: bool,
    mrope_section: list[int],        # Python list, values in embed_dim units
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    num_sections = len(mrope_section)
    num_tokens = positions.shape[1]
    rot_dim = cos_sin_cache.shape[1]
    embed_dim = rot_dim // 2

    q_out = query.clone().float()
    k_out = key.clone().float() if key is not None else None

    # cumulative section boundaries [0, s0, s0+s1, ...]
    bounds = [0]
    for s in mrope_section:
        bounds.append(bounds[-1] + s)

    for sec in range(num_sections):
        lo, hi = bounds[sec], bounds[sec + 1]  # rot_offset range [lo, hi)
        for tok in range(num_tokens):
            pos = positions[sec, tok].item()
            # cos/sin for this token position, shape [embed_dim]
            cos_full = cos_sin_cache[pos, :embed_dim]     # [embed_dim]
            sin_full = cos_sin_cache[pos, embed_dim:]     # [embed_dim]
            cos_sec = cos_full[lo:hi]  # [section_size]
            sin_sec = sin_full[lo:hi]

            # Apply rotation only to slice [lo:hi] of the rotation dims.
            if is_neox_style:
                # NeoX: dim i pairs with dim embed_dim+i.
                # Section [lo, hi) rotates pairs (lo+j, embed_dim+lo+j).
                q1 = q_out[tok, :, lo:hi].clone()                        # [heads, sec_size]
                q2 = q_out[tok, :, embed_dim + lo:embed_dim + hi].clone() # [heads, sec_size]
                q_out[tok, :, lo:hi]                     = q1 * cos_sec - q2 * sin_sec
                q_out[tok, :, embed_dim + lo:embed_dim + hi] = q2 * cos_sec + q1 * sin_sec
                if k_out is not None:
                    k1 = k_out[tok, :, lo:hi].clone()
                    k2 = k_out[tok, :, embed_dim + lo:embed_dim + hi].clone()
                    k_out[tok, :, lo:hi]                     = k1 * cos_sec - k2 * sin_sec
                    k_out[tok, :, embed_dim + lo:embed_dim + hi] = k2 * cos_sec + k1 * sin_sec
            else:
                # GPT-J: pairs are (2i, 2i+1) — the flat indices are 2*lo..2*hi
                q_slice = q_out[tok, :, 2 * lo:2 * hi]   # [heads, 2*sec_size]
                q_out[tok, :, 2 * lo:2 * hi] = _apply_rotary_emb_torch(
                    q_slice, cos_sec, sin_sec, is_neox_style=False)
                if k_out is not None:
                    k_out[tok, :, 2 * lo:2 * hi] = _apply_rotary_emb_torch(
                        k_out[tok, :, 2 * lo:2 * hi], cos_sec, sin_sec,
                        is_neox_style=False)

    return q_out, k_out


# ─── helpers ─────────────────────────────────────────────────────────────────

def _run_kernel(device, positions, query, key, cos_sin_cache, is_neox_style,
                mrope_section_list):
    """Call the XPU kernel and return (query_out, key_out) on CPU."""
    q_xpu = query.clone().to(device=device)
    k_xpu = key.clone().to(device=device) if key is not None else None
    pos_xpu = positions.to(device=device)
    cache_xpu = cos_sin_cache.to(device=device, dtype=query.dtype)

    # head_size = last dim of query when viewed as [tokens, heads, head_size]
    head_size = q_xpu.shape[-1]

    # mrope_section is passed directly as a Python list of ints.
    ops.multimodal_rotary_embedding(
        pos_xpu, q_xpu, k_xpu, head_size, cache_xpu, is_neox_style,
        mrope_section_list)

    return q_xpu.cpu().float(), (k_xpu.cpu().float()
                                 if k_xpu is not None else None)


# ─── test parameters ─────────────────────────────────────────────────────────

MINI_PYTEST_PARAMS = {
    "default": {
        "max_position": [64],
        "head_size": [32],
        "num_tokens": [8],
    }
}


@pytest.mark.parametrize("device", ["xpu"])
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("use_key", [True, False])
@pytest.mark.parametrize(
    "head_size,rot_dim,mrope_section",
    [
        # 3-section M-RoPE (typical for Qwen2-VL with head_size=128)
        (64, 64, [8, 12, 12]),   # sum=32=embed_dim
        (32, 32, [4, 4, 8]),     # sum=16=embed_dim
        # single-section M-RoPE must match standard RoPE
        (32, 32, [16]),
    ],
)
@pytest.mark.parametrize("num_tokens", [1, 16, 128])
def test_multimodal_rotary_embedding(device, is_neox_style, use_key,
                                     head_size, rot_dim, mrope_section,
                                     num_tokens):
    max_position = 512
    num_heads = 4
    num_kv_heads = 2
    base = 10000.0
    num_sections = len(mrope_section)

    cos_sin_cache = _compute_cos_sin_cache(max_position, rot_dim, base)

    # positions: different per section to exercise M-RoPE routing
    positions = torch.stack([
        torch.randint(0, max_position, (num_tokens,))
        for _ in range(num_sections)
    ])  # [num_sections, num_tokens]

    query = torch.randn(num_tokens, num_heads, head_size)
    key = torch.randn(num_tokens, num_kv_heads, head_size) if use_key else None

    # ── reference ──
    ref_q, ref_k = _ref_multimodal_rotary_embedding(
        positions, query, key, cos_sin_cache, is_neox_style, mrope_section)

    # ── kernel ──
    # Kernel accepts [num_tokens, num_heads, head_size] layout.
    xpu_q, xpu_k = _run_kernel(device, positions, query, key,
                                cos_sin_cache, is_neox_style, mrope_section)

    torch.testing.assert_close(xpu_q, ref_q, atol=1e-4, rtol=1e-4)
    if use_key:
        torch.testing.assert_close(xpu_k, ref_k, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", ["xpu"])
@pytest.mark.parametrize("is_neox_style", [True, False])
def test_mrope_matches_standard_rope_for_text_tokens(device, is_neox_style):
    """When num_sections=1 and section covers all embed_dim, M-RoPE and
    standard RoPE must produce identical results."""
    max_position = 256
    num_tokens = 32
    num_heads = 4
    head_size = 32
    rot_dim = 32
    embed_dim = rot_dim // 2
    base = 10000.0

    cos_sin_cache = _compute_cos_sin_cache(max_position, rot_dim, base)

    positions_1d = torch.randint(0, max_position, (num_tokens,))
    positions_mrope = positions_1d.unsqueeze(0)  # [1, num_tokens]

    query = torch.randn(num_tokens, num_heads, head_size)
    key = torch.randn(num_tokens, num_heads, head_size)

    # --- standard RoPE via XPU op ---
    q_std = query.clone().to(device)
    k_std = key.clone().to(device)
    pos_std = positions_1d.to(device)
    cache_xpu = cos_sin_cache.to(device)
    ops.rotary_embedding(
        pos_std, q_std, k_std, head_size, cache_xpu, is_neox_style)

    # --- M-RoPE with single section ---
    mrope_section = [embed_dim]
    q_m, k_m = _run_kernel(device, positions_mrope, query, key,
                            cos_sin_cache, is_neox_style, mrope_section)

    torch.testing.assert_close(q_m, q_std.cpu().float(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(k_m, k_std.cpu().float(), atol=1e-4, rtol=1e-4)
