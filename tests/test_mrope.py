# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the MRoPE (Multi-head Rotary Positional Embedding) SYCL kernel.

The reference implementation mirrors
vllm/model_executor/layers/rotary_embedding/mrope.py::MRotaryEmbedding.forward_native
using the same section-split logic for T/H/W multimodal positions.
"""

import pytest
import torch

from tests.register_ops import mrope

DEVICE = torch.device("xpu")


# ---------------------------------------------------------------------------
# Mini-pytest override (used by the internal CI harness)
# ---------------------------------------------------------------------------
MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [4],
        "seed": [42],
        "q_num_head,k_num_head": [(16, 2)],
        "rotary_dim": [64],
        "is_neox": [True],
    },
}


# ---------------------------------------------------------------------------
# Reference (CPU, float32) implementation
# ---------------------------------------------------------------------------

def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    """Rotate half (neox style): [-x2, x1]."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    """Rotate interleaved (gptj style): adjacent pairs (-x1, x0)."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def ref_mrope(
    positions: torch.Tensor,       # [3, num_tokens]  int64  CPU
    query: torch.Tensor,            # [num_tokens, q_num_head, head_size]  CPU fp32
    key: torch.Tensor,              # [num_tokens, k_num_head, head_size]  CPU fp32
    cos_sin_cache: torch.Tensor,    # [max_pos, rotary_dim]  CPU fp32
    rotary_dim: int,
    mrope_section: list[int],       # [t, h, w]  with t+h+w == rotary_dim//2
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for the XPU mrope kernel."""
    num_tokens = positions.shape[1]
    t_section, h_section, w_section = mrope_section
    half_rd = rotary_dim // 2

    # --- Build per-token cos/sin   shape [num_tokens, rotary_dim/2] each ---
    # For each half-dim i, select position from the appropriate row:
    #   i in [0, t_section)              → T row (positions[0])
    #   i in [t_section, t_section+h)    → H row (positions[1])
    #   else                             → W row (positions[2])
    t_boundary = t_section
    h_boundary = t_section + h_section

    # full cos_sin lookup for each of the 3 rows
    cos_sin_t = cos_sin_cache[positions[0]]   # [num_tokens, rotary_dim]
    cos_sin_h = cos_sin_cache[positions[1]]
    cos_sin_w = cos_sin_cache[positions[2]]

    cos_t, sin_t = cos_sin_t.chunk(2, dim=-1)   # [num_tokens, rotary_dim/2]
    cos_h, sin_h = cos_sin_h.chunk(2, dim=-1)
    cos_w, sin_w = cos_sin_w.chunk(2, dim=-1)

    # Compose cos/sin along the half-dim axis using section masks
    dim_idx = torch.arange(half_rd)
    t_mask = dim_idx < t_boundary                            # [half_rd]
    h_mask = (dim_idx >= t_boundary) & (dim_idx < h_boundary)
    w_mask = dim_idx >= h_boundary

    cos = (cos_t * t_mask + cos_h * h_mask + cos_w * w_mask)  # [num_tokens, half_rd]
    sin = (sin_t * t_mask + sin_h * h_mask + sin_w * w_mask)

    # Expand cos/sin for head broadcasting: [num_tokens, 1, rotary_dim]
    if is_neox_style:
        cos_full = cos.repeat(1, 2).unsqueeze(1)   # [num_tokens, 1, rotary_dim]
        sin_full = sin.repeat(1, 2).unsqueeze(1)
    else:
        # interleaved: each pair shares the same (cos, sin)
        cos_full = cos.repeat_interleave(2, dim=-1).unsqueeze(1)
        sin_full = sin.repeat_interleave(2, dim=-1).unsqueeze(1)

    rotate_fn = _rotate_neox if is_neox_style else _rotate_gptj

    def apply(x):
        x_rot  = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x_rot  = x_rot * cos_full + rotate_fn(x_rot) * sin_full
        return torch.cat([x_rot, x_pass], dim=-1)

    return apply(query), apply(key)


# ---------------------------------------------------------------------------
# Helper: build a random query / key with optional non-contiguous layout
# ---------------------------------------------------------------------------
def _make_qk(
    num_tokens: int,
    num_head: int,
    head_size: int,
    head_pad: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return [num_tokens, num_head, head_size] slice from a padded tensor."""
    padded = torch.randn(num_tokens, num_head, head_size + head_pad,
                         device=DEVICE, dtype=dtype)
    return padded[..., :head_size]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMRope:

    # ---- 1. Multimodal path (positions.ndim == 2) --------------------------

    @pytest.mark.parametrize("seed", [42, 123, 999])
    @pytest.mark.parametrize("dtype",
                             [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("num_tokens", [1, 4, 32, 128])
    @pytest.mark.parametrize("q_num_head,k_num_head", [(16, 2), (28, 4), (32, 1)])
    @pytest.mark.parametrize("rotary_dim", [64, 128])
    @pytest.mark.parametrize("is_neox", [True, False])
    @pytest.mark.parametrize("mrope_section", [
        [10, 10, 12],   # sum=32 → rotary_dim=64
        [14, 14, 36],   # sum=64 → rotary_dim=128
    ])
    def test_mrope_multimodal(
        self,
        seed,
        dtype,
        num_tokens,
        q_num_head,
        k_num_head,
        rotary_dim,
        is_neox,
        mrope_section,
    ):
        if sum(mrope_section) != rotary_dim // 2:
            pytest.skip("mrope_section sum does not match rotary_dim/2")

        torch.manual_seed(seed)
        max_pos = 10_000
        head_size = rotary_dim   # rotary_dim == head_size for these tests

        # T, H, W positions (independent per token)
        positions = torch.randint(0, max_pos, (3, num_tokens),
                                  device=DEVICE, dtype=torch.int64)

        cos_sin_cache = torch.randn(max_pos, rotary_dim,
                                    device=DEVICE, dtype=dtype)

        query = _make_qk(num_tokens, q_num_head, head_size, 0, dtype)
        key   = _make_qk(num_tokens, k_num_head, head_size, 0, dtype)

        # XPU kernel
        q_out, k_out = mrope(positions, query, key,
                             cos_sin_cache, rotary_dim, mrope_section, is_neox)

        # Reference (CPU, fp32)
        ref_q, ref_k = ref_mrope(
            positions.cpu().long(),
            query.cpu().float().contiguous(),
            key.cpu().float().contiguous(),
            cos_sin_cache.cpu().float(),
            rotary_dim, mrope_section, is_neox,
        )

        # The kernel computes in float32 internally and rounds to the output
        # dtype. Round the reference the same way for a fair comparison.
        torch.testing.assert_close(
            q_out.float().cpu(), ref_q.to(dtype).float(),
            atol=1e-5, rtol=1e-5,
            msg=f"query mismatch: dtype={dtype}, neox={is_neox}, "
                f"section={mrope_section}, tokens={num_tokens}")
        torch.testing.assert_close(
            k_out.float().cpu(), ref_k.to(dtype).float(),
            atol=1e-5, rtol=1e-5,
            msg=f"key mismatch: dtype={dtype}, neox={is_neox}, "
                f"section={mrope_section}, tokens={num_tokens}")

    # ---- 2. Non-contiguous (padded) head layout ----------------------------

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("q_head_pad,k_head_pad", [(64, 64), (128, 0)])
    @pytest.mark.parametrize("is_neox", [True, False])
    def test_mrope_noncontiguous_heads(
        self, dtype, q_head_pad, k_head_pad, is_neox
    ):
        torch.manual_seed(0)
        num_tokens, q_num_head, k_num_head = 8, 16, 2
        rotary_dim, head_size = 64, 64
        mrope_section = [10, 10, 12]
        max_pos = 10_000

        positions = torch.randint(0, max_pos, (3, num_tokens),
                                  device=DEVICE, dtype=torch.int64)
        cos_sin_cache = torch.randn(max_pos, rotary_dim,
                                    device=DEVICE, dtype=dtype)

        query = _make_qk(num_tokens, q_num_head, head_size, q_head_pad, dtype)
        key   = _make_qk(num_tokens, k_num_head, head_size, k_head_pad, dtype)

        q_out, k_out = mrope(positions, query, key,
                             cos_sin_cache, rotary_dim, mrope_section, is_neox)

        ref_q, ref_k = ref_mrope(
            positions.cpu().long(),
            query.cpu().float().contiguous(),
            key.cpu().float().contiguous(),
            cos_sin_cache.cpu().float(),
            rotary_dim, mrope_section, is_neox,
        )

        torch.testing.assert_close(q_out.float().cpu(), ref_q.to(dtype).float(), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k_out.float().cpu(), ref_k.to(dtype).float(), atol=1e-5, rtol=1e-5)

    # ---- 3. Partial rotation: rotary_dim < head_size  ----------------------
    #  Dims [rotary_dim, head_size) must be copied through unchanged.

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("rotary_dim,head_size", [(64, 128), (64, 256)])
    @pytest.mark.parametrize("is_neox", [True, False])
    def test_mrope_partial_rotation(
        self, dtype, rotary_dim, head_size, is_neox
    ):
        torch.manual_seed(7)
        num_tokens, q_num_head, k_num_head = 16, 8, 2
        mrope_section = [10, 10, 12]   # sum=32 == 64//2 ✓
        max_pos = 10_000

        positions = torch.randint(0, max_pos, (3, num_tokens),
                                  device=DEVICE, dtype=torch.int64)
        cos_sin_cache = torch.randn(max_pos, rotary_dim,
                                    device=DEVICE, dtype=dtype)

        query = torch.randn(num_tokens, q_num_head, head_size,
                            device=DEVICE, dtype=dtype)
        key   = torch.randn(num_tokens, k_num_head, head_size,
                            device=DEVICE, dtype=dtype)

        q_saved = query.clone()
        k_saved = key.clone()

        q_out, k_out = mrope(positions, query, key,
                             cos_sin_cache, rotary_dim, mrope_section, is_neox)

        ref_q, ref_k = ref_mrope(
            positions.cpu().long(),
            query.cpu().float().contiguous(),
            key.cpu().float().contiguous(),
            cos_sin_cache.cpu().float(),
            rotary_dim, mrope_section, is_neox,
        )

        torch.testing.assert_close(q_out.float().cpu(), ref_q.to(dtype).float(), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k_out.float().cpu(), ref_k.to(dtype).float(), atol=1e-5, rtol=1e-5)

        # Pass-through dims must be identical to the original input
        torch.testing.assert_close(
            q_out[..., rotary_dim:].cpu(), q_saved[..., rotary_dim:].cpu(),
            atol=0, rtol=0, msg="pass-through dims changed in query")
        torch.testing.assert_close(
            k_out[..., rotary_dim:].cpu(), k_saved[..., rotary_dim:].cpu(),
            atol=0, rtol=0, msg="pass-through dims changed in key")

    # ---- 4. Text-only path: all 3 position rows are identical --------------
    #  When T=H=W the result must match a plain RoPE rotation.

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("is_neox", [True, False])
    def test_mrope_text_only_equiv(self, dtype, is_neox):
        """With T==H==W, MRoPE is identical to plain RoPE."""
        torch.manual_seed(3)
        num_tokens, q_num_head, k_num_head = 32, 16, 2
        rotary_dim = 64
        mrope_section = [10, 10, 12]   # sum=32
        max_pos = 10_000

        # All three rows are the same 1-D positions
        pos_1d = torch.randint(0, max_pos, (num_tokens,),
                               device=DEVICE, dtype=torch.int64)
        positions = pos_1d.unsqueeze(0).expand(3, -1).contiguous()

        cos_sin_cache = torch.randn(max_pos, rotary_dim,
                                    device=DEVICE, dtype=dtype)
        query = torch.randn(num_tokens, q_num_head, rotary_dim,
                            device=DEVICE, dtype=dtype)
        key   = torch.randn(num_tokens, k_num_head, rotary_dim,
                            device=DEVICE, dtype=dtype)

        # mrope result
        q_mrope, k_mrope = mrope(positions, query, key,
                                 cos_sin_cache, rotary_dim, mrope_section, is_neox)

        # Reference plain RoPE: since all sections share the same position,
        # the section split does not matter
        ref_q, ref_k = ref_mrope(
            positions.cpu().long(),
            query.cpu().float().contiguous(),
            key.cpu().float().contiguous(),
            cos_sin_cache.cpu().float(),
            rotary_dim, mrope_section, is_neox,
        )

        torch.testing.assert_close(q_mrope.float().cpu(), ref_q.to(dtype).float(), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k_mrope.float().cpu(), ref_k.to(dtype).float(), atol=1e-5, rtol=1e-5)

    # ---- 5. Determinism: same input → same output --------------------------

    def test_mrope_deterministic(self):
        torch.manual_seed(0)
        num_tokens, q_num_head, k_num_head = 16, 8, 1
        rotary_dim = 64
        mrope_section = [10, 10, 12]
        max_pos = 1_000

        positions = torch.randint(0, max_pos, (3, num_tokens),
                                  device=DEVICE, dtype=torch.int64)
        cos_sin_cache = torch.randn(max_pos, rotary_dim,
                                    device=DEVICE, dtype=torch.bfloat16)
        query = torch.randn(num_tokens, q_num_head, rotary_dim,
                            device=DEVICE, dtype=torch.bfloat16)
        key   = torch.randn(num_tokens, k_num_head, rotary_dim,
                            device=DEVICE, dtype=torch.bfloat16)

        q1, k1 = mrope(positions, query, key, cos_sin_cache,
                        rotary_dim, mrope_section, True)
        q2, k2 = mrope(positions, query, key, cos_sin_cache,
                        rotary_dim, mrope_section, True)

        torch.testing.assert_close(q1, q2, atol=0, rtol=0)
        torch.testing.assert_close(k1, k2, atol=0, rtol=0)
