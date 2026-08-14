# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for fused_rope_norm_store_kv_fp8 kernel.

Validates the fused SYCL kernel against a pure-PyTorch reference implementation
that performs: RMSNorm + NeoX RoPE + KV cache write (FP8) + Q FP8 quant.
"""

import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401 - registers torch.ops._xpu_C ops

DEVICE = "xpu"
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0


# ---------------------------------------------------------------------------
# Reference implementation (pure PyTorch, mirrors rope_norm.py logic)
# ---------------------------------------------------------------------------

def _rms_norm(x: torch.Tensor, weight: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm returning float32 to match kernel precision."""
    x_fp32 = x.float()
    rms = torch.sqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_fp32 / rms) * weight.float()


def _apply_rotary_neox(x: torch.Tensor, cos: torch.Tensor,
                       sin: torch.Tensor) -> torch.Tensor:
    """NeoX-style RoPE. x: (T, H, D), cos/sin: (T, D/2).

    Returns float32 to match the kernel which stays in float32
    throughout (no bf16 rounding between RoPE and quant).
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat([o1, o2], dim=-1)


def _quantize_fp8_per_tensor(x: torch.Tensor,
                             scale: float) -> torch.Tensor:
    x_scaled = x.float() / scale
    return x_scaled.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def _quantize_fp8_per_token_per_head(q: torch.Tensor):
    """Returns (q_fp8, q_scale) with per-token-per-head dynamic quant."""
    q_fp32 = q.float()
    absmax = q_fp32.abs().amax(dim=-1)  # (T, H)
    q_scale = (absmax / FP8_MAX).clamp(min=1e-12)
    q_scaled = q_fp32 / q_scale.unsqueeze(-1)
    q_fp8 = q_scaled.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q_fp8, q_scale


def reference_rope_norm_store_kv_fp8(
    qkv, cos_sin_cache, positions, key_cache, value_cache, slot_mapping,
    num_heads, num_kv_heads, head_dim, k_scale_val, v_scale_val,
    q_norm_weight, k_norm_weight, qk_norm_policy, use_fp8,
):
    """Pure PyTorch reference for correctness comparison."""
    num_tokens = qkv.shape[0]
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    block_size = key_cache.shape[1]

    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = q.view(num_tokens, num_heads, head_dim)
    k = k.view(num_tokens, num_kv_heads, head_dim)
    v = v.view(num_tokens, num_kv_heads, head_dim)

    half_dim = head_dim // 2
    cos = cos_sin_cache[positions, :half_dim].float()
    sin = cos_sin_cache[positions, half_dim:].float()

    # NORM_THEN_ROPE = 2
    if qk_norm_policy == 2:
        if q_norm_weight is not None:
            q = _rms_norm(q, q_norm_weight)
        if k_norm_weight is not None:
            k = _rms_norm(k, k_norm_weight)

    q = _apply_rotary_neox(q, cos, sin)
    k = _apply_rotary_neox(k, cos, sin)

    # ROPE_THEN_NORM = 1
    if qk_norm_policy == 1:
        if q_norm_weight is not None:
            q = _rms_norm(q, q_norm_weight)
        if k_norm_weight is not None:
            k = _rms_norm(k, k_norm_weight)

    # Write KV to cache — [num_blocks, block_size, num_kv_heads, head_dim]
    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        if slot >= 0:
            bi = slot // block_size
            bo = slot % block_size
            if use_fp8:
                key_cache[bi, bo, :, :] = _quantize_fp8_per_tensor(
                    k[i], k_scale_val).view(torch.uint8)
                value_cache[bi, bo, :, :] = _quantize_fp8_per_tensor(
                    v[i], v_scale_val).view(torch.uint8)
            else:
                key_cache[bi, bo, :, :] = k[i].to(key_cache.dtype)
                value_cache[bi, bo, :, :] = v[i].to(value_cache.dtype)

    # Quantize Q
    if use_fp8:
        q_fp8, q_scale = _quantize_fp8_per_token_per_head(q)
    else:
        # Round to dtype (bf16) to match kernel's scalar_t store
        q_fp8 = q.to(qkv.dtype)
        q_scale = None

    return q_fp8, q_scale, key_cache, value_cache


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _build_cos_sin_cache(max_pos, head_dim, device="cpu"):
    half_dim = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (
        torch.arange(0, half_dim, dtype=torch.float32, device=device) /
        half_dim))
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

HEAD_DIM = 128
MAX_POS = 4096
BLOCK_SIZE = 16

CONFIGS = [
    # (num_tokens, num_heads, num_kv_heads, qk_norm_policy, use_fp8)
    (1, 16, 4, 0, True),     # No norm, FP8, GQA=4
    (4, 32, 4, 0, True),     # No norm, FP8, GQA=8
    (8, 16, 4, 1, True),     # ROPE_THEN_NORM, FP8
    (4, 32, 4, 2, True),     # NORM_THEN_ROPE, FP8
    (4, 16, 4, 0, False),    # No norm, no FP8 (bf16 passthrough)
    (32, 32, 8, 1, True),    # Larger batch, GQA=4, ROPE_THEN_NORM
]


@pytest.mark.parametrize(
    "num_tokens,num_heads,num_kv_heads,qk_norm_policy,use_fp8",
    CONFIGS,
    ids=[
        "t1_h16kv4_nonorm_fp8",
        "t4_h32kv4_nonorm_fp8",
        "t8_h16kv4_ropenorm_fp8",
        "t4_h32kv4_normrope_fp8",
        "t4_h16kv4_nonorm_bf16",
        "t32_h32kv8_ropenorm_fp8",
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("seed", [42])
@torch.inference_mode()
def test_fused_rope_norm_store_kv_fp8(
    num_tokens, num_heads, num_kv_heads, qk_norm_policy, use_fp8,
    dtype, seed,
):
    torch.manual_seed(seed)

    q_per_kv = num_heads // num_kv_heads
    assert q_per_kv in (4, 8), f"Unsupported GQA ratio: {q_per_kv}"

    q_size = num_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    total_dim = q_size + 2 * kv_size

    # Allocate inputs
    qkv = torch.randn(num_tokens, total_dim, dtype=dtype, device="cpu")
    positions = torch.randint(0, MAX_POS, (num_tokens,), dtype=torch.long,
                              device="cpu")
    cos_sin_cache = _build_cos_sin_cache(MAX_POS, HEAD_DIM, device="cpu")

    # KV cache: BSHD layout [num_blocks, block_size, num_kv_heads, head_dim]
    num_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
    kv_cache_dtype = torch.uint8 if use_fp8 else dtype
    key_cache_ref = torch.zeros(
        num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM,
        dtype=kv_cache_dtype, device="cpu")
    value_cache_ref = torch.zeros_like(key_cache_ref)
    key_cache_kernel = key_cache_ref.clone()
    value_cache_kernel = value_cache_ref.clone()

    # Slot mapping: sequential, no padding
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device="cpu")

    # Scales
    k_scale_val = 1.0
    v_scale_val = 1.0
    k_scale = torch.tensor([k_scale_val], dtype=torch.float32, device="cpu")
    v_scale = torch.tensor([v_scale_val], dtype=torch.float32, device="cpu")

    # Norm weights
    q_norm_weight = None
    k_norm_weight = None
    if qk_norm_policy != 0:
        q_norm_weight = torch.empty(HEAD_DIM, dtype=torch.float32,
                                    device="cpu").normal_(mean=1.0, std=0.1)
        k_norm_weight = torch.empty(HEAD_DIM, dtype=torch.float32,
                                    device="cpu").normal_(mean=1.0, std=0.1)

    # Q output buffer
    if use_fp8:
        out_q = torch.zeros(num_tokens, num_heads, HEAD_DIM,
                            dtype=torch.uint8, device="cpu")
    else:
        out_q = torch.zeros(num_tokens, num_heads, HEAD_DIM,
                            dtype=dtype, device="cpu")

    # --- Reference ---
    ref_q, ref_q_scale, ref_key_cache, ref_value_cache = \
        reference_rope_norm_store_kv_fp8(
            qkv.clone(), cos_sin_cache, positions,
            key_cache_ref, value_cache_ref, slot_mapping,
            num_heads, num_kv_heads, HEAD_DIM,
            k_scale_val, v_scale_val,
            q_norm_weight, k_norm_weight, qk_norm_policy, use_fp8,
        )

    # --- Kernel ---
    qkv_k = qkv.clone().to(DEVICE)
    cos_sin_cache_k = cos_sin_cache.to(DEVICE)
    positions_k = positions.to(DEVICE)
    key_cache_k = key_cache_kernel.to(DEVICE)
    value_cache_k = value_cache_kernel.to(DEVICE)
    slot_mapping_k = slot_mapping.to(DEVICE)
    out_q_k = out_q.to(DEVICE)
    k_scale_k = k_scale.to(DEVICE)
    v_scale_k = v_scale.to(DEVICE)
    q_norm_k = q_norm_weight.to(DEVICE) if q_norm_weight is not None else None
    k_norm_k = k_norm_weight.to(DEVICE) if k_norm_weight is not None else None

    q_scale_kernel = torch.ops._xpu_C.fused_rope_norm_store_kv_fp8(
        qkv_k, cos_sin_cache_k, positions_k,
        key_cache_k, value_cache_k, slot_mapping_k,
        out_q_k, num_heads, num_kv_heads, HEAD_DIM,
        k_scale_k, v_scale_k, q_norm_k, k_norm_k, qk_norm_policy, use_fp8,
    )

    # --- Compare results ---
    out_q_cpu = out_q_k.cpu()
    key_cache_cpu = key_cache_k.cpu()
    value_cache_cpu = value_cache_k.cpu()

    if use_fp8:
        # Compare Q scale: fp32 arithmetic with different reduction order
        # (tree vs serial) and FMA usage. Over 128-element RMSNorm reduction,
        # worst-case relative diff ≈ 128*eps_fp32/sqrt ≈ 5e-6.
        # rtol=1e-5 gives 2× safety margin.
        q_scale_cpu = q_scale_kernel.cpu()
        assert ref_q_scale is not None
        torch.testing.assert_close(
            q_scale_cpu, ref_q_scale,
            rtol=1e-5, atol=1e-7,
            msg="Q scale mismatch",
        )

        # Compare Q FP8 values: allow ±1 ULP difference
        ref_q_bytes = ref_q.view(torch.uint8)
        kernel_q_bytes = out_q_cpu.view(torch.uint8)
        q_diff = (ref_q_bytes.int() - kernel_q_bytes.int()).abs()
        max_q_diff = q_diff.max().item()
        assert max_q_diff <= 1, (
            f"Q FP8 mismatch: max ULP diff = {max_q_diff}")

        # Compare KV cache FP8 values: allow ±1 ULP
        for name, ref_c, kernel_c in [("K", ref_key_cache, key_cache_cpu),
                                       ("V", ref_value_cache, value_cache_cpu)]:
            ref_bytes = ref_c.view(torch.uint8)
            kernel_bytes = kernel_c.view(torch.uint8)
            diff = (ref_bytes.int() - kernel_bytes.int()).abs()
            max_diff = diff.max().item()
            assert max_diff <= 1, (
                f"{name} cache FP8 mismatch: max ULP diff = {max_diff}")
    else:
        # bf16 path: both reference and kernel store as bf16.
        # Difference comes only from FMA/reduction order in fp32
        # straddling a bf16 bucket boundary → at most 1 ULP of bf16.
        # bf16 ULP = 2^-7 ≈ 7.8e-3.
        BF16_ULP = 2**-7
        ref_q_flat = ref_q.view(num_tokens, num_heads, HEAD_DIM)
        kernel_q_flat = out_q_cpu.view(num_tokens, num_heads, HEAD_DIM)
        torch.testing.assert_close(
            kernel_q_flat.float(), ref_q_flat.float(),
            rtol=BF16_ULP, atol=BF16_ULP * 1e-4,
            msg="Q bf16 mismatch",
        )

        # KV cache: bf16 comparison (same 1 ULP tolerance)
        for name, ref_c, kernel_c in [("K", ref_key_cache, key_cache_cpu),
                                       ("V", ref_value_cache, value_cache_cpu)]:
            torch.testing.assert_close(
                kernel_c.float(), ref_c.float(),
                rtol=BF16_ULP, atol=BF16_ULP * 1e-4,
                msg=f"{name} cache bf16 mismatch",
            )

    print(f"  PASSED: tokens={num_tokens}, heads={num_heads}/"
          f"{num_kv_heads}, norm={qk_norm_policy}, fp8={use_fp8}")


@pytest.mark.parametrize("num_tokens", [8])
@torch.inference_mode()
def test_slot_mapping_with_padding(num_tokens):
    """Verify that slot_mapping=-1 (padding) tokens don't write to KV cache."""
    torch.manual_seed(7)

    num_heads, num_kv_heads = 16, 4
    dtype = torch.bfloat16
    q_size = num_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    total_dim = q_size + 2 * kv_size

    qkv = torch.randn(num_tokens, total_dim, dtype=dtype, device="cpu")
    positions = torch.randint(0, MAX_POS, (num_tokens,), dtype=torch.long)
    cos_sin_cache = _build_cos_sin_cache(MAX_POS, HEAD_DIM)

    num_blocks = 4
    key_cache = torch.zeros(
        num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM,
        dtype=torch.uint8, device="cpu")
    value_cache = torch.zeros_like(key_cache)

    # Mark half the tokens as padding (-1)
    slot_mapping = torch.arange(num_tokens, dtype=torch.long)
    slot_mapping[::2] = -1  # Every other token is padding

    k_scale = torch.tensor([1.0], dtype=torch.float32)
    v_scale = torch.tensor([1.0], dtype=torch.float32)
    out_q = torch.zeros(num_tokens, num_heads, HEAD_DIM, dtype=torch.uint8)

    # Run kernel
    qkv_k = qkv.to(DEVICE)
    key_cache_k = key_cache.to(DEVICE)
    value_cache_k = value_cache.to(DEVICE)
    torch.ops._xpu_C.fused_rope_norm_store_kv_fp8(
        qkv_k,
        cos_sin_cache.to(DEVICE),
        positions.to(DEVICE),
        key_cache_k,
        value_cache_k,
        slot_mapping.to(DEVICE),
        out_q.to(DEVICE),
        num_heads, num_kv_heads, HEAD_DIM,
        k_scale.to(DEVICE), v_scale.to(DEVICE),
        None, None, 0, True,
    )

    # The KV cache slots corresponding to padding should be zero
    key_result = key_cache_k.cpu()
    value_result = value_cache_k.cpu()
    for i in range(num_tokens):
        if slot_mapping[i] == -1:
            slot = i
            bi = slot // BLOCK_SIZE
            bo = slot % BLOCK_SIZE
            k_slice = key_result[bi, bo, :, :]
            v_slice = value_result[bi, bo, :, :]
            assert k_slice.sum() == 0, (
                f"Padding slot {i} was written to K cache!")
            assert v_slice.sum() == 0, (
                f"Padding slot {i} was written to V cache!")

    print(f"  PASSED: padding slot test with {num_tokens} tokens")


# ---------------------------------------------------------------------------
# Test: vLLM kv_cache layout compatibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kv_layout",
    ["xpu_attn", "flash_attn"],
    ids=["xpu_attn_2BSHD", "flash_attn_B2SHD"],
)
@torch.inference_mode()
def test_vllm_kv_cache_layout(kv_layout):
    """Verify the kernel produces identical results when called through
    vLLM-style unified kv_cache tensors with different dim-2 positions.

    XPU_ATTN:   kv_cache shape = (2, num_blocks, block_size, num_kv_heads, D)
    FLASH_ATTN: kv_cache shape = (num_blocks, 2, block_size, num_kv_heads, D)

    The caller splits via unbind(0) or unbind(1) respectively, yielding
    separate (num_blocks, block_size, num_kv_heads, D) views that the
    kernel writes into. The writes must land in the correct positions
    within the unified tensor.
    """
    torch.manual_seed(123)
    num_tokens, num_heads, num_kv_heads = 8, 32, 4
    dtype = torch.bfloat16

    q_size = num_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    total_dim = q_size + 2 * kv_size

    qkv = torch.randn(num_tokens, total_dim, dtype=dtype, device="cpu")
    positions = torch.randint(0, MAX_POS, (num_tokens,), dtype=torch.long)
    cos_sin_cache = _build_cos_sin_cache(MAX_POS, HEAD_DIM)
    slot_mapping = torch.arange(num_tokens, dtype=torch.long)

    num_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
    k_scale = torch.tensor([1.0], dtype=torch.float32)
    v_scale = torch.tensor([1.0], dtype=torch.float32)

    # --- Reference: standalone BSHD caches ---
    key_cache_ref = torch.zeros(
        num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM,
        dtype=torch.uint8, device="cpu")
    value_cache_ref = torch.zeros_like(key_cache_ref)
    ref_q, ref_q_scale, ref_key, ref_value = reference_rope_norm_store_kv_fp8(
        qkv.clone(), cos_sin_cache, positions,
        key_cache_ref, value_cache_ref, slot_mapping,
        num_heads, num_kv_heads, HEAD_DIM, 1.0, 1.0,
        None, None, 0, True,
    )

    # --- Kernel: write into a unified kv_cache, split like vLLM ---
    if kv_layout == "xpu_attn":
        # (2, num_blocks, block_size, num_kv_heads, head_dim)
        kv_cache = torch.zeros(
            2, num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM,
            dtype=torch.uint8, device=DEVICE)
        key_cache_k, value_cache_k = kv_cache.unbind(0)
    else:
        # (num_blocks, 2, block_size, num_kv_heads, head_dim)
        kv_cache = torch.zeros(
            num_blocks, 2, BLOCK_SIZE, num_kv_heads, HEAD_DIM,
            dtype=torch.uint8, device=DEVICE)
        key_cache_k, value_cache_k = kv_cache.unbind(1)

    out_q_k = torch.zeros(num_tokens, num_heads, HEAD_DIM,
                          dtype=torch.uint8, device=DEVICE)

    torch.ops._xpu_C.fused_rope_norm_store_kv_fp8(
        qkv.clone().to(DEVICE),
        cos_sin_cache.to(DEVICE),
        positions.to(DEVICE),
        key_cache_k,
        value_cache_k,
        slot_mapping.to(DEVICE),
        out_q_k,
        num_heads, num_kv_heads, HEAD_DIM,
        k_scale.to(DEVICE), v_scale.to(DEVICE),
        None, None, 0, True,
    )

    # Compare: kernel writes via view must match standalone reference
    key_cpu = key_cache_k.cpu()
    value_cpu = value_cache_k.cpu()

    k_diff = (ref_key.view(torch.uint8).int() - key_cpu.int()).abs().max()
    v_diff = (ref_value.view(torch.uint8).int() - value_cpu.int()).abs().max()
    assert k_diff <= 1, f"K mismatch in {kv_layout}: max ULP = {k_diff}"
    assert v_diff <= 1, f"V mismatch in {kv_layout}: max ULP = {v_diff}"

    # Verify the unified tensor was actually modified (not a detached copy)
    kv_cpu = kv_cache.cpu()
    if kv_layout == "xpu_attn":
        assert kv_cpu[0].view(torch.uint8).sum() > 0, \
            "Unified kv_cache[0] (key) not written"
        assert kv_cpu[1].view(torch.uint8).sum() > 0, \
            "Unified kv_cache[1] (value) not written"
    else:
        assert kv_cpu[:, 0].view(torch.uint8).sum() > 0, \
            "Unified kv_cache[:,0] (key) not written"
        assert kv_cpu[:, 1].view(torch.uint8).sum() > 0, \
            "Unified kv_cache[:,1] (value) not written"

    print(f"  PASSED: vLLM {kv_layout} layout compatibility")
