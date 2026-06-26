# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Accuracy tests for fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn kernel.

Tests the fused pipeline: Online Softmax Compression → RMSNorm → RoPE → MXFP4
quantization → KV cache insert, for DeepSeek-V4 indexer path.
"""

import pytest
import torch

try:
    import tests.register_ops as ops  # noqa: F401
except (ImportError, ModuleNotFoundError):
    import vllm_xpu_kernels._xpu_C  # noqa: F401

DEVICE = "xpu"
HEAD_DIM = 128
ROPE_DIM = 64
BLOCK_SIZE = 16
RMS_EPS = 1e-6
TOKEN_STRIDE = HEAD_DIM // 2  # 64 packed bytes per token
SCALE_DIM = HEAD_DIM // 32  # 4 ue8m0 bytes per token
QUANT_BLOCK = 32


def quantize_to_mxfp4(x: torch.Tensor):
    """Reference MXFP4 quantization (per-block ue8m0 scale, E2M1 values)."""
    MXFP4_BLOCK_SIZE = 32
    orig_shape = x.shape
    head_dim = orig_shape[-1]
    n_blocks = head_dim // MXFP4_BLOCK_SIZE

    x_f32 = x.float().reshape(-1, n_blocks, MXFP4_BLOCK_SIZE)

    amax = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=6 * (2**-126))
    log2_ratio = (amax * (1.0 / 6.0)).log2().ceil().clamp(-127.0, 127.0)
    scale = log2_ratio.exp2()
    ue8m0 = (log2_ratio + 127.0).to(torch.uint8)

    x_scaled = (x_f32 / scale).clamp(-6.0, 6.0)
    abs_x = x_scaled.abs()
    code = torch.zeros_like(abs_x, dtype=torch.int32)
    code = torch.where(abs_x > 0.25, 1, code)
    code = torch.where(abs_x >= 0.75, 2, code)
    code = torch.where(abs_x > 1.25, 3, code)
    code = torch.where(abs_x >= 1.75, 4, code)
    code = torch.where(abs_x > 2.5, 5, code)
    code = torch.where(abs_x >= 3.5, 6, code)
    code = torch.where(abs_x > 5.0, 7, code)
    sign = ((x_scaled.view(torch.int32) >> 31) & 1).to(torch.uint8)
    nibble = code.to(torch.uint8) | (sign << 3)

    nibble_flat = nibble.reshape(-1, head_dim)
    packed = (nibble_flat[:, 0::2] | (nibble_flat[:, 1::2] << 4)).contiguous()
    packed = packed.reshape(*orig_shape[:-1], head_dim // 2)
    scales = ue8m0.view(*orig_shape[:-1], n_blocks)
    return packed, scales


def reference_compress_norm_rope(
    state_cache, block_table, positions, rms_weight, cos_sin_cache,
    compress_ratio, overlap, state_width, rms_eps=1e-6,
):
    """PyTorch reference: compress → RMSNorm → RoPE → MXFP4 quantize."""
    device = state_cache.device
    head_dim = rms_weight.shape[0]
    rope_dim = cos_sin_cache.shape[-1]
    state_block_size = state_cache.shape[1]
    nope_dim = head_dim - rope_dim
    total = (1 + overlap) * compress_ratio
    results = []

    for pos in positions.tolist():
        src = torch.arange(pos - total + 1, pos + 1, dtype=torch.int64,
                           device=device)
        valid = src >= 0
        idx = src.clamp(min=0)
        pages = block_table[0, idx // state_block_size]
        offsets = idx % state_block_size
        raw = state_cache[pages, offsets].float()

        if overlap:
            sw = state_width
            g0_kv = raw[:compress_ratio, :head_dim]
            g1_kv = raw[compress_ratio:, head_dim:2 * head_dim]
            g0_scores = raw[:compress_ratio, sw:sw + head_dim]
            g1_scores = raw[compress_ratio:, sw + head_dim:sw + 2 * head_dim]
            kv = torch.cat([g0_kv, g1_kv])
            scores = torch.cat([g0_scores, g1_scores])
        else:
            kv = raw[:, :head_dim]
            scores = raw[:, state_width:state_width + head_dim]

        scores[~valid] = float("-inf")
        kv[~valid] = 0.0
        weights = torch.softmax(scores, dim=0)
        compressed = (kv * weights).sum(dim=0)

        var = (compressed * compressed).mean()
        normed = compressed * torch.rsqrt(var + rms_eps) * rms_weight.float()

        compressed_pos = (pos // compress_ratio) * compress_ratio
        cos, sin = cos_sin_cache[compressed_pos].float().chunk(2)
        nope, rope = normed.split([nope_dim, rope_dim])
        rope = torch.stack(
            [rope[0::2] * cos - rope[1::2] * sin,
             rope[1::2] * cos + rope[0::2] * sin],
            dim=-1,
        ).reshape(rope_dim)
        results.append(torch.cat([nope, rope]).to(state_cache.dtype))

    result = torch.stack(results)
    return quantize_to_mxfp4(result)


@pytest.mark.parametrize(
    "num_tokens,kv_block_size,compress_ratio",
    [
        (256, 16, 4),
        (256, 32, 4),
        (256, 32, 8),
        (256, 32, 16),
        (256, 32, 32),
        (256, 32, 64),
        (256, 32, 128),
    ],
    ids=[
        "CR4_KVB16", "CR4_KVB32", "CR8", "CR16", "CR32", "CR64", "CR128",
    ],
)
def test_fused_kv_compress_norm_rope_insert_indexer_mxfp4(
    num_tokens, kv_block_size, compress_ratio
):
    """Test fused compress+norm+rope+mxfp4_quant+insert kernel accuracy."""
    torch.manual_seed(42)
    device = DEVICE

    overlap = 1 if compress_ratio == 4 else 0
    coff = 1 + overlap
    state_width = coff * HEAD_DIM

    num_pages = (compress_ratio * num_tokens - 1) // BLOCK_SIZE + 2
    state_cache = torch.randn(
        num_pages, BLOCK_SIZE, 2 * state_width,
        dtype=torch.bfloat16, device=device,
    )
    block_table = torch.arange(
        num_pages, dtype=torch.int32, device=device
    ).unsqueeze(0)
    token_to_req = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    positions = torch.arange(
        compress_ratio - 1, compress_ratio * num_tokens, compress_ratio,
        dtype=torch.int64, device=device,
    )
    rms_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)
    cos_sin_cache = torch.randn(
        compress_ratio * num_tokens, ROPE_DIM, device=device
    )

    kv_n_blocks = (num_tokens + kv_block_size - 1) // kv_block_size + 1
    kv_cache = torch.zeros(
        kv_n_blocks, kv_block_size * (TOKEN_STRIDE + SCALE_DIM),
        dtype=torch.uint8, device=device,
    )

    # Run kernel
    torch.ops._xpu_C.fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn(
        state_cache, token_to_req, positions, slot_mapping, block_table,
        BLOCK_SIZE, state_width, rms_weight, RMS_EPS, cos_sin_cache,
        kv_cache, slot_mapping, kv_block_size, HEAD_DIM, ROPE_DIM,
        compress_ratio, overlap, QUANT_BLOCK,
    )

    # Reference
    ref_packed, ref_scales = reference_compress_norm_rope(
        state_cache, block_table, positions, rms_weight, cos_sin_cache,
        compress_ratio, overlap, state_width, rms_eps=RMS_EPS,
    )

    # Compare byte-by-byte
    for i in range(num_tokens):
        blk, pos = i // kv_block_size, i % kv_block_size
        val_off = pos * TOKEN_STRIDE
        actual_vals = kv_cache[blk, val_off:val_off + TOKEN_STRIDE]
        assert torch.equal(ref_packed[i], actual_vals), (
            f"Token {i}: packed values mismatch"
        )

        scale_off = kv_block_size * TOKEN_STRIDE + pos * SCALE_DIM
        actual_scales = kv_cache[blk, scale_off:scale_off + SCALE_DIM]
        assert torch.equal(ref_scales[i], actual_scales), (
            f"Token {i}: scales mismatch"
        )
