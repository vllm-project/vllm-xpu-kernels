# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for deepseek_qnorm_rope_kv_insert SYCL kernel.

Tests the fused Q-RMSNorm + GPT-J RoPE + KV cache insert kernel against a
pure-PyTorch reference for both bf16 and fp8_ds_mla cache formats.

Run:
    pytest tests/test_deepseek_qnorm_rope_kv_insert.py -v
"""
import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

DEVICE = torch.device("xpu")

# DeepSeek-V4 constants
NUM_HEADS_Q = 64
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
HALF_ROPE = ROPE_DIM // 2
QUANT_BLOCK = 64
NUM_QUANT_BLOCKS = NOPE_DIM // QUANT_BLOCK  # 7
SCALE_BYTES_PER_TOKEN = NUM_QUANT_BLOCKS + 1  # 8
TOKEN_DATA_BYTES = NOPE_DIM + ROPE_DIM * 2  # 576
FP8_MAX = 448.0
EPS = 1e-6


def _ref_rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-head RMSNorm without weight. x: [..., HEAD_DIM]"""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps)).to(x.dtype)


def reference_bf16(q, kv, cache, slot_mapping, position_ids, cos_sin_cache,
                   eps, block_size):
    """Pure PyTorch reference for bf16 cache path."""
    num_tokens = q.shape[0]
    num_heads = q.shape[1]

    # Q: per-head RMSNorm + RoPE (fused in float, then store bf16)
    for t in range(num_tokens):
        pos = position_ids[t].item()
        cos = cos_sin_cache[pos, :HALF_ROPE].float()
        sin = cos_sin_cache[pos, HALF_ROPE:].float()
        for h in range(num_heads):
            q_f = q[t, h].float()
            variance = q_f.pow(2).mean(-1, keepdim=True)
            rrms = torch.rsqrt(variance + eps)
            q_normed = q_f * rrms
            nope = q_normed[:NOPE_DIM]
            rope = q_normed[NOPE_DIM:]
            x_even = rope[0::2]
            x_odd = rope[1::2]
            new_even = x_even * cos - x_odd * sin
            new_odd = x_even * sin + x_odd * cos
            rope_out = torch.stack([new_even, new_odd], dim=-1).flatten()
            q[t, h] = torch.cat([nope, rope_out]).to(torch.bfloat16)

    # KV: RoPE + cache insert
    for t in range(num_tokens):
        slot_id = slot_mapping[t].item()
        if slot_id < 0:
            continue
        pos = position_ids[t].item()
        cos = cos_sin_cache[pos, :HALF_ROPE].float()
        sin = cos_sin_cache[pos, HALF_ROPE:].float()

        blk = slot_id // block_size
        pos_in_blk = slot_id % block_size

        # Copy NoPE
        cache[blk, pos_in_blk, :NOPE_DIM] = kv[t, :NOPE_DIM]

        # RoPE on tail
        kv_f = kv[t].float()
        rope = kv_f[NOPE_DIM:]
        x_even = rope[0::2]
        x_odd = rope[1::2]
        new_even = x_even * cos - x_odd * sin
        new_odd = x_even * sin + x_odd * cos
        rope_out = torch.stack([new_even, new_odd], dim=-1).flatten()
        cache[blk, pos_in_blk, NOPE_DIM:] = rope_out.to(torch.bfloat16)


def reference_fp8(q, kv, cache_bytes, slot_mapping, position_ids,
                  cos_sin_cache, eps, block_size):
    """Pure PyTorch reference for fp8 ds-mla cache path."""
    num_tokens = q.shape[0]
    num_heads = q.shape[1]
    block_stride = (
        block_size * TOKEN_DATA_BYTES +
        block_size * SCALE_BYTES_PER_TOKEN
    )

    # Q: per-head RMSNorm + RoPE (fused in float, then store bf16)
    for t in range(num_tokens):
        pos = position_ids[t].item()
        cos = cos_sin_cache[pos, :HALF_ROPE].float()
        sin = cos_sin_cache[pos, HALF_ROPE:].float()
        for h in range(num_heads):
            q_f = q[t, h].float()
            variance = q_f.pow(2).mean(-1, keepdim=True)
            rrms = torch.rsqrt(variance + eps)
            q_normed = q_f * rrms
            nope = q_normed[:NOPE_DIM]
            rope = q_normed[NOPE_DIM:]
            x_even = rope[0::2]
            x_odd = rope[1::2]
            new_even = x_even * cos - x_odd * sin
            new_odd = x_even * sin + x_odd * cos
            rope_out = torch.stack([new_even, new_odd], dim=-1).flatten()
            q[t, h] = torch.cat([nope, rope_out]).to(torch.bfloat16)

    # KV: RoPE + FP8 quantize + cache insert
    for t in range(num_tokens):
        slot_id = slot_mapping[t].item()
        if slot_id < 0:
            continue
        pos = position_ids[t].item()
        cos = cos_sin_cache[pos, :HALF_ROPE].float()
        sin = cos_sin_cache[pos, HALF_ROPE:].float()

        blk = slot_id // block_size
        pos_in_blk = slot_id % block_size

        block_base = blk * block_stride
        token_data_offset = block_base + pos_in_blk * TOKEN_DATA_BYTES
        token_scale_offset = (block_base
                              + block_size * TOKEN_DATA_BYTES
                              + pos_in_blk * SCALE_BYTES_PER_TOKEN)

        # NoPE: bf16 → float → bf16 round → fp8 quantize
        nope_f = kv[t, :NOPE_DIM].float()
        nope_bf16 = nope_f.to(torch.bfloat16).float()  # round to bf16

        for blk_idx in range(NUM_QUANT_BLOCKS):
            start = blk_idx * QUANT_BLOCK
            block_vals = nope_bf16[start:start + QUANT_BLOCK]
            absmax = block_vals.abs().max().clamp(min=1e-4)
            scale_raw = absmax / FP8_MAX
            # ceil to power of 2
            scale = torch.exp2(torch.ceil(torch.log2(scale_raw)))
            scaled = (block_vals / scale).clamp(-FP8_MAX, FP8_MAX)
            fp8_vals = scaled.to(torch.float8_e4m3fn)

            # Write fp8 bytes
            fp8_bytes = fp8_vals.view(torch.uint8)
            cache_bytes[token_data_offset + start:
                        token_data_offset + start + QUANT_BLOCK] = fp8_bytes

            # Write UE8M0 scale byte (biased exponent of the power-of-2 scale)
            import math
            scale_exp = int(math.log2(scale.item())) + 127
            cache_bytes[token_scale_offset + blk_idx] = scale_exp

        # Pad scale byte
        cache_bytes[token_scale_offset + NUM_QUANT_BLOCKS] = 0

        # RoPE tail: store as bf16
        kv_f = kv[t].float()
        rope = kv_f[NOPE_DIM:]
        x_even = rope[0::2]
        x_odd = rope[1::2]
        new_even = x_even * cos - x_odd * sin
        new_odd = x_even * sin + x_odd * cos
        rope_out = torch.stack([new_even, new_odd], dim=-1).flatten()
        rope_bf16 = rope_out.to(torch.bfloat16)
        rope_bytes = rope_bf16.view(torch.uint8)
        cache_bytes[token_data_offset + NOPE_DIM:
                    token_data_offset + NOPE_DIM + ROPE_DIM * 2] = rope_bytes


@pytest.mark.parametrize("num_tokens", [1, 65, 128])
@pytest.mark.parametrize("seed", [0, 42])
def test_bf16_correctness(num_tokens, seed):
    """Test bf16 cache path correctness."""
    torch.manual_seed(seed)
    block_size = 64
    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    max_pos = 4096

    q = torch.randn(num_tokens, NUM_HEADS_Q, HEAD_DIM,
                    dtype=torch.bfloat16, device=DEVICE)
    kv = torch.randn(num_tokens, HEAD_DIM,
                     dtype=torch.bfloat16, device=DEVICE)
    cache = torch.zeros(num_blocks, block_size, HEAD_DIM,
                        dtype=torch.bfloat16, device=DEVICE)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    position_ids = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    cos_sin_cache = torch.randn(max_pos, ROPE_DIM,
                                dtype=torch.float32, device=DEVICE)

    # Clone for reference
    q_ref = q.clone()
    cache_ref = cache.clone()

    # Reference
    reference_bf16(q_ref, kv, cache_ref, slot_mapping, position_ids,
                   cos_sin_cache, EPS, block_size)

    # SYCL kernel
    torch.ops._xpu_C.deepseek_qnorm_rope_kv_insert(
        q, kv, cache, slot_mapping, position_ids, cos_sin_cache,
        EPS, block_size, "bf16")

    # Check Q
    torch.testing.assert_close(
        q.float(), q_ref.float(), atol=1e-2, rtol=1.6e-2,
        msg=lambda m: f"Q mismatch: {m}")

    # Check cache: NoPE must be bit-exact copy, RoPE within 1 ULP
    assert (cache[:, :, :NOPE_DIM] == cache_ref[:, :, :NOPE_DIM]).all(), \
        "bf16 NoPE should be exact copy"
    cache_diff = (cache.float() - cache_ref.float()).abs()
    assert cache_diff.max().item() < 0.001, (
        f"Cache max diff {cache_diff.max().item():.6f}")


@pytest.mark.parametrize("num_tokens", [1, 65, 128])
@pytest.mark.parametrize("seed", [0, 42])
def test_fp8_correctness(num_tokens, seed):
    """Test fp8 ds-mla cache path correctness."""
    torch.manual_seed(seed)
    block_size = 64
    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    max_pos = 4096
    block_stride = (
        block_size * TOKEN_DATA_BYTES +
        block_size * SCALE_BYTES_PER_TOKEN
    )

    q = torch.randn(num_tokens, NUM_HEADS_Q, HEAD_DIM,
                    dtype=torch.bfloat16, device=DEVICE)
    kv = torch.randn(num_tokens, HEAD_DIM,
                     dtype=torch.bfloat16, device=DEVICE)
    # FP8 cache is flat uint8
    cache = torch.zeros(num_blocks * block_stride,
                        dtype=torch.uint8, device=DEVICE)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    position_ids = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    cos_sin_cache = torch.randn(max_pos, ROPE_DIM,
                                dtype=torch.float32, device=DEVICE)

    # Clone for reference
    q_ref = q.clone()
    cache_ref = cache.clone().cpu()
    kv_cpu = kv.cpu()
    slot_cpu = slot_mapping.cpu()
    pos_cpu = position_ids.cpu()
    cs_cpu = cos_sin_cache.cpu()

    # Reference on CPU
    q_ref_cpu = q_ref.cpu()
    reference_fp8(q_ref_cpu, kv_cpu, cache_ref, slot_cpu, pos_cpu,
                  cs_cpu, EPS, block_size)

    # SYCL kernel — cache passed as 2D [num_blocks, block_stride]
    cache_2d = cache.view(num_blocks, block_stride)
    torch.ops._xpu_C.deepseek_qnorm_rope_kv_insert(
        q, kv, cache_2d, slot_mapping, position_ids, cos_sin_cache,
        EPS, block_size, "fp8_ds_mla")

    # Check Q
    torch.testing.assert_close(
        q.float().cpu(), q_ref_cpu.float(), atol=1e-2, rtol=1.6e-2,
        msg=lambda m: f"Q mismatch: {m}")

    # Check cache bytes: scales, NoPE fp8 data, RoPE bf16
    cache_flat = cache.cpu()
    for t in range(min(num_tokens, 8)):
        slot_id = t
        blk = slot_id // block_size
        pos_in_blk = slot_id % block_size
        block_base = blk * block_stride
        data_off = block_base + pos_in_blk * TOKEN_DATA_BYTES
        scale_off = (block_base + block_size * TOKEN_DATA_BYTES
                     + pos_in_blk * SCALE_BYTES_PER_TOKEN)

        # Scales: ±1 exponent allowed (rounding mode difference)
        for s in range(NUM_QUANT_BLOCKS):
            dev_scale = cache_flat[scale_off + s].item()
            ref_scale = cache_ref[scale_off + s].item()
            assert abs(dev_scale - ref_scale) <= 1, (
                f"Token {t} scale[{s}]: dev={dev_scale} ref={ref_scale}")

        # NoPE fp8 data: allow ±3 byte diff (exponent boundary rounding)
        dev_nope = cache_flat[data_off:data_off + NOPE_DIM].to(torch.int16)
        ref_nope = cache_ref[data_off:data_off + NOPE_DIM].to(torch.int16)
        byte_diff = (dev_nope - ref_nope).abs()
        assert byte_diff.max().item() <= 3, (
            f"Token {t} NoPE fp8 byte diff: {byte_diff.max().item()}")

        # RoPE bf16 region
        rope_off = data_off + NOPE_DIM
        dev_rope = cache_flat[rope_off:rope_off + ROPE_DIM * 2]
        ref_rope = cache_ref[rope_off:rope_off + ROPE_DIM * 2]
        dev_bf16 = dev_rope.view(torch.bfloat16).float()
        ref_bf16 = ref_rope.view(torch.bfloat16).float()
        rope_diff = (dev_bf16 - ref_bf16).abs().max().item()
        assert rope_diff < 0.001, (
            f"Token {t} RoPE bf16 diff {rope_diff:.6f}")


def test_negative_slot_skipped():
    """Tokens with slot_mapping=-1 should not write to cache."""
    torch.manual_seed(0)
    num_tokens = 16
    block_size = 64
    num_blocks = 2
    max_pos = 4096

    q = torch.randn(num_tokens, NUM_HEADS_Q, HEAD_DIM,
                    dtype=torch.bfloat16, device=DEVICE)
    kv = torch.randn(num_tokens, HEAD_DIM,
                     dtype=torch.bfloat16, device=DEVICE)
    cache = torch.zeros(num_blocks, block_size, HEAD_DIM,
                        dtype=torch.bfloat16, device=DEVICE)
    slot_mapping = torch.full((num_tokens,), -1,
                              dtype=torch.int64, device=DEVICE)
    position_ids = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    cos_sin_cache = torch.randn(max_pos, ROPE_DIM,
                                dtype=torch.float32, device=DEVICE)

    torch.ops._xpu_C.deepseek_qnorm_rope_kv_insert(
        q, kv, cache, slot_mapping, position_ids, cos_sin_cache,
        EPS, block_size, "bf16")

    assert cache.abs().max().item() == 0.0


def test_nonsequential_slots():
    """Non-contiguous slot_mapping crossing block boundaries."""
    torch.manual_seed(7)
    num_tokens = 10
    block_size = 64
    max_pos = 4096
    slots = torch.tensor([0, 5, 63, 64, 65, -1, 127, 3, -1, 50],
                         dtype=torch.int64, device=DEVICE)
    num_blocks = 3

    q = torch.randn(num_tokens, NUM_HEADS_Q, HEAD_DIM,
                    dtype=torch.bfloat16, device=DEVICE)
    kv = torch.randn(num_tokens, HEAD_DIM,
                     dtype=torch.bfloat16, device=DEVICE)
    cache = torch.zeros(num_blocks, block_size, HEAD_DIM,
                        dtype=torch.bfloat16, device=DEVICE)
    position_ids = torch.randint(0, max_pos, (num_tokens,),
                                 dtype=torch.int64, device=DEVICE)
    cos_sin_cache = torch.randn(max_pos, ROPE_DIM,
                                dtype=torch.float32, device=DEVICE)

    q_ref = q.clone()
    cache_ref = cache.clone()

    reference_bf16(q_ref, kv, cache_ref, slots, position_ids,
                   cos_sin_cache, EPS, block_size)

    torch.ops._xpu_C.deepseek_qnorm_rope_kv_insert(
        q, kv, cache, slots, position_ids, cos_sin_cache,
        EPS, block_size, "bf16")

    q_diff = (q.float() - q_ref.float()).abs()
    assert q_diff.max().item() < 0.001, f"Q diff {q_diff.max().item():.6f}"

    assert (cache[:, :, :NOPE_DIM] == cache_ref[:, :, :NOPE_DIM]).all(), \
        "NoPE should be exact copy for scattered slots"
    cache_diff = (cache.float() - cache_ref.float()).abs()
    assert cache_diff.max().item() < 0.001, \
        f"Cache diff {cache_diff.max().item():.6f}"

    # Untouched positions remain zero
    untouched = [1, 2, 4, 6, 7]
    for p in untouched:
        assert cache[0, p].abs().max().item() == 0.0
