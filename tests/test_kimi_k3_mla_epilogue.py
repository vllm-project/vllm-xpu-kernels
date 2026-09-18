# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the Kimi-K3 MLA epilogue SYCL kernels.

Tests the fused prefill (q-RoPE + K concat + latent cache insert) and decode
(Q concat + latent cache insert) kernels against a pure-PyTorch reference.
bf16/fp16 only -- fp8 / fp8_ds_mla cache-layout variants are not implemented.

Run:
    pytest tests/test_kimi_k3_mla_epilogue.py -v
"""
import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401

DEVICE = torch.device("xpu")

QK_NOPE_DIM = 128
QK_ROPE_DIM = 64
QK_HEAD_DIM = QK_NOPE_DIM + QK_ROPE_DIM  # 192
KV_LORA_RANK = 512
CACHE_ENTRY = KV_LORA_RANK + QK_ROPE_DIM  # 576
HALF_ROPE = QK_ROPE_DIM // 2


def _rope_pair(x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    """GPT-J-style interleaved-pair RoPE. x: [..., 64], cos_sin: [64]."""
    cos = cos_sin[:HALF_ROPE].float()
    sin = cos_sin[HALF_ROPE:].float()
    xf = x.float()
    x_even, x_odd = xf[..., 0::2], xf[..., 1::2]
    new_even = x_even * cos - x_odd * sin
    new_odd = x_even * sin + x_odd * cos
    return torch.stack([new_even, new_odd], dim=-1).flatten(-2).to(x.dtype)


def reference_prefill(q, k_nope, k_pe, kv_c, k_out, k_cache, slot_mapping,
                      position_ids, cos_sin_cache, block_size):
    num_tokens, num_heads = k_nope.shape[0], k_nope.shape[1]
    for t in range(num_tokens):
        rope = None
        if position_ids is not None:
            rope = cos_sin_cache[position_ids[t].item()]
        for h in range(num_heads):
            k_pe_h = k_pe[t]
            if rope is not None:
                q[t, h, QK_NOPE_DIM:] = _rope_pair(q[t, h, QK_NOPE_DIM:], rope)
                k_pe_h = _rope_pair(k_pe_h, rope)
            k_out[t, h] = torch.cat([k_nope[t, h], k_pe_h])
        slot_id = slot_mapping[t].item()
        if slot_id < 0:
            continue
        k_pe_latent = k_pe[t]
        if rope is not None:
            k_pe_latent = _rope_pair(k_pe_latent, rope)
        blk, off = slot_id // block_size, slot_id % block_size
        k_cache[blk, off] = torch.cat([kv_c[t], k_pe_latent])


def reference_decode(ql_nope, q_pe, kv_c, k_pe, mqa_q, k_cache, slot_mapping,
                     position_ids, cos_sin_cache, block_size):
    num_tokens, num_heads = ql_nope.shape[0], ql_nope.shape[1]
    for t in range(num_tokens):
        rope = None
        if position_ids is not None:
            rope = cos_sin_cache[position_ids[t].item()]
        for h in range(num_heads):
            q_pe_h = q_pe[t, h]
            if rope is not None:
                q_pe_h = _rope_pair(q_pe_h, rope)
            mqa_q[t, h] = torch.cat([ql_nope[t, h], q_pe_h])
        slot_id = slot_mapping[t].item()
        if slot_id < 0:
            continue
        k_pe_latent = k_pe[t]
        if rope is not None:
            k_pe_latent = _rope_pair(k_pe_latent, rope)
        blk, off = slot_id // block_size, slot_id % block_size
        k_cache[blk, off] = torch.cat([kv_c[t], k_pe_latent])


def _make_rope(num_tokens, max_pos, apply_rope, device):
    if not apply_rope:
        return None, None
    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64,
                              device=device)
    cos_sin = torch.randn(max_pos, QK_ROPE_DIM, dtype=torch.float32,
                          device=device)
    return positions, cos_sin


@pytest.mark.parametrize("num_tokens", [1, 5, 37])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("apply_rope", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_prefill_correctness(num_tokens, num_heads, apply_rope, dtype):
    torch.manual_seed(0)
    block_size, num_blocks = 16, num_tokens + 1
    max_pos = 4096

    q = torch.randn(num_tokens, num_heads, QK_HEAD_DIM, dtype=dtype,
                    device=DEVICE)
    k_nope = torch.randn(num_tokens, num_heads, QK_NOPE_DIM,
                         dtype=dtype, device=DEVICE)
    k_pe = torch.randn(num_tokens, QK_ROPE_DIM, dtype=dtype,
                       device=DEVICE)
    kv_c = torch.randn(num_tokens, KV_LORA_RANK, dtype=dtype,
                       device=DEVICE)
    slot_mapping = torch.randperm(num_blocks * block_size,
                                  device=DEVICE)[:num_tokens].to(torch.int64)
    positions, cos_sin = _make_rope(num_tokens, max_pos, apply_rope, DEVICE)

    q_ref = q.clone()
    k_out = torch.empty(num_tokens, num_heads, QK_HEAD_DIM,
                        dtype=dtype, device=DEVICE)
    k_out_ref = k_out.clone()
    k_cache = torch.zeros(num_blocks, block_size, CACHE_ENTRY,
                          dtype=dtype, device=DEVICE)
    k_cache_ref = k_cache.clone()

    reference_prefill(q_ref, k_nope, k_pe, kv_c, k_out_ref, k_cache_ref,
                      slot_mapping, positions, cos_sin, block_size)
    torch.ops._C.fused_kimi_k3_mla_key_concat_kv_cache_insert(
        q, k_nope, k_pe, kv_c, k_out, k_cache, slot_mapping, block_size,
        positions, cos_sin)

    torch.testing.assert_close(q.float(), q_ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out.float(), k_out_ref.float(), atol=2e-2,
                               rtol=2e-2)
    torch.testing.assert_close(k_cache.float(), k_cache_ref.float(),
                               atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_tokens", [1, 5, 37])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("apply_rope", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_decode_correctness(num_tokens, num_heads, apply_rope, dtype):
    torch.manual_seed(0)
    block_size, num_blocks = 16, num_tokens + 1
    max_pos = 4096

    ql_nope = torch.randn(num_tokens, num_heads, KV_LORA_RANK,
                          dtype=dtype, device=DEVICE)
    q_pe = torch.randn(num_tokens, num_heads, QK_ROPE_DIM,
                       dtype=dtype, device=DEVICE)
    kv_c = torch.randn(num_tokens, KV_LORA_RANK, dtype=dtype,
                       device=DEVICE)
    k_pe = torch.randn(num_tokens, QK_ROPE_DIM, dtype=dtype,
                       device=DEVICE)
    slot_mapping = torch.randperm(num_blocks * block_size,
                                  device=DEVICE)[:num_tokens].to(torch.int64)
    positions, cos_sin = _make_rope(num_tokens, max_pos, apply_rope, DEVICE)

    mqa_q = torch.empty(num_tokens, num_heads, CACHE_ENTRY,
                        dtype=dtype, device=DEVICE)
    mqa_q_ref = mqa_q.clone()
    k_cache = torch.zeros(num_blocks, block_size, CACHE_ENTRY,
                          dtype=dtype, device=DEVICE)
    k_cache_ref = k_cache.clone()

    reference_decode(ql_nope, q_pe, kv_c, k_pe, mqa_q_ref, k_cache_ref,
                     slot_mapping, positions, cos_sin, block_size)
    torch.ops._C.fused_kimi_k3_mla_decode_q_concat_kv_cache_insert(
        ql_nope, q_pe, kv_c, k_pe, mqa_q, k_cache, slot_mapping, block_size,
        positions, cos_sin)

    torch.testing.assert_close(mqa_q.float(), mqa_q_ref.float(), atol=2e-2,
                               rtol=2e-2)
    torch.testing.assert_close(k_cache.float(), k_cache_ref.float(),
                               atol=2e-2, rtol=2e-2)


def test_negative_slot_skipped():
    """Tokens with slot_mapping=-1 should not write to the cache."""
    torch.manual_seed(0)
    num_tokens, num_heads, block_size, num_blocks = 8, 4, 16, 2

    ql_nope = torch.randn(num_tokens, num_heads, KV_LORA_RANK,
                          dtype=torch.bfloat16, device=DEVICE)
    q_pe = torch.randn(num_tokens, num_heads, QK_ROPE_DIM,
                       dtype=torch.bfloat16, device=DEVICE)
    kv_c = torch.randn(num_tokens, KV_LORA_RANK, dtype=torch.bfloat16,
                       device=DEVICE)
    k_pe = torch.randn(num_tokens, QK_ROPE_DIM, dtype=torch.bfloat16,
                       device=DEVICE)
    slot_mapping = torch.full((num_tokens,), -1, dtype=torch.int64,
                              device=DEVICE)
    mqa_q = torch.empty(num_tokens, num_heads, CACHE_ENTRY,
                        dtype=torch.bfloat16, device=DEVICE)
    k_cache = torch.zeros(num_blocks, block_size, CACHE_ENTRY,
                          dtype=torch.bfloat16, device=DEVICE)

    torch.ops._C.fused_kimi_k3_mla_decode_q_concat_kv_cache_insert(
        ql_nope, q_pe, kv_c, k_pe, mqa_q, k_cache, slot_mapping, block_size,
        None, None)

    assert k_cache.abs().max().item() == 0.0


def test_nonsequential_slots():
    """Non-contiguous slot_mapping crossing block boundaries."""
    torch.manual_seed(7)
    num_tokens, num_heads, block_size, num_blocks = 6, 4, 16, 3
    slots = torch.tensor([0, 5, 15, 16, 17, -1], dtype=torch.int64,
                         device=DEVICE)

    ql_nope = torch.randn(num_tokens, num_heads, KV_LORA_RANK,
                          dtype=torch.bfloat16, device=DEVICE)
    q_pe = torch.randn(num_tokens, num_heads, QK_ROPE_DIM,
                       dtype=torch.bfloat16, device=DEVICE)
    kv_c = torch.randn(num_tokens, KV_LORA_RANK, dtype=torch.bfloat16,
                       device=DEVICE)
    k_pe = torch.randn(num_tokens, QK_ROPE_DIM, dtype=torch.bfloat16,
                       device=DEVICE)
    mqa_q = torch.empty(num_tokens, num_heads, CACHE_ENTRY,
                        dtype=torch.bfloat16, device=DEVICE)
    mqa_q_ref = mqa_q.clone()
    k_cache = torch.zeros(num_blocks, block_size, CACHE_ENTRY,
                          dtype=torch.bfloat16, device=DEVICE)
    k_cache_ref = k_cache.clone()

    reference_decode(ql_nope, q_pe, kv_c, k_pe, mqa_q_ref, k_cache_ref,
                     slots, None, None, block_size)
    torch.ops._C.fused_kimi_k3_mla_decode_q_concat_kv_cache_insert(
        ql_nope, q_pe, kv_c, k_pe, mqa_q, k_cache, slots, block_size,
        None, None)

    torch.testing.assert_close(mqa_q.float(), mqa_q_ref.float(), atol=2e-2,
                               rtol=2e-2)
    torch.testing.assert_close(k_cache.float(), k_cache_ref.float(),
                               atol=2e-2, rtol=2e-2)
    # untouched block/offset stays zero
    assert k_cache[2, 1].abs().max().item() == 0.0
