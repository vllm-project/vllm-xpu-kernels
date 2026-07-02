# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU correctness test for the fused MiniMax-M3 attention pre-processing kernel
``fused_minimax_m3_qknorm_rope_kv_insert``.

Reference: Gemma RMSNorm ``x * rsqrt(mean(x^2)+eps) * (1+weight)`` followed by
partial NeoX RoPE on the leading ``rotary_dim`` dims. Adapted from the upstream
CUDA test in vllm-project/vllm#45381.
"""

import pytest
import torch

from tests.register_ops import fused_minimax_m3_qknorm_rope_kv_insert

DEVICE = torch.device("xpu")
HEAD_DIM = 128
ROTARY_DIM = 64


def make_cos_sin_cache(max_pos, rotary_dim, base, dtype, device):
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache.to(dtype)


def gemma_rmsnorm(x, weight, eps):
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps)
    out = out * (1.0 + weight.float())
    return out.to(x.dtype)


def apply_rope_neox_partial(x, positions, cos_sin_cache, rotary_dim):
    half = rotary_dim // 2
    cs = cos_sin_cache[positions].float()
    cos = cs[..., :half].unsqueeze(1)
    sin = cs[..., half:].unsqueeze(1)
    rot = x[..., :rotary_dim].float()
    x1 = rot[..., :half]
    x2 = rot[..., half:]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    out = x.clone()
    out[..., :half] = o1.to(x.dtype)
    out[..., half:rotary_dim] = o2.to(x.dtype)
    return out


def norm_rope_ref(x, weight, positions, cos_sin_cache, eps):
    normed = gemma_rmsnorm(x, weight, eps)
    return apply_rope_neox_partial(normed, positions, cos_sin_cache, ROTARY_DIM)


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 513])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 2), (16, 4), (64, 4)])
def test_dense_norm_rope(num_tokens, num_heads, num_kv_heads):
    torch.manual_seed(0)
    dtype, eps = torch.bfloat16, 1e-6
    base, max_pos = 5_000_000.0, 4096

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device=DEVICE) * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device=DEVICE) * 0.1
    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, base, dtype, DEVICE)
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int64, device=DEVICE
    )

    qsz, kvsz = num_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
    qkv = torch.randn(num_tokens, qsz + 2 * kvsz, dtype=dtype, device=DEVICE)
    qkv_orig = qkv.clone()

    fused_minimax_m3_qknorm_rope_kv_insert(
        qkv, q_w, k_w, cos_sin, positions, num_heads, num_kv_heads,
        ROTARY_DIM, eps
    )
    torch.xpu.synchronize()
    q_out, k_out, v_out = qkv.split([qsz, kvsz, kvsz], dim=-1)

    q_in, k_in, v_in = qkv_orig.split([qsz, kvsz, kvsz], dim=-1)
    q_ref = norm_rope_ref(
        q_in.view(num_tokens, num_heads, HEAD_DIM), q_w, positions, cos_sin, eps
    ).view(num_tokens, qsz)
    k_ref = norm_rope_ref(
        k_in.view(num_tokens, num_kv_heads, HEAD_DIM),
        k_w, positions, cos_sin, eps
    ).view(num_tokens, kvsz)

    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_out, k_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_out, v_in, rtol=0, atol=0)  # V untouched


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 513])
@pytest.mark.parametrize("block_size", [16, 64])
def test_sparse_full(num_tokens, block_size):
    torch.manual_seed(1)
    dtype, eps = torch.bfloat16, 1e-6
    base, max_pos = 5_000_000.0, 4096
    num_heads, num_kv_heads, num_idx_heads = 16, 4, 4

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device=DEVICE) * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device=DEVICE) * 0.1
    iq_w = torch.randn(HEAD_DIM, dtype=dtype, device=DEVICE) * 0.1
    ik_w = torch.randn(HEAD_DIM, dtype=dtype, device=DEVICE) * 0.1
    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, base, dtype, DEVICE)
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int64, device=DEVICE
    )

    qsz, kvsz = num_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
    iqsz, iksz = num_idx_heads * HEAD_DIM, HEAD_DIM
    qkv = torch.randn(
        num_tokens, qsz + 2 * kvsz + iqsz + iksz, dtype=dtype, device=DEVICE
    )
    qkv_orig = qkv.clone()
    splits = [qsz, kvsz, kvsz, iqsz, iksz]

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    kv_cache = torch.zeros(
        num_blocks, 2, block_size, num_kv_heads, HEAD_DIM,
        dtype=dtype, device=DEVICE
    )
    index_cache = torch.zeros(
        num_blocks, block_size, HEAD_DIM, dtype=dtype, device=DEVICE
    )
    slot_mapping = torch.randperm(
        num_blocks * block_size, dtype=torch.int64, device=DEVICE
    )[:num_tokens]
    index_slot_mapping = torch.roll(slot_mapping, shifts=1)

    q_out = torch.empty(num_tokens, qsz, dtype=dtype, device=DEVICE)
    index_q = torch.empty(num_tokens, iqsz, dtype=dtype, device=DEVICE)

    fused_minimax_m3_qknorm_rope_kv_insert(
        qkv, q_w, k_w, cos_sin, positions, num_heads, num_kv_heads,
        ROTARY_DIM, eps,
        iq_w, ik_w, num_idx_heads, slot_mapping, index_slot_mapping,
        kv_cache, index_cache, block_size, q_out, index_q,
    )
    torch.xpu.synchronize()

    _, k_out, _, _, index_k = qkv.split(splits, dim=-1)
    q_in, k_in, v_in, iq_orig, ik_orig = qkv_orig.split(splits, dim=-1)
    q_ref = norm_rope_ref(
        q_in.view(num_tokens, num_heads, HEAD_DIM), q_w, positions, cos_sin, eps
    ).view(num_tokens, qsz)
    k_ref = norm_rope_ref(
        k_in.view(num_tokens, num_kv_heads, HEAD_DIM),
        k_w, positions, cos_sin, eps
    ).view(num_tokens, kvsz)
    iq_ref = norm_rope_ref(
        iq_orig.view(num_tokens, num_idx_heads, HEAD_DIM),
        iq_w, positions, cos_sin, eps
    ).view(num_tokens, iqsz)
    ik_ref = norm_rope_ref(
        ik_orig.view(num_tokens, 1, HEAD_DIM), ik_w, positions, cos_sin, eps
    ).view(num_tokens, HEAD_DIM)

    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_out, k_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(index_q, iq_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(index_k, ik_ref, rtol=1e-2, atol=1e-2)

    idx_flat = index_cache.view(num_blocks * block_size, HEAD_DIM)
    k_ref_h = k_ref.view(num_tokens, num_kv_heads, HEAD_DIM)
    v_ref_h = v_in.view(num_tokens, num_kv_heads, HEAD_DIM)
    for t in range(num_tokens):
        s = slot_mapping[t].item()
        b, pos = s // block_size, s % block_size
        torch.testing.assert_close(
            kv_cache[b, 0, pos], k_ref_h[t], rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            kv_cache[b, 1, pos], v_ref_h[t], rtol=0, atol=0
        )
        index_s = index_slot_mapping[t].item()
        torch.testing.assert_close(
            idx_flat[index_s], ik_ref[t], rtol=1e-2, atol=1e-2
        )
