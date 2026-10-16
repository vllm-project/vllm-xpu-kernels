# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for absorbed-MLA (FA3 ``q_v``) varlen attention.

``scores = q @ k^T + q_v @ v^T``, ``out = softmax(scores) @ v`` over a paged
KV cache, as used by vLLM's FlashAttention MLA sparse backend (page size 1
with per-token top-k slots in ``block_table``).
"""

from typing import Optional

import pytest
import torch

from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

DEVICE = "xpu"
D_QK = 64
D_V = 512
DTYPES = [torch.bfloat16, torch.half]
# (num_q_heads, num_kv_heads)
NUM_HEADS = [(16, 1), (128, 1), (20, 1), (40, 2)]
BLOCK_SIZES = [1, 16, 64]
CAUSAL = [False, True]

MINI_PYTEST_PARAMS = {
    "default": {
        "num_heads": [(16, 1), (20, 1)],
        "dtype": [torch.bfloat16],
    },
    "test_mla_qv_paged": {
        "num_heads": [(20, 1)],
        "block_size": [16],
        "causal": [True],
        "dtype": [torch.bfloat16],
    },
}


def ref_mla_qv(
    q: torch.Tensor,
    q_v: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: list[int],
    seqused_k: list[int],
    block_table: torch.Tensor,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (out [total_q, Hq, d_v], lse [Hq, total_q])."""
    page_size = k.shape[1]
    num_heads = q.shape[1]
    heads_per_kv = num_heads // k.shape[2]
    outs = []
    lses = []
    for b in range(len(seqused_k)):
        q_begin, q_end = cu_seqlens_q[b], cu_seqlens_q[b + 1]
        q_len = q_end - q_begin
        kv_len = seqused_k[b]
        n = torch.arange(kv_len, device=q.device)
        blk = block_table[b, n // page_size].long()
        off = n % page_size
        valid = blk >= 0
        safe_blk = blk.clamp(min=0)
        kr = k[safe_blk, off].float().repeat_interleave(heads_per_kv, dim=1)
        vr = v[safe_blk, off].float().repeat_interleave(heads_per_kv, dim=1)
        qs = q[q_begin:q_end].float()
        qvs = q_v[q_begin:q_end].float()
        s = (torch.einsum("qhd,khd->hqk", qs, kr) +
             torch.einsum("qhd,khd->hqk", qvs, vr)) * scale
        mask = valid.view(1, 1, kv_len).expand(num_heads, q_len, kv_len)
        if causal:
            qi = torch.arange(q_len, device=q.device).view(q_len, 1)
            limit = kv_len - (q_len - 1 - qi)
            mask = mask & (n.view(1, kv_len) < limit).view(1, q_len, kv_len)
        s = s.masked_fill(~mask, float("-inf"))
        lse = torch.logsumexp(s, dim=-1)
        # FA convention: rows without keys report lse = +inf and out = 0.
        lse = torch.where(torch.isneginf(lse),
                          torch.full_like(lse, float("inf")), lse)
        p = torch.softmax(s, dim=-1).nan_to_num(0.0)
        outs.append(torch.einsum("hqk,khd->qhd", p, vr))
        lses.append(lse)
    return torch.cat(outs, dim=0), torch.cat(lses, dim=1)


def _run(
    q: torch.Tensor,
    q_v: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    scale: Optional[float],
    causal: bool,
    out: Optional[torch.Tensor] = None,
):
    q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    return flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        q_v=q_v,
        max_seqlen_q=max(q_lens),
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=block_table.shape[1] * k.shape[1],
        seqused_k=seqused_k,
        block_table=block_table,
        softmax_scale=scale,
        causal=causal,
        return_softmax_lse=True,
        out=out,
        fa_version=3,
    )


def _check(out, lse, ref_out, ref_lse, dtype):
    torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=2e-3, rtol=2e-3)
    assert out.dtype == dtype


def _padded_mla_cache(num_slots: int, page_size: int, num_kv_heads: int,
                      row: int, dtype: torch.dtype):
    """k / v as strided views into one padded MLA cache row ([v | k | pad]),
    mirroring vLLM's MLA cache layout."""
    cache = torch.randn(num_slots, page_size, num_kv_heads, row,
                        dtype=dtype,
                        device=DEVICE)
    v = cache[..., :D_V]
    k = cache[..., D_V:D_V + D_QK]
    return k, v


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("topk", [128, 2048])
@pytest.mark.parametrize("row", [D_V + D_QK, 1088])
@torch.inference_mode()
def test_mla_qv_sparse(num_heads, dtype, topk, row):
    """Sparse MLA decode/prefill: one query token per row of block_table,
    page size 1, compacted top-k slots with -1 padding."""
    torch.manual_seed(0)
    num_q_heads, num_kv_heads = num_heads
    num_tokens = 9
    num_slots = 4096
    scale = (D_QK + D_V)**-0.5

    k, v = _padded_mla_cache(num_slots, 1, num_kv_heads, row, dtype)
    q = torch.randn(num_tokens, num_q_heads, D_QK, dtype=dtype, device=DEVICE)
    q_v = torch.randn(num_tokens,
                      num_q_heads,
                      D_V,
                      dtype=dtype,
                      device=DEVICE)

    block_table = torch.full((num_tokens, topk),
                             -1,
                             dtype=torch.int32,
                             device=DEVICE)
    # Edge cases: no keys, a single key, partial tile, full top-k.
    counts = [0, 1, 15, 17, topk // 2, topk - 3, topk, topk, 33]
    for t, c in enumerate(counts):
        block_table[t, :c] = torch.randperm(num_slots,
                                            device=DEVICE)[:c].int()
    # Holes inside the valid range must be skipped as well.
    block_table[5, 7] = -1
    block_table[6, ::97] = -1
    seqused_k = torch.tensor(counts, dtype=torch.int32, device=DEVICE)
    cu_seqlens_q = torch.arange(num_tokens + 1,
                                dtype=torch.int32,
                                device=DEVICE)

    out, lse = _run(q, q_v, k, v, cu_seqlens_q, seqused_k, block_table,
                    scale, True)
    ref_out, ref_lse = ref_mla_qv(q, q_v, k, v, cu_seqlens_q.tolist(),
                                  counts, block_table, scale, True)
    _check(out, lse, ref_out, ref_lse, dtype)
    assert torch.all(out[0] == 0)
    assert torch.all(torch.isposinf(lse[:, 0]))


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("causal", CAUSAL)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_mla_qv_paged(num_heads, block_size, causal, dtype):
    """Varlen batch with multi-token queries over a regular paged cache."""
    torch.manual_seed(0)
    num_q_heads, num_kv_heads = num_heads
    q_lens = [1, 5, 1, 37, 3]
    kv_lens = [1, 40, 300, 37, 129]
    num_blocks = 512
    max_blocks = max((kv + block_size - 1) // block_size for kv in kv_lens)
    scale = 0.07

    k, v = _padded_mla_cache(num_blocks, block_size, num_kv_heads,
                             D_V + D_QK, dtype)
    total_q = sum(q_lens)
    q = torch.randn(total_q, num_q_heads, D_QK, dtype=dtype, device=DEVICE)
    q_v = torch.randn(total_q, num_q_heads, D_V, dtype=dtype, device=DEVICE)
    block_table = torch.randint(0,
                                num_blocks, (len(q_lens), max_blocks),
                                dtype=torch.int32,
                                device=DEVICE)
    cu = [0]
    for ql in q_lens:
        cu.append(cu[-1] + ql)
    cu_seqlens_q = torch.tensor(cu, dtype=torch.int32, device=DEVICE)
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=DEVICE)

    out = torch.empty(total_q, num_q_heads, D_V, dtype=dtype, device=DEVICE)
    res_out, lse = _run(q, q_v, k, v, cu_seqlens_q, seqused_k, block_table,
                        scale, causal, out=out)
    assert res_out.data_ptr() == out.data_ptr()
    ref_out, ref_lse = ref_mla_qv(q, q_v, k, v, cu, kv_lens, block_table,
                                  scale, causal)
    _check(out, lse, ref_out, ref_lse, dtype)


@torch.inference_mode()
def test_mla_qv_default_scale():
    """Without softmax_scale, the FA3 q_v convention (d_qk + d_v)^-0.5
    applies."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    k, v = _padded_mla_cache(64, 16, 1, D_V + D_QK, dtype)
    q = torch.randn(4, 16, D_QK, dtype=dtype, device=DEVICE)
    q_v = torch.randn(4, 16, D_V, dtype=dtype, device=DEVICE)
    block_table = torch.arange(8, dtype=torch.int32,
                               device=DEVICE).view(2, 4)
    cu = [0, 1, 4]
    kv_lens = [20, 64]
    cu_seqlens_q = torch.tensor(cu, dtype=torch.int32, device=DEVICE)
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=DEVICE)

    out, lse = _run(q, q_v, k, v, cu_seqlens_q, seqused_k, block_table, None,
                    True)
    ref_out, ref_lse = ref_mla_qv(q, q_v, k, v, cu, kv_lens, block_table,
                                  (D_QK + D_V)**-0.5, True)
    _check(out, lse, ref_out, ref_lse, dtype)


@pytest.mark.parametrize("kwargs", [
    dict(window_size=[128, 0]),
    dict(softcap=30.0),
    dict(s_aux=True),
    dict(no_block_table=True),
])
@torch.inference_mode()
def test_mla_qv_unsupported(kwargs):
    dtype = torch.bfloat16
    k, v = _padded_mla_cache(8, 16, 1, D_V + D_QK, dtype)
    q = torch.randn(1, 16, D_QK, dtype=dtype, device=DEVICE)
    q_v = torch.randn(1, 16, D_V, dtype=dtype, device=DEVICE)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    seqused_k = torch.tensor([16], dtype=torch.int32, device=DEVICE)
    block_table = torch.zeros(1, 1, dtype=torch.int32, device=DEVICE)
    kwargs = dict(kwargs)
    if kwargs.pop("s_aux", False):
        kwargs["s_aux"] = torch.zeros(16, dtype=torch.float32, device=DEVICE)
    if kwargs.pop("no_block_table", False):
        block_table = None
        seqused_k = None
        kwargs["cu_seqlens_k"] = torch.tensor([0, 16],
                                              dtype=torch.int32,
                                              device=DEVICE)
        k = k[0]
        v = v[0]
    with pytest.raises(NotImplementedError):
        flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            q_v=q_v,
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=16,
            seqused_k=seqused_k,
            block_table=block_table,
            causal=True,
            fa_version=3,
            **kwargs,
        )
