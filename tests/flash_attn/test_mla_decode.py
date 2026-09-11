# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MLA decode via the XPU paged_decode kernel.

The unified ``flash_attn_varlen_func`` (FA2) decode branch routes paged decode
with ``max_seqlen_q == 1`` through ``cutlass_paged_decode_interface``. MLA
decode is expressed as a varlen call by:

* concatenating ``q = [q_nope, q_pe]`` (head_size_qk = lora + rope),
* using the full ``kv_c_and_k_pe_cache`` reshaped to 4D
  ``[num_blocks, block_size, 1, head_qk]`` as ``k_cache``, and
* using a non-contiguous narrow view of the same buffer (first ``lora``
  channels of the last dim) as ``v_cache``.
"""


import pytest
import torch

from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

DEVICE = "xpu"


@pytest.fixture(autouse=True)
def require_compiled_decode(monkeypatch):
    def reject_fallback(*args, **kwargs):
        pytest.fail("MLA kernel test reached the Python reference fallback")

    monkeypatch.setattr(
        "vllm_xpu_kernels.flash_attn_interface._fallback_varlen_attn",
        reject_fallback)


def _ref_mla_decode(
    q_nope: torch.Tensor,        # [tokens, h_q, lora]
    q_pe: torch.Tensor,          # [tokens, h_q, rope]
    cache: torch.Tensor,         # [num_blocks, block_size, lora+rope]
    block_table: torch.Tensor,   # [b, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [b+1]
    seqused_k: torch.Tensor,     # [b]
    softmax_scale: float,
    causal: bool,
    return_softmax_lse: bool = False,
):
    lora = q_nope.shape[-1]
    rope = q_pe.shape[-1]
    head_qk = lora + rope
    block_size = cache.shape[1]
    bt = block_table.cpu().numpy()
    cu = cu_seqlens_q.cpu().tolist()
    sk = seqused_k.cpu().tolist()
    out_chunks = []
    lse_chunks = []
    for i in range(len(sk)):
        q0, q1 = cu[i], cu[i + 1]
        ql = q1 - q0
        kl = sk[i]
        nb = (kl + block_size - 1) // block_size
        idx = bt[i, :nb]
        kv = cache[idx].reshape(-1, head_qk)[:kl]   # [kl, head_qk]
        k = kv                                      # [kl, head_qk]
        v = kv[:, :lora]                            # [kl, lora]
        q = torch.cat([q_nope[q0:q1],
                       q_pe[q0:q1]], dim=-1)  # [ql, h_q, head_qk]
        attn = torch.einsum("qhd,kd->hqk", q.float(), k.float()) * softmax_scale
        if causal and ql > 1:
            mask = torch.triu(
                torch.ones(ql, kl, device=attn.device, dtype=torch.bool),
                diagonal=kl - ql + 1,
            )
            attn.masked_fill_(mask, float("-inf"))
        if return_softmax_lse:
            lse_chunks.append(torch.logsumexp(attn, dim=-1))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,kd->qhd", attn, v)
        out_chunks.append(out)
    output = torch.cat(out_chunks, dim=0)
    if return_softmax_lse:
        return output, torch.cat(lse_chunks, dim=-1)
    return output


def _make_inputs(
    batch: int,
    query_lens: list[int],
    kv_lens: list[int],
    num_heads_q: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int = 0,
):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    head_qk = kv_lora_rank + qk_rope_head_dim
    total_q = sum(query_lens)
    q_nope = torch.randn(total_q, num_heads_q, kv_lora_rank,
                         dtype=dtype, device=DEVICE, generator=g)
    q_pe = torch.randn(total_q, num_heads_q, qk_rope_head_dim,
                       dtype=dtype, device=DEVICE, generator=g)
    cache = torch.randn(num_blocks, block_size, head_qk,
                        dtype=dtype, device=DEVICE, generator=g)
    cu_seqlens_q = torch.tensor([0] + query_lens, dtype=torch.int32,
                                device=DEVICE).cumsum(0, dtype=torch.int32)
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=DEVICE)
    max_blocks = (max(kv_lens) + block_size - 1) // block_size
    block_table = torch.randint(0, num_blocks, (batch, max_blocks),
                                dtype=torch.int32, device=DEVICE,
                                generator=g)
    return q_nope, q_pe, cache, cu_seqlens_q, seqused_k, block_table


def _mla_decode_via_varlen(
    q_nope, q_pe, cache, block_table, cu_seqlens_q, seqused_k,
    max_seqlen_q, max_seqlen_k, softmax_scale,
    *, num_splits_kv=None, return_softmax_lse=False, host_kv_lens=None,
    v_head_size=None,
):
    """Pack MLA inputs and call ``flash_attn_varlen_func``.

    Mirrors what the vLLM XPU MLA backend does at `forward_mqa` time.
    """
    kv_lora_rank = q_nope.shape[-1]

    # Cache: normalize to 4D [num_blocks, block_size, 1, head_qk].
    if cache.dim() == 3:
        cache = cache.unsqueeze(-2)
    assert cache.dim() == 4 and cache.size(-2) == 1

    k_cache = cache
    v_cache = cache.narrow(
        -1, 0, kv_lora_rank if v_head_size is None else v_head_size)
    # Sanity: V must remain non-contiguous in seq stride but contiguous in
    # the last dim (kernel honors per-tensor strides).
    assert v_cache.stride(-1) == 1
    assert v_cache.stride(-2) == cache.size(-1)

    q = torch.cat([q_nope, q_pe], dim=-1)
    if not q.is_contiguous():
        q = q.contiguous()

    length_args = ({"seqused_k": seqused_k} if host_kv_lens is None else
                   {"host_kv_lens": host_kv_lens})
    return flash_attn_varlen_func(
        q,
        k_cache,
        v_cache,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=False,
        fa_version=2,
        num_splits_kv=num_splits_kv,
        return_softmax_lse=return_softmax_lse,
        **length_args,
    )


# DeepSeek-V3 shapes: kv_lora_rank=512, qk_rope_head_dim=64.
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize(
    "query_lens,kv_lens",
    [
        ([1, 1, 1, 1], [37, 128, 333, 1024]),    # batch decode
        ([1], [129]),                             # single-seq decode
        ([1, 1], [16, 1023]),                    # short + long
    ],
)
@pytest.mark.parametrize("num_heads_q", [1, 8, 16])
def test_mla_decode_deepseek_v3(block_size, query_lens, kv_lens, num_heads_q):
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    dtype = torch.bfloat16

    batch = len(query_lens)
    assert len(kv_lens) == batch
    num_blocks = max(256,
                     (max(kv_lens) + block_size - 1) // block_size * batch * 2)

    q_nope, q_pe, cache, cu_q, sk, bt = _make_inputs(
        batch, query_lens, kv_lens, num_heads_q,
        kv_lora_rank, qk_rope_head_dim, block_size, num_blocks, dtype)

    softmax_scale = (kv_lora_rank + qk_rope_head_dim) ** -0.5

    out = _mla_decode_via_varlen(
        q_nope, q_pe, cache, bt, cu_q, sk,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(kv_lens),
        softmax_scale=softmax_scale,
    )

    ref = _ref_mla_decode(q_nope, q_pe, cache, bt, cu_q, sk,
                          softmax_scale, causal=False)

    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_heads_q", [16, 32])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_splits_kv", [1, 8])
@pytest.mark.parametrize("path", ["seqused", "hostlens"])
def test_mla_decode_large_q_packed(num_heads_q, block_size, batch, dtype,
                                  num_splits_kv, path):
    """Cover Q8/Q16 dispatch, split reduction, and partial KV tiles."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")
    kv_lora_rank, rope = 512, 64
    query_lens = [1] * batch
    kv_lens = {1: [2113], 2: [129, 2113], 4: [37, 65, 129, 2113]}[batch]
    q_nope, q_pe, cache, cu_q, sk, bt = _make_inputs(
        batch=batch, query_lens=query_lens, kv_lens=kv_lens,
        num_heads_q=num_heads_q, kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=rope, block_size=block_size, num_blocks=256,
        dtype=dtype)
    softmax_scale = (kv_lora_rank + rope)**-0.5

    out, lse = _mla_decode_via_varlen(
        q_nope, q_pe, cache, bt, cu_q, sk,
        max_seqlen_q=1, max_seqlen_k=max(kv_lens),
        softmax_scale=softmax_scale,
        num_splits_kv=num_splits_kv,
        return_softmax_lse=True,
        host_kv_lens=kv_lens if path == "hostlens" else None,
    )
    ref, ref_lse = _ref_mla_decode(
        q_nope, q_pe, cache, bt, cu_q, sk, softmax_scale,
        causal=False, return_softmax_lse=True)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("head_qk", [544, 576])
@pytest.mark.parametrize("num_heads_q", [8, 32])
def test_mla_decode_default_v_width(head_qk, num_heads_q):
    """Non-split-V shapes in the 576 bucket retain the default policy."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")
    kv_lens = [37, 65, 129, 2113]
    q_nope, q_pe, cache, cu_q, sk, bt = _make_inputs(
        batch=4, query_lens=[1] * 4, kv_lens=kv_lens,
        num_heads_q=num_heads_q, kv_lora_rank=512,
        qk_rope_head_dim=head_qk - 512, block_size=64, num_blocks=256,
        dtype=torch.bfloat16)
    softmax_scale = head_qk**-0.5
    out = _mla_decode_via_varlen(
        q_nope, q_pe, cache, bt, cu_q, sk,
        max_seqlen_q=1, max_seqlen_k=max(kv_lens),
        softmax_scale=softmax_scale, v_head_size=256, num_splits_kv=1)
    ref = _ref_mla_decode(
        q_nope, q_pe, cache, bt, cu_q, sk, softmax_scale, causal=False)
    torch.testing.assert_close(out, ref[..., :256], atol=2e-2, rtol=2e-2)
