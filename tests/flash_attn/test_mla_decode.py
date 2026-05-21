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


def _ref_mla_decode(
    q_nope: torch.Tensor,        # [tokens, h_q, lora]
    q_pe: torch.Tensor,          # [tokens, h_q, rope]
    cache: torch.Tensor,         # [num_blocks, block_size, lora+rope]
    block_table: torch.Tensor,   # [b, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [b+1]
    seqused_k: torch.Tensor,     # [b]
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    lora = q_nope.shape[-1]
    rope = q_pe.shape[-1]
    head_qk = lora + rope
    block_size = cache.shape[1]
    bt = block_table.cpu().numpy()
    cu = cu_seqlens_q.cpu().tolist()
    sk = seqused_k.cpu().tolist()
    out_chunks = []
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
        attn = torch.einsum("qhd,kd->hqk", q, k).float() * softmax_scale
        if causal and ql > 1:
            mask = torch.triu(
                torch.ones(ql, kl, device=attn.device, dtype=torch.bool),
                diagonal=kl - ql + 1,
            )
            attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,kd->qhd", attn, v)
        out_chunks.append(out)
    return torch.cat(out_chunks, dim=0)


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
    v_cache = cache.narrow(-1, 0, kv_lora_rank)
    # Sanity: V must remain non-contiguous in seq stride but contiguous in
    # the last dim (kernel honors per-tensor strides).
    assert v_cache.stride(-1) == 1
    assert v_cache.stride(-2) == cache.size(-1)

    q = torch.cat([q_nope, q_pe], dim=-1)
    if not q.is_contiguous():
        q = q.contiguous()

    return flash_attn_varlen_func(
        q,
        k_cache,
        v_cache,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        seqused_k=seqused_k,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=False,
        fa_version=2,
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
    head_size_qk = kv_lora_rank + qk_rope_head_dim
    # SLM limits on Intel Xe restrict head_size_qk=576 to q_packed<=8
    # (block_size only affects per-page tiling, not SLM). Larger configs are
    # rejected by mha_varlen_fwd via TORCH_CHECK; covered by the rejection
    # test below.
    if head_size_qk > 512 and num_heads_q > 8:
        pytest.skip(
            "MLA head_size=576 requires num_heads_q<=8 due to Intel Xe SLM "
            "limits")
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


def _call_with(num_heads_q, block_size):
    kv_lora_rank, rope = 512, 64
    q_nope, q_pe, cache, cu_q, sk, bt = _make_inputs(
        batch=1, query_lens=[1], kv_lens=[64],
        num_heads_q=num_heads_q, kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=rope, block_size=block_size, num_blocks=4,
        dtype=torch.bfloat16)
    return _mla_decode_via_varlen(
        q_nope, q_pe, cache, bt, cu_q, sk,
        max_seqlen_q=1, max_seqlen_k=64,
        softmax_scale=(kv_lora_rank + rope) ** -0.5,
    )


def test_mla_decode_rejects_large_q_packed():
    """SLM-oversize configs must fail fast (not hang) for head_size=576."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")
    with pytest.raises(RuntimeError, match="num_heads_q"):
        _call_with(num_heads_q=16, block_size=64)
