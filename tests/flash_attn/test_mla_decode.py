# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MLA decode via the XPU paged_decode kernel.

The wrapper concatenates ``q = [q_nope, q_pe]`` and reuses
``kv_c_and_k_pe_cache`` for both K (full, head_dim = kv_lora_rank +
qk_rope_head_dim) and V (a non-contiguous narrow view, head_dim =
kv_lora_rank).
"""


import pytest
import torch

from vllm_xpu_kernels.flash_attn_interface import flash_mla_decode

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
        q = torch.cat([q_nope[q0:q1], q_pe[q0:q1]], dim=-1)  # [ql, h_q, head_qk]
        # attn: [h_q, ql, kl]
        attn = torch.einsum("qhd,kd->hqk", q, k).float() * softmax_scale
        if causal and ql > 1:
            mask = torch.triu(
                torch.ones(ql, kl, device=attn.device, dtype=torch.bool),
                diagonal=kl - ql + 1,
            )
            attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        # out: [ql, h_q, lora]
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


# DeepSeek-V3 shapes: kv_lora_rank=512, qk_rope_head_dim=64.
# At head_size_qk=576 the kernel only fits in Intel Xe SLM with q_packed<=8
# and block_size=64 (see assertions in `flash_mla_decode`). The matrix below
# exercises only the supported envelope; larger configs are gated to assert
# (TORCH_CHECK) and tested separately below.
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize(
    "query_lens,kv_lens",
    [
        ([1, 1, 1, 1], [37, 128, 333, 1024]),    # batch decode
        ([1], [129]),                             # single-seq decode
        ([1, 1], [16, 1023]),                    # short + long
    ],
)
@pytest.mark.parametrize("num_heads_q", [1, 8])
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

    out = flash_mla_decode(
        q_nope=q_nope, q_pe=q_pe,
        kv_c_and_k_pe_cache=cache,
        block_table=bt,
        cu_seqlens_q=cu_q,
        seqused_k=sk,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(kv_lens),
        softmax_scale=softmax_scale,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
    )

    ref = _ref_mla_decode(q_nope, q_pe, cache, bt, cu_q, sk,
                          softmax_scale, causal=False)

    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_mla_decode_rejects_large_q_packed():
    """`flash_mla_decode` should fail fast (not hang) when the configuration
    exceeds Intel Xe SLM limits."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")
    kv_lora_rank, rope = 512, 64
    block_size = 64
    q_nope, q_pe, cache, cu_q, sk, bt = _make_inputs(
        batch=1, query_lens=[1], kv_lens=[64],
        num_heads_q=16, kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=rope, block_size=block_size, num_blocks=4,
        dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="num_heads_q"):
        flash_mla_decode(q_nope=q_nope, q_pe=q_pe,
                         kv_c_and_k_pe_cache=cache, block_table=bt,
                         cu_seqlens_q=cu_q, seqused_k=sk,
                         max_seqlen_q=1, max_seqlen_k=64,
                         kv_lora_rank=kv_lora_rank,
                         qk_rope_head_dim=rope)


def test_mla_decode_rejects_large_block_size():
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")
    kv_lora_rank, rope = 512, 64
    block_size = 128
    q_nope, q_pe, cache, cu_q, sk, bt = _make_inputs(
        batch=1, query_lens=[1], kv_lens=[64],
        num_heads_q=8, kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=rope, block_size=block_size, num_blocks=4,
        dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="block_size"):
        flash_mla_decode(q_nope=q_nope, q_pe=q_pe,
                         kv_c_and_k_pe_cache=cache, block_table=bt,
                         cu_seqlens_q=cu_q, seqused_k=sk,
                         max_seqlen_q=1, max_seqlen_k=64,
                         kv_lora_rank=kv_lora_rank,
                         qk_rope_head_dim=rope)
