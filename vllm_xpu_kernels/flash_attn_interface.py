# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch

#isort: off
try:
    from . import _vllm_fa2_C  # noqa: F401
    FA2_UNAVAILABLE_REASON = None
    FA2_AVAILABLE = True
except ImportError as e:
    FA2_UNAVAILABLE_REASON = str(e)
    FA2_AVAILABLE = False

#isort: on

DEFAULT_FA_VERSION = 2


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,  # only used for non-paged prefill
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size: Optional[list[int]] = None,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    # FA3 Only
    scheduler_metadata=None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    # Version selector
    fa_version: int = DEFAULT_FA_VERSION,
    s_aux: Optional[torch.Tensor] = None,
    num_splits_kv: Optional[int] = None,
):
    """
    FlashAttention interface for variable-length sequences, with optional
    paged KV cache support.

    Args:
        q, k, v: Query, key, value tensors.
        max_seqlen_q: Maximum query sequence length in the batch.
        cu_seqlens_q: Cumulative sequence lengths for queries.
        max_seqlen_k: Maximum key/value sequence length in the batch.
        cu_seqlens_k: Cumulative sequence lengths for keys/values when not
            using paged KV cache.
        seqused_k: Number of tokens used per sequence when using paged KV.
        block_table: Optional block table for paged KV cache.
        num_splits: Backend-specific split parameter (non-KV specific),
            typically used to control work partitioning in some FA versions.
        num_splits_kv: Optional number of splits applied to KV **blocks**
            when using paged KV cache. This is forwarded to the underlying
            C++ FlashAttention op as its ``num_splits`` parameter; the split
            unit is KV blocks, not individual tokens or pages.
        fa_version: FlashAttention backend version selector.
    """
    assert cu_seqlens_k is not None or seqused_k is not None, \
        "cu_seqlens_k or seqused_k must be provided"
    assert cu_seqlens_k is None or seqused_k is None, \
        "cu_seqlens_k and seqused_k cannot be provided at the same time"
    assert block_table is None or seqused_k is not None, \
        "when enable block_table, seqused_k is needed"
    assert block_table is not None or cu_seqlens_k is not None, \
        "when block_table is disabled, cu_seqlens_k is needed"

    if softmax_scale is None:
        softmax_scale = q.shape[-1]**(-0.5)
    if k_descale is not None:
        assert sum(k_descale.stride()) == 0 and \
            k_descale.dtype == torch.float32, \
            "k_descale must be view of single float32 scalar tensor"
    if v_descale is not None:
        assert sum(v_descale.stride()) == 0 and \
            v_descale.dtype == torch.float32, \
            "v_descale must be view of single float32 scalar tensor"
    # custom op does not support non-tuple input
    real_window_size: tuple[int, int]
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)

    if fa_version == 2:
        if scheduler_metadata is not None and q_descale is not None \
            and k_descale is not None and v_descale is not None:
            raise NotImplementedError(
                "FA2 does not support scheduler_metadata, q_descale, "
                "k_descale, v_descale")
        if num_splits > 1:
            raise NotImplementedError("FA2 does not support num_splits > 1")
        if q_descale is not None:
            raise NotImplementedError("FA2 does not support q_descale")
        if scheduler_metadata is not None:
            raise NotImplementedError(
                "FA2 does not support scheduler_metadata")
        if (k_descale is not None
                and v_descale is None) or (k_descale is None
                                           and v_descale is not None):
            raise NotImplementedError(
                "FA2 only supports both KV cache descaled")
        out, softmax_lse = torch.ops._vllm_fa2_C.varlen_fwd(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            # cu_seqlens_k not used since we use seqused_k, but flash_api.cpp
            # still wants it so we pass all zeros
            dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,
            seqused_k,
            None,
            block_table,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            k_descale,
            v_descale,
            softmax_scale,
            s_aux,
            False,
            causal,
            real_window_size[0],
            real_window_size[1],
            softcap,
            return_softmax_lse,
            None,
            num_splits_kv,
        )
    else:
        raise NotImplementedError("not support yet")
    return (out, softmax_lse) if return_softmax_lse else (out)


def flash_mla_decode(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_c_and_k_pe_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    *,
    kv_lora_rank: Optional[int] = None,
    qk_rope_head_dim: Optional[int] = None,
    num_splits: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MLA decode using the XPU paged decode kernel.

    Inputs follow the FA3 MLA convention used in
    `vllm/v1/attention/backends/mla/flashattn_mla.py`:

    * ``q_nope``: ``[num_tokens, num_heads_q, kv_lora_rank]``
    * ``q_pe``:   ``[num_tokens, num_heads_q, qk_rope_head_dim]``
    * ``kv_c_and_k_pe_cache``: ``[num_blocks, block_size, kv_lora_rank +
      qk_rope_head_dim]`` or with an explicit num_heads_kv=1 dim.

    Internally this concatenates ``q = [q_nope, q_pe]`` so the kernel sees a
    single attention with ``head_size_qk = kv_lora_rank + qk_rope_head_dim``
    and ``head_size_vo = kv_lora_rank``. The kv_c portion of
    ``kv_c_and_k_pe_cache`` is passed as V via a non-contiguous narrow view of
    the same buffer (kernel honors per-tensor strides).
    """
    assert q_nope.dim() == 3 and q_pe.dim() == 3, \
        "q_nope and q_pe must be [num_tokens, num_heads_q, head_dim]"
    assert q_nope.shape[0] == q_pe.shape[0] and \
        q_nope.shape[1] == q_pe.shape[1], \
        "q_nope and q_pe must agree on tokens and num_heads_q"

    if kv_lora_rank is None:
        kv_lora_rank = q_nope.shape[-1]
    if qk_rope_head_dim is None:
        qk_rope_head_dim = q_pe.shape[-1]
    assert q_nope.shape[-1] == kv_lora_rank
    assert q_pe.shape[-1] == qk_rope_head_dim

    head_size_qk = kv_lora_rank + qk_rope_head_dim

    # The XPU paged_decode kernel only supports q_len == 1 per request: with
    # q_len > 1 + causal masking the kernel's effective-seqlen computation
    # reads past the valid KV (see comment in csrc/flash_attn/flash_api.cpp).
    # Spec-decode style multi-token-per-request is not yet supported here.
    assert max_seqlen_q == 1, (
        "flash_mla_decode currently only supports max_seqlen_q == 1; "
        f"got max_seqlen_q={max_seqlen_q}"
    )
    causal = False  # no-op when q_len == 1; seqused_k already constrains KV.

    # Normalize cache to 4D [num_blocks, block_size, num_heads_kv=1, head_dim].
    cache = kv_c_and_k_pe_cache
    if cache.dim() == 3:
        cache = cache.unsqueeze(-2)
    assert cache.dim() == 4 and cache.size(-2) == 1, \
        "kv_c_and_k_pe_cache must be [num_blocks, block_size, (1,)? , " \
        "kv_lora_rank+qk_rope_head_dim]"
    assert cache.shape[-1] == head_size_qk, \
        f"cache last dim {cache.shape[-1]} != kv_lora_rank + " \
        f"qk_rope_head_dim ({head_size_qk})"

    # SLM (shared local memory) limits on Intel Xe restrict the kernel to
    # smaller per-WG tiles when head_size grows. At head_size_qk = 576 the
    # output accumulator alone (q_packed * head_size_vo * 4 bytes) plus the
    # Q/K/V tiles cannot fit in SLM for q_packed >= 16 or for block_size >
    # 64. The same restrictions are documented in this repo's existing
    # decode tests (see `test_decode_with_paged_kv` SLM skips at h512+p128
    # and (16,1)+h256). Rather than hang at runtime we fail fast with a
    # clear message so callers (e.g. vLLM) can route to a different
    # strategy or bump TP.
    block_size = cache.size(1)
    num_q_heads = q_nope.shape[1]
    assert block_size <= 64, (
        f"flash_mla_decode: block_size={block_size} is not supported at "
        f"head_size={head_size_qk} due to Intel Xe SLM limits; use "
        f"block_size=64.")
    assert num_q_heads <= 8, (
        f"flash_mla_decode: num_heads_q={num_q_heads} per rank is not "
        f"supported at head_size={head_size_qk} due to Intel Xe SLM "
        f"limits (q_packed must be <= 8). Increase tensor parallel size "
        f"so num_heads_q per rank is <= 8.")

    # K = full cache (head_size = kv_lora_rank + qk_rope_head_dim).
    k_cache = cache
    # V = kv_c view (first kv_lora_rank elements of last dim). This is
    # non-contiguous in the seq stride (== head_size_qk), but the kernel
    # honors actual KV strides, and the last dim is still contiguous.
    v_cache = cache.narrow(-1, 0, kv_lora_rank)

    # Concatenate Q in the order matching the cache layout: [nope, pe].
    q = torch.cat([q_nope, q_pe], dim=-1)
    if not q.is_contiguous():
        q = q.contiguous()

    if softmax_scale is None:
        softmax_scale = head_size_qk**-0.5

    return torch.ops._vllm_fa2_C.mla_decode_fwd(
        q,
        k_cache,
        v_cache,
        out,
        cu_seqlens_q,
        seqused_k,
        block_table,
        max_seqlen_q,
        max_seqlen_k,
        float(softmax_scale),
        bool(causal),
        num_splits,
    )
