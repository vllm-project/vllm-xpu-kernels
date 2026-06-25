# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch

#isort: off
try:
    from . import _vllm_sparse_mla_C  # noqa: F401
    SPARSE_MLA_UNAVAILABLE_REASON = None
    SPARSE_MLA_AVAILABLE = True
except ImportError as e:
    SPARSE_MLA_UNAVAILABLE_REASON = str(e)
    SPARSE_MLA_AVAILABLE = False


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    return_softmax_lse: bool = False,
):
    """Sparse MLA prefill forward interface."""
    results = torch.ops._vllm_sparse_mla_C.mla_sparse_prefill_fwd(
        maybe_contiguous(q),
        maybe_contiguous(kv),
        maybe_contiguous(indices),
        sm_scale,
        d_v,
        maybe_contiguous(attn_sink),
        maybe_contiguous(topk_length),
        out,
        return_softmax_lse,
    )
    if return_softmax_lse:
        return tuple(results)
    return results[0]


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    num_splits: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    return_softmax_lse: bool = False,
):
    """Sparse MLA decode interface with FP8 KV cache."""
    if indices is None:
        raise NotImplementedError(
            "flash_mla_with_kvcache is only supported for sparse attention "
            "for now"
        )
    assert not causal, "causal must be False when sparse attention is enabled"
    assert is_fp8_kvcache, (
        "is_fp8_kvcache must be True when sparse attention is enabled"
    )
    assert q.dtype == torch.bfloat16, (
        "q must be bfloat16 when sparse attention is enabled"
    )
    assert q.shape[1] == 1, "seq_len_q must be 1 for sparse fp8 decode"
    assert q.shape[-1] == head_dim_v and k_cache.shape[-1] == 584, \
        "Only DeepSeek V4 style packed head_dim is supported for now"
    _ = block_table, cache_seqlens  # Unused for sparse path compatibility.

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out_tensor, lse = torch.ops._vllm_sparse_mla_C.mla_sparse_decode_fp8_fwd(
        maybe_contiguous(q),
        maybe_contiguous(k_cache),
        maybe_contiguous(indices),
        maybe_contiguous(topk_length),
        maybe_contiguous(attn_sink),
        out,
        tile_scheduler_metadata,
        num_splits,
        maybe_contiguous(extra_k_cache),
        maybe_contiguous(extra_indices_in_kvcache),
        maybe_contiguous(extra_topk_length),
        head_dim_v,
        softmax_scale,
        return_softmax_lse,
    )
    if return_softmax_lse:
        return out_tensor, lse
    return out_tensor, None
