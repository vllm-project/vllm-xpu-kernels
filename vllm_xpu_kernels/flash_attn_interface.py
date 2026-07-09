# SPDX-License-Identifier: Apache-2.0
import os
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

# Speculative-decoding fast path.
#
# In speculative decoding every sequence in the batch issues the same small
# number of query tokens (q_len = num_speculative_tokens + 1, e.g. 2/3/4). By
# default such a batch (max_seqlen_q > 1) is routed to the chunk-prefill
# kernel, which has no split-K parallelism over the KV dimension. With a long
# KV cache and only a handful of query rows the GPU is heavily under-utilized,
# so prefill is slow.
#
# Instead we reuse the split-K *decode* kernel: for a uniform, causal, paged
# batch we treat each (sequence, query-position) pair as an independent
# single-token decode "sequence" and run the whole batch in one decode
# launch. Query token ``j`` (0-indexed within the q_len window) attends
# causally to KV positions ``[0, kv_len - q_len + j]``, i.e.
# ``seqused_k - (q_len - 1 - j)`` tokens, so each pseudo-sequence is an
# ordinary q_len==1 decode with a per-position-adjusted ``seqused_k``. This is
# exact (it matches the causal reference) and gets the decode path's split-K
# parallelism.
#
# The path only triggers for small, uniform query lengths; larger uniform
# query lengths keep using chunk-prefill (which reuses the shared KV tile
# across the Q block, whereas the decode path rescans KV per query token).
# The default threshold (16) stays within the measured win region across KV
# lengths on Xe2 (crossover is ~q_len 32); override it with
# VLLM_XPU_SPEC_DECODE_MAX_QLEN (set to 0/1 to disable the fast path).
try:
    _SPEC_DECODE_MAX_QLEN = int(
        os.environ.get("VLLM_XPU_SPEC_DECODE_MAX_QLEN", "16"))
except ValueError:
    _SPEC_DECODE_MAX_QLEN = 16


def _spec_decode_varlen_fwd(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    block_table,
    max_seqlen_k,
    k_descale,
    v_descale,
    softmax_scale,
    s_aux,
    window_size,
    softcap,
):
    """Run the split-K decode kernel once for a speculative-decode batch.

    Assumes a uniform query length: every sequence contributes the same
    ``q_len`` consecutive query tokens, laid out as ``q[i * q_len + j]`` for
    sequence ``i`` and position ``j``. Each (sequence, position) pair is
    treated as an independent single-token decode "sequence", so the whole
    batch runs in a single ``varlen_fwd`` launch. Returns the output tensor
    (same row order as ``q``).
    """
    batch = cu_seqlens_q.numel() - 1
    q_len = q.shape[0] // batch
    total = q.shape[0]  # batch * q_len

    # One decode query per (sequence, position) pair; the row order already
    # matches q, so the output needs no reordering.
    cu_seqlens_q_spec = torch.arange(total + 1,
                                     dtype=torch.int32,
                                     device=q.device)
    dummy_cu_seqlens_k = torch.zeros_like(cu_seqlens_q_spec)
    # Causal per-position KV length: query row i*q_len + j (position j within
    # the q_len window of sequence i) attends to seqused_k[i] - (q_len-1-j)
    # tokens.
    pos = torch.arange(q_len, device=q.device, dtype=seqused_k.dtype)
    seqused_k_spec = (seqused_k.view(batch, 1) -
                      (q_len - 1 - pos).view(1, q_len)).clamp_(min=1).reshape(
                          total).to(torch.int32)

    # Expand the block table so each pseudo-sequence points at its parent
    # sequence's blocks.
    block_table_spec = block_table.repeat_interleave(q_len, dim=0)

    out, _ = torch.ops._vllm_fa2_C.varlen_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q_spec,
        dummy_cu_seqlens_k,
        seqused_k_spec,
        None,
        block_table_spec,
        None,  # alibi_slopes
        1,  # max_seqlen_q
        max_seqlen_k,
        0.0,  # dropout_p
        k_descale,
        v_descale,
        softmax_scale,
        s_aux,
        False,  # zero_tensors
        False,  # causal: handled by per-position seqused_k
        window_size[0],
        window_size[1],
        softcap,
        False,  # return_softmax
        None,  # gen
        None,  # num_splits (let the kernel pick via get_num_splits)
        False,  # is_mix_batch
        None,  # splits_per_seq
        None,  # work_list
    )
    return out


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _as_int32_device_tensor(x, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.int32)
    return torch.tensor(x, device=device, dtype=torch.int32)


def _normalize_descale_tensor(
    descale: Optional[torch.Tensor],
    name: str,
) -> Optional[torch.Tensor]:
    if descale is None:
        return None

    assert descale.dtype in (torch.float32, torch.bfloat16, torch.float16), \
        f"{name} must be a float32 or bfloat16 or float16 scalar tensor view"
    assert descale.untyped_storage().nbytes() == descale.element_size(), \
        f"{name} must be view of single scalar tensor"

    if descale.dtype == torch.float32:
        return descale

    descale_scalar = descale.flatten()[0].to(dtype=torch.float32)
    return descale_scalar if descale.ndim == 0 else descale_scalar.expand(
        descale.shape)


def _kv_tile_from_block_size(block_size: int) -> int:
    # Mirror of flash_api.cpp get_num_splits() / TileShapeQK<1>.
    if block_size == 16:
        return 16
    if block_size == 32:
        return 32
    return 64


def _min_blocks_for_split(kv_tile: int) -> int:
    # Mirror of XeFMHAFwdSplitKVKernel::kMinBlocksForSplit /
    # ReduceSplitK::kMinBlocksForSplit. Below this threshold a sequence is
    # processed as a single split for numerical stability.
    return 32 if kv_tile <= 64 else 128


def _infer_num_xe_cores(device: torch.device) -> int:
    # Prefer runtime query; fall back to 20 (Xe2/BMG default).
    try:
        props = torch.xpu.get_device_properties(device)
    except Exception:
        return 20
    slices = (getattr(props, "gpu_slices", None)
              or getattr(props, "num_slices", None)
              or getattr(props, "slices", None))
    subslices = (getattr(props, "gpu_subslices_per_slice", None)
                 or getattr(props, "num_subslices_per_slice", None)
                 or getattr(props, "subslices_per_slice", None))
    if slices is not None and subslices is not None:
        return max(1, int(slices) * int(subslices))
    return 20


def build_decode_split_plan(
    kv_lens,
    kv_tile: int,
    num_kv_splits: int,
    num_xe_cores: int,
    num_heads_kv: int,
):
    """Produce (splits_per_seq, work_list) for the compact-grid decode kernel.

    Inputs
    ------
    kv_lens         : per-seq KV length in tokens (list/tensor, on host).
    kv_tile         : KV tile width in tokens (must equal the kernel's
                      get<1>(TileShapeQK{})).
    num_kv_splits   : global cap on per-seq split count (buffer dim).
    num_xe_cores    : Xe-core count used for the target-WG heuristic.
    num_heads_kv    : KV heads (workload is sliced across heads_kv heads).

    Returns
    -------
    splits_per_seq  : int32 cpu tensor [batch], splits[i] >= 1.
    work_list       : int32 cpu tensor [total_wgs, 4] of
                      (seq_idx, kv_tile_start, kv_tile_count, split_idx).

    Guarantees
    ----------
    - sum(splits_per_seq) == work_list.size(0) == total_wgs
    - For every emitted work item, kv_tile_count >= 1
    - For each seq, the work items partition [0, kv_tiles) exactly once
    - splits_per_seq[i] <= num_kv_splits (so Oaccum/exp_sums/max_logits
      buffer indexing is safe)
    - splits_per_seq[i] folds in {single-split heuristic, balanced
      assignment, hard cap}; the kernel never needs to second-guess it.
    """
    if isinstance(kv_lens, torch.Tensor):
        kv_lens_list = kv_lens.to(dtype=torch.int32, device="cpu").tolist()
    else:
        kv_lens_list = [int(v) for v in kv_lens]

    tiles_per_seq = [max(1, (kv + kv_tile - 1) // kv_tile)
                     for kv in kv_lens_list]
    total_tiles = sum(tiles_per_seq)

    # Target: ~2x oversubscription of Xe cores per kv head, minimum 4 tiles
    # per WG so split-K overhead stays amortized.
    min_wgs = max(1, num_xe_cores * 2 // max(1, num_heads_kv))
    target_tiles_per_wg = max(4, total_tiles // min_wgs)

    # Mirror of the kernel's is_single_split heuristic: avoid split-reduce
    # for short sequences (numerical stability + overhead).
    min_blocks_for_split = _min_blocks_for_split(kv_tile)

    splits_per_seq = []
    work_items = []
    for i, n_tiles in enumerate(tiles_per_seq):
        if (n_tiles <= target_tiles_per_wg
                or n_tiles < min_blocks_for_split
                or num_kv_splits <= 1):
            n_splits = 1
        else:
            n_splits = ((n_tiles + target_tiles_per_wg - 1)
                        // target_tiles_per_wg)
            # Cap to the static buffer dim AND to n_tiles, so every emitted
            # work item has kv_tile_count >= 1.
            n_splits = min(n_splits, num_kv_splits, n_tiles)
        splits_per_seq.append(n_splits)

        # Even partitioning: first `rem` splits get base+1 tiles, the rest
        # get base. Guarantees count >= 1 since n_splits <= n_tiles.
        base, rem = divmod(n_tiles, n_splits)
        start = 0
        for s in range(n_splits):
            count = base + (1 if s < rem else 0)
            work_items.append([i, start, count, s])
            start += count

    splits_t = torch.tensor(splits_per_seq, dtype=torch.int32)
    work_t = torch.tensor(work_items, dtype=torch.int32)
    return splits_t, work_t


# Backward-compat shim: previously this returned only splits_per_seq.
def compute_splits_per_seq(
    kv_lens,
    kv_tile: int,
    num_kv_splits: int = 32,
    num_xe_cores: int = 20,
    num_heads_kv: int = 2,
) -> torch.Tensor:
    splits, _ = build_decode_split_plan(
        kv_lens, kv_tile, num_kv_splits, num_xe_cores, num_heads_kv)
    return splits


def ref_paged_attn(query: torch.Tensor,
                   key_cache: torch.Tensor,
                   value_cache: torch.Tensor,
                   query_lens: list[int],
                   kv_lens: list[int],
                   block_tables: torch.Tensor,
                   scale: float,
                   window_size_left: Optional[int] = None,
                   window_size_right: Optional[int] = None,
                   soft_cap: Optional[float] = None,
                   is_paged: Optional[bool] = True,
                   casual: Optional[bool] = False,
                   sink: Optional[torch.Tensor] = None,
                   q_descale: Optional[torch.Tensor] = None,
                   k_descale: Optional[torch.Tensor] = None,
                   v_descale: Optional[torch.Tensor] = None,
                   is_fp8kv: bool = False,
                   is_fp8_query: bool = False,
                   dtype: torch.dtype = torch.bfloat16,
                   return_softmax_lse: bool = False):
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    if is_paged:
        _, block_size, num_kv_heads, head_size = key_cache.shape
        v_head_size = value_cache.shape[-1]
    else:
        _, num_kv_heads, head_size = key_cache.shape
        v_head_size = value_cache.shape[-1]

    if is_fp8_query:
        query = (query.to(torch.float32) * q_descale).to(dtype)

    outputs: list[torch.Tensor] = []
    lse_list: list[torch.Tensor] = []
    start_idx = 0
    start_idx_kv = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx:start_idx + query_len] * scale

        if is_paged:
            num_kv_blocks = (kv_len + block_size - 1) // block_size
            block_indices = block_tables[i, :num_kv_blocks]

            k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
            k = k[:kv_len]
            v = value_cache[block_indices].view(-1, num_kv_heads, v_head_size)
            v = v[:kv_len]
        else:
            k = key_cache[start_idx_kv:start_idx_kv + kv_len]
            v = value_cache[start_idx_kv:start_idx_kv + kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1],
                                        dim=1).contiguous()
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1],
                                        dim=1).contiguous()

        if is_fp8kv:
            k = (k.to(torch.float32) * k_descale).to(dtype)
            v = (v.to(torch.float32) * v_descale).to(dtype)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if window_size_right > 0 or window_size_left > 0:
            if window_size_right < 0:
                window_size_right = max(kv_lens)
            if window_size_left < 0:
                window_size_left = max(kv_lens)

            mask_right = torch.triu(empty_mask,
                                    diagonal=kv_len - query_len +
                                    window_size_right + 1).bool()
            mask_left = torch.triu(empty_mask,
                                   diagonal=kv_len - query_len -
                                   window_size_left).bool().logical_not()
            mask_local = mask_right | mask_left
            attn.masked_fill_(mask_local, float("-inf"))
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        if casual:
            attn.masked_fill_(mask, float("-inf"))
        if sink is not None:
            sink_expanded = sink.view(sink.size()[0], 1,
                                      1).expand(attn.size()[0],
                                                attn.size()[1], 1)
            attn = torch.cat([attn, sink_expanded], dim=-1)
        if return_softmax_lse:
            # lse shape: [heads, query_len] -> transpose to [query_len, heads]
            lse = torch.logsumexp(attn, dim=-1).transpose(0, 1).contiguous()
            lse_list.append(lse)
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        if sink is not None:
            attn = attn[..., :-1]
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len
        start_idx_kv += kv_len

    out_tensor = torch.cat(outputs, dim=0)
    if return_softmax_lse:
        return out_tensor, torch.cat(lse_list, dim=0).float()
    return out_tensor


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
    is_mix_batch: bool = True,
    host_kv_lens: Optional[torch.Tensor] = None,
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
        host_kv_lens: Optional host-side per-seq KV lengths. If provided,
            this function converts them to an int32 device tensor and uses
            it as seqused_k.
        block_table: Optional block table for paged KV cache.
        num_splits: Backend-specific split parameter (non-KV specific),
            typically used to control work partitioning in some FA versions.
        num_splits_kv: Optional number of splits applied to KV **blocks**
            when using paged KV cache. This is forwarded to the underlying
            C++ FlashAttention op as its ``num_splits`` parameter; the split
            unit is KV blocks, not individual tokens or pages.
        fa_version: FlashAttention backend version selector.
    """
    if host_kv_lens is not None:
        if seqused_k is not None:
            raise ValueError("Provide only one of host_kv_lens and seqused_k")
        seqused_k = _as_int32_device_tensor(host_kv_lens, q.device)

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
    k_descale = _normalize_descale_tensor(k_descale, "k_descale")
    v_descale = _normalize_descale_tensor(v_descale, "v_descale")
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

        # Speculative-decoding fast path: redirect small, uniform, causal,
        # paged query batches (q_len = num_speculative_tokens + 1) from the
        # slow chunk-prefill kernel to the split-K decode kernel. See the
        # comment on _SPEC_DECODE_MAX_QLEN above for the rationale.
        batch = cu_seqlens_q.numel() - 1
        is_uniform_qlen = (batch > 0 and q.shape[0] == batch * max_seqlen_q)
        if (block_table is not None and causal and not return_softmax_lse
                and softcap == 0.0 and alibi_slopes is None and q_v is None
                and q_descale is None and scheduler_metadata is None
                and seqused_k is not None
                and 1 < max_seqlen_q <= _SPEC_DECODE_MAX_QLEN
                and is_uniform_qlen):
            return _spec_decode_varlen_fwd(
                q,
                k,
                v,
                out,
                cu_seqlens_q,
                seqused_k,
                block_table,
                max_seqlen_k,
                k_descale,
                v_descale,
                softmax_scale,
                s_aux,
                real_window_size,
                softcap,
            )

        # Compute per-seq splits and work_list on host, upload to device.
        # Only enable for decode (max_seqlen_q == 1) with paged KV cache,
        # multi-seq batches, and global num_splits_kv > 1.
        splits_per_seq_dev = None
        work_list_dev = None
        if (block_table is not None and host_kv_lens is not None
                and num_splits_kv is not None and num_splits_kv > 1
                and max_seqlen_q == 1):
            block_size = k.size(1)
            kv_tile = _kv_tile_from_block_size(block_size)
            num_xe_cores = _infer_num_xe_cores(q.device)
            num_heads_kv = k.size(2)
            splits_cpu, work_list_cpu = build_decode_split_plan(
                host_kv_lens,
                kv_tile=kv_tile,
                num_kv_splits=num_splits_kv,
                num_xe_cores=num_xe_cores,
                num_heads_kv=num_heads_kv,
            )
            if work_list_cpu.numel() > 0:
                splits_per_seq_dev = splits_cpu.to(
                    device=q.device, non_blocking=True)
                work_list_dev = work_list_cpu.to(
                    device=q.device, non_blocking=True)

        try:
            out, softmax_lse = torch.ops._vllm_fa2_C.varlen_fwd(
                q,
                k,
                v,
                out,
                cu_seqlens_q,
                # cu_seqlens_k not used since we use seqused_k, but
                # flash_api.cpp still wants it so we pass all zeros
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
                is_mix_batch,
                splits_per_seq_dev,
                work_list_dev,
            )
        except RuntimeError as e:
            if "not compiled" not in str(e):
                raise
            # Fallback to PyTorch reference implementation.
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "XPU kernel not compiled for this config, falling back "
                "to PyTorch reference attention. Performance will be "
                "significantly degraded.\n"
                "To fix: rebuild with the config line shown above.\n"
                "If this is unexpected, report at: "
                "https://github.com/vllm-project/vllm-xpu-kernels/issues/364\n"
                "Original error: %s", e)
            out, softmax_lse = _fallback_varlen_attn(
                q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_k,
                block_table, softmax_scale, causal,
                real_window_size, softcap,
                k_descale=k_descale,
                v_descale=v_descale,
                s_aux=s_aux,
                return_softmax_lse=return_softmax_lse,
            )
    else:
        raise NotImplementedError("not support yet")
    return (out, softmax_lse) if return_softmax_lse else (out)


def _fallback_varlen_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    scale: float,
    causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    s_aux: Optional[torch.Tensor] = None,
    return_softmax_lse: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """PyTorch reference fallback when XPU kernel is not compiled."""
    cu_q = cu_seqlens_q.cpu().tolist()
    num_seqs = len(cu_q) - 1
    query_lens = [cu_q[i + 1] - cu_q[i] for i in range(num_seqs)]

    # The kernel API accepts k/v_descale as (num_seqs, num_kv_heads) views of
    # a single scalar (all strides == 0).  ref_paged_attn expects a scalar
    # that broadcasts with (kv_len, num_kv_heads, head_size).
    if k_descale is not None:
        k_descale = k_descale.flatten()[0]
    if v_descale is not None:
        v_descale = v_descale.flatten()[0]

    # Determine if KV cache is FP8 and needs dequantization
    is_fp8kv = k_descale is not None and k.dtype in (
        torch.float8_e4m3fn, torch.float8_e5m2,
        torch.float8_e4m3fnuz, torch.float8_e5m2fnuz)
    # Infer the compute dtype from query (if query is also fp8, fall back to
    # float16 as the compute type)
    if q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2,
                   torch.float8_e4m3fnuz, torch.float8_e5m2fnuz):
        compute_dtype = torch.float16
    else:
        compute_dtype = q.dtype

    is_paged = block_table is not None and seqused_k is not None

    if is_paged:
        kv_lens = seqused_k.cpu().tolist()
        result = ref_paged_attn(
            query=q,
            key_cache=k,
            value_cache=v,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=block_table,
            scale=scale,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            soft_cap=softcap if softcap > 0.0 else None,
            is_paged=True,
            casual=causal,
            sink=s_aux,
            k_descale=k_descale,
            v_descale=v_descale,
            is_fp8kv=is_fp8kv,
            dtype=compute_dtype,
            return_softmax_lse=return_softmax_lse,
        )
    else:
        cu_k = cu_seqlens_k.cpu().tolist()
        kv_lens = [cu_k[i + 1] - cu_k[i] for i in range(num_seqs)]
        result = ref_paged_attn(
            query=q,
            key_cache=k,
            value_cache=v,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=torch.zeros((num_seqs, 1), dtype=torch.int32,
                                     device=q.device),
            scale=scale,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            soft_cap=softcap if softcap > 0.0 else None,
            is_paged=False,
            casual=causal,
            sink=s_aux,
            k_descale=k_descale,
            v_descale=v_descale,
            is_fp8kv=is_fp8kv,
            dtype=compute_dtype,
            return_softmax_lse=return_softmax_lse,
        )

    if return_softmax_lse:
        return result[0], result[1]
    return result, None


