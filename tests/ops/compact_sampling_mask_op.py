# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

# Mirrors vllm.v1.worker.gpu.sample.output.MAX_COMPACT_SUPPORT: bounds the
# [num_reqs, width] int32 buffer; wider rows fall back to the bitmask.
MAX_COMPACT_SUPPORT = 2048

# Set USE_TRITON_COMPACT_SAMPLING_MASK=1 to enable
# compact_sampling_mask_triton(), which runs vllm's own Triton kernel
# (_compact_sampling_mask_kernel) instead of the SYCL/PyTorch paths here.
# Requires vllm (with Triton) to be importable.
HAS_TRITON = False
USE_TRITON_COMPACT_SAMPLING_MASK = os.getenv(
    "USE_TRITON_COMPACT_SAMPLING_MASK", "False").lower() in ("1", "true")
if USE_TRITON_COMPACT_SAMPLING_MASK:
    try:
        from vllm.triton_utils import HAS_TRITON  # noqa: F401
        from vllm.v1.worker.gpu.sample.output import (  # noqa: F401
            _compact_sampling_mask_kernel)
    except ImportError:
        print("Triton is not available. compact_sampling_mask_triton() "
              "will raise if called.")

TRITON_AVAILABLE = USE_TRITON_COMPACT_SAMPLING_MASK and HAS_TRITON


def compact_sampling_mask_torch(
    logits: torch.Tensor,
    num_sampled_tokens: torch.Tensor,
    max_num_kept: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference implementation.

    Mirrors vllm's ``_compact_sampling_mask_torch``: for every row with a
    sampled token (``num_sampled_tokens > 0``), computes the finite-logit
    count, the first ``max_num_kept`` finite-logit token ids (increasing
    order), and the full finite-logit bitmask (little bit order, 1 byte per
    8 tokens).
    """
    num_reqs, vocab_size = logits.shape
    is_active = num_sampled_tokens > 0
    finite = torch.isfinite(logits) & is_active.unsqueeze(1)

    counts = finite.sum(dim=1).to(torch.int32)

    token_ids = torch.zeros(
        (num_reqs, max_num_kept), dtype=torch.int32, device=logits.device)
    for row in range(num_reqs):
        idx = finite[row].nonzero(as_tuple=True)[0]
        n = min(idx.numel(), max_num_kept)
        if n:
            token_ids[row, :n] = idx[:n].to(torch.int32)

    finite_np = finite.to("cpu", dtype=torch.uint8).numpy()
    packed_np = np.packbits(finite_np, axis=1, bitorder="little")
    packed_mask = torch.from_numpy(packed_np).to(
        device=logits.device, dtype=torch.uint8)

    return token_ids, packed_mask, counts


def compact_sampling_mask_xpu(
    logits: torch.Tensor,
    num_sampled_tokens: torch.Tensor,
    max_num_kept: int,
    max_compact_support: int = MAX_COMPACT_SUPPORT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """XPU (SYCL) implementation."""
    return torch.ops._xpu_C.compact_sampling_mask(
        logits, num_sampled_tokens, max_num_kept, max_compact_support)


def compact_sampling_mask_triton(
    logits: torch.Tensor,
    num_sampled_tokens: torch.Tensor,
    max_num_kept: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """vllm's own Triton kernel (``_compact_sampling_mask_kernel``).

    Only usable when USE_TRITON_COMPACT_SAMPLING_MASK=1 and vllm/Triton are
    importable (see TRITON_AVAILABLE above); raises otherwise.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "compact_sampling_mask_triton() requires "
            "USE_TRITON_COMPACT_SAMPLING_MASK=1 and vllm (with Triton) to "
            "be importable.")

    num_reqs, vocab_size = logits.shape
    device = logits.device

    token_ids = torch.empty((num_reqs, max_num_kept),
                            dtype=torch.int32,
                            device=device)
    packed_mask = torch.empty((num_reqs, (vocab_size + 7) // 8),
                              dtype=torch.uint8,
                              device=device)
    counts = torch.empty(num_reqs, dtype=torch.int32, device=device)

    _compact_sampling_mask_kernel[(num_reqs, )](
        logits,
        logits.stride(0),
        logits.stride(1),
        num_sampled_tokens,
        token_ids,
        token_ids.stride(0),
        packed_mask,
        packed_mask.stride(0),
        counts,
        vocab_size,
        max_num_kept,
        BLOCK_SIZE=8192,
    )
    return token_ids, packed_mask, counts


def unpack_support(
    token_ids: torch.Tensor,
    packed_mask: torch.Tensor,
    counts: torch.Tensor,
    vocab_size: int,
) -> list[np.ndarray]:
    """CPU-side helper mirroring ``SamplingMaskTensors.tolists()``: use the
    compact token ids when they fit within ``max_num_kept``, otherwise fall
    back to unpacking the bitmask.
    """
    counts_np = counts.cpu().numpy()
    token_ids_np = token_ids.cpu().numpy()
    packed_mask_np = packed_mask.cpu().numpy()
    width = token_ids_np.shape[1]

    supports = []
    for row in range(len(counts_np)):
        if counts_np[row] <= width:
            supports.append(token_ids_np[row, :counts_np[row]])
        else:
            bits = np.unpackbits(
                packed_mask_np[row], count=vocab_size, bitorder="little")
            supports.append(np.flatnonzero(bits).astype(np.int32))
    return supports
