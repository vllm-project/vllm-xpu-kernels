# SPDX-License-Identifier: Apache-2.0
"""Helpers for the 2-rank Level Zero IPC all-reduce.

XPU device addresses sit above 2**63 and torch's ``int`` schema type is
int64_t, so addresses cross the ops as their two's-complement value
(:func:`as_fptr`). The IPC helpers convert for the caller. The collective
itself is called as ``torch.ops._xpu_C.xpu_p2p_all_reduce`` with pointers
converted once at setup, keeping a Python frame out of each call.
"""

import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

_U64 = 1 << 64
_I64_MAX = (1 << 63) - 1


def as_fptr(address: int) -> int:
    """Reinterpret an unsigned device address as the signed int64 ops take."""
    return address - _U64 if address > _I64_MAX else address


def alloc_region(slot_bytes: int) -> torch.Tensor:
    """Zeroed region for a staging slot of ``slot_bytes`` on the current XPU.

    An allocation of its own, outside torch's caching allocator: the peer
    writes into it over PCIe, which must not share an allocation with other
    tensors.
    """
    return torch.ops._xpu_C.xpu_p2p_alloc_region(slot_bytes)


def export_handle(address: int) -> tuple[torch.Tensor, int, int]:
    """Export the allocation containing ``address`` for a peer process.

    Returns ``(handle_bytes, dma_buf_fd, offset)``. The fd is process-local
    and must reach the peer over ``SCM_RIGHTS``. Raises on a multi-tile
    allocation.
    """
    return torch.ops._xpu_C.xpu_ipc_export_handle(as_fptr(address))


def release_handle(handle_bytes: torch.Tensor) -> None:
    """Release an exported handle once the peer has opened it.

    Do not also close the exported fd: the driver may close it here.
    """
    torch.ops._xpu_C.xpu_ipc_release_handle(handle_bytes)


def open_handle(handle_bytes: torch.Tensor, fd: int, offset: int) -> int:
    """Open a peer's exported allocation; returns the address at ``offset``.

    Pass ``result - offset`` to :func:`close_handle`.
    """
    fptr = torch.ops._xpu_C.xpu_ipc_open_handle(handle_bytes, fd, offset)
    return fptr + _U64 if fptr < 0 else fptr


def close_handle(base_address: int) -> None:
    """Close a mapping opened by :func:`open_handle`."""
    torch.ops._xpu_C.xpu_ipc_close_handle(as_fptr(base_address))
