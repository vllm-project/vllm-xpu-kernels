# SPDX-License-Identifier: Apache-2.0
"""Python surface for the 2-rank peer-to-peer collectives.

This module exists for one reason: **XPU device addresses do not fit in a
signed int64.** Level Zero hands out USM device pointers near the top of the
address space (e.g. 0xffffd3...), and torch's ``int`` schema type is
``int64_t``, so passing such an address straight to ``torch.ops`` raises

    Expected a value of type 'int' for argument 'ptr' but instead found
    type 'int' ... Value: 18446697167134392320

The ops cast the bit pattern straight back to a pointer, so handing them the
two's-complement value is exact and lossless -- but every caller would
otherwise have to know that. :func:`as_fptr` is that conversion, and the IPC
helpers below apply it for you.

This does not arise on CUDA, which is why vLLM's ``custom_all_reduce`` can
pass ``fptr_t = int64_t`` around without a helper.

The two collectives are deliberately *not* wrapped here: their pointer
arguments are fixed for the life of the communicator, so a caller converts
them once at setup with :func:`as_fptr` and then calls
``torch.ops._xpu_C.xpu_p2p_all_reduce`` / ``xpu_p2p_all_gather`` directly,
keeping a Python frame out of a collective that costs single-digit
microseconds.
"""

import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

_U64 = 1 << 64
_I64_MAX = (1 << 63) - 1


def as_fptr(address: int) -> int:
    """Reinterpret an unsigned device address as the signed int64 ops take."""
    return address - _U64 if address > _I64_MAX else address


def as_address(fptr: int) -> int:
    """Inverse of :func:`as_fptr`."""
    return fptr + _U64 if fptr < 0 else fptr


def signal_page_bytes() -> int:
    """Bytes to allocate for each per-collective flag and counter page.

    Read this rather than hardcoding it: it is derived from the kernel's own
    maximum workgroup count, and a page sized smaller than the kernel expects
    would be overrun.
    """
    return torch.ops._xpu_C.xpu_p2p_signal_page_bytes()


def export_handle(address: int) -> tuple[torch.Tensor, int, int]:
    """Export the allocation containing ``address`` for a peer process.

    Returns ``(handle_bytes, dma_buf_fd, offset)``. The fd is only valid in
    this process and must reach the peer over ``SCM_RIGHTS``; the handle bytes
    and offset are plain data.

    Raises on a multi-tile allocation: these collectives were validated on
    single-tile devices only, and a single Level Zero IPC handle cannot
    cover more. Under the FLAT device hierarchy (the driver default) each
    tile is its own device, so this does not arise there. See
    ``reject_untested_multi_tile_allocation()`` in
    ``csrc/xpu/p2p/p2p_ipc.cpp`` to lift it.
    """
    return torch.ops._xpu_C.xpu_ipc_export_handle(as_fptr(address))


def release_handle(handle_bytes: torch.Tensor) -> None:
    """Release an exported handle; call once the peer has opened it.

    Without this the exporting process keeps its dma-buf fd and the driver's
    export bookkeeping until it exits. Do *not* also ``os.close()`` the fd
    that :func:`export_handle` returned: the driver may close it here, and a
    second close could land on an unrelated fd that reused the number.

    This is not the counterpart of :func:`close_handle`, which unmaps on the
    importing side. Releasing before the peer has opened risks invalidating
    the export reference its open resolves against, so the call belongs
    after the all-ranks agreement that both sides opened successfully.
    """
    torch.ops._xpu_C.xpu_ipc_release_handle(handle_bytes)


def open_handle(handle_bytes: torch.Tensor, fd: int, offset: int) -> int:
    """Open a peer's exported allocation; returns the device address.

    ``fd`` is the dma-buf fd as received over ``SCM_RIGHTS``. Pass
    ``result - offset`` to :func:`close_handle`.
    """
    return as_address(
        torch.ops._xpu_C.xpu_ipc_open_handle(handle_bytes, fd, offset)
    )


def close_handle(base_address: int) -> None:
    """Close a mapping opened by :func:`open_handle`."""
    torch.ops._xpu_C.xpu_ipc_close_handle(as_fptr(base_address))


def memcpy(dst: int, src: int, nbytes: int) -> None:
    """Enqueue a device-to-device copy on the current stream."""
    torch.ops._xpu_C.xpu_p2p_memcpy(as_fptr(dst), as_fptr(src), nbytes)


def queue_sync() -> None:
    """Host-synchronize the current stream."""
    torch.ops._xpu_C.xpu_p2p_queue_sync()
