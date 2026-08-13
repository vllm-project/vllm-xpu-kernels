# SPDX-License-Identifier: Apache-2.0

import mmap

import pytest
import torch

from tests import register_ops as ops

# CI/mini scope parameter overrides
MINI_PYTEST_PARAMS = {
    "test_host_register_mmap_roundtrip": {
        "device": ["xpu:0"],
    },
    "test_host_register_rejects_empty": {
        "device": ["xpu:0"],
    },
}


def _ptr_tensor(ptr: int) -> torch.Tensor:
    """Wrap a raw address in a single-element uint64 tensor.

    xpu_host_register/xpu_host_unregister take the address this way because a
    scalar int64 op argument cannot losslessly carry every 64-bit pointer bit
    pattern (addresses with bit 63 set, as seen with USM-mapped host
    allocations on XPU, overflow the signed range and fail to cast from
    Python). The mask normalizes Tensor.data_ptr()'s Python int into the exact
    bit pattern torch.uint64 expects.
    """
    return torch.tensor([ptr & 0xFFFFFFFFFFFFFFFF],
                        dtype=torch.uint64,
                        device="cpu")


@pytest.mark.parametrize("device", ["xpu:0"])
def test_host_register_mmap_roundtrip(device: str) -> None:
    """Registered mmap memory must still transfer correct bytes.

    Registration flips swap_blocks_batch from the synchronous pageable path to
    async DMA, so this guards against the copy being observed before it lands.
    """
    num_blocks = 8
    block_size_in_bytes = 4096
    total = num_blocks * block_size_in_bytes

    torch.xpu.set_device(device)

    with mmap.mmap(-1, total) as region:
        host = torch.frombuffer(memoryview(region), dtype=torch.uint8)
        host_ptr = host.data_ptr()

        assert ops.xpu_host_register(_ptr_tensor(host_ptr), total)
        try:
            expected = torch.randint(0, 256, (total, ), dtype=torch.uint8)
            dev = expected.to(device)

            # D2H into the registered region.
            sizes = torch.full((num_blocks, ),
                               block_size_in_bytes,
                               dtype=torch.uint64)
            offsets = torch.arange(num_blocks) * block_size_in_bytes
            src = (offsets + dev.data_ptr()).to(torch.uint64)
            dst = (offsets + host_ptr).to(torch.uint64)
            ops.swap_blocks_batch(src, dst, sizes)
            torch.xpu.synchronize()
            torch.testing.assert_close(host, expected)

            # H2D back out of the registered region.
            out = torch.zeros(total, dtype=torch.uint8, device=device)
            src = (offsets + host_ptr).to(torch.uint64)
            dst = (offsets + out.data_ptr()).to(torch.uint64)
            ops.swap_blocks_batch(src, dst, sizes)
            torch.xpu.synchronize()
            torch.testing.assert_close(out.cpu(), expected)
        finally:
            assert ops.xpu_host_unregister(_ptr_tensor(host_ptr))
            del host


@pytest.mark.parametrize("device", ["xpu:0"])
def test_host_register_rejects_empty(device: str) -> None:
    """Degenerate ranges report failure instead of registering nothing."""
    torch.xpu.set_device(device)
    buf = torch.zeros(4096, dtype=torch.uint8)

    assert not ops.xpu_host_register(_ptr_tensor(0), 4096)
    assert not ops.xpu_host_register(_ptr_tensor(buf.data_ptr()), 0)
    assert not ops.xpu_host_unregister(_ptr_tensor(0))

