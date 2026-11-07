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


@pytest.fixture(autouse=True, scope="module")
def _force_cpu_default_device():
    """Guard against default-device leakage from other test modules.

    Some tests (e.g. in test_cache.py) call `torch.set_default_device(...)`
    without restoring it, which is process-global state. If that leaks in
    here, plain `torch.tensor(...)`/`torch.arange(...)` calls below would
    silently land on XPU instead of CPU, breaking the CPU-only tensor
    contract that `swap_blocks_batch` requires for its address/size args.
    """
    original = torch.get_default_device()
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(original)

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

        assert ops.xpu_host_register(host_ptr, total)
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
            assert ops.xpu_host_unregister(host_ptr)
            del host


@pytest.mark.parametrize("device", ["xpu:0"])
def test_host_register_rejects_empty(device: str) -> None:
    """Degenerate ranges report failure instead of registering nothing."""
    torch.xpu.set_device(device)
    buf = torch.zeros(4096, dtype=torch.uint8)

    assert not ops.xpu_host_register(0, 4096)
    assert not ops.xpu_host_register(buf.data_ptr(), 0)
    assert not ops.xpu_host_unregister(0)
