# SPDX-License-Identifier: Apache-2.0

import mmap

import pytest
import torch

from tests import register_ops as ops

NUM_BLOCKS = 16
BLOCK_BYTES = 64 * 1024
TOTAL = NUM_BLOCKS * BLOCK_BYTES


@pytest.fixture(autouse=True, scope="module")
def _force_cpu_default_device():
    """Pointer and size tables must be CPU tensors even if another test module
    leaked an XPU default device."""
    original = torch.get_default_device()
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(original)


def _ptrs(base: int, block_ids: list[int]) -> torch.Tensor:
    return torch.tensor([base + b * BLOCK_BYTES for b in block_ids],
                        dtype=torch.uint64,
                        device="cpu")


def _sizes(n: int) -> torch.Tensor:
    return torch.full((n, ), BLOCK_BYTES, dtype=torch.uint64, device="cpu")


@pytest.mark.parametrize("device", ["xpu:0"])
def test_roundtrip_registered_mmap(device: str) -> None:
    """D2H then H2D through registered mmap memory on a side stream, with
    permuted block ids, must preserve bytes."""
    torch.xpu.set_device(device)
    stream = torch.xpu.Stream()
    src_ids = list(range(NUM_BLOCKS))
    host_ids = src_ids[::-1]

    with mmap.mmap(-1, TOTAL) as region:
        host = torch.frombuffer(memoryview(region), dtype=torch.uint8)
        assert ops.xpu_host_register(host.data_ptr(), TOTAL)
        try:
            expected = torch.randint(0,
                                     256, (NUM_BLOCKS, BLOCK_BYTES),
                                     dtype=torch.uint8)
            dev = expected.to(device)
            stream.wait_stream(torch.xpu.current_stream())

            ops.xpu_memcpy_batch_async(_ptrs(host.data_ptr(), host_ids),
                                       _ptrs(dev.data_ptr(), src_ids),
                                       _sizes(NUM_BLOCKS), stream)
            stream.synchronize()
            torch.testing.assert_close(
                host.view(NUM_BLOCKS, BLOCK_BYTES)[host_ids], expected)

            out = torch.zeros_like(dev)
            stream.wait_stream(torch.xpu.current_stream())
            ops.xpu_memcpy_batch_async(_ptrs(out.data_ptr(), src_ids),
                                       _ptrs(host.data_ptr(), host_ids),
                                       _sizes(NUM_BLOCKS), stream)
            stream.synchronize()
            torch.testing.assert_close(out.cpu(), expected)
        finally:
            assert ops.xpu_host_unregister(host.data_ptr())
            del host


@pytest.mark.parametrize("device", ["xpu:0"])
def test_ordered_on_target_stream(device: str) -> None:
    """Copies run in order with prior work on the given stream, and skip
    zero-sized entries."""
    torch.xpu.set_device(device)
    stream = torch.xpu.Stream()
    src = torch.zeros(TOTAL, dtype=torch.uint8, device=device)
    dst = torch.zeros_like(src)
    with torch.xpu.stream(stream):
        src.fill_(7)
    sizes = _sizes(NUM_BLOCKS)
    sizes[0] = 0
    ops.xpu_memcpy_batch_async(_ptrs(dst.data_ptr(), range(NUM_BLOCKS)),
                               _ptrs(src.data_ptr(), range(NUM_BLOCKS)),
                               sizes, stream)
    stream.synchronize()
    out = dst.cpu().view(NUM_BLOCKS, BLOCK_BYTES)
    assert torch.all(out[0] == 0)
    assert torch.all(out[1:] == 7)


@pytest.mark.parametrize("device", ["xpu:0"])
def test_empty_batch(device: str) -> None:
    torch.xpu.set_device(device)
    empty = torch.empty(0, dtype=torch.uint64)
    ops.xpu_memcpy_batch_async(empty, empty, empty)
