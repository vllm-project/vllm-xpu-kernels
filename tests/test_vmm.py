# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401

DEVICES = [i for i in range(torch.xpu.device_count())]
ops = torch.ops._C


@pytest.mark.parametrize("device", DEVICES)
def test_vmm_round_trip(device) -> None:
    """Back the second granule of a two-granule reservation, use it from a
    kernel through a tensor view, then tear everything down."""
    granularity = ops.vmm_granularity(device)
    assert granularity > 0
    base = ops.vmm_reserve(2 * granularity)
    assert base != 0
    ptr = base + granularity
    handle = ops.vmm_create(granularity, device)
    try:
        ops.vmm_map(ptr, granularity, handle)
        try:
            ops.vmm_set_access(ptr, granularity, device)
            view = ops.vmm_tensor_from_ptr(ptr, granularity, device)
            assert view.device == torch.device("xpu", device)
            assert view.dtype == torch.uint8
            assert view.data_ptr() == ptr
            assert view.untyped_storage().nbytes() == granularity
            view.fill_(3)
            view[:256].copy_(
                torch.arange(256, device=view.device).to(torch.uint8))
            torch.xpu.synchronize(device)
            assert view[:256].cpu().tolist() == list(range(256))
            assert int(view[256:].sum()) == 3 * (granularity - 256)
            del view
        finally:
            ops.vmm_unmap(ptr, granularity)
    finally:
        ops.vmm_release(handle)
        ops.vmm_free_reserved(base, 2 * granularity)


@pytest.mark.parametrize("device", DEVICES)
def test_vmm_commit_reduces_free_memory(device) -> None:
    """Callers size what they map from getMemoryInfo, so mapped physical memory
    must be reported as used."""
    size = 1 << 30
    size -= size % ops.vmm_granularity(device)
    torch.xpu.synchronize(device)
    free_before, _ = torch.ops._C_cache_ops.getMemoryInfo(device)
    base = ops.vmm_reserve(size)
    handle = ops.vmm_create(size, device)
    try:
        ops.vmm_map(base, size, handle)
        free_after, _ = torch.ops._C_cache_ops.getMemoryInfo(device)
        ops.vmm_unmap(base, size)
    finally:
        ops.vmm_release(handle)
        ops.vmm_free_reserved(base, size)
    assert free_before - free_after >= size // 2


def test_vmm_reports_errors() -> None:
    with pytest.raises(RuntimeError, match="Unknown VMM handle"):
        ops.vmm_release(-1)
    with pytest.raises(RuntimeError, match="Unknown VMM handle"):
        ops.vmm_map(0, 1, -1)
    with pytest.raises(RuntimeError, match="must be positive"):
        ops.vmm_reserve(0)
