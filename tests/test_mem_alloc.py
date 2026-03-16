# SPDX-License-Identifier: Apache-2.0

import gc

import pytest
import torch
from packaging import version

xpumem_allocator = pytest.importorskip("vllm_xpu_kernels.xpumem_allocator")


@pytest.mark.skipif(not torch.xpu.is_available(),
                    reason="XPU is not available")
def test_init_module_requires_callables() -> None:
    with pytest.raises(TypeError, match="Both arguments must be callables"):
        xpumem_allocator.init_module(1, 2)


@pytest.mark.skipif(not torch.xpu.is_available(),
                    reason="XPU is not available")
def test_unmap_and_release_requires_four_args() -> None:
    with pytest.raises(TypeError, match="Expected a tuple of size 4"):
        xpumem_allocator.python_unmap_and_release(0, 1, 2)


@pytest.mark.skipif(not torch.xpu.is_available(),
                    reason="XPU is not available")
def test_create_and_allocate_requires_four_args() -> None:
    with pytest.raises(TypeError, match="Expected a tuple of size 4"):
        xpumem_allocator.python_create_and_allocate(0, 1, 2)


@pytest.mark.skipif(not torch.xpu.is_available(),
                    reason="XPU is not available")
def test_unmap_and_release_unknown_pointer_raises() -> None:
    bogus_ptr = 0x12345678
    with pytest.raises(RuntimeError, match="pointer not found in memory map"):
        xpumem_allocator.python_unmap_and_release(0, 0, bogus_ptr, 0)


@pytest.mark.skipif(not torch.xpu.is_available(),
                    reason="XPU is not available")
def test_create_and_allocate_unknown_pointer_raises() -> None:
    bogus_ptr = 0x12345678
    with pytest.raises(RuntimeError, match="pointer not found in memory map"):
        xpumem_allocator.python_create_and_allocate(0, 0, bogus_ptr, 0)


@pytest.mark.skipif(not torch.xpu.is_available(),
                    reason="XPU is not available")
@pytest.mark.skipif(version.parse(torch.__version__) <= version.parse("2.10"),
                    reason="Torch version should be newer or equal 2.11")
def test_pluggable_allocator_allocate_release() -> None:
    allocation_records: dict[int, tuple[int, int, int]] = {}
    freed_ptrs: set[int] = set()

    def python_malloc_fn(allocation_handle):
        device, size, ptr, handle = allocation_handle
        allocation_records[ptr] = (device, size, handle)
        return None

    def python_free_fn(ptr):
        device, size, handle = allocation_records.pop(ptr)
        freed_ptrs.add(ptr)
        return (device, size, ptr, handle)

    xpumem_allocator.init_module(python_malloc_fn, python_free_fn)

    mem_mod = getattr(torch.xpu, "memory", None)
    assert mem_mod is not None, "torch.xpu.memory is not available"
    assert hasattr(mem_mod, "XPUPluggableAllocator"), (
        "torch.xpu.memory.XPUPluggableAllocator is not available")
    assert hasattr(mem_mod, "MemPool") and hasattr(mem_mod, "use_mem_pool"), (
        "torch.xpu.memory MemPool APIs are not available")

    alloc_cls = mem_mod.XPUPluggableAllocator
    pluggable_allocator = alloc_cls(xpumem_allocator.__file__, "my_malloc",
                                    "my_free")
    mem_pool = mem_mod.MemPool(pluggable_allocator._allocator)

    ptr = None
    with mem_mod.use_mem_pool(mem_pool):
        tensor = torch.empty(256, dtype=torch.float32, device="xpu")
        ptr = tensor.data_ptr()
        assert ptr in allocation_records, "allocation not tracked by callback"
        del tensor

    # MemPool may cache freed blocks; force allocator/pool teardown first.
    del mem_pool
    del pluggable_allocator

    gc.collect()
    assert ptr is not None
    assert ptr in freed_ptrs or ptr not in allocation_records, (
        "allocation was not released after pool teardown")
