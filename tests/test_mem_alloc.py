# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

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
