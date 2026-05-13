# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401
from tests import register_ops as ops

MEMCPY_HOST_TO_DEVICE = 0
MEMCPY_DEVICE_TO_HOST = 1
MEMCPY_DEVICE_TO_DEVICE = 2

XPU_DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

# CI/mini scope parameter overrides
MINI_PYTEST_PARAMS = {
    "default": {
        "device": ["xpu:0"],
    },
}


@pytest.mark.parametrize("device", XPU_DEVICES)
def test_xpu_memcpy_sync_host_to_device(device: str) -> None:
    torch.xpu.set_device(device)
    src = torch.arange(0, 1024, dtype=torch.int32, device="cpu")
    dst = torch.empty_like(src, device=device)

    ops.xpu_memcpy_sync(
        dst.data_ptr(),
        src.data_ptr(),
        src.numel() * src.element_size(),
        MEMCPY_HOST_TO_DEVICE,
        torch.xpu.current_device(),
    )

    torch.xpu.synchronize()
    assert torch.equal(dst.cpu(), src)


@pytest.mark.parametrize("device", XPU_DEVICES)
def test_xpu_memcpy_sync_device_to_host(device: str) -> None:
    torch.xpu.set_device(device)
    src = torch.arange(0, 2048, dtype=torch.int32, device=device)
    dst = torch.empty(src.shape, dtype=src.dtype, device="cpu")

    ops.xpu_memcpy_sync(
        dst.data_ptr(),
        src.data_ptr(),
        dst.numel() * dst.element_size(),
        MEMCPY_DEVICE_TO_HOST,
        torch.xpu.current_device(),
    )

    assert torch.equal(dst, src.cpu())


@pytest.mark.parametrize("device", XPU_DEVICES)
def test_xpu_memcpy_sync_device_to_device(device: str) -> None:
    torch.xpu.set_device(device)
    src = torch.arange(0, 1536, dtype=torch.float32, device=device)
    dst = torch.zeros_like(src)

    ops.xpu_memcpy_sync(
        dst.data_ptr(),
        src.data_ptr(),
        src.numel() * src.element_size(),
        MEMCPY_DEVICE_TO_DEVICE,
        torch.xpu.current_device(),
    )

    torch.xpu.synchronize()
    assert torch.equal(dst, src)


@pytest.mark.parametrize("device", XPU_DEVICES)
def test_xpu_memcpy_sync_invalid_kind(device: str) -> None:
    torch.xpu.set_device(device)
    src = torch.arange(0, 4, dtype=torch.int32, device="cpu")
    dst = torch.empty_like(src, device=device)

    with pytest.raises(RuntimeError, match="Unsupported memcpy kind"):
        ops.xpu_memcpy_sync(
            dst.data_ptr(),
            src.data_ptr(),
            src.numel() * src.element_size(),
            99,
            torch.xpu.current_device(),
        )
