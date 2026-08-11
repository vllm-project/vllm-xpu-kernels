# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc

import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401

XPU_DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

# Skip entire module in mini scope
SKIP_IN_MINI_SCOPE = True

# CI scope parameter overrides
MINI_PYTEST_PARAMS = {
    "default": {
        "device": ["xpu:0"],
    },
}


@pytest.mark.parametrize("device", XPU_DEVICES)
def test_cpu_write(device):
    torch.set_default_device(device)
    cpu_tensor = torch.zeros(10,
                             10,
                             device="cpu",
                             pin_memory=True,
                             dtype=torch.int32)
    xpu_view = torch.ops._C.get_xpu_view_from_cpu_tensor(cpu_tensor)
    assert xpu_view.device.type == "xpu"

    assert xpu_view[0, 0] == 0
    assert xpu_view[2, 3] == 0
    assert xpu_view[4, 5] == 0

    cpu_tensor[0, 0] = 1
    cpu_tensor[2, 3] = 2
    cpu_tensor[4, 5] = -1

    xpu_view.mul_(2)
    assert xpu_view[0, 0] == 2
    assert xpu_view[2, 3] == 4
    assert xpu_view[4, 5] == -2


@pytest.mark.parametrize("device", XPU_DEVICES)
def test_gpu_write(device):
    torch.set_default_device(device)
    cpu_tensor = torch.zeros(10,
                             10,
                             device="cpu",
                             pin_memory=True,
                             dtype=torch.int32)
    xpu_view = torch.ops._C.get_xpu_view_from_cpu_tensor(cpu_tensor)
    assert xpu_view.device.type == "xpu"

    assert xpu_view[0, 0] == 0
    assert xpu_view[2, 3] == 0
    assert xpu_view[4, 5] == 0

    xpu_view[0, 0] = 1
    xpu_view[2, 3] = 2
    xpu_view[4, 5] = -1
    xpu_view.mul_(2)

    assert cpu_tensor[0, 0] == 2
    assert cpu_tensor[2, 3] == 4
    assert cpu_tensor[4, 5] == -2


@pytest.mark.parametrize("device", XPU_DEVICES)
def test_non_pinned_cpu_tensor(device):
    # Non-pinned CPU tensors are internally copied into a pinned buffer,
    # so the resulting XPU view reflects the values at creation time but
    # is decoupled from further writes to the original `cpu_tensor`.
    torch.set_default_device(device)
    cpu_tensor = torch.arange(100,
                              dtype=torch.int32,
                              device="cpu").view(10, 10)
    assert not cpu_tensor.is_pinned()
    xpu_view = torch.ops._C.get_xpu_view_from_cpu_tensor(cpu_tensor)
    assert xpu_view.device.type == "xpu"

    assert xpu_view[0, 0] == 0
    assert xpu_view[2, 3] == 23
    assert xpu_view[9, 9] == 99

    # Writes to the original (unpinned) CPU tensor must not affect the view,
    # since a private pinned copy was made.
    cpu_tensor[0, 0] = -1
    assert xpu_view[0, 0] == 0

    # The view itself remains writable and independently usable.
    xpu_view.mul_(2)
    assert xpu_view[2, 3] == 46
    assert xpu_view[9, 9] == 198


@pytest.mark.parametrize("device", XPU_DEVICES)
def test_empty_cpu_tensor(device):
    torch.set_default_device(device)
    cpu_tensor = torch.empty(0, dtype=torch.int32, device="cpu")
    xpu_view = torch.ops._C.get_xpu_view_from_cpu_tensor(cpu_tensor)
    assert xpu_view.device.type == "xpu"
    assert xpu_view.numel() == 0
    assert xpu_view.shape == cpu_tensor.shape


@pytest.mark.parametrize("device", XPU_DEVICES)
def test_view_lifetime_after_owner_drop(device):
    torch.set_default_device(device)
    cpu_tensor = torch.arange(100,
                              dtype=torch.int32,
                              device="cpu",
                              pin_memory=True).view(10, 10)
    xpu_view = torch.ops._C.get_xpu_view_from_cpu_tensor(cpu_tensor)

    # Drop the original owner reference and force Python GC.
    del cpu_tensor
    gc.collect()

    # Exercise both read and write from the XPU view after owner drop.
    assert xpu_view[2, 3].item() == 23
    xpu_view.add_(1)
    assert xpu_view[0, 0].item() == 1
    assert xpu_view[9, 9].item() == 100
