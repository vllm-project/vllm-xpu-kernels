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
