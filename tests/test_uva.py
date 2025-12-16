# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401

XPU_DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

SKIP_TEST_FOR_MINI_SCOPE = os.getenv("XPU_KERNEL_PYTEST_PROFILER") == "MINI"


@pytest.mark.parametrize("device", XPU_DEVICES)
@pytest.mark.skipif(SKIP_TEST_FOR_MINI_SCOPE,
                    reason="Skip UVA tests for the mini pytest profiler.")
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
@pytest.mark.skipif(SKIP_TEST_FOR_MINI_SCOPE,
                    reason="Skip UVA tests for the mini pytest profiler.")
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
