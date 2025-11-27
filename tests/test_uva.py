# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401

CUDA_DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_cpu_write(device):
    torch.set_default_device(device)
    cpu_tensor = torch.zeros(10,
                             10,
                             device="cpu",
                             pin_memory=True,
                             dtype=torch.int32)
    cuda_view = torch.ops._C.get_xpu_view_from_cpu_tensor(cpu_tensor)
    assert cuda_view.device.type == "xpu"

    assert cuda_view[0, 0] == 0
    assert cuda_view[2, 3] == 0
    assert cuda_view[4, 5] == 0

    cpu_tensor[0, 0] = 1
    cpu_tensor[2, 3] = 2
    cpu_tensor[4, 5] = -1

    cuda_view.mul_(2)
    assert cuda_view[0, 0] == 2
    assert cuda_view[2, 3] == 4
    assert cuda_view[4, 5] == -2


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_gpu_write(device):
    torch.set_default_device(device)
    cpu_tensor = torch.zeros(10,
                             10,
                             device="cpu",
                             pin_memory=True,
                             dtype=torch.int32)
    cuda_view = torch.ops._C.get_xpu_view_from_cpu_tensor(cpu_tensor)
    assert cuda_view.device.type == "xpu"

    assert cuda_view[0, 0] == 0
    assert cuda_view[2, 3] == 0
    assert cuda_view[4, 5] == 0

    cuda_view[0, 0] = 1
    cuda_view[2, 3] = 2
    cuda_view[4, 5] = -1
    cuda_view.mul_(2)

    assert cpu_tensor[0, 0] == 2
    assert cpu_tensor[2, 3] == 4
    assert cpu_tensor[4, 5] == -2
