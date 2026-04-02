# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401

DEVICES = [i for i in range(torch.xpu.device_count())]


@pytest.mark.parametrize("device", DEVICES)
def test_get_memory_info(device) -> None:
    free, total = torch.ops._C_cache_ops.getMemoryInfo(device)

    ref_free, ref_total = torch.xpu.mem_get_info(device)

    if not torch.ops._xpu_C.is_pvc(device):
        assert total == ref_total
    assert free == 0
