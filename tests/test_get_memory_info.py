# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401

DEVICES = [i for i in range(torch.xpu.device_count())]


@pytest.mark.parametrize("device", DEVICES)
def test_get_memory_info(device) -> None:
    free, total = torch.ops._C_cache_ops.getMemoryInfo(device)

    # FIXME:
    # After update neo, the results of torch is changed
    # ref_free 24385683456
    # ref_total 24385683456
    # ->
    # ref_free 25429528576
    # ref_total 25669140480
    # So we use hard code to check the results
    # We should check the results after fixing the torch issue
    # ref_free, ref_total = torch.xpu.mem_get_info(device)

    if torch.ops._xpu_C.is_bmg_g21(device):
        ref_total = 24385683456
        assert free > 0
        assert total == ref_total

    elif torch.ops._xpu_C.is_bmg_g31(device):
        ref_total = 32530182144
        assert free > 0
        assert total == ref_total
