# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

DEVICES = [i for i in range(torch.xpu.device_count())]

ARCH_OPS = ["is_bmg", "is_pvc", "is_cri", "is_nvl_p"]


@pytest.mark.parametrize("device", DEVICES)
def test_arch_query_ops(device) -> None:
    results = {
        name: getattr(torch.ops._xpu_C, name)(device)
        for name in ARCH_OPS
    }
    for name, value in results.items():
        assert isinstance(value, bool), name
    # A device belongs to at most one of these product families.
    assert sum(results.values()) <= 1, results


def test_arch_query_default_device() -> None:
    if not DEVICES:
        pytest.skip("No XPU device available")
    assert torch.ops._xpu_C.is_cri() == torch.ops._xpu_C.is_cri(0)
    assert torch.ops._xpu_C.is_nvl_p() == torch.ops._xpu_C.is_nvl_p(0)
