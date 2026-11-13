# SPDX-License-Identifier: Apache-2.0
"""The mono decode op contracts that need no layer: availability, the device
checks of the interface, an unregistered layer."""

import pytest
import torch

from tests.mono_decode.mono_utils import requires_mono, rows
from vllm_xpu_kernels import mono_decode_interface as mdi


@requires_mono
def test_available():
    assert mdi.is_available()
    assert mdi.supports()
    assert mdi.supports(torch.device("xpu", 0))
    assert not mdi.supports(torch.device("cpu"))
    assert mdi.max_context() >= 32768
    words = mdi.health(0)
    assert words.dtype == torch.int32 and words.shape == (8, )
    assert words.device == torch.device("xpu", 0)


def test_supports_needs_32_xe_cores():
    # the layer kernels keep 32 work-groups resident, one per Xe core
    if not (mdi.is_available() and torch.xpu.is_available()):
        pytest.skip("needs the mono decode ops and an XPU")
    for i in range(torch.xpu.device_count()):
        if torch.xpu.get_device_properties(i).gpu_subslice_count < 32:
            assert not mdi.supports(torch.device("xpu", i))


@requires_mono
def test_rejects_host_tensors():
    with pytest.raises(RuntimeError, match="must be an XPU tensor"):
        torch.ops._xpu_C.mono_register_layer(900, False, 0, 0,
                                             [torch.zeros(4)])


@requires_mono
def test_unregistered_layer():
    hs, rs = rows(1, torch.Generator().manual_seed(0), "xpu:0")
    m0 = torch.zeros(1, dtype=torch.int32, device="xpu:0")
    with pytest.raises(RuntimeError, match="not registered"):
        torch.ops._xpu_C.mono_decode_layer(901, hs, rs, torch.empty_like(hs),
                                           torch.empty_like(rs), m0, m0,
                                           None, None)
