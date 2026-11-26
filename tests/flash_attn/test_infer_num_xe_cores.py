# SPDX-License-Identifier: Apache-2.0
"""Xe-core count used by the paged-decode split plan.

`_infer_num_xe_cores` feeds `build_decode_split_plan`; it must report the
same core count that flash_api.cpp::get_num_splits() reads through SYCL
(gpu_slices * gpu_subslices_per_slice). The mocked cases run without a GPU.
"""
from types import SimpleNamespace

import pytest
import torch

from vllm_xpu_kernels import flash_attn_interface as fai


def _mock_props(monkeypatch, props=None, exc=None):

    def fake_get_device_properties(device):
        if exc is not None:
            raise exc
        return props

    monkeypatch.setattr(torch.xpu, "get_device_properties",
                        fake_get_device_properties)


@pytest.mark.parametrize(
    "props, expected",
    [
        # What torch actually exposes: only the total subslice count.
        # 32-core Arc Pro B70.
        (SimpleNamespace(gpu_subslice_count=32), 32),
        # 20-core Arc B580 / Arc Pro B60.
        (SimpleNamespace(gpu_subslice_count=20), 20),
        # A slice/subslice pair, if a backend ever exposes one, still wins.
        (SimpleNamespace(gpu_slices=2,
                         gpu_subslices_per_slice=16,
                         gpu_subslice_count=8), 32),
        # Nothing usable: keep the historical default.
        (SimpleNamespace(), 20),
        (SimpleNamespace(gpu_subslice_count=0), 20),
    ],
)
def test_infer_num_xe_cores_from_props(monkeypatch, props, expected):
    _mock_props(monkeypatch, props=props)
    assert fai._infer_num_xe_cores(torch.device("xpu:0")) == expected


def test_infer_num_xe_cores_query_failure(monkeypatch):
    _mock_props(monkeypatch, exc=RuntimeError("no XPU"))
    assert fai._infer_num_xe_cores(torch.device("xpu:0")) == 20


def test_split_plan_uses_all_cores(monkeypatch):
    # One decode sequence, 4 KV heads, 64-token tiles on a 32-core part:
    # the plan should target ~2x oversubscription of 32 cores, not of 20.
    _mock_props(monkeypatch, props=SimpleNamespace(gpu_subslice_count=32))
    cores = fai._infer_num_xe_cores(torch.device("xpu:0"))
    splits, work = fai.build_decode_split_plan([16384],
                                               kv_tile=64,
                                               num_kv_splits=32,
                                               num_xe_cores=cores,
                                               num_heads_kv=4)
    assert int(splits[0]) == 16
    assert work.shape[0] == 16


@pytest.mark.skipif(not (hasattr(torch, "xpu") and torch.xpu.is_available()),
                    reason="needs an Intel GPU")
def test_infer_num_xe_cores_matches_device():
    for i in range(torch.xpu.device_count()):
        props = torch.xpu.get_device_properties(i)
        assert fai._infer_num_xe_cores(
            torch.device(f"xpu:{i}")) == props.gpu_subslice_count
