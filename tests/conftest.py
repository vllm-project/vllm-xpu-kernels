# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from tests.utils import create_kv_caches_with_random


def pytest_generate_tests(metafunc):
    use_mini_pytest_profiler = os.getenv("XPU_KERNEL_PYTEST_PROFILER",
                                         "") == "MINI"
    if not use_mini_pytest_profiler:
        return

    module = metafunc.module

    func_pytest_params = getattr(module, "MINI_PYTEST_PARAMS", {})
    profile = func_pytest_params.get(metafunc.function.__name__, None)

    if not profile:
        profile = func_pytest_params.get('default', None)

    if not profile:
        return

    for param_name, values in profile.items():
        split_names = [name.strip() for name in param_name.split(",")]
        if all(name in metafunc.fixturenames for name in split_names):
            new_markers = []
            for mark in metafunc.definition.own_markers:
                if mark.name == "parametrize" and mark.args[0] != param_name:
                    new_markers.append(mark)
                metafunc.definition.own_markers = new_markers
            metafunc.parametrize(param_name, values)


@pytest.fixture
def reset_default_device():
    """
    Some tests, such as `test_punica_ops.py`, explicitly set the
    default device, which can affect subsequent tests. Adding this fixture
    helps avoid this problem.
    """
    import torch
    original_device = torch.get_default_device()
    yield
    torch.set_default_device(original_device)


@pytest.fixture()
def kv_cache_factory():
    return create_kv_caches_with_random
