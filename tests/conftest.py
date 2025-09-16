# SPDX-License-Identifier: Apache-2.0
import os


def pytest_generate_tests(metafunc):
    use_mini_pytest_profile = os.getenv("PYTEST_PROFILE", "") == "MINI"
    if not use_mini_pytest_profile:
        return

    module = metafunc.module

    func_pytest_params = getattr(module, "MINI_PYTEST_PARAMS", {})
    profile = func_pytest_params.get(metafunc.function.__name__, None)

    if not profile:
        profile = func_pytest_params.get('default', None)

    if not profile:
        return

    for param_name, values in profile.items():
        if param_name in metafunc.fixturenames:
            new_markers = []
            for mark in metafunc.definition.own_markers:
                if mark.name == "parametrize" and mark.args[0] != param_name:
                    new_markers.append(mark)
                metafunc.definition.own_markers = new_markers
            metafunc.parametrize(param_name, values)
