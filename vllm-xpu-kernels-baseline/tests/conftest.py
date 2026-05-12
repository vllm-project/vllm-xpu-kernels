# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from tests.utils import create_kv_caches_with_random


def _get_test_scope():
    """Determine the active test scope from environment variables.

    Priority:
      1. XPU_KERNEL_TEST_SCOPE (new unified env var)
      2. XPU_KERNEL_PYTEST_PROFILER=MINI (legacy, maps to "mini")
      3. Default: "full"

    Returns one of: "full", "ci", "mini", or "ondemand:<profile_name>"
    """
    scope = os.getenv("XPU_KERNEL_TEST_SCOPE", "").strip().lower()
    if scope:
        return scope

    # Legacy compatibility
    if os.getenv("XPU_KERNEL_PYTEST_PROFILER", "").strip().upper() == "MINI":
        return "mini"

    return "full"


def _resolve_scope_params(module, func_name, scope):
    """Resolve parameter overrides for the given scope.

    Lookup order:
    1. ondemand: profile-defined function entry or default entry
    2. mini: MINI_PYTEST_PARAMS[func_name]
    3. mini: MINI_PYTEST_PARAMS["default"]
    4. ci/full: use original test parameters

    Returns:
      - dict of {param_name: values} to override, or
      - None to skip this test entirely (when the entry is explicitly None)
      - empty dict {} to run with original params (no override)
    """
    if scope.startswith("ondemand:"):
        # On-demand profiles are loaded from tests/test_scope_profiles.py
        # Support both "ondemand:llama3" and "ondemand::llama3"
        profile_name = scope.split(":", 1)[1].lstrip(":")
        try:
            from tests.test_scope_profiles import ONDEMAND_PROFILES
        except ImportError:
            return {}
        profile = ONDEMAND_PROFILES.get(profile_name, {})
        # Match by module file path (relative)
        module_file = getattr(module, "__file__", "")
        for path_key, funcs in profile.items():
            if module_file.endswith(path_key):
                entry = funcs.get(func_name, funcs.get("default", None))
                return entry  # may be None (skip) or dict
        return None  # module not in profile → skip

    scope_key = scope  # "ci" or "mini"

    # ci scope always uses the original/default parametrization.
    if scope_key == "ci":
        return {}

    # mini scope always uses the current file's MINI_PYTEST_PARAMS.
    if scope_key == "mini":
        legacy = getattr(module, "MINI_PYTEST_PARAMS", {})
        entry = legacy.get(func_name)
        if entry is not None:
            return entry
        entry = legacy.get("default")
        if entry is not None:
            return entry

    return {}


def _apply_param_overrides(metafunc, profile):
    """Replace @pytest.mark.parametrize values in markers (not via 
    metafunc.parametrize).

    Instead of calling metafunc.parametrize() directly (which conflicts with
    pytest's built-in marker processing), we replace the marker objects with
    new ones carrying the overridden values. The built-in pytest_generate_tests
    hook then processes all markers uniformly.
    """
    new_markers = []
    for mark in metafunc.definition.own_markers:
        if (mark.name == "parametrize" and mark.args[0] in profile):
            param_name = mark.args[0]
            split_names = [n.strip() for n in param_name.split(",")]
            if all(n in metafunc.fixturenames for n in split_names):
                # Replace with a new marker carrying override values
                new_mark = pytest.mark.parametrize(param_name,
                                                   profile[param_name])
                new_markers.append(new_mark.mark)
                continue
        new_markers.append(mark)
    metafunc.definition.own_markers = new_markers


def _skip_test(metafunc, reason):
    """Skip a test by collapsing parametrize markers to single values and adding
    skip.

    We cannot use ``pytest.skip()`` inside ``pytest_generate_tests`` because the
    resulting ``Skipped`` exception propagates up through ``_genfunctions`` and
    aborts the entire **module** collection, discarding items already generated
    for other test functions in the same file.

    Instead we:
      1. Replace each ``@pytest.mark.parametrize`` marker with a single-value
         version so pytest still maps the parameter names to fixtures (avoiding
         "fixture not found" errors).
      2. Add a ``@pytest.mark.skip`` marker to the **Python function object**
         itself (``metafunc.function``).  Markers on
         ``metafunc.definition.own_markers`` do NOT propagate to the
         ``Function`` items that ``_genfunctions`` creates; those items read 
         markers from the function object's ``pytestmark`` attribute instead.
    """
    new_markers = []
    for m in metafunc.definition.own_markers:
        if m.name == "parametrize" and m.args:
            # Keep the marker but collapse to a single placeholder value
            param_name = m.args[0]
            original_values = m.args[1]
            single = [original_values[0]] if original_values else [None]
            new_markers.append(
                pytest.mark.parametrize(param_name, single).mark)
        else:
            new_markers.append(m)
    metafunc.definition.own_markers = new_markers

    # Add skip marker to the actual function object so the resulting
    # Function items inherit it via pytestmark.
    skip_mark = pytest.mark.skip(reason=reason).mark
    func = metafunc.function
    existing = list(getattr(func, "pytestmark", []))
    existing.append(skip_mark)
    func.pytestmark = existing


def pytest_generate_tests(metafunc):
    """Hook to apply test scope parameter overrides.

    Controlled by XPU_KERNEL_TEST_SCOPE env var:
      - "full"  (or unset): run all tests with all parameters (no override)
      - "ci":   run all tests with reduced parameter sets
      - "mini": run subset of tests with minimal parameters
      - "ondemand:<profile>": run model-specific tests and shapes

    See docs/test_scope_design.md for details.
    """
    scope = _get_test_scope()

    if scope == "full":
        return

    module = metafunc.module
    func_name = metafunc.function.__name__

    # --- Module-level skip for mini scope ---
    if scope == "mini" and getattr(module, "SKIP_IN_MINI_SCOPE", False):
        _skip_test(metafunc, "Skipped in mini scope (SKIP_IN_MINI_SCOPE=True)")
        return

    profile = _resolve_scope_params(module, func_name, scope)

    # None means explicitly skip this test
    if profile is None:
        _skip_test(metafunc, f"Skipped in {scope} scope")
        return

    # Empty dict means no override (run with original params)
    if not profile:
        return

    _apply_param_overrides(metafunc, profile)


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
