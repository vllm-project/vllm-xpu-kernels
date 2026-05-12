# Test Scope Design Document

## Overview

This document describes the multi-scope test system for `vllm-xpu-kernels`. The system supports four test scopes, controlled by the environment variable `XPU_KERNEL_TEST_SCOPE`, to balance CI speed vs. coverage across different scenarios.

## Test Scopes

| Scope | Env Value | Description | Use Case |
|-------|-----------|-------------|----------|
| **full** | `full` (or unset) | All tests × all parameters | Nightly CI, pre-release validation |
| **ci** | `ci` | All tests × reduced parameters | PR CI, push-to-main CI |
| **mini** | `mini` | Subset of tests × minimal parameters | Simulator, quick smoke test |
| **on-demand** | `ondemand:<profile>` | Profile-specific tests × model-specific shapes | Model-targeted validation (e.g. `ondemand:llama`) |

## Environment Variable

```bash
# Full scope (default) — nightly
export XPU_KERNEL_TEST_SCOPE=full
# or simply unset it

# CI scope — PR CI
export XPU_KERNEL_TEST_SCOPE=ci

# Mini scope — simulator / quick validation
export XPU_KERNEL_TEST_SCOPE=mini

# On-demand scope — model-specific
export XPU_KERNEL_TEST_SCOPE=ondemand:llama
export XPU_KERNEL_TEST_SCOPE=ondemand:deepseek
```

## Architecture

### Per-Test-Module Configuration

Each test module defines a `TEST_SCOPE_PARAMS` dictionary with scope-specific parameter overrides:

```python
TEST_SCOPE_PARAMS = {
    "ci": {                          # CI scope: reduced shapes
        "default": {                 # applies to all functions unless overridden
            "num_tokens": [1, 128],
            "hidden_size": [64],
        },
        "test_specific_fn": {        # per-function override
            "num_tokens": [1],
        },
    },
    "mini": {                        # Mini scope: minimal shapes
        "default": {
            "num_tokens": [1],
            "hidden_size": [32],
        },
    },
}
```

### On-Demand Profiles

On-demand profiles are defined centrally in `tests/test_scope_profiles.py`. Each profile maps test functions to model-relevant parameter sets:

```python
ONDEMAND_PROFILES = {
    "llama3": {
        "tests/test_activation.py": {
            "test_act_and_mul": {
                "num_tokens": [1, 128],
                "d": [11008],
                "activation": ["silu_and_mul"],
            },
        },
        ...
    },
    "deepseek": { ... },
}
```

### conftest.py Hook

The `pytest_generate_tests` hook in `conftest.py`:

1. Reads `XPU_KERNEL_TEST_SCOPE` env var
2. For `full` scope: no modification (run everything)
3. For `ci` / `mini` scope: looks up `TEST_SCOPE_PARAMS[scope]` in the test module
4. For `ondemand:<profile>` scope: looks up the profile in `test_scope_profiles.py`
5. Replaces `@pytest.mark.parametrize` values with the scoped subset
6. For `mini` scope: also skips entire tests if marked with `skip_for_mini=True`

### Test Skipping for Mini Scope

Tests can be skipped entirely in mini scope using:

```python
SKIP_IN_MINI_SCOPE = True  # Module-level flag to skip all tests in this module
```

Or per-function via `TEST_SCOPE_PARAMS`:

```python
TEST_SCOPE_PARAMS = {
    "mini": {
        "test_expensive_fn": None,   # None means skip this test in mini scope
    },
}
```

## Backward Compatibility

- The old `XPU_KERNEL_PYTEST_PROFILER=MINI` environment variable is still supported and maps to `XPU_KERNEL_TEST_SCOPE=mini`.
- The old `MINI_PYTEST_PARAMS` dictionary is still consumed as a fallback for `ci` and `mini` scopes when `TEST_SCOPE_PARAMS` is not defined.

## CI Workflow Integration

```yaml
# PR CI (ci scope)
- name: test
  run: |
    XPU_KERNEL_TEST_SCOPE=ci pytest -v -s tests/

# Nightly CI (full scope)
- name: test
  run: |
    pytest -v -s tests/
```

## Migration Guide

For existing test files, the `MINI_PYTEST_PARAMS` still works. To adopt the new system:

1. Rename `MINI_PYTEST_PARAMS` → embedded into `TEST_SCOPE_PARAMS["ci"]` (and optionally `["mini"]`)
2. Add more granular scope definitions as needed
3. The old `SKIP_TEST_FOR_MINI_SCOPE` pattern is replaced by `SKIP_IN_MINI_SCOPE = True`
