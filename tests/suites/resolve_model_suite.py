# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Iterable

from .kernel_test_map import KERNEL_TO_TESTS
from .model_kernel_map import MODEL_TO_KERNELS


def _split_csv(values: str | None) -> list[str]:
    if not values:
        return []
    return [item.strip() for item in values.split(",") if item.strip()]


def _ordered_unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def resolve_suite(
    models: list[str],
    profiles: list[str],
    extra_kernels: list[str],
) -> tuple[list[str], list[str]]:
    resolved_kernels: list[str] = []

    unknown_models = [
        model for model in models if model not in MODEL_TO_KERNELS
    ]
    if unknown_models:
        raise ValueError(
            f"Unknown model(s): {', '.join(unknown_models)}. "
            f"Known models: {', '.join(sorted(MODEL_TO_KERNELS))}")

    missing_profiles: list[str] = []
    for model in models:
        profile_map = MODEL_TO_KERNELS[model]
        for profile in profiles:
            kernels = profile_map.get(profile)
            if kernels is None:
                missing_profiles.append(f"{model}:{profile}")
                continue
            resolved_kernels.extend(kernels)

    if missing_profiles:
        raise ValueError("Missing profile mapping(s): " +
                         ", ".join(missing_profiles) +
                         ". Add them to tests/suites/model_kernel_map.py")

    resolved_kernels.extend(extra_kernels)
    resolved_kernels = _ordered_unique(resolved_kernels)

    missing_kernel_tests = [
        kernel for kernel in resolved_kernels if kernel not in KERNEL_TO_TESTS
    ]
    if missing_kernel_tests:
        raise ValueError("Kernel(s) missing test mapping: " +
                         ", ".join(missing_kernel_tests) +
                         ". Add them to tests/suites/kernel_test_map.py")

    tests: list[str] = []
    for kernel in resolved_kernels:
        tests.extend(KERNEL_TO_TESTS[kernel])

    return resolved_kernels, _ordered_unique(tests)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(
        "Resolve a model-aware kernel UT suite into pytest node ids / paths."))
    parser.add_argument(
        "--models",
        help="Comma-separated model names, e.g. llama,qwen2 or deepseek_v3",
    )
    parser.add_argument(
        "--profiles",
        default="default",
        help="Comma-separated additive profiles, default: default",
    )
    parser.add_argument(
        "--kernels",
        help="Optional extra comma-separated kernels to include manually",
    )
    parser.add_argument(
        "--format",
        choices=("lines", "args", "json"),
        default="lines",
        help="Output format for resolved tests",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List known model names and exit",
    )
    parser.add_argument(
        "--list-kernels",
        action="store_true",
        help="List known kernel names and exit",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_models:
        for model in sorted(MODEL_TO_KERNELS):
            print(model)
        return 0

    if args.list_kernels:
        for kernel in sorted(KERNEL_TO_TESTS):
            print(kernel)
        return 0

    models = _split_csv(args.models)
    if not models:
        parser.error(
            "--models is required unless --list-models/--list-kernels is used")

    profiles = _split_csv(args.profiles) or ["default"]
    extra_kernels = _split_csv(args.kernels)

    try:
        kernels, tests = resolve_suite(models, profiles, extra_kernels)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(
            json.dumps(
                {
                    "models": models,
                    "profiles": profiles,
                    "kernels": kernels,
                    "tests": tests,
                },
                indent=2,
            ))
        return 0

    if args.format == "args":
        print(" ".join(shlex.quote(test) for test in tests))
        return 0

    for test in tests:
        print(test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
