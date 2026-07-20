# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.machinery
from pathlib import Path


def load_native_extension_alias(package: str, target_name: str) -> None:
    package_dir = Path(__file__).resolve().parent
    has_native_target = any(
        (package_dir / f"{target_name}{suffix}").exists()
        for suffix in importlib.machinery.EXTENSION_SUFFIXES)
    if not has_native_target:
        raise ModuleNotFoundError(
            f"No native extension module {package}.{target_name} was found")

    importlib.import_module(f".{target_name}", package)
