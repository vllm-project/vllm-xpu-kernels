# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from setuptools import Distribution


class BuildCacheTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        setup_path = Path(__file__).resolve().parents[2] / "setup.py"
        spec = importlib.util.spec_from_file_location("build_setup", setup_path)
        cls.build_setup = importlib.util.module_from_spec(spec)
        with patch("setuptools.setup"), patch(
                "setuptools_scm.get_version", return_value="0.0.0"), patch.dict(
                    os.environ, {
                        "VLLM_USE_PRECOMPILED": "0",
                        "VLLM_PRECOMPILED_WHEEL_LOCATION": "",
                    }):
            spec.loader.exec_module(cls.build_setup)

    def configure(self, mode, available):
        module = self.build_setup
        environment = {} if mode is None else {"VLLM_COMPILER_CACHE": mode}
        with tempfile.TemporaryDirectory() as temp, patch.dict(
                os.environ, environment), patch.object(
                    module, "which",
                    side_effect=lambda name: name if name in available else
                    None), patch.object(
                        module.cmake_build_ext, "did_config", {}), patch.object(
                            module.cmake_build_ext, "compute_num_jobs",
                            return_value=1), patch.object(
                                module.subprocess, "check_call") as call:
            command = module.cmake_build_ext(Distribution())
            command.build_temp = temp
            command.configure(module.CMakeExtension("test", temp))
        return [
            arg for arg in call.call_args.args[0]
            if arg.startswith(("-DCMAKE_C_COMPILER_LAUNCHER=",
                               "-DCMAKE_CXX_COMPILER_LAUNCHER="))
        ]

    def assert_launcher(self, mode, available, launcher):
        self.assertEqual(self.configure(mode, available), [
            f"-DCMAKE_C_COMPILER_LAUNCHER={launcher}",
            f"-DCMAKE_CXX_COMPILER_LAUNCHER={launcher}",
        ])

    def test_auto_preserves_cache_preference_and_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            for available, launcher in (
                ({"sccache", "ccache"}, "sccache"),
                ({"ccache"}, "ccache"),
                (set(), ""),
            ):
                with self.subTest(available=available):
                    self.assert_launcher(None, available, launcher)

    def test_explicit_cache_overrides_auto_preference(self):
        self.assert_launcher("ccache", {"sccache", "ccache"}, "ccache")
        self.assert_launcher("sccache", {"sccache", "ccache"}, "sccache")

    def test_disabled_cache_explicitly_clears_cmake_launchers(self):
        self.assert_launcher("none", {"sccache", "ccache"}, "")

    def test_missing_requested_cache_fails(self):
        with self.assertRaisesRegex(RuntimeError, "not on PATH"):
            self.configure("ccache", {"sccache"})

    def test_invalid_cache_fails(self):
        with self.assertRaisesRegex(ValueError, "VLLM_COMPILER_CACHE must be"):
            self.configure("invalid", {"ccache"})


if __name__ == "__main__":
    unittest.main()
