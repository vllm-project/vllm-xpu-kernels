# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest.mock import patch

from tools.envs import get_compiler_cache


class BuildCacheTest(unittest.TestCase):

    def select_cache(self, mode, available):
        environment = {} if mode is None else {"VLLM_COMPILER_CACHE": mode}
        with patch.dict(os.environ, environment, clear=True), patch(
                "tools.envs.which",
                side_effect=lambda name: name if name in available else None):
            return get_compiler_cache()

    def test_auto_preserves_cache_preference_and_fallback(self):
        for available, launcher in (
            ({"sccache", "ccache"}, "sccache"),
            ({"ccache"}, "ccache"),
            (set(), ""),
        ):
            with self.subTest(available=available):
                self.assertEqual(self.select_cache(None, available), launcher)

    def test_explicit_cache_overrides_auto_preference(self):
        for launcher in ("ccache", "sccache"):
            with self.subTest(launcher=launcher):
                self.assertEqual(
                    self.select_cache(launcher, {"sccache", "ccache"}),
                    launcher)

    def test_disabled_cache_does_not_probe_installed_tools(self):
        with patch.dict(os.environ, {"VLLM_COMPILER_CACHE": "none"}), patch(
                "tools.envs.which") as which:
            self.assertEqual(get_compiler_cache(), "")
            which.assert_not_called()

    def test_missing_requested_cache_fails(self):
        with self.assertRaisesRegex(RuntimeError, "not on PATH"):
            self.select_cache("ccache", {"sccache"})

    def test_invalid_cache_fails(self):
        with self.assertRaisesRegex(ValueError, "VLLM_COMPILER_CACHE must be"):
            self.select_cache("invalid", {"ccache"})


if __name__ == "__main__":
    unittest.main()
