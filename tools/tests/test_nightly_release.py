# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from packaging.version import Version

from tools.nightly_release import (build_metadata, expired_nightlies,
                                   prune_nightlies, validate_wheels)

NOW = datetime(2026, 10, 29, 3, 0, 0, tzinfo=timezone.utc)
COMMIT = "abcdef1234567890" * 2 + "12345678"


class NightlyReleaseTest(unittest.TestCase):

    def test_version_bumps_final_tag_and_is_unique_on_reruns(self):
        first = build_metadata("v0.1.16.1", COMMIT, 42, 1, NOW)
        rerun = build_metadata("v0.1.16.1", COMMIT, 42, 2, NOW)
        self.assertEqual(first["version"],
                         "0.1.16.2.dev20261029030000+gabcdef1.r42.a1")
        self.assertGreater(Version(first["version"]), Version("0.1.16.1"))
        self.assertLess(Version(first["version"]), Version("0.1.16.2"))
        self.assertNotEqual(first["version"], rerun["version"])
        self.assertNotEqual(first["release_tag"], rerun["release_tag"])
        self.assertEqual(first["commit"], COMMIT)

    def test_prerelease_tag_uses_same_base_version_and_utc(self):
        local_time = NOW.astimezone(timezone(timedelta(hours=8)))
        for tag in ("v0.1.17rc1", "v0.1.17.dev2"):
            metadata = build_metadata(tag, COMMIT, 42, 1, local_time)
            self.assertTrue(metadata["version"].startswith(
                "0.1.17.dev20261029030000+"))

    def test_retention_only_deletes_expired_published_nightlies(self):
        old = "2026-09-28T03:00:00Z"
        releases = [
            {"id": 1, "tag_name": "nightly-old", "prerelease": True,
             "published_at": old},
            {"id": 2, "tag_name": "v0.1.16", "published_at": old},
            {"id": 3, "tag_name": "v0.1.17rc1", "prerelease": True,
             "published_at": old},
            {"id": 4, "tag_name": "nightly-draft", "prerelease": True,
             "draft": True, "published_at": old},
            {"id": 5, "tag_name": "nightly-boundary", "prerelease": True,
             "published_at": "2026-09-29T03:00:00Z"},
            {"id": 6, "tag_name": "nightly-not-a-prerelease",
             "prerelease": False, "published_at": old},
        ]
        self.assertEqual(expired_nightlies(releases, NOW, 30), [releases[0]])
        with patch("tools.nightly_release.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                [], 0, stdout=json.dumps([releases[:3], releases[3:]]))
            prune_nightlies("owner/repo", NOW, 30)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args.args[0],
                             ["gh", "api", "--method", "DELETE",
                              "repos/owner/repo/releases/1"])
        with self.assertRaisesRegex(ValueError, "must be positive"):
            expired_nightlies(releases, NOW, 0)

    def test_wheel_filename_and_metadata_must_match_nightly_version(self):
        version = build_metadata("v0.1.16", COMMIT, 42, 1, NOW)["version"]
        with tempfile.TemporaryDirectory() as temp:
            dist = Path(temp)
            wheel = dist / (f"vllm_xpu_kernels-{version}"
                            "-cp39-abi3-linux_x86_64.whl")
            with ZipFile(wheel, "w") as archive:
                archive.writestr(
                    f"vllm_xpu_kernels-{version}.dist-info/METADATA",
                    f"Name: vllm-xpu-kernels\nVersion: {version}\n")
            validate_wheels(dist, version)
            with ZipFile(wheel, "w") as archive:
                archive.writestr(
                    "vllm_xpu_kernels-0.1.16.dist-info/METADATA",
                    "Name: vllm-xpu-kernels\nVersion: 0.1.16\n")
            with self.assertRaisesRegex(ValueError, "wheel metadata"):
                validate_wheels(dist, version)
            with self.assertRaisesRegex(ValueError, "Unexpected nightly"):
                validate_wheels(dist, "0.1.17.dev1")
        with self.assertRaisesRegex(ValueError, "development version"):
            validate_wheels(Path("."), "0.1.16")

    def test_missing_wheel_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(
                ValueError, "No nightly wheels"):
            validate_wheels(Path(temp), "0.1.17.dev1")


if __name__ == "__main__":
    unittest.main()
