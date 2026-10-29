# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

from tools.generate_wheel_index import generate_index

BASE_URL = (
    "https://github.com/vllm-project/vllm-xpu-kernels/releases/download/v1.0/"
)


def asset(name, digest=None):
    return {
        "name": name,
        "browser_download_url": BASE_URL + name.replace("+", "%2B"),
        "digest": digest,
    }


class LinkParser(HTMLParser):

    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.append(dict(attrs)["href"])


class WheelIndexTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "site"

    def links(self, relative_path):
        parser = LinkParser()
        parser.feed((self.output / relative_path).read_text(encoding="utf-8"))
        return parser.links

    def test_multiple_binaries_and_hashes(self):
        names = [
            "vllm_xpu_kernels-1.0+torch214-cp39-abi3-linux_x86_64.whl",
            "vllm_xpu_kernels-1.0+torch214-cp312-cp312-linux_x86_64.whl",
            "vllm_xpu_kernels-1.0+torch214-cp39-abi3-linux_aarch64.whl",
        ]
        digest = "a" * 64
        releases = [{
            "assets": [asset(name, f"sha256:{digest}") for name in names]
        }]
        self.assertEqual(generate_index(releases, self.output), 3)
        self.assertEqual(
            set(self.links("whl/vllm-xpu-kernels/index.html")),
            {BASE_URL + name.replace("+", "%2B") + f"#sha256={digest}"
             for name in names},
        )
        self.assertEqual(self.links("whl/index.html"), ["vllm-xpu-kernels/"])
        self.assertEqual(self.links("index.html"), ["whl/", "whl/nightly/"])

    def test_drafts_and_non_package_assets_are_excluded(self):
        wheel = "vllm_xpu_kernels-1.0-cp39-abi3-linux_x86_64.whl"
        releases = [
            {"draft": True, "assets": [asset(wheel)]},
            {"assets": [
                asset("other_package-1.0-py3-none-any.whl"),
                asset("vllm_xpu_kernels-not-a-wheel.whl"),
                asset("build_info.txt"),
            ]},
        ]
        self.assertEqual(generate_index(releases, self.output), 0)
        self.assertEqual(self.links("whl/vllm-xpu-kernels/index.html"), [])

    def test_prereleases_and_legacy_assets_without_hashes(self):
        name = "vllm_xpu_kernels-1.1rc1-cp39-abi3-linux_x86_64.whl"
        releases = [{"prerelease": True, "assets": [asset(name)]}]
        generate_index(releases, self.output)
        self.assertEqual(self.links("whl/vllm-xpu-kernels/index.html"),
                         [BASE_URL + name])

    def test_nightly_channel_is_isolated_from_releases_and_rcs(self):
        stable = asset("vllm_xpu_kernels-1.0-cp39-abi3-linux_x86_64.whl")
        rc = asset("vllm_xpu_kernels-1.1rc1-cp39-abi3-linux_x86_64.whl")
        nightly = asset(
            "vllm_xpu_kernels-1.1.dev20261029030000+gabcdef1"
            "-cp39-abi3-linux_x86_64.whl", "sha256:" + "b" * 64)
        releases = [
            {"tag_name": "v1.0", "assets": [stable]},
            {"tag_name": "v1.1rc1", "prerelease": True, "assets": [rc]},
            {"tag_name": "nightly-20261029-abcdef1", "prerelease": True,
             "assets": [nightly]},
            {"tag_name": "nightly-draft", "draft": True, "assets": [stable]},
        ]
        self.assertEqual(generate_index(releases, self.output), 3)
        self.assertEqual(
            set(self.links("whl/vllm-xpu-kernels/index.html")),
            {stable["browser_download_url"], rc["browser_download_url"]})
        self.assertEqual(
            self.links("whl/nightly/vllm-xpu-kernels/index.html"),
            [nightly["browser_download_url"] + "#sha256=" + "b" * 64])
        self.assertEqual(self.links("whl/nightly/index.html"),
                         ["vllm-xpu-kernels/"])

    def test_invalid_digest_is_rejected(self):
        name = "vllm_xpu_kernels-1.0-cp39-abi3-linux_x86_64.whl"
        with self.assertRaisesRegex(ValueError, "Invalid SHA-256"):
            generate_index([{"assets": [asset(name, "sha256:invalid")]}],
                           self.output)

    def test_paginated_cli_and_html_escaping(self):
        name = "vllm_xpu_kernels-1.0-cp39-abi3-linux_x86_64.whl"
        wheel = asset(name)
        wheel["browser_download_url"] += "?a=1&b=2"
        releases_json = Path(self.temp.name) / "releases.json"
        releases_json.write_text(json.dumps([[{"assets": []}],
                                             [{"assets": [wheel]}]]),
                                 encoding="utf-8")
        script = (Path(__file__).resolve().parents[1]
                  / "generate_wheel_index.py")
        subprocess.run(
            [sys.executable, str(script), "--releases-json",
             str(releases_json), "--output", str(self.output)],
            check=True,
            capture_output=True,
        )
        self.assertEqual(self.links("whl/vllm-xpu-kernels/index.html"),
                         [wheel["browser_download_url"]])
        html = (self.output / "whl/vllm-xpu-kernels/index.html").read_text()
        self.assertIn("?a=1&amp;b=2", html)


if __name__ == "__main__":
    unittest.main()
