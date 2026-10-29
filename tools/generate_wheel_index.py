# SPDX-License-Identifier: Apache-2.0
"""Generate a static Simple Repository index from GitHub Releases JSON."""

import argparse
import json
from html import escape
from pathlib import Path

from packaging.utils import (InvalidWheelFilename, canonicalize_name,
                             parse_wheel_filename)

PACKAGE = "vllm-xpu-kernels"


def write_page(path, title, links):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(links)
    path.write_text(
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head><meta charset="utf-8">\n'
        f"<title>{escape(title)}</title></head>\n"
        f"<body><h1>{escape(title)}</h1>\n{body}\n</body></html>\n",
        encoding="utf-8",
    )


def generate_index(releases, output):
    channels = {"": {}, "nightly/": {}}
    for release in releases:
        if release.get("draft"):
            continue
        channel = ("nightly/" if release.get("tag_name", "").startswith(
            "nightly-") else "")
        wheels = channels[channel]
        for asset in release.get("assets", []):
            name = asset["name"]
            try:
                distribution, version, _, _ = parse_wheel_filename(name)
            except InvalidWheelFilename:
                continue
            if canonicalize_name(distribution) != PACKAGE:
                continue
            url = asset["browser_download_url"]
            digest = asset.get("digest")
            if digest:
                algorithm, _, value = digest.partition(":")
                if (algorithm != "sha256" or len(value) != 64
                        or any(c not in "0123456789abcdef" for c in value)):
                    raise ValueError(f"Invalid SHA-256 digest for {name}")
                url += f"#sha256={value}"
            wheels[url] = (version, name)

    for channel, wheels in channels.items():
        links = [
            f'<a href="{escape(url, quote=True)}">{escape(name)}</a><br>'
            for url, (_, name) in sorted(
                wheels.items(), key=lambda item: (*item[1], item[0]))
        ]
        write_page(output / "whl" / channel / PACKAGE / "index.html",
                   f"Links for {PACKAGE} ({channel.rstrip('/') or 'release'})",
                   links)
    write_page(output / "whl" / "nightly" / "index.html", "Nightly wheel index",
               [f'<a href="{PACKAGE}/">{PACKAGE}</a><br>'])
    write_page(output / "whl" / "index.html", "Wheel index",
               [f'<a href="{PACKAGE}/">{PACKAGE}</a><br>'])
    write_page(output / "index.html", "vllm-xpu-kernels wheels",
               ['<a href="whl/">Wheel index</a>',
                '<a href="whl/nightly/">Nightly wheel index</a>'])
    (output / ".nojekyll").touch()
    return sum(len(wheels) for wheels in channels.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--releases-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    releases = json.loads(args.releases_json.read_text(encoding="utf-8"))
    # gh api --paginate --slurp returns an array of response pages.
    if releases and isinstance(releases[0], list):
        releases = [release for page in releases for release in page]
    count = generate_index(releases, args.output)
    print(f"Indexed {count} wheels in {args.output / 'whl'}")


if __name__ == "__main__":
    main()
