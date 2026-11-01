# SPDX-License-Identifier: Apache-2.0
"""Prepare nightly wheel metadata and prune expired nightly releases."""

import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


def build_metadata(version_tag, commit, run_id, run_attempt, now):
    version = Version(version_tag)
    release = list(version.release)
    if not version.is_prerelease:
        release[-1] += 1
    base_version = ".".join(str(part) for part in release)
    timestamp = now.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")
    short_sha = commit[:7]
    return {
        "commit": commit,
        "short_sha": short_sha,
        "build_date": timestamp,
        "version": (f"{base_version}.dev{timestamp}"
                    f"+g{short_sha}.r{run_id}.a{run_attempt}"),
        "release_tag": (f"nightly-{timestamp}-{short_sha}"
                        f"-{run_id}-{run_attempt}"),
    }


def validate_wheels(dist, expected_version):
    expected_version = Version(expected_version)
    if not expected_version.is_devrelease:
        raise ValueError("Nightly wheels must have a development version")
    wheels = sorted(dist.glob("*.whl"))
    if not wheels:
        raise ValueError("No nightly wheels found")
    for wheel in wheels:
        name, version, _, _ = parse_wheel_filename(wheel.name)
        if name != "vllm-xpu-kernels" or version != expected_version:
            raise ValueError(f"Unexpected nightly wheel: {wheel.name}")
        with ZipFile(wheel) as archive:
            metadata_files = [
                name for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_files) != 1:
                raise ValueError(f"Expected one METADATA file in {wheel.name}")
            metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
        if (canonicalize_name(metadata["Name"]) != "vllm-xpu-kernels"
                or Version(metadata["Version"]) != expected_version):
            raise ValueError(f"Unexpected wheel metadata: {wheel.name}")


def expired_nightlies(releases, now, retention_days):
    if retention_days < 1:
        raise ValueError("retention_days must be positive")
    cutoff = now - timedelta(days=retention_days)
    return [
        release for release in releases
        if release.get("tag_name", "").startswith("nightly-")
        and release.get("prerelease") and not release.get("draft")
        and release.get("published_at")
        and datetime.fromisoformat(
            release["published_at"].replace("Z", "+00:00")) < cutoff
    ]


def prune_nightlies(repository, now, retention_days):
    result = subprocess.run(
        ["gh", "api", "--paginate", "--slurp",
         f"repos/{repository}/releases?per_page=100"],
        check=True, capture_output=True, text=True)
    releases = [release for page in json.loads(result.stdout)
                for release in page]
    for release in expired_nightlies(releases, now, retention_days):
        subprocess.run(
            ["gh", "api", "--method", "DELETE",
             f"repos/{repository}/releases/{release['id']}"],
            check=True)
        print(f"Deleted expired nightly release: {release['tag_name']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    metadata = commands.add_parser("metadata")
    metadata.add_argument("--version-tag", required=True)
    metadata.add_argument("--commit", required=True)
    metadata.add_argument("--run-id", type=int, required=True)
    metadata.add_argument("--run-attempt", type=int, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--dist", type=Path, required=True)
    validate.add_argument("--version", required=True)
    prune = commands.add_parser("prune")
    prune.add_argument("--repository", required=True)
    prune.add_argument("--retention-days", type=int, default=30)
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    if args.command == "metadata":
        for key, value in build_metadata(
                args.version_tag, args.commit, args.run_id, args.run_attempt,
                now).items():
            print(f"{key}={value}")
    elif args.command == "validate":
        validate_wheels(args.dist, args.version)
    else:
        prune_nightlies(args.repository, now, args.retention_days)


if __name__ == "__main__":
    main()
