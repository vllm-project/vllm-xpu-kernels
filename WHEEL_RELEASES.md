# Publishing wheels with GitHub Releases and Pages

GitHub Releases stores wheel binaries; GitHub Pages hosts a pip-compatible
index linking to those assets. Wheels are not committed to Git or hosted on
Pages. Existing PyPI publishing is unchanged.

## Publish release wheels

Build and validate the wheels using the existing native build process, then
collect them in `release-dist/`. From this checkout, replace `vX.Y.Z` with an
existing version tag matching the wheel version:

```bash
gh release create vX.Y.Z release-dist/*.whl \
  --verify-tag --draft --generate-notes

# Review the draft and assets before publishing.
gh release edit vX.Y.Z --draft=false
```

One release can contain multiple wheels with distinct filenames. Do not rename
wheels to claim a different version or binary compatibility.

Publication refreshes the index automatically. To add wheels to an existing
release and refresh immediately:

```bash
gh release upload vX.Y.Z release-dist/*.whl
gh workflow run wheel-index.yaml --ref main
```

Asset uploads alone do not trigger deployment. A six-hour scheduled refresh
also keeps the index up to date.

## Install with pip

Install compatible PyTorch XPU and runtime dependencies first; `--no-deps`
below skips dependency installation. Replace the example version as needed:

```bash
pip install --only-binary=vllm-xpu-kernels \
  --index-url https://vllm-project.github.io/vllm-xpu-kernels/whl/ \
  --no-deps "vllm-xpu-kernels==0.1.16.1"
```

## Nightly wheels

**wheel-per-commit** builds development wheels after successful push or daily
scheduled tests on `main`, using the tested commit. Push builds only upload an
Actions artifact, retained for 30 days. Daily scheduled tests start at 19:00 UTC
(03:00 UTC+8); after they succeed, the wheel is built and published to the
nightly channel. Scheduled runs can publish the same commit again even when
`main` has not changed.

To build manually, run **Actions > wheel-per-commit > Run workflow > main**.
Enable **Publish a nightly prerelease after building** to also publish it;
otherwise, the build only uploads an artifact. Manual builds bypass the test
wait. To explicitly publish from the command line:

```bash
gh workflow run wheel-per-commit.yaml \
  --repo vllm-project/vllm-xpu-kernels --ref main \
  -F publish_nightly=true
```

Manually running the unit-test workflow does not trigger a wheel build or
publication.

Nightlies follow FlashInfer's UTC date-based naming convention:

- Wheel version: `0.1.17.dev20261102`
- Release tag: `nightly-v0.1.17-20261102`
- Release title: `Nightly Release v0.1.17-20261102`

The base version still comes from the nearest version tag: final tags increment
the last version component, while prerelease tags retain their release version.
The source commit remains in release notes, and `build_info.txt` records both
the commit and build timestamp. Re-publishing the same base version on the same
UTC date replaces the existing nightly release and tag with the new build; it
does not create another daily release. Because the wheel version is unchanged,
clients may need to force reinstall without their cached wheel to pick up a
same-day replacement.
Artifact-only per-commit builds retain unique timestamp, commit, run ID and
attempt suffixes.

Once a daily nightly is available for a base version, the nightly index omits
legacy long-timestamp builds of that same base version so pip does not prefer
them over the new daily versions. Their release assets remain available until
normal retention cleanup.

Nightlies do not change the formal Latest release. Publishing refreshes the
separate `/whl/nightly/` index; other releases, including RCs, remain under
`/whl/`.

```bash
pip install --upgrade --pre --no-deps --only-binary=vllm-xpu-kernels \
  --index-url https://vllm-project.github.io/vllm-xpu-kernels/whl/nightly/ \
  vllm-xpu-kernels
```

Pin the full development version with `==` for a specific build. After successful
publishing, nightly prereleases older than 30 days are deleted (Git tags remain);
formal releases and RCs are not removed.
