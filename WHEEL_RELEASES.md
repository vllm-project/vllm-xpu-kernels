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

**wheel-per-commit** publishes nightly wheels after successful push or daily
scheduled tests on `main`, using the tested commit. To build manually, run
**Actions > wheel-per-commit > Run workflow > main**; this bypasses the test wait.

Nightlies use unique `.dev` versions and `nightly-` prerelease tags without
changing the formal Latest release. Publishing refreshes the separate
`/whl/nightly/` index; other releases, including RCs, remain under `/whl/`.

```bash
pip install --upgrade --pre --no-deps --only-binary=vllm-xpu-kernels \
  --index-url https://vllm-project.github.io/vllm-xpu-kernels/whl/nightly/ \
  vllm-xpu-kernels
```

Pin the full development version with `==` for a specific build. After successful
publishing, nightly prereleases older than 30 days are deleted (Git tags remain);
formal releases and RCs are not removed.
