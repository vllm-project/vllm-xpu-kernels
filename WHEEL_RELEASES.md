# Publishing wheels with GitHub Releases and Pages

This repository supports a FlashInfer-style distribution layout:

- **GitHub Releases** stores the wheel binaries as release assets. One release
  can contain multiple wheels.
- **GitHub Pages** hosts a small, static
  [Simple Repository API](https://packaging.python.org/en/latest/specifications/simple-repository-api/)
  index whose download links point directly to those release assets.

No wheel binaries are committed to Git or copied to Pages. This is a separate
package index, not an upload to PyPI; existing PyPI publishing is unchanged.

## Enable GitHub Pages

1. Merge `.github/workflows/wheel-index.yaml`, the generator, and its tests
   into the repository's default branch.
2. Open **Settings > Pages > Build and deployment**, and select
   **GitHub Actions** as the source. No `gh-pages` branch is needed.
3. Ensure repository/organization Actions policy permits the pinned actions in
   the workflow. Deployment uses the built-in `GITHUB_TOKEN` with
   `pages: write` and `id-token: write`; no personal access token is needed.
4. Open **Settings > Environments > github-pages**. If deployment branches or
   tags are restricted, allow the default branch and the release tags
   (for example `v*`). Release-triggered runs retain a tag event ref even though
   the workflow checks out the generator from the default branch.
5. In **Actions > Publish wheel index**, select the default branch and click
   **Run workflow**. Approve deployment if the environment requires approval.
6. After deployment, open:

   ```text
   https://vllm-project.github.io/vllm-xpu-kernels/whl/
   https://vllm-project.github.io/vllm-xpu-kernels/whl/vllm-xpu-kernels/
   ```

These are the default project-site URLs, assuming the organization does not
already configure a custom Pages domain. The deployment's environment URL is
the source of truth for the site base URL. For a fork, replace the owner and
repository path with the fork's values.

Only public, non-draft releases should be used for distribution. Releases with
a `nightly-` tag prefix go to `/whl/nightly/`; all other releases, including RCs,
go to `/whl/`. pip normally selects stable versions unless `--pre` or an explicit
pre-release version is requested. Assets for other packages are ignored.
SHA-256 fragments are included when GitHub supplies an asset digest; older
assets without a digest remain installable without a hash fragment.

The workflow refreshes on release publication, editing, or deletion, and on
generator changes on `main`. Uploading, replacing, or deleting an asset on an
existing release does **not** emit a release event: run the workflow manually
afterward, or wait for the six-hour scheduled refresh. Scheduled workflows can
be delayed or disabled by GitHub; manual refresh remains available.

## Publish multiple wheels

Keep the existing native build process. The index workflow does not compile
kernels, choose a build matrix, or upload artifacts from per-commit builds.
Build and validate each supported binary with the appropriate XPU toolchain,
then collect the wheels in a dedicated directory such as `release-dist/`.

The release tag should match the package version, for example tag `vX.Y.Z`
for package version `X.Y.Z`. Publish using the release UI, or:

```bash
# Replace vX.Y.Z with an existing, reviewed version tag.
gh release create vX.Y.Z \
  --repo vllm-project/vllm-xpu-kernels \
  --verify-tag --draft --generate-notes

gh release upload vX.Y.Z release-dist/*.whl \
  --repo vllm-project/vllm-xpu-kernels

# Check all intended assets, then make the release public.
gh release view vX.Y.Z --repo vllm-project/vllm-xpu-kernels
gh release edit vX.Y.Z --draft=false \
  --repo vllm-project/vllm-xpu-kernels
```

Upload assets before publishing the draft so that the initial index refresh
sees the complete release. The release page lists all binaries, and the index
collects wheels across all published releases, including existing releases.
For an existing public release, upload the new wheels and manually refresh:

```bash
gh release upload vX.Y.Z release-dist/*.whl \
  --repo vllm-project/vllm-xpu-kernels
gh workflow run wheel-index.yaml --ref main \
  --repo vllm-project/vllm-xpu-kernels
```

Prefer a new package version over replacing published wheels: installers may
cache the old files. Each asset filename within a release must be unique.

### Binary compatibility and variants

Python, ABI, and platform tags distinguish wheels that an installer can
automatically select, for example `cp312-cp312` versus `cp39-abi3`, or
`manylinux_2_28_x86_64` versus `manylinux_2_28_aarch64`. Only publish combinations
that were actually built and validated. An `abi3` tag does not make the PyTorch
C++ ABI stable, and the package's `Requires-Python` still applies.

Do not rename a wheel to claim a different ABI, architecture, or manylinux
compatibility. Platform tags must describe the actual binary and its runtime
dependencies.

pip does **not** detect an installed Torch version, oneAPI version, GPU family,
or attention-kernel preset when choosing wheel tags. For multiple builds with
the same Python/platform tags, use distinct package versions and explicitly
pin the intended variant. For example, a future build convention could use
PEP 440 local versions such as `X.Y.Z+torch214.xpu20261`. Build with
`VLLM_VERSION_OVERRIDE` to set the version in both the filename and wheel
metadata; do not just rename the output. Consumers must then pin the complete
local version. A plain `==X.Y.Z` requirement can also match local variants and
is not sufficient to choose between them.

Each release/nightly channel uses one package listing, not separate Torch/XPU
channels. If those channels are needed later, define a compatibility matrix
and channel naming convention first.

## Install with pip

With compatible PyTorch XPU and runtime dependencies already installed:

```bash
pip install --only-binary=vllm-xpu-kernels \
  --index-url https://vllm-project.github.io/vllm-xpu-kernels/whl/ \
  --no-deps "vllm-xpu-kernels==0.1.16.1"
```

Replace the example version with the desired published version. `--no-deps`
assumes dependencies are already installed; without it, this single-package
index cannot resolve dependencies such as Torch. Use the appropriate PyTorch
XPU source for those dependencies.

Alternatively, keep PyPI as the main dependency index and use the package
listing as a wheel link source:

```bash
pip install --only-binary=vllm-xpu-kernels \
  --find-links https://vllm-project.github.io/vllm-xpu-kernels/whl/vllm-xpu-kernels/ \
  "vllm-xpu-kernels==0.1.16.1"
```

`--find-links` must point to the **package listing**, not `/whl/`, because pip
does not recursively follow directory links. `--extra-index-url .../whl/`
also works, but pip combines candidates from all indexes rather than giving
one source priority. Both approaches can select a PyPI copy of the same
version. To force the Release-hosted binary, use the first command or install
its direct Release asset URL. This index does not replace the PyTorch XPU
dependency index.

## Domain names

**You do not need to buy a domain.** The default HTTPS project-site URL is
sufficient for pip and browsers.

An optional custom domain requires a domain you control and access to its DNS.
Follow GitHub's
[custom domain configuration guide](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site).
Verify the domain, configure DNS, set **Settings > Pages > Custom domain**,
and enable **Enforce HTTPS** after the certificate is available. For example,
a dedicated `wheels.example.com` project domain would serve the index at
`https://wheels.example.com/whl/` without the repository path. Update install
commands to match the actual deployed URL; binary links still point to GitHub
Releases.

## Nightly wheels

The existing **wheel-per-commit** workflow builds and publishes nightly wheels
after a successful **run unit tests on Intel GPU.** run on `main`, for both
pushes and the existing daily scheduled test run. This reuses the native build
job rather than adding another nightly build matrix. The checkout uses the
upstream run's exact `head_sha`, not the latest moving `main`.

You can also run **Actions > wheel-per-commit > Run workflow > main** manually.
Manual builds do not wait for a test run. Dispatches on other branches, PR test
runs, failed test runs, and runs from other repositories cannot publish.
Merge these changes into `main` before triggering the workflow.

The build derives a base version from the nearest reachable `v[0-9]*` tag:
it increments the final numeric component of a final release, or retains the
base of a prerelease tag. A UTC timestamp, commit SHA, workflow run ID, and
attempt number produce a unique development version, for example:

```text
Tag used for versioning: v0.1.16
Wheel version:          0.1.17.dev20261029030000+gabcdef1.r12345.a1
Release tag:            nightly-20261029030000-abcdef1-12345-1
```

`VLLM_VERSION_OVERRIDE` sets the version during the native build. Before
publication, the workflow checks both the filename and wheel metadata against
that version. Existing ABI/platform build settings are unchanged; these checks
do not replace native compatibility testing.

A separate Ubuntu publishing job downloads only this run's wheel artifact and
uses `contents: write` to create a draft prerelease at the built commit, upload
the wheels and build information, and publish it with `--latest=false`.
The native build job has read-only repository permissions. Retrying publication
can resume an existing draft; already-published releases are not overwritten.

After publication, releases older than 30 days are deleted **only** if they
have the `nightly-` tag prefix, are prereleases, and are not drafts. This prefix
is reserved for nightly releases. Formal releases and RC releases are retained.
Deletion removes release assets but leaves Git tags intact; old nightly URLs
are therefore not permanent. Cleanup runs after successful publishing, not on
an independent daily schedule.

The index workflow refreshes after a successful `wheel-per-commit` completion.
This explicit `workflow_run` trigger is necessary because releases created
with `GITHUB_TOKEN` do not trigger another workflow's `release` event. Both
workflows must exist on the default branch. The six-hour index refresh and
manual index dispatch remain available as fallbacks.

Nightly index URLs:

```text
https://vllm-project.github.io/vllm-xpu-kernels/whl/nightly/
https://vllm-project.github.io/vllm-xpu-kernels/whl/nightly/vllm-xpu-kernels/
```

With compatible PyTorch XPU and runtime dependencies already installed:

```bash
pip install --upgrade --pre --no-deps --only-binary=vllm-xpu-kernels \
  --index-url https://vllm-project.github.io/vllm-xpu-kernels/whl/nightly/ \
  vllm-xpu-kernels
```

Use the full development version with `==` to pin a particular retained build.
For `--find-links`, use the nightly **package listing** URL. The formal release
index excludes nightly-tagged releases, even when installing with `--pre`.
