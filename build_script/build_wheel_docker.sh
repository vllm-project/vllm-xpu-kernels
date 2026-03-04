#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Build vllm-xpu-kernels wheel using the root Dockerfile.xpu.
#
# This script:
#   1. Builds a Docker image from Dockerfile.xpu (unmodified)
#   2. Starts a container and builds the wheel inside it
#   3. Copies the built wheel to the host
#
# Usage:
#   ./build_wheel_docker.sh [OPTIONS]
#
# Options:
#   --output-dir <dir>     Host directory for the built wheel  (default: ./dist)
#   --image-name <name>    Docker image tag                    (default: vllm-xpu-kernels-builder)
#   --gpu-profile <name>   GPU runtime profile                 (default: default)
#   --version <ver>        Override wheel version via VLLM_VERSION_OVERRIDE
#   --max-jobs <n>         Limit parallel compilation jobs     (default: auto)
#   --no-cache             Build Docker image without layer cache
#   -h, --help             Show this help message
#
# Prerequisites:
#   - Docker installed and running
#   - Network access (to pull base image & pip packages)

set -euo pipefail

# ─── Defaults ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/dist"
IMAGE_NAME="vllm-xpu-kernels-builder"
CONTAINER_NAME="vllm-xpu-wheel-build-$$"
DOCKER_NO_CACHE=""
GPU_RUNTIME_PROFILE="default"
VLLM_VERSION_OVERRIDE=""
MAX_JOBS=""

# ─── Parse arguments ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --image-name)
            IMAGE_NAME="$2"; shift 2 ;;
        --gpu-profile)
            GPU_RUNTIME_PROFILE="$2"; shift 2 ;;
        --version)
            VLLM_VERSION_OVERRIDE="$2"; shift 2 ;;
        --max-jobs)
            MAX_JOBS="$2"; shift 2 ;;
        --no-cache)
            DOCKER_NO_CACHE="--no-cache"; shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *)
            echo "ERROR: Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ─── Cleanup on exit ───────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Cleaning up container '${CONTAINER_NAME}'..."
    docker rm -f "${CONTAINER_NAME}" > /dev/null 2>&1 || true
}
trap cleanup EXIT

# ─── Step 1: Build Docker image from Dockerfile.xpu ────────────────────────
echo "=========================================="
echo " Step 1/3: Building Docker image '${IMAGE_NAME}'"
echo "           from Dockerfile.xpu"
echo "=========================================="

docker build \
    ${DOCKER_NO_CACHE} \
    --build-arg GPU_RUNTIME_PROFILE="${GPU_RUNTIME_PROFILE}" \
    -f "${REPO_ROOT}/Dockerfile.xpu" \
    -t "${IMAGE_NAME}" \
    "${REPO_ROOT}"

echo "Docker image '${IMAGE_NAME}' built successfully."

# ─── Step 2: Run container to build the wheel ──────────────────────────────
echo ""
echo "=========================================="
echo " Step 2/3: Building wheel inside container"
echo "           Container: ${CONTAINER_NAME}"
echo "=========================================="

# Compose the in-container build script
BUILD_CMD='
set -euo pipefail

echo ">>> Copying source tree into container..."
cp -a /workspace/src /workspace/vllm-xpu-kernels
cd /workspace/vllm-xpu-kernels
git config --global --add safe.directory /workspace/vllm-xpu-kernels
rm -rf build/ dist/ vllm_xpu_kernels.egg-info/ .deps/

# setuptools-scm needs git metadata for version detection.
# The bind-mounted .git may reference absolute worktree paths that do not
# exist inside the container, so we re-init if needed.
if [ -d .git ]; then
    # Attempt to read the version; if it fails, re-initialise git
    git describe --tags --always > /dev/null 2>&1 || {
        echo ">>> Re-initialising git for setuptools-scm..."
        rm -rf .git
        git init -q
        git add -A
        git commit -q -m "build" --allow-empty
        git tag -a v0.0.0 -m "placeholder" 2>/dev/null || true
    }
else
    git init -q
    git add -A
    git commit -q -m "build" --allow-empty
    git tag -a v0.0.0 -m "placeholder" 2>/dev/null || true
fi
'

# Version override
if [ -n "${VLLM_VERSION_OVERRIDE}" ]; then
    BUILD_CMD+="
export VLLM_VERSION_OVERRIDE=\"${VLLM_VERSION_OVERRIDE}\"
echo \">>> Version override: \${VLLM_VERSION_OVERRIDE}\"
"
fi

# MAX_JOBS
if [ -n "${MAX_JOBS}" ]; then
    BUILD_CMD+="
export MAX_JOBS=\"${MAX_JOBS}\"
echo \">>> MAX_JOBS=\${MAX_JOBS}\"
"
fi

BUILD_CMD+='
echo ">>> Installing build dependencies..."
uv pip install \
    "cmake>=3.26" \
    ninja \
    "packaging>=24.2" \
    "setuptools>=77.0.3,<80.0.0" \
    "setuptools-scm>=8" \
    "torch==2.10.0+xpu" \
    wheel \
    regex \
    jinja2 \
    build

echo ">>> Building wheel..."
python setup.py bdist_wheel --dist-dir /workspace/dist --py-limited-api=cp38

echo ">>> Wheel build complete."
ls -lh /workspace/dist/*.whl
'

docker run \
    --name "${CONTAINER_NAME}" \
    -v "${REPO_ROOT}:/workspace/src:ro" \
    "${IMAGE_NAME}" \
    bash -c "${BUILD_CMD}"

# ─── Step 3: Copy wheel to host ────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Step 3/3: Copying wheel to ${OUTPUT_DIR}"
echo "=========================================="

mkdir -p "${OUTPUT_DIR}"
docker cp "${CONTAINER_NAME}:/workspace/dist/." "${OUTPUT_DIR}/"

echo ""
echo "Built wheel(s):"
ls -lh "${OUTPUT_DIR}"/*.whl 2>/dev/null || echo "  (no wheels found – build may have failed)"

echo ""
echo "Done! Wheel(s) available in: ${OUTPUT_DIR}"
