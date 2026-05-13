#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Build vllm-xpu-kernels wheel inside a Docker container and copy it to the host.
#
# Usage:
#   ./build_wheel.sh [--output-dir <dir>] [--image-name <name>] [--gpu-profile <name>] [--no-cache]
#
# Options:
#   --output-dir     Directory on host to store the built wheel (default: ./dist)
#   --image-name     Docker image name/tag (default: vllm-xpu-kernels-builder)
#   --gpu-profile    GPU runtime profile name from gpu_runtime_packages.json (default: default)
#   --version        Set wheel version via VLLM_VERSION_OVERRIDE (default: auto from setuptools-scm)
#   --no-cache       Build Docker image without cache
#
# Prerequisites:
#   - Docker installed and running
#   - GPU/XPU not required at build time (only compilation)

set -euo pipefail

# ─── Defaults ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/dist"
IMAGE_NAME="vllm-xpu-kernels-builder"
CONTAINER_NAME="vllm-xpu-wheel-build-$$"
DOCKER_NO_CACHE=""
GPU_RUNTIME_PROFILE="default"
VLLM_VERSION_OVERRIDE=""

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
        --no-cache)
            DOCKER_NO_CACHE="--no-cache"; shift ;;
        -h|--help)
            head -18 "$0" | tail -16; exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ─── Step 1: Build Docker image ────────────────────────────────────────────
echo "=========================================="
echo " Step 1/3: Building Docker image '${IMAGE_NAME}'"
echo "=========================================="

BUILD_ARGS="--build-arg GPU_RUNTIME_PROFILE=${GPU_RUNTIME_PROFILE}"
if [ -n "${VLLM_VERSION_OVERRIDE}" ]; then
    BUILD_ARGS="${BUILD_ARGS} --build-arg VLLM_VERSION_OVERRIDE=${VLLM_VERSION_OVERRIDE}"
    echo "  Version override: ${VLLM_VERSION_OVERRIDE}"
fi

docker build \
    ${DOCKER_NO_CACHE} \
    ${BUILD_ARGS} \
    -f "${SCRIPT_DIR}/Dockerfile.build_wheel" \
    -t "${IMAGE_NAME}" \
    "${SCRIPT_DIR}"

# ─── Step 2: Run container to build the wheel ──────────────────────────────
echo "=========================================="
echo " Step 2/3: Building wheel inside container '${CONTAINER_NAME}'"
echo "=========================================="

docker run \
    --name "${CONTAINER_NAME}" \
    "${IMAGE_NAME}"

# ─── Step 3: Copy wheel out ────────────────────────────────────────────────
echo "=========================================="
echo " Step 3/3: Copying wheel to ${OUTPUT_DIR}"
echo "=========================================="

mkdir -p "${OUTPUT_DIR}"

# Copy all .whl files from the container's /workspace/dist/ to host
docker cp "${CONTAINER_NAME}:/workspace/dist/." "${OUTPUT_DIR}/"

# Show what was built
echo ""
echo "Built wheel(s):"
ls -lh "${OUTPUT_DIR}"/*.whl 2>/dev/null || echo "  (no wheels found – build may have failed)"

# ─── Cleanup ────────────────────────────────────────────────────────────────
echo ""
echo "Cleaning up container '${CONTAINER_NAME}'..."
docker rm "${CONTAINER_NAME}" > /dev/null 2>&1 || true

echo ""
echo "Done! Wheel(s) available in: ${OUTPUT_DIR}"