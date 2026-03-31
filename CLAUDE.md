# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vllm-xpu-kernels is a vLLM component that provides optimized custom kernels for Intel GPUs (XPU) to accelerate LLM inference. Kernels are written in SYCL/DPC++ and leverage oneDNN for deep learning primitives.

## Build System

The project uses a hybrid Python/C++ build system:
- **setup.py**: Entry point for pip installs, delegates to CMake via custom `cmake_build_ext` class
- **CMakeLists.txt**: Builds SYCL kernels and creates multiple Python extension modules
- **pyproject.toml**: Standard Python project metadata and tool configuration

### Requirements

- Intel oneAPI 2025.3+ (source `/opt/intel/oneapi/setvars.sh` before building)
- PyTorch 2.10.0+xpu
- CMake >= 3.26
- Ninja build system
- Python 3.9 – 3.12

### Deployment Workflow

- **Never use system `python3` or bare `pip`/`pip install`.** All Python commands must go through `uv` and `.venv/bin/python`.

#### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

# Always make sure `pre-commit` and its hooks are installed:
uv pip install -r requirements/lint.txt
pre-commit install
```

#### Installing dependencies

```bash
# If you are also making C/C++ changes:
uv pip install --no-build-isolation -e . 
```

#### Running tests

> Requires [Environment setup](#environment-setup) and [Installing dependencies](#installing-dependencies).

```bash
# Run a specific test file (use .venv/bin/python directly;
# `source activate` does not persist in non-interactive shells):
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

#### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`). For example:

```text
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```

### Build Configuration

Key environment variables (defined in `tools/envs.py`):
- `MAX_JOBS`: Number of parallel compilation jobs (default: auto-detect based on CPU cores and memory)
- `CMAKE_BUILD_TYPE`: Debug, Release, or RelWithDebInfo
- `VLLM_USE_PRECOMPILED=1`: Skip build and use precompiled binaries
- `VERBOSE=1`: Verbose CMake output
- `VLLM_XPU_AOT_DEVICES`: Override AOT compilation targets (default: "pvc,bmg,bmg-g21-a0,bmg-g31-a0")

## Architecture

### Extension Modules

The build produces multiple Python extension modules (all in `vllm_xpu_kernels/`):

| Module | Source | Purpose |
|--------|--------|---------|
| `_C` | `csrc/*.cpp` | Core ops: RMS norm, activations, RoPE, cache ops, FP8/MxFP4 quant, topk |
| `_vllm_fa2_C` | `csrc/flash_attn/*.cpp` | Flash attention kernels (variable-length) |
| `_moe_C` | `csrc/moe/*.cpp` | MoE ops: align, gather, sum, grouped topk |
| `_xpu_C` | `csrc/xpu/*.cpp` | XPU-specific ops: LoRA, grouped GEMM, GDN attention, DeepSeek RoPE |
| `xpumem_allocator` | `csrc/utils/mem_alloc.cpp` | XPU memory allocator with Python callbacks |

### Code Organization

```
csrc/                          # C++/SYCL kernel source
├── torch_bindings.cpp         # PyTorch op registration (_C module)
├── cache.cpp                  # KV cache operations
├── layernorm.cpp              # RMS norm, layer norm
├── activation.cpp             # SiLU, GeLU, SwiGLU
├── pos_encoding_kernels.cpp   # RoPE implementations
├── quantization/              # FP8, MxFP4 quantization
├── flash_attn/                # Flash attention kernels
├── moe/                       # MoE operations
├── xpu/                       # XPU-specific implementations
│   ├── torch_bindings.cpp     # _xpu_C module registration
│   ├── lora/                  # LoRA shrink/expand
│   ├── grouped_gemm/          # Grouped GEMM (XE default + XE2)
│   ├── attn/xe_2/             # Flash attention XE2 kernels (SYCL-TLA)
│   ├── gdn_attn/xe_2/         # GDN attention XE2 kernels
│   ├── onednn/                # oneDNN-based kernels (FP8 GEMM)
│   └── sycl/                  # SYCL kernels (DeepSeek RoPE)
└── utils/                     # Memory utilities

vllm_xpu_kernels/              # Python package
├── __init__.py                # Exports flash_attn_varlen_func
├── flash_attn_interface.py    # Python interface for flash attention
├── fused_moe_interface.py     # Python interface for MoE ops
└── quantization/              # Quantization utilities

tests/                         # Test suite
├── register_ops.py            # Op registration and dispatch testing
├── test_*.py                  # Kernel-specific tests
├── utils.py                   # Test utilities
├── conftest.py                # pytest fixtures
└── */                         # Subdirectories for flash_attn, fused_moe, etc.

benchmark/                     # Performance benchmarks
├── benchmark_*.py             # Kernel-specific benchmarks
└── src/                       # Benchmark utilities
```

### Op Registration Pattern

Ops are registered with PyTorch's dispatcher using `TORCH_LIBRARY_EXPAND` macros:

1. **Declaration** in `csrc/ops.h`: Function signatures
2. **Implementation** in kernel files (e.g., `csrc/layernorm.cpp`): SYCL kernel code
3. **Registration** in `csrc/torch_bindings.cpp`: `ops.def()` for schema, `ops.impl()` for XPU dispatch
4. **Python usage**: `torch.ops._C.rms_norm(...)` or `torch.ops._xpu_C.grouped_gemm(...)`

### SYCL-TLA (Cutlass) Integration

XE2 kernels (attention, grouped GEMM) use SYCL-TLA (Intel's SYCL port of CUTLASS):
- Fetched from https://github.com/intel/sycl-tla.git at CMake configure time
- Static libraries built as separate CMake subprojects
- Linked into corresponding Python extension modules

## Testing

### Running Tests

> Requires [Environment setup](#environment-setup) and [Installing dependencies](#installing-dependencies).

```bash
# Run a specific test file (use .venv/bin/python directly;
# `source activate` does not persist in non-interactive shells):
.venv/bin/python -m pytest tests/path/to/test_file.py -v

# Full test suite
.venv/bin/python -m pytest tests/

# Single test file
.venv/bin/python -m pytest tests/test_layernorm.py

# Single test function
.venv/bin/python -m pytest tests/test_layernorm.py::test_rms_norm

# With verbose output
.venv/bin/python -m pytest -v -s tests/
```

### Test Environment Variables

- `ZE_AFFINITY_MASK=0,1`: Limit GPU visibility for testing
- `SKIP_HANG_KERNEL=1`: Skip tests that may hang
- `SKIP_ACC_ERROR_KERNEL=1`: Skip tests with accuracy issues
- `VLLM_XPU_FORCE_XE_DEFAULT_KERNEL=1`: Force XE default kernels instead of XE2
- `XPU_KERNEL_PYTEST_PROFILER=MINI`: Use mini test parameters (defined in `MINI_PYTEST_PARAMS` per module)

### Test Structure

Tests use custom fixtures in `conftest.py`:
- `reset_default_device`: Reset torch default device after tests
- `kv_cache_factory`: Factory for creating KV caches with random data

## Code Quality

### Pre-commit Hooks

> Requires [Environment setup](#environment-setup).

```bash
# Install hooks
pre-commit install

# Run all hooks manually
pre-commit run --all-files

# Run specific hook
pre-commit run yapf --all-files
pre-commit run ruff --all-files
```

Configured hooks (`yapf`, `ruff`, `isort`, `clang-format`, `cmake-format`, `codespell`, `mypy`, `shellcheck`, `pymarkdown`, SPDX header checks).

### Code Style

- **Python**: yapf formatter, ruff linter (line length 80)
- **C++**: clang-format with `.clang-format` config
- **CMake**: cmake-format and cmake-lint

## CI/CD

GitHub Actions workflows in `.github/workflows/`:
- **ut.yaml**: Unit tests on PVC and BMG hardware (self-hosted runners)
- **wheel-per-commit.yaml**: Build wheels per commit
- **pre-commit.yml**: Pre-commit checks

Tests run in Docker containers with Intel GPU access (`--device /dev/dri --privileged`).

## Important Notes

- **AOT Compilation**: Kernels are compiled ahead-of-time for specific Intel GPU architectures (PVC, BMG). The `VLLM_XPU_AOT_DEVICES` env var controls which targets are built.
- **SYCL First Header**: All SYCL files include `csrc/sycl_first.h` first (forced via compiler flag) to ensure proper SYCL setup.
- **Import Order**: Importing `vllm_xpu_kernels._C` (or other modules) registers ops with PyTorch's dispatcher. This happens automatically when vLLM imports this package.
- **Memory**: Each compile process can use ~8GB of memory; the build system auto-limits parallelism based on available memory.
