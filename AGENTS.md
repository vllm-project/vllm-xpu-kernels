# AGENTS.md

This file is the shared guidance for AI coding agents working in this
repository. Keep tool-specific entry points, such as `CLAUDE.md`, thin and point
them here so the project rules do not drift.

## Project Overview

`vllm-xpu-kernels` is a vLLM component that provides optimized custom kernels
for Intel GPUs (XPU). Kernels are written in SYCL/DPC++ and use oneDNN and
SYCL-TLA where appropriate. The package registers PyTorch custom ops that vLLM
can dispatch on Intel GPU hardware.

## Source Of Truth

- Prefer current build metadata over stale docs when versions disagree:
  `pyproject.toml`, `setup.py`, `CMakeLists.txt`, and `tools/envs.py`.
- Python package metadata currently requires Python `>=3.9,<3.14`; CMake still
  searches `3.9` through `3.12`, so verify interpreter support before changing
  build behavior.
- PyTorch XPU support is currently `2.12.0+xpu` in `pyproject.toml`, with CMake
  expecting torch `2.12.0` for XPU.
- oneAPI 2025.3 is the expected toolchain. Source
  `/opt/intel/oneapi/setvars.sh` before building native extensions.

## Environment And Commands

- Do not use system `python3` or bare `pip` for project work. Use `uv` and the
  repository virtualenv.
- Create the environment with `uv venv --python 3.12` unless a task explicitly
  targets another supported interpreter.
- Use `.venv/bin/python` in non-interactive commands because shell activation
  does not persist across tool calls.
- Install lint tooling with `uv pip install pre-commit` and run
  `pre-commit install` when setting up a fresh checkout.
- For editable development installs, use:

```bash
uv pip install --no-build-isolation -e .
```

- Run targeted tests with:

```bash
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

- Run all pre-commit hooks with:

```bash
pre-commit run --all-files
```

## Build System

- `setup.py` drives CMake through a custom `cmake_build_ext` command.
- `CMakeLists.txt` defines extension targets and native SYCL/SYCL-TLA build
  options.
- `pyproject.toml` defines Python metadata, build dependencies, and formatter or
  lint configuration.
- Native builds require CMake `>=3.26`, Ninja, oneAPI `icx`/`icpx`, PyTorch XPU,
  and sufficient memory. The build estimates parallel jobs assuming about 8 GB
  per compile process.

Important build environment variables:

- `MAX_JOBS`: override compile parallelism.
- `CMAKE_BUILD_TYPE`: `Debug`, `Release`, or `RelWithDebInfo`.
- `VLLM_USE_PRECOMPILED=1`: extract precompiled shared objects instead of
  building from source.
- `VLLM_PRECOMPILED_WHEEL_LOCATION`: local or remote wheel for precompiled
  extraction.
- `VERBOSE=1`: enable verbose CMake makefile output.
- `BUILD_SYCL_TLA_KERNELS`: enable or disable SYCL-TLA based kernels.
- `VLLM_XPU_ENABLE_XE2` and `VLLM_XPU_ENABLE_XE_DEFAULT`: architecture-family
  toggles.
- `BASIC_KERNELS_ENABLED`, `FA2_KERNELS_ENABLED`, `MOE_KERNELS_ENABLED`,
  `GDN_KERNELS_ENABLED`, `MQA_LOGITS_KERNELS_ENABLED`,
  `XPU_SPECIFIC_KERNELS_ENABLED`, `XPUMEM_ALLOCATOR_ENABLED`: extension or
  kernel-family toggles forwarded from `setup.py` to CMake.
- `VLLM_CHUNK_PREFILL_CONFIG`: selects the chunk prefill attention kernel
  config used for prompt processing. Accepts a preset name with or without the
  `.conf` suffix from `csrc/xpu/attn/kernel_configs/`, or an absolute path to a
  custom config file. Current presets are `chunk_prefill_default.conf` for common
  Llama, Qwen, DeepSeek, and Falcon coverage, and `chunk_prefill_full.conf` for
  all supported variants.
- `VLLM_PAGED_DECODE_CONFIG`: selects the paged decode attention kernel config
  used for token generation. It follows the same preset-name or absolute-path
  rules. Current presets are `paged_decode_default.conf` for common Llama, Qwen,
  DeepSeek, and Falcon coverage, and `paged_decode_full.conf` for all supported
  variants.
- If either config variable is unset, CMake currently defaults to the matching
  `*_full.conf` preset. Use the `*_default.conf` presets to reduce compile time
  when broad model coverage is not needed.
- `VLLM_XPU_AOT_DEVICES` and `VLLM_XPU_XE2_AOT_DEVICES`: override AOT target
  devices. Empty values intentionally disable AOT for the corresponding set.
- `VLLM_CUTLASS_SRC_DIR`: use a local SYCL-TLA checkout instead of FetchContent.

## Architecture Notes

The build produces Python extension modules under `vllm_xpu_kernels/`:

| Module | Main sources | Purpose |
| --- | --- | --- |
| `_C` | `csrc/*.cpp`, `csrc/quantization/**` | Core ops such as RMS norm, activations, RoPE, cache ops, quantization, tensor utilities, memory info, and top-k per row. |
| `_vllm_fa2_C` | `csrc/flash_attn/*.cpp`, `csrc/xpu/attn/**` | Flash attention and variable-length attention interfaces. |
| `_moe_C` | `csrc/moe/*.cpp` | MoE ops such as align, gather, sum, grouped top-k, and fused grouped top-k. |
| `_xpu_C` | `csrc/xpu/*.cpp` and subdirectories | XPU-specific ops such as LoRA, grouped GEMM, GDN attention, MQA logits, samplers, and RoPE variants. |
| `xpumem_allocator` | `csrc/utils/mem_alloc.cpp` | XPU memory allocator hooks with Python callbacks. |

Op registration generally follows this pattern:

1. Declare signatures in `csrc/ops.h` or the relevant module header.
2. Implement CPU-hosted binding logic and SYCL kernels in `csrc/**`.
3. Register schemas and dispatch with `TORCH_LIBRARY_EXPAND` or related PyTorch
   dispatcher macros in `torch_bindings.cpp` files.
4. Exercise Python-facing behavior through `torch.ops._C.*`,
   `torch.ops._xpu_C.*`, or wrapper modules in `vllm_xpu_kernels/`.

## Testing Guidance

- Prefer the smallest test that covers the touched kernel, wrapper, or build
  path.
- Use `.venv/bin/python -m pytest ...` for Python tests.
- GPU tests may require Intel hardware and Level Zero access; if unavailable,
  state that clearly instead of treating failures as code failures.
- Useful test environment variables include `ZE_AFFINITY_MASK`,
  `SKIP_HANG_KERNEL=1`, `SKIP_ACC_ERROR_KERNEL=1`,
  `VLLM_XPU_FORCE_XE_DEFAULT_KERNEL=1`, and
  `XPU_KERNEL_PYTEST_PROFILER=MINI`.
- `tests/register_ops.py` is useful for checking op registration and dispatch
  expectations.

## Code Style

- Keep changes focused and consistent with nearby code.
- Python uses yapf and ruff with an 80-column target from `pyproject.toml`.
- C++ and SYCL code use the repository `.clang-format` configuration.
- CMake is checked by cmake-format and cmake-lint.
- All SYCL compilations force-include `csrc/sycl_first.h`; keep direct SYCL
  include changes compatible with that setup.
- Avoid broad refactors when adding or fixing a kernel. Update tests and docs
  only where they directly support the change.

## Documentation And Release Notes

- Update `KERNEL_CONFIGURATION.md` when changing attention kernel config
  presets, config syntax, or build-time selection behavior.
- Update `README.md` only for user-facing behavior or setup changes.
- Release-note files are version-specific; do not edit old release notes unless
  the task explicitly asks for it.
- `custom-kernels-inventory.md` tracks registered custom kernels. Update it when
  adding, removing, or renaming Python-visible ops.

## Git Hygiene

- The working tree may contain unrelated user changes or generated artifacts.
  Do not revert changes you did not make.
- Do not commit, create branches, or run destructive git commands unless the
  user explicitly asks.
- Always sign off commits with `git commit -s` or an equivalent `Signed-off-by:`
  trailer.
- If the user asks for commit text, include appropriate trailers such as
  `Co-authored-by: GitHub Copilot` only when requested or consistent with their
  workflow.
