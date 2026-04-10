# vllm-xpu-kernels

A [vLLM](https://github.com/vllm-project/vllm) component that provides optimized custom kernels for Intel GPUs (XPU) to accelerate LLM inference.

## Table of Contents

- [About](#about)
- [Supported Kernels](#supported-kernels)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
  - [How It Works](#how-it-works)
  - [Installation](#installation)
  - [Build Options](#build-options)
  - [Using with vLLM](#using-with-vllm)
- [Testing](#testing)
- [Benchmarks](#benchmarks)
- [Design Notes](#design-notes)
- [License](#license)

---

## About

vLLM defines and implements many custom Torch ops and kernels. This repository provides custom implementations for the Intel XPU (GPU) backend, enabling high-throughput LLM inference on Intel hardware.

Kernels are written in SYCL/DPC++ and leverage [oneDNN](https://github.com/oneapi-src/oneDNN) for deep learning primitives. The library follows the PyTorch custom op registration and dispatch pattern — importing it at startup registers all ops for seamless use within vLLM.

## Supported Kernels

| Category | Operations |
|---|---|
| **Normalization** | RMS norm, fused add-RMS norm, layer norm |
| **Activation** | SiLU-and-mul, mul-and-SiLU, GeLU (fast/new/quick/tanh), SwigluOAI |
| **Attention** | Flash attention (variable-length), GDN attention, XE2 attention variants |
| **Positional Encoding** | Rotary embedding (NeoX and GPT-J styles), DeepSeek scaling RoPE |
| **Mixture of Experts** | TopK scoring (softmax/sigmoid), grouped TopK, fused grouped TopK; MoE align sum, MoE gather, expert remapping |
| **LoRA** | LoRA operator support |
| **Quantization** | FP8, MxFP4 quantization and GEMM |
| **GEMM** | Grouped GEMM |
| **Misc** | TopK per row, memory utilities |

## Requirements

- **Python**: 3.9 – 3.12
- **PyTorch**: 2.10.0+xpu
- **oneAPI**: 2025.3 ([Base Toolkit download](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html))
- **CMake**: ≥ 3.26
- **Ninja** build system

## Getting Started

### How It Works

vLLM calls `import vllm_xpu_kernels._C` at startup, which registers all custom ops into the PyTorch dispatcher. From that point on, XPU ops are dispatched automatically whenever vLLM runs on Intel GPU hardware — no additional code changes are required in vLLM itself.

### Installation

**1. Install oneAPI 2025.3**

Download and install the [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html), then source the environment:

```bash
source /opt/intel/oneapi/setvars.sh
```

**2. Create a virtual environment and install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate

git clone https://github.com/vllm-project/vllm-xpu-kernels
cd vllm-xpu-kernels

pip install -r requirements.txt
```

### Build Options

**Development install** (editable, source in current directory):

```bash
pip install --extra-index-url=https://download.pytorch.org/whl/xpu -e . -v
# Faster: skip build isolation if dependencies are already present
pip install --no-build-isolation -e . -v
```

**Standard install** (to site-packages):

```bash
pip install --extra-index-url=https://download.pytorch.org/whl/xpu .
# or
pip install --no-build-isolation .
```

**Build a wheel** (output goes to `dist/`):

```bash
pip wheel --extra-index-url=https://download.pytorch.org/whl/xpu .
# or
pip wheel --no-build-isolation .
```

**Incremental rebuild** (fastest for iterative development):

```bash
python -m build --wheel --no-isolation
```

### Using with vLLM

After [vLLM RFC#33214](https://github.com/vllm-project/vllm/issues/33214) was completed, vLLM-XPU migrated to a `vllm-xpu-kernels`-based implementation. Installing the latest vLLM for XPU will pull in `vllm-xpu-kernels` automatically as a wheel dependency — no manual integration is required.

## Testing

Run the full test suite with pytest:

```bash
pytest tests/
```

Individual test modules cover activations, cache operations, attention, MoE, LoRA, quantization, and memory utilities. See the [`tests/`](tests/) directory for the complete list.

## Benchmarks

Benchmark scripts for individual kernels are in the [`benchmark/`](benchmark/) directory:

```bash
python benchmark/benchmark_layernorm.py
python benchmark/benchmark_lora.py
python benchmark/benchmark_grouped_topk.py
# etc.
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
