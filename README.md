## About

This repository is a vLLM plugin that provides custom kernels for Intel GPUs (called XPU in PyTorch).

## Getting started
We currently use PyTorch 2.10 and oneAPI 2025.3.

### How it works
vLLM defines and implements many custom Torch ops/kernels in the vLLM codebase for CUDA. As a PyTorch-supported device, Intel GPUs can provide equivalent ops/kernels for the XPU backend. This repository follows the Torch op registration/dispatch pattern. On the vLLM side, `import vllm_xpu_kernels._C` is executed at startup, which registers all custom ops for direct use.

### Prepare

Install the oneAPI 2025.3 deep learning essentials [dependency](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).

Create a new virtual environment, then install the build and Torch dependencies:

```
pip install -r requirements.txt
```

### Build & Install
For a development install in the current directory:

```
pip install --extra-index-url=https://download.pytorch.org/whl/xpu -e . -v
# or for faster build, you can use --no-build-isolation
pip install --no-build-isolation -e . -v
```

For an installation to the system directory:

```
pip install --extra-index-url=https://download.pytorch.org/whl/xpu  .
# or for faster build, you can use --no-build-isolation
pip install --no-build-isolation . 
```

To build a wheel (the generated `.whl` file will be placed in the `dist` folder):

```
pip wheel --extra-index-url=https://download.pytorch.org/whl/xpu  .
# or for faster build, you can use --no-build-isolation
pip wheel --no-build-isolation  .
```

For incremental builds:

```
python3 -m build --wheel --no-isolation
```

### How to use in vLLM
After vLLM [RFC#33214](https://github.com/vllm-project/vllm/issues/33214) was completed, vLLM-XPU migrated to a vLLM-XPU-kernels-based implementation. You can now pull the latest vLLM code and install vLLM-XPU manually; vLLM-XPU-kernels will be installed automatically as a wheel dependency.

### Why statically link DNNL instead of using shared linking?

We chose to **statically link oneDNN (DNNL)** rather than using it as a shared library for the following reasons:

#### 1. **Version Compatibility**

Static linking ensures that the application always uses the exact DNNL version we ship. With shared libraries, system-installed versions may be incompatible or introduce subtle bugs due to API/ABI changes.

#### 2. **Performance Consistency**

With static linking, we avoid performance variability caused by different DNNL builds or configurations that may exist on the host system.

#### 3. **Avoiding Runtime Errors**

Shared libraries require correct paths and environment setup (`LD_LIBRARY_PATH` on Linux). Static linking avoids runtime issues where DNNL cannot be found or loaded.

#### 4. **Aligning with PyTorch**

Another key reason for static linking is consistency with the PyTorch ecosystem. PyTorch also statically links libraries such as DNNL to ensure deterministic and reliable behavior across environments.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
