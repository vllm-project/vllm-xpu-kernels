## About

This repo is designed as a vLLM plugin which provides custom kernels for Intel GPU (known as XPU in PyTorch).

## Getting started
Currently we use PyTorch 2.8, oneapi 2025.1.

### How it works
python3 setup.py build - will build a `_C.abi3.so` under build directory
python3 setup.py install - will copy above .so to `vllm_xpu_kernels` folder
python3 setup.py develop - will be local install if we use develop build or system/virtual env lib path if we use install.

On vllm side, we will `import vllm_xpu_kernels._C` at start time which should register all custom ops so we can directly use.

### Prepare

Install oneapi 2025.1 deep learning essential [dependency](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).

Create a new virtual env, install build dependency and torch dependency

```
pip install -r requirements.txt
```

### Build & Install
Build development installation to current directory:

```
VLLM_TARGET_DEVICE=xpu python3 setup.py develop
```

or installation to system directory:

```
VLLM_TARGET_DEVICE=xpu python3 setup.py install
```

or build wheel (generated .whl in dist folder)

```
VLLM_TARGET_DEVICE=xpu python3 setup.py bdist_wheel
```

### How to use in vLLM
Please refer to temporary branch https://github.com/jikunshang/vllm/tree/xpu_kernel to install & test vllm which replaces `rms_norm` kernel from IPEX to vllm-xpu-kernels.

### Why Static Linking DNNL Instead of Shared Linking?

We chose to **statically link oneDNN (DNNL)** rather than using it as a shared library for the following reasons:

#### 1. **Version Compatibility**

Static linking ensures our application always uses the exact version of DNNL. With shared libraries, there's a risk that system-installed versions might be incompatible or introduce subtle bugs due to API/ABI changes.

#### 2. **Performance Consistency**

By linking statically, we avoid potential performance variability introduced by different builds or configurations of DNNL that might be present on the host system.

#### 3. **Avoiding Runtime Errors**

Using shared libraries requires correct paths and environment setup (`LD_LIBRARY_PATH` on Linux). Static linking avoids issues where DNNL cannot be found or loaded at runtime.

#### 4. **Aligning with PyTorch**

One key reason to use static linking is to maintain consistency with the PyTorch ecosystem. PyTorch itself statically links libraries like DNNL to ensure deterministic and reliable behavior across different environments.
