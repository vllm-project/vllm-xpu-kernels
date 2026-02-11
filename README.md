## About

This repo is designed as a vLLM plugin which provides custom kernels for Intel GPU (known as XPU in PyTorch).

## Getting started
Currently we use PyTorch 2.10, oneapi 2025.3.

### How it works
vLLM define and implement a lot custom torch ops/kernels in vllm code base for cuda. as a torch supported device, Intel GPU can also do similar work to provide same ops/kernels in vLLM for xpu device. We followed torch op register/dispatch method in this repo. On vllm side, we will do `import vllm_xpu_kernels._C` at start time which should register all custom ops so we can directly use.

### Prepare

Install oneapi 2025.3 deep learning essential [dependency](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).

Create a new virtual env, install build dependency and torch dependency

```
pip install -r requirements.txt
```

### Build & Install
Build development installation to current directory:

```
pip install --extra-index-url=https://download.pytorch.org/whl/xpu -e . -v
# or for faster build, you can use --no-build-isolation
pip install --no-build-isolation -e . -v
```

or installation to system directory:

```
pip install --extra-index-url=https://download.pytorch.org/whl/xpu  .
# or for faster build, you can use --no-build-isolation
pip install --no-build-isolation . 
```

or build wheel (generated .whl in dist folder)

```
pip wheel --extra-index-url=https://download.pytorch.org/whl/xpu  .
# or for faster build, you can use --no-build-isolation
pip wheel --no-build-isolation  .
```

Incremental build

```
python3 -m build --wheel --no-isolation
```

### How to use in vLLM
As vLLM [RFC#33214](https://github.com/vllm-project/vllm/issues/33214) completed, vLLM-xpu is migrated to vLLM-xpu-kernels based implementation. You can pull latest vllm code and install vllm-xpu manually now.

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

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
