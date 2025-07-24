## About

This repo is designed as a vLLM plugin for Intel GPU (also know as XPU in pytorch) which will provide custom kernels used in vLLM.

## Getting started

current we use torch 2.8, oneapi is 2025.1.

### how this project works
Build system of this project will build a `_C.abi3.so` under build dir, and install step will copy this to `vllm_xpu_kernels` folder(will be local install if we use develop build or system/virtual env lib path if we use install). On vllm side, we will
`import vllm_xpu_kernels._C` at start time which should register all custom ops so we can directly use.

### prepare

Install oneapi 2025.1 deep learning essential [dependency](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).

create a new virtual env, install build dependency and torch dependency

```
pip install -r requirements.txt
```

### build & install
development install command (which will install to current dir):

```
VLLM_TARGET_DEVICE=xpu python3 setup.py develop
```

or

install to system dir

```
VLLM_TARGET_DEVICE=xpu python3 setup.py install
```

or build wheel(generate on dist folder)

```
VLLM_TARGET_DEVICE=xpu python3 setup.py bdist_wheel
```

### how to use in vLLM

please use this branch https://github.com/jikunshang/vllm/tree/xpu_kernel to install & test vllm. which already replace rms norm kernel from ipex to vllm-xpu-kernels.
