#pragma once
#include <Python.h>

#ifdef VLLM_USE_STABLE_TORCH_LIBRARY
  #include <torch/csrc/stable/library.h>
#else
  #include <torch/library.h>
#endif

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

// A version of the TORCH_LIBRARY macro that expands the NAME, i.e. so NAME
// could be a macro instead of a literal token.
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

// A version of the TORCH_LIBRARY_IMPL macro that expands the NAME, i.e. so NAME
// could be a macro instead of a literal token.
#define TORCH_LIBRARY_IMPL_EXPAND(NAME, DEVICE, MODULE) \
  TORCH_LIBRARY_IMPL(NAME, DEVICE, MODULE)

#ifdef VLLM_USE_STABLE_TORCH_LIBRARY
  #define VLLM_TORCH_LIBRARY_FRAGMENT_EXPAND(NAME, MODULE) \
    STABLE_TORCH_LIBRARY_FRAGMENT(NAME, MODULE)
  #define VLLM_TORCH_LIBRARY_IMPL_EXPAND(NAME, DEVICE, MODULE) \
    STABLE_TORCH_LIBRARY_IMPL(NAME, DEVICE, MODULE)
  #define VLLM_TORCH_IMPL(MODULE, NAME, FUNC) MODULE.impl(NAME, TORCH_BOX(FUNC))
#else
  #define VLLM_TORCH_LIBRARY_FRAGMENT_EXPAND(NAME, MODULE) \
    TORCH_LIBRARY_EXPAND(NAME, MODULE)
  #define VLLM_TORCH_LIBRARY_IMPL_EXPAND(NAME, DEVICE, MODULE) \
    TORCH_LIBRARY_IMPL_EXPAND(NAME, DEVICE, MODULE)
  #define VLLM_TORCH_IMPL(MODULE, NAME, FUNC) MODULE.impl(NAME, FUNC)
#endif

#ifndef VLLM_TORCH_OP_NAMESPACE
  #define VLLM_TORCH_OP_NAMESPACE TORCH_EXTENSION_NAME
#endif

#ifndef VLLM_TORCH_CACHE_OP_NAMESPACE
  #define VLLM_TORCH_CACHE_OP_NAMESPACE CONCAT(TORCH_EXTENSION_NAME, _cache_ops)
#endif

// REGISTER_EXTENSION allows the shared library to be loaded and initialized
// via python's import statement.
#define REGISTER_EXTENSION(NAME)           \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() { \
    static struct PyModuleDef module = {   \
        PyModuleDef_HEAD_INIT,             \
        STRINGIFY(NAME),                   \
        nullptr,                           \
        0,                                 \
        nullptr,                           \
        nullptr,                           \
        nullptr,                           \
        nullptr,                           \
        nullptr};                          \
    return PyModule_Create(&module);       \
  }
