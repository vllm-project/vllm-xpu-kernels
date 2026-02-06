// An XPUPluggableAllocator based on SYCL USM APIs.
// This provides similar functionality to cu_mem.cpp but for Intel XPU devices.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <iostream>
#include <unordered_map>
#include <mutex>

#include <torch/extension.h>
#include <c10/xpu/XPUFunctions.h>
#include <sycl/sycl.hpp>


extern "C" {

//#define PY_SSIZE_T_CLEAN
//#include <Python.h>

//#include <torch/extension.h>
//#include <c10/xpu/XPUFunctions.h>
//#include <sycl/sycl.hpp>

// Error handling
char error_msg[10240];  // 10KB buffer to store error messages
int no_error = 0;
int error_code = no_error;  // store error code

#define XPU_CHECK(condition)                                                 \
  do {                                                                      \
    try {                                                                   \
      condition;                                                            \
      error_code = no_error;                                                \
    } catch (const sycl::exception& e) {                                    \
      error_code = -1;                                                      \
      snprintf(error_msg, sizeof(error_msg),                                \
               "XPU SYCL Error: %s at %s:%d", e.what(), __FILE__, __LINE__); \
      std::cerr << error_msg << std::endl;                                 \
    } catch (...) {                                                         \
      error_code = -1;                                                      \
      snprintf(error_msg, sizeof(error_msg),                                \
               "XPU Unknown Error at %s:%d", __FILE__, __LINE__);          \
      std::cerr << error_msg << std::endl;                                 \
    }                                                                       \
  } while (0)

// Global references to Python callables
// NOTE: this is borrowed reference, so we don't need to DECREF them.
// This brings the limitation that the allocator needs to be singleton.
static PyObject* g_python_malloc_callback = nullptr;
static PyObject* g_python_free_callback = nullptr;

// Memory metadata storage
// For XPU, we use a simpler approach than CUDA's handle system
// We store metadata (device, size) associated with each pointer
struct MemoryMetadata {
  int device;
  size_t size;
  void* ptr;
};

// Thread-safe map to store memory metadata
static std::unordered_map<void*, MemoryMetadata> g_memory_map;
static std::mutex g_memory_map_mutex;

// ---------------------------------------------------------------------------
// Helper functions:

void ensure_device(int device) {
  // Ensure the device context is set
  c10::xpu::check_device_index(device);
  c10::xpu::set_device(device);
}

void* create_and_allocate(int device, size_t size) {
  std::cout << "use device:" << device << " size:" << size << std::endl;
  ensure_device(device);
  
  auto& sycl_device = c10::xpu::get_raw_device(device);
  auto& sycl_context = c10::xpu::get_device_context();
  
  void* ptr = nullptr;
  XPU_CHECK(ptr = sycl::malloc_shared(size, sycl_device, sycl_context));
  
  if (error_code != 0 || ptr == nullptr) {
    std::cout << "failed to allocate" << std::endl;
    return nullptr;
  }
  
  // Store metadata
  {
    std::lock_guard<std::mutex> lock(g_memory_map_mutex);
    g_memory_map[ptr] = {device, size, ptr};
  }
  
  return ptr;
 }

void unmap_and_release(int device, size_t size, void* ptr) {
  ensure_device(device);
  
  // Free the memory
  XPU_CHECK(sycl::free(ptr, c10::xpu::get_device_context()));
  std::cout << "!!!freed" << std::endl; 
  // Remove from metadata map
  {
    std::lock_guard<std::mutex> lock(g_memory_map_mutex);
    g_memory_map.erase(ptr);
  }
  
  if (error_code != 0) {
    return;
  }
}

PyObject* create_tuple_from_c_integers(unsigned long long a,
                                       unsigned long long b,
                                       unsigned long long c,
                                       unsigned long long d) {
  // Create a new tuple of size 4
  PyObject* tuple = PyTuple_New(4);
  if (!tuple) {
    return NULL;  // Return NULL on failure
  }

  // Convert integers to Python objects and set them in the tuple
  PyTuple_SetItem(
      tuple, 0,
      PyLong_FromUnsignedLongLong(a));  // Steals reference to the PyLong
  PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLongLong(b));
  PyTuple_SetItem(tuple, 2, PyLong_FromUnsignedLongLong(c));
  PyTuple_SetItem(tuple, 3, PyLong_FromUnsignedLongLong(d));

  // Note: PyTuple_SetItem "steals" a reference to each object,
  // so we do not need to Py_DECREF the PyLong objects explicitly.

  return tuple;  // Return the created tuple
}

// ---------------------------------------------------------------------------
// Our exported C functions that call Python:

// Use sycl::queue* instead of CUstream
void* my_malloc(ssize_t size, int device, sycl::queue* queue) {
  ensure_device(device);

  // For XPU, we use a simpler allocation model than CUDA
  // No need for address reservation or granularity alignment like CUDA
  // SYCL USM handles this automatically
  
  if (!g_python_malloc_callback) {
    std::cerr << "ERROR: g_python_malloc_callback not set.\n";
    return nullptr;
  }

  // Allocate the actual memory first (before acquiring GIL for better performance)
  void* ptr = create_and_allocate(device, size);
  
  if (ptr == nullptr) {
    return nullptr;
  }

  // Acquire GIL for Python callback
  PyGILState_STATE gstate = PyGILState_Ensure();

  // For XPU, we pass: device, size, ptr (as unsigned long long), and a dummy handle (0)
  // The handle is 0 for XPU since we don't use CUDA-style handles
  PyObject* arg_tuple = create_tuple_from_c_integers(
      (unsigned long long)device, (unsigned long long)size,
      reinterpret_cast<unsigned long long>(ptr), (unsigned long long)0);

  // Call g_python_malloc_callback
  PyObject* py_result =
      PyObject_CallFunctionObjArgs(g_python_malloc_callback, arg_tuple, NULL);
  Py_DECREF(arg_tuple);
  std::cout << "!!!calling my_alloc, py_result:" << py_result << std::endl;
  if (!py_result) {
    PyErr_Print();
    PyGILState_Release(gstate);
    // Clean up allocated memory on error
    unmap_and_release(device, size, ptr);
    return nullptr;
  }

  // py_result might be None or a tuple, we don't need to check it for XPU
  Py_XDECREF(py_result);
  PyGILState_Release(gstate);

  return ptr;
}

// Use sycl::queue* instead of CUstream
void my_free(void* ptr, ssize_t size, int device, sycl::queue* queue) {
  // Get memory metadata from the pointer
  if (!g_python_free_callback) {
    std::cerr << "ERROR: g_python_free_callback not set.\n";
    return;
  }

  // Get metadata
  MemoryMetadata metadata;
  {
    std::lock_guard<std::mutex> lock(g_memory_map_mutex);
    auto it = g_memory_map.find(ptr);
    if (it == g_memory_map.end()) {
      std::cerr << "ERROR: Pointer not found in memory map.\n";
      return;
    }
    metadata = it->second;
  }

  // Acquire GIL
  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject* py_ptr =
      PyLong_FromUnsignedLongLong(reinterpret_cast<unsigned long long>(ptr));

  PyObject* py_result =
      PyObject_CallFunctionObjArgs(g_python_free_callback, py_ptr, NULL);
  Py_DECREF(py_ptr);

  if (!py_result || !PyTuple_Check(py_result) || PyTuple_Size(py_result) != 4) {
    PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 4");
    PyGILState_Release(gstate);
    return;
  }

  unsigned long long recv_device, recv_size;
  unsigned long long recv_ptr, recv_handle;
  // Unpack the tuple into four C integers
  if (!PyArg_ParseTuple(py_result, "KKKK", &recv_device, &recv_size,
                        &recv_ptr, &recv_handle)) {
    // PyArg_ParseTuple sets an error if it fails
    PyGILState_Release(gstate);
    return;
  }

  PyGILState_Release(gstate);

  // Verify the received data matches
  if (recv_ptr != reinterpret_cast<unsigned long long>(ptr) ||
      recv_device != static_cast<unsigned long long>(metadata.device) ||
      recv_size != static_cast<unsigned long long>(metadata.size)) {
    std::cerr << "WARNING: Mismatch in memory metadata.\n";
  }

  // Free memory
  unmap_and_release(metadata.device, metadata.size, ptr);
}

// ---------------------------------------------------------------------------
// Python extension boilerplate:

// Python-exposed function: init_module(python_malloc, python_free)
static PyObject* py_init_module(PyObject* self, PyObject* args) {
  PyObject* malloc_callback = nullptr;
  PyObject* free_callback = nullptr;

  if (!PyArg_ParseTuple(args, "OO", &malloc_callback, &free_callback)) {
    return nullptr;
  }

  if (!PyCallable_Check(malloc_callback) || !PyCallable_Check(free_callback)) {
    PyErr_SetString(PyExc_TypeError, "Both arguments must be callables");
    return nullptr;
  }

  // Save the Python callables
  // This module does not handle GC of these objects, so they must be kept alive
  // outside of this module.
  g_python_malloc_callback = malloc_callback;
  g_python_free_callback = free_callback;

  Py_RETURN_NONE;
}

static PyObject* python_unmap_and_release(PyObject* self, PyObject* args) {
  if (!args || !PyTuple_Check(args) || PyTuple_Size(args) != 4) {
    PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 4");
    return nullptr;
  }

  unsigned long long recv_device, recv_size;
  unsigned long long recv_ptr, recv_handle;
  // Unpack the tuple into four C integers
  if (!PyArg_ParseTuple(args, "KKKK", &recv_device, &recv_size, &recv_ptr,
                        &recv_handle)) {
    // PyArg_ParseTuple sets an error if it fails
    return nullptr;
  }

  void* ptr = reinterpret_cast<void*>(recv_ptr);
  unmap_and_release(static_cast<int>(recv_device), recv_size, ptr);

  if (error_code != 0) {
    error_code = no_error;
    PyErr_SetString(PyExc_RuntimeError, error_msg);
    return nullptr;
  }

  Py_RETURN_NONE;
}

static PyObject* python_create_and_allocate(PyObject* self, PyObject* args) {
  if (!args || !PyTuple_Check(args) || PyTuple_Size(args) != 4) {
    PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 4");
    return nullptr;
  }

  unsigned long long recv_device, recv_size;
  unsigned long long recv_ptr, recv_handle;
  // Unpack the tuple into four C integers
  if (!PyArg_ParseTuple(args, "KKKK", &recv_device, &recv_size, &recv_ptr,
                        &recv_handle)) {
    // PyArg_ParseTuple sets an error if it fails
    return nullptr;
  }

  void* ptr = create_and_allocate(static_cast<int>(recv_device), recv_size);
  std::cout << "calling python_create_and_allocate" << std::endl;
  if (error_code != 0 || ptr == nullptr) {
    error_code = no_error;
    PyErr_SetString(PyExc_RuntimeError, error_msg);
    return nullptr;
  }

  Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"init_module", (PyCFunction)py_init_module, METH_VARARGS,
     "Initialize module with python_malloc and python_free callables."},
    {"python_create_and_allocate", (PyCFunction)python_create_and_allocate, METH_VARARGS,
     "Create and allocate memory on the XPU device."},
    {"python_unmap_and_release", (PyCFunction)python_unmap_and_release,
     METH_VARARGS, "Unmap and release memory on the XPU device."},
    {NULL, NULL, 0, NULL}  // sentinel
};

static struct PyModuleDef xpumem_allocator_module = {
    PyModuleDef_HEAD_INIT, "xpumem_allocator",
    "SYCL USM-based allocator for XPUPluggableAllocator", -1, module_methods};

PyMODINIT_FUNC PyInit_xpumem_allocator(void) {
  // Initialize the module
  PyObject* module = PyModule_Create(&xpumem_allocator_module);
  if (!module) {
    return NULL;
  }
  return module;
}
}  // extern "C"
