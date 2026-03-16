// An XPUPluggableAllocator based on SYCL USM APIs.
// This provides similar functionality to cu_mem.cpp but for Intel XPU devices.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cctype>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <optional>
#include <sstream>

#include <torch/extension.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

extern "C" {

// Error handling
char error_msg[10240];  // 10KB buffer to store error messages
int no_error = 0;
int error_code = no_error;  // store error code

#define XPU_CHECK(condition)               \
  do {                                     \
    try {                                  \
      condition;                           \
      error_code = no_error;               \
    } catch (const sycl::exception& e) {   \
      error_code = -1;                     \
      snprintf(                            \
          error_msg,                       \
          sizeof(error_msg),               \
          "XPU SYCL Error: %s at %s:%d",   \
          e.what(),                        \
          __FILE__,                        \
          __LINE__);                       \
      std::cerr << error_msg << std::endl; \
    } catch (...) {                        \
      error_code = -1;                     \
      snprintf(                            \
          error_msg,                       \
          sizeof(error_msg),               \
          "XPU Unknown Error at %s:%d",    \
          __FILE__,                        \
          __LINE__);                       \
      std::cerr << error_msg << std::endl; \
    }                                      \
  } while (0)

// Global references to Python callables.
static PyObject* g_python_malloc_callback = nullptr;
static PyObject* g_python_free_callback = nullptr;

static bool parse_debug_env_enabled() {
  const char* env = std::getenv("XPUMEM_DEBUG_LOG_METADATA");
  if (env == nullptr) {
    return false;
  }
  std::string v(env);
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return v == "1" || v == "true" || v == "on" || v == "yes";
}

static std::atomic<bool> g_debug_logging{parse_debug_env_enabled()};

struct MemoryMetadata {
  int device;
  size_t size;
  void* ptr;
  bool mapped;
  std::optional<sycl::ext::oneapi::experimental::physical_mem> physical_mem;

  std::string toStr() const {
    std::ostringstream os;
    os << "{ptr=" << ptr << ", device=" << device << ", size=" << size
       << ", mapped=" << (mapped ? 1 : 0)
       << ", has_physical=" << (physical_mem.has_value() ? 1 : 0) << "}";
    return os.str();
  }
};

static std::unordered_map<void*, MemoryMetadata> g_memory_map;
static std::mutex g_memory_map_mutex;

void log_metadata_lifecycle(const char* stage, const MemoryMetadata& metadata) {
  if (!g_debug_logging.load(std::memory_order_relaxed)) {
    return;
  }
  std::cout << "[xpumem] " << stage << " " << metadata.toStr() << std::endl;
}

void ensure_device(int device) {
  c10::xpu::check_device_index(device);
  c10::xpu::set_device(device);
}

void map_physical_memory(MemoryMetadata& metadata) {
  if (metadata.mapped) {
    log_metadata_lifecycle("map skipped(already mapped)", metadata);
    return;
  }

  log_metadata_lifecycle("map begin", metadata);

  auto sycl_device = c10::xpu::get_raw_device(metadata.device);
  auto sycl_context = c10::xpu::get_device_context();

  metadata.physical_mem.emplace(sycl_device, sycl_context, metadata.size);
  metadata.physical_mem->map(
      reinterpret_cast<uintptr_t>(metadata.ptr),
      metadata.size,
      sycl::ext::oneapi::experimental::address_access_mode::read_write);
  metadata.mapped = true;
  log_metadata_lifecycle("map end", metadata);
}

void unmap_physical_memory(MemoryMetadata& metadata) {
  if (!metadata.mapped) {
    log_metadata_lifecycle("unmap skipped(already unmapped)", metadata);
    return;
  }

  log_metadata_lifecycle("unmap begin", metadata);

  auto sycl_context = c10::xpu::get_device_context();
  sycl::ext::oneapi::experimental::unmap(
      metadata.ptr, metadata.size, sycl_context);
  metadata.physical_mem.reset();
  metadata.mapped = false;
  log_metadata_lifecycle("unmap end", metadata);
}

void free_virtual_address(const MemoryMetadata& metadata) {
  log_metadata_lifecycle("free_virtual_address begin", metadata);
  auto sycl_context = c10::xpu::get_device_context();
  sycl::ext::oneapi::experimental::free_virtual_mem(
      reinterpret_cast<uintptr_t>(metadata.ptr), metadata.size, sycl_context);
  log_metadata_lifecycle("free_virtual_address end", metadata);
}

// Reserve virtual address space and map physical memory, then track metadata.
void* reserve_and_map_new(int device, size_t size) {
  ensure_device(device);

  auto sycl_context = c10::xpu::get_device_context();
  uintptr_t va = 0;
  XPU_CHECK(
      va = sycl::ext::oneapi::experimental::reserve_virtual_mem(
          size, sycl_context));
  if (error_code != 0 || va == 0) {
    return nullptr;
  }

  MemoryMetadata metadata = {
      device,
      size,
      reinterpret_cast<void*>(va),
      false,
      std::nullopt,
  };
  log_metadata_lifecycle("reserve_and_map initialized", metadata);

  XPU_CHECK(map_physical_memory(metadata));
  if (error_code != 0 || !metadata.mapped) {
    XPU_CHECK(free_virtual_address(metadata));
    return nullptr;
  }

  {
    std::lock_guard<std::mutex> lock(g_memory_map_mutex);
    g_memory_map[metadata.ptr] = std::move(metadata);
    log_metadata_lifecycle(
        "reserve_and_map inserted", g_memory_map[reinterpret_cast<void*>(va)]);
  }

  return reinterpret_cast<void*>(va);
}

// Sleep path: unmap physical memory while keeping VA reservation and metadata.
void unmap_for_sleep(void* ptr) {
  std::lock_guard<std::mutex> lock(g_memory_map_mutex);
  auto it = g_memory_map.find(ptr);
  if (it == g_memory_map.end()) {
    error_code = -1;
    snprintf(
        error_msg,
        sizeof(error_msg),
        "XPU Error: pointer not found in memory map for sleep unmap");
    return;
  }

  ensure_device(it->second.device);
  log_metadata_lifecycle("sleep before unmap", it->second);
  c10::xpu::syncStreamsOnDevice(it->second.device);
  XPU_CHECK(unmap_physical_memory(it->second));
  log_metadata_lifecycle("sleep after unmap", it->second);
}

// Wake path: remap physical memory to a previously reserved VA.
void remap_for_wakeup(void* ptr) {
  std::lock_guard<std::mutex> lock(g_memory_map_mutex);
  auto it = g_memory_map.find(ptr);
  if (it == g_memory_map.end()) {
    error_code = -1;
    snprintf(
        error_msg,
        sizeof(error_msg),
        "XPU Error: pointer not found in memory map for wake remap");
    return;
  }

  ensure_device(it->second.device);
  log_metadata_lifecycle("wake before map", it->second);
  XPU_CHECK(map_physical_memory(it->second));
  log_metadata_lifecycle("wake after map", it->second);
}

// Fully release an allocation: unmap physical memory and free virtual address.
void release_allocation(void* ptr) {
  MemoryMetadata metadata;
  {
    std::lock_guard<std::mutex> lock(g_memory_map_mutex);
    auto it = g_memory_map.find(ptr);
    if (it == g_memory_map.end()) {
      error_code = -1;
      snprintf(
          error_msg,
          sizeof(error_msg),
          "XPU Error: pointer not found in memory map");
      return;
    }
    log_metadata_lifecycle("release erase-from-map", it->second);
    metadata = std::move(it->second);
    g_memory_map.erase(it);
  }

  ensure_device(metadata.device);
  log_metadata_lifecycle("release before unmap", metadata);
  XPU_CHECK(unmap_physical_memory(metadata));
  if (error_code != 0) {
    return;
  }

  XPU_CHECK(free_virtual_address(metadata));
  log_metadata_lifecycle("release done", metadata);
}

PyObject* create_tuple_from_c_integers(
    unsigned long long a,
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
      tuple,
      0,
      PyLong_FromUnsignedLongLong(a));  // Steals reference to the PyLong
  PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLongLong(b));
  PyTuple_SetItem(tuple, 2, PyLong_FromUnsignedLongLong(c));
  PyTuple_SetItem(tuple, 3, PyLong_FromUnsignedLongLong(d));

  return tuple;
}

// Our exported C functions that call Python:

// Allocate XPU memory and notify Python allocator callback with allocation
// tuple.
void* my_malloc(ssize_t size, int device, sycl::queue* queue) {
  (void)queue;
  ensure_device(device);

  if (!g_python_malloc_callback) {
    std::cerr << "ERROR: g_python_malloc_callback not set.\n";
    return nullptr;
  }

  // Reserve VA and map physical memory first.
  void* ptr = reserve_and_map_new(device, size);

  if (ptr == nullptr) {
    return nullptr;
  }

  // Acquire GIL for Python callback
  PyGILState_STATE gstate = PyGILState_Ensure();

  // Pass a single handle tuple to keep CUDA cumem allocator parity
  PyObject* arg_tuple = create_tuple_from_c_integers(
      (unsigned long long)device,
      (unsigned long long)size,
      reinterpret_cast<unsigned long long>(ptr),
      (unsigned long long)0);

  PyObject* py_result =
      PyObject_CallFunctionObjArgs(g_python_malloc_callback, arg_tuple, NULL);
  Py_DECREF(arg_tuple);
  if (!py_result) {
    PyGILState_Release(gstate);
    // Clean up allocated memory on error
    release_allocation(ptr);
    return nullptr;
  }

  // py_result might be None or a tuple, we don't need to check it for XPU
  Py_XDECREF(py_result);
  PyGILState_Release(gstate);

  return ptr;
}

// Validate and release an allocation after Python callback confirms metadata.
void my_free(void* ptr, ssize_t size, int device, sycl::queue* queue) {
  (void)size;
  (void)device;
  (void)queue;

  if (!g_python_free_callback) {
    return;
  }

  int meta_device = 0;
  size_t meta_size = 0;
  {
    std::lock_guard<std::mutex> lock(g_memory_map_mutex);
    auto it = g_memory_map.find(ptr);
    if (it == g_memory_map.end()) {
      return;
    }
    meta_device = it->second.device;
    meta_size = it->second.size;
  }

  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject* py_ptr =
      PyLong_FromUnsignedLongLong(reinterpret_cast<unsigned long long>(ptr));

  PyObject* py_result =
      PyObject_CallFunctionObjArgs(g_python_free_callback, py_ptr, NULL);
  Py_DECREF(py_ptr);

  if (!py_result) {
    PyGILState_Release(gstate);
    return;
  }

  if (!PyTuple_Check(py_result) || PyTuple_Size(py_result) != 4) {
    PyErr_SetString(
        PyExc_TypeError, "Expected python_free to return a tuple of size 4");
    Py_XDECREF(py_result);
    PyGILState_Release(gstate);
    return;
  }

  unsigned long long recv_device, recv_size;
  unsigned long long recv_ptr, recv_handle;
  // Unpack the tuple into four C integers
  if (!PyArg_ParseTuple(
          py_result,
          "KKKK",
          &recv_device,
          &recv_size,
          &recv_ptr,
          &recv_handle)) {
    Py_DECREF(py_result);
    PyGILState_Release(gstate);
    return;
  }

  Py_DECREF(py_result);
  PyGILState_Release(gstate);

  if (recv_ptr != reinterpret_cast<unsigned long long>(ptr) ||
      recv_device != static_cast<unsigned long long>(meta_device) ||
      recv_size != static_cast<unsigned long long>(meta_size)) {
    return;
  }

  release_allocation(ptr);
}

// Python-exposed function: init_module(python_malloc, python_free)
// Register Python malloc/free callbacks used by allocator entry points.
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

  // Save callbacks as borrowed references.
  g_python_malloc_callback = malloc_callback;
  g_python_free_callback = free_callback;
  Py_RETURN_NONE;
}

// Python sleep hook: unmap physical memory for an existing allocation tuple.
static PyObject* python_unmap_and_release(PyObject* self, PyObject* args) {
  if (!args || !PyTuple_Check(args) || PyTuple_Size(args) != 4) {
    PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 4");
    return nullptr;
  }

  unsigned long long recv_device, recv_size;
  unsigned long long recv_ptr, recv_handle;
  // Unpack the tuple into four C integers
  if (!PyArg_ParseTuple(
          args, "KKKK", &recv_device, &recv_size, &recv_ptr, &recv_handle)) {
    return nullptr;
  }

  void* ptr = reinterpret_cast<void*>(recv_ptr);
  // Sleep path: only unmap physical memory and keep reserved VA/metadata.
  (void)recv_device;
  (void)recv_size;
  (void)recv_handle;
  unmap_for_sleep(ptr);

  if (PyErr_Occurred()) {
    return nullptr;
  }

  if (error_code != 0) {
    error_code = no_error;
    PyErr_SetString(PyExc_RuntimeError, error_msg);
    return nullptr;
  }

  Py_RETURN_NONE;
}

// Python wake hook: remap physical memory for a previously reserved allocation.
static PyObject* python_create_and_allocate(PyObject* self, PyObject* args) {
  if (!args || !PyTuple_Check(args) || PyTuple_Size(args) != 4) {
    PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 4");
    return nullptr;
  }

  unsigned long long recv_device, recv_size;
  unsigned long long recv_ptr, recv_handle;
  if (!PyArg_ParseTuple(
          args, "KKKK", &recv_device, &recv_size, &recv_ptr, &recv_handle)) {
    return nullptr;
  }

  (void)recv_device;
  (void)recv_size;
  (void)recv_handle;
  remap_for_wakeup(reinterpret_cast<void*>(recv_ptr));

  if (error_code != 0) {
    error_code = no_error;
    PyErr_SetString(PyExc_RuntimeError, error_msg);
    return nullptr;
  }

  Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"init_module",
     (PyCFunction)py_init_module,
     METH_VARARGS,
     "Initialize module with python_malloc and python_free callables."},
    {"python_create_and_allocate",
     (PyCFunction)python_create_and_allocate,
     METH_VARARGS,
     "Create and allocate memory on the XPU device."},
    {"python_unmap_and_release",
     (PyCFunction)python_unmap_and_release,
     METH_VARARGS,
     "Unmap and release memory on the XPU device."},
    {NULL, NULL, 0, NULL}  // sentinel
};

static struct PyModuleDef xpumem_allocator_module = {
    PyModuleDef_HEAD_INIT,
    "xpumem_allocator",
    "SYCL USM-based allocator for XPUPluggableAllocator",
    -1,
    module_methods};

PyMODINIT_FUNC PyInit_xpumem_allocator(void) {
  // Initialize the module
  PyObject* module = PyModule_Create(&xpumem_allocator_module);
  if (!module) {
    return NULL;
  }
  return module;
}
}  // extern "C"
