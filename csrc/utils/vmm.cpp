// Virtual memory management (VMM) on PyTorch's SYCL context, for device
// buffers that grow in place, such as vLLM's extensible KV cache. Memory mapped
// here is usable by every kernel on PyTorch's queues. Mirrors the CUDA cuMem*
// API: pointers and sizes are passed as int64, and physical memory as an opaque
// int64 handle.

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <c10/xpu/XPUFunctions.h>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "ops.h"

namespace syclex = sycl::ext::oneapi::experimental;

namespace {

#define VMM_SYCL_CALL(call)                          \
  try {                                              \
    call;                                            \
  } catch (const sycl::exception& e) {               \
    TORCH_CHECK(false, #call " failed: ", e.what()); \
  }

// Physical memory objects keyed by the handle returned to Python; dropping the
// last reference releases the memory.
std::mutex g_handles_mutex;
std::unordered_map<int64_t, syclex::physical_mem> g_handles;
int64_t g_next_handle = 1;

const sycl::context& vmm_context() { return c10::xpu::get_device_context(); }

sycl::device& vmm_device(int64_t device_index) {
  c10::xpu::check_device_index(static_cast<c10::DeviceIndex>(device_index));
  return c10::xpu::get_raw_device(static_cast<c10::DeviceIndex>(device_index));
}

syclex::physical_mem take_handle(int64_t handle) {
  std::lock_guard<std::mutex> lock(g_handles_mutex);
  auto it = g_handles.find(handle);
  TORCH_CHECK(it != g_handles.end(), "Unknown VMM handle ", handle, ".");
  syclex::physical_mem mem = std::move(it->second);
  g_handles.erase(it);
  return mem;
}

syclex::physical_mem get_handle(int64_t handle) {
  std::lock_guard<std::mutex> lock(g_handles_mutex);
  auto it = g_handles.find(handle);
  TORCH_CHECK(it != g_handles.end(), "Unknown VMM handle ", handle, ".");
  return it->second;
}

}  // namespace

int64_t vmm_granularity(int64_t device_index) {
  auto& device = vmm_device(device_index);
  size_t granularity = 0;
  VMM_SYCL_CALL(
      granularity = syclex::get_mem_granularity(
          device, vmm_context(), syclex::granularity_mode::minimum));
  return static_cast<int64_t>(granularity);
}

int64_t vmm_reserve(int64_t size) {
  TORCH_CHECK(size > 0, "VMM reservation size must be positive.");
  uintptr_t ptr = 0;
  VMM_SYCL_CALL(
      ptr = syclex::reserve_virtual_mem(
          static_cast<size_t>(size), vmm_context()));
  return static_cast<int64_t>(ptr);
}

void vmm_free_reserved(int64_t ptr, int64_t size) {
  VMM_SYCL_CALL(
      syclex::free_virtual_mem(
          static_cast<uintptr_t>(ptr),
          static_cast<size_t>(size),
          vmm_context()));
}

int64_t vmm_create(int64_t size, int64_t device_index) {
  TORCH_CHECK(size > 0, "VMM physical memory size must be positive.");
  auto& device = vmm_device(device_index);
  std::optional<syclex::physical_mem> mem;
  VMM_SYCL_CALL(mem.emplace(device, vmm_context(), static_cast<size_t>(size)));
  std::lock_guard<std::mutex> lock(g_handles_mutex);
  int64_t handle = g_next_handle++;
  g_handles.emplace(handle, std::move(*mem));
  return handle;
}

void vmm_map(int64_t ptr, int64_t size, int64_t handle) {
  syclex::physical_mem mem = get_handle(handle);
  VMM_SYCL_CALL(mem.map(
      static_cast<uintptr_t>(ptr),
      static_cast<size_t>(size),
      syclex::address_access_mode::read_write));
}

void vmm_set_access(int64_t ptr, int64_t size, int64_t device_index) {
  // Access applies to the whole context, which spans every device.
  vmm_device(device_index);
  VMM_SYCL_CALL(
      syclex::set_access_mode(
          reinterpret_cast<const void*>(ptr),
          static_cast<size_t>(size),
          syclex::address_access_mode::read_write,
          vmm_context()));
}

void vmm_unmap(int64_t ptr, int64_t size) {
  VMM_SYCL_CALL(
      syclex::unmap(
          reinterpret_cast<const void*>(ptr),
          static_cast<size_t>(size),
          vmm_context()));
}

void vmm_release(int64_t handle) { take_handle(handle); }

// Wraps mapped memory as a uint8 tensor without taking ownership. The device is
// given explicitly: PyTorch cannot infer it from a VMM address, which is not a
// USM allocation.
torch::Tensor
vmm_tensor_from_ptr(int64_t ptr, int64_t num_bytes, int64_t device_index) {
  TORCH_CHECK(num_bytes >= 0, "num_bytes must be non-negative.");
  vmm_device(device_index);
  const at::Device device(
      at::kXPU, static_cast<c10::DeviceIndex>(device_index));
  return at::for_blob(reinterpret_cast<void*>(ptr), {num_bytes})
      .options(at::TensorOptions().dtype(at::kByte).device(device))
      .target_device(device)
      .deleter([](void*) {})
      .make_tensor();
}
