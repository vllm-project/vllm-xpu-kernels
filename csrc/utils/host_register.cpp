#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

#include "ops.h"
#include "utils.h"

namespace vllm {
namespace xpu {

namespace {

// Page-locks an existing host allocation and imports it into the current
// device's SYCL context, the SYCL counterpart of cudaHostRegister. Host
// memory that cannot be obtained from the caching host allocator -- notably
// a shared mmap region -- is otherwise pageable, which forces staged (H2D)
// or synchronous (D2H) copies. Registration is scoped to the device's
// context, so each device that transfers to or from the range must
// register it separately. Returns false if registration is unsupported or
// failed, in which case transfers remain correct but slower.
bool hostRegister(void* ptr, size_t n_bytes) {
  if (ptr == nullptr || n_bytes == 0) return false;

  auto ctx = vllm::xpu::vllmGetQueue().get_context();
  try {
    syclex::prepare_for_device_copy(ptr, n_bytes, ctx);
  } catch (const sycl::exception& e) {
    TORCH_WARN(
        "prepare_for_device_copy failed (",
        e.what(),
        "); transfers stay correct but fall back to pageable host memory");
    return false;
  }
  return true;
}

// Releases a host range previously passed to hostRegister.
bool hostUnregister(void* ptr) {
  if (ptr == nullptr) return false;

  auto ctx = vllm::xpu::vllmGetQueue().get_context();
  try {
    syclex::release_from_device_copy(ptr, ctx);
  } catch (const sycl::exception& e) {
    TORCH_WARN("release_from_device_copy failed (", e.what(), ")");
    return false;
  }
  return true;
}

}  // namespace

}  // namespace xpu
}  // namespace vllm

namespace {

// Extracts the raw address from a single-element uint64 tensor. Addresses
// are passed this way (rather than as a scalar int64) because a pointer's
// full 64-bit bit pattern (e.g. bit 63 set) can exceed the signed int64
// range and would otherwise fail to cast from Python.
uint64_t ptrFromTensor(const torch::Tensor& ptr) {
  // Must live on the CPU: the address is dereferenced with data_ptr<uint64_t>
  // on the host, so a device-resident tensor here would be undefined
  // behavior rather than a clean error.
  TORCH_CHECK(ptr.device().is_cpu(), "ptr must be on CPU");
  TORCH_CHECK(
      ptr.numel() == 1,
      "ptr must be a single-element tensor, got ",
      ptr.numel());
  TORCH_CHECK(
      ptr.scalar_type() == torch::kUInt64,
      "ptr must be a uint64 tensor, got ",
      ptr.scalar_type());
  return *ptr.data_ptr<uint64_t>();
}

}  // namespace

bool xpu_host_register(const torch::Tensor& ptr, int64_t n_bytes) {
  TORCH_CHECK(n_bytes >= 0, "n_bytes must be non-negative");
  uint64_t addr = ptrFromTensor(ptr);
  if (addr == 0 || n_bytes == 0) return false;
  return vllm::xpu::hostRegister(
      reinterpret_cast<void*>(static_cast<uintptr_t>(addr)),
      static_cast<size_t>(n_bytes));
}

bool xpu_host_unregister(const torch::Tensor& ptr) {
  uint64_t addr = ptrFromTensor(ptr);
  if (addr == 0) return false;
  return vllm::xpu::hostUnregister(
      reinterpret_cast<void*>(static_cast<uintptr_t>(addr)));
}
