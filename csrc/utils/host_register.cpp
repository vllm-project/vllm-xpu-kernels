#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>

#include "ops.h"
#include "utils.h"
#include "utils/mem_cpy.h"

namespace vllm {
namespace xpu {

namespace {

// Ranges registered through hostRegister. SYCL reports an imported range as
// usm::alloc::unknown, so callers that must tell a registered (DMA-capable)
// range from pageable memory look it up here.
std::mutex& registeredMutex() {
  static std::mutex m;
  return m;
}
std::map<uintptr_t, size_t>& registeredRanges() {
  static std::map<uintptr_t, size_t> r;
  return r;
}

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
  {
    std::lock_guard<std::mutex> g(registeredMutex());
    registeredRanges()[reinterpret_cast<uintptr_t>(ptr)] = n_bytes;
  }
  return true;
}

// Releases a host range previously passed to hostRegister.
bool hostUnregister(void* ptr) {
  if (ptr == nullptr) return false;

  {
    std::lock_guard<std::mutex> g(registeredMutex());
    registeredRanges().erase(reinterpret_cast<uintptr_t>(ptr));
  }
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

bool isHostRegistered(const void* ptr, size_t n_bytes) {
  const auto p = reinterpret_cast<uintptr_t>(ptr);
  std::lock_guard<std::mutex> g(registeredMutex());
  auto& ranges = registeredRanges();
  auto it = ranges.upper_bound(p);
  if (it == ranges.begin()) return false;
  --it;
  return p >= it->first && p + n_bytes <= it->first + it->second;
}

}  // namespace xpu
}  // namespace vllm

bool xpu_host_register(int64_t ptr, int64_t n_bytes) {
  TORCH_CHECK(n_bytes >= 0, "n_bytes must be non-negative");
  if (ptr == 0 || n_bytes == 0) return false;
  return vllm::xpu::hostRegister(
      reinterpret_cast<void*>(static_cast<uintptr_t>(ptr)),
      static_cast<size_t>(n_bytes));
}

bool xpu_host_unregister(int64_t ptr) {
  if (ptr == 0) return false;
  return vllm::xpu::hostUnregister(
      reinterpret_cast<void*>(static_cast<uintptr_t>(ptr)));
}
