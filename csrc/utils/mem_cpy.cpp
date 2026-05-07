#include <ATen/xpu/CachingHostAllocator.h>
#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ops.h"
#include "utils.h"
#include "utils/mem_cpy.h"

namespace vllm {
namespace xpu {

namespace {

inline void record_host_alloc_event_if_possible(void* ptr, const void* hctx) {
  // hctx is only valid when the host memory is actually allocated
  // by PyTorch’s XPU caching host allocator; otherwise,
  // calling recordEvent may be invalid.
  if (ptr != nullptr && hctx != nullptr) {
    at::getHostAllocator(at::kXPU)->record_event(
        ptr, const_cast<void*>(hctx), at::xpu::getCurrentXPUStream());
  }
}

inline void
memcpy_sync(sycl::queue& queue, void* dst, const void* src, size_t n_bytes) {
  queue.memcpy(dst, src, n_bytes).wait();
}

inline void
memcpy_async(sycl::queue& queue, void* dst, const void* src, size_t n_bytes) {
  queue.memcpy(dst, src, n_bytes);
}

inline void async_h2d_with_staging(
    sycl::queue& queue,
    void* dst_device,
    const void* src_host_pageable,
    size_t n_bytes) {
  auto staging = at::getHostAllocator(at::kXPU)->allocate(n_bytes);
  void* staging_ptr = staging.get();
  TORCH_CHECK(staging_ptr, "Failed to allocate pinned host memory for staging");

  std::memcpy(staging_ptr, src_host_pageable, n_bytes);

  memcpy_async(queue, dst_device, staging_ptr, n_bytes);

  // The staging buffer is managed by the allocator,
  // so record the event on it to ensure the staging buffer remains alive
  // until the DMA transfer completes.
  record_host_alloc_event_if_possible(staging_ptr, staging.get_context());
}

}  // namespace

void memcpyHostToDevice(
    void* dst,
    const void* src,
    size_t n_bytes,
    bool async,
    const void* hctx,
    bool is_pinned) {
  if (n_bytes == 0) return;

  auto& queue = vllm::xpu::vllmGetQueue();

  if (!async) {
    memcpy_sync(queue, dst, src, n_bytes);
    return;
  }

  if (is_pinned) {
    // Pinned host → device: can be transferred via direct asynchronous DMA
    memcpy_async(queue, dst, src, n_bytes);
    // Only record the event if src is pinned memory allocated
    // by the caching host allocator
    record_host_alloc_event_if_possible(const_cast<void*>(src), hctx);
    return;
  }

  // Pageable host → device: requires a staging buffer.
  async_h2d_with_staging(queue, dst, src, n_bytes);
}

void memcpyDeviceToHost(
    void* dst,
    const void* src,
    size_t n_bytes,
    bool async,
    const void* hctx,
    bool is_pinned) {
  if (n_bytes == 0) return;

  auto& queue = vllm::xpu::vllmGetQueue();

  if (!async) {
    memcpy_sync(queue, dst, src, n_bytes);
    return;
  }

  if (is_pinned) {
    // Device → pinned host: supports asynchronous copies;
    // use recordEvent to defer reuse until the copy completes
    memcpy_async(queue, dst, src, n_bytes);
    record_host_alloc_event_if_possible(dst, hctx);
    return;
  }

  // Device → pageable host: asynchronous copies are unsafe
  // unless you introduce staging plus a completion callback.
  // Async device-to-host copy to non-pinned memory is unsafe;
  // falling back to synchronous to avoid data corruption.");
  memcpy_sync(queue, dst, src, n_bytes);
}

void memcpyDeviceToDevice(
    void* dst, const void* src, size_t n_bytes, bool async) {
  if (n_bytes == 0) return;

  auto& queue = vllm::xpu::vllmGetQueue();
  if (!async) {
    memcpy_sync(queue, dst, src, n_bytes);
    return;
  }
  memcpy_async(queue, dst, src, n_bytes);
}

// The caller must specify whether the host pointer is pinned,
// and provide hctx if the pointer comes from the caching host allocator.
void xpuAsyncMemcpy(
    void* dst,
    const void* src,
    size_t n_bytes,
    xpuMemcpyKind kind,
    const void* hctx,
    bool is_pinned) {
  switch (kind) {
    case HostToDevice:
      memcpyHostToDevice(dst, src, n_bytes, /*async=*/true, hctx, is_pinned);
      break;
    case DeviceToHost:
      memcpyDeviceToHost(dst, src, n_bytes, /*async=*/true, hctx, is_pinned);
      break;
    case DeviceToDevice:
      memcpyDeviceToDevice(dst, src, n_bytes, /*async=*/true);
      break;
    default:
      TORCH_CHECK(false, "Unknown dpcpp memory kind");
  }
}

// Infer which XPU device a USM device pointer was allocated on by probing
// each device's SYCL context.  Returns the device index on success.
// This is O(num_xpu_devices) but avoids threading an explicit device argument
// through the entire call chain when all callers already have the pointer.
static at::DeviceIndex infer_xpu_device_from_ptr(const void* device_ptr) {
  const int n_devs = c10::xpu::device_count();
  for (int i = 0; i < n_devs; i++) {
    auto ctx = vllm::xpu::vllmGetQueue(i).get_context();
    auto type = sycl::get_pointer_type(device_ptr, ctx);
    if (type == sycl::usm::alloc::device || type == sycl::usm::alloc::shared) {
      return static_cast<at::DeviceIndex>(i);
    }
  }
  TORCH_CHECK(false, "Cannot determine XPU device from pointer");
  return -1;
}

void xpuAsyncMemcpyBatch(
    const uint64_t* src_ptrs,
    const uint64_t* dst_ptrs,
    const uint64_t* sizes,
    int64_t n) {
  if (n == 0) return;

  // Scan the first non-zero entry to determine copy direction.
  // Also capture the device-side pointer so we can infer which XPU to use.
  const void* device_probe = nullptr;
  bool needs_staging = false;
  bool dst_is_pageable = false;  // D2H to pageable host -> sync copy
  for (int64_t i = 0; i < n; i++) {
    if (sizes[i] == 0) continue;
    const void* first_src = reinterpret_cast<const void*>(src_ptrs[i]);
    const void* first_dst = reinterpret_cast<const void*>(dst_ptrs[i]);

    // Use device 0's context as a probe: we only need pointer *type* here,
    // and USM pointer types are consistent across all devices on the same
    // platform (host/unknown are always host; device is always device on its
    // own platform).  The actual device index is resolved below via
    // infer_xpu_device_from_ptr().
    auto probe_ctx = vllm::xpu::vllmGetQueue(0).get_context();
    auto src_type = sycl::get_pointer_type(first_src, probe_ctx);
    auto dst_type = sycl::get_pointer_type(first_dst, probe_ctx);
    bool src_is_host =
        (src_type == sycl::usm::alloc::host ||
         src_type == sycl::usm::alloc::unknown);
    bool dst_is_device = (dst_type == sycl::usm::alloc::device);
    needs_staging = src_is_host && dst_is_device;
    // D2H to pageable host requires synchronous copy to avoid corruption.
    dst_is_pageable = !dst_is_device && (dst_type == sycl::usm::alloc::unknown);
    // Device-side pointer: dst for H2D, src for D2H or D2D.
    device_probe = needs_staging ? first_dst : first_src;
    break;
  }

  if (device_probe == nullptr) return;  // all sizes are zero

  // Infer the target XPU device from the device pointer and set the guard so
  // that vllmGetQueue() returns the correct in-order queue.
  const at::DeviceIndex dev = infer_xpu_device_from_ptr(device_probe);
  const at::DeviceGuard device_guard(at::Device(at::kXPU, dev));

  auto& queue = vllm::xpu::vllmGetQueue();

  // Compute total bytes needed for the H2D staging buffer.
  uint64_t total_bytes = 0;
  for (int64_t i = 0; i < n; i++) {
    total_bytes += sizes[i];
  }

  if (needs_staging) {
    // H2D: allocate one contiguous pinned staging buffer, snapshot all source
    // blocks, then submit all async DMAs.  This avoids N separate allocator
    // round-trips and protects against caller mutation after return.
    auto staging = at::getHostAllocator(at::kXPU)->allocate(
        static_cast<size_t>(total_bytes));
    char* staging_ptr = static_cast<char*>(staging.get());
    TORCH_CHECK(staging_ptr, "Failed to allocate pinned staging buffer");

    // Phase 1: snapshot all source blocks into staging (pure CPU work).
    size_t staging_offset = 0;
    for (int64_t i = 0; i < n; i++) {
      size_t sz = static_cast<size_t>(sizes[i]);
      if (sz == 0) continue;
      std::memcpy(
          staging_ptr + staging_offset,
          reinterpret_cast<const void*>(src_ptrs[i]),
          sz);
      staging_offset += sz;
    }

    // Phase 2: submit async DMA from staging to device in a tight loop,
    // maximising PCIe/copy-engine throughput without interleaved CPU work.
    staging_offset = 0;
    for (int64_t i = 0; i < n; i++) {
      size_t sz = static_cast<size_t>(sizes[i]);
      if (sz == 0) continue;
      queue.memcpy(
          reinterpret_cast<void*>(dst_ptrs[i]),
          staging_ptr + staging_offset,
          sz);
      staging_offset += sz;
    }

    // Keep the staging buffer alive until all submitted DMAs complete.
    if (staging.get_context() != nullptr) {
      at::getHostAllocator(at::kXPU)->record_event(
          staging_ptr,
          const_cast<void*>(staging.get_context()),
          at::xpu::getCurrentXPUStream());
    }
  } else {
    // D2H or D2D: dst_is_pageable was probed once from the first non-zero
    // entry (all entries share the same direction and memory class).
    // Pageable D2H is unsafe with async DMA; fall back to sync copy.
    for (int64_t i = 0; i < n; i++) {
      size_t sz = static_cast<size_t>(sizes[i]);
      if (sz == 0) continue;

      const void* src = reinterpret_cast<const void*>(src_ptrs[i]);
      void* dst = reinterpret_cast<void*>(dst_ptrs[i]);

      if (dst_is_pageable) {
        queue.memcpy(dst, src, sz).wait();
      } else {
        queue.memcpy(dst, src, sz);
      }
    }
  }
}

}  // namespace xpu
}  // namespace vllm

enum class MemcpyKind : int64_t {
  HostToDevice = 0,
  DeviceToHost = 1,
  DeviceToDevice = 2,
};

void xpu_memcpy_sync(
    int64_t dst_ptr,
    int64_t src_ptr,
    int64_t n_bytes,
    int64_t kind,
    int64_t device) {
  TORCH_CHECK(n_bytes >= 0, "n_bytes must be non-negative");
  if (n_bytes == 0) {
    return;
  }

  if (device >= 0) {
    c10::xpu::check_device_index(static_cast<int>(device));
    c10::xpu::set_device(static_cast<int>(device));
  }

  void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(dst_ptr));
  const void* src =
      reinterpret_cast<const void*>(static_cast<uintptr_t>(src_ptr));

  switch (static_cast<MemcpyKind>(kind)) {
    case MemcpyKind::HostToDevice:
      vllm::xpu::memcpyHostToDevice(
          dst,
          src,
          static_cast<size_t>(n_bytes),
          /*async=*/false,
          /*hctx=*/nullptr,
          /*is_pinned=*/false);
      break;
    case MemcpyKind::DeviceToHost:
      vllm::xpu::memcpyDeviceToHost(
          dst,
          src,
          static_cast<size_t>(n_bytes),
          /*async=*/false,
          /*hctx=*/nullptr,
          /*is_pinned=*/false);
      break;
    case MemcpyKind::DeviceToDevice:
      vllm::xpu::memcpyDeviceToDevice(
          dst, src, static_cast<size_t>(n_bytes), /*async=*/false);
      break;
    default:
      TORCH_CHECK(
          false, "Unsupported memcpy kind: ", kind, " (0=H2D, 1=D2H, 2=D2D)");
  }
}
