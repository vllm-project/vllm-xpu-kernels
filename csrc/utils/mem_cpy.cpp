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
    memcpy_sync(queue, dst, src, n_bytes);
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
