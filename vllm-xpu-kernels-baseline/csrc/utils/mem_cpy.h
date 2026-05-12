#pragma once
#include <cstddef>
#include <cstdint>

namespace vllm {
namespace xpu {

enum xpuMemcpyKind { HostToDevice, DeviceToHost, DeviceToDevice };

/**
 * @brief Performs asynchronous memory copy between host and XPU device.
 *
 * Handles data transfer in three directions: host-to-device, device-to-host,
 * and device-to-device. For host-involved transfers, supports both pinned
 * (page-locked) and pageable host memory through the caching host allocator.
 *
 * @param dst        Destination memory address (device or host pointer)
 * @param src        Source memory address (device or host pointer)
 * @param n_bytes    Number of bytes to copy
 * @param kind       Direction of memory transfer (HostToDevice, DeviceToHost,
 *                   or DeviceToDevice)
 * @param hctx       Host context pointer from caching host allocator. Required
 *                   when is_pinned is true for optimized DMA transfer; nullptr
 *                   for non-pinned memory or D2D transfers
 * @param is_pinned  Whether the host pointer is page-locked (pinned) memory.
 *                   Must be true for hctx to be valid
 */
void xpuAsyncMemcpy(
    void* dst,
    const void* src,
    size_t n_bytes,
    xpuMemcpyKind kind,
    const void* hctx,
    bool is_pinned);

/**
 * @brief Batch async memcpy: copies N independent (src, dst, size) triples
 *        in a single call, amortising per-copy overhead.
 *
 * The copy direction is auto-detected from the first non-zero entry's USM
 * pointer types.  All entries must share the same direction.
 *
 * For H2D: snapshots all source blocks through a single contiguous pinned
 *   staging buffer so the caller may safely mutate host memory immediately.
 * For D2H / D2D: direct async DMA without staging.
 *
 * @param src_ptrs   Array of N raw source addresses
 * @param dst_ptrs   Array of N raw destination addresses
 * @param sizes      Array of N byte counts
 * @param n          Number of entries
 */
void xpuAsyncMemcpyBatch(
    const uint64_t* src_ptrs,
    const uint64_t* dst_ptrs,
    const uint64_t* sizes,
    int64_t n);

}  // namespace xpu
}  // namespace vllm
