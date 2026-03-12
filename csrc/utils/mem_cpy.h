#pragma once
#include <cstddef>

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

}  // namespace xpu
}  // namespace vllm
