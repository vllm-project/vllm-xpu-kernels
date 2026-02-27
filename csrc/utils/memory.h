#pragma once

namespace vllm {
namespace xpu {

enum xpuMemcpyKind { HostToDevice, DeviceToHost, DeviceToDevice };

void xpuMemcpy(void* dst, const void* src, size_t n_bytes, xpuMemcpyKind kind);

}  // namespace xpu
}  // namespace vllm
