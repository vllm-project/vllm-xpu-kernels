#pragma once
#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

namespace vllm {
namespace xpu {
// ============================================================================
// Unified function to get max workgroup size
// - With KernelClass template: queries kernel-specific limit
// - Without template: queries device default limit
// ============================================================================
template <class KernelClass = void>
static int64_t getMaxWorkGroupSize(
    at::DeviceIndex dev_id = c10::xpu::current_device()) {
  auto q = c10::xpu::getCurrentXPUStream(dev_id).queue();
  auto ctx = q.get_context();
  auto dev = q.get_device();

  if constexpr (!std::is_void_v<KernelClass>) {
    // Kernel-specific version
    auto kid = sycl::get_kernel_id<KernelClass>();
    auto kbundle =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx, {kid});
    sycl::kernel k = kbundle.get_kernel(kid);
    return k.get_info<sycl::info::kernel_device_specific::work_group_size>(dev);
  } else {
    // Device default version
    auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
    return dev_prop->max_work_group_size;
  }
}

// ============================================================================
// Convenience overload for a kernel instance
// ============================================================================
template <class KernelClass>
static int64_t getMaxWorkGroupSize(
    KernelClass /*kfn*/, at::DeviceIndex dev_id = c10::xpu::current_device()) {
  return getMaxWorkGroupSize<KernelClass>(dev_id);
}

}  // namespace xpu
}  // namespace vllm

template <typename ScalarType, int Dims = 1>
using WorkgroupLocal = sycl::local_accessor<ScalarType, Dims>;