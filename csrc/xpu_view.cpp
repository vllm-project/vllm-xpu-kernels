#include <ATen/ATen.h>
#include "ops.h"
#include <c10/core/Device.h>
#include <c10/xpu/XPUFunctions.h>

namespace vllm::xpu {

/**
 * @class XPUHostViewAllocator
 * @brief Custom allocator that wraps existing host memory pointers for use with
 * XPU devices
 *
 * This allocator does not actually allocate new memory, but instead uses
 * pre-allocated host memory and wraps it in c10::DataPtr format for use within
 * the XPU system
 */

class XPUHostViewAllocator : public c10::Allocator {
 public:
  /**
   * @brief Constructor
   * @param host_ptr Pre-allocated host memory pointer
   * @param size Size of the host memory (in bytes)
   */
  XPUHostViewAllocator(void* host_ptr, size_t size)
      : host_ptr_(host_ptr), size_(size) {}

  /**
   * @brief Allocate memory (actually just validates and wraps existing host
   * memory)
   * @param n Requested memory size
   * @return Wrapped data pointer
   * @throws Exception if requested size exceeds pre-allocated memory size
   */
  c10::DataPtr allocate(size_t n) override {
    // Verify requested memory size doesn't exceed pre-allocated memory size
    TORCH_CHECK(
        n <= size_, "Requested size exceeds allocated host pointer size");
    // Return wrapped data pointer with no-op deleter since memory is externally
    // managed
    auto device_id = c10::xpu::current_device();
    return {
        host_ptr_,     // Actual data pointer
        host_ptr_,     // Context pointer (same as data pointer here)
        [](void*) {},  // No-op deleter, doesn't actually free memory
        c10::Device(c10::DeviceType::XPU, device_id)  // Device type set to XPU
    };
  }

  /**
   * @brief Check if data pointer is a simple pointer (requires no special
   * handling)
   * @param data_ptr Data pointer to check
   * @return Always returns true, indicating this is a simple data pointer
   */
  bool is_simple_data_ptr(const c10::DataPtr& data_ptr) const override {
    return true;
  }

  /**
   * @brief Data copy method
   * @param dest Destination memory address
   * @param src Source memory address
   * @param count Number of bytes to copy
   */
  void
  copy_data(void* dest, const void* src, std::size_t count) const override {
    std::memcpy(dest, src, count);
  }

 private:
  void* const host_ptr_;  // Pre-allocated host memory pointer
  const size_t size_;     // Size of pre-allocated memory
};
}  // namespace vllm::xpu

// This function assumes that `cpu_tensor` is a CPU tensor allocated with pinned
// memory, and that UVA (Unified Virtual Addressing) is enabled.
torch::Tensor get_xpu_view_from_cpu_tensor(torch::Tensor& cpu_tensor) {
  TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");
  TORCH_CHECK(cpu_tensor.is_contiguous(), "Input tensor must be contiguous");
  TORCH_CHECK(
      cpu_tensor.is_pinned(),
      "Input tensor must be allocated with pinned memory");
  // Get raw host pointer from CPU tensor
  void* host_ptr = cpu_tensor.data_ptr();

  // We'll use the same sizes, strides, and dtype as the CPU tensor.
  // TODO: check if layout is respected.
  auto sizes = cpu_tensor.sizes();
  auto strides = cpu_tensor.strides();
  auto scalar_type = cpu_tensor.scalar_type();

  size_t byte_size = cpu_tensor.numel() * cpu_tensor.element_size();
  vllm::xpu::XPUHostViewAllocator allocator(host_ptr, byte_size);
  c10::DataPtr data_ptr = allocator.allocate(byte_size);
  c10::Storage storage(
      c10::Storage::use_byte_size_t(), byte_size, std::move(data_ptr));

  auto impl = c10::make_intrusive<at::TensorImpl>(
      std::move(storage),
      c10::DispatchKeySet(c10::DispatchKey::XPU),
      at::scalarTypeToTypeMeta(scalar_type));

  // Set sizes and strides during construction to avoid extra copy
  impl->set_sizes_and_strides(sizes, strides);

  // Due to from_blob can only accept the device pointer, we use
  //  pluggable-allocator and storage to create tensor
  //  instead of tensor::from_blob
  torch::Tensor xpu_tensor = torch::Tensor::wrap_tensor_impl(impl);

  TORCH_CHECK(
      xpu_tensor.device().is_xpu(), "Resulting tensor is not on XPU device");
  return xpu_tensor;
}
