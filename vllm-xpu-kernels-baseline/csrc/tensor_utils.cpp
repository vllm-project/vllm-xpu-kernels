#include <torch/all.h>
#include <vector>

torch::Tensor weak_ref_tensor(torch::Tensor& tensor) {
  // Ensure tensor is on XPU
  if (!tensor.is_xpu()) {
    throw std::runtime_error("Tensor must be on XPU device");
  }

  // Get the raw data pointer
  void* data_ptr = tensor.data_ptr();

  // Get tensor sizes and strides
  std::vector<int64_t> sizes = tensor.sizes().vec();
  std::vector<int64_t> strides = tensor.strides().vec();

  // Get tensor options (dtype, device)
  auto options = tensor.options();

  // Create a new tensor from the raw data pointer
  auto new_tensor = torch::from_blob(data_ptr, sizes, strides, options);

  return new_tensor;
}
