#include <torch/all.h>

#include "utils.h"
#include "dispatch_utils.h"

#include "exponential_kernels.hpp"

void exponential_2d_(
    torch::Tensor& tensor, int64_t seed, int64_t offset, const double lambda) {
  auto dtype = tensor.dtype();
  int batch_size = tensor.size(0);
  int vocal_size = tensor.size(1);

  auto& queue = vllm::xpu::vllmGetQueue();

#define EXPONENTIAL_2D_LAUNCHER(T)             \
  using scalar_t = T;                          \
  RAND::exponential_2d_kernel_launcher(        \
      queue,                                   \
      reinterpret_cast<T*>(tensor.data_ptr()), \
      batch_size,                              \
      vocal_size,                              \
      seed,                                    \
      offset,                                  \
      lambda);

  if (dtype == torch::kFloat32) {
    EXPONENTIAL_2D_LAUNCHER(float)
  } else if (dtype == torch::kFloat16) {
    EXPONENTIAL_2D_LAUNCHER(sycl::half)
  } else if (dtype == torch::kBFloat16) {
    EXPONENTIAL_2D_LAUNCHER(sycl::ext::oneapi::bfloat16)
  } else {
    TORCH_CHECK(false, "Unsupported dtype: ", dtype);
  }
#undef EXPONENTIAL_2D_LAUNCHER
}

void exponential_2d_(
    torch::Tensor& tensor,
    torch::Tensor& seeds,  // should on CPU
    const double lambda) {
  auto seeds_ptr = reinterpret_cast<int64_t*>(seeds.data_ptr());
  int64_t seed = seeds_ptr[0];
  int64_t offset = seeds_ptr[1];
  exponential_2d_(tensor, seed, offset, lambda);
}
