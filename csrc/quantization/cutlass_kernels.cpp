#include "grouped_gemm_fp8.h"

#include <sycl/sycl.hpp>
#include <cassert>
#include <vector>

#include <ATen/Tensor.h>
/* #include "pytorch_shim.h" */

#include "core/registration.h"
#include <torch/all.h>
#include "xpu/utils.h"

namespace gpu::cutlass_kernel {

at::Tensor grouped_gemm_func(
    at::Tensor& input,
    at::Tensor& weight, 
    at::Tensor& mnks, // [M, K], half
    at::Tensor& res
   ) {
  auto dpcpp_queue = vllm::xpu::vllmGetQueue();
  if (input.scalar_type() != at::kFloat8_e4m3fn) {
    std::cout << "error:wrong datatype" << std::endl;
    return at::Tensor();
  }

  grouped_gemm::kernel_functor(
      &dpcpp_queue,
      input.data_ptr(),
      weight.data_ptr(),
      mnks.data_ptr(),
      res.data_ptr()
      );
  return res;
}

} // namespace gpu::cutlass_kernel

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("cutlass_grouped_gemm(Tensor input, Tensor weight, Tensor mnks) -> Tensor");
  ops.impl("cutlass_grouped_gemm", torch::kXPU, gpu::cutlass_kernel::grouped_gemm_func);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME);
