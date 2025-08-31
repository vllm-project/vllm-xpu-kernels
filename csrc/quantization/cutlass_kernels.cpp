

// #include <sycl/sycl.hpp>
// #include <cassert>
// #include <vector>

// #include <ATen/Tensor.h>
/* #include "pytorch_shim.h" */

#include "core/registration.h"
#include <torch/all.h>
#include "xpu/utils.h"
#include "grouped_gemm.h"

namespace gpu::cutlass_kernel {

/* gemm2(group_A, w2, output, offset) */

at::Tensor grouped_gemm_func(
    at::Tensor& input,
    at::Tensor& weight, 
    at::Tensor& res,
    at::Tensor& offset,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t num_of_expert
   ) {
  auto dpcpp_queue = vllm::xpu::vllmGetQueue();
  if (input.scalar_type() != at::kBFloat16) {
    std::cout << "error:wrong datatype, current only support bfloat16" << std::endl;
    return at::Tensor();
  }

  grouped_gemm::kernel_functor(
      &dpcpp_queue,
      input.data_ptr(),
      weight.data_ptr(),
      res.data_ptr(),
      offset.data_ptr(),
      hidden_size,
      intermediate_size,
      num_of_expert
      );
  return res;
}

} // namespace gpu::cutlass_kernel

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("cutlass_grouped_gemm(Tensor input, Tensor weight, Tensor res, Tensor offset, int hidden_size, int intermediate_size, int num_of_expert) -> Tensor");
  ops.impl("cutlass_grouped_gemm", torch::kXPU, gpu::cutlass_kernel::grouped_gemm_func);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME);
