#include "base_gemm.h"
#include "scaled_mm.h"

#include <sycl/sycl.hpp>
#include <cassert>
#include <vector>

#include <ATen/Tensor.h>
/* #include "pytorch_shim.h" */

#include "core/registration.h"
#include <torch/all.h>
#include "xpu/utils.h"

namespace gpu::cutlass_kernel {

at::Tensor basic_gemm_func(
    const at::Tensor& inputA,
    const at::Tensor& inputB,
    const at::Tensor& inputC,
    const at::Tensor& res,
    double alpha,
    double beta) {
  int m = inputA.size(0);
  int n = inputB.size(0);
  int k = inputA.size(1);

  auto dpcpp_queue = vllm::xpu::vllmGetQueue();
  basic_gemm::gemm_functor<cutlass::bfloat16_t>(
      &dpcpp_queue,
      inputA.data_ptr(),
      inputB.data_ptr(),
      inputC.data_ptr(),
      res.data_ptr(),
      m,
      n,
      k,
      alpha,
      beta);
  return res;
}

at::Tensor scaled_mm_func(
    at::Tensor& inputA, // [M, K], fp8_e4m3
    at::Tensor& inputB, // [N, K], fp8_e4m3
    at::Tensor& scaleA, // [M, K], half
    at::Tensor& scaleB, // [N, K], half
    at::Tensor& res, // [M, N], float
    double alpha,
    double beta) {
  int m = inputA.size(0);
  int n = inputB.size(0);
  int k = inputA.size(1);

  auto dpcpp_queue = vllm::xpu::vllmGetQueue();
  if (inputA.scalar_type() != at::kFloat8_e4m3fn) {
    std::cout << "error:wrong datatype" << std::endl;
    return at::Tensor();
  }

  scaled_mm::kernel_functor(
      &dpcpp_queue,
      inputA.data_ptr(),
      inputB.data_ptr(),
      scaleA.data_ptr(),
      scaleB.data_ptr(),
      res.data_ptr(),
      m,
      n,
      k,
      alpha,
      beta);
  return res;
}

} // namespace gpu::cutlass_kernel

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("cutlass_gemm(Tensor inputA, Tensor inputB, Tensor inputC, Tensor res, float alpha, float beta) -> Tensor");
  ops.impl("cutlass_gemm", torch::kXPU, gpu::cutlass_kernel::basic_gemm_func);

  ops.def("cutlass_scaled_mm(Tensor inputA, Tensor inputB, Tensor scaleA, Tensor scaleB, Tensor res, float alpha, float beta) -> Tensor");
  ops.impl("cutlass_scaled_gemm", torch::kXPU, gpu::cutlass_kernel::scaled_mm_func);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME);


