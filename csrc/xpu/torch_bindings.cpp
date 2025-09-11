#include "core/registration.h"
#include "xpu/ops.h"
#include "xpu/cutlass_backend/cutlass_kernels.h"

#include <torch/library.h>
#include <torch/version.h>

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, xpu_ops) {
  at::Tag stride_tag = at::Tag::needs_fixed_stride_order;

  xpu_ops.def(
      "fp8_gemm_w8a16(Tensor! A, Tensor! B, bool trans_B, Tensor? B_scale_, "
      "Tensor? bias_) -> Tensor");
  xpu_ops.impl("fp8_gemm_w8a16", torch::kXPU, &fp8_gemm_w8a16);
  
  xpu_ops.def("cutlass_grouped_gemm(Tensor ptr_A, Tensor ptr_B, Tensor ptr_D, Tensor ptr_alpha, Tensor ptr_beta, Tensor offset, int N, int K, int groups) -> Tensor");
  xpu_ops.impl("cutlass_grouped_gemm", torch::kXPU, gpu::cutlass_kernel::grouped_gemm_func);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
