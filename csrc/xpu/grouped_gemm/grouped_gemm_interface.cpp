#include "csrc/utils.h"
#include "grouped_gemm_interface.h"
#include <stdio.h>

#ifdef VLLM_XPU_ENABLE_XE2
  #include "xe_2/grouped_gemm_xe2.h"
#endif
#ifdef VLLM_XPU_ENABLE_XE3P
  #include "xe_3/grouped_gemm_xe3.h"
#endif

torch::Tensor cutlass_grouped_gemm_interface(
    torch::Tensor ptr_A,
    const c10::optional<at::Tensor>& ptr_A_scale,
    torch::Tensor ptr_B,
    const c10::optional<at::Tensor>& ptr_B_scale,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t num_experts) {
#ifdef VLLM_XPU_ENABLE_XE3P
  if (vllm::xpu::is_xe3p_arch()) {
    // Xe3P consumes block-scaled activations (MXFP8 / MXFP4 / block-FP8)
    // natively, so ptr_A_scale is forwarded to the device kernel.
    return cutlass_grouped_gemm_xe3(
        ptr_A,
        ptr_A_scale,
        ptr_B,
        ptr_B_scale,
        ptr_bias,
        ptr_D,
        rows_per_expert,
        N,
        K,
        num_experts);
  }
#endif
#ifdef VLLM_XPU_ENABLE_XE2
  // BMG / PVC / LNL and the Xe3 client parts run the xe_2 kernel, which
  // consumes high-precision activations only.
  (void)ptr_A_scale;
  return cutlass_grouped_gemm_xe2(
      ptr_A,
      ptr_B,
      ptr_B_scale,
      ptr_bias,
      ptr_D,
      rows_per_expert,
      N,
      K,
      num_experts);
#else
  TORCH_CHECK(
      false, "No cutlass grouped GEMM kernel is enabled in this build.");
#endif
}
