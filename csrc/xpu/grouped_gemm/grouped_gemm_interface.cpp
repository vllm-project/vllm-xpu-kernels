#include "csrc/utils.h"
#include "grouped_gemm_interface.h"
#define VLLM_XPU_ENABLE_XE2

#ifdef VLLM_XPU_ENABLE_XE2
  #include "xe_2/grouped_gemm_xe2.h"
#endif

torch::Tensor cutlass_grouped_gemm_interface(
    torch::Tensor ptr_A,
    torch::Tensor ptr_B,
    const c10::optional<at::Tensor>& ptr_scales,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor expert_first_token_offset,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    bool is_B_int4,
    bool is_B_mxfp4) {
  if (vllm::xpu::is_xe2_arch()) {
#ifdef VLLM_XPU_ENABLE_XE2
    // Use XE2 cutlass kernel
    return cutlass_grouped_gemm_xe2(
        ptr_A,
        ptr_B,
        ptr_scales,
        ptr_bias,
        ptr_D,
        expert_first_token_offset,
        N,
        K,
        num_experts,
        is_B_int4,
        is_B_mxfp4);
#else
    TORCH_CHECK(false, "XE2 cutlass kernel is not enabled in this build.");
#endif
  } else {
#if 0
    // FIXME: confirm groups meaning here.
    int64_t groups = num_experts;
    return cutlass_grouped_gemm_default(
        ptr_A,
        ptr_B,
        ptr_bias,
        ptr_D,
        expert_first_token_offset,
        N,
        K,
        groups);
#endif
  }
}
