#include "csrc/utils.h"
#include "fused_moe_interface.h"

#ifdef VLLM_XPU_ENABLE_XE2
  #include "xe2/fused_moe_gate_up_xe2.h"
#endif

torch::Tensor fused_moe_gate_up(
    torch::Tensor& ptr_A,
    const c10::optional<at::Tensor>& ptr_A_scale,
    torch::Tensor& ptr_B,
    const c10::optional<at::Tensor>& ptr_B_scale,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor& ptr_D,
    torch::Tensor& rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    std::string activation,
    double gemm1_clamp_limit) {
  if (vllm::xpu::is_xe2_arch() || vllm::xpu::is_xe3_arch()) {
#ifdef VLLM_XPU_ENABLE_XE2
    // Use XE2 cutlass kernel (also used as WA for XE3/XE3P)
    return fused_moe_gate_up_xe2(
        ptr_A,
        ptr_A_scale,
        ptr_B,
        ptr_B_scale,
        ptr_bias,
        ptr_D,
        rows_per_expert,
        N,
        K,
        num_experts,
        activation,
        gemm1_clamp_limit);
#else
    TORCH_CHECK(false, "XE2 cutlass kernel is not enabled in this build.");
#endif
  } else {
    TORCH_CHECK(false, "Only XE2/XE3 cutlass kernel is supported currently.");
  }
}
