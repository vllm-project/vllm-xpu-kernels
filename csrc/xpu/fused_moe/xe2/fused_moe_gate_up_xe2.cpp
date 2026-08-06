#include "fused_moe_gate_up_xe2.h"
#include "csrc/xpu/fused_moe/common/fused_moe_gate_up_impl.h"

torch::Tensor fused_moe_gate_up_xe2(
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
  return FusedMOE::fused_moe_gate_up_impl(
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
}
