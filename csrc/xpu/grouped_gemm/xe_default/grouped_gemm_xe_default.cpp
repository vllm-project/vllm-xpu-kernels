#include <torch/all.h>
#include "grouped_gemm_xe_default.h"
#include "grouped_gemm.hpp"

torch::Tensor cutlass_grouped_gemm_xe_default(
    torch::Tensor ptr_A,
    torch::Tensor ptr_B,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor expert_first_token_offset,
    int64_t N,
    int64_t K,
    int64_t num_experts) {
  auto ptr_bias_ = ptr_bias;
  if (ptr_bias.has_value()) {
    auto expert_token_count = (expert_first_token_offset.slice(
                                   0, 1, expert_first_token_offset.size(0)) -
                               expert_first_token_offset.slice(0, 0, -1))
                                  .to(torch::kInt64);
    ptr_bias_ =
        ptr_bias->repeat_interleave(expert_token_count, 0).to(torch::kFloat32);
  }
  return gpu::cutlass_kernel::grouped_gemm_func(
      ptr_A,
      ptr_B,
      ptr_bias_,
      ptr_D,
      expert_first_token_offset,
      N,
      K,
      num_experts);
};