#include <torch/all.h>
#include "grouped_gemm_xe3.h"
#include "csrc/xpu/grouped_gemm/xe_3/grouped_gemm.hpp"

namespace {

// Equivalent to `bias.repeat_interleave(rows_per_expert, 0)`, but computed
// entirely on device. The tensor overload of repeat_interleave reads the
// repeat counts on the host, which synchronizes and cannot be captured into
// an XPU graph.
torch::Tensor expand_bias_to_rows(
    const torch::Tensor& bias,
    const torch::Tensor& rows_per_expert,
    int64_t total_m) {
  auto row_offsets = rows_per_expert.to(torch::kInt64).cumsum(0);
  auto rows = torch::arange(total_m, row_offsets.options());
  auto expert_ids = torch::searchsorted(
      row_offsets, rows, /*out_int32=*/false, /*right=*/true);
  return bias.to(torch::kFloat32).index_select(0, expert_ids);
}

}  // namespace

torch::Tensor cutlass_grouped_gemm_xe3(
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
  auto ptr_bias_ = ptr_bias;
  if (ptr_bias.has_value()) {
    ptr_bias_ = expand_bias_to_rows(*ptr_bias, rows_per_expert, ptr_A.size(0));
  }
  return gpu::cutlass_kernel::grouped_gemm_func(
      ptr_A,
      ptr_A_scale,
      ptr_B,
      ptr_B_scale,
      ptr_bias_,
      ptr_D,
      rows_per_expert,
      N,
      K,
      num_experts);
};
