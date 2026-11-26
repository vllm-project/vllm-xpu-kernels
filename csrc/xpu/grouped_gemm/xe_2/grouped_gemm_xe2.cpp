#include <torch/all.h>
#include "grouped_gemm_xe2.h"
#include "grouped_gemm_xe2_interface.hpp"

torch::Tensor cutlass_grouped_gemm_xe2(
    torch::Tensor ptr_A,
    torch::Tensor ptr_B,
    const c10::optional<at::Tensor>& ptr_scales,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t num_experts) {
  return MoE::cutlass_grouped_gemm_xe2_impl(
      ptr_A,
      ptr_B,
      ptr_scales,
      ptr_bias,
      ptr_D,
      rows_per_expert,
      N,
      K,
      num_experts);
}

void cutlass_batched_gemm_xe2(
    torch::Tensor input,
    torch::Tensor weight,
    const c10::optional<at::Tensor>& bias,
    torch::Tensor output,
    torch::Tensor rows_per_expert,
    int64_t tile_m) {
  const auto capacity = input.size(1);
  if (capacity == 0) {
    return;
  }
  auto flat_input = input.view({-1, input.size(2)});
  auto flat_output = output.view({-1, output.size(2)});
  MoE::cutlass_grouped_gemm_xe2_impl(
      flat_input,
      weight,
      c10::nullopt,
      bias,
      flat_output,
      rows_per_expert,
      output.size(2),
      input.size(2),
      input.size(0),
      capacity,
      tile_m);
}
