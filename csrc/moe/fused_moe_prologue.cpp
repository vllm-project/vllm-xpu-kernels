#include <torch/all.h>
#include "utils.h"
#include "fused_moe_prologue.hpp"

template<typename TA>
void fused_moe_prologue_impl(
    torch::Tensor input,
    torch::Tensor token_selected_experts,
    torch::Tensor token_final_scales,
    torch::Tensor workspace,
    int64_t hidden_size,
    int64_t inter_size,
    int64_t ep_rank,
    int64_t ep_size,
    int64_t num_experts_on_rank) {
  int experts_per_token = token_selected_experts.size(1);
  int64_t num_rows = input.size(0);

  assert(ep_rank >= 0 && ep_rank < ep_size);
  auto const num_experts_total =
      static_cast<int>(num_experts_on_rank * ep_size);
  auto& stream = at::xpu::getCurrentXPUStream(input.device().index()).queue();
  if constexpr(std::is_same_v<TA, at::Float4_e2m1fn_x2>){
    hidden_size = hidden_size / 2;
  }

  assert(token_selected_experts.dtype() == torch::kInt64);
  auto const* token_selected_experts_ =
      reinterpret_cast<int64_t const*>(token_selected_experts.data_ptr());
  auto const* input_activations =
      reinterpret_cast<TA const*>(input.data_ptr());
  auto const* token_topk_unpermuted_scales =
      reinterpret_cast<float const*>(token_final_scales.data_ptr());
  int const num_experts_per_node = num_experts_total / ep_size;
  int start_expert = num_experts_per_node * ep_rank;
  int end_expert = start_expert + num_experts_per_node;
  auto expanded_num_rows = num_rows * experts_per_token;

  // workspace configure
  auto ws_ptr = reinterpret_cast<uint8_t*>(workspace.data_ptr());
  size_t num_moe_inputs = experts_per_token * num_rows;
  size_t const permuted_elems = num_moe_inputs * hidden_size;
  size_t const interbuf_elems = num_moe_inputs * inter_size;

  constexpr int dtype_size = sizeof(TA);

  size_t const permuted_row_to_unpermuted_row_size =
      num_moe_inputs * sizeof(int);
  size_t const permuted_token_selected_experts_size =
      num_moe_inputs * sizeof(int);
  size_t src_to_dest_map_size = experts_per_token * num_rows * sizeof(int);
  size_t const expert_first_token_offset_size =
      (num_experts_per_node + 1) * sizeof(int64_t);

  int64_t const num_tokens_per_block =
      computeNumTokensPerBlock(num_rows, num_experts_per_node);
  int64_t const num_blocks_per_seq = ceilDiv(num_rows, num_tokens_per_block);
  size_t const blocked_expert_counts_size =
      num_experts_per_node * num_blocks_per_seq * sizeof(int);
  size_t const blocked_expert_counts_cumsum_size = blocked_expert_counts_size;
  size_t const blocked_row_to_unpermuted_row_size =
      num_experts_per_node * num_rows * sizeof(int);
  size_t const permuted_data_size = permuted_elems * dtype_size;
  size_t const permuted_token_final_scales_size =
      num_moe_inputs * sizeof(float);

  int map_offset = 0;
  std::map<std::string, std::pair<size_t, size_t>> ws_map;

#define ADD_NAME(name, size)                             \
  do {                                                   \
    size_t aligned_size = ((size) + 255) & ~255ULL;      \
    ws_map[#name] = std::pair{aligned_size, map_offset}; \
    map_offset += aligned_size;                          \
  } while (false)
#define ADD(name) ADD_NAME(name, name##_size)

  ADD(permuted_row_to_unpermuted_row);
  ADD(permuted_token_selected_experts);
  ADD_NAME(unpermuted_row_to_permuted_row, src_to_dest_map_size);
  ADD(blocked_expert_counts);
  ADD(blocked_expert_counts_cumsum);
  ADD(blocked_row_to_unpermuted_row);
  ADD(expert_first_token_offset);
  ADD(permuted_token_final_scales);
  ADD_NAME(overlapped_gemm1_gemm2_inputs, permuted_data_size);

  auto getWsPtr = [&](auto type, std::string const& name) {
    return ws_map.at(name).first ? reinterpret_cast<decltype(type)*>(
                                       ws_ptr + ws_map.at(name).second)
                                 : nullptr;
  };
  auto permuted_token_selected_experts_ =
      getWsPtr(int{}, "permuted_token_selected_experts");
  auto permuted_row_to_unpermuted_row_ =
      getWsPtr(int{}, "permuted_row_to_unpermuted_row");
  auto unpermuted_row_to_permuted_row =
      getWsPtr(int{}, "unpermuted_row_to_permuted_row");
  auto expert_first_token_offset_ =
      getWsPtr(int64_t{}, "expert_first_token_offset");
  auto blocked_expert_counts_ = getWsPtr(int{}, "blocked_expert_counts");
  auto blocked_expert_counts_cumsum_ =
      getWsPtr(int{}, "blocked_expert_counts_cumsum");
  auto blocked_row_to_unpermuted_row_ =
      getWsPtr(int{}, "blocked_row_to_unpermuted_row");
  auto permuted_data_ = getWsPtr(TA{}, "overlapped_gemm1_gemm2_inputs");
  auto permuted_token_final_scales_ =
      getWsPtr(float{}, "permuted_token_final_scales");
  bool use_per_expert_act_scale = false;
  at::DeviceGuard device_guard(input.device());
  // TODO: fused prologe
  threeStepBuildExpertMapsSortFirstToken(
      token_selected_experts_,
      permuted_token_selected_experts_,
      permuted_row_to_unpermuted_row_,
      unpermuted_row_to_permuted_row,
      expert_first_token_offset_,
      blocked_expert_counts_,
      blocked_expert_counts_cumsum_,
      blocked_row_to_unpermuted_row_,
      num_rows,
      num_experts_per_node,
      experts_per_token,
      start_expert,
      stream);

  TA * gemm1_input_expand = reinterpret_cast<TA *>(permuted_data_);
  expandInputRowsKernelLauncher(
      input_activations,
      gemm1_input_expand,
      token_topk_unpermuted_scales,
      permuted_token_final_scales_,
      permuted_row_to_unpermuted_row_,
      num_rows,
      hidden_size,
      experts_per_token,
      num_experts_per_node,
      use_per_expert_act_scale,
      expert_first_token_offset_,
      nullptr,
      stream);
}

void fused_moe_prologue(
    torch::Tensor input,
    torch::Tensor token_selected_experts,
    torch::Tensor token_final_scales,
    torch::Tensor workspace,
    int64_t hidden_size,
    int64_t inter_size,
    int64_t ep_rank,
    int64_t ep_size,
    int64_t num_experts_on_rank) {
  auto input_type = input.dtype();
  auto call_impl = [&](auto type) {
    using T = decltype(type);
    fused_moe_prologue_impl<T>(input, token_selected_experts, token_final_scales, workspace, hidden_size, inter_size, ep_rank, ep_size, num_experts_on_rank);
  };

  if (input_type == at::kBFloat16){
    call_impl(at::BFloat16{});
  } else if (input_type == at::kHalf) {
    call_impl(at::Half{});
  } else if (input_type == at::kFloat8_e4m3fn) {
    call_impl(at::Float8_e4m3fn{});
  } else if (input_type == at::kFloat4_e2m1fn_x2){
    call_impl(at::Float4_e2m1fn_x2{});
  }
}


