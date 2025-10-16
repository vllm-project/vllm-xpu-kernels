#include <torch/all.h>
#include "utils.h"
#include <iostream>

typedef at::BFloat16 bfloat16;

void fused_moe(torch::Tensor output, 
               torch::Tensor input,
               torch::Tensor token_selected_experts,
               torch::Tensor token_final_scales,
               torch::Tensor fc1_expert_weights,
               torch::Tensor fc2_expert_weights){
  std::cout << "register fused_moe" << std::endl;
  int experts_per_token = token_selected_experts.size(1);
  int64_t num_rows = input.size(0);
  int64_t hidden_size = fc2_expert_weights.size(1);
  int64_t inter_size = fc2_expert_weights.size(2);
  int const num_experts_on_rank = fc2_expert_weights.size(0);
  // TODO: EP
  int ep_size = 1;
  auto const num_experts_total = static_cast<int>(num_experts_on_rank * ep_size);
  auto& stream =
      at::xpu::getCurrentXPUStream(output.device().index()).queue();
  
  auto const* input_activations = reinterpret_cast<bfloat16 const*>(input.data_ptr());
  auto const* fc1_expert_weights_ = reinterpret_cast<bfloat16 const*>(fc1_expert_weights.data_ptr());
  auto const* fc2_expert_weights_ = reinterpret_cast<bfloat16 const*>(fc2_expert_weights.data_ptr());  
  auto* final_output = reinterpret_cast<bfloat16 const*>(output.data_ptr()); 
  auto const* token_topk_unpermuted_scales = reinterpret_cast<bfloat16 const*>(token_final_scales.data_ptr());
  int const num_experts_per_node = num_experts_total / ep_size;
  int start_expert = num_experts_per_node * 0;
  int end_expert = start_expert + num_experts_per_node;
  auto expanded_num_rows = num_rows * experts_per_token;

  // TODO: fused prologe
  // threeStepBuildExpertMapsSortFirstToken(token_selected_experts, 
  //                                        permuted_token_selected_experts_,
  //                                        permuted_row_to_unpermuted_row_,
  //                                        unpermuted_row_to_permuted_row,
  //                                        expert_first_token_offset_, 
  //                                        blocked_expert_counts_,
  //                                        blocked_expert_counts_cumsum_,
  //                                        blocked_row_to_unpermuted_row_, 
  //                                        num_rows,
  //                                        num_experts_per_node, 
  //                                        experts_per_token, 
  //                                        start_expert, 
  //                                        enable_pdl, 
  //                                        stream);



}
 
