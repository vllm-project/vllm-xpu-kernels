#include <torch/all.h>
#include "utils.h"
#include <iostream>

typedef at::BFloat16 bfloat16;

std::map<std::string, std::pair<size_t, size_t>> ws_map;
int align64(int size){
  // todo
  return size;
}

void fused_moe(torch::Tensor output, 
               torch::Tensor input,
               torch::Tensor token_selected_experts,
               torch::Tensor token_final_scales,
               torch::Tensor fc1_expert_weights,
               torch::Tensor fc2_expert_weights,
               torch::Tensor workspace){
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
  auto* final_output = reinterpret_cast<bfloat16*>(output.data_ptr()); 
  auto const* token_topk_unpermuted_scales = reinterpret_cast<bfloat16 const*>(token_final_scales.data_ptr());
  int const num_experts_per_node = num_experts_total / ep_size;
  int start_expert = num_experts_per_node * 0;
  int end_expert = start_expert + num_experts_per_node;
  auto expanded_num_rows = num_rows * experts_per_token;


  // workspace configure
  auto ws_ptr = reinterpret_cast<uint8_t*>(workspace.data_ptr());
  size_t num_moe_inputs = experts_per_token * num_rows;
  size_t const permuted_elems = num_moe_inputs * hidden_size;
  size_t const interbuf_elems = num_moe_inputs * inter_size;

  constexpr float dtype_size = 2.0f; 

  size_t const permuted_row_to_unpermuted_row_size = num_moe_inputs * sizeof(int);
  size_t const permuted_token_selected_experts_size = num_moe_inputs * sizeof(int);  
  size_t src_to_dest_map_size = experts_per_token * num_rows * sizeof(int);
  size_t const expert_first_token_offset_size = (num_experts_per_node + 1) * sizeof(int64_t);
  
  size_t const num_blocks_per_seq = 1;
  size_t const blocked_expert_counts_size = num_experts_per_node * num_blocks_per_seq * sizeof(int);
  size_t const blocked_expert_counts_cumsum_size = blocked_expert_counts_size;
  size_t const blocked_row_to_unpermuted_row_size = num_experts_per_node * num_rows * sizeof(int);


  int map_offset = 0;
  ws_map["permuted_token_selected_experts"] = std::pair{permuted_token_selected_experts_size, map_offset};  
  map_offset += permuted_token_selected_experts_size;
  ws_map["permuted_row_to_unpermuted_row"] = std::pair{permuted_row_to_unpermuted_row_size, map_offset};
  map_offset += permuted_row_to_unpermuted_row_size;
  ws_map["src_to_dest_map"] = std::pair{src_to_dest_map_size, map_offset};
  map_offset += src_to_dest_map_size;
  ws_map["expert_first_token_offset"] = std::pair{expert_first_token_offset_size, map_offset};
  map_offset += expert_first_token_offset_size;
  ws_map["blocked_expert_counts"] = std::pair{blocked_expert_counts_size, map_offset};
  map_offset += blocked_expert_counts_size;
  ws_map["blocked_expert_counts_cumsum"] = std::pair{blocked_expert_counts_cumsum_size, map_offset};
  map_offset += blocked_expert_counts_cumsum_size;
  ws_map["blocked_row_to_unpermuted_row"] = std::pair{blocked_row_to_unpermuted_row_size, map_offset};
  map_offset += blocked_row_to_unpermuted_row_size;

  std::cout << "total workspace size(byte): " << map_offset << std::endl;





   

  auto getWsPtr = [&](auto type, std::string const& name) {
    return ws_map.at(name).first
               ? reinterpret_cast<decltype(type)*>(ws_ptr + ws_map.at(name).second)
               : nullptr;
  };
  auto permuted_token_selected_experts_ = getWsPtr(int{}, "permuted_token_selected_experts"); 
  auto permuted_row_to_unpermuted_row_ = getWsPtr(int{}, "permuted_row_to_unpermuted_row");
  auto unpermuted_row_to_permuted_row = getWsPtr(int{}, "src_to_dest_map");
  auto expert_first_token_offset_ = getWsPtr(int64_t{}, "expert_first_token_offset");
  auto blocked_expert_counts_ = getWsPtr(int{}, "blocked_expert_counts");
  auto blocked_expert_counts_cumsum_ = getWsPtr(int{}, "blocked_expert_counts_cumsum");
  auto blocked_row_to_unpermuted_row_ = getWsPtr(int{}, "blocked_row_to_unpermuted_row");


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
  //                                        stream);



}
 
