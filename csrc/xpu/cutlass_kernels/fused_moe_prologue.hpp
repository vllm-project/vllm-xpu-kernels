#include <torch/all.h>
#include "utils.h"


template <typename T>
inline T ceilDiv(T a, T b) {
    return (a + b - 1) / b;
}

int64_t computeNumTokensPerBlock(int64_t const num_tokens, int64_t const num_experts_per_node) {
  for (int64_t num_tokens_per_block = 32; num_tokens_per_block <= 1024; num_tokens_per_block *= 2) {
    int64_t const num_blocks_per_seq = ceilDiv(num_tokens, num_tokens_per_block);
    if (num_blocks_per_seq * num_experts_per_node <= num_tokens_per_block) {
      return num_tokens_per_block;
    }
  }
  return 1024;
}




void threeStepBuildExpertMapsSortFirstToken(
    int const* token_selected_experts, int* permuted_token_selected_experts,
    int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset, int* blocked_expert_counts,
    int* blocked_expert_counts_cumsum, int* blocked_row_to_unpermuted_row, int64_t const num_tokens,
    int64_t const num_experts_per_node, int64_t const num_experts_per_token,
    int const start_expert_id, sycl::queue& stream){
  std::cout << "three steps" << std::endl;
  int64_t const num_tokens_per_block = computeNumTokensPerBlock(num_rows, num_experts_per_node);
  int64_t const num_blocks_per_seq = ceilDiv(num_rows, num_tokens_per_block); 
  blockExpertPrefixSum(token_selected_experts, blocked_expert_counts, blocked_row_to_unpermuted_row,
                       num_tokens, num_experts_per_node, num_experts_per_token,
                       num_tokens_per_block, num_blocks_per_seq, start_expert_id, stream);
  stream.wait();
  std::cout << "finish 3 steps" << std::endl;
}
