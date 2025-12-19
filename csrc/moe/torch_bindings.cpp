#include "core/registration.h"
#include "moe_ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  // Calculate the result of moe by summing up the partial results
  // from all selected experts.
  m.def("moe_sum(Tensor input, Tensor! output) -> ()");
  m.impl("moe_sum", torch::kXPU, &moe_sum);

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts,"
      "                     int block_size, Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad) -> ()");
  m.impl("moe_align_block_size", torch::kXPU, &moe_align_block_size);

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size, but for the batched case.
  m.def(
      "batched_moe_align_block_size(int max_tokens_per_batch,"
      "                     int block_size, Tensor expert_num_tokens,"
      "                     Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad) -> ()");
  m.impl(
      "batched_moe_align_block_size",
      torch::kXPU,
      &batched_moe_align_block_size);

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  m.def(
      "moe_lora_align_block_size(Tensor topk_ids,"
      "                     Tensor token_lora_mapping,"
      "                     int num_experts,"
      "                     int block_size, int max_loras, "
      "                     int max_num_tokens_padded, "
      "                     int max_num_m_blocks, "
      "                     Tensor !sorted_token_ids,"
      "                     Tensor !experts_ids,"
      "                     Tensor !num_tokens_post_pad,"
      "                     Tensor !adapter_enabled,"
      "                     Tensor !lora_ids) -> () ");
  m.impl("moe_lora_align_block_size", torch::kXPU, &moe_lora_align_block_size);

  // Apply grouped topk routing to select experts.
  m.def(
      "grouped_topk(Tensor scores, Tensor scores_with_bias, int n_group, int "
      "topk_group, int topk, bool renormalize, float "
      "routed_scaling_factor) -> (Tensor, Tensor)");
  m.impl("grouped_topk", torch::kXPU, &grouped_topk);

  // Fused Grouped TopK
  m.def(
      "fused_grouped_topk(Tensor hidden_states, Tensor gating_output, int "
      "n_topk, "
      "bool renormalize, int n_expert_group, int n_topk_group, str "
      "scoring_func, float routed_scaling_factor, Tensor? bias=None) -> "
      "(Tensor, Tensor)");
  m.impl("fused_grouped_topk", torch::kXPU, &fused_grouped_topk);
  // Apply topk softmax to the gating outputs.
  m.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output, bool renormalize) -> ()");
  m.impl("topk_softmax", torch::kXPU, &topk_softmax);
  // Apply topk softmax to the gating outputs.
  m.def(
      "moe_gather(Tensor! output, Tensor moe_output, Tensor topk_weights, "
      "Tensor "
      "unpermuted_row_to_permuted_row, int num_experts) -> ()");
  m.impl("moe_gather", torch::kXPU, &moe_gather);
  m.def(
      "fused_moe_prologue(Tensor input, Tensor token_selected_experts, "
      "Tensor "
      "token_final_scales, Tensor workspace, int hidden_size, int inter_size, "
      "int num_experts_on_rank) -> "
      "()");
  m.impl("fused_moe_prologue", torch::kXPU, &fused_moe_prologue);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
