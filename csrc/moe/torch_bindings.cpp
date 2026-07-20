#include "core/registration.h"
#include "moe_ops.h"

VLLM_TORCH_LIBRARY_FRAGMENT_EXPAND(VLLM_TORCH_OP_NAMESPACE, m) {
  // Calculate the result of moe by summing up the partial results
  // from all selected experts.
  m.def(
      "moe_sum(Tensor input, Tensor! output, Tensor? topk_ids, "
      "Tensor? expert_map) -> ()");

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts,"
      "                     int block_size, Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad,"
      "                     Tensor? maybe_expert_map) -> ()");

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size, but for the batched case.
  m.def(
      "batched_moe_align_block_size(int max_tokens_per_batch,"
      "                     int block_size, Tensor expert_num_tokens,"
      "                     Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad) -> ()");

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  m.def(
      "moe_lora_align_block_size(Tensor topk_ids,"
      "                     Tensor token_lora_mapping,"
      "                     int num_experts,"
      "                     int block_size, int max_loras, "
      "                     int max_num_tokens_padded, "
      "                     int max_num_m_blocks, "
      "                     Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad,"
      "                     Tensor! adapter_enabled,"
      "                     Tensor! lora_ids,"
      "                     Tensor? maybe_expert_map) -> () ");

  // Apply grouped topk routing to select experts.
  m.def(
      "grouped_topk(Tensor scores, Tensor scores_with_bias, int n_group, int "
      "topk_group, int topk, bool renormalize, float "
      "routed_scaling_factor) -> (Tensor, Tensor)");

  // Apply topk softmax to the gating outputs.
  m.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output, bool renormalize, Tensor? "
      "bias) -> ()");

  // Apply topk sigmoid to the gating outputs.
  m.def(
      "topk_sigmoid(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output, bool renormalize, "
      "Tensor? bias, float routed_scaling_factor) -> ()");

  // Apply topk softplus sqrt to the gating outputs.
  m.def(
      "topk_softplus_sqrt(Tensor! topk_weights, Tensor! topk_indices, "
      "Tensor! token_expert_indices, Tensor gating_output, bool renormalize, "
      "float routed_scaling_factor, Tensor? correction_bias=None, Tensor? "
      "input_ids=None, Tensor? tid2eid=None) -> ()");

  // Apply topk softmax to the gating outputs.
  m.def(
      "moe_gather(Tensor! output, Tensor moe_output, Tensor topk_weights, "
      "Tensor unpermuted_row_to_permuted_row, "
      "int num_experts) -> ()");

  m.def(
      "fused_moe_prologue(Tensor input, Tensor? input_scales, Tensor "
      "token_selected_experts, "
      "Tensor "
      "token_final_scales, Tensor workspace, int hidden_size, int inter_size, "
      "int block_k, "
      "int ep_rank, int ep_size,"
      "int num_experts_on_rank) -> "
      "()");

  m.def(
      "init_expert_map(Tensor expert_map,"
      "int num_experts, "
      "int ep_rank, int ep_size) -> "
      "()");

  m.def(
      "remap_hidden_states(Tensor hidden_states, Tensor? hidden_states_scales, "
      "Tensor remapped_hidden_states,"
      "Tensor? remapped_hidden_states_scales,"
      "Tensor? expert_map, Tensor rows_per_expert,"
      "Tensor unpermuted_row_to_permuted_row, Tensor topk_ids,"
      "int total_experts_num, int "
      "local_experts_num) -> "
      "()");
}

VLLM_TORCH_LIBRARY_IMPL_EXPAND(VLLM_TORCH_OP_NAMESPACE, XPU, m) {
  VLLM_TORCH_IMPL(m, "moe_sum", &moe_sum);
  VLLM_TORCH_IMPL(m, "moe_align_block_size", &moe_align_block_size);
  VLLM_TORCH_IMPL(
      m, "batched_moe_align_block_size", &batched_moe_align_block_size);
  VLLM_TORCH_IMPL(m, "moe_lora_align_block_size", &moe_lora_align_block_size);
  VLLM_TORCH_IMPL(m, "grouped_topk", &grouped_topk);
  VLLM_TORCH_IMPL(m, "topk_softmax", &topk_softmax);
  VLLM_TORCH_IMPL(m, "topk_sigmoid", &topk_sigmoid);
  VLLM_TORCH_IMPL(m, "topk_softplus_sqrt", &topk_softplus_sqrt);
  VLLM_TORCH_IMPL(m, "moe_gather", &moe_gather);
  VLLM_TORCH_IMPL(m, "fused_moe_prologue", &fused_moe_prologue);
  VLLM_TORCH_IMPL(m, "init_expert_map", &init_expert_map);
  VLLM_TORCH_IMPL(m, "remap_hidden_states", &remap_hidden_states);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
