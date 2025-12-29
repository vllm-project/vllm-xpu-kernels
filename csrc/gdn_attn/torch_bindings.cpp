#include "core/registration.h"
#include "ops.h"

#include <torch/library.h>
#include <torch/version.h>

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, xpu_ops) {
  at::Tag stride_tag = at::Tag::needs_fixed_stride_order;

  xpu_ops.def(
      "gdn_attention(Tensor! core_attn_out, Tensor! z, Tensor projected_states_qkvz, Tensor projected_states_ba,"
      "int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,"
      "Tensor! conv_state, Tensor! ssm_state, Tensor conv_weights, Tensor? conv_bias, str activation, Tensor A_log, Tensor dt_bias,"
      "int num_prefills, int num_decodes, Tensor? has_initial_state, Tensor non_spec_query_start_loc,"
      "Tensor non_spec_state_indices_tensor, int num_actual_tokens, int tp_size) -> ()");
  xpu_ops.impl("gdn_attention", torch::kXPU, &gdn_attention);

  xpu_ops.def(
      "causal_conv1d(Tensor! core_attn_out, Tensor! z, Tensor projected_states_qkvz, Tensor projected_states_ba,"
      "int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,"
      "Tensor! conv_state, Tensor! ssm_state, Tensor conv_weights, Tensor? conv_bias, str activation, Tensor A_log, Tensor dt_bias,"
      "int num_prefills, int num_decodes, Tensor? has_initial_state, Tensor non_spec_query_start_loc,"
      "Tensor non_spec_state_indices_tensor, int num_actual_tokens, int tp_size) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  xpu_ops.impl("causal_conv1d", torch::kXPU, &causal_conv1d);

  xpu_ops.def(
      "gated_delta_rule(Tensor! core_attn_out, Tensor! q, Tensor k, Tensor v,"
      "Tensor! b, Tensor! a, Tensor! A_log, Tensor! dt_bias, Tensor! ssm_state, Tensor! query_start_loc, Tensor! cache_indices,"
      "Tensor? has_initial_state, int num_prefills, int num_decodes) -> ()");
  xpu_ops.impl("gated_delta_rule", torch::kXPU, &gated_delta_rule);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
