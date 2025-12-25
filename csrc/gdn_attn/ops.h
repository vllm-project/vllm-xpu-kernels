#pragma once

#include <torch/all.h>

void gdn_attention(
    torch::Tensor& core_attn_out,
    torch::Tensor& z,
    const torch::Tensor& projected_states_qkvz,
    const torch::Tensor& projected_states_ba,
    const int64_t num_k_heads,
    const int64_t num_v_heads,
    const int64_t head_k_dim,
    const int64_t head_v_dim,
    torch::Tensor& conv_state,
    torch::Tensor& ssm_state,
    const torch::Tensor& conv_weights,
    const std::optional<torch::Tensor>& conv_bias,
    const std::string& activation,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    const int64_t num_prefills,
    const int64_t num_decodes,
    const std::optional<torch::Tensor>& has_initial_state,
    const torch::Tensor& non_spec_query_start_loc,
    const torch::Tensor& non_spec_state_indices_tensor,
    const int64_t num_actual_tokens
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> causal_conv1d(
    torch::Tensor& core_attn_out, // [total_seqlen, num_v_heads, head_v_dim]
    torch::Tensor& z,              // [total_seqlen, num_v_heads, head_v_dim]
    const torch::Tensor& projected_states_qkvz, // [total_seqlen, num_k_heads * (2 * head_k_dim + 2 * head_v_dim * num_v_heads / num_k_heads)]
    const torch::Tensor& projected_states_ba, // [total_seqlen, num_k_heads * (2 * num_v_heads / num_k_heads)]
    const int64_t num_k_heads,
    const int64_t num_v_heads,
    const int64_t head_k_dim,
    const int64_t head_v_dim,
    torch::Tensor& conv_state,  // [cache_batch_size, width - 1, num_k_heads * (2 * head_k_dim + head_v_dim * num_v_heads / num_k_heads)]
    torch::Tensor& ssm_state,   // [cache_batch_size, num_v_heads, head_k_dim, head_v_dim]
    const torch::Tensor& conv_weights,   // [num_k_heads * (2 * head_k_dim + head_v_dim * num_v_heads / num_k_heads), width]
    const std::optional<torch::Tensor>& conv_bias, // [num_k_heads * (2 * head_k_dim + head_v_dim * num_v_heads / num_k_heads)] or None
    const std::string& activation,
    const torch::Tensor& A_log,  // [num_v_heads]
    const torch::Tensor& dt_bias, // [num_v_heads]
    const int64_t num_prefills,
    const int64_t num_decodes,
    const std::optional<torch::Tensor>& has_initial_state, // [batch_size] or None 
    const torch::Tensor& non_spec_query_start_loc,   // [batch_size + 1]
    const torch::Tensor& non_spec_state_indices_tensor,  // [batch_size]
    const int64_t num_actual_tokens
);

void gated_delta_rule(
    torch::Tensor& core_attn_out, // [total_seqlen, num_v_heads, head_v_dim]
    const torch::Tensor& q, // [total_seqlen, num_k_heads, head_k_dim]
    const torch::Tensor& k, // [total_seqlen, num_k_heads, head_k_dim]
    const torch::Tensor& v, // [total_seqlen, num_v_heads, head_v_dim]
    const torch::Tensor& b, // [total_seqlen, num_v_heads]
    const torch::Tensor& a, // [total_seqlen, num_v_heads]
    const torch::Tensor& A_log, // [num_v_heads]
    const torch::Tensor& dt_bias, // [num_v_heads]
    torch::Tensor& ssm_state, // [cache_batch_size, num_v_heads, head_k_dim, head_v_dim]
    const torch::Tensor& query_start_loc,  // [batch_size + 1]
    const torch::Tensor& cache_indices, // [batch_size]
    const std::optional<torch::Tensor>& has_initial_state // [batch_size] or None
);
