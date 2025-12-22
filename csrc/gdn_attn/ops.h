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
    const int64_t tp_size);