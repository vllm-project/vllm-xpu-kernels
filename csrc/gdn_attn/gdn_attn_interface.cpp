#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"


void gdn_attention(
    torch::Tensor& core_attn_out,
    torch::Tensor& z,
    const torch::Tensor& projected_states_qkvz,
    const torch::Tensor& projected_states_ba,
    const int64_t num_k_heads,
    const int64_t num_v_heads,
    const int64_t head_k_dim,
    const int64_t head_v_dim,
    const int64_t tp_size) {
    // Implement the GDN attention mechanism here

    TORCH_CHECK(projected_states_qkvz.is_contiguous(), "projected_states_qkvz must be contiguous");
    TORCH_CHECK(projected_states_ba.is_contiguous(), "projected_states_ba must be contiguous");
    TORCH_CHECK(core_attn_out.is_contiguous(), "core_attn_out must be contiguous");

    // check core_attn_out shape
    int num_tokens = core_attn_out.size(0);
    TORCH_CHECK(core_attn_out.size(1) == num_v_heads / tp_size);
    TORCH_CHECK(core_attn_out.size(2) == head_v_dim);

    // check z shape
    TORCH_CHECK(z.size(0) == core_attn_out.size(0));
    TORCH_CHECK(z.size(1) == core_attn_out.size(1));
    TORCH_CHECK(z.size(2) == core_attn_out.size(2));

    // check projected_states_qkvz shape
    TORCH_CHECK(projected_states_qkvz.size(0) == num_tokens);
    TORCH_CHECK(projected_states_qkvz.size(1) == num_k_heads * (2 * head_k_dim + 2 * head_v_dim * num_v_heads / num_k_heads) / tp_size);

    // check projected_states_ba shape
    TORCH_CHECK(projected_states_ba.size(0) == num_tokens);
    TORCH_CHECK(projected_states_ba.size(1) == 2 * num_v_heads / tp_size);
}