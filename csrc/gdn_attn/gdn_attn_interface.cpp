#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"

#include "causal_conv1d.hpp"
// #include "gated_delta_rule.hpp"

void gdn_attention(
    torch::Tensor& core_attn_out,
    torch::Tensor& z,
    const torch::Tensor& projected_states_qkvz,
    const torch::Tensor& projected_states_ba,
    const int64_t num_k_heads,
    const int64_t num_v_heads,
    const int64_t head_k_dim,
    const int64_t head_v_dim,
    const int64_t tp_size,
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
) {
    // Implement the GDN attention mechanism here

    TORCH_CHECK(projected_states_qkvz.is_contiguous(), "projected_states_qkvz must be contiguous");
    TORCH_CHECK(projected_states_ba.is_contiguous(), "projected_states_ba must be contiguous");
    TORCH_CHECK(core_attn_out.is_contiguous(), "core_attn_out must be contiguous");

    // check core_attn_out shape
    int num_tokens = core_attn_out.size(0);
    TORCH_CHECK(num_tokens == num_actual_tokens);
    TORCH_CHECK(core_attn_out.size(1) == num_v_heads / tp_size);
    TORCH_CHECK(core_attn_out.size(2) == head_v_dim);

    // check z shape
    TORCH_CHECK(z.size(0) == core_attn_out.size(0));
    TORCH_CHECK(z.size(1) == core_attn_out.size(1));
    TORCH_CHECK(z.size(2) == core_attn_out.size(2));

    // check projected_states_qkvz shape
    TORCH_CHECK(projected_states_qkvz.size(1) == num_k_heads * (2 * head_k_dim + 2 * head_v_dim * num_v_heads / num_k_heads) / tp_size);

    // check projected_states_ba shape
    TORCH_CHECK(projected_states_ba.size(1) == 2 * num_v_heads / tp_size);

    auto& queue = vllm::xpu::vllmGetQueue();
    auto dtype = projected_states_qkvz.dtype();
    auto device = projected_states_qkvz.device();
    torch::Tensor q = torch::empty({num_actual_tokens, num_k_heads, head_k_dim},
                            torch::dtype(dtype).device(device).requires_grad(false));
    torch::Tensor k = torch::empty({num_actual_tokens, num_k_heads, head_k_dim},
                            torch::dtype(dtype).device(device).requires_grad(false));
    torch::Tensor v = torch::empty({num_actual_tokens, num_v_heads, head_v_dim},
                            torch::dtype(dtype).device(device).requires_grad(false));
    gdn::ActMode act_mode;

    if(activation == "silu"){
        act_mode = gdn::ActMode::silu;
    }else if(activation == "swish"){
        act_mode = gdn::ActMode::swish;
    }else{
        TORCH_CHECK(false);
    }
    const int pad_slot_id = -1;

    gdn::causal_conv1d(
        queue,
        q,
        k,
        v,
        z,
        head_k_dim,
        head_v_dim,
        num_k_heads,
        num_v_heads,
        projected_states_qkvz,
        conv_weights,
        conv_bias,
        conv_state,
        non_spec_query_start_loc,
        non_spec_state_indices_tensor,
        has_initial_state,
        act_mode,
        pad_slot_id,
        num_actual_tokens
    );

    gdn::gated_delta_rule(
        queue,
        core_attn_out,
        q,
        k,
        v,
        projected_states_ba,
        A_log,
        dt_bias,
        initial_state,
        ssm_state,
        non_spec_query_start_loc,
        non_spec_state_indices_tensor,
    );


}