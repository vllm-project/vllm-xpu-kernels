#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "utils.h"
#include "dispatch_utils.h"

#include "causal_conv1d.hpp"
#include "gated_delta_rule.hpp"
#ifdef VLLM_XPU_ENABLE_XE2
  #include "xe_2/chunk_causal_conv1d_xe2.hpp"
  #include "xe_2/chunk_gated_delta_rule_xe2.h"
#endif

void gdn_attention(
    torch::Tensor&
        core_attn_out,  // [total_seqlen, num_v_heads / tp_size, head_v_dim]
    torch::Tensor& z,   // [total_seqlen, num_v_heads / tp_size, head_v_dim]
    const torch::Tensor&
        projected_states_qkvz,  // [total_seqlen, num_k_heads / tp_size * (2 *
                                // head_k_dim + 2 * head_v_dim * num_v_heads /
                                // num_k_heads)]
    const torch::Tensor&
        projected_states_ba,  // [total_seqlen, num_k_heads / tp_size * (2 *
                              // num_v_heads / num_k_heads)]
    const int64_t num_k_heads,
    const int64_t num_v_heads,
    const int64_t head_k_dim,
    const int64_t head_v_dim,
    torch::Tensor&
        conv_state,  // [cache_batch_size, width - 1, num_k_heads / tp_size * (2
                     // * head_k_dim + head_v_dim * num_v_heads / num_k_heads)]
    torch::Tensor& ssm_state,  // [cache_batch_size, num_v_heads / tp_size,
                               // head_v_dim, head_k_dim]
    const torch::Tensor&
        conv_weights,  // [num_k_heads / tp_size * (2 * head_k_dim + head_v_dim
                       // * num_v_heads / num_k_heads), width]
    const std::optional<torch::Tensor>&
        conv_bias,  // [num_k_heads / tp_size * (2 * head_k_dim + head_v_dim *
                    // num_v_heads / num_k_heads)] or None
    const std::string& activation,
    const torch::Tensor& A_log,    // [num_v_heads / tp_size]
    const torch::Tensor& dt_bias,  // [num_v_heads / tp_size]
    const int64_t num_prefills,
    const int64_t num_decodes,
    const std::optional<torch::Tensor>&
        has_initial_state,                               // [batch_size] or None
    const torch::Tensor& non_spec_query_start_loc,       // [batch_size + 1]
    const torch::Tensor& non_spec_state_indices_tensor,  // [batch_size]
    const int64_t num_actual_tokens,
    const int64_t tp_size) {
  TORCH_CHECK(
      core_attn_out.is_contiguous(), "core_attn_out must be contiguous");
  TORCH_CHECK(z.is_contiguous(), "z must be contiguous");
  TORCH_CHECK(
      projected_states_qkvz.is_contiguous(),
      "projected_states_qkvz must be contiguous");
  TORCH_CHECK(
      projected_states_ba.is_contiguous(),
      "projected_states_ba must be contiguous");
  TORCH_CHECK(
      conv_state[0].is_contiguous(),
      "conv_state of each batch must be contiguous");
  TORCH_CHECK(
      ssm_state[0].is_contiguous(),
      "ssm_state of each batch must be contiguous");
  TORCH_CHECK(conv_weights.is_contiguous(), "conv_weights must be contiguous");
  TORCH_CHECK(A_log.is_contiguous(), "A_log must be contiguous");
  TORCH_CHECK(dt_bias.is_contiguous(), "dt_bias must be contiguous");
  TORCH_CHECK(
      non_spec_query_start_loc.is_contiguous(),
      "non_spec_query_start_loc must be contiguous");
  TORCH_CHECK(
      non_spec_state_indices_tensor.is_contiguous(),
      "non_spec_state_indices_tensor must be contiguous");

  // check core_attn_out shape
  TORCH_CHECK(core_attn_out.size(0) == num_actual_tokens);
  TORCH_CHECK(core_attn_out.size(1) == num_v_heads / tp_size);
  TORCH_CHECK(core_attn_out.size(2) == head_v_dim);

  // check z shape
  TORCH_CHECK(z.size(0) == core_attn_out.size(0));
  TORCH_CHECK(z.size(1) == core_attn_out.size(1));
  TORCH_CHECK(z.size(2) == core_attn_out.size(2));

  // check projected_states_qkvz shape
  TORCH_CHECK(projected_states_qkvz.size(0) == num_actual_tokens);
  TORCH_CHECK(
      projected_states_qkvz.size(1) ==
      num_k_heads / tp_size *
          (2 * head_k_dim + 2 * head_v_dim * num_v_heads / num_k_heads));

  // check projected_states_ba shape
  TORCH_CHECK(projected_states_ba.size(0) == num_actual_tokens);
  TORCH_CHECK(projected_states_ba.size(1) == 2 * num_v_heads / tp_size);

  auto& queue = vllm::xpu::vllmGetQueue();
  auto dtype = projected_states_qkvz.dtype();
  auto device = projected_states_qkvz.device();
  gdn::ActMode act_mode;

  if (activation == "silu") {
    act_mode = gdn::ActMode::silu;
  } else if (activation == "swish") {
    act_mode = gdn::ActMode::swish;
  } else {
    TORCH_CHECK(false);
  }
  const int pad_slot_id = -1;

#define NATIVE_LAUNCHER                                           \
  do {                                                            \
    torch::Tensor q = torch::empty(                               \
        {num_actual_tokens, num_k_heads / tp_size, head_k_dim},   \
        torch::dtype(dtype).device(device).requires_grad(false)); \
    torch::Tensor k = torch::empty(                               \
        {num_actual_tokens, num_k_heads / tp_size, head_k_dim},   \
        torch::dtype(dtype).device(device).requires_grad(false)); \
    torch::Tensor v = torch::empty(                               \
        {num_actual_tokens, num_v_heads / tp_size, head_v_dim},   \
        torch::dtype(dtype).device(device).requires_grad(false)); \
    torch::Tensor b = torch::empty(                               \
        {num_actual_tokens, num_v_heads / tp_size},               \
        torch::dtype(dtype).device(device).requires_grad(false)); \
    torch::Tensor a = torch::empty(                               \
        {num_actual_tokens, num_v_heads / tp_size},               \
        torch::dtype(dtype).device(device).requires_grad(false)); \
    gdn::causal_conv1d(                                           \
        queue,                                                    \
        q,                                                        \
        k,                                                        \
        v,                                                        \
        z,                                                        \
        b,                                                        \
        a,                                                        \
        projected_states_qkvz,                                    \
        projected_states_ba,                                      \
        conv_weights,                                             \
        conv_bias,                                                \
        conv_state,                                               \
        non_spec_query_start_loc,                                 \
        non_spec_state_indices_tensor,                            \
        has_initial_state,                                        \
        act_mode,                                                 \
        pad_slot_id,                                              \
        num_prefills,                                             \
        num_decodes);                                             \
    gdn::gated_delta_rule(                                        \
        queue,                                                    \
        core_attn_out,                                            \
        q,                                                        \
        k,                                                        \
        v,                                                        \
        b,                                                        \
        a,                                                        \
        A_log,                                                    \
        dt_bias,                                                  \
        ssm_state,                                                \
        non_spec_query_start_loc,                                 \
        non_spec_state_indices_tensor,                            \
        has_initial_state,                                        \
        num_prefills,                                             \
        num_decodes);                                             \
  } while (0)

#ifdef VLLM_XPU_ENABLE_XE2
  if (num_prefills > 0) {
    int batch_size = non_spec_query_start_loc.size(0) - 1;
    int padding_size = batch_size * (gdn::chunk_size_xe2 - 1);

    torch::Tensor q = torch::zeros(
        {num_actual_tokens + padding_size, num_k_heads / tp_size, head_k_dim},
        torch::dtype(dtype).device(device).requires_grad(false));
    torch::Tensor k = torch::zeros(
        {num_actual_tokens + padding_size, num_k_heads / tp_size, head_k_dim},
        torch::dtype(dtype).device(device).requires_grad(false));
    torch::Tensor v = torch::zeros(
        {num_actual_tokens + padding_size, num_v_heads / tp_size, head_v_dim},
        torch::dtype(dtype).device(device).requires_grad(false));
    torch::Tensor b = torch::zeros(
        {num_v_heads / tp_size, num_actual_tokens + padding_size},
        torch::dtype(torch::kFloat32).device(device).requires_grad(false));
    torch::Tensor a = torch::zeros(
        {num_v_heads / tp_size, num_actual_tokens + padding_size},
        torch::dtype(torch::kFloat32).device(device).requires_grad(false));

    gdn::chunk_causal_conv1d_xe2(
        queue,
        q,
        k,
        v,
        z,
        b,
        a,
        projected_states_qkvz,
        projected_states_ba,
        conv_weights,
        conv_bias,
        conv_state,
        non_spec_query_start_loc,
        non_spec_state_indices_tensor,
        has_initial_state,
        act_mode,
        pad_slot_id,
        num_prefills,
        num_decodes);

    chunk_gated_delta_rule_xe2(
        queue,
        core_attn_out,
        q,
        k,
        v,
        b,
        a,
        A_log,
        dt_bias,
        ssm_state,
        non_spec_query_start_loc,
        non_spec_state_indices_tensor,
        has_initial_state,
        num_prefills,
        num_decodes);
  } else {
    NATIVE_LAUNCHER;
  }
#else
  NATIVE_LAUNCHER;
#endif
#undef NATIVE_LAUNCHER
}