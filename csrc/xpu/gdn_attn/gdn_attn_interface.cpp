#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "utils.h"
#include "dispatch_utils.h"

#include "causal_conv1d.hpp"
#include "gated_delta_rule.hpp"
#ifdef VLLM_XPU_ENABLE_XE2
  #include "xe_2/chunk_causal_conv1d_xe2.hpp"
  #include "xe_2/chunk_causal_conv1d_tiled_xe2.hpp"
  #include "xe_2/chunk_gated_delta_rule_xe2.h"
#endif

void gdn_attention(
    torch::Tensor& core_attn_out,  // [num_actual_tokens, num_v_heads / tp_size,
                                   // head_v_dim]
    torch::Tensor& z,  // [num_actual_tokens, num_v_heads / tp_size, head_v_dim]
    const torch::Tensor&
        projected_states_qkvz,  // [num_actual_tokens, num_k_heads / tp_size *
                                // (2 * head_k_dim + 2 * head_v_dim *
                                // num_v_heads / num_k_heads)]
    const torch::Tensor&
        projected_states_ba,  // [num_actual_tokens, num_k_heads / tp_size * (2
                              // * num_v_heads / num_k_heads)]
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
    const int64_t num_spec_decodes,
    const std::optional<torch::Tensor>&
        has_initial_state,  // [num_prefills] or None
    const std::optional<torch::Tensor>&
        non_spec_query_start_loc,  // [num_prefills + num_decodes + 1]
    const std::optional<torch::Tensor>&
        non_spec_token_indx,  // [non_spec_token]
    const std::optional<torch::Tensor>&
        non_spec_state_indices_tensor,  // [num_prefills + num_decodes]
    const std::optional<torch::Tensor>&
        spec_query_start_loc,  // [num_spec_decodes + 1]
    const std::optional<torch::Tensor>& spec_token_indx,  // [spec_token]
    const std::optional<torch::Tensor>&
        spec_state_indices_tensor,  // [num_spec_decodes, num_speculative_tokens
                                    // + 1]
    const std::optional<torch::Tensor>&
        num_accepted_tokens,  // [num_spec_decodes]
    const int64_t num_actual_tokens,
    const int64_t tp_size,
    const bool reorder_input) {
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

  int non_spec_token = 0;
  if (num_prefills + num_decodes > 0) {
    if (has_initial_state.has_value()) {
      TORCH_CHECK(
          has_initial_state->is_contiguous(),
          "has_initial_state must be contiguous");
      TORCH_CHECK(
          has_initial_state->dtype() == torch::kBool,
          "has_initial_state must be of bool dtype");
      TORCH_CHECK(
          has_initial_state->dim() == 1,
          "has_initial_state must be 1D of shape [num_prefills + num_decodes]");
      TORCH_CHECK(
          num_prefills + num_decodes > 0 &&
              has_initial_state->size(0) == num_prefills + num_decodes,
          "has_initial_state must have size [num_prefills + num_decodes]");
    }

    TORCH_CHECK(
        non_spec_query_start_loc->is_contiguous(),
        "non_spec_query_start_loc must be contiguous");
    TORCH_CHECK(
        non_spec_query_start_loc->dtype() == torch::kInt32,
        "non_spec_query_start_loc must be of int32 dtype");
    TORCH_CHECK(
        non_spec_query_start_loc->dim() == 1,
        "non_spec_query_start_loc must be 1D of shape [num_prefills + "
        "num_decodes + 1]");
    TORCH_CHECK(
        non_spec_query_start_loc->size(0) == num_prefills + num_decodes + 1,
        "non_spec_query_start_loc must have size [num_prefills + num_decodes + "
        "1]");

    if (non_spec_token_indx.has_value()) {
      TORCH_CHECK(
          non_spec_token_indx->is_contiguous(),
          "non_spec_token_indx must be contiguous");
      TORCH_CHECK(
          non_spec_token_indx->dtype() == torch::kInt32,
          "non_spec_token_indx must be of int32 dtype");
      TORCH_CHECK(
          non_spec_token_indx->dim() == 1,
          "non_spec_token_indx must be 1D of shape [non_spec_token]");
      non_spec_token = non_spec_token_indx->size(0);
    } else {
      non_spec_token = num_actual_tokens;
    }

    TORCH_CHECK(
        non_spec_state_indices_tensor->is_contiguous(),
        "non_spec_state_indices_tensor must be contiguous");
    TORCH_CHECK(
        non_spec_state_indices_tensor->dtype() == torch::kInt32,
        "non_spec_state_indices_tensor must be of int32 dtype");
    TORCH_CHECK(
        non_spec_state_indices_tensor->dim() == 1,
        "non_spec_state_indices_tensor must be 1D of shape [num_prefills + "
        "num_decodes]");
    TORCH_CHECK(
        num_prefills + num_decodes > 0 && non_spec_state_indices_tensor->size(
                                              0) == num_prefills + num_decodes,
        "non_spec_state_indices_tensor must have size [num_prefills + "
        "num_decodes, non_spec_token]");
  }

  int spec_token = 0;
  int num_speculative_tokens = 0;
  if (num_spec_decodes > 0) {
    TORCH_CHECK(
        spec_query_start_loc->is_contiguous(),
        "spec_query_start_loc must be contiguous");
    TORCH_CHECK(
        spec_query_start_loc->dtype() == torch::kInt32,
        "spec_query_start_loc must be of int32 dtype");
    TORCH_CHECK(
        spec_query_start_loc->dim() == 1,
        "spec_query_start_loc must be 1D of shape [num_spec_decodes + 1]");
    TORCH_CHECK(
        spec_query_start_loc->size(0) >= num_spec_decodes + 1,
        "spec_query_start_loc must have size at least [num_spec_decodes + 1]");

    TORCH_CHECK(
        spec_token_indx->is_contiguous(), "spec_token_indx must be contiguous");
    TORCH_CHECK(
        spec_token_indx->dtype() == torch::kInt32,
        "spec_token_indx must be of int32 dtype");
    TORCH_CHECK(
        spec_token_indx->dim() == 1,
        "spec_token_indx must be 1D of shape [spec_token]");
    spec_token = spec_token_indx->size(0);

    TORCH_CHECK(
        spec_state_indices_tensor->is_contiguous(),
        "spec_state_indices_tensor must be contiguous");
    TORCH_CHECK(
        spec_state_indices_tensor->dtype() == torch::kInt32,
        "spec_state_indices_tensor must be of int32 dtype");
    TORCH_CHECK(
        spec_state_indices_tensor->dim() == 2,
        "spec_state_indices_tensor must be 2D of shape [num_spec_decodes, "
        "num_speculative_tokens + 1]");
    TORCH_CHECK(
        num_spec_decodes > 0 &&
            spec_state_indices_tensor->size(0) >= num_spec_decodes,
        "spec_state_indices_tensor must have size at least [num_spec_decodes, "
        "num_speculative_tokens + 1]");
    num_speculative_tokens = spec_state_indices_tensor->size(1) - 1;

    TORCH_CHECK(
        num_accepted_tokens->is_contiguous(),
        "num_accepted_tokens must be contiguous");
    TORCH_CHECK(
        num_accepted_tokens->dtype() == torch::kInt32,
        "num_accepted_tokens must be of int32 dtype");
    TORCH_CHECK(
        num_accepted_tokens->dim() == 1,
        "num_accepted_tokens must be 1D of shape [num_spec_decodes]");
    TORCH_CHECK(
        num_accepted_tokens->size(0) >= num_spec_decodes,
        "num_accepted_tokens size must be at least num_spec_decodes");
  }

  TORCH_CHECK(spec_token == num_spec_decodes * (num_speculative_tokens + 1));
  TORCH_CHECK(non_spec_token + spec_token == num_actual_tokens);

  // check core_attn_out / z / projected_states_{qkvz,ba} shapes.
  // Callers running under torch.compile + cudagraph capture pad the leading
  // dim to the captured graph size while num_actual_tokens stays at the real
  // (unpadded) count, so accept size(0) >= num_actual_tokens and narrow to the
  // active prefix below.
  TORCH_CHECK(
      core_attn_out.size(0) >= num_actual_tokens,
      "core_attn_out.size(0) (",
      core_attn_out.size(0),
      ") must be >= num_actual_tokens (",
      num_actual_tokens,
      ")");
  TORCH_CHECK(core_attn_out.size(1) == num_v_heads / tp_size);
  TORCH_CHECK(core_attn_out.size(2) == head_v_dim);

  TORCH_CHECK(z.size(0) == core_attn_out.size(0));
  TORCH_CHECK(z.size(1) == core_attn_out.size(1));
  TORCH_CHECK(z.size(2) == core_attn_out.size(2));

  TORCH_CHECK(
      projected_states_qkvz.size(0) >= num_actual_tokens,
      "projected_states_qkvz.size(0) (",
      projected_states_qkvz.size(0),
      ") must be >= num_actual_tokens (",
      num_actual_tokens,
      ")");
  TORCH_CHECK(
      projected_states_qkvz.size(1) ==
      num_k_heads / tp_size *
          (2 * head_k_dim + 2 * head_v_dim * num_v_heads / num_k_heads));

  TORCH_CHECK(
      projected_states_ba.size(0) >= num_actual_tokens,
      "projected_states_ba.size(0) (",
      projected_states_ba.size(0),
      ") must be >= num_actual_tokens (",
      num_actual_tokens,
      ")");
  TORCH_CHECK(projected_states_ba.size(1) == 2 * num_v_heads / tp_size);

  // Narrowing dim 0 of a contiguous tensor yields a contiguous view that
  // shares storage with the original, so writes through core_attn_out_active
  // land in the caller's buffer at offsets [0, num_actual_tokens); padded
  // trailing slots are left untouched (matches the CUDA path in
  // gdn_linear_attn.py).
  auto core_attn_out_active = core_attn_out.narrow(0, 0, num_actual_tokens);
  auto z_active = z.narrow(0, 0, num_actual_tokens);
  auto projected_states_qkvz_active =
      projected_states_qkvz.narrow(0, 0, num_actual_tokens);
  auto projected_states_ba_active =
      projected_states_ba.narrow(0, 0, num_actual_tokens);

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

  std::optional<torch::Tensor> empty_tensor{std::nullopt};

  if (spec_token > 0) {
    std::optional<torch::Tensor> spec_query_start_loc_active{
        spec_query_start_loc->narrow(0, 0, num_spec_decodes + 1)};
    std::optional<torch::Tensor> spec_state_indices_tensor_active{
        spec_state_indices_tensor->narrow(0, 0, num_spec_decodes)};
    std::optional<torch::Tensor> num_accepted_tokens_active{
        num_accepted_tokens->narrow(0, 0, num_spec_decodes)};

    torch::Tensor q = torch::empty(
        {spec_token, num_k_heads / tp_size, head_k_dim},
        torch::dtype(dtype).device(device).requires_grad(false));
    torch::Tensor k = torch::empty(
        {spec_token, num_k_heads / tp_size, head_k_dim},
        torch::dtype(dtype).device(device).requires_grad(false));
    torch::Tensor v = torch::empty(
        {spec_token, num_v_heads / tp_size, head_v_dim},
        torch::dtype(dtype).device(device).requires_grad(false));
    torch::Tensor b = torch::empty(
        {spec_token, num_v_heads / tp_size},
        torch::dtype(dtype).device(device).requires_grad(false));
    torch::Tensor a = torch::empty(
        {spec_token, num_v_heads / tp_size},
        torch::dtype(dtype).device(device).requires_grad(false));

    gdn::causal_conv1d(
        queue,
        q,
        k,
        v,
        z_active,
        b,
        a,
        projected_states_qkvz_active,
        projected_states_ba_active,
        conv_weights,
        conv_bias,
        conv_state,
        spec_query_start_loc_active,
        spec_token_indx,
        spec_state_indices_tensor_active,
        empty_tensor,
        num_accepted_tokens_active,
        act_mode,
        pad_slot_id,
        num_prefills,
        num_decodes,
        num_spec_decodes,
        reorder_input);
    gdn::gated_delta_rule(
        queue,
        core_attn_out_active,
        q,
        k,
        v,
        b,
        a,
        A_log,
        dt_bias,
        ssm_state,
        spec_query_start_loc_active,
        spec_token_indx,
        spec_state_indices_tensor_active,
        empty_tensor,
        num_accepted_tokens_active,
        num_prefills,
        num_decodes,
        num_spec_decodes);
  }

  if (non_spec_token > 0) {
#define NATIVE_LAUNCHER                                           \
  do {                                                            \
    torch::Tensor q = torch::empty(                               \
        {non_spec_token, num_k_heads / tp_size, head_k_dim},      \
        torch::dtype(dtype).device(device).requires_grad(false)); \
    torch::Tensor k = torch::empty(                               \
        {non_spec_token, num_k_heads / tp_size, head_k_dim},      \
        torch::dtype(dtype).device(device).requires_grad(false)); \
    torch::Tensor v = torch::empty(                               \
        {non_spec_token, num_v_heads / tp_size, head_v_dim},      \
        torch::dtype(dtype).device(device).requires_grad(false)); \
    torch::Tensor b = torch::empty(                               \
        {non_spec_token, num_v_heads / tp_size},                  \
        torch::dtype(dtype).device(device).requires_grad(false)); \
    torch::Tensor a = torch::empty(                               \
        {non_spec_token, num_v_heads / tp_size},                  \
        torch::dtype(dtype).device(device).requires_grad(false)); \
    gdn::causal_conv1d(                                           \
        queue,                                                    \
        q,                                                        \
        k,                                                        \
        v,                                                        \
        z_active,                                                 \
        b,                                                        \
        a,                                                        \
        projected_states_qkvz_active,                             \
        projected_states_ba_active,                               \
        conv_weights,                                             \
        conv_bias,                                                \
        conv_state,                                               \
        non_spec_query_start_loc,                                 \
        non_spec_token_indx,                                      \
        non_spec_state_indices_tensor,                            \
        has_initial_state,                                        \
        empty_tensor,                                             \
        act_mode,                                                 \
        pad_slot_id,                                              \
        num_prefills,                                             \
        num_decodes,                                              \
        num_spec_decodes,                                         \
        reorder_input);                                           \
    gdn::gated_delta_rule(                                        \
        queue,                                                    \
        core_attn_out_active,                                     \
        q,                                                        \
        k,                                                        \
        v,                                                        \
        b,                                                        \
        a,                                                        \
        A_log,                                                    \
        dt_bias,                                                  \
        ssm_state,                                                \
        non_spec_query_start_loc,                                 \
        non_spec_token_indx,                                      \
        non_spec_state_indices_tensor,                            \
        has_initial_state,                                        \
        empty_tensor,                                             \
        num_prefills,                                             \
        num_decodes,                                              \
        num_spec_decodes);                                        \
  } while (0)

#ifdef VLLM_XPU_ENABLE_XE2
    // XE2 chunk path handles all non-spec tokens whenever there are prefills,
    // even when spec_decodes are also present. The XE2 kernels accept an
    // optional token_indx so they can read mixed_qkvz/mixed_ba and write z /
    // core_attn_out directly at the interleaved global slots indicated by
    // non_spec_token_indx, avoiding host-side gather/scatter.
    if (num_prefills > 0) {
      int batch_size = non_spec_query_start_loc->size(0) - 1;
      int padding_size = batch_size * (gdn::chunk_size_xe2 - 1);

      const int* token_indx_ptr =
          non_spec_token_indx.has_value()
              ? reinterpret_cast<const int*>(non_spec_token_indx->data_ptr())
              : nullptr;

      torch::Tensor q = torch::zeros(
          {non_spec_token + padding_size, num_k_heads / tp_size, head_k_dim},
          torch::dtype(dtype).device(device).requires_grad(false));
      torch::Tensor k = torch::zeros(
          {non_spec_token + padding_size, num_k_heads / tp_size, head_k_dim},
          torch::dtype(dtype).device(device).requires_grad(false));
      torch::Tensor v = torch::zeros(
          {non_spec_token + padding_size, num_v_heads / tp_size, head_v_dim},
          torch::dtype(dtype).device(device).requires_grad(false));
      torch::Tensor b = torch::zeros(
          {num_v_heads / tp_size, non_spec_token + padding_size},
          torch::dtype(torch::kFloat32).device(device).requires_grad(false));
      torch::Tensor a = torch::zeros(
          {num_v_heads / tp_size, non_spec_token + padding_size},
          torch::dtype(torch::kFloat32).device(device).requires_grad(false));

      // Use SLM-tiled conv1d for prefill when token count is large enough
      // to benefit from the tiling overhead (tile_size=8).
      if (non_spec_token >= gdn::conv1d_tile_size) {
        gdn::chunk_causal_conv1d_tiled_xe2(
            queue,
            q,
            k,
            v,
            z_active,
            b,
            a,
            projected_states_qkvz_active,
            projected_states_ba_active,
            conv_weights,
            conv_bias,
            conv_state,
            *non_spec_query_start_loc,
            *non_spec_state_indices_tensor,
            has_initial_state,
            act_mode,
            pad_slot_id,
            num_prefills,
            num_decodes,
            reorder_input,
            token_indx_ptr,
            non_spec_token);
      } else {
        gdn::chunk_causal_conv1d_xe2(
            queue,
            q,
            k,
            v,
            z_active,
            b,
            a,
            projected_states_qkvz_active,
            projected_states_ba_active,
            conv_weights,
            conv_bias,
            conv_state,
            *non_spec_query_start_loc,
            *non_spec_state_indices_tensor,
            has_initial_state,
            act_mode,
            pad_slot_id,
            num_prefills,
            num_decodes,
            reorder_input,
            token_indx_ptr,
            non_spec_token);
      }

      chunk_gated_delta_rule_xe2(
          queue,
          core_attn_out_active,
          q,
          k,
          v,
          b,
          a,
          A_log,
          dt_bias,
          ssm_state,
          *non_spec_query_start_loc,
          *non_spec_state_indices_tensor,
          has_initial_state,
          num_prefills,
          num_decodes,
          token_indx_ptr);
    } else {
      NATIVE_LAUNCHER;
    }
#else
    NATIVE_LAUNCHER;
#endif
#undef NATIVE_LAUNCHER
  }
}
