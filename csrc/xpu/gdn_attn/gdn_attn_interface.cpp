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
    const int64_t tp_size,
    const bool reorder_input,
    // Spec-decoding extension. When provided together (both non-empty), the
    // op evaluates K candidate tokens per sequence and uses the per-token
    // slot ring; mirrors FLA Triton IS_SPEC_DECODING.
    const std::optional<torch::Tensor>& spec_state_indices_tensor,
    const std::optional<torch::Tensor>& num_accepted_tokens) {
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

  // Resolve spec-decoding state. Both tensors must be set or both unset;
  // mismatched is rejected. When set, route through the native kernels
  // with IS_SPEC=true (the chunk path handles non-spec prefill only).
  const bool is_spec =
      spec_state_indices_tensor.has_value() && num_accepted_tokens.has_value();
  TORCH_CHECK(
      spec_state_indices_tensor.has_value() == num_accepted_tokens.has_value(),
      "spec_state_indices_tensor and num_accepted_tokens must both be "
      "provided or both omitted");
  const int* spec_state_indices_ptr = nullptr;
  const int* num_accepted_tokens_ptr = nullptr;
  int spec_stride = 0;
  if (is_spec) {
    const auto& spec_idx = spec_state_indices_tensor.value();
    const auto& num_acc = num_accepted_tokens.value();
    TORCH_CHECK(
        spec_idx.is_contiguous(),
        "spec_state_indices_tensor must be contiguous");
    TORCH_CHECK(
        num_acc.is_contiguous(), "num_accepted_tokens must be contiguous");
    TORCH_CHECK(
        spec_idx.dim() == 2,
        "spec_state_indices_tensor must be 2D [num_seqs, K]");
    TORCH_CHECK(
        num_acc.dim() == 1, "num_accepted_tokens must be 1D [num_seqs]");
    // Both tensors are reinterpret_cast<int*> on the kernel side; require
    // int32 explicitly so a runaway int64 caller fails loudly instead of
    // silently misreading bytes.
    TORCH_CHECK(
        spec_idx.scalar_type() == at::kInt,
        "spec_state_indices_tensor must be int32, got ",
        spec_idx.scalar_type());
    TORCH_CHECK(
        num_acc.scalar_type() == at::kInt,
        "num_accepted_tokens must be int32, got ",
        num_acc.scalar_type());
    TORCH_CHECK(
        spec_idx.size(0) == num_acc.size(0),
        "spec_state_indices_tensor.size(0) (",
        spec_idx.size(0),
        ") must match num_accepted_tokens.size(0) (",
        num_acc.size(0),
        ")");
    TORCH_CHECK(
        spec_idx.size(0) == non_spec_query_start_loc.size(0) - 1,
        "spec_state_indices_tensor.size(0) (",
        spec_idx.size(0),
        ") must equal query_start_loc.size(0) - 1 (",
        non_spec_query_start_loc.size(0) - 1,
        "); pass spec_query_start_loc as non_spec_query_start_loc when "
        "spec args are set");
    spec_state_indices_ptr = reinterpret_cast<int*>(spec_idx.data_ptr());
    num_accepted_tokens_ptr = reinterpret_cast<int*>(num_acc.data_ptr());
    spec_stride = spec_idx.size(1);
  }

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
        z_active,                                                 \
        b,                                                        \
        a,                                                        \
        projected_states_qkvz_active,                             \
        projected_states_ba_active,                               \
        conv_weights,                                             \
        conv_bias,                                                \
        conv_state,                                               \
        non_spec_query_start_loc,                                 \
        non_spec_state_indices_tensor,                            \
        has_initial_state,                                        \
        act_mode,                                                 \
        pad_slot_id,                                              \
        num_prefills,                                             \
        num_decodes,                                              \
        reorder_input,                                            \
        is_spec,                                                  \
        spec_state_indices_ptr,                                   \
        num_accepted_tokens_ptr,                                  \
        spec_stride);                                             \
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
        non_spec_state_indices_tensor,                            \
        has_initial_state,                                        \
        num_prefills,                                             \
        num_decodes,                                              \
        is_spec,                                                  \
        spec_state_indices_ptr,                                   \
        num_accepted_tokens_ptr,                                  \
        spec_stride);                                             \
  } while (0)

#ifdef VLLM_XPU_ENABLE_XE2
  // Spec-decoding always routes through the native kernels: even when
  // seq_len > 1 (K=2..3 candidates) the spec slot ring needs the per-token
  // writeback path, which the chunk kernels don't implement.
  if (!is_spec && num_prefills > 0) {
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
        z_active,
        b,
        a,
        projected_states_qkvz_active,
        projected_states_ba_active,
        conv_weights,
        conv_bias,
        conv_state,
        non_spec_query_start_loc,
        non_spec_state_indices_tensor,
        has_initial_state,
        act_mode,
        pad_slot_id,
        num_prefills,
        num_decodes,
        reorder_input);

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