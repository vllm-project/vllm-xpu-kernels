#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "utils.h"
#include "dispatch_utils.h"

#include "causal_conv1d.hpp"
#include "gated_delta_rule.hpp"
#ifdef VLLM_XPU_ENABLE_XE2
  #include "xe_2/chunk_causal_conv1d_xe2.hpp"
  #include "xe_2/chunk_causal_conv1d_tiled_xe2.hpp"
  #include "xe_2/l2norm.h"
  #include "xe_2/chunk_gated_delta_rule_xe2.h"
#endif

// Conv stage of GatedDeltaNet linear attention, spec-decode path. Runs the
// causal conv1d (plus activation) for the spec-decode tokens, scatters z,
// advances the per-seq conv cache, and returns the intermediate
// {q, k, v, b, a} tensors consumed by gated_delta_rule_spec. Only spec tokens
// are processed; spec_token may be < num_actual_tokens when a non-spec group
// coexists in the same batch (handled by causal_conv1d_non_spec).
std::vector<torch::Tensor> causal_conv1d_spec(
    torch::Tensor& z,  // [num_actual_tokens, num_v_heads / tp_size, head_v_dim]
    const torch::Tensor& projected_states_qkvz,
    const torch::Tensor& projected_states_ba,
    const int64_t num_k_heads,
    const int64_t num_v_heads,
    const int64_t head_k_dim,
    const int64_t head_v_dim,
    torch::Tensor& conv_state,
    const torch::Tensor& conv_weights,
    const std::optional<torch::Tensor>& conv_bias,
    const std::string& activation,
    const int64_t num_prefills,
    const int64_t num_decodes,
    const int64_t num_spec_decodes,
    const torch::Tensor& spec_query_start_loc,
    const torch::Tensor& spec_token_indx,
    const torch::Tensor& spec_state_indices_tensor,
    const torch::Tensor& num_accepted_tokens,
    const int64_t num_actual_tokens,
    const int64_t tp_size,
    const bool reorder_input) {
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
  TORCH_CHECK(conv_weights.is_contiguous(), "conv_weights must be contiguous");

  TORCH_CHECK(
      num_spec_decodes > 0, "causal_conv1d_spec requires num_spec_decodes > 0");

  TORCH_CHECK(
      spec_query_start_loc.is_contiguous(),
      "spec_query_start_loc must be contiguous");
  TORCH_CHECK(
      spec_query_start_loc.dtype() == torch::kInt32,
      "spec_query_start_loc must be of int32 dtype");
  TORCH_CHECK(
      spec_query_start_loc.dim() == 1,
      "spec_query_start_loc must be 1D of shape [num_spec_decodes + 1]");
  TORCH_CHECK(
      spec_query_start_loc.size(0) == num_spec_decodes + 1,
      "spec_query_start_loc must have size [num_spec_decodes + 1]");

  TORCH_CHECK(
      spec_token_indx.is_contiguous(), "spec_token_indx must be contiguous");
  TORCH_CHECK(
      spec_token_indx.dtype() == torch::kInt32,
      "spec_token_indx must be of int32 dtype");
  TORCH_CHECK(
      spec_token_indx.dim() == 1,
      "spec_token_indx must be 1D of shape [spec_token]");
  const int spec_token = spec_token_indx.size(0);

  TORCH_CHECK(
      spec_state_indices_tensor.is_contiguous(),
      "spec_state_indices_tensor must be contiguous");
  TORCH_CHECK(
      spec_state_indices_tensor.dtype() == torch::kInt32,
      "spec_state_indices_tensor must be of int32 dtype");
  TORCH_CHECK(
      spec_state_indices_tensor.dim() == 2,
      "spec_state_indices_tensor must be 2D of shape [num_spec_decodes, "
      "num_speculative_tokens + 1]");
  TORCH_CHECK(
      spec_state_indices_tensor.size(0) == num_spec_decodes,
      "spec_state_indices_tensor must have size [num_spec_decodes, "
      "num_speculative_tokens + 1]");
  const int num_speculative_tokens = spec_state_indices_tensor.size(1) - 1;
  const int64_t required_conv_state_len =
      conv_weights.size(1) - 1 + num_speculative_tokens;
  TORCH_CHECK(
      conv_state.size(1) >= required_conv_state_len,
      "conv_state must have at least ",
      required_conv_state_len,
      " rows for speculative decoding, but got ",
      conv_state.size(1));

  TORCH_CHECK(
      num_accepted_tokens.is_contiguous(),
      "num_accepted_tokens must be contiguous");
  TORCH_CHECK(
      num_accepted_tokens.dtype() == torch::kInt32,
      "num_accepted_tokens must be of int32 dtype");
  TORCH_CHECK(
      num_accepted_tokens.dim() == 1,
      "num_accepted_tokens must be 1D of shape [num_spec_decodes]");
  TORCH_CHECK(
      num_accepted_tokens.size(0) == num_spec_decodes,
      "num_accepted_tokens size must be num_spec_decodes");

  TORCH_CHECK(spec_token == num_spec_decodes * (num_speculative_tokens + 1));
  TORCH_CHECK(spec_token > 0, "spec_token must be > 0 for causal_conv1d_spec");

  TORCH_CHECK(
      z.size(0) >= num_actual_tokens,
      "z.size(0) (",
      z.size(0),
      ") must be >= num_actual_tokens (",
      num_actual_tokens,
      ")");
  TORCH_CHECK(z.size(1) == num_v_heads / tp_size);
  TORCH_CHECK(z.size(2) == head_v_dim);

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

  // Narrow the (possibly cudagraph-padded) leading dim to the active prefix so
  // the conv kernels never read or write padded rows.
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
      spec_query_start_loc,
      spec_token_indx,
      spec_state_indices_tensor,
      empty_tensor,
      num_accepted_tokens,
      act_mode,
      pad_slot_id,
      num_prefills,
      num_decodes,
      num_spec_decodes,
      reorder_input);

  return {q, k, v, b, a};
}

// Conv stage of GatedDeltaNet linear attention, non-spec (prefill + decode)
// path. Runs the causal conv1d (plus activation) for the non-spec tokens,
// writes z, advances the per-seq conv cache, and returns the intermediate
// {q, k, v, b, a} tensors consumed by gated_delta_rule_non_spec. Uses the XE2
// chunk kernel when prefills are present (and XE2 is enabled), otherwise the
// native kernel. Only non-spec tokens are processed; non_spec_token may be
// < num_actual_tokens when a spec-decode group coexists in the same batch
// (handled by causal_conv1d_spec).
std::vector<torch::Tensor> causal_conv1d_non_spec(
    torch::Tensor& z,  // [num_actual_tokens, num_v_heads / tp_size, head_v_dim]
    const torch::Tensor& projected_states_qkvz,
    const torch::Tensor& projected_states_ba,
    const int64_t num_k_heads,
    const int64_t num_v_heads,
    const int64_t head_k_dim,
    const int64_t head_v_dim,
    torch::Tensor& conv_state,
    const torch::Tensor& conv_weights,
    const std::optional<torch::Tensor>& conv_bias,
    const std::string& activation,
    const int64_t num_prefills,
    const int64_t num_decodes,
    const int64_t num_spec_decodes,
    const std::optional<torch::Tensor>& has_initial_state,
    const torch::Tensor& non_spec_query_start_loc,
    const std::optional<torch::Tensor>& non_spec_token_indx,
    const torch::Tensor& non_spec_state_indices_tensor,
    const int64_t num_actual_tokens,
    const int64_t tp_size,
    const bool reorder_input) {
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
  TORCH_CHECK(conv_weights.is_contiguous(), "conv_weights must be contiguous");

  TORCH_CHECK(
      num_prefills + num_decodes > 0,
      "causal_conv1d_non_spec requires num_prefills + num_decodes > 0");

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
        has_initial_state->size(0) == num_prefills + num_decodes,
        "has_initial_state must have size [num_prefills + num_decodes]");
  }

  TORCH_CHECK(
      non_spec_query_start_loc.is_contiguous(),
      "non_spec_query_start_loc must be contiguous");
  TORCH_CHECK(
      non_spec_query_start_loc.dtype() == torch::kInt32,
      "non_spec_query_start_loc must be of int32 dtype");
  TORCH_CHECK(
      non_spec_query_start_loc.dim() == 1,
      "non_spec_query_start_loc must be 1D of shape [num_prefills + "
      "num_decodes + 1]");
  TORCH_CHECK(
      non_spec_query_start_loc.size(0) == num_prefills + num_decodes + 1,
      "non_spec_query_start_loc must have size [num_prefills + num_decodes + "
      "1]");

  int non_spec_token = num_actual_tokens;
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
  }

  TORCH_CHECK(
      non_spec_state_indices_tensor.is_contiguous(),
      "non_spec_state_indices_tensor must be contiguous");
  TORCH_CHECK(
      non_spec_state_indices_tensor.dtype() == torch::kInt32,
      "non_spec_state_indices_tensor must be of int32 dtype");
  TORCH_CHECK(
      non_spec_state_indices_tensor.dim() == 1,
      "non_spec_state_indices_tensor must be 1D of shape [num_prefills + "
      "num_decodes]");
  TORCH_CHECK(
      non_spec_state_indices_tensor.size(0) == num_prefills + num_decodes,
      "non_spec_state_indices_tensor must have size [num_prefills + "
      "num_decodes]");

  TORCH_CHECK(
      non_spec_token > 0,
      "non_spec_token must be > 0 for causal_conv1d_non_spec");

  TORCH_CHECK(
      z.size(0) >= num_actual_tokens,
      "z.size(0) (",
      z.size(0),
      ") must be >= num_actual_tokens (",
      num_actual_tokens,
      ")");
  TORCH_CHECK(z.size(1) == num_v_heads / tp_size);
  TORCH_CHECK(z.size(2) == head_v_dim);

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

  // Narrow the (possibly cudagraph-padded) leading dim to the active prefix so
  // the conv kernels never read or write padded rows.
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

#define NATIVE_CONV_LAUNCHER                                      \
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
    return {q, k, v, b, a};                                       \
  } while (0)

#ifdef VLLM_XPU_ENABLE_XE2
  if (num_prefills > 0) {
    int batch_size = non_spec_query_start_loc.size(0) - 1;
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

    // Determine whether fused l2norm is valid for the chosen conv1d path.
    // Tiled kernel: only valid when all Q+K features fit in a single
    // feature chunk, i.e. 2 * head_k_dim <= feats_per_wg (256).
    // Untiled kernel: always valid (entire QKV in one WG).
    constexpr int tiled_feats_per_wg = 256;  // wg_size(64) * elems_per_item(4)
    const bool use_tiled = (non_spec_token >= gdn::conv1d_tile_size);
    const bool fuse_l2norm =
        use_tiled ? (2 * head_k_dim <= tiled_feats_per_wg) : true;

    if (use_tiled) {
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
          non_spec_query_start_loc,
          non_spec_state_indices_tensor,
          has_initial_state,
          act_mode,
          pad_slot_id,
          num_prefills,
          num_decodes,
          reorder_input,
          token_indx_ptr,
          non_spec_token,
          fuse_l2norm);
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
          non_spec_query_start_loc,
          non_spec_state_indices_tensor,
          has_initial_state,
          act_mode,
          pad_slot_id,
          num_prefills,
          num_decodes,
          reorder_input,
          token_indx_ptr,
          non_spec_token,
          fuse_l2norm);
    }

    // Run standalone l2norm kernel when not fused into conv1d
    if (!fuse_l2norm) {
      l2norm(queue, q, k);
    }

    return {q, k, v, b, a};
  } else {
    NATIVE_CONV_LAUNCHER;
  }
#else
  NATIVE_CONV_LAUNCHER;
#endif
#undef NATIVE_CONV_LAUNCHER
}

// Delta stage of GatedDeltaNet linear attention. Consumes the intermediate
// {q, k, v, b, a} tensors produced by causal_conv1d, advances ssm_state, and
// writes core_attn_out. A single fused invocation may contain both a
// spec-decode group and a non-spec (prefill + decode) group. The delta stage is
// split into two ops -- gated_delta_rule_spec and gated_delta_rule_non_spec --
// each taking only the {q, k, v, b, a} intermediates and index tensors of its
// own path, so neither carries the other path's optional arguments. The legacy
// fused gdn_attention drives whichever path(s) are active.

// Spec-decode delta stage: consumes the spec {q, k, v, b, a} produced by
// causal_conv1d and writes core_attn_out / advances ssm_state via the native
// gated_delta_rule kernel.
void gated_delta_rule_spec(
    torch::Tensor& core_attn_out,  // [num_actual_tokens, num_v_heads / tp_size,
                                   // head_v_dim]
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& b,
    const torch::Tensor& a,
    const int64_t num_v_heads,
    const int64_t head_v_dim,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    torch::Tensor& ssm_state,
    const int64_t num_prefills,
    const int64_t num_decodes,
    const int64_t num_spec_decodes,
    const torch::Tensor& spec_query_start_loc,
    const torch::Tensor& spec_token_indx,
    const torch::Tensor& spec_state_indices_tensor,
    const torch::Tensor& num_accepted_tokens,
    const int64_t num_actual_tokens,
    const int64_t tp_size) {
  TORCH_CHECK(
      core_attn_out.is_contiguous(), "core_attn_out must be contiguous");
  TORCH_CHECK(
      ssm_state[0].is_contiguous(),
      "ssm_state of each batch must be contiguous");
  TORCH_CHECK(A_log.is_contiguous(), "A_log must be contiguous");
  TORCH_CHECK(dt_bias.is_contiguous(), "dt_bias must be contiguous");

  TORCH_CHECK(
      core_attn_out.size(0) >= num_actual_tokens,
      "core_attn_out.size(0) (",
      core_attn_out.size(0),
      ") must be >= num_actual_tokens (",
      num_actual_tokens,
      ")");
  TORCH_CHECK(core_attn_out.size(1) == num_v_heads / tp_size);
  TORCH_CHECK(core_attn_out.size(2) == head_v_dim);

  TORCH_CHECK(
      num_spec_decodes > 0,
      "gated_delta_rule_spec requires num_spec_decodes > 0");

  // spec_token may be < num_actual_tokens when a spec-decode group coexists
  // with a non-spec group in the same batch: each delta op processes only its
  // own tokens (located via spec_token_indx) while writing into the shared
  // core_attn_out. Requiring spec_token == num_actual_tokens would wrongly
  // reject that mixed case, so we only require it to be positive.
  const int spec_token = spec_token_indx.size(0);
  TORCH_CHECK(
      spec_token > 0, "spec_token must be > 0 for gated_delta_rule_spec");

  // Narrow the (possibly cudagraph-padded) leading dim to the active prefix.
  auto core_attn_out_active = core_attn_out.narrow(0, 0, num_actual_tokens);

  auto& queue = vllm::xpu::vllmGetQueue();
  std::optional<torch::Tensor> empty_tensor{std::nullopt};

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
      spec_query_start_loc,
      spec_token_indx,
      spec_state_indices_tensor,
      empty_tensor,
      num_accepted_tokens,
      num_prefills,
      num_decodes,
      num_spec_decodes);
}

// Non-spec (prefill + decode) delta stage: consumes the non-spec
// {q, k, v, b, a} produced by causal_conv1d and writes core_attn_out /
// advances ssm_state. Uses the XE2 chunk kernel when prefills are present
// (and XE2 is enabled), otherwise the native gated_delta_rule kernel.
void gated_delta_rule_non_spec(
    torch::Tensor& core_attn_out,  // [num_actual_tokens, num_v_heads / tp_size,
                                   // head_v_dim]
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& b,
    const torch::Tensor& a,
    const int64_t num_v_heads,
    const int64_t head_v_dim,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    torch::Tensor& ssm_state,
    const int64_t num_prefills,
    const int64_t num_decodes,
    const int64_t num_spec_decodes,
    const std::optional<torch::Tensor>& has_initial_state,
    const torch::Tensor& non_spec_query_start_loc,
    const std::optional<torch::Tensor>& non_spec_token_indx,
    const torch::Tensor& non_spec_state_indices_tensor,
    const int64_t num_actual_tokens,
    const int64_t tp_size) {
  TORCH_CHECK(
      core_attn_out.is_contiguous(), "core_attn_out must be contiguous");
  TORCH_CHECK(
      ssm_state[0].is_contiguous(),
      "ssm_state of each batch must be contiguous");
  TORCH_CHECK(A_log.is_contiguous(), "A_log must be contiguous");
  TORCH_CHECK(dt_bias.is_contiguous(), "dt_bias must be contiguous");

  TORCH_CHECK(
      core_attn_out.size(0) >= num_actual_tokens,
      "core_attn_out.size(0) (",
      core_attn_out.size(0),
      ") must be >= num_actual_tokens (",
      num_actual_tokens,
      ")");
  TORCH_CHECK(core_attn_out.size(1) == num_v_heads / tp_size);
  TORCH_CHECK(core_attn_out.size(2) == head_v_dim);

  TORCH_CHECK(
      num_prefills + num_decodes > 0,
      "gated_delta_rule_non_spec requires num_prefills + num_decodes > 0");

  // non_spec_token may be < num_actual_tokens when a non-spec group coexists
  // with a spec-decode group in the same batch: each delta op processes only
  // its own tokens (located via non_spec_token_indx) while writing into the
  // shared core_attn_out. Requiring non_spec_token == num_actual_tokens would
  // wrongly reject that mixed case, so we only require it to be positive.
  const int non_spec_token = non_spec_token_indx.has_value()
                                 ? non_spec_token_indx->size(0)
                                 : num_actual_tokens;
  TORCH_CHECK(
      non_spec_token > 0,
      "non_spec_token must be > 0 for gated_delta_rule_non_spec");

  // Narrow the (possibly cudagraph-padded) leading dim to the active prefix.
  auto core_attn_out_active = core_attn_out.narrow(0, 0, num_actual_tokens);

  auto& queue = vllm::xpu::vllmGetQueue();
  std::optional<torch::Tensor> empty_tensor{std::nullopt};

#ifdef VLLM_XPU_ENABLE_XE2
  if (num_prefills > 0) {
    // The XE2 delta kernel writes core_attn_out in chunk-contiguous blocks
    // after a SINGLE token_indx lookup per chunk (kernels_xe2.hpp O_tensor
    // epilogue) — it silently assumes token_indx values are run-contiguous
    // within every sequence. Mixed spec/non-spec batches produce shuffled
    // index sets, so rows land at wrong global positions (silent corruption
    // or overrun). Make the assumption true by construction: with token_indx
    // present, write into a compact staging buffer with identity indexing,
    // then scatter per-row.
    if (non_spec_token_indx.has_value()) {
      torch::Tensor o_compact = torch::zeros(
          {non_spec_token,
           core_attn_out_active.size(1),
           core_attn_out_active.size(2)},
          core_attn_out_active.options());
      chunk_gated_delta_rule_xe2(
          queue,
          o_compact,
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
          num_decodes,
          /*token_indx=*/nullptr);
      core_attn_out_active.index_copy_(
          0, non_spec_token_indx->to(torch::kLong), o_compact);
    } else {
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
          num_decodes,
          /*token_indx=*/nullptr);
    }
  } else {
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
        non_spec_query_start_loc,
        non_spec_token_indx,
        non_spec_state_indices_tensor,
        has_initial_state,
        empty_tensor,
        num_prefills,
        num_decodes,
        num_spec_decodes);
  }
#else
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
      non_spec_query_start_loc,
      non_spec_token_indx,
      non_spec_state_indices_tensor,
      has_initial_state,
      empty_tensor,
      num_prefills,
      num_decodes,
      num_spec_decodes);
#endif
}

// Legacy fused entry point kept for API backward compatibility. It performs
// the exact same work as the original gdn_attention by chaining the split ops:
// the conv stage (causal_conv1d_spec / causal_conv1d_non_spec) produces the
// {q,k,v,b,a} intermediates (and writes z / updates conv_state); the delta
// stage (gated_delta_rule_spec / gated_delta_rule_non_spec) consumes them (and
// writes core_attn_out / updates ssm_state). Both the spec-decode and non-spec
// paths may be active in a single call; each active path runs its own
// conv->delta pipeline, preserving the original fused-op semantics where both
// paths could coexist.
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
    const int64_t num_spec_decodes,
    const std::optional<torch::Tensor>& has_initial_state,
    const std::optional<torch::Tensor>& non_spec_query_start_loc,
    const std::optional<torch::Tensor>& non_spec_token_indx,
    const std::optional<torch::Tensor>& non_spec_state_indices_tensor,
    const std::optional<torch::Tensor>& spec_query_start_loc,
    const std::optional<torch::Tensor>& spec_token_indx,
    const std::optional<torch::Tensor>& spec_state_indices_tensor,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    const int64_t num_actual_tokens,
    const int64_t tp_size,
    const bool reorder_input) {
  // Determine which paths are active using the same conditions the conv/delta
  // ops use internally.
  int non_spec_token = 0;
  if (num_prefills + num_decodes > 0) {
    non_spec_token = non_spec_token_indx.has_value()
                         ? non_spec_token_indx->size(0)
                         : num_actual_tokens;
  }
  int spec_token = 0;
  if (num_spec_decodes > 0) {
    spec_token = spec_token_indx->size(0);
  }
  const bool spec_active = spec_token > 0;
  const bool non_spec_active = non_spec_token > 0;

  if (spec_active) {
    std::vector<torch::Tensor> intermediates = causal_conv1d_spec(
        z,
        projected_states_qkvz,
        projected_states_ba,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_state,
        conv_weights,
        conv_bias,
        activation,
        num_prefills,
        num_decodes,
        num_spec_decodes,
        *spec_query_start_loc,
        *spec_token_indx,
        *spec_state_indices_tensor,
        *num_accepted_tokens,
        num_actual_tokens,
        tp_size,
        reorder_input);

    gated_delta_rule_spec(
        core_attn_out,
        intermediates[0],
        intermediates[1],
        intermediates[2],
        intermediates[3],
        intermediates[4],
        num_v_heads,
        head_v_dim,
        A_log,
        dt_bias,
        ssm_state,
        num_prefills,
        num_decodes,
        num_spec_decodes,
        *spec_query_start_loc,
        *spec_token_indx,
        *spec_state_indices_tensor,
        *num_accepted_tokens,
        num_actual_tokens,
        tp_size);
  }

  if (non_spec_active) {
    std::vector<torch::Tensor> intermediates = causal_conv1d_non_spec(
        z,
        projected_states_qkvz,
        projected_states_ba,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_state,
        conv_weights,
        conv_bias,
        activation,
        num_prefills,
        num_decodes,
        num_spec_decodes,
        has_initial_state,
        *non_spec_query_start_loc,
        non_spec_token_indx,
        *non_spec_state_indices_tensor,
        num_actual_tokens,
        tp_size,
        reorder_input);

    gated_delta_rule_non_spec(
        core_attn_out,
        intermediates[0],
        intermediates[1],
        intermediates[2],
        intermediates[3],
        intermediates[4],
        num_v_heads,
        head_v_dim,
        A_log,
        dt_bias,
        ssm_state,
        num_prefills,
        num_decodes,
        num_spec_decodes,
        has_initial_state,
        *non_spec_query_start_loc,
        non_spec_token_indx,
        *non_spec_state_indices_tensor,
        num_actual_tokens,
        tp_size);
  }
}
