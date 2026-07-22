#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "kda_attention.hpp"
#include "utils.h"

namespace {

void check_int32_tensor(
    const std::optional<torch::Tensor>& tensor, const char* name, int64_t dim) {
  TORCH_CHECK(tensor.has_value(), name, " must be provided");
  TORCH_CHECK(tensor->is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor->scalar_type() == at::kInt, name, " must be int32");
  TORCH_CHECK(tensor->dim() == dim, name, " must be ", dim, "D");
}

template <typename T, typename CacheT>
void launch_kda(
    sycl::queue& queue,
    torch::Tensor& core_attn_out,
    const torch::Tensor& q_proj,
    const torch::Tensor& k_proj,
    const torch::Tensor& v_proj,
    const torch::Tensor& raw_gate,
    const torch::Tensor& beta,
    torch::Tensor& conv_state,
    torch::Tensor& recurrent_state,
    const torch::Tensor& q_conv_weight,
    const torch::Tensor& k_conv_weight,
    const torch::Tensor& v_conv_weight,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    const std::optional<torch::Tensor>& has_initial_state,
    const std::optional<torch::Tensor>& non_spec_query_start_loc,
    const std::optional<torch::Tensor>& non_spec_token_indx,
    const std::optional<torch::Tensor>& non_spec_state_indices,
    const std::optional<torch::Tensor>& spec_query_start_loc,
    const std::optional<torch::Tensor>& spec_token_indx,
    const std::optional<torch::Tensor>& spec_state_indices,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    int num_prefills,
    int num_decodes,
    int num_spec_decodes,
    int num_actual_tokens,
    int num_heads,
    int head_dim,
    int width,
    int64_t conv_state_dim_stride,
    int64_t conv_state_time_stride) {
  const int hidden_dim = num_heads * head_dim;
  auto options =
      torch::dtype(q_proj.dtype()).device(q_proj.device()).requires_grad(false);
  auto q = torch::empty({num_actual_tokens, hidden_dim}, options);
  auto k = torch::empty({num_actual_tokens, hidden_dim}, options);
  auto v = torch::empty({num_actual_tokens, hidden_dim}, options);

  T* q_ptr = reinterpret_cast<T*>(q.data_ptr());
  T* k_ptr = reinterpret_cast<T*>(k.data_ptr());
  T* v_ptr = reinterpret_cast<T*>(v.data_ptr());
  T* output_ptr = reinterpret_cast<T*>(core_attn_out.data_ptr());
  CacheT* conv_state_ptr = reinterpret_cast<CacheT*>(conv_state.data_ptr());
  float* recurrent_state_ptr =
      reinterpret_cast<float*>(recurrent_state.data_ptr());
  const T* raw_gate_ptr = reinterpret_cast<const T*>(raw_gate.data_ptr());
  const float* beta_ptr = reinterpret_cast<const float*>(beta.data_ptr());
  const float* a_log_ptr = reinterpret_cast<const float*>(a_log.data_ptr());
  const float* dt_bias_ptr = reinterpret_cast<const float*>(dt_bias.data_ptr());

#define LAUNCH_CONV(                                            \
    WIDTH,                                                      \
    IS_SPEC,                                                    \
    QUERY_START,                                                \
    TOKEN_INDX,                                                 \
    STATE_INDICES,                                              \
    STATE_STRIDE,                                               \
    HAS_INITIAL,                                                \
    ACCEPTED,                                                   \
    BATCH_SIZE)                                                 \
  kda::launch_causal_conv1d<T, CacheT, WIDTH, IS_SPEC>(         \
      queue,                                                    \
      q_ptr,                                                    \
      k_ptr,                                                    \
      v_ptr,                                                    \
      reinterpret_cast<const T*>(q_proj.data_ptr()),            \
      reinterpret_cast<const T*>(k_proj.data_ptr()),            \
      reinterpret_cast<const T*>(v_proj.data_ptr()),            \
      reinterpret_cast<const float*>(q_conv_weight.data_ptr()), \
      reinterpret_cast<const float*>(k_conv_weight.data_ptr()), \
      reinterpret_cast<const float*>(v_conv_weight.data_ptr()), \
      conv_state_ptr,                                           \
      conv_state.stride(0),                                     \
      conv_state_dim_stride,                                    \
      conv_state_time_stride,                                   \
      QUERY_START,                                              \
      TOKEN_INDX,                                               \
      STATE_INDICES,                                            \
      STATE_STRIDE,                                             \
      HAS_INITIAL,                                              \
      ACCEPTED,                                                 \
      BATCH_SIZE,                                               \
      hidden_dim)

#define WIDTH_DISPATCH(  \
    IS_SPEC,             \
    QUERY_START,         \
    TOKEN_INDX,          \
    STATE_INDICES,       \
    STATE_STRIDE,        \
    HAS_INITIAL,         \
    ACCEPTED,            \
    BATCH_SIZE)          \
  switch (width) {       \
    case 2:              \
      LAUNCH_CONV(       \
          2,             \
          IS_SPEC,       \
          QUERY_START,   \
          TOKEN_INDX,    \
          STATE_INDICES, \
          STATE_STRIDE,  \
          HAS_INITIAL,   \
          ACCEPTED,      \
          BATCH_SIZE);   \
      break;             \
    case 3:              \
      LAUNCH_CONV(       \
          3,             \
          IS_SPEC,       \
          QUERY_START,   \
          TOKEN_INDX,    \
          STATE_INDICES, \
          STATE_STRIDE,  \
          HAS_INITIAL,   \
          ACCEPTED,      \
          BATCH_SIZE);   \
      break;             \
    case 4:              \
      LAUNCH_CONV(       \
          4,             \
          IS_SPEC,       \
          QUERY_START,   \
          TOKEN_INDX,    \
          STATE_INDICES, \
          STATE_STRIDE,  \
          HAS_INITIAL,   \
          ACCEPTED,      \
          BATCH_SIZE);   \
      break;             \
    case 5:              \
      LAUNCH_CONV(       \
          5,             \
          IS_SPEC,       \
          QUERY_START,   \
          TOKEN_INDX,    \
          STATE_INDICES, \
          STATE_STRIDE,  \
          HAS_INITIAL,   \
          ACCEPTED,      \
          BATCH_SIZE);   \
      break;             \
  }

#define LAUNCH_RECURRENT(                        \
    BUCKET,                                      \
    IS_SPEC,                                     \
    QUERY_START,                                 \
    TOKEN_INDX,                                  \
    STATE_INDICES,                               \
    STATE_STRIDE,                                \
    HAS_INITIAL,                                 \
    ACCEPTED,                                    \
    BATCH_SIZE)                                  \
  kda::launch_recurrent_kda<T, BUCKET, IS_SPEC>( \
      queue,                                     \
      output_ptr,                                \
      q_ptr,                                     \
      k_ptr,                                     \
      v_ptr,                                     \
      raw_gate_ptr,                              \
      beta_ptr,                                  \
      a_log_ptr,                                 \
      dt_bias_ptr,                               \
      recurrent_state_ptr,                       \
      recurrent_state.stride(0),                 \
      QUERY_START,                               \
      TOKEN_INDX,                                \
      STATE_INDICES,                             \
      STATE_STRIDE,                              \
      HAS_INITIAL,                               \
      ACCEPTED,                                  \
      BATCH_SIZE,                                \
      num_heads,                                 \
      head_dim)

#define BUCKET_DISPATCH(                    \
    IS_SPEC,                                \
    QUERY_START,                            \
    TOKEN_INDX,                             \
    STATE_INDICES,                          \
    STATE_STRIDE,                           \
    HAS_INITIAL,                            \
    ACCEPTED,                               \
    BATCH_SIZE)                             \
  switch (head_dim / kda::sub_group_size) { \
    case 1:                                 \
      LAUNCH_RECURRENT(                     \
          1,                                \
          IS_SPEC,                          \
          QUERY_START,                      \
          TOKEN_INDX,                       \
          STATE_INDICES,                    \
          STATE_STRIDE,                     \
          HAS_INITIAL,                      \
          ACCEPTED,                         \
          BATCH_SIZE);                      \
      break;                                \
    case 2:                                 \
      LAUNCH_RECURRENT(                     \
          2,                                \
          IS_SPEC,                          \
          QUERY_START,                      \
          TOKEN_INDX,                       \
          STATE_INDICES,                    \
          STATE_STRIDE,                     \
          HAS_INITIAL,                      \
          ACCEPTED,                         \
          BATCH_SIZE);                      \
      break;                                \
    case 4:                                 \
      LAUNCH_RECURRENT(                     \
          4,                                \
          IS_SPEC,                          \
          QUERY_START,                      \
          TOKEN_INDX,                       \
          STATE_INDICES,                    \
          STATE_STRIDE,                     \
          HAS_INITIAL,                      \
          ACCEPTED,                         \
          BATCH_SIZE);                      \
      break;                                \
    case 8:                                 \
      LAUNCH_RECURRENT(                     \
          8,                                \
          IS_SPEC,                          \
          QUERY_START,                      \
          TOKEN_INDX,                       \
          STATE_INDICES,                    \
          STATE_STRIDE,                     \
          HAS_INITIAL,                      \
          ACCEPTED,                         \
          BATCH_SIZE);                      \
      break;                                \
  }

  const int non_spec_batch_size = num_prefills + num_decodes;
  if (non_spec_batch_size > 0) {
    const int* query_start =
        reinterpret_cast<const int*>(non_spec_query_start_loc->data_ptr());
    const int* token_indx =
        non_spec_token_indx.has_value()
            ? reinterpret_cast<const int*>(non_spec_token_indx->data_ptr())
            : nullptr;
    const int* state_indices =
        reinterpret_cast<const int*>(non_spec_state_indices->data_ptr());
    const bool* initial_state =
        has_initial_state.has_value()
            ? reinterpret_cast<const bool*>(has_initial_state->data_ptr())
            : nullptr;

    WIDTH_DISPATCH(
        false,
        query_start,
        token_indx,
        state_indices,
        0,
        initial_state,
        nullptr,
        non_spec_batch_size);
    BUCKET_DISPATCH(
        false,
        query_start,
        token_indx,
        state_indices,
        0,
        initial_state,
        nullptr,
        non_spec_batch_size);
  }

  if (num_spec_decodes > 0) {
    const int* query_start =
        reinterpret_cast<const int*>(spec_query_start_loc->data_ptr());
    const int* token_indx =
        reinterpret_cast<const int*>(spec_token_indx->data_ptr());
    const int* state_indices =
        reinterpret_cast<const int*>(spec_state_indices->data_ptr());
    const int* accepted =
        reinterpret_cast<const int*>(num_accepted_tokens->data_ptr());
    const int64_t state_stride = spec_state_indices->stride(0);

    WIDTH_DISPATCH(
        true,
        query_start,
        token_indx,
        state_indices,
        state_stride,
        nullptr,
        accepted,
        num_spec_decodes);
    BUCKET_DISPATCH(
        true,
        query_start,
        token_indx,
        state_indices,
        state_stride,
        nullptr,
        accepted,
        num_spec_decodes);
  }

#undef BUCKET_DISPATCH
#undef LAUNCH_RECURRENT
#undef WIDTH_DISPATCH
#undef LAUNCH_CONV
}

}  // namespace

void kda_attention(
    torch::Tensor& core_attn_out,
    const torch::Tensor& q_proj,
    const torch::Tensor& k_proj,
    const torch::Tensor& v_proj,
    const torch::Tensor& raw_gate,
    const torch::Tensor& beta,
    torch::Tensor& conv_state,
    torch::Tensor& recurrent_state,
    const torch::Tensor& q_conv_weight,
    const torch::Tensor& k_conv_weight,
    const torch::Tensor& v_conv_weight,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    const int64_t num_prefills,
    const int64_t num_decodes,
    const int64_t num_spec_decodes,
    const std::optional<torch::Tensor>& has_initial_state,
    const std::optional<torch::Tensor>& non_spec_query_start_loc,
    const std::optional<torch::Tensor>& non_spec_token_indx,
    const std::optional<torch::Tensor>& non_spec_state_indices,
    const std::optional<torch::Tensor>& spec_query_start_loc,
    const std::optional<torch::Tensor>& spec_token_indx,
    const std::optional<torch::Tensor>& spec_state_indices,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    const int64_t num_actual_tokens) {
  TORCH_CHECK(num_actual_tokens >= 0, "num_actual_tokens must be non-negative");
  TORCH_CHECK(
      num_prefills >= 0 && num_decodes >= 0 && num_spec_decodes >= 0,
      "sequence counts must be non-negative");
  TORCH_CHECK(q_proj.dim() == 2, "q_proj must be 2D");
  TORCH_CHECK(
      k_proj.sizes() == q_proj.sizes(), "k_proj shape must match q_proj");
  TORCH_CHECK(
      v_proj.sizes() == q_proj.sizes(), "v_proj shape must match q_proj");
  TORCH_CHECK(raw_gate.dim() == 4, "raw_gate must be 4D");
  TORCH_CHECK(raw_gate.size(0) == 1, "raw_gate batch dimension must be 1");
  TORCH_CHECK(beta.dim() == 3, "beta must be 3D");
  TORCH_CHECK(beta.size(0) == 1, "beta batch dimension must be 1");

  const int64_t num_heads = raw_gate.size(2);
  const int64_t head_dim = raw_gate.size(3);
  const int64_t hidden_dim = num_heads * head_dim;
  TORCH_CHECK(q_proj.size(1) == hidden_dim, "q_proj hidden dimension mismatch");
  TORCH_CHECK(beta.size(2) == num_heads, "beta head dimension mismatch");
  TORCH_CHECK(
      q_proj.size(0) >= num_actual_tokens &&
          raw_gate.size(1) >= num_actual_tokens &&
          beta.size(1) >= num_actual_tokens,
      "input leading dimensions must be >= num_actual_tokens");
  TORCH_CHECK(
      core_attn_out.dim() == 4 && core_attn_out.size(0) == 1 &&
          core_attn_out.size(1) >= num_actual_tokens &&
          core_attn_out.size(2) == num_heads &&
          core_attn_out.size(3) == head_dim,
      "core_attn_out must have shape [1, >=num_actual_tokens, heads, dim]");

  TORCH_CHECK(
      q_proj.is_contiguous() && k_proj.is_contiguous() &&
          v_proj.is_contiguous() && raw_gate.is_contiguous() &&
          beta.is_contiguous() && core_attn_out.is_contiguous(),
      "KDA activation tensors must be contiguous");
  TORCH_CHECK(
      q_proj.scalar_type() == k_proj.scalar_type() &&
          q_proj.scalar_type() == v_proj.scalar_type() &&
          q_proj.scalar_type() == raw_gate.scalar_type() &&
          q_proj.scalar_type() == core_attn_out.scalar_type(),
      "KDA activation tensor dtypes must match");
  TORCH_CHECK(
      q_proj.scalar_type() == at::kHalf ||
          q_proj.scalar_type() == at::kBFloat16 ||
          q_proj.scalar_type() == at::kFloat,
      "KDA activations must be float16, bfloat16, or float32");
  TORCH_CHECK(beta.scalar_type() == at::kFloat, "beta must be float32");

  TORCH_CHECK(
      q_conv_weight.dim() == 2 && q_conv_weight.size(0) == hidden_dim,
      "q_conv_weight must have shape [heads * dim, width]");
  TORCH_CHECK(
      k_conv_weight.sizes() == q_conv_weight.sizes() &&
          v_conv_weight.sizes() == q_conv_weight.sizes(),
      "KDA convolution weight shapes must match");
  TORCH_CHECK(
      q_conv_weight.scalar_type() == at::kFloat &&
          k_conv_weight.scalar_type() == at::kFloat &&
          v_conv_weight.scalar_type() == at::kFloat,
      "KDA convolution weights must be float32");
  TORCH_CHECK(
      q_conv_weight.is_contiguous() && k_conv_weight.is_contiguous() &&
          v_conv_weight.is_contiguous(),
      "KDA convolution weights must be contiguous");
  const int64_t width = q_conv_weight.size(1);
  TORCH_CHECK(width >= 2 && width <= 5, "KDA convolution width must be 2-5");

  TORCH_CHECK(
      a_log.scalar_type() == at::kFloat && a_log.numel() == num_heads &&
          a_log.is_contiguous(),
      "A_log must be contiguous float32 with one value per head");
  TORCH_CHECK(
      dt_bias.scalar_type() == at::kFloat && dt_bias.numel() == hidden_dim &&
          dt_bias.is_contiguous(),
      "dt_bias must be contiguous float32 with shape [heads * dim]");

  TORCH_CHECK(conv_state.dim() == 3, "conv_state must be 3D");
  TORCH_CHECK(conv_state.is_contiguous(), "conv_state must be contiguous");
  TORCH_CHECK(
      conv_state.scalar_type() == at::kHalf ||
          conv_state.scalar_type() == at::kBFloat16 ||
          conv_state.scalar_type() == at::kFloat,
      "conv_state must be float16, bfloat16, or float32");
  TORCH_CHECK(
      recurrent_state.dim() == 4 && recurrent_state.size(1) == num_heads &&
          recurrent_state.size(2) == head_dim &&
          recurrent_state.size(3) == head_dim,
      "recurrent_state must have shape [slots, heads, dim, dim]");
  TORCH_CHECK(
      recurrent_state.scalar_type() == at::kFloat &&
          recurrent_state.is_contiguous(),
      "recurrent_state must be contiguous float32");
  TORCH_CHECK(
      recurrent_state.size(0) == conv_state.size(0),
      "KDA cache slot counts must match");

  int64_t conv_state_dim_stride;
  int64_t conv_state_time_stride;
  if (conv_state.size(1) == 3 * hidden_dim && conv_state.size(2) == width - 1) {
    conv_state_dim_stride = conv_state.stride(1);
    conv_state_time_stride = conv_state.stride(2);
  } else {
    TORCH_CHECK(
        conv_state.size(1) == width - 1 && conv_state.size(2) == 3 * hidden_dim,
        "conv_state must use DS or SD layout");
    conv_state_dim_stride = conv_state.stride(2);
    conv_state_time_stride = conv_state.stride(1);
  }

  TORCH_CHECK(
      head_dim % kda::sub_group_size == 0 && head_dim <= 256,
      "KDA head_dim must be a multiple of 32 and <= 256");

  const int64_t non_spec_batch_size = num_prefills + num_decodes;
  if (non_spec_batch_size > 0) {
    check_int32_tensor(non_spec_query_start_loc, "non_spec_query_start_loc", 1);
    check_int32_tensor(non_spec_state_indices, "non_spec_state_indices", 1);
    TORCH_CHECK(
        non_spec_query_start_loc->size(0) >= non_spec_batch_size + 1,
        "non_spec_query_start_loc size mismatch");
    TORCH_CHECK(
        non_spec_state_indices->size(0) >= non_spec_batch_size,
        "non_spec_state_indices size mismatch");
    if (non_spec_token_indx.has_value()) {
      check_int32_tensor(non_spec_token_indx, "non_spec_token_indx", 1);
    }
    if (has_initial_state.has_value()) {
      TORCH_CHECK(
          has_initial_state->is_contiguous() &&
              has_initial_state->scalar_type() == at::kBool &&
              has_initial_state->dim() == 1 &&
              has_initial_state->size(0) >= non_spec_batch_size,
          "has_initial_state must be contiguous bool with one value per "
          "non-spec sequence");
    }
  }

  int64_t spec_tokens = 0;
  if (num_spec_decodes > 0) {
    check_int32_tensor(spec_query_start_loc, "spec_query_start_loc", 1);
    check_int32_tensor(spec_token_indx, "spec_token_indx", 1);
    check_int32_tensor(spec_state_indices, "spec_state_indices", 2);
    check_int32_tensor(num_accepted_tokens, "num_accepted_tokens", 1);
    TORCH_CHECK(
        spec_query_start_loc->size(0) >= num_spec_decodes + 1,
        "spec_query_start_loc size mismatch");
    TORCH_CHECK(
        spec_state_indices->size(0) >= num_spec_decodes,
        "spec_state_indices batch size mismatch");
    TORCH_CHECK(
        num_accepted_tokens->size(0) >= num_spec_decodes,
        "num_accepted_tokens size mismatch");
    spec_tokens = spec_token_indx->size(0);
    TORCH_CHECK(
        spec_tokens == num_spec_decodes * spec_state_indices->size(1),
        "spec token and state-index counts must match");
  }

  const int64_t non_spec_tokens =
      non_spec_batch_size == 0
          ? 0
          : (non_spec_token_indx.has_value() ? non_spec_token_indx->size(0)
                                             : num_actual_tokens - spec_tokens);
  TORCH_CHECK(
      non_spec_tokens + spec_tokens == num_actual_tokens,
      "metadata token counts must equal num_actual_tokens");
  if (num_actual_tokens == 0) {
    return;
  }

  auto& queue = vllm::xpu::vllmGetQueue();
  auto launch = [&](auto activation_tag, auto cache_tag) {
    using T = decltype(activation_tag);
    using CacheT = decltype(cache_tag);
    launch_kda<T, CacheT>(
        queue,
        core_attn_out,
        q_proj,
        k_proj,
        v_proj,
        raw_gate,
        beta,
        conv_state,
        recurrent_state,
        q_conv_weight,
        k_conv_weight,
        v_conv_weight,
        a_log,
        dt_bias,
        has_initial_state,
        non_spec_query_start_loc,
        non_spec_token_indx,
        non_spec_state_indices,
        spec_query_start_loc,
        spec_token_indx,
        spec_state_indices,
        num_accepted_tokens,
        num_prefills,
        num_decodes,
        num_spec_decodes,
        num_actual_tokens,
        num_heads,
        head_dim,
        width,
        conv_state_dim_stride,
        conv_state_time_stride);
  };
  auto dispatch_cache = [&](auto activation_tag) {
    if (conv_state.scalar_type() == at::kBFloat16) {
      launch(activation_tag, sycl::ext::oneapi::bfloat16{});
    } else if (conv_state.scalar_type() == at::kHalf) {
      launch(activation_tag, sycl::half{});
    } else {
      launch(activation_tag, float{});
    }
  };

  if (q_proj.scalar_type() == at::kBFloat16) {
    dispatch_cache(sycl::ext::oneapi::bfloat16{});
  } else if (q_proj.scalar_type() == at::kHalf) {
    dispatch_cache(sycl::half{});
  } else {
    dispatch_cache(float{});
  }
}
