#include <sycl/sycl.hpp>
#include <torch/all.h>

#include <limits>

#include "kda_attention.hpp"
#include "utils.h"

namespace {

void check_cache_layout(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.size(0) > 0, name, " must contain at least one slot");
  TORCH_CHECK(
      tensor[0].is_contiguous(), name, " of each slot must be contiguous");
  TORCH_CHECK(
      tensor.size(0) == 1 || tensor.stride(0) >= tensor[0].numel(),
      name,
      " slots must not overlap");
}

void check_device(
    const torch::Tensor& tensor,
    const torch::Device& device,
    const char* name) {
  TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
}

void check_int32_tensor(
    const std::optional<torch::Tensor>& tensor,
    const char* name,
    int64_t dim,
    const torch::Device& device) {
  TORCH_CHECK(tensor.has_value(), name, " must be provided");
  check_device(*tensor, device, name);
  TORCH_CHECK(tensor->is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor->scalar_type() == at::kInt, name, " must be int32");
  TORCH_CHECK(tensor->dim() == dim, name, " must be ", dim, "D");
}

void validate_metadata(
    int64_t num_prefills,
    int64_t num_decodes,
    int64_t num_spec_decodes,
    const std::optional<torch::Tensor>& has_initial_state,
    const std::optional<torch::Tensor>& non_spec_query_start_loc,
    const std::optional<torch::Tensor>& non_spec_token_indx,
    const std::optional<torch::Tensor>& non_spec_state_indices,
    const std::optional<torch::Tensor>& spec_query_start_loc,
    const std::optional<torch::Tensor>& spec_token_indx,
    const std::optional<torch::Tensor>& spec_state_indices,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    int64_t num_actual_tokens,
    const torch::Device& device) {
  TORCH_CHECK(num_actual_tokens >= 0, "num_actual_tokens must be non-negative");
  TORCH_CHECK(
      num_prefills >= 0 && num_decodes >= 0 && num_spec_decodes >= 0,
      "sequence counts must be non-negative");
  constexpr int64_t max_kernel_int = std::numeric_limits<int>::max();
  TORCH_CHECK(
      num_actual_tokens <= max_kernel_int,
      "num_actual_tokens exceeds the KDA kernel limit");
  TORCH_CHECK(
      num_prefills <= max_kernel_int && num_decodes <= max_kernel_int &&
          num_spec_decodes <= max_kernel_int &&
          num_prefills + num_decodes <= max_kernel_int,
      "sequence counts exceed the KDA kernel limit");

  const int64_t non_spec_batch_size = num_prefills + num_decodes;
  if (non_spec_batch_size > 0) {
    check_int32_tensor(
        non_spec_query_start_loc, "non_spec_query_start_loc", 1, device);
    check_int32_tensor(
        non_spec_state_indices, "non_spec_state_indices", 1, device);
    TORCH_CHECK(
        non_spec_query_start_loc->size(0) >= non_spec_batch_size + 1,
        "non_spec_query_start_loc must contain one offset per sequence plus "
        "the final offset");
    TORCH_CHECK(
        non_spec_state_indices->size(0) >= non_spec_batch_size,
        "non_spec_state_indices must contain one slot per sequence");
    if (non_spec_token_indx.has_value()) {
      check_int32_tensor(non_spec_token_indx, "non_spec_token_indx", 1, device);
    }
    if (has_initial_state.has_value()) {
      check_device(*has_initial_state, device, "has_initial_state");
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
    check_int32_tensor(spec_query_start_loc, "spec_query_start_loc", 1, device);
    check_int32_tensor(spec_token_indx, "spec_token_indx", 1, device);
    check_int32_tensor(spec_state_indices, "spec_state_indices", 2, device);
    check_int32_tensor(num_accepted_tokens, "num_accepted_tokens", 1, device);
    TORCH_CHECK(
        spec_query_start_loc->size(0) >= num_spec_decodes + 1,
        "spec_query_start_loc must contain one offset per sequence plus the "
        "final offset");
    TORCH_CHECK(
        spec_state_indices->size(0) >= num_spec_decodes,
        "spec_state_indices must contain one row per sequence");
    TORCH_CHECK(
        spec_state_indices->size(1) > 0,
        "spec_state_indices must contain at least one slot per sequence");
    TORCH_CHECK(
        num_accepted_tokens->size(0) >= num_spec_decodes,
        "num_accepted_tokens must contain one value per spec sequence");
    spec_tokens = spec_token_indx->numel();
    TORCH_CHECK(
        spec_tokens == num_spec_decodes * spec_state_indices->size(1),
        "spec token and state-index counts must match");
  }

  const int64_t non_spec_tokens =
      non_spec_batch_size == 0
          ? 0
          : (non_spec_token_indx.has_value() ? non_spec_token_indx->numel()
                                             : num_actual_tokens - spec_tokens);
  TORCH_CHECK(
      non_spec_tokens >= 0 &&
          non_spec_tokens + spec_tokens == num_actual_tokens,
      "metadata token counts must equal num_actual_tokens");
}

struct ConvShape {
  int64_t hidden_dim;
  int64_t width;
  int64_t dim_stride;
  int64_t time_stride;
};

ConvShape validate_conv_inputs(
    const torch::Tensor& q_proj,
    const torch::Tensor& k_proj,
    const torch::Tensor& v_proj,
    torch::Tensor& conv_state,
    const torch::Tensor& q_conv_weight,
    const torch::Tensor& k_conv_weight,
    const torch::Tensor& v_conv_weight,
    int64_t num_actual_tokens) {
  const auto device = q_proj.device();
  TORCH_CHECK(q_proj.is_xpu(), "q_proj must be on XPU");
  TORCH_CHECK(q_proj.dim() == 2, "q_proj must be 2D");
  TORCH_CHECK(
      k_proj.sizes() == q_proj.sizes(), "k_proj shape must match q_proj");
  TORCH_CHECK(
      v_proj.sizes() == q_proj.sizes(), "v_proj shape must match q_proj");
  TORCH_CHECK(
      q_proj.size(0) >= num_actual_tokens,
      "projection leading dimensions must be >= num_actual_tokens");
  TORCH_CHECK(
      q_proj.is_contiguous() && k_proj.is_contiguous() &&
          v_proj.is_contiguous(),
      "KDA projections must be contiguous");
  TORCH_CHECK(
      q_proj.scalar_type() == k_proj.scalar_type() &&
          q_proj.scalar_type() == v_proj.scalar_type(),
      "KDA projection dtypes must match");
  TORCH_CHECK(
      q_proj.scalar_type() == at::kHalf ||
          q_proj.scalar_type() == at::kBFloat16 ||
          q_proj.scalar_type() == at::kFloat,
      "KDA projections must be float16, bfloat16, or float32");
  check_device(k_proj, device, "k_proj");
  check_device(v_proj, device, "v_proj");

  const int64_t hidden_dim = q_proj.size(1);
  TORCH_CHECK(hidden_dim > 0, "projection hidden dimension must be positive");
  TORCH_CHECK(
      hidden_dim <= std::numeric_limits<int>::max(),
      "projection hidden dimension exceeds the KDA kernel limit");
  TORCH_CHECK(
      q_conv_weight.dim() == 2 && q_conv_weight.size(0) == hidden_dim,
      "q_conv_weight must have shape [hidden_dim, width]");
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
  check_device(q_conv_weight, device, "q_conv_weight");
  check_device(k_conv_weight, device, "k_conv_weight");
  check_device(v_conv_weight, device, "v_conv_weight");

  const int64_t width = q_conv_weight.size(1);
  TORCH_CHECK(width >= 2 && width <= 5, "KDA convolution width must be 2-5");
  TORCH_CHECK(conv_state.dim() == 3, "conv_state must be 3D");
  TORCH_CHECK(
      conv_state.scalar_type() == at::kHalf ||
          conv_state.scalar_type() == at::kBFloat16 ||
          conv_state.scalar_type() == at::kFloat,
      "conv_state must be float16, bfloat16, or float32");
  check_device(conv_state, device, "conv_state");
  check_cache_layout(conv_state, "conv_state");

  int64_t dim_stride;
  int64_t time_stride;
  if (conv_state.size(1) == 3 * hidden_dim && conv_state.size(2) == width - 1) {
    dim_stride = conv_state.stride(1);
    time_stride = conv_state.stride(2);
  } else {
    TORCH_CHECK(
        conv_state.size(1) == width - 1 && conv_state.size(2) == 3 * hidden_dim,
        "conv_state must have DS shape [slots, 3 * hidden_dim, width - 1] "
        "or SD shape [slots, width - 1, 3 * hidden_dim]");
    dim_stride = conv_state.stride(2);
    time_stride = conv_state.stride(1);
  }
  return {hidden_dim, width, dim_stride, time_stride};
}

struct RecurrentShape {
  int64_t num_heads;
  int64_t head_dim;
};

RecurrentShape validate_recurrent_inputs(
    torch::Tensor& core_attn_out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& raw_gate,
    const torch::Tensor& beta,
    torch::Tensor& recurrent_state,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    int64_t num_actual_tokens) {
  const auto device = q.device();
  TORCH_CHECK(q.is_xpu(), "q must be on XPU");
  TORCH_CHECK(raw_gate.dim() == 4, "raw_gate must be 4D");
  TORCH_CHECK(raw_gate.size(0) == 1, "raw_gate batch dimension must be 1");
  const int64_t num_heads = raw_gate.size(2);
  const int64_t head_dim = raw_gate.size(3);
  const int64_t hidden_dim = num_heads * head_dim;
  TORCH_CHECK(num_heads > 0, "KDA num_heads must be positive");
  TORCH_CHECK(
      head_dim == 32 || head_dim == 64 || head_dim == 128 || head_dim == 256,
      "KDA head_dim must be one of 32, 64, 128, or 256");

  TORCH_CHECK(
      q.dim() == 2 && q.size(0) == num_actual_tokens && q.size(1) == hidden_dim,
      "q must have shape [num_actual_tokens, heads * dim]");
  TORCH_CHECK(k.sizes() == q.sizes(), "k shape must match q");
  TORCH_CHECK(v.sizes() == q.sizes(), "v shape must match q");
  TORCH_CHECK(
      q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
          raw_gate.is_contiguous() && beta.is_contiguous() &&
          core_attn_out.is_contiguous(),
      "KDA recurrent activation tensors must be contiguous");
  TORCH_CHECK(
      q.scalar_type() == k.scalar_type() &&
          q.scalar_type() == v.scalar_type() &&
          q.scalar_type() == raw_gate.scalar_type() &&
          q.scalar_type() == core_attn_out.scalar_type(),
      "KDA recurrent activation tensor dtypes must match");
  TORCH_CHECK(
      q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16 ||
          q.scalar_type() == at::kFloat,
      "KDA recurrent activations must be float16, bfloat16, or float32");
  TORCH_CHECK(
      raw_gate.size(1) >= num_actual_tokens,
      "raw_gate token dimension must be >= num_actual_tokens");
  TORCH_CHECK(
      beta.dim() == 3 && beta.size(0) == 1 &&
          beta.size(1) >= num_actual_tokens && beta.size(2) == num_heads,
      "beta must have shape [1, >=num_actual_tokens, heads]");
  TORCH_CHECK(beta.scalar_type() == at::kFloat, "beta must be float32");
  TORCH_CHECK(
      core_attn_out.dim() == 4 && core_attn_out.size(0) == 1 &&
          core_attn_out.size(1) >= num_actual_tokens &&
          core_attn_out.size(2) == num_heads &&
          core_attn_out.size(3) == head_dim,
      "core_attn_out must have shape [1, >=num_actual_tokens, heads, dim]");
  check_device(k, device, "k");
  check_device(v, device, "v");
  check_device(raw_gate, device, "raw_gate");
  check_device(beta, device, "beta");
  check_device(core_attn_out, device, "core_attn_out");

  TORCH_CHECK(
      recurrent_state.dim() == 4 && recurrent_state.size(1) == num_heads &&
          recurrent_state.size(2) == head_dim &&
          recurrent_state.size(3) == head_dim,
      "recurrent_state must have shape [slots, heads, dim, dim]");
  TORCH_CHECK(
      recurrent_state.scalar_type() == at::kFloat,
      "recurrent_state must be float32");
  check_device(recurrent_state, device, "recurrent_state");
  check_cache_layout(recurrent_state, "recurrent_state");
  TORCH_CHECK(
      a_log.scalar_type() == at::kFloat && a_log.numel() == num_heads &&
          a_log.is_contiguous() && a_log.dim() == 4 && a_log.size(0) == 1 &&
          a_log.size(1) == 1 && a_log.size(2) == num_heads &&
          a_log.size(3) == 1,
      "A_log must have contiguous float32 shape [1, 1, heads, 1]");
  TORCH_CHECK(
      dt_bias.scalar_type() == at::kFloat && dt_bias.numel() == hidden_dim &&
          dt_bias.is_contiguous() && dt_bias.dim() == 1,
      "dt_bias must have contiguous float32 shape [heads * dim]");
  check_device(a_log, device, "A_log");
  check_device(dt_bias, device, "dt_bias");
  return {num_heads, head_dim};
}

template <typename T, typename CacheT>
void launch_kda_conv(
    sycl::queue& queue,
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    const torch::Tensor& q_proj,
    const torch::Tensor& k_proj,
    const torch::Tensor& v_proj,
    torch::Tensor& conv_state,
    const torch::Tensor& q_conv_weight,
    const torch::Tensor& k_conv_weight,
    const torch::Tensor& v_conv_weight,
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
    int hidden_dim,
    int width,
    int64_t conv_state_dim_stride,
    int64_t conv_state_time_stride) {
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
      reinterpret_cast<T*>(q.data_ptr()),                       \
      reinterpret_cast<T*>(k.data_ptr()),                       \
      reinterpret_cast<T*>(v.data_ptr()),                       \
      reinterpret_cast<const T*>(q_proj.data_ptr()),            \
      reinterpret_cast<const T*>(k_proj.data_ptr()),            \
      reinterpret_cast<const T*>(v_proj.data_ptr()),            \
      reinterpret_cast<const float*>(q_conv_weight.data_ptr()), \
      reinterpret_cast<const float*>(k_conv_weight.data_ptr()), \
      reinterpret_cast<const float*>(v_conv_weight.data_ptr()), \
      reinterpret_cast<CacheT*>(conv_state.data_ptr()),         \
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

#define WIDTH_DISPATCH(                                        \
    IS_SPEC,                                                   \
    QUERY_START,                                               \
    TOKEN_INDX,                                                \
    STATE_INDICES,                                             \
    STATE_STRIDE,                                              \
    HAS_INITIAL,                                               \
    ACCEPTED,                                                  \
    BATCH_SIZE)                                                \
  switch (width) {                                             \
    case 2:                                                    \
      LAUNCH_CONV(                                             \
          2,                                                   \
          IS_SPEC,                                             \
          QUERY_START,                                         \
          TOKEN_INDX,                                          \
          STATE_INDICES,                                       \
          STATE_STRIDE,                                        \
          HAS_INITIAL,                                         \
          ACCEPTED,                                            \
          BATCH_SIZE);                                         \
      break;                                                   \
    case 3:                                                    \
      LAUNCH_CONV(                                             \
          3,                                                   \
          IS_SPEC,                                             \
          QUERY_START,                                         \
          TOKEN_INDX,                                          \
          STATE_INDICES,                                       \
          STATE_STRIDE,                                        \
          HAS_INITIAL,                                         \
          ACCEPTED,                                            \
          BATCH_SIZE);                                         \
      break;                                                   \
    case 4:                                                    \
      LAUNCH_CONV(                                             \
          4,                                                   \
          IS_SPEC,                                             \
          QUERY_START,                                         \
          TOKEN_INDX,                                          \
          STATE_INDICES,                                       \
          STATE_STRIDE,                                        \
          HAS_INITIAL,                                         \
          ACCEPTED,                                            \
          BATCH_SIZE);                                         \
      break;                                                   \
    case 5:                                                    \
      LAUNCH_CONV(                                             \
          5,                                                   \
          IS_SPEC,                                             \
          QUERY_START,                                         \
          TOKEN_INDX,                                          \
          STATE_INDICES,                                       \
          STATE_STRIDE,                                        \
          HAS_INITIAL,                                         \
          ACCEPTED,                                            \
          BATCH_SIZE);                                         \
      break;                                                   \
    default:                                                   \
      TORCH_CHECK(false, "unsupported KDA convolution width"); \
  }

  const int non_spec_batch_size = num_prefills + num_decodes;
  if (non_spec_batch_size > 0) {
    WIDTH_DISPATCH(
        false,
        reinterpret_cast<const int*>(non_spec_query_start_loc->data_ptr()),
        non_spec_token_indx.has_value()
            ? reinterpret_cast<const int*>(non_spec_token_indx->data_ptr())
            : nullptr,
        reinterpret_cast<const int*>(non_spec_state_indices->data_ptr()),
        0,
        has_initial_state.has_value()
            ? reinterpret_cast<const bool*>(has_initial_state->data_ptr())
            : nullptr,
        nullptr,
        non_spec_batch_size);
  }
  if (num_spec_decodes > 0) {
    WIDTH_DISPATCH(
        true,
        reinterpret_cast<const int*>(spec_query_start_loc->data_ptr()),
        reinterpret_cast<const int*>(spec_token_indx->data_ptr()),
        reinterpret_cast<const int*>(spec_state_indices->data_ptr()),
        spec_state_indices->stride(0),
        nullptr,
        reinterpret_cast<const int*>(num_accepted_tokens->data_ptr()),
        num_spec_decodes);
  }
#undef WIDTH_DISPATCH
#undef LAUNCH_CONV
}

template <typename T>
void launch_kda_recurrent(
    sycl::queue& queue,
    torch::Tensor& core_attn_out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& raw_gate,
    const torch::Tensor& beta,
    torch::Tensor& recurrent_state,
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
    int num_heads,
    int head_dim) {
#define LAUNCH_RECURRENT(                                   \
    BUCKET,                                                 \
    IS_SPEC,                                                \
    QUERY_START,                                            \
    TOKEN_INDX,                                             \
    STATE_INDICES,                                          \
    STATE_STRIDE,                                           \
    HAS_INITIAL,                                            \
    ACCEPTED,                                               \
    BATCH_SIZE)                                             \
  kda::launch_recurrent_kda<T, BUCKET, IS_SPEC>(            \
      queue,                                                \
      reinterpret_cast<T*>(core_attn_out.data_ptr()),       \
      reinterpret_cast<const T*>(q.data_ptr()),             \
      reinterpret_cast<const T*>(k.data_ptr()),             \
      reinterpret_cast<const T*>(v.data_ptr()),             \
      reinterpret_cast<const T*>(raw_gate.data_ptr()),      \
      reinterpret_cast<const float*>(beta.data_ptr()),      \
      reinterpret_cast<const float*>(a_log.data_ptr()),     \
      reinterpret_cast<const float*>(dt_bias.data_ptr()),   \
      reinterpret_cast<float*>(recurrent_state.data_ptr()), \
      recurrent_state.stride(0),                            \
      QUERY_START,                                          \
      TOKEN_INDX,                                           \
      STATE_INDICES,                                        \
      STATE_STRIDE,                                         \
      HAS_INITIAL,                                          \
      ACCEPTED,                                             \
      BATCH_SIZE,                                           \
      num_heads,                                            \
      head_dim)

#define BUCKET_DISPATCH(                                    \
    IS_SPEC,                                                \
    QUERY_START,                                            \
    TOKEN_INDX,                                             \
    STATE_INDICES,                                          \
    STATE_STRIDE,                                           \
    HAS_INITIAL,                                            \
    ACCEPTED,                                               \
    BATCH_SIZE)                                             \
  switch (head_dim / kda::sub_group_size) {                 \
    case 1:                                                 \
      LAUNCH_RECURRENT(                                     \
          1,                                                \
          IS_SPEC,                                          \
          QUERY_START,                                      \
          TOKEN_INDX,                                       \
          STATE_INDICES,                                    \
          STATE_STRIDE,                                     \
          HAS_INITIAL,                                      \
          ACCEPTED,                                         \
          BATCH_SIZE);                                      \
      break;                                                \
    case 2:                                                 \
      LAUNCH_RECURRENT(                                     \
          2,                                                \
          IS_SPEC,                                          \
          QUERY_START,                                      \
          TOKEN_INDX,                                       \
          STATE_INDICES,                                    \
          STATE_STRIDE,                                     \
          HAS_INITIAL,                                      \
          ACCEPTED,                                         \
          BATCH_SIZE);                                      \
      break;                                                \
    case 4:                                                 \
      LAUNCH_RECURRENT(                                     \
          4,                                                \
          IS_SPEC,                                          \
          QUERY_START,                                      \
          TOKEN_INDX,                                       \
          STATE_INDICES,                                    \
          STATE_STRIDE,                                     \
          HAS_INITIAL,                                      \
          ACCEPTED,                                         \
          BATCH_SIZE);                                      \
      break;                                                \
    case 8:                                                 \
      LAUNCH_RECURRENT(                                     \
          8,                                                \
          IS_SPEC,                                          \
          QUERY_START,                                      \
          TOKEN_INDX,                                       \
          STATE_INDICES,                                    \
          STATE_STRIDE,                                     \
          HAS_INITIAL,                                      \
          ACCEPTED,                                         \
          BATCH_SIZE);                                      \
      break;                                                \
    default:                                                \
      TORCH_CHECK(false, "unsupported KDA head dimension"); \
  }

  const int non_spec_batch_size = num_prefills + num_decodes;
  if (non_spec_batch_size > 0) {
    BUCKET_DISPATCH(
        false,
        reinterpret_cast<const int*>(non_spec_query_start_loc->data_ptr()),
        non_spec_token_indx.has_value()
            ? reinterpret_cast<const int*>(non_spec_token_indx->data_ptr())
            : nullptr,
        reinterpret_cast<const int*>(non_spec_state_indices->data_ptr()),
        0,
        has_initial_state.has_value()
            ? reinterpret_cast<const bool*>(has_initial_state->data_ptr())
            : nullptr,
        nullptr,
        non_spec_batch_size);
  }
  if (num_spec_decodes > 0) {
    BUCKET_DISPATCH(
        true,
        reinterpret_cast<const int*>(spec_query_start_loc->data_ptr()),
        reinterpret_cast<const int*>(spec_token_indx->data_ptr()),
        reinterpret_cast<const int*>(spec_state_indices->data_ptr()),
        spec_state_indices->stride(0),
        nullptr,
        reinterpret_cast<const int*>(num_accepted_tokens->data_ptr()),
        num_spec_decodes);
  }
#undef BUCKET_DISPATCH
#undef LAUNCH_RECURRENT
}

template <typename Fn>
void dispatch_activation(at::ScalarType dtype, Fn&& fn) {
  if (dtype == at::kBFloat16) {
    fn(sycl::ext::oneapi::bfloat16{});
  } else if (dtype == at::kHalf) {
    fn(sycl::half{});
  } else if (dtype == at::kFloat) {
    fn(float{});
  } else {
    TORCH_CHECK(false, "unsupported KDA activation dtype");
  }
}

}  // namespace

std::vector<torch::Tensor> kda_causal_conv1d(
    const torch::Tensor& q_proj,
    const torch::Tensor& k_proj,
    const torch::Tensor& v_proj,
    torch::Tensor& conv_state,
    const torch::Tensor& q_conv_weight,
    const torch::Tensor& k_conv_weight,
    const torch::Tensor& v_conv_weight,
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
  validate_metadata(
      num_prefills,
      num_decodes,
      num_spec_decodes,
      has_initial_state,
      non_spec_query_start_loc,
      non_spec_token_indx,
      non_spec_state_indices,
      spec_query_start_loc,
      spec_token_indx,
      spec_state_indices,
      num_accepted_tokens,
      num_actual_tokens,
      q_proj.device());
  const auto shape = validate_conv_inputs(
      q_proj,
      k_proj,
      v_proj,
      conv_state,
      q_conv_weight,
      k_conv_weight,
      v_conv_weight,
      num_actual_tokens);

  auto options =
      torch::dtype(q_proj.dtype()).device(q_proj.device()).requires_grad(false);
  auto q = torch::empty({num_actual_tokens, shape.hidden_dim}, options);
  auto k = torch::empty({num_actual_tokens, shape.hidden_dim}, options);
  auto v = torch::empty({num_actual_tokens, shape.hidden_dim}, options);
  if (num_actual_tokens == 0) {
    return {q, k, v};
  }

  auto& queue = vllm::xpu::vllmGetQueue();
  dispatch_activation(q_proj.scalar_type(), [&](auto tag) {
    using T = decltype(tag);
    dispatch_activation(conv_state.scalar_type(), [&](auto cache_tag) {
      using CacheT = decltype(cache_tag);
      launch_kda_conv<T, CacheT>(
          queue,
          q,
          k,
          v,
          q_proj,
          k_proj,
          v_proj,
          conv_state,
          q_conv_weight,
          k_conv_weight,
          v_conv_weight,
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
          shape.hidden_dim,
          shape.width,
          shape.dim_stride,
          shape.time_stride);
    });
  });
  return {q, k, v};
}

void kda_gated_delta_rule(
    torch::Tensor& core_attn_out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& raw_gate,
    const torch::Tensor& beta,
    torch::Tensor& recurrent_state,
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
  validate_metadata(
      num_prefills,
      num_decodes,
      num_spec_decodes,
      has_initial_state,
      non_spec_query_start_loc,
      non_spec_token_indx,
      non_spec_state_indices,
      spec_query_start_loc,
      spec_token_indx,
      spec_state_indices,
      num_accepted_tokens,
      num_actual_tokens,
      q.device());
  const auto shape = validate_recurrent_inputs(
      core_attn_out,
      q,
      k,
      v,
      raw_gate,
      beta,
      recurrent_state,
      a_log,
      dt_bias,
      num_actual_tokens);
  if (num_actual_tokens == 0) {
    return;
  }

  auto& queue = vllm::xpu::vllmGetQueue();
  dispatch_activation(q.scalar_type(), [&](auto tag) {
    using T = decltype(tag);
    launch_kda_recurrent<T>(
        queue,
        core_attn_out,
        q,
        k,
        v,
        raw_gate,
        beta,
        recurrent_state,
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
        shape.num_heads,
        shape.head_dim);
  });
}

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
  auto qkv = kda_causal_conv1d(
      q_proj,
      k_proj,
      v_proj,
      conv_state,
      q_conv_weight,
      k_conv_weight,
      v_conv_weight,
      num_prefills,
      num_decodes,
      num_spec_decodes,
      has_initial_state,
      non_spec_query_start_loc,
      non_spec_token_indx,
      non_spec_state_indices,
      spec_query_start_loc,
      spec_token_indx,
      spec_state_indices,
      num_accepted_tokens,
      num_actual_tokens);
  kda_gated_delta_rule(
      core_attn_out,
      qkv[0],
      qkv[1],
      qkv[2],
      raw_gate,
      beta,
      recurrent_state,
      a_log,
      dt_bias,
      num_prefills,
      num_decodes,
      num_spec_decodes,
      has_initial_state,
      non_spec_query_start_loc,
      non_spec_token_indx,
      non_spec_state_indices,
      spec_query_start_loc,
      spec_token_indx,
      spec_state_indices,
      num_accepted_tokens,
      num_actual_tokens);
}
