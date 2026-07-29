#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "chunk_kda_launcher_xe2.hpp"
#include "chunk_kda_xe2.h"

namespace {

// Upper bound on the chunk-aligned "virtual" token space: every sequence
// contributes at most ceil(len / chunk) chunks, so padding one extra chunk per
// sequence is always enough.
int64_t virtual_seqlen(int64_t batch_size, int64_t num_actual_tokens) {
  const int64_t cs = kda_xe2::chunk_size;
  const int64_t chunks = (num_actual_tokens + cs - 1) / cs + batch_size;
  return chunks * cs;
}

// `fwd_o` may run several work-groups per (sequence, head), one per value
// slice, and each of them needs its own scratch plane for the intra-chunk
// attention tile.
int64_t o2_planes(int64_t batch_size, int64_t num_heads, int64_t head_dim) {
  return kda_xe2::chunk_kda_fwd_o_dv_groups(
      static_cast<int>(batch_size),
      static_cast<int>(num_heads),
      static_cast<int>(head_dim));
}

}  // namespace

bool chunk_kda_xe2_supported(int64_t head_dim) {
  // The DPAS tiles are chunk_size x chunk_size, so the head dimension must be
  // a whole number of tiles.
  return head_dim >= kda_xe2::chunk_size && head_dim % kda_xe2::chunk_size == 0;
}

int64_t chunk_kda_xe2_workspace_bytes(
    int64_t batch_size,
    int64_t num_actual_tokens,
    int64_t num_heads,
    int64_t head_dim,
    int64_t element_size) {
  const int64_t vt = virtual_seqlen(batch_size, num_actual_tokens);
  // Ka, Kb, Qt, Vp, W, U are [heads, vt, dim]; A is [o2_planes, heads, vt,
  // chunk]; Tl is [heads, vt / chunk, dim] in fp32.
  const int64_t wide = 6 * num_heads * vt * head_dim * element_size;
  const int64_t a_bytes = o2_planes(batch_size, num_heads, head_dim) *
                          num_heads * vt * kda_xe2::chunk_size * element_size;
  const int64_t tl_bytes =
      num_heads * (vt / kda_xe2::chunk_size) * head_dim * 4;
  return wide + a_bytes + tl_bytes;
}

void chunk_kda_xe2(
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
    const torch::Tensor& query_start_loc,
    const torch::Tensor& state_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const std::optional<torch::Tensor>& token_indx,
    int64_t batch_size,
    int64_t num_actual_tokens,
    int64_t num_heads,
    int64_t head_dim) {
  TORCH_CHECK(
      chunk_kda_xe2_supported(head_dim),
      "chunk_kda_xe2 requires head_dim to be a positive multiple of ",
      kda_xe2::chunk_size,
      ", got ",
      head_dim);

  const int64_t vt = virtual_seqlen(batch_size, num_actual_tokens);
  const auto dtype = q.scalar_type();
  const auto device = q.device();
  const auto opts = torch::dtype(dtype).device(device).requires_grad(false);

  // One allocation for the six [heads, vt, dim] operands keeps the number of
  // caching-allocator round trips down and guarantees they stay adjacent.
  torch::Tensor wide = torch::empty({6, num_heads, vt, head_dim}, opts);
  torch::Tensor A = torch::empty(
      {o2_planes(batch_size, num_heads, head_dim),
       num_heads,
       vt,
       kda_xe2::chunk_size},
      opts);
  torch::Tensor Tl = torch::empty(
      {num_heads, vt / kda_xe2::chunk_size, head_dim},
      torch::dtype(torch::kFloat32).device(device).requires_grad(false));

  const int* token_indx_ptr =
      token_indx.has_value()
          ? reinterpret_cast<const int*>(token_indx->data_ptr())
          : nullptr;
  const bool* has_initial_state_ptr =
      has_initial_state.has_value()
          ? reinterpret_cast<const bool*>(has_initial_state->data_ptr())
          : nullptr;

#define KDA_CHUNK_LAUNCH(scalar_t, state_t)                       \
  do {                                                            \
    auto* base = reinterpret_cast<scalar_t*>(wide.data_ptr());    \
    const int64_t plane = num_heads * vt * head_dim;              \
    kda_xe2::chunk_kda_launcher<scalar_t, state_t>(               \
        queue,                                                    \
        reinterpret_cast<scalar_t*>(core_attn_out.data_ptr()),    \
        reinterpret_cast<const scalar_t*>(q.data_ptr()),          \
        reinterpret_cast<const scalar_t*>(k.data_ptr()),          \
        reinterpret_cast<const scalar_t*>(v.data_ptr()),          \
        reinterpret_cast<const scalar_t*>(raw_gate.data_ptr()),   \
        reinterpret_cast<const float*>(beta.data_ptr()),          \
        reinterpret_cast<const float*>(a_log.data_ptr()),         \
        reinterpret_cast<const float*>(dt_bias.data_ptr()),       \
        base,                                                     \
        base + plane,                                             \
        base + 2 * plane,                                         \
        base + 3 * plane,                                         \
        reinterpret_cast<scalar_t*>(A.data_ptr()),                \
        base + 4 * plane,                                         \
        base + 5 * plane,                                         \
        reinterpret_cast<float*>(Tl.data_ptr()),                  \
        reinterpret_cast<state_t*>(recurrent_state.data_ptr()),   \
        recurrent_state.stride(0),                                \
        reinterpret_cast<const int*>(query_start_loc.data_ptr()), \
        reinterpret_cast<const int*>(state_indices.data_ptr()),   \
        has_initial_state_ptr,                                    \
        token_indx_ptr,                                           \
        static_cast<int>(batch_size),                             \
        static_cast<int>(vt),                                     \
        static_cast<int>(num_heads),                              \
        static_cast<int>(head_dim));                              \
  } while (0)

#define KDA_CHUNK_DISPATCH_STATE(scalar_t)                                \
  do {                                                                    \
    if (recurrent_state.scalar_type() == at::kFloat) {                    \
      KDA_CHUNK_LAUNCH(scalar_t, float);                                  \
    } else if (recurrent_state.scalar_type() == at::kBFloat16) {          \
      KDA_CHUNK_LAUNCH(scalar_t, cutlass::bfloat16_t);                    \
    } else if (recurrent_state.scalar_type() == at::kHalf) {              \
      KDA_CHUNK_LAUNCH(scalar_t, cutlass::half_t);                        \
    } else {                                                              \
      TORCH_CHECK(                                                        \
          false,                                                          \
          "recurrent_state dtype must be float32/float16/bfloat16, got ", \
          recurrent_state.scalar_type());                                 \
    }                                                                     \
  } while (0)

  if (dtype == at::kBFloat16) {
    KDA_CHUNK_DISPATCH_STATE(cutlass::bfloat16_t);
  } else if (dtype == at::kHalf) {
    KDA_CHUNK_DISPATCH_STATE(cutlass::half_t);
  } else {
    TORCH_CHECK(false, "chunk_kda_xe2 activations must be float16 or bfloat16");
  }

#undef KDA_CHUNK_DISPATCH_STATE
#undef KDA_CHUNK_LAUNCH
}
