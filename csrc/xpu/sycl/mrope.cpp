// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// SYCL kernel for Multi-head Rotary Positional Embedding (MRoPE).
//
// Key differences vs deepseek_scaling_rope.cpp:
//  1. positions is [3, num_tokens] (T/H/W multimodal positions) rather than
//     [num_tokens].
//  2. The rotary dim space [0, rotary_dim/2) is split into three sections:
//       [0, mrope_section_t)                       → use T positions
//       [mrope_section_t, mrope_section_t+h)        → use H positions
//       [mrope_section_t+h, rotary_dim/2)           → use W positions
//  3. head_size >= rotary_dim is supported; dimensions beyond rotary_dim are
//     copied through unchanged.

#include <sycl/sycl.hpp>
#include "utils.h"
#include "dispatch_utils.h"
#include <cmath>
#include <c10/macros/Macros.h>

namespace vllm {

// ---------------------------------------------------------------------------
// Kernel class
// ---------------------------------------------------------------------------

template <typename T, int64_t rotary_dim, bool is_neox>
class mrope_kernel {
 public:
  static constexpr int     sg_size         = 16;
  static constexpr int64_t half_rotary_dim = rotary_dim / 2;

  mrope_kernel(
      const int64_t* positions,   // [3, num_tokens], row-major
      const T*       query,       // [num_tokens, q_num_head, head_size] (may be non-contiguous)
      const T*       key,         // [num_tokens, k_num_head, head_size] (may be non-contiguous)
      const T*       cos_sin_cache, // [max_pos, rotary_dim]  (first half=cos, second half=sin)
      T*             query_out,   // [num_tokens, q_num_head, head_size] contiguous
      T*             key_out,     // [num_tokens, k_num_head, head_size] contiguous
      const int64_t  num_tokens,
      const int64_t  q_num_head,
      const int64_t  k_num_head,
      const int64_t  head_size,
      const int64_t  mrope_section_t,   // # half-dims belonging to T (temporal)
      const int64_t  mrope_section_h,   // # half-dims belonging to H (height)
      const int64_t  q_num_head_d,      // stride: between q heads in input
      const int64_t  q_batch_d,         // stride: between tokens in q input
      const int64_t  k_num_head_d,      // stride: between k heads in input
      const int64_t  k_batch_d          // stride: between tokens in k input
  )
      : positions(positions),
        query(query),
        key(key),
        cos_sin_cache(cos_sin_cache),
        query_out(query_out),
        key_out(key_out),
        num_tokens(num_tokens),
        q_num_head(q_num_head),
        k_num_head(k_num_head),
        head_size(head_size),
        mrope_section_t(mrope_section_t),
        mrope_section_h(mrope_section_h),
        q_num_head_d(q_num_head_d),
        q_batch_d(q_batch_d),
        k_num_head_d(k_num_head_d),
        k_batch_d(k_batch_d) {}

  // ------------------------------------------------------------------------
  // Inner rotation helper
  //
  // Applies MRoPE rotation to one head vector `pe` (length head_size) and
  // writes the result into `res`.
  //
  //  • Neox style (is_neox=true):
  //      for i in [0, rotary_dim/2):
  //        j = i + rotary_dim/2
  //        res[i] = pe[i]*cos[i] - pe[j]*sin[i]
  //        res[j] = pe[j]*cos[i] + pe[i]*sin[i]
  //
  //  • GptJ / interleaved style (is_neox=false):
  //      for i in [0, rotary_dim/2):
  //        pair = (pe[2i], pe[2i+1])
  //        rotate = (-pe[2i+1], pe[2i])
  //        res[2i]   = pe[2i]*cos[i]   - pe[2i+1]*sin[i]
  //        res[2i+1] = pe[2i+1]*cos[i] + pe[2i]*sin[i]
  //
  //  The mrope section selects which row of cos_sin_cache to use for dim i:
  //    i in [0, section_t)          → T position
  //    i in [section_t, section_t+h) → H position
  //    i in [section_t+h, rot/2)    → W position
  //
  //  Dims beyond rotary_dim are copied through unchanged.
  // ------------------------------------------------------------------------
  void rotary_embedding_mrope(
      const int64_t t_pos,
      const int64_t h_pos,
      const int64_t w_pos,
      const T*      pe,
      T*            res) const {

    const int64_t t_boundary = mrope_section_t;
    const int64_t h_boundary = mrope_section_t + mrope_section_h;

    if constexpr (is_neox) {
      for (int64_t i = 0; i < half_rotary_dim; ++i) {
        const int64_t j = i + half_rotary_dim;

        // Select position based on section
        int64_t pos;
        if (i < t_boundary) {
          pos = t_pos;
        } else if (i < h_boundary) {
          pos = h_pos;
        } else {
          pos = w_pos;
        }
        const int64_t row = pos * rotary_dim;
        // Upcast to float32 for accurate FMA computation, then convert back
        const float cv = static_cast<float>(cos_sin_cache[row + i]);
        const float sv = static_cast<float>(cos_sin_cache[row + half_rotary_dim + i]);
        const float pei = static_cast<float>(pe[i]);
        const float pej = static_cast<float>(pe[j]);

        res[i] = static_cast<T>(pei * cv - pej * sv);
        res[j] = static_cast<T>(pej * cv + pei * sv);
      }
    } else {
      // GptJ / interleaved: elements come in adjacent pairs (2i, 2i+1)
      for (int64_t i = 0; i < half_rotary_dim; ++i) {
        int64_t pos;
        if (i < t_boundary) {
          pos = t_pos;
        } else if (i < h_boundary) {
          pos = h_pos;
        } else {
          pos = w_pos;
        }
        const int64_t row = pos * rotary_dim;
        // Upcast to float32 for accurate FMA computation, then convert back
        const float cv = static_cast<float>(cos_sin_cache[row + i]);
        const float sv = static_cast<float>(cos_sin_cache[row + half_rotary_dim + i]);
        const float p0 = static_cast<float>(pe[2 * i]);
        const float p1 = static_cast<float>(pe[2 * i + 1]);

        res[2 * i]     = static_cast<T>(p0 * cv - p1 * sv);
        res[2 * i + 1] = static_cast<T>(p1 * cv + p0 * sv);
      }
    }

    // Pass through any remaining head dims that are not rotated
    for (int64_t i = rotary_dim; i < head_size; ++i) {
      res[i] = pe[i];
    }
  }

  // ------------------------------------------------------------------------
  // SYCL kernel entry point
  //
  // Grid:  global(num_tokens, sg_per_heads * sg_size)  → 3-D: (num_tokens, sg, lane)
  // Heads: q_num_head + k_num_head encoded in dims 1–2 together.
  // ------------------------------------------------------------------------
  [[sycl::reqd_sub_group_size(sg_size)]] void
  operator()(sycl::nd_item<3> idx) const {
    const int64_t token_idx = idx.get_global_id(0);
    const int64_t sg_idx    = idx.get_local_id(1);
    const int64_t local_id  = idx.get_global_id(2);
    const int64_t head_idx  = sg_idx * sg_size + local_id;

    // T/H/W positions for this token
    // positions layout: [3, num_tokens], row-major → pos[section][tok] = positions[section*num_tokens + tok]
    const int64_t t_pos = positions[token_idx];
    const int64_t h_pos = positions[num_tokens + token_idx];
    const int64_t w_pos = positions[2 * num_tokens + token_idx];

    if (head_idx < q_num_head) {
      // Query head
      const int64_t qi_idx  = token_idx * q_batch_d + head_idx * q_num_head_d;
      const int64_t qo_idx  = token_idx * q_num_head * head_size + head_idx * head_size;
      rotary_embedding_mrope(t_pos, h_pos, w_pos, &query[qi_idx], &query_out[qo_idx]);
    } else if (head_idx < q_num_head + k_num_head) {
      // Key head
      const int64_t k_local = head_idx - q_num_head;
      const int64_t ki_idx  = token_idx * k_batch_d + k_local * k_num_head_d;
      const int64_t ko_idx  = token_idx * k_num_head * head_size + k_local * head_size;
      rotary_embedding_mrope(t_pos, h_pos, w_pos, &key[ki_idx], &key_out[ko_idx]);
    }
  }

 private:
  const int64_t* positions;
  const T*       query;
  const T*       key;
  const T*       cos_sin_cache;
  T*             query_out;
  T*             key_out;
  const int64_t  num_tokens;
  const int64_t  q_num_head;
  const int64_t  k_num_head;
  const int64_t  head_size;
  const int64_t  mrope_section_t;
  const int64_t  mrope_section_h;
  const int64_t  q_num_head_d;
  const int64_t  q_batch_d;
  const int64_t  k_num_head_d;
  const int64_t  k_batch_d;
};

}  // namespace vllm

// ---------------------------------------------------------------------------
// Dispatch helper (templated over T)
// ---------------------------------------------------------------------------

template <typename T>
void call_mrope(
    const int64_t* positions,
    const T*       query,
    const T*       key,
    const T*       cos_sin_cache,
    T*             query_out,
    T*             key_out,
    int64_t        num_tokens,
    int64_t        q_num_head,
    int64_t        k_num_head,
    int64_t        head_size,
    int64_t        rotary_dim,
    int64_t        mrope_section_t,
    int64_t        mrope_section_h,
    bool           is_neox,
    int64_t        q_num_head_d,
    int64_t        q_batch_d,
    int64_t        k_num_head_d,
    int64_t        k_batch_d) {

  static constexpr std::array<int, 5> allowed_dims = {32, 64, 96, 128, 256};
  auto it = std::find(allowed_dims.begin(), allowed_dims.end(), rotary_dim);

  TORCH_CHECK(
      it != allowed_dims.end(),
      "mrope: Invalid rotary_dim (", rotary_dim,
      "). Supported values: 32, 64, 96, 128, 256");

  const int rot_idx  = std::distance(allowed_dims.begin(), it);
  const int neox_idx = is_neox ? 1 : 0;
  const int func_idx = neox_idx * static_cast<int>(allowed_dims.size()) + rot_idx;

  using LaunchFn = void (*)(
      sycl::queue&,
      const int64_t*, const T*, const T*, const T*,
      T*, T*,
      int64_t, int64_t, int64_t, int64_t,
      int64_t, int64_t,
      int64_t, int64_t, int64_t, int64_t);

#define MROPE_REGISTER_CASE(dim, neox)                                        \
  [](sycl::queue& q,                                                          \
     const int64_t* pos,                                                      \
     const T*  qi,                                                            \
     const T*  ki,                                                            \
     const T*  cache,                                                         \
     T*        qo,                                                            \
     T*        ko,                                                            \
     int64_t   ntok,                                                          \
     int64_t   qnh,                                                           \
     int64_t   knh,                                                           \
     int64_t   hs,                                                            \
     int64_t   sec_t,                                                         \
     int64_t   sec_h,                                                         \
     int64_t   q_nhd,                                                         \
     int64_t   q_bd,                                                          \
     int64_t   k_nhd,                                                         \
     int64_t   k_bd) {                                                        \
    constexpr int64_t sg_sz      = 16;                                        \
    int64_t sg_per_heads         = (qnh + knh + sg_sz - 1) / sg_sz;          \
    sycl::range<3> local(1, sg_per_heads, sg_sz);                             \
    sycl::range<3> global(ntok, sg_per_heads, sg_sz);                         \
    at::DeviceGuard dg(at::Device(at::kXPU, at::xpu::current_device()));      \
    q.submit([&](sycl::handler& cgh) {                                        \
      cgh.parallel_for(                                                       \
          sycl::nd_range<3>(global, local),                                   \
          vllm::mrope_kernel<T, dim, neox>{                                   \
              pos, qi, ki, cache, qo, ko,                                     \
              ntok, qnh, knh, hs,                                             \
              sec_t, sec_h,                                                   \
              q_nhd, q_bd, k_nhd, k_bd});                                     \
    });                                                                       \
  }

  static constexpr std::array<LaunchFn, allowed_dims.size() * 2> table = {
      MROPE_REGISTER_CASE(32,  false),
      MROPE_REGISTER_CASE(64,  false),
      MROPE_REGISTER_CASE(96,  false),
      MROPE_REGISTER_CASE(128, false),
      MROPE_REGISTER_CASE(256, false),
      MROPE_REGISTER_CASE(32,  true),
      MROPE_REGISTER_CASE(64,  true),
      MROPE_REGISTER_CASE(96,  true),
      MROPE_REGISTER_CASE(128, true),
      MROPE_REGISTER_CASE(256, true),
  };

  auto& queue = vllm::xpu::vllmGetQueue();
  table[func_idx](
      queue,
      positions, query, key, cos_sin_cache,
      query_out, key_out,
      num_tokens, q_num_head, k_num_head, head_size,
      mrope_section_t, mrope_section_h,
      q_num_head_d, q_batch_d, k_num_head_d, k_batch_d);

#undef MROPE_REGISTER_CASE
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/**
 * @brief Multi-head Rotary Positional Embedding for multimodal models
 *        (Qwen2-VL, Llama4, etc.).
 *
 * @param positions     [3, num_tokens] int64 – T / H / W position indices per token.
 *                      For text-only tokens all three rows are equal.
 * @param query         [num_tokens, q_num_head, head_size]  (may be non-contiguous)
 * @param key           [num_tokens, k_num_head, head_size]  (may be non-contiguous)
 * @param cos_sin_cache [max_seq_len, rotary_dim]  (first half = cos, second half = sin)
 * @param rotary_dim    Length of rotary portion of each head (≤ head_size).
 * @param mrope_section List of 3 ints [t, h, w] where t+h+w = rotary_dim/2.
 *                      Each value gives the number of rotary half-dims assigned
 *                      to temporal / height / width positions respectively.
 * @param is_neox_style true → neox (split first/second half),
 *                      false → gptj (interleaved pairs).
 * @return Pair (query_out, key_out) with rotary embedding applied.
 */
std::tuple<torch::Tensor, torch::Tensor> mrope(
    const torch::Tensor&              positions,
    const torch::Tensor&              query,
    const torch::Tensor&              key,
    const torch::Tensor&              cos_sin_cache,
    int64_t                           rotary_dim,
    const std::vector<int64_t>&       mrope_section,
    bool                              is_neox_style) {

  // ----- Validate inputs --------------------------------------------------
  TORCH_CHECK(
      positions.dim() == 2 && positions.size(0) == 3,
      "mrope: positions must have shape [3, num_tokens], got ",
      positions.sizes());
  TORCH_CHECK(
      mrope_section.size() == 3,
      "mrope: mrope_section must have 3 elements [t, h, w]");
  TORCH_CHECK(
      positions.scalar_type() == torch::kInt64,
      "mrope: positions must be int64");
  TORCH_CHECK(
      query.dim() == 3 && key.dim() == 3,
      "mrope: query and key must be 3-D [num_tokens, num_heads, head_size]");

  const int64_t num_tokens = positions.size(1);
  const int64_t q_num_head = query.size(1);
  const int64_t k_num_head = key.size(1);
  const int64_t head_size  = query.size(2);

  TORCH_CHECK(
      key.size(2) == head_size,
      "mrope: query and key must have the same head_size");
  TORCH_CHECK(
      rotary_dim <= head_size,
      "mrope: rotary_dim (", rotary_dim, ") must be ≤ head_size (", head_size, ")");
  TORCH_CHECK(
      cos_sin_cache.size(1) == rotary_dim,
      "mrope: cos_sin_cache second dim (", cos_sin_cache.size(1),
      ") must equal rotary_dim (", rotary_dim, ")");
  TORCH_CHECK(
      mrope_section[0] + mrope_section[1] + mrope_section[2] == rotary_dim / 2,
      "mrope: sum(mrope_section) must equal rotary_dim/2 (", rotary_dim / 2, ")");

  const int64_t mrope_section_t = mrope_section[0];
  const int64_t mrope_section_h = mrope_section[1];

  const auto q_stride = query.strides();
  const auto k_stride = key.strides();
  const int64_t q_batch_d   = q_stride[0];
  const int64_t q_num_head_d = q_stride[1];
  const int64_t k_batch_d   = k_stride[0];
  const int64_t k_num_head_d = k_stride[1];

  auto query_out = at::empty_like(query);
  auto key_out   = at::empty_like(key);

  // Ensure output is contiguous [num_tokens, num_heads, head_size]
  TORCH_CHECK(query_out.is_contiguous());
  TORCH_CHECK(key_out.is_contiguous());

  switch (query.scalar_type()) {
    case torch::kFloat:
      call_mrope<float>(
          reinterpret_cast<const int64_t*>(positions.data_ptr()),
          reinterpret_cast<const float*>(query.data_ptr()),
          reinterpret_cast<const float*>(key.data_ptr()),
          reinterpret_cast<const float*>(cos_sin_cache.data_ptr()),
          reinterpret_cast<float*>(query_out.data_ptr()),
          reinterpret_cast<float*>(key_out.data_ptr()),
          num_tokens, q_num_head, k_num_head, head_size,
          rotary_dim, mrope_section_t, mrope_section_h, is_neox_style,
          q_num_head_d, q_batch_d, k_num_head_d, k_batch_d);
      break;

    case torch::kFloat16:
      call_mrope<sycl::half>(
          reinterpret_cast<const int64_t*>(positions.data_ptr()),
          reinterpret_cast<const sycl::half*>(query.data_ptr()),
          reinterpret_cast<const sycl::half*>(key.data_ptr()),
          reinterpret_cast<const sycl::half*>(cos_sin_cache.data_ptr()),
          reinterpret_cast<sycl::half*>(query_out.data_ptr()),
          reinterpret_cast<sycl::half*>(key_out.data_ptr()),
          num_tokens, q_num_head, k_num_head, head_size,
          rotary_dim, mrope_section_t, mrope_section_h, is_neox_style,
          q_num_head_d, q_batch_d, k_num_head_d, k_batch_d);
      break;

    case torch::kBFloat16:
      call_mrope<sycl::ext::oneapi::bfloat16>(
          reinterpret_cast<const int64_t*>(positions.data_ptr()),
          reinterpret_cast<const sycl::ext::oneapi::bfloat16*>(query.data_ptr()),
          reinterpret_cast<const sycl::ext::oneapi::bfloat16*>(key.data_ptr()),
          reinterpret_cast<const sycl::ext::oneapi::bfloat16*>(cos_sin_cache.data_ptr()),
          reinterpret_cast<sycl::ext::oneapi::bfloat16*>(query_out.data_ptr()),
          reinterpret_cast<sycl::ext::oneapi::bfloat16*>(key_out.data_ptr()),
          num_tokens, q_num_head, k_num_head, head_size,
          rotary_dim, mrope_section_t, mrope_section_h, is_neox_style,
          q_num_head_d, q_batch_d, k_num_head_d, k_batch_d);
      break;

    default:
      TORCH_CHECK(
          false,
          "mrope: unsupported dtype. Only float32, float16, bfloat16 are supported.");
  }

  return {query_out, key_out};
}
