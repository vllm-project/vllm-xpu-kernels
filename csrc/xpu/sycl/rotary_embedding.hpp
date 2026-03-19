// SPDX-License-Identifier: Apache-2.0
// Common rotary embedding primitives shared by multimodal_rope and
// deepseek_scaling_rope kernels.

#pragma once

#include <sycl/sycl.hpp>

namespace vllm {

// ── Element-level rotation ──────────────────────────────────────────────────
// Applies the rotary embedding to a single (rot_offset) pair of elements.
//
// When used in-place, pass the same pointer for both `input` and `output`.
// Reads are completed before writes, so aliasing is safe.

template <typename scalar_t, bool IS_NEOX>
inline void apply_token_rotary_embedding(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr,
    int rot_offset,
    int embed_dim) {
  if constexpr (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    const int x_index = rot_offset;
    const int y_index = embed_dim + rot_offset;
    const scalar_t cos = cos_ptr[x_index];
    const scalar_t sin = sin_ptr[x_index];

    const scalar_t x = input[x_index];
    const scalar_t y = input[y_index];
    output[x_index] = x * cos - y * sin;
    output[y_index] = y * cos + x * sin;
  } else {
    // GPT-J style rotary embedding.
    // Interleaved layout: elements are paired as [x0,y0, x1,y1, ...].
    // Use vec2 to load/rotate/store a pair in one shot.
    using v2_type = sycl::vec<scalar_t, 2>;
    const auto* input_v2 = reinterpret_cast<const v2_type*>(input);
    auto* output_v2 = reinterpret_cast<v2_type*>(output);

    const scalar_t c = cos_ptr[rot_offset];
    const scalar_t s = sin_ptr[rot_offset];
    const v2_type c2 = {c, c};
    const v2_type s2 = {s, s};
    const v2_type t = input_v2[rot_offset];
    const v2_type tr = {-t[1], t[0]};
    output_v2[rot_offset] = t * c2 + tr * s2;
  }
}

// ── Per-token multi-head rotation ───────────────────────────────────────────
// Loops over all heads (query and optionally key) for one token, distributing
// work across SYCL work-items.  `cache_ptr` points to [cos | sin] of size
// rot_dim for the current token.

template <typename scalar_t, bool IS_NEOX>
inline void apply_rotary_embedding(
    scalar_t* __restrict__ query,  // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,    // nullptr or
                                   // [batch_size, seq_len, num_kv_heads,
                                   // head_size] or [num_tokens, num_kv_heads,
                                   // head_size]
    const scalar_t* cache_ptr,
    const int head_size,
    const int num_heads,
    const int num_kv_heads,
    const int rot_dim,
    const int token_idx,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t head_stride,
    const sycl::nd_item<3>& item_ct1) {
  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = item_ct1.get_local_id(2); i < nq;
       i += item_ct1.get_local_range(2)) {
    const int head_idx = i / embed_dim;
    const int64_t token_head =
        token_idx * query_stride + head_idx * head_stride;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, IS_NEOX>(
        query + token_head,
        query + token_head,
        cos_ptr,
        sin_ptr,
        rot_offset,
        embed_dim);
  }

  if (key != nullptr) {
    const int nk = num_kv_heads * embed_dim;
    for (int i = item_ct1.get_local_id(2); i < nk;
         i += item_ct1.get_local_range(2)) {
      const int head_idx = i / embed_dim;
      const int64_t token_head =
          token_idx * key_stride + head_idx * head_stride;
      const int rot_offset = i % embed_dim;
      apply_token_rotary_embedding<scalar_t, IS_NEOX>(
          key + token_head,
          key + token_head,
          cos_ptr,
          sin_ptr,
          rot_offset,
          embed_dim);
    }
  }
}

}  // namespace vllm
