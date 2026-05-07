// SPDX-License-Identifier: Apache-2.0
// Common rotary embedding primitives shared by multimodal_rope and
// deepseek_scaling_rope kernels.
//
// This file provides element-level and vectorized rotation functions for
// both NeoX (split-half) and GPT-J (interleaved) layouts:
//
//   NeoX:  [x0 x1 … x_{d/2-1} | y0 y1 … y_{d/2-1}]
//   GPT-J: [x0 y0 x1 y1 x2 y2 …]
//
// Vectorized variants:
//   - apply_token_rotary_embedding_vec<VEC_SIZE>:  NeoX vec4  (4 offsets/call)
//   - apply_token_rotary_embedding_gptj_vec4:      GPT-J vec4 (2 pairs/call)

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
    // GPT-NeoX style: x and y are separated by embed_dim.
    const int x_index = rot_offset;
    const int y_index = embed_dim + rot_offset;
    const scalar_t cos = cos_ptr[x_index];
    const scalar_t sin = sin_ptr[x_index];

    const scalar_t x = input[x_index];
    const scalar_t y = input[y_index];
    output[x_index] = x * cos - y * sin;
    output[y_index] = y * cos + x * sin;
  } else {
    // GPT-J style: interleaved [x0, y0, x1, y1, ...].
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

// ── Vectorized NeoX rotation (vec4) ─────────────────────────────────────────
// Processes VEC_SIZE consecutive rotation offsets in a single shot using SYCL
// vector types.  Requires rot_offset to be VEC_SIZE-aligned and
// rot_offset + VEC_SIZE <= embed_dim.

template <typename scalar_t, int VEC_SIZE>
inline void apply_token_rotary_embedding_vec(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr,
    int rot_offset,
    int embed_dim) {
  using vec_t = sycl::vec<scalar_t, VEC_SIZE>;

  const int x_index = rot_offset;
  const int y_index = embed_dim + rot_offset;

  vec_t x = *reinterpret_cast<const vec_t*>(input + x_index);
  vec_t y = *reinterpret_cast<const vec_t*>(input + y_index);
  vec_t cos_v = *reinterpret_cast<const vec_t*>(cos_ptr + rot_offset);
  vec_t sin_v = *reinterpret_cast<const vec_t*>(sin_ptr + rot_offset);

  *reinterpret_cast<vec_t*>(output + x_index) = x * cos_v - y * sin_v;
  *reinterpret_cast<vec_t*>(output + y_index) = y * cos_v + x * sin_v;
}

// ── Vectorized GPT-J rotation (vec4, 2 pairs per call) ──────────────────────
// Interleaved layout: input = [..., x_i, y_i, x_{i+1}, y_{i+1}, ...]
// One vec4 load grabs 2 consecutive (x,y) pairs and rotates them together.
//
// rot_offset: the first of 2 consecutive rotation offsets to process.
//             Must satisfy rot_offset + 2 <= embed_dim.
//             The corresponding memory range is
//             input[rot_offset*2 .. rot_offset*2+3].

template <typename scalar_t>
inline void apply_token_rotary_embedding_gptj_vec4(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr,
    int rot_offset) {
  using vec4_t = sycl::vec<scalar_t, 4>;
  using vec2_t = sycl::vec<scalar_t, 2>;

  // Load 2 consecutive interleaved pairs: {x0, y0, x1, y1}
  vec4_t v = *reinterpret_cast<const vec4_t*>(input + rot_offset * 2);

  // Load cos/sin for 2 consecutive offsets
  vec2_t cs = *reinterpret_cast<const vec2_t*>(cos_ptr + rot_offset);
  vec2_t ss = *reinterpret_cast<const vec2_t*>(sin_ptr + rot_offset);

  // Expand to vec4: {c0, c0, c1, c1}, {s0, s0, s1, s1}
  vec4_t c4 = {cs[0], cs[0], cs[1], cs[1]};
  vec4_t s4 = {ss[0], ss[0], ss[1], ss[1]};

  // Swizzle: {x0, y0, x1, y1} → {-y0, x0, -y1, x1}
  vec4_t vr = {-v[1], v[0], -v[3], v[2]};

  // Rotation: output = v * cos + vr * sin
  *reinterpret_cast<vec4_t*>(output + rot_offset * 2) = v * c4 + vr * s4;
}

}  // namespace vllm
