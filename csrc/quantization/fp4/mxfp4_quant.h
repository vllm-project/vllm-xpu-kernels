// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <type_traits>

#include "quantization/moe_scale_layout.h"
#include "quantization/swiglu_limit.h"
#include "quantization/utils.h"

#include "quantization/fp4/mxfp4_quant_asm.h"

namespace vllm {
namespace mxfp4 {

// Maximum representable FP4 E2M1 value.
static constexpr float FP4_MAX = 6.0f;

// Convert a single float32 value to a 4-bit FP4 E2M1 code (stored in the
// lower 4 bits of a uint8_t). Matches the to_mxfp reference: rounds to bf16
// first, then to e2m1 with round-to-nearest-even (RNE).
inline uint8_t float_to_fp4_e2m1(float x) {
  // E2M1 midpoints shifted by half a bf16 ulp, so that comparing x matches
  // comparing bf16(x); >= where the tie rounds up to the even code.
  const float b[7] = {
      0.25f + 0x1p-10f,
      0.75f - 0x1p-9f,
      1.25f + 0x1p-8f,
      1.75f - 0x1p-8f,
      2.5f + 0x1p-7f,
      3.5f - 0x1p-7f,
      5.0f + 0x1p-6f};

  uint8_t sign = (x < 0.0f) ? 0x8u : 0x0u;
  float a = sycl::fabs(x);

  // midpoint rounding: branchless
  uint8_t code = (a > b[0]) + (a >= b[1]) + (a > b[2]) + (a >= b[3]) +
                 (a > b[4]) + (a >= b[5]) + (a > b[6]);

  return code | sign;
}

// Quantise an input tensor [M, N] to MXFP4 block format.
// output_q  : packed FP4 nibbles, two elements per uint8 byte.
//             Byte at index i stores elements [2i, 2i+1]:
//             byte[i] = fp4[2i+1] << 4 | fp4[2i]
// output_s  : UE8M0-rounded scale (power-of-two) stored as float32.
//             One scale per group_size input elements.
template <typename scalar_t>
class per_token_group_quant_mxfp4_kernel {
 private:
  uint8_t* out;
  float* scale;
  scalar_t const* input;
  const int group_size;  // = 32 for MX format
  const int groups_per_block;
  float eps;
  const int scale_num_cols;  // N / group_size  (columns in scale matrix)
  const int scale_stride;    // stride(1) of scale tensor
  bool is_column_major;
  MoEScaleLayout moe_layout;

 public:
  per_token_group_quant_mxfp4_kernel(
      uint8_t* out_,
      float* scale_,
      scalar_t const* input_,
      const int group_size_,
      const int groups_per_block_,
      float eps_,
      const int scale_num_cols_ = 0,
      const int scale_stride_ = 0,
      bool is_column_major_ = false,
      MoEScaleLayout moe_layout_ = {})
      : out(out_),
        scale(scale_),
        input(input_),
        group_size(group_size_),
        groups_per_block(groups_per_block_),
        eps(eps_),
        scale_num_cols(scale_num_cols_),
        scale_stride(scale_stride_),
        is_column_major(is_column_major_),
        moe_layout(moe_layout_) {}

  void operator()
      [[sycl::reqd_sub_group_size(32)]] (sycl::nd_item<1> item) const {
    constexpr int THREADS_PER_GROUP = 32;

    int local_id = item.get_local_id(0);
    int local_group_id = local_id / THREADS_PER_GROUP;
    int lane_id = local_id % THREADS_PER_GROUP;

    int block_group_id = item.get_group(0) * groups_per_block;
    int global_group_id = block_group_id + local_group_id;

    int64_t group_offset = static_cast<int64_t>(global_group_id) * group_size;
    scalar_t const* group_input = input + group_offset;
    // Two FP4 nibbles packed into one byte, so the output offset is halved.
    uint8_t* group_output = out + group_offset / 2;

    // Resolve the scale output pointer for this group.
    float* scale_output = nullptr;
    if (is_column_major && !moe_layout.enabled()) {
      const int row_idx = global_group_id / scale_num_cols;
      const int col_idx = global_group_id % scale_num_cols;
      scale_output = scale + col_idx * scale_stride + row_idx;
    } else if (!moe_layout.enabled()) {
      scale_output = scale + global_group_id;
    }

    float local_absmax = eps;
    if (lane_id < group_size) {
      float val = static_cast<float>(group_input[lane_id]);
      local_absmax = sycl::max(local_absmax, sycl::fabs(val));
    }
    local_absmax = sycl::reduce_over_group(
        item.get_sub_group(), local_absmax, sycl::maximum<float>());

    float y_s = local_absmax / FP4_MAX;
    y_s = sycl::exp2(sycl::ceil(sycl::log2(sycl::fmax(sycl::fabs(y_s), eps))));

    if (lane_id == 0) {
      if (moe_layout.enabled()) {
        moe_layout.store(
            global_group_id / moe_layout.scale_k,
            global_group_id % moe_layout.scale_k,
            y_s);
      } else {
        *scale_output = y_s;
      }
    }
    sycl::group_barrier(item.get_group());

    const float inv_scale = 1.0f / y_s;

    if (lane_id < group_size) {
      float scaled_val =
          mxfp4_scale_clamp_asm(group_input[lane_id], inv_scale);
      uint8_t fp4_val = float_to_fp4_e2m1(scaled_val);

      // Pack: even lane → low nibble, odd lane (lane+1) → high nibble.
      // Retrieve the adjacent (odd) thread's FP4 code via sub-group shuffle.
      uint8_t next_fp4 = static_cast<uint8_t>(sycl::shift_group_left(
          item.get_sub_group(), static_cast<uint32_t>(fp4_val), 1u));

      if (lane_id % 2 == 0) {
        // byte[k] = fp4[2k+1] << 4 | fp4[2k]
        group_output[lane_id / 2] =
            ((next_fp4 & 0x0Fu) << 4) | (fp4_val & 0x0Fu);
      }
    }
  }
};

// Vectorized MXFP4 quantiser: one work-group per row (or per rows_per_wg
// rows), each work-item reads CPT consecutive 16-byte chunks. CPT is picked in
// mxfp4_quant.cpp and always divides the chunks in a group; when
// CPT == chunks_per_group the cross-lane absmax reduction disappears.
// FUSE_SILU_MUL folds a SwiGLU activation into the load: the row then holds
// the gate and up halves back to back and is twice as wide.
// MULTI_ROW = false requires rows_per_wg == 1 and drops the row-packing index
// math, which is measurable on the latency-bound few-row launches.
template <
    typename scalar_t,
    int CPT = 1,
    bool FLAT_GROUPS = false,
    bool FUSE_SILU_MUL = false,
    bool CLAMP = false,
    bool MULTI_ROW = true>
class per_token_group_quant_mxfp4_vec_kernel {
 private:
  uint8_t* __restrict__ out;
  float* __restrict__ scale;
  scalar_t const* __restrict__ input;
  const int hidden_size;
  const int group_size;  // = 32 for MX format
  float eps;
  const int64_t scale_stride_token;
  const int64_t scale_stride_group;
  const int num_rows;
  const int rows_per_wg;
  MoEScaleLayout moe_layout;
  // Only read with FUSE_SILU_MUL and CLAMP.
  float swiglu_limit;

 public:
  per_token_group_quant_mxfp4_vec_kernel(
      uint8_t* out_,
      float* scale_,
      scalar_t const* input_,
      const int hidden_size_,
      const int group_size_,
      float eps_,
      const int64_t scale_stride_token_,
      const int64_t scale_stride_group_,
      const int num_rows_,
      const int rows_per_wg_,
      MoEScaleLayout moe_layout_ = {},
      float swiglu_limit_ = kSwigluLimitDisabled)
      : out(out_),
        scale(scale_),
        input(input_),
        hidden_size(hidden_size_),
        group_size(group_size_),
        eps(eps_),
        scale_stride_token(scale_stride_token_),
        scale_stride_group(scale_stride_group_),
        num_rows(num_rows_),
        rows_per_wg(rows_per_wg_),
        moe_layout(moe_layout_),
        swiglu_limit(swiglu_limit_) {}

  void operator()
      [[sycl::reqd_sub_group_size(32)]] (sycl::nd_item<1> item) const {
    const int tid = static_cast<int>(item.get_local_id(0));
    const int local_range = static_cast<int>(item.get_local_range(0));
    auto sg = item.get_sub_group();

    constexpr int VEC = 16 / sizeof(scalar_t);
    constexpr int OUT_VEC = CPT * VEC / 2;
    static_assert(!FLAT_GROUPS || CPT * VEC == 32);
    const int width = FLAT_GROUPS ? 32 : hidden_size;
    const int chunks_per_group = (FLAT_GROUPS ? 32 : group_size) / VEC;
    const int lanes_per_group = chunks_per_group / CPT;
    const int num_chunks = width / VEC;
    const int threads_per_row = num_chunks / CPT;

    // Packing rows_per_wg rows into one work-group keeps every lane issuing
    // loads when a row is narrower than a sub-group. Under FLAT_GROUPS a
    // 32-element group is itself treated as a row.
    int stride;
    int row_idx;
    int t0;
    if constexpr (FLAT_GROUPS) {
      stride = 1;
      row_idx = static_cast<int>(item.get_global_id(0));
      t0 = 0;
      if (row_idx >= num_rows) return;
    } else if constexpr (!MULTI_ROW) {
      stride = local_range;
      row_idx = static_cast<int>(item.get_group(0));
      t0 = tid;
    } else {
      stride = local_range / rows_per_wg;
      row_idx =
          static_cast<int>(item.get_group(0)) * rows_per_wg + tid / stride;
      t0 = tid % stride;
      if (row_idx >= num_rows) return;
    }

    // With FUSE_SILU_MUL the row holds gate and up back to back, so it is
    // twice as wide and the up half starts num_chunks vectors in.
    static_assert(!(FUSE_SILU_MUL && FLAT_GROUPS));
    const int64_t row_stride =
        static_cast<int64_t>(width) * (FUSE_SILU_MUL ? 2 : 1);
    const scalar_t* row_input =
        input + static_cast<int64_t>(row_idx) * row_stride;
    uint8_t* row_output = out + static_cast<int64_t>(row_idx) * (width / 2);
    const auto* vin =
        reinterpret_cast<const vec_n_t<scalar_t, VEC>*>(row_input);
    auto* vout = reinterpret_cast<vec_n_t<uint8_t, OUT_VEC>*>(row_output);

    for (int base = 0; base < threads_per_row; base += stride) {
      const int t = base + t0;
      const bool active = t < threads_per_row;
      const int c0 = t * CPT;

      float val[CPT][VEC];
      float lane_absmax = 0.0f;
      if (active) {
        vec_n_t<scalar_t, VEC> v[CPT];
#pragma unroll
        for (int j = 0; j < CPT; ++j) {
          v[j] = vin[c0 + j];
        }
        if constexpr (FUSE_SILU_MUL) {
          vec_n_t<scalar_t, VEC> u[CPT];
#pragma unroll
          for (int j = 0; j < CPT; ++j) {
            u[j] = vin[c0 + j + num_chunks];
          }
#pragma unroll
          for (int j = 0; j < CPT; ++j) {
#pragma unroll
            for (int k = 0; k < VEC; ++k) {
              // Mirror activation.cpp's silu_and_mul rounding exactly so the
              // fused path is bit-identical to running the two kernels.
              // The limit is pre-rounded to scalar_t, so the clamped values
              // are exact in scalar_t and can stay in float.
              float g = static_cast<float>(v[j].val[k]);
              float up = static_cast<float>(u[j].val[k]);
              if constexpr (CLAMP) {
                g = swiglu_clamp_gate(g, swiglu_limit);
                up = swiglu_clamp_up(up, swiglu_limit);
              }
              const scalar_t sg =
                  static_cast<scalar_t>(g / (1.0f + sycl::exp(-g)));
              // The product of two 16-bit floats is exact in float, so this
              // rounds like sg * up in scalar_t.
              const float f = static_cast<float>(
                  static_cast<scalar_t>(static_cast<float>(sg) * up));
              val[j][k] = f;
              lane_absmax = sycl::max(lane_absmax, sycl::fabs(f));
            }
          }
        } else {
#pragma unroll
          for (int j = 0; j < CPT; ++j) {
#pragma unroll
            for (int k = 0; k < VEC; ++k) {
              const float f = static_cast<float>(v[j].val[k]);
              val[j][k] = f;
              lane_absmax = sycl::max(lane_absmax, sycl::fabs(f));
            }
          }
        }
      }

      float gmax = lane_absmax;
#pragma unroll
      for (int off = 1; off < lanes_per_group; off <<= 1) {
        gmax = sycl::max(gmax, sycl::permute_group_by_xor(sg, gmax, off));
      }

      float y_s = gmax / FP4_MAX;
      y_s =
          sycl::exp2(sycl::ceil(sycl::log2(sycl::fmax(sycl::fabs(y_s), eps))));

      if (active && (lanes_per_group <= 1 || (t % lanes_per_group) == 0)) {
        const int g_idx = c0 / chunks_per_group;
        if constexpr (FLAT_GROUPS) {
          scale[row_idx] = y_s;
        } else {
          if (moe_layout.enabled()) {
            moe_layout.store(row_idx, g_idx, y_s);
          } else {
            const int64_t scale_idx =
                static_cast<int64_t>(row_idx) * scale_stride_token +
                static_cast<int64_t>(g_idx) * scale_stride_group;
            scale[scale_idx] = y_s;
          }
        }
      }

      if (active) {
        const float inv_scale = 1.0f / y_s;
        vec_n_t<uint8_t, OUT_VEC> o;
#pragma unroll
        for (int j = 0; j < CPT; ++j) {
#pragma unroll
          for (int k = 0; k < VEC; k += 2) {
            float s0 = sycl::fmax(
                -FP4_MAX, sycl::fmin(val[j][k] * inv_scale, FP4_MAX));
            float s1 = sycl::fmax(
                -FP4_MAX, sycl::fmin(val[j][k + 1] * inv_scale, FP4_MAX));
            uint8_t n0 = float_to_fp4_e2m1(s0);
            uint8_t n1 = float_to_fp4_e2m1(s1);
            o.val[j * (VEC / 2) + k / 2] =
                static_cast<uint8_t>(((n1 & 0x0Fu) << 4) | (n0 & 0x0Fu));
          }
        }
        vout[t] = o;
      }
    }
  }
};

// Quantise one whole 32-element group within a single work-item, so the absmax
// reduction stays local. Returns the scale; packed bytes land in `o`.
template <typename scalar_t, int VEC, int CPT>
inline float mxfp4_quant_one_group(
    const vec_n_t<scalar_t, VEC>* __restrict__ vin,
    int c0,
    float eps,
    vec_n_t<uint8_t, CPT * VEC / 2>& o) {
  vec_n_t<scalar_t, VEC> v[CPT];
#pragma unroll
  for (int j = 0; j < CPT; ++j) {
    v[j] = vin[c0 + j];
  }

  float val[CPT][VEC];
  float absmax = 0.0f;
#pragma unroll
  for (int j = 0; j < CPT; ++j) {
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      const float f = static_cast<float>(v[j].val[k]);
      val[j][k] = f;
      absmax = sycl::max(absmax, sycl::fabs(f));
    }
  }

  const float y_s = sycl::exp2(
      sycl::ceil(sycl::log2(sycl::fmax(sycl::fabs(absmax / FP4_MAX), eps))));
  const float inv_scale = 1.0f / y_s;

#pragma unroll
  for (int j = 0; j < CPT; ++j) {
#pragma unroll
    for (int k = 0; k < VEC; k += 2) {
      const float s0 =
          sycl::fmax(-FP4_MAX, sycl::fmin(val[j][k] * inv_scale, FP4_MAX));
      const float s1 =
          sycl::fmax(-FP4_MAX, sycl::fmin(val[j][k + 1] * inv_scale, FP4_MAX));
      const uint8_t n0 = float_to_fp4_e2m1(s0);
      const uint8_t n1 = float_to_fp4_e2m1(s1);
      o.val[j * (VEC / 2) + k / 2] =
          static_cast<uint8_t>(((n1 & 0x0Fu) << 4) | (n0 & 0x0Fu));
    }
  }
  return y_s;
}

inline float pow2_ceil_scale(float x, float& inv_scale) {
  const uint32_t xb = sycl::bit_cast<uint32_t>(x);
  const uint32_t e = (xb >> 23) + ((xb & 0x007FFFFFu) != 0u);
  float y_s;
  if (xb >= 0x00800000u && e <= 253u) {
    y_s = sycl::bit_cast<float>(e << 23);
    inv_scale = sycl::bit_cast<float>((254u - e) << 23);
  } else {
    const bool tiny = xb < 0x00800000u;
    const uint32_t sb = sycl::bit_cast<uint32_t>(tiny ? x * 0x1p64f : x);
    const uint32_t se =
        sycl::min((sb >> 23) + ((sb & 0x007FFFFFu) != 0u), 255u);
    y_s = sycl::bit_cast<float>(se << 23) * (tiny ? 0x1p-64f : 1.0f);
    if (sycl::isnan(x)) y_s = x;
    inv_scale = 1.0f / y_s;
  }
  return y_s;
}

inline float mxfp4_pow2_scale(float absmax, float eps, float& inv_scale) {
  return pow2_ceil_scale(
      sycl::fmax(sycl::fabs(absmax / FP4_MAX), eps), inv_scale);
}

template <typename scalar_t>
inline uint32_t mxfp4_pack8(const float (&val)[8], float inv_scale) {
  uint32_t o;
  uint32_t word = 0;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const float s =
        sycl::fmax(-FP4_MAX, sycl::fmin(val[k] * inv_scale, FP4_MAX));
    const uint8_t code = float_to_fp4_e2m1(s);
    word |= static_cast<uint32_t>(code & 0x0Fu) << (4 * k);
  }
  o = word;
  return o;
}

template <typename scalar_t>
inline float mxfp4_quant_group32(
    const vec_n_t<scalar_t, 8>* __restrict__ vin,
    bool active,
    float eps,
    vec_n_t<uint32_t, 4>& o) {
  static_assert(sizeof(scalar_t) == 2);
  vec_n_t<scalar_t, 8> v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] = active ? vin[j] : vec_n_t<scalar_t, 8>{};
  }
  float val[4][8];
  float absmax = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      val[j][k] = static_cast<float>(v[j].val[k]);
      absmax = sycl::max(absmax, sycl::fabs(val[j][k]));
    }
  }
  float inv_scale;
  const float y_s = mxfp4_pow2_scale(absmax, eps, inv_scale);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    o.val[j] = mxfp4_pack8<scalar_t>(val[j], inv_scale);
  }
  return y_s;
}

// The scale dtype (float or e8m0) is a uniform runtime switch at the final
// store (binary size).
template <typename scalar_t>
class per_token_group_quant_mxfp4_flat_kernel {
 private:
  uint8_t* __restrict__ out;
  void* __restrict__ scale;
  scalar_t const* __restrict__ input;
  const int64_t num_groups;
  float eps;
  const bool scale_e8m0;

 public:
  per_token_group_quant_mxfp4_flat_kernel(
      uint8_t* out_,
      void* scale_,
      scalar_t const* input_,
      const int64_t num_groups_,
      float eps_,
      const bool scale_e8m0_)
      : out(out_),
        scale(scale_),
        input(input_),
        num_groups(num_groups_),
        eps(eps_),
        scale_e8m0(scale_e8m0_) {}

  void operator()
      [[sycl::reqd_sub_group_size(32)]] (sycl::nd_item<1> item) const {
    const int64_t g = static_cast<int64_t>(item.get_global_id(0));
    const bool active = g < num_groups;
    vec_n_t<uint32_t, 4> o;
    const float y_s = mxfp4_quant_group32<scalar_t>(
        reinterpret_cast<const vec_n_t<scalar_t, 8>*>(input) + g * 4,
        active,
        eps,
        o);
    if (!active) {
      return;
    }
    reinterpret_cast<vec_n_t<uint32_t, 4>*>(out)[g] = o;
    if (scale_e8m0) {
      static_cast<uint8_t*>(scale)[g] = MoEScaleLayout::encode(y_s);
    } else {
      static_cast<float*>(scale)[g] = y_s;
    }
  }
};

// The scale-tile write-out order (col_major) and the scale dtype (float or
// e8m0, staged as float in SLM) are uniform runtime switches (binary size).
template <typename scalar_t>
class per_token_group_quant_mxfp4_tile_kernel {
 private:
  uint8_t* __restrict__ out;
  void* __restrict__ scale;
  scalar_t const* __restrict__ input;
  sycl::local_accessor<float, 1> tile;
  const int groups_per_row;
  const int groups_per_wg;
  float eps;
  const int num_rows;
  const int rows_per_wg;
  const int64_t scale_stride_token;
  const int64_t scale_stride_group;
  const bool col_major;
  const bool scale_e8m0;

 public:
  per_token_group_quant_mxfp4_tile_kernel(
      uint8_t* out_,
      void* scale_,
      scalar_t const* input_,
      sycl::local_accessor<float, 1> tile_,
      const int groups_per_row_,
      const int groups_per_wg_,
      float eps_,
      const int num_rows_,
      const int rows_per_wg_,
      const int64_t scale_stride_token_,
      const int64_t scale_stride_group_,
      const bool col_major_,
      const bool scale_e8m0_)
      : out(out_),
        scale(scale_),
        input(input_),
        tile(tile_),
        groups_per_row(groups_per_row_),
        groups_per_wg(groups_per_wg_),
        eps(eps_),
        num_rows(num_rows_),
        rows_per_wg(rows_per_wg_),
        scale_stride_token(scale_stride_token_),
        scale_stride_group(scale_stride_group_),
        col_major(col_major_),
        scale_e8m0(scale_e8m0_) {}

  void operator()
      [[sycl::reqd_sub_group_size(32)]] (sycl::nd_item<2> item) const {
    const int tid = static_cast<int>(item.get_local_id(1));
    const int local_range = static_cast<int>(item.get_local_range(1));
    const int row_base = static_cast<int>(item.get_group(0)) * rows_per_wg;
    const int k_base = static_cast<int>(item.get_group(1)) * groups_per_wg;
    const int rows = sycl::min(rows_per_wg, num_rows - row_base);
    const int tile_groups = rows * groups_per_wg;

    const int row_chunks = groups_per_row * 4;
    const int64_t first_chunk =
        static_cast<int64_t>(row_base) * row_chunks + k_base * 4;
    const auto* vin =
        reinterpret_cast<const vec_n_t<scalar_t, 8>*>(input) + first_chunk;
    auto* vout = reinterpret_cast<uint32_t*>(out) + first_chunk;

    for (int base = 0; base < tile_groups; base += local_range) {
      const int gi = base + tid;
      const bool active = gi < tile_groups;
      const int r = gi / groups_per_wg;
      const int off = r * row_chunks + (gi - r * groups_per_wg) * 4;
      vec_n_t<uint32_t, 4> o;
      const float y_s =
          mxfp4_quant_group32<scalar_t>(vin + off, active, eps, o);
      if (active) {
        *reinterpret_cast<vec_n_t<uint32_t, 4>*>(vout + off) = o;
        tile[gi] = y_s;
      }
    }

    sycl::group_barrier(item.get_group());
    for (int idx = tid; idx < tile_groups; idx += local_range) {
      int r, k;
      if (col_major) {
        k = idx / rows;
        r = idx - k * rows;
      } else {
        r = idx / groups_per_wg;
        k = idx - r * groups_per_wg;
      }
      const int64_t dst =
          static_cast<int64_t>(row_base + r) * scale_stride_token +
          static_cast<int64_t>(k_base + k) * scale_stride_group;
      const float y_s = tile[r * groups_per_wg + k];
      if (scale_e8m0) {
        static_cast<uint8_t*>(scale)[dst] = MoEScaleLayout::encode(y_s);
      } else {
        static_cast<float*>(scale)[dst] = y_s;
      }
    }
  }
};

// MoE variant with SLM-staged scale stores.
//
// The MoE scale surface is group-major (scale(row, k) at
// `dst[e]*scale_k + k*ld + (row - src[e])`), so one row per work-group writes
// single e8m0 bytes `ld` apart, i.e. one cache line per byte (774 vs 946 GB/s
// at [39040, 2048] on CRI). Here a work-group owns ROWS rows x K groups and
// stages its bytes in SLM, then re-maps lanes so the row index varies fastest
// and each sub-group writes a contiguous run. Requires CPT*VEC == group_size.
template <typename scalar_t, int CPT>
class per_token_group_quant_mxfp4_moe_slm_kernel {
 private:
  uint8_t* __restrict__ out;
  scalar_t const* __restrict__ input;
  sycl::local_accessor<uint8_t, 1> tile;
  const int hidden_size;
  float eps;
  const int num_rows;
  const int rows_per_wg;
  MoEScaleLayout moe_layout;

 public:
  per_token_group_quant_mxfp4_moe_slm_kernel(
      uint8_t* out_,
      scalar_t const* input_,
      sycl::local_accessor<uint8_t, 1> tile_,
      const int hidden_size_,
      float eps_,
      const int num_rows_,
      const int rows_per_wg_,
      MoEScaleLayout moe_layout_)
      : out(out_),
        input(input_),
        tile(tile_),
        hidden_size(hidden_size_),
        eps(eps_),
        num_rows(num_rows_),
        rows_per_wg(rows_per_wg_),
        moe_layout(moe_layout_) {}

  void operator()
      [[sycl::reqd_sub_group_size(32)]] (sycl::nd_item<1> item) const {
    constexpr int VEC = 16 / sizeof(scalar_t);
    constexpr int OUT_VEC = CPT * VEC / 2;

    const int tid = static_cast<int>(item.get_local_id(0));
    const int groups_per_row = hidden_size / (CPT * VEC);
    // The work-group is rounded up to whole sub-groups, so tail lanes own no
    // group and sit out both phases.
    const int tile_threads = rows_per_wg * groups_per_row;
    const int row_local = tid / groups_per_row;
    const int k = tid - row_local * groups_per_row;
    const int row_base = static_cast<int>(item.get_group(0)) * rows_per_wg;
    const int row_idx = row_base + row_local;

    if (tid < tile_threads && row_idx < num_rows) {
      const auto* vin = reinterpret_cast<const vec_n_t<scalar_t, VEC>*>(
          input + static_cast<int64_t>(row_idx) * hidden_size);
      auto* vout = reinterpret_cast<vec_n_t<uint8_t, OUT_VEC>*>(
          out + static_cast<int64_t>(row_idx) * (hidden_size / 2));

      vec_n_t<uint8_t, OUT_VEC> o;
      const float y_s =
          mxfp4_quant_one_group<scalar_t, VEC, CPT>(vin, k * CPT, eps, o);
      vout[k] = o;
      tile[row_local * groups_per_row + k] = MoEScaleLayout::encode(y_s);
    }

    sycl::group_barrier(item.get_group());

    // Re-map so consecutive lanes hold consecutive rows of one group index.
    const int r2 = tid % rows_per_wg;
    const int k2 = tid / rows_per_wg;
    const int row_out = row_base + r2;
    if (tid < tile_threads && row_out < num_rows) {
      moe_layout.store_byte(row_out, k2, tile[r2 * groups_per_row + k2]);
    }
  }
};

}  // namespace mxfp4
}  // namespace vllm
