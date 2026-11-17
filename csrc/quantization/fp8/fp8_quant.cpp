#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/xpu/XPUContext.h>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <limits>
#include <type_traits>

#include "dispatch_utils.h"
#include "ops.h"
#include "utils.h"

#include "quantization/fp8/fp8_quant.h"
#include "quantization/fp8/quant_utils.h"
#include "quantization/moe_scale_layout.h"
#include "quantization/mx_quant_utils.h"
#include "quantization/utils.h"

namespace vllm {

// STRIDE_I_ZERO: true if scale_stride_i == 0 (per-tensor or per-channel)
// STRIDE_J_ZERO: true if scale_stride_j == 0 (per-tensor or per-token)
template <
    typename scalar_t,
    typename fp8_type,
    bool STRIDE_I_ZERO,
    bool STRIDE_J_ZERO>
class scaled_fp8_quant_kernel_strided_group_shape {
 private:
  fp8_type* out;
  const scalar_t* input;
  const float* scale;
  int hidden_size;
  int64_t in_row_stride;
  int64_t out_row_stride;
  int group_m;
  int group_n;
  int64_t scale_stride_i;
  int64_t scale_stride_j;

 public:
  scaled_fp8_quant_kernel_strided_group_shape(
      fp8_type* out_,
      const scalar_t* input_,
      const float* scale_,
      int hidden_size,
      int64_t in_row_stride_,
      int64_t out_row_stride_,
      int group_m_,
      int group_n_,
      int64_t scale_stride_i_,
      int64_t scale_stride_j_)
      : out(out_),
        input(input_),
        scale(scale_),
        hidden_size(hidden_size),
        in_row_stride(in_row_stride_),
        out_row_stride(out_row_stride_),
        group_m(group_m_),
        group_n(group_n_),
        scale_stride_i(scale_stride_i_),
        scale_stride_j(scale_stride_j_) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t token_idx = item.get_group(0);
    const int tid = item.get_local_id(0);

    const scalar_t* token_in = input + token_idx * in_row_stride;
    fp8_type* token_out = out + token_idx * out_row_stride;

    // Precompute row-level base offset for scale access (compile-time
    // eliminated
    // when STRIDE_I_ZERO)
    const int64_t scale_row_base =
        STRIDE_I_ZERO ? 0
                      : static_cast<int>(token_idx) / group_m * scale_stride_i;
    auto get_inv_scale = [&](int gj) {
      return 1.0f / scale[scale_row_base + gj * scale_stride_j];
    };

    int cached_gj = -1;
    float cached_inv_scale = 0.0f;
    auto get_inv_scale_cached = [&](int gj) {
      if (gj != cached_gj) {
        cached_inv_scale = 1.0f / scale[scale_row_base + gj * scale_stride_j];
        cached_gj = gj;
      }
      return cached_inv_scale;
    };

    constexpr int VEC_SIZE = 4;
    if (STRIDE_J_ZERO && hidden_size % VEC_SIZE == 0) {
      // Per-tensor or per-token: single scale per row, vectorize full row
      fp8::ConvertWithScaleOp<true, fp8_type> op{get_inv_scale(0)};
      vectorize_with_alignment<VEC_SIZE>(
          token_in, token_out, hidden_size, tid, item.get_local_range(0), op);
    } else if (group_n % VEC_SIZE == 0) {
      // Multiple column groups with vectorization
      const int num_groups_n = hidden_size / group_n;

      for (int gj = 0; gj < num_groups_n; gj++) {
        fp8::ConvertWithScaleOp<true, fp8_type> op{get_inv_scale(gj)};
        vectorize_with_alignment<VEC_SIZE>(
            token_in + gj * group_n,
            token_out + gj * group_n,
            group_n,
            tid,
            item.get_local_range(0),
            op);
      }
    } else {
      // Scalar path for small column groups (group_n < VEC_SIZE)
      for (int n = tid; n < hidden_size; n += item.get_local_range(0)) {
        const int gj = n / group_n;
        fp8::ConvertWithScaleOp<true, fp8_type> op{get_inv_scale_cached(gj)};
        op(token_out[n], token_in[n]);
      }
    }
  }
};

// SI is the scale row stride: 0 = zero (one scale row for all tokens),
// 1 = non-zero, -1 = uniform runtime switch. The one-row kernels (no
// ELEM_SCALE) fix it at compile time, which is up to 6% faster on CRI; the
// ELEM_SCALE kernels keep the runtime switch to save binary size.
template <
    typename scalar_t,
    typename fp8_type,
    bool ELEM_SCALE,
    int VEC,
    int CPT,
    int SI = -1>
class scaled_fp8_quant_kernel_strided_group_shape_vec {
 private:
  fp8_type* out;
  const scalar_t* input;
  const float* scale;
  int hidden_size;
  int64_t in_row_stride;
  int64_t out_row_stride;
  int group_m;
  int group_n;
  int chunks_per_group;
  int64_t num_tokens;
  int64_t scale_stride_i;
  int64_t scale_stride_j;
  int col_tiles;

 public:
  scaled_fp8_quant_kernel_strided_group_shape_vec(
      fp8_type* out_,
      const scalar_t* input_,
      const float* scale_,
      int hidden_size_,
      int64_t in_row_stride_,
      int64_t out_row_stride_,
      int group_m_,
      int group_n_,
      int chunks_per_group_,
      int64_t num_tokens_,
      int64_t scale_stride_i_,
      int64_t scale_stride_j_,
      int col_tiles_)
      : out(out_),
        input(input_),
        scale(scale_),
        hidden_size(hidden_size_),
        in_row_stride(in_row_stride_),
        out_row_stride(out_row_stride_),
        group_m(group_m_),
        group_n(group_n_),
        chunks_per_group(chunks_per_group_),
        num_tokens(num_tokens_),
        scale_stride_i(scale_stride_i_),
        scale_stride_j(scale_stride_j_),
        col_tiles(col_tiles_) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t gid = item.get_group(0);
    const int64_t row_blocks = item.get_group_range(0) / col_tiles;
    const int ct = ELEM_SCALE ? static_cast<int>(gid % col_tiles) : 0;
    const int64_t rb = ELEM_SCALE ? gid / col_tiles : gid;
    const int local_range = static_cast<int>(item.get_local_range(0));
    const int thread_begin =
        ct * local_range + static_cast<int>(item.get_local_id(0));

    const int num_chunks = hidden_size / VEC;
    const int threads_per_row = num_chunks / CPT;
    for (int t = thread_begin; t < threads_per_row;
         t += ELEM_SCALE ? threads_per_row : local_range) {
      const int c0 = t * CPT;

      constexpr int NUM_SCALES = ELEM_SCALE ? VEC * CPT : 1;
      const int64_t col_off =
          ELEM_SCALE
              ? 0
              : static_cast<int64_t>(c0 / chunks_per_group) * scale_stride_j;
      float inv_scale[NUM_SCALES];
      auto load_inv_scale = [&](int64_t scale_row_base) {
        if constexpr (ELEM_SCALE) {
          // The VEC * CPT columns are contiguous, so step the group index
          // instead of dividing every column by group_n.
          const int col0 = c0 * VEC;
          int g = col0 / group_n;
          int rem = col0 - g * group_n;
#pragma unroll
          for (int i = 0; i < NUM_SCALES; ++i) {
            inv_scale[i] =
                1.0f /
                scale
                    [scale_row_base + static_cast<int64_t>(g) * scale_stride_j];
            if (++rem == group_n) {
              rem = 0;
              ++g;
            }
          }
        } else {
          inv_scale[0] = 1.0f / scale[scale_row_base + col_off];
        }
      };
      // Only the multi-row ELEM_SCALE loop hoists a row-invariant scale; the
      // one-row path loads it next to the input, without a branch in between,
      // so the two loads overlap (a zero stride yields scale row 0).
      const bool hoist_scale = ELEM_SCALE && scale_stride_i == 0;
      if (hoist_scale) {
        load_inv_scale(0);
      }

      const auto* vin = reinterpret_cast<const vec_n_t<scalar_t, VEC>*>(input);
      auto* vout = reinterpret_cast<vec_n_t<fp8_type, VEC>*>(out);
      const int64_t in_vstride = in_row_stride / VEC;
      const int64_t out_vstride = out_row_stride / VEC;

      const int64_t row_end = ELEM_SCALE ? num_tokens : rb + 1;
      const int64_t row_step = ELEM_SCALE ? row_blocks : 1;
      for (int64_t row = rb; row < row_end; row += row_step) {
        const auto* row_in = vin + row * in_vstride;
        auto* row_out = vout + row * out_vstride;

        if (!hoist_scale) {
          // Skipping the integer division matters on the latency-bound
          // per-tensor launches.
          int64_t scale_row_base = 0;
          if constexpr (SI == 1) {
            scale_row_base =
                static_cast<int64_t>(static_cast<int>(row) / group_m) *
                scale_stride_i;
          } else if (SI == -1 && scale_stride_i != 0) {
            scale_row_base =
                static_cast<int64_t>(static_cast<int>(row) / group_m) *
                scale_stride_i;
          }
          load_inv_scale(scale_row_base);
        }

        vec_n_t<scalar_t, VEC> xv[CPT];
#pragma unroll
        for (int j = 0; j < CPT; ++j) {
          xv[j] = row_in[c0 + j];
        }

#pragma unroll
        for (int j = 0; j < CPT; ++j) {
          constexpr bool kPackStore = VEC % 4 == 0;
          vec_n_t<fp8_type, VEC> o;
          PackedBytes<kPackStore ? VEC : 4> packed;
#pragma unroll
          for (int k = 0; k < VEC; ++k) {
            const float inv = inv_scale[ELEM_SCALE ? j * VEC + k : 0];
            fp8::ConvertWithScaleOp<true, fp8_type> op{inv};
            op(o.val[k], xv[j].val[k]);
            if constexpr (kPackStore) {
              packed.set(k, o.val[k].x);
            }
          }
          if constexpr (kPackStore) {
            packed.store(row_out + c0 + j);
          } else {
            row_out[c0 + j] = o;
          }
        }
      }
    }
  }
};

template <typename scalar_t, typename fp8_type>
class scaled_fp8_quant_kernel_strided_dynamic {
 private:
  fp8_type* out;
  const scalar_t* input;
  const float* scale;
  int64_t hidden_size;
  int64_t in_row_stride;
  int64_t out_row_stride;

 public:
  scaled_fp8_quant_kernel_strided_dynamic(
      fp8_type* out_,
      const scalar_t* input_,
      const float* scale_,
      int hidden_size_,
      int64_t in_row_stride_,
      int64_t out_row_stride_)
      : out(out_),
        input(input_),
        scale(scale_),
        hidden_size(hidden_size_),
        in_row_stride(in_row_stride_),
        out_row_stride(out_row_stride_) {}
  void operator()(sycl::nd_item<1> item) const {
    int64_t token_idx = item.get_group(0);
    int tid = item.get_local_id(0);

    const scalar_t* token_in = input + token_idx * in_row_stride;
    fp8_type* token_out = out + token_idx * out_row_stride;
    // Invert the scale so that we can use multiplications to avoid expensive
    // division.
    const float inverted_scale = 1.0f / (*scale);
    fp8::ConvertWithScaleOp<true, fp8_type> op{inverted_scale};
    constexpr int VEC_SIZE = 4;
    vectorize_with_alignment<VEC_SIZE>(
        token_in, token_out, hidden_size, tid, item.get_local_range(0), op);
  }
};

template <typename scalar_t, typename fp8_type>
class per_token_group_quant_8bit_kernel {
 private:
  fp8_type* out;
  float* scale;
  scalar_t const* input;
  const int group_size;
  const int groups_per_block;
  float eps;
  bool scale_ue8m0;
  const int scale_num_rows;
  const int scale_stride;
  bool is_column_major;
  MoEScaleLayout moe_layout;

 public:
  per_token_group_quant_8bit_kernel(
      fp8_type* out_,
      float* scale_,
      scalar_t const* input_,
      const int group_size_,
      const int groups_per_block_,
      float eps_,
      bool scale_ue8m0_,
      const int scale_num_rows_ = 0,
      const int scale_stride_ = 0,
      bool is_column_major_ = false,
      MoEScaleLayout moe_layout_ = {})
      : out(out_),
        scale(scale_),
        input(input_),
        group_size(group_size_),
        groups_per_block(groups_per_block_),
        eps(eps_),
        scale_ue8m0(scale_ue8m0_),
        scale_num_rows(scale_num_rows_),
        scale_stride(scale_stride_),
        is_column_major(is_column_major_),
        moe_layout(moe_layout_) {}
  void operator()
      [[sycl::reqd_sub_group_size(32)]] (sycl::nd_item<1> item) const {
    constexpr int threads_per_group = 32;

    int local_id = item.get_local_id(0);
    int local_group_id = local_id / threads_per_group;
    int lane_id = local_id % threads_per_group;

    // Global 32-thread group
    int block_group_id = item.get_group(0) * groups_per_block;
    int global_group_id = block_group_id + local_group_id;

    // Base offset
    int64_t block_group_offset =
        static_cast<int64_t>(global_group_id) * group_size;

    float local_absmax = eps;
    float* scale_output = nullptr;

    scalar_t const* group_input = &input[block_group_offset];
    fp8_type* group_output = &out[block_group_offset];
    if (moe_layout.enabled()) {
      // scale_output stays null: the MoE surface is addressed by descriptor.
    } else if (is_column_major) {
      const int row_idx = global_group_id / scale_num_rows;
      const int col_idx = global_group_id % scale_num_rows;
      scale_output =
          reinterpret_cast<float*>(scale) + (col_idx * scale_stride + row_idx);
    } else {
      scale_output = &scale[global_group_id];
    }

    bool const can_vectorize = group_size % 4 == 0;

    if (can_vectorize) {
      local_absmax =
          thread_max_vec(group_input, group_size, lane_id, threads_per_group);
    } else {
      for (int i = lane_id; i < group_size; i += threads_per_group) {
        float const x = static_cast<float>(group_input[i]);
        local_absmax = sycl::max(local_absmax, sycl::fabs(x));
      }
    }

    local_absmax = sycl::reduce_over_group(
        item.get_sub_group(), local_absmax, sycl::maximum<float>());

    float y_s = sycl::max(
        local_absmax / fp8::quant_type_max_v<fp8_type>,
        fp8::min_scaling_factor<fp8_type>::val());

    if (scale_ue8m0) {
      y_s = sycl::exp2(
          sycl::ceil(sycl::log2(sycl::fmax(sycl::fabs(y_s), 1e-10f))));
    }

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
    group_barrier(item.get_group());

    const float inverted_scale = 1.0f / (y_s);
    constexpr int VEC_SIZE = 4;
    if (can_vectorize) {
      fp8::ConvertWithScaleOp<true, fp8_type> op{inverted_scale};
      vectorize_with_alignment<VEC_SIZE>(
          group_input,
          group_output,
          group_size,
          lane_id,
          threads_per_group,
          op);
    } else {
      for (int i = lane_id; i < group_size; i += threads_per_group) {
        fp8::ConvertWithScaleOp<true, fp8_type> op{inverted_scale};
        op(group_output[i], group_input[i]);
      }
    }
  }
};

// Vectorized per-token group quant: one work-group per row, each lane reads VEC
// contiguous elements via 16-byte loads (coalesced), and the per-group absmax
// is a segmented reduction over the lanes_per_group consecutive lanes that
// cover a group. This maximizes global-memory bandwidth versus the
// 1-element-per-thread path. Used for the common case where group_size is a
// multiple of VEC.
template <typename scalar_t, typename fp8_type, int CPT>
class per_token_group_quant_8bit_vec_kernel {
 private:
  fp8_type* out;
  float* scale;
  scalar_t const* input;
  const int group_size;
  const int hidden;
  const int num_groups_per_row;
  float eps;
  bool scale_ue8m0;
  const int scale_num_rows;
  const int scale_stride;
  bool is_column_major;
  MoEScaleLayout moe_layout;

 public:
  per_token_group_quant_8bit_vec_kernel(
      fp8_type* out_,
      float* scale_,
      scalar_t const* input_,
      const int group_size_,
      const int hidden_,
      const int num_groups_per_row_,
      float eps_,
      bool scale_ue8m0_,
      const int scale_num_rows_,
      const int scale_stride_,
      bool is_column_major_,
      MoEScaleLayout moe_layout_ = {})
      : out(out_),
        scale(scale_),
        input(input_),
        group_size(group_size_),
        hidden(hidden_),
        num_groups_per_row(num_groups_per_row_),
        eps(eps_),
        scale_ue8m0(scale_ue8m0_),
        scale_num_rows(scale_num_rows_),
        scale_stride(scale_stride_),
        is_column_major(is_column_major_),
        moe_layout(moe_layout_) {}

  void operator()
      [[sycl::reqd_sub_group_size(32)]] (sycl::nd_item<1> item) const {
    const int row = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));
    const int local_range = static_cast<int>(item.get_local_range(0));
    auto sg = item.get_sub_group();
    const int lane = sg.get_local_id()[0];

    constexpr int VEC = 16 / sizeof(scalar_t);
    const int chunks_per_group = group_size / VEC;
    const int lanes_per_group = chunks_per_group / CPT;
    const int num_chunks = hidden / VEC;
    const int threads_per_row = num_chunks / CPT;

    scalar_t const* row_in = input + static_cast<int64_t>(row) * hidden;
    fp8_type* row_out = out + static_cast<int64_t>(row) * hidden;
    const auto* vin = reinterpret_cast<const vec_n_t<scalar_t, VEC>*>(row_in);
    auto* vout = reinterpret_cast<vec_n_t<fp8_type, VEC>*>(row_out);

    for (int base = 0; base < threads_per_row; base += local_range) {
      const int t = base + tid;
      const bool active = t < threads_per_row;
      const int c0 = t * CPT;

      vec_n_t<scalar_t, VEC> xv[CPT];
      float lane_absmax = active ? eps : 0.0f;
      if (active) {
#pragma unroll
        for (int j = 0; j < CPT; ++j) {
          xv[j] = vin[c0 + j];
        }
#pragma unroll
        for (int j = 0; j < CPT; ++j) {
#pragma unroll
          for (int k = 0; k < VEC; ++k) {
            lane_absmax = sycl::max(
                lane_absmax, sycl::fabs(static_cast<float>(xv[j].val[k])));
          }
        }
      }

      float gmax = lane_absmax;
#pragma unroll
      for (int off = 1; off < lanes_per_group; off <<= 1) {
        gmax = sycl::max(gmax, sycl::permute_group_by_xor(sg, gmax, off));
      }

      float y_s = sycl::max(
          gmax / fp8::quant_type_max_v<fp8_type>,
          fp8::min_scaling_factor<fp8_type>::val());
      if (scale_ue8m0) {
        y_s = sycl::exp2(
            sycl::ceil(sycl::log2(sycl::fmax(sycl::fabs(y_s), 1e-10f))));
      }

      if (active && (lanes_per_group <= 1 || (lane % lanes_per_group) == 0)) {
        const int g_idx = c0 / chunks_per_group;
        if (moe_layout.enabled()) {
          moe_layout.store(row, g_idx, y_s);
        } else {
          const int global_group = row * num_groups_per_row + g_idx;
          float* scale_output;
          if (is_column_major) {
            const int row_idx = global_group / scale_num_rows;
            const int col_idx = global_group % scale_num_rows;
            scale_output = scale + (col_idx * scale_stride + row_idx);
          } else {
            scale_output = scale + global_group;
          }
          *scale_output = y_s;
        }
      }

      if (active) {
        const float inv = 1.0f / y_s;
        fp8::ConvertWithScaleOp<true, fp8_type> op{inv};
#pragma unroll
        for (int j = 0; j < CPT; ++j) {
          vec_n_t<fp8_type, VEC> o;
#pragma unroll
          for (int k = 0; k < VEC; ++k) {
            op(o.val[k], static_cast<float>(xv[j].val[k]));
          }
          vout[c0 + j] = o;
        }
      }
    }
  }
};

// MXFP8 of 16-bit input, one lane per 16-byte vector, so a group consists of 4
// lanes: vector v is in group v / 4. kMoE selects the routed MoE scale
// surface; row- vs column-major is a uniform runtime switch (binary size).
template <typename scalar_t, typename fp8_type, typename scale_t, bool kMoE>
class per_token_group_quant_mxfp8_kernel {
 private:
  fp8_type* __restrict__ out;
  scale_t* __restrict__ scale;
  scalar_t const* __restrict__ input;
  const int num_vecs;
  float eps;
  const int scale_num_rows;
  const int scale_stride;
  MoEScaleLayout moe_layout;
  mx::ScaleLayout layout;

 public:
  per_token_group_quant_mxfp8_kernel(
      fp8_type* out_,
      scale_t* scale_,
      scalar_t const* input_,
      const int num_vecs_,
      float eps_,
      const int scale_num_rows_,
      const int scale_stride_,
      MoEScaleLayout moe_layout_,
      mx::ScaleLayout layout_)
      : out(out_),
        scale(scale_),
        input(input_),
        num_vecs(num_vecs_),
        eps(eps_),
        scale_num_rows(scale_num_rows_),
        scale_stride(scale_stride_),
        moe_layout(moe_layout_),
        layout(layout_) {}

  void operator()
      [[sycl::reqd_sub_group_size(32),
        sycl::reqd_work_group_size(mx::kWorkGroupSize)]] (
          sycl::nd_item<1> item) const {
    constexpr int VEC = mx::kVec;
    const int v = mx::flat_index(item);
    const bool active = v < num_vecs;

    float x[VEC];
    float absmax = eps;
    if (active) {
      const auto xv = reinterpret_cast<const vec_n_t<scalar_t, VEC>*>(input)[v];
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        x[k] = static_cast<float>(xv.val[k]);
        absmax = sycl::fmax(absmax, sycl::fabs(x[k]));
      }
    }
    auto sg = item.get_sub_group();
    absmax = mx::group_absmax(sg, absmax, mx::kLanesPerGroup);
    float inv_scale;
    const float y_s = mx::mxfp8_scale<fp8_type>(absmax, inv_scale);

    const bool packed_e8m0 = !kMoE && !std::is_same_v<scale_t, float> &&
                             layout == mx::ScaleLayout::kRowMajor;
    if (packed_e8m0) {
      using scale8_t = vec_n_t<uint8_t, 8>;
      const uint32_t e = MoEScaleLayout::encode(y_s);
      scale8_t packed;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        packed.val[k] = static_cast<uint8_t>(
            sycl::select_from_group(sg, e, k * mx::kLanesPerGroup));
      }
      if (active && sg.get_local_linear_id() == 0) {
        const int g = v / mx::kLanesPerGroup;
        const int num_g = num_vecs / mx::kLanesPerGroup;
        if (g + 8 <= num_g && reinterpret_cast<uintptr_t>(scale + g) % 8 == 0) {
          *reinterpret_cast<scale8_t*>(scale + g) = packed;
        } else {
          for (int k = 0; k < 8 && g + k < num_g; ++k) {
            scale[g + k] = packed.val[k];
          }
        }
      }
    }
    if (!active) return;

    if (!packed_e8m0 && v % mx::kLanesPerGroup == 0) {
      const int g = v / mx::kLanesPerGroup;
      scale_t s;
      if constexpr (std::is_same_v<scale_t, float>) {
        s = y_s;
      } else {
        s = MoEScaleLayout::encode(y_s);
      }
      if constexpr (kMoE) {
        moe_layout.store(g / moe_layout.scale_k, g % moe_layout.scale_k, y_s);
      } else if (layout == mx::ScaleLayout::kColMajor) {
        scale[(g % scale_num_rows) * scale_stride + g / scale_num_rows] = s;
      } else {
        scale[g] = s;
      }
    }
    reinterpret_cast<mx::fp8x8_t*>(out)[v] =
        mx::mxfp8_pack8<fp8_type>(x, inv_scale);
  }
};

template <typename scalar_t, typename fp8_type>
class dynamic_per_token_scaled_fp8_quant_kernel {
 private:
  fp8_type* out;
  float* scale;
  scalar_t const* input;
  float const* scale_ub;
  const int hidden_size;

 public:
  dynamic_per_token_scaled_fp8_quant_kernel(
      fp8_type* out_,
      float* scale_,
      scalar_t const* input_,
      float const* scale_ub_,
      const int hidden_size_)
      : out(out_),
        scale(scale_),
        input(input_),
        scale_ub(scale_ub_),
        hidden_size(hidden_size_) {}

  void operator()(sycl::nd_item<1> item) const {
    int const tid = item.get_local_id(0);
    int const token_idx = item.get_group(0);

    // Use int64 to avoid overflowing an int32 when calculating this offset
    int64_t offset = static_cast<int64_t>(token_idx) * hidden_size;
    scalar_t const* token_input = &input[offset];
    fp8_type* token_output = &out[offset];

    // For vectorization, token_input and token_output pointers need to be
    // aligned at 8-byte and 4-byte addresses respectively.
    bool const can_vectorize = hidden_size % 4 == 0;

    float absmax_val = 0.0f;
    if (can_vectorize) {
      absmax_val = thread_max_vec(
          token_input, hidden_size, tid, item.get_local_range(0));
    } else {
      for (int i = tid; i < hidden_size; i += item.get_local_range(0)) {
        float const x = static_cast<float>(token_input[i]);
        absmax_val = sycl::max(absmax_val, sycl::fabs(x));
      }
    }

    float const block_absmax_val_maybe = sycl::reduce_over_group(
        item.get_group(), absmax_val, sycl::maximum<float>());
    // __shared__ float token_scale;
    auto& token_scale =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1]>(
            item.get_group());
    if (tid == 0) {
      if (scale_ub) {
        token_scale[0] = sycl::min(block_absmax_val_maybe, *scale_ub);
      } else {
        token_scale[0] = block_absmax_val_maybe;
      }
      // token scale computation
      token_scale[0] = sycl::max(
          token_scale[0] / fp8::quant_type_max_v<fp8_type>,
          fp8::min_scaling_factor<fp8_type>::val());
      scale[token_idx] = token_scale[0];
    }
    group_barrier(item.get_group());

    // Note that we don't use inverted scales so we can match FBGemm impl.
    const float inverted_scale = 1.0f / (token_scale[0]);
    constexpr int VEC_SIZE = 4;
    if (can_vectorize) {
      fp8::ConvertWithScaleOp<true, fp8_type> op{inverted_scale};
      vectorize_with_alignment<VEC_SIZE>(
          token_input,
          token_output,
          hidden_size,
          tid,
          item.get_local_range(0),
          op);
    } else {
      for (int i = tid; i < hidden_size; i += item.get_local_range(0)) {
        fp8::ConvertWithScaleOp<true, fp8_type> op{inverted_scale};
        op(token_output[i], token_input[i]);
      }
    }
  }
};

}  // namespace vllm

void static_scaled_fp8_quant(
    torch::Tensor& out,          // [..., d]
    torch::Tensor const& input,  // [..., d]
    torch::Tensor const& scale,  // various shapes
    std::optional<std::tuple<int64_t, int64_t>>
        opt_group_shape)  // optional explicit (group_m, group_n)
{
  TORCH_CHECK(
      input.stride(-1) == 1, "last dimension of input must be contiguous");
  TORCH_CHECK(
      out.stride(-1) == 1, "last dimension of output must be contiguous");

  const int hidden_size = input.size(-1);                  // N (columns)
  const int64_t num_tokens = input.numel() / hidden_size;  // M (rows)

  // Determine group_m, group_n, and scale strides from scale shape
  // Scale indexing: scale[gi * scale_stride_j + gj * scale_stride_i]
  // where gi = m / group_m, gj = n / group_n
  int group_m, group_n;
  int64_t scale_stride_i, scale_stride_j;

  if (scale.dim() == 0 || scale.numel() == 1) {
    // Per-tensor: one scale for the entire tensor
    group_m = num_tokens;
    group_n = hidden_size;
    scale_stride_i = 0;
    scale_stride_j = 0;
  } else if (scale.dim() == 1) {
    // 1D scale: require explicit group_shape to disambiguate per-channel vs
    // per-token (avoids edge case where num_tokens == hidden_size)
    TORCH_CHECK(
        opt_group_shape.has_value(),
        "1D scale requires explicit group_shape to disambiguate "
        "per-channel vs per-token quantization. "
        "Use group_shape=(-1, 1) for per-channel or group_shape=(1, "
        "-1) for per-token.");

    const auto& [opt_group_m, opt_group_n] = opt_group_shape.value();
    group_m = opt_group_m == -1 ? num_tokens : static_cast<int>(opt_group_m);
    group_n = opt_group_n == -1 ? hidden_size : static_cast<int>(opt_group_n);

    // Validate the explicit group shape matches the 1D scale
    const int64_t scale_len = scale.numel();
    const int64_t expected_scale_m = num_tokens / group_m;
    const int64_t expected_scale_n = hidden_size / group_n;
    const int64_t expected_scale_numel = expected_scale_m * expected_scale_n;

    TORCH_CHECK(
        scale_len == expected_scale_numel,
        "1D scale length (",
        scale_len,
        ") does not match expected size (",
        expected_scale_numel,
        ") for group_shape (",
        opt_group_m,
        ", ",
        opt_group_n,
        ") with input shape (",
        num_tokens,
        ", ",
        hidden_size,
        ")");

    // For 1D scale, determine strides based on which dim is trivial
    // Scale indexing: scale[gi * scale_stride_i + gj * scale_stride_j]
    // where gi = m / group_m (row group), gj = n / group_n (col group)
    if (expected_scale_m == 1) {
      // Per-channel style: one scale in M dim, scale varies along N
      // gi = 0 always, gj varies, so stride_1 traverses the scale
      scale_stride_i = 0;
      scale_stride_j = scale.stride(0);
    } else if (expected_scale_n == 1) {
      // Per-token style: one scale in N dim, scale varies along M
      // gj = 0 always, gi varies, so stride_0 traverses the scale
      scale_stride_i = scale.stride(0);
      scale_stride_j = 0;
    } else {
      TORCH_CHECK(
          false,
          "1D scale can only be used when one of the scale dimensions is 1. "
          "For 2D group scaling, use a 2D scale tensor.");
    }
  } else if (scale.dim() == 2) {
    // 2D scale: infer group sizes from scale dimensions (or use explicit if
    // provided)
    const int64_t scale_size_0 = scale.size(0);
    const int64_t scale_size_1 = scale.size(1);

    TORCH_CHECK(
        num_tokens % scale_size_0 == 0,
        "num_tokens (",
        num_tokens,
        ") must be divisible by scale.size(0) (",
        scale_size_0,
        ")");
    TORCH_CHECK(
        hidden_size % scale_size_1 == 0,
        "hidden_size (",
        hidden_size,
        ") must be divisible by scale.size(1) (",
        scale_size_1,
        ")");

    // Infer from 2D scale shape
    int inferred_group_m = num_tokens / scale_size_0;
    int inferred_group_n = hidden_size / scale_size_1;

    // Use explicit if provided, otherwise use inferred
    if (opt_group_shape.has_value()) {
      const auto& [opt_group_m, opt_group_n] = opt_group_shape.value();
      group_m = opt_group_m == -1 ? num_tokens : static_cast<int>(opt_group_m);
      group_n = opt_group_n == -1 ? hidden_size : static_cast<int>(opt_group_n);

      // Validate explicit matches inferred
      TORCH_CHECK(
          group_m == inferred_group_m && group_n == inferred_group_n,
          "Explicit group_shape (",
          opt_group_m,
          ", ",
          opt_group_n,
          ") does not match inferred group shape (",
          inferred_group_m,
          ", ",
          inferred_group_n,
          ") from 2D scale tensor shape (",
          scale_size_0,
          ", ",
          scale_size_1,
          ")");
    } else {
      group_m = inferred_group_m;
      group_n = inferred_group_n;
    }

    scale_stride_i = scale.stride(0);
    scale_stride_j = scale.stride(1);
  } else {
    TORCH_CHECK(
        false,
        "scale must be 0D, 1D, or 2D tensor, but got ",
        scale.dim(),
        "D");
  }

  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(1024);

  const int64_t in_row_stride = input.stride(-2);
  const int64_t out_row_stride = out.stride(-2);

  const at::DeviceGuard device_guard(input.device());

  auto& queue = vllm::xpu::vllmGetQueue();
  const int eu_count = static_cast<int>(
      queue.get_device().get_info<sycl::info::device::max_compute_units>());

  constexpr int kWorkGroupMultiple = 32;
  constexpr int kMaxWorkGroup = 1024;
  constexpr int kWorkItemsPerEU = 128;
  constexpr int kMaxScalesPerItem = 16;
  constexpr int64_t kMaxRowsPerItem = 8;
  const int vec = 16 / input.element_size();
  // The vectorized kernel is only built for 16-bit input and e4m3 output;
  // the other combinations take the generic kernel to save binary size.
  bool vectorizable = input.element_size() == 2 &&
                      out.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                      reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
                      reinterpret_cast<uintptr_t>(out.data_ptr()) % vec == 0 &&
                      in_row_stride % vec == 0 && out_row_stride % vec == 0 &&
                      hidden_size % vec == 0;
  const bool elem_scale = scale_stride_j != 0 && group_n % vec != 0;
  const int num_chunks = hidden_size / vec;
  const int chunks_per_group =
      (scale_stride_j == 0 || elem_scale) ? num_chunks : group_n / vec;
  int chunks_per_item = 1;
  vllm::dispatch_quant_chunks(
      num_tokens, num_chunks, chunks_per_group, eu_count, [&](auto tag) {
        chunks_per_item = decltype(tag)::value;
      });
  // On BMG, one 16-byte chunk per work-item measured 10-30% faster than
  // several adjacent chunks per work-item for T >= 128; Xe3P prefers the
  // wider items.
  if (!elem_scale && vllm::xpu::is_bmg()) {
    chunks_per_item = 1;
  }
  if (elem_scale) {
    chunks_per_item = std::min(chunks_per_item, kMaxScalesPerItem / vec);
  }
  const int threads_per_row = num_chunks / chunks_per_item;
  const int work_group = std::clamp(
      (threads_per_row + kWorkGroupMultiple - 1) / kWorkGroupMultiple *
          kWorkGroupMultiple,
      kWorkGroupMultiple,
      elem_scale ? kWorkItemsPerEU : kMaxWorkGroup);
  const int col_tiles =
      elem_scale ? std::max(1, (threads_per_row + work_group - 1) / work_group)
                 : 1;
  const int64_t row_blocks =
      elem_scale
          ? std::min(
                num_tokens,
                std::max(
                    (num_tokens + kMaxRowsPerItem - 1) / kMaxRowsPerItem,
                    std::max<int64_t>(
                        1,
                        static_cast<int64_t>(eu_count) * kWorkItemsPerEU /
                            (static_cast<int64_t>(col_tiles) * work_group))))
          : num_tokens;
  vectorizable = vectorizable && (!elem_scale || row_blocks < num_tokens);
  const sycl::nd_range<1> range(
      sycl::range<1>(static_cast<size_t>(col_tiles) * row_blocks * work_group),
      sycl::range<1>(work_group));

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
              auto* out_ptr = out.data_ptr<fp8_t>();
              auto* in_ptr = input.data_ptr<scalar_t>();
              auto* scale_ptr = scale.data_ptr<float>();

              constexpr int VEC = 16 / sizeof(scalar_t);

              if (vectorizable) {
                auto launch = [&](auto elem_tag, auto cpt_tag) {
                  auto go = [&](auto si_tag) {
                    // vectorizable is always false for fp32 inputs and
                    // e5m2 output; skip the instantiation.
                    if constexpr (
                        sizeof(scalar_t) == 2 &&
                        std::is_same_v<fp8_t, c10::Float8_e4m3fn>)
                      queue.submit([&](sycl::handler& cgh) {
                        cgh.parallel_for(
                            range,
                            vllm::
                                scaled_fp8_quant_kernel_strided_group_shape_vec<
                                    scalar_t,
                                    fp8_t,
                                    decltype(elem_tag)::value,
                                    VEC,
                                    decltype(cpt_tag)::value,
                                    decltype(si_tag)::value>(
                                    out_ptr,
                                    in_ptr,
                                    scale_ptr,
                                    hidden_size,
                                    in_row_stride,
                                    out_row_stride,
                                    group_m,
                                    group_n,
                                    chunks_per_group,
                                    num_tokens,
                                    scale_stride_i,
                                    scale_stride_j,
                                    col_tiles));
                      });
                  };
                  if constexpr (decltype(elem_tag)::value) {
                    go(std::integral_constant<int, -1>{});
                  } else if (scale_stride_i == 0) {
                    go(std::integral_constant<int, 0>{});
                  } else {
                    go(std::integral_constant<int, 1>{});
                  }
                };

                auto dispatch = [&](auto elem_tag) {
                  // elem_scale caps chunks_per_item at kMaxScalesPerItem /
                  // VEC, so 4 is unreachable there for 2-byte inputs.
                  constexpr bool kHasCpt4 = !decltype(elem_tag)::value ||
                                            4 * VEC <= kMaxScalesPerItem;
                  if (kHasCpt4 && chunks_per_item == 4) {
                    if constexpr (kHasCpt4) {
                      launch(elem_tag, std::integral_constant<int, 4>{});
                    }
                  } else if (chunks_per_item == 2) {
                    launch(elem_tag, std::integral_constant<int, 2>{});
                  } else {
                    launch(elem_tag, std::integral_constant<int, 1>{});
                  }
                };
                if (elem_scale) {
                  dispatch(std::true_type{});
                } else {
                  dispatch(std::false_type{});
                }
                return;
              }

              VLLM_DISPATCH_BOOL(scale_stride_i == 0, S0_ZERO, [&] {
                VLLM_DISPATCH_BOOL(scale_stride_j == 0, S1_ZERO, [&] {
                  queue.submit([&](sycl::handler& cgh) {
                    auto kernel =
                        vllm::scaled_fp8_quant_kernel_strided_group_shape<
                            scalar_t,
                            fp8_t,
                            S0_ZERO,
                            S1_ZERO>(
                            out_ptr,
                            in_ptr,
                            scale_ptr,
                            hidden_size,
                            in_row_stride,
                            out_row_stride,
                            group_m,
                            group_n,
                            scale_stride_i,
                            scale_stride_j);
                    cgh.parallel_for(
                        sycl::nd_range<1>(grid * block, block), kernel);
                  });
                });
              });
            });
      });
}

void dynamic_scaled_fp8_quant(
    torch::Tensor& out,          // [..., d]
    torch::Tensor const& input,  // [..., d]
    torch::Tensor& scale)        // [1]
{
  TORCH_CHECK(
      scale.scalar_type() == at::kFloat, "scale must be a float tensor");
  TORCH_CHECK(scale.numel() == 1, "scale must be a single-element tensor");
  TORCH_CHECK(
      input.stride(-1) == 1, "last dimension of input must be contiguous");
  TORCH_CHECK(
      out.stride(-1) == 1, "last dimension of output must be contiguous");

  int64_t hidden_size = input.size(-1);
  int64_t num_tokens = input.numel() / hidden_size;
  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(1024);

  int64_t in_row_stride = input.stride(-2);
  int64_t out_row_stride = out.stride(-2);

  const at::DeviceGuard device_guard(input.device());

  auto& queue = vllm::xpu::vllmGetQueue();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
              queue.memset(scale.data_ptr<float>(), 0, sizeof(float));
              // Launch the kernel
              queue.submit([&](sycl::handler& cgh) {
                auto max_reduce_kernel =
                    vllm::segmented_max_reduction_strided<scalar_t, fp8_t>(
                        scale.data_ptr<float>(),
                        input.data_ptr<scalar_t>(),
                        hidden_size,
                        in_row_stride,
                        static_cast<int64_t>(num_tokens));
                cgh.parallel_for(
                    sycl::nd_range<1>(grid * block, block), max_reduce_kernel);
              });
              queue.submit([&](sycl::handler& cgh) {
                auto kernel = vllm::
                    scaled_fp8_quant_kernel_strided_dynamic<scalar_t, fp8_t>(
                        out.data_ptr<fp8_t>(),
                        input.data_ptr<scalar_t>(),
                        scale.data_ptr<float>(),
                        hidden_size,
                        in_row_stride,
                        out_row_stride);
                cgh.parallel_for(
                    sycl::nd_range<1>(grid * block, block), kernel);
              });
            });
      });
}

void per_token_group_quant_fp8(
    const torch::Tensor& input,
    torch::Tensor& output_q,
    torch::Tensor& output_s,
    int64_t group_size,
    double eps,
    double fp8_min,
    double fp8_max,
    bool scale_ue8m0,
    bool dummy_is_scale_transposed,
    bool dummy_is_tma_aligned,
    const c10::optional<torch::Tensor>& expert_scale_desc) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(output_q.is_contiguous());

  const int num_groups = input.numel() / group_size;

  TORCH_CHECK(input.numel() % group_size == 0);
  TORCH_CHECK(output_s.dim() == 2);

  if (num_groups == 0) {
    return;
  }

  const at::DeviceGuard device_guard(input.device());

  constexpr int THREADS_PER_GROUP = 32;

  int groups_per_block = 1;

  if (num_groups % 16 == 0) {
    groups_per_block = 16;
  } else if (num_groups % 8 == 0) {
    groups_per_block = 8;
  } else if (num_groups % 4 == 0) {
    groups_per_block = 4;
  } else if (num_groups % 2 == 0) {
    groups_per_block = 2;
  }

  auto dst_type = output_q.scalar_type();
  const int num_blocks = num_groups / groups_per_block;
  const int num_threads = groups_per_block * THREADS_PER_GROUP;

  const auto moe_layout = vllm::make_moe_scale_layout(
      output_s,
      expert_scale_desc,
      static_cast<int>(input.size(-1) / group_size),
      input.size(-1) > 0 ? input.numel() / input.size(-1) : 0);
  const bool is_column_major =
      !moe_layout.enabled() && output_s.stride(0) < output_s.stride(1);
  const int scale_num_rows = output_s.size(1);
  const int scale_stride = output_s.stride(1);
  const bool scale_is_e8m0 =
      !moe_layout.enabled() &&
      (output_s.scalar_type() == at::ScalarType::Float8_e8m0fnu ||
       output_s.scalar_type() == at::ScalarType::Byte);
  float* const scale_ptr = moe_layout.enabled() || scale_is_e8m0
                               ? nullptr
                               : output_s.data_ptr<float>();

  // Fast vectorized path: one work-group per row with 16-byte loads. Requires
  // group_size to be a multiple of the (dtype-dependent) vector width.
  // Derived from the input rather than output_s, which is a padded per-expert
  // surface (not [num_tokens, scale_k]) when a MoE descriptor is given.
  const int hidden = static_cast<int>(input.size(-1));
  const int num_tokens = static_cast<int>(input.numel() / hidden);
  const int elem_size = input.element_size();
  const int vec_width = 16 / elem_size;
  // The vectorized kernel reduces each group's absmax with a segmented
  // sub-group XOR shuffle over lanes_per_group = group_size / vec_width lanes.
  // The sub-group is pinned to 32 lanes (reqd_sub_group_size(32)), so once
  // lanes_per_group exceeds 32 the XOR partner falls outside the sub-group and
  // the reduction (and the per-group scale-write predicate) become undefined,
  // producing wrong scales and corrupted output. Restrict the fast path to
  // groups that fit inside a single sub-group and fall back to the scalar
  // kernel (which reduces over the whole sub-group) otherwise.
  const bool use_vec = (group_size % vec_width == 0) &&
                       (hidden % vec_width == 0) &&
                       (group_size / vec_width <= 32);

  const int64_t mx_num_vecs = input.numel() / vllm::mx::kVec;
  // Groups are numbered over the flat tensor, so hidden must be a multiple of
  // the group size (32).
  const bool use_mx = group_size == vllm::mx::kGroupSize && scale_ue8m0 &&
                      elem_size == 2 && hidden % vllm::mx::kGroupSize == 0 &&
                      mx_num_vecs <= std::numeric_limits<int>::max() &&
                      reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
                      reinterpret_cast<uintptr_t>(output_q.data_ptr()) % 8 == 0;
  TORCH_CHECK(
      !scale_is_e8m0 || use_mx,
      "e8m0 output_s requires group_size 32, scale_ue8m0, 16-bit input, a "
      "hidden size divisible by 32 and aligned tensors");
  if (use_mx) {
    if (mx_num_vecs == 0) return;
    auto& mqueue = vllm::xpu::vllmGetQueue();
    const int64_t wg = vllm::mx::kWorkGroupSize;
    sycl::range<1> mgrid((mx_num_vecs + wg - 1) / wg);
    sycl::range<1> mblock(wg);
    VLLM_DISPATCH_HALF_TYPES(
        input.scalar_type(), "per_token_group_quant_mxfp8_scale_type", [&] {
          VLLM_DISPATCH_FP8_TYPES(
              output_q.scalar_type(), "per_token_group_quant_mxfp8_type", [&] {
                auto launch = [&](auto out_s, auto layout_tag) {
                  constexpr auto layout = decltype(layout_tag)::value;
                  constexpr bool kMoE = layout == vllm::mx::ScaleLayout::kMoE;
                  mqueue.submit([&](sycl::handler& cgh) {
                    cgh.parallel_for(
                        sycl::nd_range<1>(mgrid * mblock, mblock),
                        vllm::per_token_group_quant_mxfp8_kernel<
                            scalar_t,
                            fp8_t,
                            std::remove_pointer_t<decltype(out_s)>,
                            kMoE>(
                            output_q.data_ptr<fp8_t>(),
                            out_s,
                            input.data_ptr<scalar_t>(),
                            static_cast<int>(mx_num_vecs),
                            eps,
                            scale_num_rows,
                            scale_stride,
                            moe_layout,
                            layout));
                  });
                };
                using L = vllm::mx::ScaleLayout;
                auto by_layout = [&](auto out_s) {
                  if (moe_layout.enabled()) {
                    // MoE scales go through moe_layout; out_s is unused.
                    if constexpr (std::is_same_v<decltype(out_s), float*>) {
                      launch(out_s, std::integral_constant<L, L::kMoE>{});
                    }
                  } else if (is_column_major) {
                    launch(out_s, std::integral_constant<L, L::kColMajor>{});
                  } else {
                    launch(out_s, std::integral_constant<L, L::kRowMajor>{});
                  }
                };
                if (scale_is_e8m0) {
                  by_layout(reinterpret_cast<uint8_t*>(output_s.data_ptr()));
                } else {
                  by_layout(scale_ptr);
                }
              });
        });
    return;
  }

  if (use_vec) {
    const int num_groups_per_row = hidden / group_size;
    const int chunks_per_group = group_size / vec_width;
    const int num_chunks = hidden / vec_width;
    auto& vqueue = vllm::xpu::vllmGetQueue();
    const int eu_count = static_cast<int>(
        vqueue.get_device().get_info<sycl::info::device::max_compute_units>());
    const long long saturate_wi = static_cast<long long>(eu_count) * 128;
    const long long base_wi = static_cast<long long>(num_tokens) * num_chunks;
    // Wider per-thread tiles are only specialised for 16-bit input and e4m3
    // output (binary size); the rest keep one chunk per thread.
    const bool wide = input.element_size() == 2 &&
                      output_q.scalar_type() == at::ScalarType::Float8_e4m3fn;
    int cpt = 1;
    for (int c : {4, 2}) {
      if (wide && chunks_per_group % c == 0 && base_wi / c >= saturate_wi) {
        cpt = c;
        break;
      }
    }
    const int threads_per_row = num_chunks / cpt;
    int wg = std::min<int>(((threads_per_row + 31) / 32) * 32, 1024);
    if (wg < 32) wg = 32;
    sycl::range<1> vgrid(num_tokens);
    sycl::range<1> vblock(wg);
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "per_token_group_quant_8bit_vec_scale_type", [&] {
          VLLM_DISPATCH_FP8_TYPES(
              output_q.scalar_type(),
              "per_token_group_quant_8bit_vec_fp8_type",
              [&] {
                auto launch = [&](auto cpt_tag) {
                  // cpt is always 1 without `wide`, so fold the wider tags
                  // onto the CPT = 1 instantiation.
                  constexpr int CPT =
                      sizeof(scalar_t) == 2 &&
                              std::is_same_v<fp8_t, c10::Float8_e4m3fn>
                          ? decltype(cpt_tag)::value
                          : 1;
                  vqueue.submit([&](sycl::handler& cgh) {
                    auto kernel = vllm::per_token_group_quant_8bit_vec_kernel<
                        scalar_t,
                        fp8_t,
                        CPT>(
                        output_q.data_ptr<fp8_t>(),
                        scale_ptr,
                        input.data_ptr<scalar_t>(),
                        group_size,
                        hidden,
                        num_groups_per_row,
                        eps,
                        scale_ue8m0,
                        scale_num_rows,
                        scale_stride,
                        is_column_major,
                        moe_layout);
                    cgh.parallel_for(
                        sycl::nd_range<1>(vgrid * vblock, vblock), kernel);
                  });
                };
                if (cpt == 4) {
                  launch(std::integral_constant<int, 4>{});
                } else if (cpt == 2) {
                  launch(std::integral_constant<int, 2>{});
                } else {
                  launch(std::integral_constant<int, 1>{});
                }
              });
        });
    return;
  }

  sycl::range<1> grid(num_blocks);
  sycl::range<1> block(num_threads);
  auto& queue = vllm::xpu::vllmGetQueue();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "per_token_group_quant_8bit_kernel_scale_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            output_q.scalar_type(),
            "per_token_group_quant_8bit_kernel_fp8_type",
            [&] {
              // Launch the kernel
              queue.submit([&](sycl::handler& cgh) {
                auto kernel =
                    vllm::per_token_group_quant_8bit_kernel<scalar_t, fp8_t>(
                        output_q.data_ptr<fp8_t>(),
                        scale_ptr,
                        input.data_ptr<scalar_t>(),
                        group_size,
                        groups_per_block,
                        eps,
                        scale_ue8m0,
                        scale_num_rows,
                        scale_stride,
                        is_column_major,
                        moe_layout);
                cgh.parallel_for(
                    sycl::nd_range<1>(grid * block, block), kernel);
              });
            });
      });
}

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out,          // [..., d]
    torch::Tensor const& input,  // [..., d]
    torch::Tensor& scales,
    std::optional<at::Tensor> const& scale_ub) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(std::min(hidden_size, 1024));

  const at::DeviceGuard device_guard(input.device());

  auto& queue = vllm::xpu::vllmGetQueue();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "dynamic_per_token_scaled_fp8_quant_kernel_scalar_type",
      [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(),
            "dynamic_per_token_scaled_fp8_quant_kernel_fp8_type",
            [&] {
              // Launch the kernel
              queue.submit([&](sycl::handler& cgh) {
                auto kernel = vllm::
                    dynamic_per_token_scaled_fp8_quant_kernel<scalar_t, fp8_t>(
                        out.data_ptr<fp8_t>(),
                        scales.data_ptr<float>(),
                        input.data_ptr<scalar_t>(),
                        scale_ub.has_value() ? scale_ub->data_ptr<float>()
                                             : nullptr,
                        hidden_size);
                cgh.parallel_for(
                    sycl::nd_range<1>(grid * block, block), kernel);
              });
            });
      });
}
