#include <sycl/sycl.hpp>

#include <algorithm>
#include <optional>
#include <ATen/DeviceGuard.h>
#include "utils.h"
#include "dispatch_utils.h"
#include "quantization/utils.h"

namespace vllm {

template <typename scalar_t, int NUM_DIMS, int VEC_SIZE, bool HasWeight>
class rms_norm_kernel {
 public:
  rms_norm_kernel(
      scalar_t* out_,
      const scalar_t* input_,
      const int64_t input_stride_d2_,  // input.stride(-2)
      const int64_t input_stride_d3_,  // input.stride(-3)
      const int64_t input_stride_d4_,  // input.stride(-4)
      const int64_t input_shape_d2_,   // input.size(-2)
      const int64_t input_shape_d3_,   // input.size(-3)
      const scalar_t* weight_,
      const float epsilon_,
      const int num_tokens_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_)
      : out(out_),
        input(input_),
        input_stride_d2(input_stride_d2_),
        input_stride_d3(input_stride_d3_),
        input_stride_d4(input_stride_d4_),
        input_shape_d2(input_shape_d2_),
        input_shape_d3(input_shape_d3_),
        weight(weight_),
        epsilon(epsilon_),
        num_tokens(num_tokens_),
        hidden_size(hidden_size_),
        s_variance(s_variance_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();
    float variance = 0.0f;

    const scalar_t* input_row;
    if constexpr (NUM_DIMS == 2) {
      // 2D for layernorm normal case [batch_size, hidden]
      input_row = input + item_ct1.get_group(2) * input_stride_d2;
    } else if constexpr (NUM_DIMS == 3) {
      // 3D for q/k norm [batch_size, num_heads, head_size]
      int batch_idx = item_ct1.get_group(2) / input_shape_d2;
      int head_idx = item_ct1.get_group(2) % input_shape_d2;
      input_row =
          input + batch_idx * input_stride_d3 + head_idx * input_stride_d2;
    } else if constexpr (NUM_DIMS == 4) {
      // 4D for transformers model_impl qk norm [batch, seq, head, head_dim]
      int batch_idx = item_ct1.get_group(2) / (input_shape_d3 * input_shape_d2);
      int remaining = item_ct1.get_group(2) % (input_shape_d3 * input_shape_d2);
      int seq_idx = remaining / input_shape_d2;
      int head_idx = remaining % input_shape_d2;
      input_row = input + batch_idx * input_stride_d4 +
                  seq_idx * input_stride_d3 + head_idx * input_stride_d2;
    }

    auto vec_op = [&variance](const vec_n_t<scalar_t, VEC_SIZE>& vec) {
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        float x = static_cast<float>(vec.val[i]);
        variance += x * x;
      }
    };

    int64_t const num_vec_elems = hidden_size / VEC_SIZE;
    auto const* vec_in =
        reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(input_row);
    for (int i = item_ct1.get_local_id(2); i < num_vec_elems;
         i += item_ct1.get_local_range(2)) {
      vec_n_t<scalar_t, VEC_SIZE> tmp = vec_in[i];
      vec_op(tmp);
    }

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());
    if (item_ct1.get_local_id(2) == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size + epsilon);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    scalar_t* out_row = out + item_ct1.get_group(2) * hidden_size;
    auto* v_in =
        reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(input_row);
    auto* v_w = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(weight);
    auto* v_out = reinterpret_cast<vec_n_t<scalar_t, VEC_SIZE>*>(out_row);
    int64_t const out_num_vec_elems = hidden_size / VEC_SIZE;
    float s_variance_val = *s_variance_ptr;
    for (int idx = item_ct1.get_local_id(2); idx < out_num_vec_elems;
         idx += item_ct1.get_local_range(2)) {
      vec_n_t<scalar_t, VEC_SIZE> dst;
      vec_n_t<scalar_t, VEC_SIZE> src1 = v_in[idx];
      vec_n_t<scalar_t, VEC_SIZE> src2;
      if constexpr (HasWeight) {
        src2 = v_w[idx];
      }
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        float x = static_cast<float>(src1.val[j]);
        scalar_t normalized = static_cast<scalar_t>(x * s_variance_val);
        if constexpr (HasWeight) {
          dst.val[j] = normalized * src2.val[j];
        } else {
          dst.val[j] = normalized;
        }
      }
      v_out[idx] = dst;
    }
  }

 private:
  scalar_t* __restrict__ out;          // [..., hidden_size]
  const scalar_t* __restrict__ input;  // [..., hidden_size]
  const int64_t input_stride_d2;
  const int64_t input_stride_d3;
  const int64_t input_stride_d4;
  const int64_t input_shape_d2;
  const int64_t input_shape_d3;
  const scalar_t* __restrict__ weight;  // [hidden_size]
  const float epsilon;
  const int num_tokens;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;
};

template <typename scalar_t, int NUM_DIMS, bool HasWeight>
class rms_norm_kernel<scalar_t, NUM_DIMS, 0, HasWeight> {
 public:
  rms_norm_kernel(
      scalar_t* out_,
      const scalar_t* input_,
      const int64_t input_stride_d2_,  // input.stride(-2)
      const int64_t input_stride_d3_,  // input.stride(-3)
      const int64_t input_stride_d4_,  // input.stride(-4)
      const int64_t input_shape_d2_,   // input.size(-2)
      const int64_t input_shape_d3_,   // input.size(-3)
      const scalar_t* weight_,
      const float epsilon_,
      const int num_tokens_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_)
      : out(out_),
        input(input_),
        input_stride_d2(input_stride_d2_),
        input_stride_d3(input_stride_d3_),
        input_stride_d4(input_stride_d4_),
        input_shape_d2(input_shape_d2_),
        input_shape_d3(input_shape_d3_),
        weight(weight_),
        epsilon(epsilon_),
        num_tokens(num_tokens_),
        hidden_size(hidden_size_),
        s_variance(s_variance_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();
    float variance = 0.0f;

    const scalar_t* input_row;
    if constexpr (NUM_DIMS == 2) {
      // 2D for layernorm normal case [batch_size, hidden]
      input_row = input + item_ct1.get_group(2) * input_stride_d2;
    } else if constexpr (NUM_DIMS == 3) {
      // 3D for q/k norm [batch_size, num_heads, head_size]
      int batch_idx = item_ct1.get_group(2) / input_shape_d2;
      int head_idx = item_ct1.get_group(2) % input_shape_d2;
      input_row =
          input + batch_idx * input_stride_d3 + head_idx * input_stride_d2;
    } else if constexpr (NUM_DIMS == 4) {
      // 4D for transformers model_impl qk norm [batch, seq, head, head_dim]
      int batch_idx = item_ct1.get_group(2) / (input_shape_d3 * input_shape_d2);
      int remaining = item_ct1.get_group(2) % (input_shape_d3 * input_shape_d2);
      int seq_idx = remaining / input_shape_d2;
      int head_idx = remaining % input_shape_d2;
      input_row = input + batch_idx * input_stride_d4 +
                  seq_idx * input_stride_d3 + head_idx * input_stride_d2;
    }

    auto scalar_op = [&variance](const scalar_t& val) {
      float x = static_cast<float>(val);
      variance += x * x;
    };

#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < hidden_size;
         i += item_ct1.get_local_range(2)) {
      scalar_op(input_row[i]);
    }

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());
    if (item_ct1.get_local_id(2) == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size + epsilon);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    scalar_t* out_row = out + item_ct1.get_group(2) * hidden_size;
#pragma unroll
    for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
         idx += item_ct1.get_local_range(2)) {
      float x = (float)input_row[idx];
      scalar_t normalized = static_cast<scalar_t>(x * (*s_variance_ptr));
      if constexpr (HasWeight) {
        out_row[idx] = normalized * weight[idx];
      } else {
        out_row[idx] = normalized;
      }
    }
  }

 private:
  scalar_t* __restrict__ out;          // [..., hidden_size]
  const scalar_t* __restrict__ input;  // [..., hidden_size]
  const int64_t input_stride_d2;
  const int64_t input_stride_d3;
  const int64_t input_stride_d4;
  const int64_t input_shape_d2;
  const int64_t input_shape_d3;
  const scalar_t* __restrict__ weight;  // [hidden_size]
  const float epsilon;
  const int num_tokens;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;
};

// Multi-row kernel: each work-group processes ROWS_PER_WG rows.
// The work-group is organized as (ROWS_PER_WG, 1, items_per_row).
// Each "sub-row" uses dimension 0 to index which row it handles,
// and dimension 2 for the column within that row.
template <
    typename scalar_t,
    int NUM_DIMS,
    int VEC_SIZE,
    int ROWS_PER_WG,
    bool HasWeight>
class rms_norm_multi_row_kernel {
 public:
  rms_norm_multi_row_kernel(
      scalar_t* out_,
      const scalar_t* input_,
      const int64_t input_stride_d2_,
      const int64_t input_stride_d3_,
      const int64_t input_stride_d4_,
      const int64_t input_shape_d2_,
      const int64_t input_shape_d3_,
      const scalar_t* weight_,
      const float epsilon_,
      const int num_tokens_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_)
      : out(out_),
        input(input_),
        input_stride_d2(input_stride_d2_),
        input_stride_d3(input_stride_d3_),
        input_stride_d4(input_stride_d4_),
        input_shape_d2(input_shape_d2_),
        input_shape_d3(input_shape_d3_),
        weight(weight_),
        epsilon(epsilon_),
        num_tokens(num_tokens_),
        hidden_size(hidden_size_),
        s_variance(s_variance_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    // dim 0 = row within work-group, dim 2 = column work-item
    const int row_in_wg = item_ct1.get_local_id(0);
    const int col_id = item_ct1.get_local_id(2);
    const int col_range = item_ct1.get_local_range(2);

    // Global row index (guaranteed valid by dispatch: num_tokens % ROWS_PER_WG
    // == 0)
    const int global_row = item_ct1.get_group(2) * ROWS_PER_WG + row_in_wg;

    // s_variance layout: one float per row in the work-group
    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();

    const scalar_t* input_row;
    if constexpr (NUM_DIMS == 2) {
      input_row = input + global_row * input_stride_d2;
    } else if constexpr (NUM_DIMS == 3) {
      int batch_idx = global_row / input_shape_d2;
      int head_idx = global_row % input_shape_d2;
      input_row =
          input + batch_idx * input_stride_d3 + head_idx * input_stride_d2;
    } else if constexpr (NUM_DIMS == 4) {
      int batch_idx = global_row / (input_shape_d3 * input_shape_d2);
      int remaining = global_row % (input_shape_d3 * input_shape_d2);
      int seq_idx = remaining / input_shape_d2;
      int head_idx = remaining % input_shape_d2;
      input_row = input + batch_idx * input_stride_d4 +
                  seq_idx * input_stride_d3 + head_idx * input_stride_d2;
    }

    float variance = 0.0f;
    const int64_t num_vec_elems = hidden_size / VEC_SIZE;
    auto const* vec_in =
        reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(input_row);

    for (int i = col_id; i < num_vec_elems; i += col_range) {
      vec_n_t<scalar_t, VEC_SIZE> tmp = vec_in[i];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        float x = static_cast<float>(tmp.val[j]);
        variance += x * x;
      }
    }

    // Reduce across column work-items for this row using sub-group shifts.
    // With linearized local id = row_in_wg * col_range + col_id,
    // sub-groups of 32 may span multiple rows. We reduce within each
    // row's lanes using shift_group_left-based tree reduction.
    auto sg = item_ct1.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int row_lane_offset = lane % col_range;

    // Tree reduction within col_range consecutive lanes
    for (int offset = col_range / 2; offset > 0; offset >>= 1) {
      float other = sycl::shift_group_left(sg, variance, offset);
      if (row_lane_offset < offset) variance += other;
    }

    // Lane 0 of each row's segment has the final sum
    if (row_lane_offset == 0) {
      s_variance_ptr[row_in_wg] = sycl::rsqrt(variance / hidden_size + epsilon);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Phase 2: normalize
    scalar_t* out_row = out + global_row * hidden_size;
    auto* v_in =
        reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(input_row);
    auto* v_w = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(weight);
    auto* v_out = reinterpret_cast<vec_n_t<scalar_t, VEC_SIZE>*>(out_row);
    float s_var = s_variance_ptr[row_in_wg];

    for (int idx = col_id; idx < num_vec_elems; idx += col_range) {
      vec_n_t<scalar_t, VEC_SIZE> dst;
      vec_n_t<scalar_t, VEC_SIZE> src1 = v_in[idx];
      vec_n_t<scalar_t, VEC_SIZE> src2;
      if constexpr (HasWeight) {
        src2 = v_w[idx];
      }
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        float x = static_cast<float>(src1.val[j]);
        scalar_t normalized = static_cast<scalar_t>(x * s_var);
        if constexpr (HasWeight) {
          dst.val[j] = normalized * src2.val[j];
        } else {
          dst.val[j] = normalized;
        }
      }
      v_out[idx] = dst;
    }
  }

 private:
  scalar_t* __restrict__ out;
  const scalar_t* __restrict__ input;
  const int64_t input_stride_d2;
  const int64_t input_stride_d3;
  const int64_t input_stride_d4;
  const int64_t input_shape_d2;
  const int64_t input_shape_d3;
  const scalar_t* __restrict__ weight;
  const float epsilon;
  const int num_tokens;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;
};

template <typename scalar_t, bool HasWeight>
void call_rms_norm_kernel(
    torch::Tensor& out,
    torch::Tensor& input,
    const scalar_t* weight_ptr,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  int num_dims = input.dim();
  int64_t input_stride_d2 = input.stride(-2);
  int64_t input_stride_d3 = (num_dims >= 3) ? input.stride(-3) : 0;
  int64_t input_stride_d4 = (num_dims >= 4) ? input.stride(-4) : 0;
  int64_t input_shape_d2 = (num_dims >= 3) ? input.size(-2) : 0;
  int64_t input_shape_d3 = (num_dims >= 4) ? input.size(-3) : 0;

  auto out_ptr = out.data_ptr<scalar_t>();
  auto input_ptr = input.data_ptr<scalar_t>();

  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  auto& queue = vllm::xpu::vllmGetQueue();

  constexpr int vec_size = (sizeof(scalar_t) == 2) ? 8 : 4;
  constexpr int req_alignment_bytes = vec_size * sizeof(scalar_t);

  auto inp_addr = reinterpret_cast<std::uintptr_t>(input_ptr);
  auto out_addr = reinterpret_cast<std::uintptr_t>(out_ptr);
  auto wt_addr = reinterpret_cast<std::uintptr_t>(weight_ptr);

  // Base pointers must be aligned
  bool ptrs_aligned =
      (inp_addr % req_alignment_bytes == 0) &&
      (out_addr % req_alignment_bytes == 0) &&
      (weight_ptr == nullptr || wt_addr % req_alignment_bytes == 0);

  // hidden_size must be divisible by vec_size (so vectorized loop covers all
  // elements)
  bool hidden_divisible = (hidden_size % vec_size == 0);

  // Strides must be divisible by vec_size so that each row starts at an aligned
  // offset (input_row = input + batch_idx * stride_d3 + head_idx * stride_d2)
  bool strides_aligned = (input_stride_d2 % vec_size == 0) &&
                         (input_stride_d3 % vec_size == 0 || num_dims < 3) &&
                         (input_stride_d4 % vec_size == 0 || num_dims < 4);

  bool can_vec = ptrs_aligned && hidden_divisible && strides_aligned;
  // Multi-row optimization: when hidden_size is small (e.g., head_dim=128),
  // a single row doesn't have enough work to fill a work-group. Pack multiple
  // rows into one work-group to improve occupancy.
  if (can_vec) {
    const int items_per_row = hidden_size / vec_size;  // e.g., 128/8 = 16
    constexpr int subgroup_size = 32;
    constexpr int ROWS_PER_WG = 16;
    if (items_per_row <= subgroup_size && num_tokens % ROWS_PER_WG == 0 &&
        subgroup_size % items_per_row == 0) {
      // Use multi-row kernel: pack multiple rows per work-group
      const int num_groups = (num_tokens + ROWS_PER_WG - 1) / ROWS_PER_WG;
      // Work-group: dim0 = ROWS_PER_WG, dim1 = 1, dim2 = items_per_row
      sycl::range<3> block(ROWS_PER_WG, 1, items_per_row);
      sycl::range<3> grid(1, 1, num_groups);
      VLLM_DISPATCH_RANK234(num_dims, [&]() {
        queue.submit([&](sycl::handler& cgh) {
          sycl::local_accessor<float, 1> s_variance(
              sycl::range<1>(ROWS_PER_WG), cgh);
          cgh.parallel_for(
              sycl::nd_range<3>(grid * block, block),
              rms_norm_multi_row_kernel<
                  sycl_t,
                  tensor_rank,
                  vec_size,
                  ROWS_PER_WG,
                  HasWeight>(
                  (sycl_t*)out_ptr,
                  (const sycl_t*)input_ptr,
                  input_stride_d2,
                  input_stride_d3,
                  input_stride_d4,
                  input_shape_d2,
                  input_shape_d3,
                  (const sycl_t*)weight_ptr,
                  epsilon,
                  num_tokens,
                  hidden_size,
                  s_variance));
        });
      });
      return;
    }
    sycl::range<3> grid(1, 1, num_tokens);
    sycl::range<3> block(
        1, 1, std::min(hidden_size / vec_size, max_block_size));
    VLLM_DISPATCH_RANK234(num_dims, [&]() {
      queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            rms_norm_kernel<sycl_t, tensor_rank, vec_size, HasWeight>(
                (sycl_t*)out_ptr,
                (const sycl_t*)input_ptr,
                input_stride_d2,
                input_stride_d3,
                input_stride_d4,
                input_shape_d2,
                input_shape_d3,
                (const sycl_t*)weight_ptr,
                epsilon,
                num_tokens,
                hidden_size,
                s_variance));
      });
    });
  } else {
    sycl::range<3> grid(1, 1, num_tokens);
    sycl::range<3> block(1, 1, std::min(hidden_size, max_block_size));
    VLLM_DISPATCH_RANK234(num_dims, [&]() {
      queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            rms_norm_kernel<sycl_t, tensor_rank, 0, HasWeight>(
                (sycl_t*)out_ptr,
                (const sycl_t*)input_ptr,
                input_stride_d2,
                input_stride_d3,
                input_stride_d4,
                input_shape_d2,
                input_shape_d3,
                (const sycl_t*)weight_ptr,
                epsilon,
                num_tokens,
                hidden_size,
                s_variance));
      });
    });
  }
}

template <typename scalar_t, int width, bool HasWeight>
class fused_add_rms_norm_kernel {
 public:
  fused_add_rms_norm_kernel(
      scalar_t* __restrict__ input_,     // [..., hidden_size]
      scalar_t* __restrict__ residual_,  // [..., hidden_size]
      const int64_t input_stride_,
      const scalar_t* __restrict__ weight_,  // [hidden_size]
      const float epsilon_,
      const int num_tokens_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_)
      : input(input_),
        residual(residual_),
        input_stride(input_stride_),
        weight(weight_),
        epsilon(epsilon_),
        num_tokens(num_tokens_),
        hidden_size(hidden_size_),
        s_variance(s_variance_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    static_assert(width > 0, "Use width=0 specialization for scalar path");
    using vec_t = vec_n_t<scalar_t, width>;

    const int vec_hidden_size = hidden_size / width;
    const int64_t vec_input_stride = input_stride / width;

    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();
    float variance = 0.0f;

    auto* __restrict__ input_v = reinterpret_cast<vec_t*>(input);
    auto* __restrict__ residual_v = reinterpret_cast<vec_t*>(residual);
    auto* __restrict__ weight_v = reinterpret_cast<const vec_t*>(weight);

    for (int idx = item_ct1.get_local_id(2); idx < vec_hidden_size;
         idx += item_ct1.get_local_range(2)) {
      int id = item_ct1.get_group(2) * vec_hidden_size + idx;
      int64_t strided_id = item_ct1.get_group(2) * vec_input_stride + idx;
      vec_t temp = input_v[strided_id];
      vec_t res = residual_v[id];
#pragma unroll
      for (int i = 0; i < width; i++) {
        temp.val[i] += res.val[i];
        float x = static_cast<float>(temp.val[i]);
        variance += x * x;
      }
      residual_v[id] = temp;
    }

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());
    if (item_ct1.get_local_id(2) == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size + epsilon);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    float s_var = *s_variance_ptr;
    for (int idx = item_ct1.get_local_id(2); idx < vec_hidden_size;
         idx += item_ct1.get_local_range(2)) {
      int id = item_ct1.get_group(2) * vec_hidden_size + idx;
      int64_t strided_id = item_ct1.get_group(2) * vec_input_stride + idx;
      vec_t res = residual_v[id];
      vec_t w;
      if constexpr (HasWeight) {
        w = weight_v[idx];
      }
      vec_t out;
#pragma unroll
      for (int i = 0; i < width; i++) {
        float x = static_cast<float>(res.val[i]);
        scalar_t normalized = static_cast<scalar_t>(x * s_var);
        if constexpr (HasWeight) {
          out.val[i] = normalized * w.val[i];
        } else {
          out.val[i] = normalized;
        }
      }
      input_v[strided_id] = out;
    }
  }

 private:
  scalar_t* __restrict__ input;     // [..., hidden_size]
  scalar_t* __restrict__ residual;  // [..., hidden_size]
  const int64_t input_stride;
  const scalar_t* __restrict__ weight;  // [hidden_size]
  const float epsilon;
  const int num_tokens;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;  // local memory for variance
};

template <typename scalar_t, bool HasWeight>
class fused_add_rms_norm_kernel<scalar_t, 0, HasWeight> {
 public:
  fused_add_rms_norm_kernel(
      scalar_t* __restrict__ input_,     // [..., hidden_size]
      scalar_t* __restrict__ residual_,  // [..., hidden_size]
      const int64_t input_stride_,
      const scalar_t* __restrict__ weight_,  // [hidden_size]
      const float epsilon_,
      const int num_tokens_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_)
      : input(input_),
        residual(residual_),
        input_stride(input_stride_),
        weight(weight_),
        epsilon(epsilon_),
        num_tokens(num_tokens_),
        hidden_size(hidden_size_),
        s_variance(s_variance_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();
    float variance = 0.0f;

    for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
         idx += item_ct1.get_local_range(2)) {
      scalar_t z = (scalar_t)input[item_ct1.get_group(2) * input_stride + idx];
      z += residual[item_ct1.get_group(2) * hidden_size + idx];
      float x = (float)z;
      variance += x * x;
      residual[item_ct1.get_group(2) * hidden_size + idx] = z;
    }

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());
    if (item_ct1.get_local_id(2) == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size + epsilon);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
         idx += item_ct1.get_local_range(2)) {
      float x = (float)residual[item_ct1.get_group(2) * hidden_size + idx];
      scalar_t normalized = static_cast<scalar_t>(x * (*s_variance_ptr));
      if constexpr (HasWeight) {
        input[item_ct1.get_group(2) * input_stride + idx] =
            normalized * weight[idx];
      } else {
        input[item_ct1.get_group(2) * input_stride + idx] = normalized;
      }
    }
  }

 private:
  scalar_t* __restrict__ input;     // [..., hidden_size]
  scalar_t* __restrict__ residual;  // [..., hidden_size]
  const int64_t input_stride;
  const scalar_t* __restrict__ weight;  // [hidden_size]
  const float epsilon;
  const int num_tokens;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;  // local memory for variance
};

template <typename scalar_t, bool HasWeight>
void call_fused_add_rms_norm_kernel(
    torch::Tensor& input,
    torch::Tensor& residual,
    const scalar_t* weight_ptr,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  auto input_ptr = input.data_ptr<scalar_t>();
  auto residual_ptr = residual.data_ptr<scalar_t>();
  int64_t input_stride = input.stride(-2);

  constexpr int vector_width = (sizeof(scalar_t) == 2) ? 8 : 4;
  constexpr int req_alignment_bytes = vector_width * sizeof(scalar_t);
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input_ptr);
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual_ptr);
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight_ptr);
  bool ptrs_are_aligned =
      inp_ptr % req_alignment_bytes == 0 &&
      res_ptr % req_alignment_bytes == 0 &&
      (weight_ptr == nullptr || wt_ptr % req_alignment_bytes == 0);
  bool offsets_are_multiple_of_vector_width =
      hidden_size % vector_width == 0 && input_stride % vector_width == 0;
  bool can_vec = ptrs_are_aligned && offsets_are_multiple_of_vector_width;

  sycl::range<3> grid(1, 1, num_tokens);
  auto& queue = vllm::xpu::vllmGetQueue();

  if (can_vec) {
    sycl::range<3> block(
        1, 1, std::min(hidden_size / vector_width, max_block_size));
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          fused_add_rms_norm_kernel<sycl_t, vector_width, HasWeight>(
              (sycl_t*)input_ptr,
              (sycl_t*)residual_ptr,
              input_stride,
              (const sycl_t*)weight_ptr,
              epsilon,
              num_tokens,
              hidden_size,
              s_variance));
    });
  } else {
    sycl::range<3> block(1, 1, std::min(hidden_size, max_block_size));
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          fused_add_rms_norm_kernel<sycl_t, 0, HasWeight>(
              (sycl_t*)input_ptr,
              (sycl_t*)residual_ptr,
              input_stride,
              (const sycl_t*)weight_ptr,
              epsilon,
              num_tokens,
              hidden_size,
              s_variance));
    });
  }
}

}  // namespace vllm

void rms_norm(
    torch::Tensor& out,
    torch::Tensor& input,
    std::optional<torch::Tensor> weight,
    double epsilon) {
  const at::DeviceGuard device_guard(input.device());
  TORCH_CHECK(out.is_contiguous());
  if (input.stride(-1) != 1) {
    input = input.contiguous();
  }
  TORCH_CHECK(input.stride(-1) == 1);
  const bool has_weight = weight.has_value();
  if (has_weight) {
    TORCH_CHECK(weight->is_contiguous());
  }
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "call_rms_norm_kernel", [&] {
        const scalar_t* weight_ptr =
            has_weight ? weight->data_ptr<scalar_t>() : nullptr;
        if (has_weight) {
          vllm::call_rms_norm_kernel<scalar_t, true>(
              out, input, weight_ptr, epsilon);
        } else {
          vllm::call_rms_norm_kernel<scalar_t, false>(
              out, input, weight_ptr, epsilon);
        }
      });
}

void fused_add_rms_norm(
    torch::Tensor& input,
    torch::Tensor& residual,
    std::optional<torch::Tensor> weight,
    double epsilon) {
  const at::DeviceGuard device_guard(input.device());
  const bool has_weight = weight.has_value();
  if (has_weight) {
    TORCH_CHECK(weight->is_contiguous());
  }

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "call_fused_add_rms_norm_kernel", [&] {
        const scalar_t* weight_ptr =
            has_weight ? weight->data_ptr<scalar_t>() : nullptr;
        if (has_weight) {
          vllm::call_fused_add_rms_norm_kernel<scalar_t, true>(
              input, residual, weight_ptr, epsilon);
        } else {
          vllm::call_fused_add_rms_norm_kernel<scalar_t, false>(
              input, residual, weight_ptr, epsilon);
        }
      });
}
