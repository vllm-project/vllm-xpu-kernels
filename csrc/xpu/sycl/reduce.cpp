// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Row-wise reduce-max / reduce-min (value + index) over the last dimension
// of a 2D [batch, dim] tensor.
// Ported from xpu-perf's projects/micro_perf/vendor_ops/INTEL/ops/sycl_ext/
// reduce_max_kernel.cpp and reduce_min_kernel.cpp, unified into a single
// template kernel with the min/max comparison as a functor template
// parameter (ReduceMaxOp / ReduceMinOp below).
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>

#include <ATen/DeviceGuard.h>

#include "utils.h"
#include "dispatch_utils.h"

namespace vllm {

constexpr int kVectorSizeInBytes = 16;

// ---- min/max comparison functors ----
struct ReduceMaxOp {
  using Compare = std::greater<float>;
  static constexpr float kIdentity = -std::numeric_limits<float>::infinity();
};

struct ReduceMinOp {
  using Compare = std::less<float>;
  static constexpr float kIdentity = std::numeric_limits<float>::infinity();
};

template <typename scalar_t, int size>
struct alignas(sizeof(scalar_t) * size) ReduceInputVector {
  scalar_t data[size];
};

// NaN-aware "keep lhs" predicate: NaN always wins (propagates), ties and
// NaN-vs-NaN break by the lower index, otherwise Op::Compare decides.
template <typename Op>
inline bool reduce_keep_lhs(float lhs_v, int lhs_i, float rhs_v, int rhs_i) {
  bool lhs_nan = std::isnan(lhs_v);
  bool rhs_nan = std::isnan(rhs_v);

  if (lhs_nan) {
    if (rhs_nan) return lhs_i < rhs_i;
    return true;
  }

  if (lhs_v == rhs_v) return lhs_i < rhs_i;

  return typename Op::Compare{}(lhs_v, rhs_v);
}

// Mirrors the tuned xpu-perf max/min "auto" rows-per-group selection.
template <typename scalar_t>
inline int reduce_auto_rows_per_group(int batch, int dim_size) {
  // One row per group selects the baseline kernel for large dimensions.
  if (dim_size > 4096) return 1;
  if (dim_size != 1024 && dim_size != 2048 && dim_size != 4096) return 4;

  if (batch <= 1024) return 4;
  if (batch <= 2048) return 8;

  if (dim_size == 4096) {
    if (sizeof(scalar_t) == 2) return batch <= 8192 ? 16 : 32;
    return batch <= 8192 ? 4 : 2;
  }

  if (batch <= 4096) return 16;
  if (sizeof(scalar_t) == 2) return 32;
  if (dim_size == 1024) {
    if (batch <= 8192) return 16;
    if (batch <= 32768) return 32;
    return 8;
  }
  return batch <= 16384 ? 32 : 4;
}

// ==========================================================================
// Baseline kernel: 1 work-group per row.
// dim_size <= 1024 uses a plain strided scan (2 elems/iter); dim_size > 1024
// uses a 4-wide register-accumulated scan to cut loop overhead. Only 16-bit
// dtypes take this split (matches the ported reference kernels); fp32 always
// uses the plain per-lane strided scan below.
// ==========================================================================
template <typename Op, typename scalar_t>
class reduce_baseline_kernel {
 public:
  reduce_baseline_kernel(
      const scalar_t* input,
      scalar_t* out_values,
      int32_t* out_indices,
      int dim_size,
      int wg_size,
      sycl::local_accessor<float, 1> local_val,
      sycl::local_accessor<int, 1> local_idx)
      : input_(input),
        out_values_(out_values),
        out_indices_(out_indices),
        dim_size_(dim_size),
        wg_size_(wg_size),
        local_val_(local_val),
        local_idx_(local_idx) {}

  void operator()(sycl::nd_item<1> item) const {
    int lid = static_cast<int>(item.get_local_id(0));
    int row = static_cast<int>(item.get_group(0));
    std::size_t base =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(dim_size_);

    float thread_val = Op::kIdentity;
    int thread_idx = 0;

    if constexpr (sizeof(scalar_t) == 2) {
      if (dim_size_ <= 1024) {
        for (int i = lid; i < dim_size_; i += (wg_size_ << 1)) {
          float v = static_cast<float>(input_[base + i]);
          if (!reduce_keep_lhs<Op>(thread_val, thread_idx, v, i)) {
            thread_val = v;
            thread_idx = i;
          }
          int j = i + wg_size_;
          if (j < dim_size_) {
            float v2 = static_cast<float>(input_[base + j]);
            if (!reduce_keep_lhs<Op>(thread_val, thread_idx, v2, j)) {
              thread_val = v2;
              thread_idx = j;
            }
          }
        }
      } else {
        constexpr int kInputVec = 4;
        float acc_v[kInputVec];
        int acc_i[kInputVec];
#pragma unroll
        for (int k = 0; k < kInputVec; ++k) {
          acc_v[k] = Op::kIdentity;
          acc_i[k] = 0;
        }

        int vec_start = lid * kInputVec;
        int vec_step = wg_size_ * kInputVec;
        int vec_tail = (dim_size_ / kInputVec) * kInputVec;

        for (int i = vec_start; i < vec_tail; i += vec_step) {
          float v0 = static_cast<float>(input_[base + i + 0]);
          float v1 = static_cast<float>(input_[base + i + 1]);
          float v2 = static_cast<float>(input_[base + i + 2]);
          float v3 = static_cast<float>(input_[base + i + 3]);

          if (!reduce_keep_lhs<Op>(acc_v[0], acc_i[0], v0, i + 0)) {
            acc_v[0] = v0;
            acc_i[0] = i + 0;
          }
          if (!reduce_keep_lhs<Op>(acc_v[1], acc_i[1], v1, i + 1)) {
            acc_v[1] = v1;
            acc_i[1] = i + 1;
          }
          if (!reduce_keep_lhs<Op>(acc_v[2], acc_i[2], v2, i + 2)) {
            acc_v[2] = v2;
            acc_i[2] = i + 2;
          }
          if (!reduce_keep_lhs<Op>(acc_v[3], acc_i[3], v3, i + 3)) {
            acc_v[3] = v3;
            acc_i[3] = i + 3;
          }
        }

        for (int i = vec_tail + lid; i < dim_size_; i += wg_size_) {
          float v = static_cast<float>(input_[base + i]);
          if (!reduce_keep_lhs<Op>(acc_v[0], acc_i[0], v, i)) {
            acc_v[0] = v;
            acc_i[0] = i;
          }
        }

        thread_val = acc_v[0];
        thread_idx = acc_i[0];
#pragma unroll
        for (int k = 1; k < kInputVec; ++k) {
          if (!reduce_keep_lhs<Op>(
                  thread_val, thread_idx, acc_v[k], acc_i[k])) {
            thread_val = acc_v[k];
            thread_idx = acc_i[k];
          }
        }
      }
    } else {
      for (int i = lid; i < dim_size_; i += wg_size_) {
        float v = static_cast<float>(input_[base + i]);
        if (!reduce_keep_lhs<Op>(thread_val, thread_idx, v, i)) {
          thread_val = v;
          thread_idx = i;
        }
      }
    }

    local_val_[lid] = thread_val;
    local_idx_[lid] = thread_idx;
    sycl::group_barrier(item.get_group());

    for (int stride = wg_size_ / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        float cand_v = local_val_[lid + stride];
        int cand_i = local_idx_[lid + stride];
        if (!reduce_keep_lhs<Op>(
                local_val_[lid], local_idx_[lid], cand_v, cand_i)) {
          local_val_[lid] = cand_v;
          local_idx_[lid] = cand_i;
        }
      }
      sycl::group_barrier(item.get_group());
    }

    if (lid == 0) {
      out_values_[row] = static_cast<scalar_t>(local_val_[0]);
      out_indices_[row] = local_idx_[0];
    }
  }

 private:
  const scalar_t* input_;
  scalar_t* out_values_;
  int32_t* out_indices_;
  int dim_size_;
  int wg_size_;
  sycl::local_accessor<float, 1> local_val_;
  sycl::local_accessor<int, 1> local_idx_;
};

// ==========================================================================
// Multi-row kernel: several short rows share one work-group (threads_per_row
// lanes each) when dim_size is small enough that a single row can't fill a
// work-group profitably.
// ==========================================================================
template <typename Op, typename scalar_t>
class reduce_multirow_kernel {
 public:
  reduce_multirow_kernel(
      const scalar_t* input,
      scalar_t* out_values,
      int32_t* out_indices,
      int dim_size,
      int batch,
      int rows_per_group,
      int threads_per_row,
      sycl::local_accessor<float, 1> local_val,
      sycl::local_accessor<int, 1> local_idx)
      : input_(input),
        out_values_(out_values),
        out_indices_(out_indices),
        dim_size_(dim_size),
        batch_(batch),
        rows_per_group_(rows_per_group),
        threads_per_row_(threads_per_row),
        local_val_(local_val),
        local_idx_(local_idx) {}

  void operator()(sycl::nd_item<1> item) const {
    int lid = static_cast<int>(item.get_local_id(0));
    int group_id = static_cast<int>(item.get_group(0));
    int row_in_group = lid / threads_per_row_;
    int lane = lid % threads_per_row_;
    int row = group_id * rows_per_group_ + row_in_group;

    float thread_val = Op::kIdentity;
    int thread_idx = 0;

    if (row < batch_) {
      std::size_t base =
          static_cast<std::size_t>(row) * static_cast<std::size_t>(dim_size_);
      constexpr int kInputVec = kVectorSizeInBytes / sizeof(scalar_t);
      using InputVector = ReduceInputVector<scalar_t, kInputVec>;
      int vec_end = (dim_size_ / kInputVec) * kInputVec;
      for (int i = lane * kInputVec; i < vec_end;
           i += threads_per_row_ * kInputVec) {
        InputVector values = *reinterpret_cast<const InputVector*>(
            input_ + base + static_cast<std::size_t>(i));
#pragma unroll
        for (int j = 0; j < kInputVec; ++j) {
          float v = static_cast<float>(values.data[j]);
          if (!reduce_keep_lhs<Op>(thread_val, thread_idx, v, i + j)) {
            thread_val = v;
            thread_idx = i + j;
          }
        }
      }
      // There is no tail in multi-row kernel.
    }

    local_val_[lid] = thread_val;
    local_idx_[lid] = thread_idx;
    sycl::group_barrier(item.get_group());

    int seg_base = row_in_group * threads_per_row_;
    for (int stride = threads_per_row_ / 2; stride > 0; stride >>= 1) {
      if (lane < stride) {
        int left = seg_base + lane;
        int right = left + stride;
        float cand_v = local_val_[right];
        int cand_i = local_idx_[right];
        if (!reduce_keep_lhs<Op>(
                local_val_[left], local_idx_[left], cand_v, cand_i)) {
          local_val_[left] = cand_v;
          local_idx_[left] = cand_i;
        }
      }
      sycl::group_barrier(item.get_group());
    }

    if (lane == 0 && row < batch_) {
      out_values_[row] = static_cast<scalar_t>(local_val_[seg_base]);
      out_indices_[row] = local_idx_[seg_base];
    }
  }

 private:
  const scalar_t* input_;
  scalar_t* out_values_;
  int32_t* out_indices_;
  int dim_size_;
  int batch_;
  int rows_per_group_;
  int threads_per_row_;
  sycl::local_accessor<float, 1> local_val_;
  sycl::local_accessor<int, 1> local_idx_;
};

template <typename Op, typename scalar_t>
void launch_reduce_baseline_kernel(
    sycl::queue& q,
    const scalar_t* input,
    scalar_t* out_values,
    int32_t* out_indices,
    int batch,
    int dim_size) {
  int wg_size = 256;
  if (dim_size < wg_size) {
    wg_size = 1;
    while (wg_size * 2 <= dim_size)
      wg_size *= 2;
  }

  sycl::range<1> local_range(static_cast<std::size_t>(wg_size));
  sycl::range<1> global_range(static_cast<std::size_t>(batch) * local_range[0]);

  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> local_val(local_range, cgh);
    sycl::local_accessor<int, 1> local_idx(local_range, cgh);
    reduce_baseline_kernel<Op, scalar_t> kfn(
        input,
        out_values,
        out_indices,
        dim_size,
        wg_size,
        local_val,
        local_idx);
    cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), kfn);
  });
}

template <typename Op, typename scalar_t>
void launch_reduce_multirow_kernel(
    sycl::queue& q,
    const scalar_t* input,
    scalar_t* out_values,
    int32_t* out_indices,
    int batch,
    int dim_size,
    int rows_per_group) {
  constexpr int kWgSize = 256;
  const int threads_per_row = kWgSize / rows_per_group;
  const int group_count = (batch + rows_per_group - 1) / rows_per_group;

  sycl::range<1> local_range(static_cast<std::size_t>(kWgSize));
  sycl::range<1> global_range(
      static_cast<std::size_t>(group_count) * local_range[0]);

  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> local_val(local_range, cgh);
    sycl::local_accessor<int, 1> local_idx(local_range, cgh);
    reduce_multirow_kernel<Op, scalar_t> kfn(
        input,
        out_values,
        out_indices,
        dim_size,
        batch,
        rows_per_group,
        threads_per_row,
        local_val,
        local_idx);
    cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), kfn);
  });
}

template <typename Op, typename scalar_t>
void launch_reduce_kernel(
    torch::Tensor& values, torch::Tensor& indices, const torch::Tensor& input) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  const int batch = static_cast<int>(input.size(0));
  const int dim_size = static_cast<int>(input.size(1));
  const int rows_per_group =
      reduce_auto_rows_per_group<sycl_t>(batch, dim_size);

  auto* d_in =
      reinterpret_cast<const sycl_t*>(input.const_data_ptr<scalar_t>());
  auto* d_val = reinterpret_cast<sycl_t*>(values.data_ptr<scalar_t>());
  auto* d_idx = indices.data_ptr<int32_t>();

  auto& q = vllm::xpu::vllmGetQueue();

  constexpr int kInputVec = kVectorSizeInBytes / sizeof(sycl_t);
  const bool use_multirow = dim_size <= 4096 && rows_per_group > 1 &&
                            (reinterpret_cast<std::uintptr_t>(d_in) &
                             (kVectorSizeInBytes - 1)) == 0 &&
                            dim_size % kInputVec == 0;
  if (use_multirow) {
    launch_reduce_multirow_kernel<Op, sycl_t>(
        q, d_in, d_val, d_idx, batch, dim_size, rows_per_group);
  } else {
    launch_reduce_baseline_kernel<Op, sycl_t>(
        q, d_in, d_val, d_idx, batch, dim_size);
  }
}

template <typename Op>
void reduce(
    torch::Tensor& values, torch::Tensor& indices, const torch::Tensor& input) {
  const at::DeviceGuard device_guard(input.device());
  CHECK_DEVICE(input);
  CHECK_DEVICE(values);
  CHECK_DEVICE(indices);
  TORCH_CHECK(
      values.device() == input.device() && indices.device() == input.device(),
      "input/values/indices must be on the same device");
  TORCH_CHECK(input.dim() == 2, "input must be 2D [batch, dim]");
  TORCH_CHECK(
      values.dim() == 2 && indices.dim() == 2,
      "values/indices must be 2D [batch, 1]");
  TORCH_CHECK(
      values.size(0) == input.size(0) && values.size(1) == 1,
      "values shape must be [batch, 1]");
  TORCH_CHECK(
      indices.size(0) == input.size(0) && indices.size(1) == 1,
      "indices shape must be [batch, 1]");
  TORCH_CHECK(
      input.is_contiguous() && values.is_contiguous() &&
          indices.is_contiguous(),
      "input/values/indices must be contiguous");
  TORCH_CHECK(
      indices.scalar_type() == at::ScalarType::Int,
      "indices dtype must be int32");
  TORCH_CHECK(
      values.scalar_type() == input.scalar_type(),
      "values/input dtype mismatch");
  TORCH_CHECK(
      input.size(0) > 0 && input.size(1) > 0,
      "batch/dim_size must be positive");

  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "reduce", [&] {
    vllm::launch_reduce_kernel<Op, scalar_t>(values, indices, input);
  });
}

}  // namespace vllm

// Row-wise argmax: writes the max value and its index over the last
// dimension of a 2D [batch, dim] input into values/indices ([batch, 1]).
void reduce_max_row_wise(
    torch::Tensor& values, torch::Tensor& indices, const torch::Tensor& input) {
  vllm::reduce<vllm::ReduceMaxOp>(values, indices, input);
}

// Row-wise argmin: writes the min value and its index over the last
// dimension of a 2D [batch, dim] input into values/indices ([batch, 1]).
void reduce_min_row_wise(
    torch::Tensor& values, torch::Tensor& indices, const torch::Tensor& input) {
  vllm::reduce<vllm::ReduceMinOp>(values, indices, input);
}
