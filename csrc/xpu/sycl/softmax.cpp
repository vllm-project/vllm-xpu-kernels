// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Row-wise softmax over the last dimension of a 2D [batch, dim] tensor.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

#include <ATen/DeviceGuard.h>

#include "utils.h"
#include "dispatch_utils.h"
#include "quantization/utils.h"

namespace vllm {

template <typename scalar_t, int N>
struct softmax_packed_io {
  using vec_type = vec_n_t<scalar_t, N>;
  static float get(const vec_type& v, int j) {
    return static_cast<float>(v.val[j]);
  }
  static void set(vec_type& v, int j, float f) {
    v.val[j] = static_cast<scalar_t>(f);
  }
};

template <int N>
struct softmax_packed_io<sycl::ext::oneapi::bfloat16, N> {
  static_assert(N % 2 == 0, "bfloat16 packed IO needs an even element count");
  using vec_type = vec_n_t<uint32_t, N / 2>;
  static float get(const vec_type& v, int j) {
    uint32_t w = v.val[j >> 1];
    return sycl::bit_cast<float>((j & 1) ? (w & 0xFFFF0000u) : (w << 16));
  }
  static void set(vec_type& v, int j, float f) {
    uint32_t h =
        sycl::bit_cast<uint16_t>(static_cast<sycl::ext::oneapi::bfloat16>(f));
    uint32_t& w = v.val[j >> 1];
    w = (j & 1) ? ((w & 0x0000FFFFu) | (h << 16)) : ((w & 0xFFFF0000u) | h);
  }
};

// Use 16-byte transactions in the double-read kernel. The row-cache kernel
// keeps vec4 to avoid increasing register pressure alongside its SLM cache.
constexpr std::size_t kSoftmaxVecBytes = 16;

template <typename scalar_t>
constexpr int softmax_vec_size() {
  return static_cast<int>(kSoftmaxVecBytes / sizeof(scalar_t));
}

template <typename scalar_t>
int pick_softmax_wg_size(int dim_size, int device_wg_max) {
  int wg_size;
  if constexpr (std::is_same_v<scalar_t, float>) {
    if (dim_size <= 2048) {
      wg_size = 128;
    } else if (dim_size <= 8192) {
      wg_size = 256;
    } else if (dim_size <= 16384) {
      wg_size = 512;
    } else {
      wg_size = 1024;
    }
  } else if constexpr (
      std::is_same_v<scalar_t, sycl::half> ||
      std::is_same_v<scalar_t, sycl::ext::oneapi::bfloat16>) {
    if (dim_size <= 4096) {
      wg_size = 128;
    } else if (dim_size <= 16384) {
      wg_size = 256;
    } else if (dim_size <= 32768) {
      wg_size = 512;
    } else {
      wg_size = 1024;
    }
  }

  if (wg_size > device_wg_max) {
    wg_size = 1;
    while (wg_size * 2 <= device_wg_max) {
      wg_size *= 2;
    }
  }

  if (dim_size < wg_size) {
    wg_size = 1;
    while (wg_size * 2 <= dim_size) {
      wg_size *= 2;
    }
  }

  return wg_size;
}

// Merges the reduction state (running max/sum of a partial online-softmax)
// held by each work-item / sub-group into a single row-wide max and sum.
inline void merge_row_reduction(
    int lid,
    int wg_size,
    sycl::nd_item<1> item,
    sycl::local_accessor<float, 1> local_max,
    sycl::local_accessor<float, 1> local_sum,
    float thread_max,
    float thread_sum,
    float scale_log2e) {
  if (wg_size >= 512) {
    local_max[lid] = thread_max;
    local_sum[lid] = thread_sum;
    sycl::group_barrier(item.get_group());

    for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        float m0 = local_max[lid];
        float s0 = local_sum[lid];
        float m1 = local_max[lid + stride];
        float s1 = local_sum[lid + stride];
        float merged_max = sycl::fmax(m0, m1);
        float merged_sum =
            s0 * sycl::native::exp2(scale_log2e * (m0 - merged_max)) +
            s1 * sycl::native::exp2(scale_log2e * (m1 - merged_max));
        local_max[lid] = merged_max;
        local_sum[lid] = merged_sum;
      }
      sycl::group_barrier(item.get_group());
    }
  } else {
    sycl::sub_group sg = item.get_sub_group();
    int sg_size = static_cast<int>(sg.get_local_range().size());
    int sg_id = lid / sg_size;
    int num_sg = (wg_size + sg_size - 1) / sg_size;

    float sg_max =
        sycl::reduce_over_group(sg, thread_max, sycl::maximum<float>());
    float sg_sum = sycl::reduce_over_group(
        sg,
        thread_sum * sycl::native::exp2(scale_log2e * (thread_max - sg_max)),
        sycl::plus<float>());

    if (sg.leader()) {
      local_max[sg_id] = sg_max;
      local_sum[sg_id] = sg_sum;
    }
    sycl::group_barrier(item.get_group());

    if (lid == 0) {
      float row_max = local_max[0];
      float row_sum = local_sum[0];
      for (int g = 1; g < num_sg; ++g) {
        float m1 = local_max[g];
        float s1 = local_sum[g];
        float merged_max = sycl::fmax(row_max, m1);
        float merged_sum =
            row_sum * sycl::native::exp2(scale_log2e * (row_max - merged_max)) +
            s1 * sycl::native::exp2(scale_log2e * (m1 - merged_max));
        row_max = merged_max;
        row_sum = merged_sum;
      }
      local_max[0] = row_max;
      local_sum[0] = row_sum;
    }
    sycl::group_barrier(item.get_group());
  }
}

template <typename scalar_t>
class softmax_row_cache_kernel {
 public:
  softmax_row_cache_kernel(
      const scalar_t* input,
      scalar_t* output,
      int dim_size,
      int wg_size,
      float scale_log2e,
      sycl::local_accessor<float, 1> local_max,
      sycl::local_accessor<float, 1> local_sum,
      sycl::local_accessor<float, 1> row_cache)
      : input_(input),
        output_(output),
        dim_size_(dim_size),
        wg_size_(wg_size),
        scale_log2e_(scale_log2e),
        local_max_(local_max),
        local_sum_(local_sum),
        row_cache_(row_cache) {}

  void operator()(sycl::nd_item<1> item) const {
    constexpr int kVec = 4;
    using pio = softmax_packed_io<scalar_t, kVec>;
    using vec_type = typename pio::vec_type;

    int lid = static_cast<int>(item.get_local_id(0));
    int row = static_cast<int>(item.get_group(0));
    std::size_t base =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(dim_size_);

    float thread_max = -std::numeric_limits<float>::infinity();
    float thread_sum = 0.0f;

    const scalar_t* row_in = input_ + base;
    scalar_t* row_out = output_ + base;
    const int n_vec = dim_size_ / kVec;
    auto* v_in = reinterpret_cast<const vec_type*>(row_in);

    for (int vi = lid; vi < n_vec; vi += wg_size_) {
      vec_type tmp = v_in[vi];
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        float v = pio::get(tmp, j);
        row_cache_[vi * kVec + j] = v;
        float new_max = sycl::fmax(thread_max, v);
        thread_sum = thread_sum * sycl::native::exp2(
                                      scale_log2e_ * (thread_max - new_max)) +
                     sycl::native::exp2(scale_log2e_ * (v - new_max));
        thread_max = new_max;
      }
    }

    merge_row_reduction(
        lid,
        wg_size_,
        item,
        local_max_,
        local_sum_,
        thread_max,
        thread_sum,
        scale_log2e_);

    float row_max = local_max_[0];
    float inv_sum = 1.0f / local_sum_[0];

    auto* v_out = reinterpret_cast<vec_type*>(row_out);
    for (int vi = lid; vi < n_vec; vi += wg_size_) {
      vec_type tout;
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        float v = row_cache_[vi * kVec + j];
        float p = sycl::native::exp2(scale_log2e_ * (v - row_max)) * inv_sum;
        pio::set(tout, j, p);
      }
      v_out[vi] = tout;
    }
  }

 private:
  const scalar_t* input_;
  scalar_t* output_;
  int dim_size_;
  int wg_size_;
  float scale_log2e_;
  sycl::local_accessor<float, 1> local_max_;
  sycl::local_accessor<float, 1> local_sum_;
  sycl::local_accessor<float, 1> row_cache_;
};

// Generic double-read fallback: works for any dtype/dim_size/alignment.
// Internally picks a 16-byte vector or scalar strided load/store depending on
// whether the row is aligned (decided once on the host, passed in as can_vec).
template <typename scalar_t, int kVec>
class softmax_kernel {
 public:
  softmax_kernel(
      const scalar_t* input,
      scalar_t* output,
      int dim_size,
      int wg_size,
      float scale_log2e,
      bool can_vec,
      sycl::local_accessor<float, 1> local_max,
      sycl::local_accessor<float, 1> local_sum)
      : input_(input),
        output_(output),
        dim_size_(dim_size),
        wg_size_(wg_size),
        scale_log2e_(scale_log2e),
        can_vec_(can_vec),
        local_max_(local_max),
        local_sum_(local_sum) {}

  void operator()(sycl::nd_item<1> item) const {
    int lid = static_cast<int>(item.get_local_id(0));
    int row = static_cast<int>(item.get_group(0));
    std::size_t base =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(dim_size_);

    float thread_max = -std::numeric_limits<float>::infinity();
    float thread_sum = 0.0f;

    using pio = softmax_packed_io<scalar_t, kVec>;
    using vec_type = typename pio::vec_type;
    const scalar_t* row_in = input_ + base;
    scalar_t* row_out = output_ + base;

    if (can_vec_) {
      const int n_vec = dim_size_ / kVec;
      auto* v_in = reinterpret_cast<const vec_type*>(row_in);
      for (int vi = lid; vi < n_vec; vi += wg_size_) {
        vec_type tmp = v_in[vi];
#pragma unroll
        for (int j = 0; j < kVec; ++j) {
          float v = pio::get(tmp, j);
          float new_max = sycl::fmax(thread_max, v);
          thread_sum = thread_sum * sycl::native::exp2(
                                        scale_log2e_ * (thread_max - new_max)) +
                       sycl::native::exp2(scale_log2e_ * (v - new_max));
          thread_max = new_max;
        }
      }
    } else {
      int step = wg_size_ * 2;
      int i = lid;
      for (; i + wg_size_ < dim_size_; i += step) {
        float v0 = static_cast<float>(row_in[i]);
        float new_max0 = sycl::fmax(thread_max, v0);
        thread_sum = thread_sum * sycl::native::exp2(
                                      scale_log2e_ * (thread_max - new_max0)) +
                     sycl::native::exp2(scale_log2e_ * (v0 - new_max0));
        thread_max = new_max0;

        int i1 = i + wg_size_;
        float v1 = static_cast<float>(row_in[i1]);
        float new_max1 = sycl::fmax(thread_max, v1);
        thread_sum = thread_sum * sycl::native::exp2(
                                      scale_log2e_ * (thread_max - new_max1)) +
                     sycl::native::exp2(scale_log2e_ * (v1 - new_max1));
        thread_max = new_max1;
      }
      for (; i < dim_size_; i += wg_size_) {
        float v = static_cast<float>(row_in[i]);
        float new_max = sycl::fmax(thread_max, v);
        thread_sum = thread_sum * sycl::native::exp2(
                                      scale_log2e_ * (thread_max - new_max)) +
                     sycl::native::exp2(scale_log2e_ * (v - new_max));
        thread_max = new_max;
      }
    }

    merge_row_reduction(
        lid,
        wg_size_,
        item,
        local_max_,
        local_sum_,
        thread_max,
        thread_sum,
        scale_log2e_);

    float row_max = local_max_[0];
    float inv_sum = 1.0f / local_sum_[0];

    if (can_vec_) {
      const int n_vec = dim_size_ / kVec;
      auto* v_in = reinterpret_cast<const vec_type*>(row_in);
      auto* v_out = reinterpret_cast<vec_type*>(row_out);
      for (int vi = lid; vi < n_vec; vi += wg_size_) {
        vec_type tin = v_in[vi];
        vec_type tout;
#pragma unroll
        for (int j = 0; j < kVec; ++j) {
          float v = pio::get(tin, j);
          float p = sycl::native::exp2(scale_log2e_ * (v - row_max)) * inv_sum;
          pio::set(tout, j, p);
        }
        v_out[vi] = tout;
      }
    } else {
      int step_out = wg_size_ * 2;
      int i_out = lid;
      for (; i_out + wg_size_ < dim_size_; i_out += step_out) {
        float v0 = static_cast<float>(row_in[i_out]);
        float p0 = sycl::native::exp2(scale_log2e_ * (v0 - row_max)) * inv_sum;
        row_out[i_out] = static_cast<scalar_t>(p0);

        int i1 = i_out + wg_size_;
        float v1 = static_cast<float>(row_in[i1]);
        float p1 = sycl::native::exp2(scale_log2e_ * (v1 - row_max)) * inv_sum;
        row_out[i1] = static_cast<scalar_t>(p1);
      }
      for (; i_out < dim_size_; i_out += wg_size_) {
        float v = static_cast<float>(row_in[i_out]);
        float p = sycl::native::exp2(scale_log2e_ * (v - row_max)) * inv_sum;
        row_out[i_out] = static_cast<scalar_t>(p);
      }
    }
  }

 private:
  const scalar_t* input_;
  scalar_t* output_;
  int dim_size_;
  int wg_size_;
  float scale_log2e_;
  bool can_vec_;
  sycl::local_accessor<float, 1> local_max_;
  sycl::local_accessor<float, 1> local_sum_;
};

template <typename scalar_t>
void launch_softmax_kernel(torch::Tensor& out, const torch::Tensor& input) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  const int batch = static_cast<int>(input.size(0));
  const int dim_size = static_cast<int>(input.size(1));

  auto* d_in = reinterpret_cast<const sycl_t*>(input.data_ptr<scalar_t>());
  auto* d_out = reinterpret_cast<sycl_t*>(out.data_ptr<scalar_t>());

  auto& q = vllm::xpu::vllmGetQueue();
  const int device_wg_max = static_cast<int>(
      q.get_device().get_info<sycl::info::device::max_work_group_size>());

  const int wg_size = pick_softmax_wg_size<sycl_t>(dim_size, device_wg_max);
  const float scale_log2e = static_cast<float>(1.4426950408889634);

  sycl::range<1> local_range(static_cast<std::size_t>(wg_size));
  sycl::range<1> global_range(static_cast<std::size_t>(batch) * local_range[0]);

  constexpr int kRowCacheVec = 4;  // Keep 4 for row cache kernel
  constexpr std::size_t kRowCacheVecBytes = kRowCacheVec * sizeof(scalar_t);

  constexpr std::size_t kRowCacheMinBytes = 8192;
  const std::size_t row_cache_min_dim = kRowCacheMinBytes / sizeof(scalar_t);

  const bool can_row_cache_vec =
      static_cast<std::size_t>(dim_size) >= row_cache_min_dim &&
      ((reinterpret_cast<std::uintptr_t>(d_in) & (kRowCacheVecBytes - 1)) ==
       0) &&
      ((reinterpret_cast<std::uintptr_t>(d_out) & (kRowCacheVecBytes - 1)) ==
       0) &&
      ((dim_size & (kRowCacheVec - 1)) == 0) &&
      (dim_size >= kRowCacheVec * wg_size);

  const std::size_t local_mem_size = static_cast<std::size_t>(
      q.get_device().get_info<sycl::info::device::local_mem_size>());
  const std::size_t reduction_bytes =
      2 * static_cast<std::size_t>(wg_size) * sizeof(float);
  const std::size_t row_cache_bytes =
      static_cast<std::size_t>(dim_size) * sizeof(float);
  constexpr std::size_t kSlmReserve =
      2048;  // headroom for driver/runtime SLM usage
  // Cap far below the device-reported max: letting one workgroup claim most
  // of a Xe-core's SLM collapses how many workgroups can be co-resident on
  // that core, which regresses large dim_size despite the reduced HBM
  // traffic. This keeps small/medium rows cached while excluding the sizes
  // that regress.
  constexpr std::size_t kRowCacheSlmCap = 49152;
  const std::size_t slm_budget = std::min(local_mem_size, kRowCacheSlmCap);
  const bool use_row_cache =
      can_row_cache_vec && slm_budget > kSlmReserve &&
      (reduction_bytes + row_cache_bytes <= slm_budget - kSlmReserve);

  if (use_row_cache) {
    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> local_max(local_range, cgh);
      sycl::local_accessor<float, 1> local_sum(local_range, cgh);
      sycl::local_accessor<float, 1> row_cache(
          sycl::range<1>(static_cast<std::size_t>(dim_size)), cgh);
      vllm::softmax_row_cache_kernel<sycl_t> kfn(
          d_in,
          d_out,
          dim_size,
          wg_size,
          scale_log2e,
          local_max,
          local_sum,
          row_cache);
      cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), kfn);
    });
    return;
  }

  constexpr int kVec = softmax_vec_size<sycl_t>();
  const bool can_vec =
      ((reinterpret_cast<std::uintptr_t>(d_in) & (kSoftmaxVecBytes - 1)) ==
       0) &&
      ((reinterpret_cast<std::uintptr_t>(d_out) & (kSoftmaxVecBytes - 1)) ==
       0) &&
      ((dim_size & (kVec - 1)) == 0) && (dim_size >= kVec * wg_size);

  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> local_max(local_range, cgh);
    sycl::local_accessor<float, 1> local_sum(local_range, cgh);
    vllm::softmax_kernel<sycl_t, kVec> kfn(
        d_in,
        d_out,
        dim_size,
        wg_size,
        scale_log2e,
        can_vec,
        local_max,
        local_sum);
    cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), kfn);
  });
}

}  // namespace vllm

void softmax(torch::Tensor& out, torch::Tensor& input) {
  const at::DeviceGuard device_guard(input.device());
  CHECK_DEVICE(input);
  CHECK_DEVICE(out);
  TORCH_CHECK(
      input.device() == out.device(),
      "input/output must be on the same device");
  TORCH_CHECK(
      input.dim() == 2 && out.dim() == 2,
      "input/output must be 2D [batch, dim]");
  TORCH_CHECK(
      input.size(0) == out.size(0) && input.size(1) == out.size(1),
      "input/output shape mismatch");
  TORCH_CHECK(
      input.is_contiguous() && out.is_contiguous(),
      "input/output must be contiguous");
  TORCH_CHECK(
      input.scalar_type() == out.scalar_type(), "input/output dtype mismatch");
  TORCH_CHECK(
      input.size(0) > 0 && input.size(1) > 0,
      "batch/dim_size must be positive");

  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softmax", [&] {
    vllm::launch_softmax_kernel<scalar_t>(out, input);
  });
}
