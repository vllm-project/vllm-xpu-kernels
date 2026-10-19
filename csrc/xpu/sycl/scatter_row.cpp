// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Row-wise scatter: dst[index[i], :] = src[i, :] for each row i, using
// 16-byte-wide vector loads/stores (dtype-agnostic byte copy).
// Ported from xpu-perf's projects/micro_perf/vendor_ops/INTEL/ops/sycl_ext/
// scatter_kernel.cpp.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>

#include <ATen/DeviceGuard.h>

#include "utils.h"
#include "dispatch_utils.h"
#include "sycl_launch.h"

namespace vllm {

constexpr int kScatterRowSimd = 32;
constexpr int kScatterRowAlignment = 16;

// 128-bit vector split into two 64-bit halves for the store — better
// codegen on Xe2 than a single 128-bit store.
struct ScatterRowVec128 {
  uint64_t lo, hi;
};

template <int N>
struct ScatterRowVecType {};
template <>
struct ScatterRowVecType<16> {
  using type = ScatterRowVec128;
};
template <>
struct ScatterRowVecType<8> {
  using type = uint64_t;
};
template <>
struct ScatterRowVecType<4> {
  using type = uint32_t;
};

template <int N>
inline typename ScatterRowVecType<N>::type
scatter_row_ld_vec(const char* addr) {
  using vec_t = typename ScatterRowVecType<N>::type;
  vec_t v;
  std::memcpy(&v, addr, sizeof(vec_t));
  return v;
}

template <int N>
inline void
scatter_row_st_vec(char* addr, typename ScatterRowVecType<N>::type v) {
  if constexpr (N == 16) {
    *reinterpret_cast<uint64_t*>(addr) = v.lo;
    *reinterpret_cast<uint64_t*>(addr + 8) = v.hi;
  } else {
    std::memcpy(addr, &v, sizeof(v));
  }
}

// One row per group-1 index; the row is split into ALIGNMENT-byte chunks
// spread across group-0 x local-1 so a work-group of `wg_size` lanes covers
// `wg_size * ALIGNMENT` bytes per step.
class scatter_row_kernel {
 public:
  scatter_row_kernel(
      char* out,
      const char* input,
      const int64_t* idx,
      int64_t slice_bytes,
      int64_t input_stride,
      int64_t out_stride)
      : out_(out),
        input_(input),
        idx_(idx),
        slice_bytes_(slice_bytes),
        input_stride_(input_stride),
        out_stride_(out_stride) {}

  void operator() [[sycl::reqd_sub_group_size(kScatterRowSimd)]] (
      sycl::nd_item<2> item) const {
    int64_t dst_row = idx_[item.get_group(1)];

    int64_t off = (static_cast<int64_t>(item.get_local_range(1)) *
                       static_cast<int64_t>(item.get_group(0)) +
                   static_cast<int64_t>(item.get_local_id(1))) *
                  kScatterRowAlignment;
    if (off >= slice_bytes_) return;

    int64_t input_offset =
        static_cast<int64_t>(item.get_group(1)) * input_stride_;
    auto vec =
        scatter_row_ld_vec<kScatterRowAlignment>(input_ + input_offset + off);
    scatter_row_st_vec<kScatterRowAlignment>(
        out_ + dst_row * out_stride_ + off, vec);
  }

 private:
  char* out_;
  const char* input_;
  const int64_t* idx_;
  int64_t slice_bytes_;
  int64_t input_stride_;
  int64_t out_stride_;
};

inline void launch_scatter_row_kernel(
    sycl::queue& q,
    char* out,
    const char* input,
    const int64_t* idx,
    int64_t num_indices,
    int64_t slice_bytes,
    int64_t input_stride,
    int64_t out_stride) {
  constexpr int64_t kMaxThreads = 512;
  int64_t num_threads_needed =
      (slice_bytes + kScatterRowAlignment - 1) / kScatterRowAlignment;
  // Round up to a multiple of the sub-group size.
  num_threads_needed =
      ((num_threads_needed + kScatterRowSimd - 1) / kScatterRowSimd) *
      kScatterRowSimd;
  const int64_t wg_size = std::min(kMaxThreads, num_threads_needed);

  const int64_t num_groups_dim0 =
      (slice_bytes + wg_size * kScatterRowAlignment - 1) /
      (wg_size * kScatterRowAlignment);

  sycl::range<2> local_range(1, static_cast<std::size_t>(wg_size));
  sycl::range<2> global_range(
      static_cast<std::size_t>(num_groups_dim0),
      static_cast<std::size_t>(num_indices) *
          static_cast<std::size_t>(wg_size));

  scatter_row_kernel kfn(
      out, input, idx, slice_bytes, input_stride, out_stride);
  vllm::xpu::sycl_kernel_submit(
      q, sycl::nd_range<2>(global_range, local_range), kfn);
}

// Row-copy is only vectorizable when every base pointer/stride/slice size
// lands on an ALIGNMENT-byte boundary.
inline bool scatter_row_vectorized_eligible(
    const char* dst_ptr,
    const char* src_ptr,
    int64_t slice_bytes,
    int64_t src_stride,
    int64_t dst_stride) {
  auto check_align = [](uintptr_t v) {
    return (v % kScatterRowAlignment) == 0;
  };
  return check_align(reinterpret_cast<uintptr_t>(dst_ptr)) &&
         check_align(reinterpret_cast<uintptr_t>(src_ptr)) &&
         check_align(static_cast<uintptr_t>(slice_bytes)) &&
         check_align(static_cast<uintptr_t>(src_stride)) &&
         check_align(static_cast<uintptr_t>(dst_stride));
}

}  // namespace vllm

// Row-wise scatter: dst[index[i], :] = src[i, :] for i in [0, index.size(0)).
// dtype-agnostic (plain byte copy); dim_size * element_size must be a
// multiple of 16 bytes, and index must be int64.
void scatter_row(
    torch::Tensor& dst, const torch::Tensor& src, const torch::Tensor& index) {
  const at::DeviceGuard device_guard(src.device());
  CHECK_DEVICE(dst);
  CHECK_DEVICE(src);
  CHECK_DEVICE(index);
  TORCH_CHECK(
      dst.dim() == 2 && src.dim() == 2, "dst/src must be 2D [batch, dim]");
  TORCH_CHECK(index.dim() == 1, "index must be 1D [src_batch]");
  TORCH_CHECK(
      dst.is_contiguous() && src.is_contiguous() && index.is_contiguous(),
      "dst/src/index must be contiguous");
  TORCH_CHECK(
      index.scalar_type() == at::ScalarType::Long, "index must be int64");
  TORCH_CHECK(dst.scalar_type() == src.scalar_type(), "dst/src dtype mismatch");
  TORCH_CHECK(
      dst.device() == src.device() && dst.device() == index.device(),
      "dst/src/index must be on the same device");

  const int64_t num_indices = index.size(0);
  TORCH_CHECK(
      num_indices == src.size(0), "index length must match src.size(0)");
  const int64_t dim_size = src.size(1);
  TORCH_CHECK(dst.size(1) == dim_size, "dst/src dim_size mismatch");

  const int64_t element_size = dst.element_size();
  const int64_t slice_bytes = dim_size * element_size;
  const int64_t src_stride = slice_bytes;
  const int64_t dst_stride = slice_bytes;

  char* dst_ptr = reinterpret_cast<char*>(dst.data_ptr());
  const char* src_ptr = reinterpret_cast<const char*>(src.const_data_ptr());
  const int64_t* idx_ptr = index.const_data_ptr<int64_t>();

  TORCH_CHECK(
      vllm::scatter_row_vectorized_eligible(
          dst_ptr, src_ptr, slice_bytes, src_stride, dst_stride),
      "scatter_row: alignment check failed (slice_bytes=",
      slice_bytes,
      "); dim_size * element_size must be a multiple of ",
      vllm::kScatterRowAlignment);

  if (num_indices == 0 || slice_bytes == 0) return;

  auto& q = vllm::xpu::vllmGetQueue();
  vllm::launch_scatter_row_kernel(
      q,
      dst_ptr,
      src_ptr,
      idx_ptr,
      num_indices,
      slice_bytes,
      src_stride,
      dst_stride);
}
