#pragma once

#include <sycl/sycl.hpp>
#include <cmath>

#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>

#include "quantization/fp8/quant_utils.h"

using namespace at;

namespace vllm {

template <typename scalar_t>
inline float thread_max_vec(
    scalar_t const* input,
    int64_t const num_elems,
    int const tid,
    int const step) {
  // Vectorized input/output to better utilize memory bandwidth.
  using vec4_t = fp8::vec4_t<scalar_t>;
  vec4_t const* vectorized_in = reinterpret_cast<vec4_t const*>(input);

  int64_t const num_vec_elems = num_elems >> 2;
  float absmax_val = 0.0f;

#pragma unroll 4
  for (int64_t i = tid; i < num_vec_elems; i += step) {
    vec4_t in_vec = vectorized_in[i];
    absmax_val =
        sycl::max(absmax_val, sycl::fabs(static_cast<float>(in_vec.x)));
    absmax_val =
        sycl::max(absmax_val, sycl::fabs(static_cast<float>(in_vec.y)));
    absmax_val =
        sycl::max(absmax_val, sycl::fabs(static_cast<float>(in_vec.z)));
    absmax_val =
        sycl::max(absmax_val, sycl::fabs(static_cast<float>(in_vec.w)));
  }

  // Handle the remaining elements if num_elems is not divisible by 4
  for (int64_t i = num_vec_elems * 4 + tid; i < num_elems; i += step) {
    absmax_val =
        sycl::max(absmax_val, sycl::fabs(static_cast<float>(input[i])));
  }

  return absmax_val;
}

template <typename scalar_t, typename fp8_type>
class segmented_max_reduction_strided {
 private:
  float* scale;
  const scalar_t* input;
  int64_t hidden_size;
  int64_t in_row_stride;
  int64_t num_tokens;

 public:
  segmented_max_reduction_strided(
      float* scale_,
      const scalar_t* input_,
      int64_t hidden_size_,
      int64_t in_row_stride_,
      int64_t num_tokens_)
      : scale(scale_),
        input(input_),
        hidden_size(hidden_size_),
        in_row_stride(in_row_stride_),
        num_tokens(num_tokens_) {}
  void operator()(sycl::nd_item<1> item) const {
    // NOTE: `scale` must be initialized before lanching the reduction kernel.
    auto& cache =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1024]>(
            item.get_group());
    const int tid = item.get_local_id(0);
    int64_t token_idx = item.get_group(0);

    // one block per token. Guard in case gridDim.x > num_tokens.
    if (token_idx >= num_tokens) {
      return;
    }

    const scalar_t* row_ptr = input + token_idx * in_row_stride;

    // each thread scans elements of the row in a strided fashion.
    float thread_max = 0.0f;
    for (int e = tid; e < hidden_size; e += item.get_local_range(0)) {
      float x = static_cast<float>(row_ptr[e]);
      thread_max = sycl::max(thread_max, sycl::fabs(x));
    }
    cache[tid] = thread_max;
    group_barrier(item.get_group());

    // parallel reduction to find row max.
    for (int offset = item.get_local_range(0) / 2; offset > 0; offset >>= 1) {
      if (tid < offset) {
        cache[tid] = sycl::max(cache[tid], cache[tid + offset]);
      }
      group_barrier(item.get_group());
    }

    // thread 0 updates global scale (per-tensor) atomically.
    if (tid == 0) {
      using atomic_t = sycl::atomic_ref<
          float,
          sycl::memory_order::relaxed,
          sycl::memory_scope::device,
          sycl::access::address_space::global_space>;
      atomic_t atomic_max(*scale);
      atomic_max.fetch_max(cache[0] / fp8::quant_type_max_v<fp8_type>);
    }
  }
};

}  // namespace vllm
