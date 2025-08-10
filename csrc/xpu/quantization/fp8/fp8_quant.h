#pragma once

#include <sycl/sycl.hpp>
#include <cmath>

#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>

#include "utils.h"

using namespace at;

namespace vllm {

template <bool is_scale_inverted, typename fp8_type>
inline fp8_type scaled_fp8_conversion(
    float const val,
    float const scale,
    int tid = 0,
    int token_id = 0) {
  float x = 0.0f;
  if constexpr (is_scale_inverted) {
    x = val * scale;
  } else {
    x = val / scale;
  }

  const float fp8_max = static_cast<float>(quant_type_max_v<fp8_type>);
  float r = sycl::fmax(-fp8_max, sycl::fmin(x, fp8_max));
  // sycl::ext::oneapi::experimental::printf(
  //     "token_id: %d tid: %d val: %f, scale: %f, x: %f,
  //     quant_type_max_v<fp8_type>: %f, r: %f\n", token_id, tid, val, scale, x,
  //     static_cast<float>(quant_type_max_v<fp8_type>),
  //     r);
  // sycl::ext::oneapi::experimental::printf("scaled_fp8_conversion: %f\n", r);
  return static_cast<fp8_type>(r);
}

template <typename scalar_t, bool is_scale_inverted, typename fp8_type>
inline void scaled_fp8_conversion_vec(
    fp8_type* out,
    scalar_t const* input,
    float const scale,
    int64_t const num_elems,
    int const tid,
    int const step) {
  using float8x4_t = q8x4_t<fp8_type>;
  // Vectorized input/output to better utilize memory bandwidth.
  auto const* vectorized_in = reinterpret_cast<vec4_t<scalar_t> const*>(input);
  auto* vectorized_out = reinterpret_cast<float8x4_t*>(out);

  int64_t const num_vec_elems = num_elems >> 2;

#pragma unroll 4
  for (int64_t i = tid; i < num_vec_elems; i += step) {
    vec4_t<scalar_t> in_vec = vectorized_in[i];
    float8x4_t out_vec;

    out_vec.x = scaled_fp8_conversion<is_scale_inverted, fp8_type>(
        static_cast<float>(in_vec.x), scale);
    out_vec.y = scaled_fp8_conversion<is_scale_inverted, fp8_type>(
        static_cast<float>(in_vec.y), scale);
    out_vec.z = scaled_fp8_conversion<is_scale_inverted, fp8_type>(
        static_cast<float>(in_vec.z), scale);
    out_vec.w = scaled_fp8_conversion<is_scale_inverted, fp8_type>(
        static_cast<float>(in_vec.w), scale);
    vectorized_out[i] = out_vec;
  }

  // Handle the remaining elements if num_elems is not divisible by 4
  for (int64_t i = num_vec_elems * 4 + tid; i < num_elems; i += step) {
    out[i] = scaled_fp8_conversion<is_scale_inverted, fp8_type>(
        static_cast<float>(input[i]), scale);
  }
}

template <typename scalar_t>
inline float thread_max_vec(
    scalar_t const* input,
    int64_t const num_elems,
    int const tid,
    int const step) {
  // Vectorized input/output to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vectorized_in =
      reinterpret_cast<vec4_t<scalar_t> const*>(input);

  int64_t const num_vec_elems = num_elems >> 2;
  float absmax_val = 0.0f;

#pragma unroll 4
  for (int64_t i = tid; i < num_vec_elems; i += step) {
    vec4_t<scalar_t> in_vec = vectorized_in[i];
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
class segmented_max_reduction {
 private:
  float* scale;
  const scalar_t* input;
  int64_t num_elems;

 public:
  segmented_max_reduction(
      float* scale_,
      const scalar_t* input_,
      int64_t num_elems_)
      : scale(scale_), input(input_), num_elems(num_elems_) {}
  void operator()(sycl::nd_item<1> item) const {
    auto& cache =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1024]>(
            item.get_group());
    int64_t i = item.get_global_linear_id();

    // First store maximum for all values processes by
    // the current thread in cache[item.get_local_id(0)]
    float tmp = 0.0;
    while (i < num_elems) {
      float x = static_cast<float>(input[i]);
      tmp = sycl::max(tmp, sycl::fabs(x));
      i += item.get_local_range(0) * item.get_group_range(0);
    }
    cache[item.get_local_id(0)] = tmp;

    group_barrier(item.get_group());

    // Now perform parallel reduction within the thread block
    int ib = item.get_local_range(0) / 2;
    while (ib != 0) {
      if (item.get_local_id(0) < ib &&
          cache[item.get_local_id(0) + ib] > cache[item.get_local_id(0)]) {
        cache[item.get_local_id(0)] = cache[item.get_local_id(0) + ib];
      }
      group_barrier(item.get_group());
      ib /= 2;
    }
    // Finally, since cache[0] contains the maximum for this thread block,
    // atomically write the max to the target location
    // TODO: Do we need if statement?
    if (item.get_local_id(0) == 0) {
      using atomic_t = sycl::atomic_ref<
          float,
          sycl::memory_order::relaxed,
          sycl::memory_scope::device,
          sycl::access::address_space::global_space>;
      atomic_t atomic_max(*scale);
      atomic_max.fetch_max(cache[0] / quant_type_max_v<fp8_type>);
    }
  }
};

} // namespace vllm
