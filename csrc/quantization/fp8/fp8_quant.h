#pragma once

#include <sycl/sycl.hpp>
#include <cmath>

#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>

#include "quantization/fp8/quant_utils.h"

using namespace at;

namespace vllm {

template <typename scalar_t>
inline float thread_max_vec(scalar_t const* input, int64_t const num_elems,
                            int const tid, int const step) {
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
class segmented_max_reduction {
 private:
  float* scale;
  const scalar_t* input;
  int64_t num_elems;

 public:
  segmented_max_reduction(float* scale_, const scalar_t* input_,
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
      using atomic_t =
          sycl::atomic_ref<float, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>;
      atomic_t atomic_max(*scale);
      atomic_max.fetch_max(cache[0] / fp8::quant_type_max_v<fp8_type>);
    }
  }
};

}  // namespace vllm
