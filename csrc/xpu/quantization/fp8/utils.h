#pragma once

#include <sycl/sycl.hpp>
#include <cmath>

#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <torch/all.h>

namespace vllm {

template <typename scalar_t>
struct alignas(8) vec4_t {
  scalar_t x;
  scalar_t y;
  scalar_t z;
  scalar_t w;
};

template <typename quant_type_t>
struct alignas(4) q8x4_t {
  static_assert(
      std::is_same_v<quant_type_t, c10::Float8_e5m2> ||
      std::is_same_v<quant_type_t, c10::Float8_e4m3fn>);
  quant_type_t x;
  quant_type_t y;
  quant_type_t z;
  quant_type_t w;
};

template <
    typename T,
    typename = std::enable_if_t<
        std::is_same_v<T, c10::Float8_e5m2> ||
        std::is_same_v<T, c10::Float8_e4m3fn>>>
struct quant_type_max {
  static constexpr T val() {
    return std::numeric_limits<T>::max();
  }
};

template <typename T>
static constexpr T quant_type_max_v = quant_type_max<T>::val();

template <
    typename T,
    typename = std::enable_if_t<
        std::is_same_v<T, c10::Float8_e5m2> ||
        std::is_same_v<T, c10::Float8_e4m3fn>>>
struct min_scaling_factor {
  static inline float val() {
    return 1.0f / (quant_type_max_v<T> * 512.0f);
  }
};

} // namespace vllm 
