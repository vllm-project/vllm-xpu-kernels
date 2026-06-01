// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Provides ScaledQuant<quant_type_t, is_scale_inverted> for use in XPU
// fused quantization kernels.

#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <type_traits>
#include <torch/all.h>
#include "quantization/fp8/quant_utils.h"

namespace vllm {

// ---------------------------------------------------------------------------
// float -> int8  (round-nearest, saturate to [-128, 127])
// ---------------------------------------------------------------------------
static inline int8_t float_to_int8_rn(float const x) {
  return static_cast<int8_t>(sycl::clamp(sycl::rint(x), -128.0f, 127.0f));
}

// ---------------------------------------------------------------------------
// float -> fp8  (clamp to [-fp8_max, fp8_max], then cast)
// ---------------------------------------------------------------------------
template <typename fp8_type>
static inline fp8_type float_to_fp8(float const x) {
  const float fp8_max = static_cast<float>(fp8::quant_type_max_v<fp8_type>);
  return static_cast<fp8_type>(sycl::fmax(-fp8_max, sycl::fmin(x, fp8_max)));
}

// ---------------------------------------------------------------------------
// ScaledQuant<quant_type_t, is_scale_inverted>
//
//   is_scale_inverted = false  →  quant_fn(x, scale) computes x / scale
//   is_scale_inverted = true   →  quant_fn(x, scale) computes x * scale
//                                 (scale is already the inverted scale)
// ---------------------------------------------------------------------------

template <typename quant_type_t, bool is_scale_inverted, typename Enable = void>
struct ScaledQuant;

// int8 specialization
template <typename quant_type_t, bool is_scale_inverted>
struct ScaledQuant<
    quant_type_t,
    is_scale_inverted,
    std::enable_if_t<std::is_same_v<quant_type_t, int8_t>>> {
  static inline int8_t quant_fn(float const x, float const scale) {
    if constexpr (is_scale_inverted) {
      return float_to_int8_rn(x * scale);
    } else {
      return float_to_int8_rn(x / scale);
    }
  }
};

// FP8 (e4m3fn and e5m2) specialization
template <typename quant_type_t, bool is_scale_inverted>
struct ScaledQuant<
    quant_type_t,
    is_scale_inverted,
    std::enable_if_t<
        std::is_same_v<quant_type_t, at::Float8_e4m3fn> ||
        std::is_same_v<quant_type_t, at::Float8_e5m2>>> {
  static inline quant_type_t quant_fn(float const x, float const scale) {
    if constexpr (is_scale_inverted) {
      return float_to_fp8<quant_type_t>(x * scale);
    } else {
      return float_to_fp8<quant_type_t>(x / scale);
    }
  }
};

}  // namespace vllm
