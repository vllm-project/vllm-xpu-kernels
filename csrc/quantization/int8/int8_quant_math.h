// SPDX-License-Identifier: Apache-2.0
// Scalar math of the dynamic per-token symmetric int8 quant.
//
// Bit-for-bit mirror of vLLM's dynamic_per_token_int8_quant_ref
// (vllm/_xpu_ops.py)
// (use_sym_quant=True, bits=8):
//   absmax = max(|min(row)|, |max(row)|)                 (exact in fp32)
//   scale  = clamp(absmax / 127, min=1e-5)               (fp32, IEEE div)
//   scale  = scale.to(input.dtype)                       (RNE, integer ops)
//   q      = clamp(round_half_even(x_fp32 / scale_fp32), -128, 127)
// For 16-bit input the rounding uses an exact tie test; see quantize_one.
// Header-only and free of SYCL so the same code is compiled into the device
// kernel and into the host-side CPU check.
#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

// icpx defaults to -ffp-model=fast and SYCL fp32 division to 2.5 ulp; either
// would break the exact match. The device TU (int8_quant.cpp) is therefore
// compiled with -ffp-model=precise -foffload-fp32-prec-div (CMakeLists.txt);
// '#pragma float_control' is ignored for spir64 and cannot do it.

namespace vllm {
namespace int8q {

inline uint32_t f32_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

inline float f32_from_bits(uint32_t u) {
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

// Round half to even, written out. Exact for |t| < 2^23.
inline float rint_half_even(float t) {
  const float f = __builtin_floorf(t);
  const float d = t - f;  // exact: t and f share the exponent range
  if (d > 0.5f) return f + 1.0f;
  if (d < 0.5f) return f;
  const float half_f = f * 0.5f;
  return (__builtin_floorf(half_f) == half_f) ? f : f + 1.0f;
}

// float -> bf16 -> float, round to nearest even, in integer arithmetic.
inline float round_to_bf16(float s) {
  if (s != s) return s;  // NaN
  uint32_t u = f32_bits(s);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return f32_from_bits(u & 0xFFFF0000u);
}

// float -> fp16 -> float, round to nearest even, for finite s >= 0 in fp16
// range (the scale here is in [1e-5, 65504 / 127]). Below 2^-14 fp16 is
// subnormal with a fixed quantum of 2^-24.
inline float round_to_fp16(float s) {
  if (s != s) return s;        // NaN
  if (s < 6.103515625e-05f) {  // 2^-14
    return rint_half_even(s * 16777216.0f) * 5.9604644775390625e-08f;
  }
  uint32_t u = f32_bits(s);
  u += 0xFFFu + ((u >> 13) & 1u);
  return f32_from_bits(u & 0xFFFFE000u);
}

// scale.to(input.dtype). A static_cast through c10::Half on the device did not
// round (measured on Xe2: the f32 scale came out unrounded for fp16 input), so
// the rounding is spelled out for both 16-bit types.
template <typename scalar_t>
inline float round_to_dtype(float s) {
  if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
    return round_to_bf16(s);
  } else if constexpr (std::is_same_v<scalar_t, c10::Half>) {
    return round_to_fp16(s);
  } else {
    return s;
  }
}

template <typename scalar_t>
inline float token_scale_from_absmax(float absmax) {
  float s = absmax / 127.0f;
  // torch.clamp(min=1e-5): NaN stays NaN.
  s = (s < 1e-5f) ? 1e-5f : s;
  return round_to_dtype<scalar_t>(s);
}

// x / scale, rounded half to even, against the exact quotient. Measured on
// Xe2: quotients that are exactly q + 0.5 came out truncated toward zero, also
// with explicit half-even rounding of the quotient and with
// -foffload-fp32-prec-div, consistent with a device quotient just under the
// tie. So ties are not decided from the quotient. For 16-bit x and scale (at
// most 11 significant bits each) (f + 0.5) * scale is exact in fp32, so
// comparing it with x decides the rounding exactly, and an inexact quotient is
// never close enough to a tie for an IEEE fp32 division to round onto it. The
// quotient only has to pick the right f.
template <typename scalar_t>
inline int8_t quantize_one(float x, float scale) {
  const float t = x / scale;
  float q;
  if constexpr (
      std::is_same_v<scalar_t, c10::BFloat16> ||
      std::is_same_v<scalar_t, c10::Half>) {
    const float f = __builtin_floorf(t);
    const float boundary = (f + 0.5f) * scale;
    if (x > boundary) {
      q = f + 1.0f;
    } else if (x < boundary) {
      q = f;
    } else {
      const float half_f = f * 0.5f;
      q = (__builtin_floorf(half_f) == half_f) ? f : f + 1.0f;
    }
  } else {
    q = rint_half_even(t);
  }
  q = q < -128.0f ? -128.0f : q;
  q = q > 127.0f ? 127.0f : q;
  return static_cast<int8_t>(q);
}

inline float abs_f(float x) { return __builtin_fabsf(x); }
inline float max_f(float a, float b) { return a > b ? a : b; }

}  // namespace int8q
}  // namespace vllm
