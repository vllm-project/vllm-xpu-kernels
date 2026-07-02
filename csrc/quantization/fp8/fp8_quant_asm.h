#pragma once

#include <cmath>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Half.h>

#include "quantization/fp8/quant_utils.h"
#include "quantization/utils.h"

namespace vllm {
namespace fp8 {

#if defined(__SYCL_DEVICE_ONLY__) && defined(VLLM_XPU_ENABLE_XE3) && \
    !defined(VLLM_FP8_QUANT_DISABLE_ASM)
#define VLLM_FP8_QUANT_ASM_ENABLED 1
#endif

template <typename Tin, typename Tout>
inline Tout scaled_quant_asm(Tin const& src, float inv_scale) {
  float x = static_cast<float>(src) * inv_scale;
  const float fp8_max = static_cast<float>(quant_type_max_v<Tout>);
  x = sycl::fmax(-fp8_max, sycl::fmin(x, fp8_max));
  return static_cast<Tout>(x);
}

#ifdef VLLM_FP8_QUANT_ASM_ENABLED


inline at::Float8_e4m3fn fcvt_half_e4m3(sycl::half h) {
  uint8_t dst;
  asm("{\n"
      ".decl IN_HF v_type=G type=HF num_elts=32 alias=<%1,0>\n"
      ".decl OUT_B v_type=G type=B  num_elts=32 alias=<%0,0>\n"
      "fcvt (M1_NM, 32) OUT_B(0,0)<1>  IN_HF(0,0)<1;1,0>\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(h));
  return sycl::bit_cast<at::Float8_e4m3fn>(dst);
}

inline at::Float8_e5m2 fcvt_half_e5m2(sycl::half h) {
  uint8_t dst;
  asm("{\n"
      ".decl IN_HF  v_type=G type=HF num_elts=32 alias=<%1,0>\n"
      ".decl OUT_UB v_type=G type=UB num_elts=32 alias=<%0,0>\n"
      "fcvt (M1_NM, 32) OUT_UB(0,0)<1>  IN_HF(0,0)<1;1,0>\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(h));
  return sycl::bit_cast<at::Float8_e5m2>(dst);
}

template <typename Tout>
inline sycl::half fp8_scale_clamp_half(float x, float inv_scale) {
  x *= inv_scale;
  const float fp8_max = static_cast<float>(quant_type_max_v<Tout>);
  x = sycl::fmax(-fp8_max, sycl::fmin(x, fp8_max));
  return static_cast<sycl::half>(x);
}

// half -> e4m3
template <>
inline at::Float8_e4m3fn scaled_quant_asm<at::Half, at::Float8_e4m3fn>(
    at::Half const& src, float inv_scale) {
  return fcvt_half_e4m3(
      fp8_scale_clamp_half<at::Float8_e4m3fn>(static_cast<float>(src),
                                              inv_scale));
}

// half -> e5m2
template <>
inline at::Float8_e5m2 scaled_quant_asm<at::Half, at::Float8_e5m2>(
    at::Half const& src, float inv_scale) {
  return fcvt_half_e5m2(
      fp8_scale_clamp_half<at::Float8_e5m2>(static_cast<float>(src),
                                            inv_scale));
}

// bfloat16 -> e4m3
template <>
inline at::Float8_e4m3fn scaled_quant_asm<at::BFloat16, at::Float8_e4m3fn>(
    at::BFloat16 const& src, float inv_scale) {
  return fcvt_half_e4m3(
      fp8_scale_clamp_half<at::Float8_e4m3fn>(static_cast<float>(src),
                                              inv_scale));
}

// bfloat16 -> e5m2
template <>
inline at::Float8_e5m2 scaled_quant_asm<at::BFloat16, at::Float8_e5m2>(
    at::BFloat16 const& src, float inv_scale) {
  return fcvt_half_e5m2(
      fp8_scale_clamp_half<at::Float8_e5m2>(static_cast<float>(src),
                                            inv_scale));
}

// float -> e4m3
template <>
inline at::Float8_e4m3fn scaled_quant_asm<float, at::Float8_e4m3fn>(
    float const& src, float inv_scale) {
  return fcvt_half_e4m3(fp8_scale_clamp_half<at::Float8_e4m3fn>(src, inv_scale));
}

// float -> e5m2
template <>
inline at::Float8_e5m2 scaled_quant_asm<float, at::Float8_e5m2>(
    float const& src, float inv_scale) {
  return fcvt_half_e5m2(fp8_scale_clamp_half<at::Float8_e5m2>(src, inv_scale));
}

#endif  // VLLM_FP8_QUANT_ASM_ENABLED

template <bool is_scale_inverted, typename scalar_t, typename fp8_type>
struct ScaledQuantOpAsm {
  float scale;
  inline void operator()(fp8_type& dst, scalar_t const& src) const {
    float inv_scale = is_scale_inverted ? scale : (1.0f / scale);
    dst = scaled_quant_asm<scalar_t, fp8_type>(src, inv_scale);
  }
};

}  // namespace fp8
}  // namespace vllm
