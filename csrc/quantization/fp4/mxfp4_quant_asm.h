#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

namespace vllm {
namespace mxfp4 {

static constexpr float kMxFp4Max = 6.0f;

#if defined(__SYCL_DEVICE_ONLY__) && defined(VLLM_XPU_ENABLE_XE3) && \
    !defined(VLLM_MXFP4_QUANT_DISABLE_ASM)
#define VLLM_MXFP4_QUANT_ASM_ENABLED 1
#endif

template <typename scalar_t>
inline float mxfp4_scale_clamp_asm(scalar_t const& src, float inv_scale) {
  float x = static_cast<float>(src) * inv_scale;
  return sycl::fmax(-kMxFp4Max, sycl::fmin(x, kMxFp4Max));
}

#ifdef VLLM_MXFP4_QUANT_ASM_ENABLED

// half -> sat(half*scale, +/-6) -> float
template <>
inline float mxfp4_scale_clamp_asm<at::Half>(
    at::Half const& src, float inv_scale) {
  sycl::half a = sycl::bit_cast<sycl::half>(src);
  float dst;
  asm("{\n"
      ".decl IN_HF   v_type=G type=HF num_elts=16 alias=<%1,0>\n"
      ".decl OUT_F   v_type=G type=F  num_elts=16 alias=<%0,0>\n"
      ".decl SCALE_F v_type=G type=F  num_elts=16 alias=<%2,0>\n"
      ".decl TMP_F   v_type=G type=F  num_elts=16 align=32\n"
      "mov  (M1_NM, 16) TMP_F(0,0)<1>  IN_HF(0,0)<1;1,0>\n"
      "mul  (M1_NM, 16) TMP_F(0,0)<1>  TMP_F(0,0)<1;1,0>  SCALE_F(0,0)<0;1,0>\n"
      "min  (M1_NM, 16) TMP_F(0,0)<1>  TMP_F(0,0)<1;1,0>  0x40C00000:f\n"
      "max  (M1_NM, 16) OUT_F(0,0)<1>  TMP_F(0,0)<1;1,0>  0xC0C00000:f\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(a), "rw"(inv_scale));
  return dst;
}

// bfloat16 -> sat(bf16*scale, +/-6) -> float
template <>
inline float mxfp4_scale_clamp_asm<at::BFloat16>(
    at::BFloat16 const& src, float inv_scale) {
  sycl::ext::oneapi::bfloat16 a =
      sycl::bit_cast<sycl::ext::oneapi::bfloat16>(src);
  float dst;
  asm("{\n"
      ".decl IN_BF   v_type=G type=BF num_elts=16 alias=<%1,0>\n"
      ".decl OUT_F   v_type=G type=F  num_elts=16 alias=<%0,0>\n"
      ".decl SCALE_F v_type=G type=F  num_elts=16 alias=<%2,0>\n"
      ".decl TMP_F   v_type=G type=F  num_elts=16 align=32\n"
      "mov  (M1_NM, 16) TMP_F(0,0)<1>  IN_BF(0,0)<1;1,0>\n"
      "mul  (M1_NM, 16) TMP_F(0,0)<1>  TMP_F(0,0)<1;1,0>  SCALE_F(0,0)<0;1,0>\n"
      "min  (M1_NM, 16) TMP_F(0,0)<1>  TMP_F(0,0)<1;1,0>  0x40C00000:f\n"
      "max  (M1_NM, 16) OUT_F(0,0)<1>  TMP_F(0,0)<1;1,0>  0xC0C00000:f\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(a), "rw"(inv_scale));
  return dst;
}

// float -> sat(float*scale, +/-6) -> float
template <>
inline float mxfp4_scale_clamp_asm<float>(
    float const& src, float inv_scale) {
  float dst;
  asm("{\n"
      ".decl IN_F    v_type=G type=F num_elts=16 alias=<%1,0>\n"
      ".decl OUT_F   v_type=G type=F num_elts=16 alias=<%0,0>\n"
      ".decl SCALE_F v_type=G type=F num_elts=16 alias=<%2,0>\n"
      ".decl TMP_F   v_type=G type=F num_elts=16 align=32\n"
      "mul  (M1_NM, 16) TMP_F(0,0)<1>  IN_F(0,0)<1;1,0>   SCALE_F(0,0)<0;1,0>\n"
      "min  (M1_NM, 16) TMP_F(0,0)<1>  TMP_F(0,0)<1;1,0>  0x40C00000:f\n"
      "max  (M1_NM, 16) OUT_F(0,0)<1>  TMP_F(0,0)<1;1,0>  0xC0C00000:f\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(src), "rw"(inv_scale));
  return dst;
}

#endif  // VLLM_MXFP4_QUANT_ASM_ENABLED

}  // namespace mxfp4
}  // namespace vllm
