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
inline float
mxfp4_scale_clamp_asm<at::Half>(at::Half const& src, float inv_scale) {
  sycl::half a = sycl::bit_cast<sycl::half>(src);
  float dst;
  asm("{\n"
      ".decl IN_HF   v_type=G type=HF num_elts=32 alias=<%1,0>\n"
      ".decl OUT_F   v_type=G type=F  num_elts=32 alias=<%0,0>\n"
      ".decl SCALE_F v_type=G type=F  num_elts=32 alias=<%2,0>\n"
      ".decl TMP_F   v_type=G type=F  num_elts=32 align=32\n"
      "mov  (M1_NM, 32) TMP_F(0,0)<1>  IN_HF(0,0)<1;1,0>\n"
      "mul  (M1_NM, 32) TMP_F(0,0)<1>  TMP_F(0,0)<1;1,0>  SCALE_F(0,0)<0;1,0>\n"
      "min  (M1_NM, 32) TMP_F(0,0)<1>  TMP_F(0,0)<1;1,0>  0x40C00000:f\n"
      "max  (M1_NM, 32) OUT_F(0,0)<1>  TMP_F(0,0)<1;1,0>  0xC0C00000:f\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(a), "rw"(inv_scale));
  return dst;
}

// bfloat16 -> sat(bf16*scale, +/-6) -> float
template <>
inline float
mxfp4_scale_clamp_asm<at::BFloat16>(at::BFloat16 const& src, float inv_scale) {
  sycl::ext::oneapi::bfloat16 a =
      sycl::bit_cast<sycl::ext::oneapi::bfloat16>(src);
  float dst;
  asm("{\n"
      ".decl IN_BF   v_type=G type=BF num_elts=32 alias=<%1,0>\n"
      ".decl OUT_F   v_type=G type=F  num_elts=32 alias=<%0,0>\n"
      ".decl SCALE_F v_type=G type=F  num_elts=32 alias=<%2,0>\n"
      ".decl TMP_F   v_type=G type=F  num_elts=32 align=32\n"
      "mov  (M1_NM, 32) TMP_F(0,0)<1>  IN_BF(0,0)<1;1,0>\n"
      "mul  (M1_NM, 32) TMP_F(0,0)<1>  TMP_F(0,0)<1;1,0>  SCALE_F(0,0)<0;1,0>\n"
      "min  (M1_NM, 32) TMP_F(0,0)<1>  TMP_F(0,0)<1;1,0>  0x40C00000:f\n"
      "max  (M1_NM, 32) OUT_F(0,0)<1>  TMP_F(0,0)<1;1,0>  0xC0C00000:f\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(a), "rw"(inv_scale));
  return dst;
}

// float -> sat(float*scale, +/-6) -> float
template <>
inline float mxfp4_scale_clamp_asm<float>(float const& src, float inv_scale) {
  float dst;
  asm("{\n"
      ".decl IN_F    v_type=G type=F num_elts=32 alias=<%1,0>\n"
      ".decl OUT_F   v_type=G type=F num_elts=32 alias=<%0,0>\n"
      ".decl SCALE_F v_type=G type=F num_elts=32 alias=<%2,0>\n"
      ".decl TMP_F   v_type=G type=F num_elts=32 align=32\n"
      "mul  (M1_NM, 32) TMP_F(0,0)<1>  IN_F(0,0)<1;1,0>   SCALE_F(0,0)<0;1,0>\n"
      "min  (M1_NM, 32) TMP_F(0,0)<1>  TMP_F(0,0)<1;1,0>  0x40C00000:f\n"
      "max  (M1_NM, 32) OUT_F(0,0)<1>  TMP_F(0,0)<1;1,0>  0xC0C00000:f\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(src), "rw"(inv_scale));
  return dst;
}

// Convert eight bf16 values to eight packed e2m1 values.
// Cooperative subgroup operation.
inline uint32_t
dnscl_bf16x8_to_e2m1(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  uint32_t dst;
  asm("{\n"
      ".decl OUT_UD v_type=G type=UD num_elts=32 alias=<%0,0>\n"
      ".decl A_UD   v_type=G type=UD num_elts=32 alias=<%1,0>\n"
      ".decl B_UD   v_type=G type=UD num_elts=32 alias=<%2,0>\n"
      ".decl C_UD   v_type=G type=UD num_elts=32 alias=<%3,0>\n"
      ".decl D_UD   v_type=G type=UD num_elts=32 alias=<%4,0>\n"
      ".decl T0_UD  v_type=G type=UD num_elts=32 align=wordx32\n"
      ".decl T1_UD  v_type=G type=UD num_elts=32 align=wordx32\n"
      "dnscl.bftoe2m1.mode0.rne (M1_NM, 16) T0_UD.0 A_UD.0 C_UD.0 %%null.0\n"
      "dnscl.bftoe2m1.mode0.rne (M5_NM, 16) T0_UD.64 A_UD.64 C_UD.64 "
      "%%null.0\n"
      "dnscl.bftoe2m1.mode2.rne (M1_NM, 16) T1_UD.0 B_UD.0 D_UD.0 %%null.0\n"
      "dnscl.bftoe2m1.mode2.rne (M5_NM, 16) T1_UD.64 B_UD.64 D_UD.64 "
      "%%null.0\n"
      "or (M1_NM, 32) OUT_UD(0,0)<1> T0_UD(0,0)<1;1,0> T1_UD(0,0)<1;1,0>\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(a), "rw"(b), "rw"(c), "rw"(d));
  return dst;
}

#endif  // VLLM_MXFP4_QUANT_ASM_ENABLED

}  // namespace mxfp4
}  // namespace vllm
