#pragma once

#include <cstdint>
#include <type_traits>
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

#ifdef VLLM_FP8_QUANT_ASM_ENABLED

inline uint8_t fcvt_half_e4m3(sycl::half h) {
  uint8_t dst;
  asm("{\n"
      ".decl IN_HF v_type=G type=HF num_elts=32 alias=<%1,0>\n"
      ".decl OUT_B v_type=G type=B  num_elts=32 alias=<%0,0>\n"
      "fcvt (M1_NM, 32) OUT_B(0,0)<1>  IN_HF(0,0)<1;1,0>\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(h));
  return dst;
}

inline uint8_t fcvt_half_e5m2(sycl::half h) {
  uint8_t dst;
  asm("{\n"
      ".decl IN_HF  v_type=G type=HF num_elts=32 alias=<%1,0>\n"
      ".decl OUT_UB v_type=G type=UB num_elts=32 alias=<%0,0>\n"
      "fcvt (M1_NM, 32) OUT_UB(0,0)<1>  IN_HF(0,0)<1;1,0>\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(h));
  return dst;
}

// Four conversions in one block so they issue back to back; byte k of the
// result holds the conversion of the k-th input.
inline uint32_t
fcvt4_half_e4m3(sycl::half a, sycl::half b, sycl::half c, sycl::half d) {
  uint32_t dst;
  asm("{\n"
      ".decl OUT_B v_type=G type=B  num_elts=128 alias=<%0,0>\n"
      ".decl A_HF  v_type=G type=HF num_elts=32 alias=<%1,0>\n"
      ".decl B_HF  v_type=G type=HF num_elts=32 alias=<%2,0>\n"
      ".decl C_HF  v_type=G type=HF num_elts=32 alias=<%3,0>\n"
      ".decl D_HF  v_type=G type=HF num_elts=32 alias=<%4,0>\n"
      "fcvt (M1_NM, 32) OUT_B(0,0)<4>  A_HF(0,0)<1;1,0>\n"
      "fcvt (M1_NM, 32) OUT_B(0,1)<4>  B_HF(0,0)<1;1,0>\n"
      "fcvt (M1_NM, 32) OUT_B(0,2)<4>  C_HF(0,0)<1;1,0>\n"
      "fcvt (M1_NM, 32) OUT_B(0,3)<4>  D_HF(0,0)<1;1,0>\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(a), "rw"(b), "rw"(c), "rw"(d));
  return dst;
}

inline uint32_t
fcvt4_half_e5m2(sycl::half a, sycl::half b, sycl::half c, sycl::half d) {
  uint32_t dst;
  asm("{\n"
      ".decl OUT_UB v_type=G type=UB num_elts=128 alias=<%0,0>\n"
      ".decl A_HF   v_type=G type=HF num_elts=32 alias=<%1,0>\n"
      ".decl B_HF   v_type=G type=HF num_elts=32 alias=<%2,0>\n"
      ".decl C_HF   v_type=G type=HF num_elts=32 alias=<%3,0>\n"
      ".decl D_HF   v_type=G type=HF num_elts=32 alias=<%4,0>\n"
      "fcvt (M1_NM, 32) OUT_UB(0,0)<4>  A_HF(0,0)<1;1,0>\n"
      "fcvt (M1_NM, 32) OUT_UB(0,1)<4>  B_HF(0,0)<1;1,0>\n"
      "fcvt (M1_NM, 32) OUT_UB(0,2)<4>  C_HF(0,0)<1;1,0>\n"
      "fcvt (M1_NM, 32) OUT_UB(0,3)<4>  D_HF(0,0)<1;1,0>\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(a), "rw"(b), "rw"(c), "rw"(d));
  return dst;
}
#endif

// Rounding to fp16 with round-to-odd makes the following fp16 -> fp8 rounding
// of the hardware fcvt equal to a direct fp32 -> fp8 rounding: fp16 keeps at
// least two more mantissa bits than e4m3 / e5m2, also for their subnormals.
inline sycl::half fp32_to_fp16_round_to_odd(float x) {
  const sycl::half h = static_cast<sycl::half>(x);
  const float back = static_cast<float>(h);
  const uint16_t hb = sycl::bit_cast<uint16_t>(h);
  const bool fix = back != x && (hb & 1u) == 0;
  const uint16_t step = sycl::fabs(back) > sycl::fabs(x) ? 0xFFFFu : 1u;
  return sycl::bit_cast<sycl::half>(
      static_cast<uint16_t>(hb + (fix ? step : 0u)));
}

// fp32 -> fp16 step in front of the fcvt to Tout. For e4m3, truncating the
// fp32 mantissa to fp16 width with a sticky LSB is a cheaper round-to-odd:
// the narrowing conversion is then exact for fp16 normals, and fp16
// subnormals lie below half the smallest e4m3 subnormal, so they round to
// zero either way. e5m2 subnormals reach into the fp16 subnormal range and
// need the full version.
template <typename Tout>
inline sycl::half fp32_to_fp16_for_fp8(float x) {
  if constexpr (std::is_same_v<Tout, at::Float8_e4m3fn>) {
    const uint32_t b = sycl::bit_cast<uint32_t>(x);
    const uint32_t t = (b & ~0x1FFFu) | ((b & 0x1FFFu) ? 0x2000u : 0u);
    return static_cast<sycl::half>(sycl::bit_cast<float>(t));
  } else {
    return fp32_to_fp16_round_to_odd(x);
  }
}

template <typename Tout>
inline constexpr float fp8_max_asm_v =
    std::is_same_v<Tout, at::Float8_e4m3fn> ? 448.0f : 57344.0f;

template <typename Tin, typename Tout>
inline uint8_t scaled_quant_asm(Tin const& src, float inv_scale) {
  float x = static_cast<float>(src) * inv_scale;
  constexpr float fp8_max = fp8_max_asm_v<Tout>;
  x = sycl::fmax(-fp8_max, sycl::fmin(x, fp8_max));
#ifdef VLLM_FP8_QUANT_ASM_ENABLED
  const sycl::half h = fp32_to_fp16_for_fp8<Tout>(x);
  if constexpr (std::is_same_v<Tout, at::Float8_e4m3fn>) {
    return fcvt_half_e4m3(h);
  } else {
    return fcvt_half_e5m2(h);
  }
}
#else
  return sycl::bit_cast<uint8_t>(static_cast<Tout>(x));
}
#endif

// Four scaled conversions packed into one word, byte k from x[k]. The Xe3p
// path issues the four fcvt back to back.
template <typename Tout, bool EXACT_IN_FP16 = false>
inline uint32_t scaled_quant4_asm(const float (&x)[4], float inv_scale) {
  constexpr float fp8_max = fp8_max_asm_v<Tout>;
  float y[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    y[k] = sycl::fmax(-fp8_max, sycl::fmin(x[k] * inv_scale, fp8_max));
  }
#ifdef VLLM_FP8_QUANT_ASM_ENABLED
  sycl::half h[4];
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    h[k] = EXACT_IN_FP16 ? static_cast<sycl::half>(y[k])
                         : fp32_to_fp16_for_fp8<Tout>(y[k]);
  }
  if constexpr (std::is_same_v<Tout, at::Float8_e4m3fn>) {
    return fcvt4_half_e4m3(h[0], h[1], h[2], h[3]);
  } else {
    return fcvt4_half_e5m2(h[0], h[1], h[2], h[3]);
  }
#else
  uint32_t w = 0;
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    w |= static_cast<uint32_t>(sycl::bit_cast<uint8_t>(static_cast<Tout>(y[k])))
         << (8 * k);
  }
  return w;
#endif
}

template <Fp8KVCacheDataType kv_dt>
struct kv_fp8_type;  // primary intentionally undefined; kAuto never uses it.
template <>
struct kv_fp8_type<Fp8KVCacheDataType::kFp8E4M3> {
  using type = at::Float8_e4m3fn;
};
template <>
struct kv_fp8_type<Fp8KVCacheDataType::kFp8E5M2> {
  using type = at::Float8_e5m2;
};

template <bool is_scale_inverted, typename scalar_t, typename fp8_type>
struct ScaledQuantOpAsm {
  float scale;
  inline void operator()(fp8_type& dst, scalar_t const& src) const {
    float inv_scale = is_scale_inverted ? scale : (1.0f / scale);
    uint8_t u8_value = scaled_quant_asm<scalar_t, fp8_type>(src, inv_scale);
    dst = sycl::bit_cast<fp8_type>(u8_value);
  }
};

// Xe3p counterpart of CopyWithScaleOp for vectorization_utils. Unlike
// CopyWithScaleOp it multiplies, so the caller passes the inverted scale.
template <typename OutT, typename InT, Fp8KVCacheDataType kv_dt>
struct CopyWithInvScaleOpAsm {
  float scale;

  inline void operator()(OutT& dst, const InT src) const {
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      dst = static_cast<OutT>(src);
    } else {
      // The fp8 format comes from kv_dt; OutT is just the uint8_t storage type.
      using fp8_t = typename kv_fp8_type<kv_dt>::type;
      dst = scaled_quant_asm<InT, fp8_t>(src, scale);
    }
  }
};

}  // namespace fp8
}  // namespace vllm
