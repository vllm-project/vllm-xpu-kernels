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
#endif

template <typename Tout>
inline constexpr float fp8_max_asm_v =
    std::is_same_v<Tout, at::Float8_e4m3fn> ? 448.0f : 57344.0f;

template <typename Tin, typename Tout>
inline uint8_t scaled_quant_asm(Tin const& src, float inv_scale) {
  float x = static_cast<float>(src) * inv_scale;
  constexpr float fp8_max = fp8_max_asm_v<Tout>;
  x = sycl::fmax(-fp8_max, sycl::fmin(x, fp8_max));
#ifdef VLLM_FP8_QUANT_ASM_ENABLED
  sycl::half h = static_cast<sycl::half>(x);
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
