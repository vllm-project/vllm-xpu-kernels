#pragma once
#include <cmath>

#include <ATen/ScalarType.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>

namespace vllm {

enum class Fp8KVCacheDataType {
  kAuto = 0,
  kFp8E4M3 = 1,
  kFp8E5M2 = 2,
};

namespace fp8 {

template <
    typename T,
    typename = std::enable_if_t<
        std::is_same_v<T, at::Float8_e5m2> ||
        std::is_same_v<T, at::Float8_e4m3fn>>>
struct quant_type_max {
  static constexpr T val() { return std::numeric_limits<T>::max(); }
};

template <typename T>
static constexpr T quant_type_max_v = quant_type_max<T>::val();

template <
    typename T,
    typename = std::enable_if_t<
        std::is_same_v<T, at::Float8_e5m2> ||
        std::is_same_v<T, at::Float8_e4m3fn>>>
struct min_scaling_factor {
  static inline float val() { return 1.0f / (quant_type_max_v<T> * 512.0f); }
};

// Used by vectorization_utils to copy/convert one element
template <typename OutT, typename InT, Fp8KVCacheDataType kv_dt>
struct CopyWithScaleOp {
  float scale;
  inline void operator()(OutT& dst, const InT src) const {
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      dst = static_cast<OutT>(src);
    } else {
      using out_dtype = std::conditional_t<
          kv_dt == Fp8KVCacheDataType::kFp8E5M2,
          at::Float8_e5m2,
          at::Float8_e4m3fn>;
      float fp8_max = quant_type_max_v<out_dtype>;
      float x = (float)src / scale;
      x = sycl::fmax(-fp8_max, sycl::fmin(x, fp8_max));
      auto fp8_val = static_cast<out_dtype>(x);
      dst = sycl::bit_cast<OutT>(fp8_val);
    }
  }
};

// convert a float value to fp8 type with scaling
template <bool is_scale_inverted, typename fp8_type>
struct ConvertWithScaleOp {
  float scale;

  inline void operator()(fp8_type& dst, float const src) const {
    float x = is_scale_inverted ? (src * scale) : (src / scale);
    const float fp8_max = static_cast<float>(quant_type_max_v<fp8_type>);
    float r = sycl::fmax(-fp8_max, sycl::fmin(x, fp8_max));
    dst = static_cast<fp8_type>(r);
  }
};

}  // namespace fp8
}  // namespace vllm

// The following macro is used to dispatch the conversion function based on
// the data type of the key and value cache. The FN is a macro that calls a
// function with template<typename scalar_t, typename cache_t,
// Fp8KVCacheDataType kv_dt>.
// TODO: change uint8_t to sycl fp8 dtype when it is supported
#define DISPATCH_BY_KV_CACHE_DTYPE(SRC_DTYPE, KV_DTYPE, FN)                    \
  if (KV_DTYPE == "auto") {                                                    \
    if (SRC_DTYPE == at::ScalarType::Float) {                                  \
      FN(float, float, vllm::Fp8KVCacheDataType::kAuto);                       \
    } else if (SRC_DTYPE == at::ScalarType::Half) {                            \
      FN(at::Half, at::Half, vllm::Fp8KVCacheDataType::kAuto);                 \
    } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                        \
      FN(at::BFloat16, at::BFloat16, vllm::Fp8KVCacheDataType::kAuto);         \
    } else {                                                                   \
      TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE);   \
    }                                                                          \
  } else {                                                                     \
    if (KV_DTYPE == "fp8" || KV_DTYPE == "fp8_e4m3") {                         \
      if (SRC_DTYPE == at::ScalarType::Float) {                                \
        FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);                \
      } else if (SRC_DTYPE == at::ScalarType::Half) {                          \
        FN(at::Half, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);             \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                      \
        FN(at::BFloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);         \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE); \
      }                                                                        \
    } else if (KV_DTYPE == "fp8_e5m2") {                                       \
      if (SRC_DTYPE == at::ScalarType::Float) {                                \
        FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2);                \
      } else if (SRC_DTYPE == at::ScalarType::Half) {                          \
        FN(at::Half, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2);             \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                      \
        FN(at::BFloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2);         \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE); \
      }                                                                        \
    } else {                                                                   \
      TORCH_CHECK(false, "Unsupported data type of kv cache: ", KV_DTYPE);     \
    }                                                                          \
  }