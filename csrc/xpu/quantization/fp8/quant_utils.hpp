#include <ATen/ScalarType.h>

namespace vllm {

enum class Fp8KVCacheDataType {
  kAuto = 0,
  kFp8E4M3 = 1,
  kFp8E5M2 = 2,
};

namespace fp8 {

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ Tout scaled_convert(const Tin& x, const float scale) {
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    return static_cast<at::Float8_e4m3fn>(x * scale);
  } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
    return static_cast<at::Float8_e5m2>(x * scale);
  }

  assert(false);
  return {};  // Squash missing return statement warning
}

}  // namespace fp8
}  // namespace vllm

// The following macro is used to dispatch the conversion function based on
// the data type of the key and value cache. The FN is a macro that calls a
// function with template<typename scalar_t, typename cache_t,
// Fp8KVCacheDataType kv_dt>.
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
        FN(float, at::Float8_e4m3fn, vllm::Fp8KVCacheDataType::kFp8E4M3);      \
      } else if (SRC_DTYPE == at::ScalarType::Half) {                          \
        FN(at::Half, at::Float8_e4m3fn, vllm::Fp8KVCacheDataType::kFp8E4M3);   \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                      \
        FN(at::BFloat16, at::Float8_e4m3fn,                                    \
           vllm::Fp8KVCacheDataType::kFp8E4M3);                                \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE); \
      }                                                                        \
    } else if (KV_DTYPE == "fp8_e5m2") {                                       \
      if (SRC_DTYPE == at::ScalarType::Float) {                                \
        FN(float, at::Float8_e5m2, vllm::Fp8KVCacheDataType::kFp8E5M2);        \
      } else if (SRC_DTYPE == at::ScalarType::Half) {                          \
        FN(at::Half, at::Float8_e5m2, vllm::Fp8KVCacheDataType::kFp8E5M2);     \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                      \
        FN(at::BFloat16, at::Float8_e5m2, vllm::Fp8KVCacheDataType::kFp8E5M2); \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE); \
      }                                                                        \
    } else {                                                                   \
      TORCH_CHECK(false, "Unsupported data type of kv cache: ", KV_DTYPE);     \
    }                                                                          \
  }