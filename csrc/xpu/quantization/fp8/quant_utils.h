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

template <typename scalar_t>
struct alignas(8) vec4_t {
  scalar_t x;
  scalar_t y;
  scalar_t z;
  scalar_t w;
};

template <typename dtype_t>
struct alignas(4) dtypex4_t {
  static_assert(std::is_same_v<dtype_t, float> ||
                    std::is_same_v<dtype_t, at::Half> ||
                    std::is_same_v<dtype_t, at::BFloat16> ||
                    std::is_same_v<dtype_t, at::Float8_e4m3fn> ||
                    std::is_same_v<dtype_t, at::Float8_e5m2>,
                "Unsupported cache type for dtypex4_t");
  dtype_t x;
  dtype_t y;
  dtype_t z;
  dtype_t w;
};

template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, at::Float8_e5m2> ||
                                      std::is_same_v<T, at::Float8_e4m3fn>>>
struct quant_type_max {
  static constexpr T val() { return std::numeric_limits<T>::max(); }
};

template <typename T>
static constexpr T quant_type_max_v = quant_type_max<T>::val();

template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, at::Float8_e5m2> ||
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
      float x = (float)src / scale;
      if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
        dst = static_cast<at::Float8_e4m3fn>(x);
      } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
        dst = static_cast<at::Float8_e5m2>(x);
      }
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

// The vector width is fixed at 4 to avoid excessive branching in the kernel,
// which could degrade performance.
template <typename scalar_t, typename dtype_t, typename ScaOp>
void scaled_convert_vec(const scalar_t* src, dtype_t* dst, int num_elems,
                        int local_idx, int local_range, ScaOp&& scalar_op) {
  using srcx4_t = vec4_t<scalar_t>;
  using distx4_t = dtypex4_t<dtype_t>;

  int64_t const num_vec_elems = num_elems >> 2;

  auto const* vectorized_in = reinterpret_cast<srcx4_t const*>(src);
  auto* vectorized_out = reinterpret_cast<distx4_t*>(dst);

#pragma unroll 4
  for (int64_t i = local_idx; i < num_vec_elems; i += local_range) {
    srcx4_t in_vec = vectorized_in[i];
    distx4_t out_vec;
    scalar_op(out_vec.x, in_vec.x);
    scalar_op(out_vec.y, in_vec.y);
    scalar_op(out_vec.z, in_vec.z);
    scalar_op(out_vec.w, in_vec.w);
    vectorized_out[i] = out_vec;
  }

  // Handle the remaining elements if num_elems is not divisible by 4
  for (int64_t i = num_vec_elems * 4 + local_idx; i < num_elems;
       i += local_range) {
    scalar_op(dst[i], src[i]);
  }
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