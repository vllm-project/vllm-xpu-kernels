#pragma once
#include <memory>
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

namespace vllm {
namespace xpu {

static inline sycl::queue& vllmGetQueue(at::DeviceIndex device_index = -1) {
  auto current_stream = c10::xpu::getCurrentXPUStream(device_index);
  auto& queue = current_stream.queue();
  return queue;
}

template <typename T>
struct SyclTypeTrait {
  using Type = T;
};

template <>
struct SyclTypeTrait<c10::Half> {
  using Type = sycl::half;
};

template <>
struct SyclTypeTrait<c10::BFloat16> {
  using Type = sycl::ext::oneapi::bfloat16;
};

template <typename T>
struct AccumulateType {
 private:
  static constexpr bool is_narrow_float =
      std::is_same_v<T, at::Half> || std::is_same_v<T, at::BFloat16> ||
      std::is_same_v<T, c10::Float8_e4m3fn> ||
      std::is_same_v<T, c10::Float8_e5m2>;

  static constexpr bool is_integer =
      std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> ||
      std::is_same_v<T, char> || std::is_same_v<T, int16_t> ||
      std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>;

  static constexpr bool is_complex = std::is_same_v<T, c10::complex<float>> ||
                                     std::is_same_v<T, c10::complex<double>>;

 public:
  using type = std::conditional_t<
      is_narrow_float, float,
      std::conditional_t<
          std::is_floating_point_v<T>, T,
          std::conditional_t<is_integer, int64_t,
                             std::conditional_t<is_complex, T, T>>>>;
};

template <typename T>
using acc_type = typename AccumulateType<T>::type;

// aligned vector generates vectorized load/store on XPU
template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vec {
  scalar_t val[vec_size];

  scalar_t& operator[](int index) { return val[index]; }

  scalar_t const& operator[](int index) const { return val[index]; }
};

// From float to float.
inline void from_float(float& dst, float src) { dst = src; }
// From float32 to float16.
inline void from_float(sycl::half& dst, float src) {
  dst = sycl::half(src);
}
// From float32 to bfloat16.
inline void from_float(sycl::ext::oneapi::bfloat16& dst, float src) {
  dst = sycl::ext::oneapi::bfloat16(src);
}

// From float to float.
inline  float to_float(float u) { return u; }
// From float16 to float32.
inline float to_float(sycl::half u) { return float(u); }
// From bfloat16 to float32.
inline float to_float(sycl::ext::oneapi::bfloat16 u) {
  return float(u);
}


}  // namespace xpu

}  // namespace vllm
