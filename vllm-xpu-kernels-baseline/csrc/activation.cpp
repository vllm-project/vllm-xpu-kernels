#include <sycl/sycl.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "utils.h"
#include "dispatch_utils.h"

#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include "quantization/fp8/quant_utils.h"

#define VLLM_LDG(arg) *(arg)

namespace vllm {

template <typename T>
inline T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + sycl::exp((float)-x)));
}

template <typename T>
inline T gelu_fast_kernel(const T& x) {
  const float f = (float)x;
  const T t =
      (T)tanhf(((T)(f * 0.79788456f)) * (((T)1.0) + (T)(0.044715f * f) * x));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
inline T gelu_new_kernel(const T& x) {
  // 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
  const float x3 = (float)(x * x * x);
  const T t = (T)tanhf((T)(0.79788456f * (float)(x + (T)(0.044715f * x3))));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
inline T gelu_quick_kernel(const T& x) {
  // x * sigmoid(1.702 * x)
  return (T)(((float)x) / (1.0f + (T)sycl::exp(-1.702f * (float)x)));
}

template <typename T>
inline T relu2_no_mul_kernel(const T& x) {
  // square(relu(x))
  const float f = (float)x;
  const float r = f > 0.0f ? f : 0.0f;
  return (T)(r * r);
}

template <typename T>
inline T gelu_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = (float)x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + sycl::erf(f * ALPHA)));
}

template <typename T>
inline T gelu_tanh_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  const float f = (float)x;
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  float x_cube = f * f * f;
  float inner = BETA * (f + KAPPA * x_cube);
  return (T)(0.5f * f * (1.0f + sycl::tanh(inner)));
}

template <typename T>
inline T fatrelu_kernel(const T& x, const float threshold) {
  const float f = (float)x;
  return (T)(f > threshold ? f : 0.0f);
}

template <
    typename scalar_t,
    scalar_t (*ACT_FN)(const scalar_t&),
    bool act_first>
inline scalar_t compute(const scalar_t& x, const scalar_t& y) {
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
class act_kernel {
 public:
  act_kernel(
      scalar_t* __restrict__ out,          // [..., d]
      const scalar_t* __restrict__ input,  // [..., d]
      const int d)
      : out_(out), input_(input), d_(d) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    const int64_t token_idx = item_ct1.get_group(2);
    for (int64_t idx = item_ct1.get_local_id(2); idx < d_;
         idx += item_ct1.get_local_range(2)) {
      const scalar_t x = input_[token_idx * d_ + idx];
      out_[token_idx * d_ + idx] = ACT_FN(x);
    }
  }

 private:
  scalar_t* __restrict__ out_;          // [..., d]
  const scalar_t* __restrict__ input_;  // [..., d]
  const int d_;
};

template <
    typename scalar_t,
    scalar_t (*ACT_FN)(const scalar_t&),
    bool act_first>
class act_and_mul_kernel {
 public:
  act_and_mul_kernel(
      scalar_t* __restrict__ out,          // [..., d]
      const scalar_t* __restrict__ input,  // [..., 2, d]
      const int d)
      : out_(out), input_(input), d_(d) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    const int64_t token_idx = item_ct1.get_group(2);
    for (int64_t idx = item_ct1.get_local_id(2); idx < d_;
         idx += item_ct1.get_local_range(2)) {
      const scalar_t x = input_[token_idx * 2 * d_ + idx];
      const scalar_t y = input_[token_idx * 2 * d_ + d_ + idx];
      out_[token_idx * d_ + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
    }
  }

 private:
  scalar_t* __restrict__ out_;          // [..., d]
  const scalar_t* __restrict__ input_;  // [..., 2, d]
  const int d_;
};

// Vectorized version of act_and_mul_kernel using aligned vector loads/stores.
// Each work-item processes VEC_SIZE elements per iteration, reducing memory
// transactions and improving bandwidth utilization.
template <
    typename scalar_t,
    scalar_t (*ACT_FN)(const scalar_t&),
    bool act_first,
    int VEC_SIZE>
class act_and_mul_vec_kernel {
 public:
  act_and_mul_vec_kernel(
      scalar_t* __restrict__ out,
      const scalar_t* __restrict__ input,
      const int d)
      : out_(out), input_(input), d_(d) {}

  void operator()(sycl::nd_item<1> item) const {
    using vec_t = vllm::xpu::aligned_vec<scalar_t, VEC_SIZE>;
    const int64_t token_idx = item.get_group(0);
    const int64_t offset = item.get_local_linear_id();
    const int64_t step = item.get_local_range(0);
    const int64_t bound = d_ / VEC_SIZE;

    for (int64_t i = offset; i < bound; i += step) {
      auto x_vec =
          reinterpret_cast<const vec_t*>(input_)[token_idx * bound * 2 + i];
      auto y_vec = reinterpret_cast<const vec_t*>(
          input_)[token_idx * bound * 2 + i + bound];
      vec_t out_vec;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        out_vec[j] = compute<scalar_t, ACT_FN, act_first>(x_vec[j], y_vec[j]);
      }
      reinterpret_cast<vec_t*>(out_)[token_idx * bound + i] = out_vec;
    }
  }

 private:
  scalar_t* __restrict__ out_;
  const scalar_t* __restrict__ input_;
  const int d_;
};

template <
    typename scalar_t,
    scalar_t (*ACT_FN)(const scalar_t&),
    typename fp8_type,
    int VEC_SIZE>
class act_and_mul_quant_vec_kernel {
 public:
  act_and_mul_quant_vec_kernel(
      fp8_type* __restrict__ out,          // [..., d]
      const scalar_t* __restrict__ input,  // [..., 2 * d]
      const float* __restrict__ scale,     // [1]
      const int d)
      : out_(out), input_(input), scale_(scale), d_(d) {}

  void operator()(sycl::nd_item<1> item) const {
    using vec_t = vllm::xpu::aligned_vec<scalar_t, VEC_SIZE>;

    const int64_t token_idx = item.get_group(0);
    const int64_t offset = item.get_local_linear_id();
    const int64_t step = item.get_local_range(0);
    const int64_t bound = d_ / VEC_SIZE;

    const float inv_scale = 1.0f / (*scale_);
    const float fp8_max = static_cast<float>(fp8::quant_type_max_v<fp8_type>);

    // x and y halves are laid out contiguously: [x0..xd-1, y0..yd-1]
    const auto* v_x =
        reinterpret_cast<const vec_t*>(input_) + token_idx * bound * 2;
    const auto* v_y = v_x + bound;

    for (int64_t i = offset; i < bound; i += step) {
      vec_t xv = v_x[i];
      vec_t yv = v_y[i];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        float val = static_cast<float>(ACT_FN(xv[j]) * yv[j]) * inv_scale;
        float clamped = sycl::fmax(-fp8_max, sycl::fmin(val, fp8_max));
        out_[token_idx * d_ + i * VEC_SIZE + j] =
            static_cast<fp8_type>(clamped);
      }
    }
  }

 private:
  fp8_type* __restrict__ out_;          // [..., d]
  const scalar_t* __restrict__ input_;  // [..., 2 * d]
  const float* __restrict__ scale_;     // [1]
  const int d_;
};

template <
    typename scalar_t,
    scalar_t (*ACT_FN)(const scalar_t&, const float),
    int VEC_SIZE>
class act_and_mul_with_param_vec_kernel {
 public:
  act_and_mul_with_param_vec_kernel(
      scalar_t* __restrict__ out,
      const scalar_t* __restrict__ input,
      const int d,
      const float param)
      : out_(out), input_(input), d_(d), param_(param) {}

  void operator()(sycl::nd_item<1> item) const {
    using vec_t = vllm::xpu::aligned_vec<scalar_t, VEC_SIZE>;
    const int64_t token_idx = item.get_group(0);
    const int64_t offset = item.get_local_linear_id();
    const int64_t step = item.get_local_range(0);
    const int64_t bound = d_ / VEC_SIZE;

    for (int64_t i = offset; i < bound; i += step) {
      auto x_vec =
          reinterpret_cast<const vec_t*>(input_)[token_idx * bound * 2 + i];
      auto y_vec = reinterpret_cast<const vec_t*>(
          input_)[token_idx * bound * 2 + i + bound];
      vec_t out_vec;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        out_vec[j] = ACT_FN(x_vec[j], param_) * y_vec[j];
      }
      reinterpret_cast<vec_t*>(out_)[token_idx * bound + i] = out_vec;
    }
  }

 private:
  scalar_t* __restrict__ out_;
  const scalar_t* __restrict__ input_;
  const int d_;
  const float param_;
};

template <typename T>
[[intel::device_indirectly_callable]] inline __attribute__((always_inline)) T
swigluoai_and_mul(const T& gate, const T& up, float alpha, float limit) {
  // clamp gate: min=None, max=limit
  const float gate_f = (float)gate;
  const float clamped_gate = gate_f > limit ? limit : gate_f;

  // clamp up: min=-limit, max=limit
  const float up_f = (float)up;
  const float clamped_up =
      up_f > limit ? limit : (up_f < -limit ? -limit : up_f);

  // glu = gate * sigmoid(gate * alpha)
  const float sigmoid_val = 1.0f / (1.0f + sycl::exp(-clamped_gate * alpha));
  const float glu = clamped_gate * sigmoid_val;

  // (up + 1) * glu
  return (T)((clamped_up + 1.0f) * glu);
}

template <typename T>
[[intel::device_indirectly_callable]] inline __attribute__((always_inline)) T
swiglustep_and_mul(const T& gate, const T& up, float limit) {
  // gate = silu(gate).clamp(max=limit)
  const float gate_f = (float)gate;
  const float silu_gate = gate_f / (1.0f + sycl::exp(-gate_f));
  const float clamped_gate = silu_gate > limit ? limit : silu_gate;

  // up = up.clamp(min=-limit, max=limit)
  const float up_f = (float)up;
  const float clamped_up =
      up_f > limit ? limit : (up_f < -limit ? -limit : up_f);

  return (T)(clamped_gate * clamped_up);
}

template <
    typename scalar_t,
    scalar_t (*ACT_FN)(
        const scalar_t&, const scalar_t&, const float, const float)>
class swigluoai_and_mul_kernel {
 public:
  swigluoai_and_mul_kernel(
      scalar_t* __restrict__ out,          // [..., d]
      const scalar_t* __restrict__ input,  // [..., 2, d]
      const int d,
      const float alpha,
      const float limit)
      : out(out), input(input), d(d), alpha(alpha), limit(limit) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t token_idx = item.get_group(0);
    for (int64_t idx = item.get_local_id(0); idx < d;
         idx += item.get_local_range(0)) {
      // gate = x[..., ::2]  (even indices)
      const scalar_t gate = VLLM_LDG(&input[token_idx * 2 * d + 2 * idx]);
      // up = x[..., 1::2]   (odd indices)
      const scalar_t up = VLLM_LDG(&input[token_idx * 2 * d + 2 * idx + 1]);

      out[token_idx * d + idx] = ACT_FN(gate, up, alpha, limit);
    }
  }

 private:
  scalar_t* out;          // [..., d]
  const scalar_t* input;  // [..., topk, d]
  const int d;
  const float alpha;
  const float limit;
};

template <
    typename scalar_t,
    scalar_t (*ACT_FN)(const scalar_t&, const scalar_t&, const float)>
class swiglustep_and_mul_kernel {
 public:
  swiglustep_and_mul_kernel(
      scalar_t* __restrict__ out,          // [..., d]
      const scalar_t* __restrict__ input,  // [..., 2 * d]
      const int d,
      const float limit)
      : out(out), input(input), d(d), limit(limit) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t token_idx = item.get_group(0);
    for (int64_t idx = item.get_local_id(0); idx < d;
         idx += item.get_local_range(0)) {
      // gate = first half, up = second half (contiguous chunks)
      const scalar_t gate = VLLM_LDG(&input[token_idx * 2 * d + idx]);
      const scalar_t up = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);

      out[token_idx * d + idx] = ACT_FN(gate, up, limit);
    }
  }

 private:
  scalar_t* out;
  const scalar_t* input;
  const int d;
  const float limit;
};

}  // namespace vllm

// Launch activation and gating kernel.
// Use ACT_FIRST (bool) indicating whether to apply the activation function
// first.
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, ACT_FIRST)     \
  using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;   \
  int d = input.size(-1) / 2;                                \
  int64_t num_tokens = input.numel() / input.size(-1);       \
  sycl::range<3> grid(1, 1, num_tokens);                     \
  sycl::range<3> block(1, 1, std::min(d, 1024));             \
  if (num_tokens == 0) {                                     \
    return;                                                  \
  }                                                          \
  auto out_ptr = out.data_ptr<scalar_t>();                   \
  auto input_ptr = input.data_ptr<scalar_t>();               \
  at::DeviceGuard device_guard(input.device());              \
  auto& queue = vllm::xpu::vllmGetQueue();                   \
  queue.submit([&](sycl::handler& cgh) {                     \
    cgh.parallel_for(                                        \
        sycl::nd_range<3>(grid * block, block),              \
        vllm::act_and_mul_kernel<sycl_t, KERNEL, ACT_FIRST>( \
            (sycl_t*)out_ptr, (sycl_t*)input_ptr, d));       \
  });

// Vectorized launch: dispatch to vec_size=1,2,4,8,16 based on d and dtype.
#define VEC_LAUNCH_ACT_AND_MUL(KERNEL, ACT_FIRST, N)                  \
  case N: {                                                           \
    queue.submit([&](sycl::handler& cgh) {                            \
      cgh.parallel_for(                                               \
          sycl::nd_range<1>(num_tokens * wg_size, wg_size),           \
          vllm::act_and_mul_vec_kernel<sycl_t, KERNEL, ACT_FIRST, N>( \
              (sycl_t*)out_ptr, (sycl_t*)input_ptr, d));              \
    });                                                               \
    break;                                                            \
  }

#define VEC_LAUNCH_ACT_AND_MUL_WITH_PARAM(KERNEL, N)                  \
  case N: {                                                           \
    queue.submit([&](sycl::handler& cgh) {                            \
      cgh.parallel_for(                                               \
          sycl::nd_range<1>(num_tokens * wg_size, wg_size),           \
          vllm::act_and_mul_with_param_vec_kernel<sycl_t, KERNEL, N>( \
              (sycl_t*)out_ptr, (sycl_t*)input_ptr, d, param));       \
    });                                                               \
    break;                                                            \
  }

#define LAUNCH_ACTIVATION_GATE_KERNEL_VEC(KERNEL, ACT_FIRST)             \
  using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;               \
  int d = input.size(-1) / 2;                                            \
  int64_t num_tokens = input.numel() / input.size(-1);                   \
  if (num_tokens == 0) {                                                 \
    return;                                                              \
  }                                                                      \
  auto out_ptr = out.data_ptr<scalar_t>();                               \
  auto input_ptr = input.data_ptr<scalar_t>();                           \
  at::DeviceGuard device_guard(input.device());                          \
  auto& queue = vllm::xpu::vllmGetQueue();                               \
  int vec_size = static_cast<int>(sizeof(float) * 4 / sizeof(scalar_t)); \
  {                                                                      \
    int64_t tmp_wg =                                                     \
        std::min(static_cast<int64_t>(d), static_cast<int64_t>(1024));   \
    while (vec_size > 1 && (vec_size >> 1) * tmp_wg >= d) {              \
      vec_size = vec_size >> 1;                                          \
    }                                                                    \
  }                                                                      \
  if (d % vec_size != 0) vec_size = 1;                                   \
  int64_t wg_size = std::min(                                            \
      static_cast<int64_t>(d / vec_size), static_cast<int64_t>(1024));   \
  switch (vec_size) {                                                    \
    VEC_LAUNCH_ACT_AND_MUL(KERNEL, ACT_FIRST, 1);                        \
    VEC_LAUNCH_ACT_AND_MUL(KERNEL, ACT_FIRST, 2);                        \
    VEC_LAUNCH_ACT_AND_MUL(KERNEL, ACT_FIRST, 4);                        \
    VEC_LAUNCH_ACT_AND_MUL(KERNEL, ACT_FIRST, 8);                        \
    VEC_LAUNCH_ACT_AND_MUL(KERNEL, ACT_FIRST, 16);                       \
    default:                                                             \
      TORCH_CHECK(false, "Unsupported vector size: ", vec_size);         \
  }

#define LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM_VEC(KERNEL, PARAM)      \
  using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;               \
  int d = input.size(-1) / 2;                                            \
  int64_t num_tokens = input.numel() / input.size(-1);                   \
  if (num_tokens == 0) {                                                 \
    return;                                                              \
  }                                                                      \
  auto out_ptr = out.data_ptr<scalar_t>();                               \
  auto input_ptr = input.data_ptr<scalar_t>();                           \
  const float param = static_cast<float>(PARAM);                         \
  at::DeviceGuard device_guard(input.device());                          \
  auto& queue = vllm::xpu::vllmGetQueue();                               \
  int vec_size = static_cast<int>(sizeof(float) * 4 / sizeof(scalar_t)); \
  {                                                                      \
    int64_t tmp_wg =                                                     \
        std::min(static_cast<int64_t>(d), static_cast<int64_t>(1024));   \
    while (vec_size > 1 && (vec_size >> 1) * tmp_wg >= d) {              \
      vec_size = vec_size >> 1;                                          \
    }                                                                    \
  }                                                                      \
  if (d % vec_size != 0) vec_size = 1;                                   \
  int64_t wg_size = std::min(                                            \
      static_cast<int64_t>(d / vec_size), static_cast<int64_t>(1024));   \
  switch (vec_size) {                                                    \
    VEC_LAUNCH_ACT_AND_MUL_WITH_PARAM(KERNEL, 1);                        \
    VEC_LAUNCH_ACT_AND_MUL_WITH_PARAM(KERNEL, 2);                        \
    VEC_LAUNCH_ACT_AND_MUL_WITH_PARAM(KERNEL, 4);                        \
    VEC_LAUNCH_ACT_AND_MUL_WITH_PARAM(KERNEL, 8);                        \
    VEC_LAUNCH_ACT_AND_MUL_WITH_PARAM(KERNEL, 16);                       \
    default:                                                             \
      TORCH_CHECK(false, "Unsupported vector size: ", vec_size);         \
  }

void silu_and_mul(
    torch::Tensor& out,    // [..., d]
    torch::Tensor& input)  // [..., 2 * d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "silu_and_mul", [&] {
    LAUNCH_ACTIVATION_GATE_KERNEL_VEC(vllm::silu_kernel, true);
  });
}

// Fused SiLU + Mul + FP8 Quantization
// Input: [..., 2*d] in FP16/BF16, Output: [..., d] in FP8
// Dispatches to the vectorized kernel (VEC_SIZE=1..8) based on alignment.
#define LAUNCH_ACT_AND_MUL_QUANT_VEC(KERNEL, N)                               \
  case N: {                                                                   \
    int64_t wg_size =                                                         \
        std::min(static_cast<int64_t>(d / N), static_cast<int64_t>(1024));    \
    VLLM_DISPATCH_FP8_TYPES(                                                  \
        out.scalar_type(), "act_and_mul_quant_vec_kernel_fp8", [&] {          \
          auto out_ptr = out.data_ptr<fp8_t>();                               \
          queue.submit([&](sycl::handler& cgh) {                              \
            cgh.parallel_for(                                                 \
                sycl::nd_range<1>(num_tokens * wg_size, wg_size),             \
                vllm::act_and_mul_quant_vec_kernel<sycl_t, KERNEL, fp8_t, N>( \
                    out_ptr, (sycl_t*)input_ptr, scale_ptr, d));              \
          });                                                                 \
        });                                                                   \
    break;                                                                    \
  }

#define LAUNCH_ACTIVATION_GATE_QUANT_KERNEL(KERNEL)                          \
  using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;                   \
  int d = input.size(-1) / 2;                                                \
  int64_t num_tokens = input.numel() / input.size(-1);                       \
  if (num_tokens == 0) {                                                     \
    return;                                                                  \
  }                                                                          \
  auto input_ptr = input.data_ptr<scalar_t>();                               \
  auto scale_ptr = scale.data_ptr<float>();                                  \
  at::DeviceGuard device_guard(input.device());                              \
  auto& queue = vllm::xpu::vllmGetQueue();                                   \
  /* Compute vec_size like non-quant path: gcd(4*sizeof(float)/sizeof, d) */ \
  int vec_size = static_cast<int>(sizeof(float) * 4 / sizeof(sycl_t));       \
  {                                                                          \
    int64_t tmp_wg =                                                         \
        std::min(static_cast<int64_t>(d), static_cast<int64_t>(1024));       \
    while (vec_size > 1 && (vec_size >> 1) * tmp_wg >= d) {                  \
      vec_size = vec_size >> 1;                                              \
    }                                                                        \
  }                                                                          \
  if (d % vec_size != 0) vec_size = 1;                                       \
  switch (vec_size) {                                                        \
    LAUNCH_ACT_AND_MUL_QUANT_VEC(KERNEL, 1);                                 \
    LAUNCH_ACT_AND_MUL_QUANT_VEC(KERNEL, 2);                                 \
    LAUNCH_ACT_AND_MUL_QUANT_VEC(KERNEL, 4);                                 \
    LAUNCH_ACT_AND_MUL_QUANT_VEC(KERNEL, 8);                                 \
    LAUNCH_ACT_AND_MUL_QUANT_VEC(KERNEL, 16);                                \
    default:                                                                 \
      TORCH_CHECK(false, "Unsupported vector size: ", vec_size);             \
  }

void silu_and_mul_quant(
    torch::Tensor& out,    // [..., d] FP8
    torch::Tensor& input,  // [..., 2 * d] FP16/BF16
    torch::Tensor& scale)  // [1] FP32
{
  TORCH_CHECK(
      out.dtype() == torch::kFloat8_e4m3fn ||
      out.dtype() == torch::kFloat8_e5m2);
  TORCH_CHECK(
      input.dtype() == torch::kFloat16 || input.dtype() == torch::kBFloat16);
  TORCH_CHECK(input.size(-1) % 2 == 0);
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "silu_and_mul_quant", [&] {
    LAUNCH_ACTIVATION_GATE_QUANT_KERNEL(vllm::silu_kernel);
  });
}

void mul_and_silu(
    torch::Tensor& out,    // [..., d]
    torch::Tensor& input)  // [..., 2 * d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mul_and_silu", [&] {
    LAUNCH_ACTIVATION_GATE_KERNEL_VEC(vllm::silu_kernel, false);
  });
}

void gelu_and_mul(
    torch::Tensor& out,    // [..., d]
    torch::Tensor& input)  // [..., 2 * d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_and_mul", [&] {
    LAUNCH_ACTIVATION_GATE_KERNEL_VEC(vllm::gelu_kernel, true);
  });
}

void gelu_tanh_and_mul(
    torch::Tensor& out,    // [..., d]
    torch::Tensor& input)  // [..., 2 * d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_tanh_and_mul", [&] {
    LAUNCH_ACTIVATION_GATE_KERNEL_VEC(vllm::gelu_tanh_kernel, true);
  });
}

void fatrelu_and_mul(
    torch::Tensor& out,    // [..., d]
    torch::Tensor& input,  // [..., 2 * d]
    double threshold) {
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fatrelu_and_mul", [&] {
    LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM_VEC(
        vllm::fatrelu_kernel, threshold);
  });
}

// Launch element-wise activation kernel.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                   \
  using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type; \
  int d = input.size(-1);                                  \
  int64_t num_tokens = input.numel() / input.size(-1);     \
  sycl::range<3> grid(1, 1, num_tokens);                   \
  sycl::range<3> block(1, 1, std::min(d, 1024));           \
  if (num_tokens == 0) {                                   \
    return;                                                \
  }                                                        \
  auto out_ptr = out.data_ptr<scalar_t>();                 \
  auto input_ptr = input.data_ptr<scalar_t>();             \
  at::DeviceGuard device_guard(input.device());            \
  auto& queue = vllm::xpu::vllmGetQueue();                 \
  queue.submit([&](sycl::handler& cgh) {                   \
    cgh.parallel_for(                                      \
        sycl::nd_range<3>(grid * block, block),            \
        vllm::act_kernel<sycl_t, KERNEL>(                  \
            (sycl_t*)out_ptr, (sycl_t*)input_ptr, d));     \
  });

#define LAUNCH_SWIGLUOAI_AND_MUL(KERNEL, ALPHA, LIMIT)                    \
  int d = input.size(-1) / 2;                                             \
  int64_t num_tokens = input.numel() / input.size(-1);                    \
  sycl::range<1> grid(num_tokens);                                        \
  sycl::range<1> block(std::min(d, 1024));                                \
  at::DeviceGuard device_guard(input.device());                           \
  auto& queue = vllm::xpu::vllmGetQueue();                                \
  VLLM_DISPATCH_FLOATING_TYPES(                                           \
      input.scalar_type(), "clamp_swiglu_kernel_with_params", [&] {       \
        queue.submit([&](sycl::handler& cgh) {                            \
          cgh.parallel_for(                                               \
              sycl::nd_range<1>(grid * block, block),                     \
              vllm::swigluoai_and_mul_kernel<scalar_t, KERNEL<scalar_t>>( \
                  out.data_ptr<scalar_t>(),                               \
                  input.data_ptr<scalar_t>(),                             \
                  d,                                                      \
                  ALPHA,                                                  \
                  LIMIT));                                                \
        });                                                               \
      });

void gelu_new(
    torch::Tensor& out,    // [..., d]
    torch::Tensor& input)  // [..., d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_new", [&] {
    LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel);
  });
}

void gelu_fast(
    torch::Tensor& out,    // [..., d]
    torch::Tensor& input)  // [..., d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_fast", [&] {
    LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel);
  });
}

void gelu_quick(
    torch::Tensor& out,    // [..., d]
    torch::Tensor& input)  // [..., d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_quick", [&] {
    LAUNCH_ACTIVATION_KERNEL(vllm::gelu_quick_kernel);
  });
}

void relu2_no_mul(
    torch::Tensor& out,    // [..., d]
    torch::Tensor& input)  // [..., d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu2_no_mul", [&] {
    LAUNCH_ACTIVATION_KERNEL(vllm::relu2_no_mul_kernel);
  });
}

void swigluoai_and_mul(
    torch::Tensor& out,    // [..., d]
    torch::Tensor& input,  // [..., 2 * d]
    double alpha,
    double limit) {
  LAUNCH_SWIGLUOAI_AND_MUL(vllm::swigluoai_and_mul, alpha, limit);
}

#define LAUNCH_SWIGLUSTEP_AND_MUL(KERNEL, LIMIT)                           \
  int d = input.size(-1) / 2;                                              \
  int64_t num_tokens = input.numel() / input.size(-1);                     \
  sycl::range<1> grid(num_tokens);                                         \
  sycl::range<1> block(std::min(d, 1024));                                 \
  at::DeviceGuard device_guard(input.device());                            \
  auto& queue = vllm::xpu::vllmGetQueue();                                 \
  VLLM_DISPATCH_FLOATING_TYPES(                                            \
      input.scalar_type(), "swiglustep_and_mul_kernel", [&] {              \
        queue.submit([&](sycl::handler& cgh) {                             \
          cgh.parallel_for(                                                \
              sycl::nd_range<1>(grid * block, block),                      \
              vllm::swiglustep_and_mul_kernel<scalar_t, KERNEL<scalar_t>>( \
                  out.data_ptr<scalar_t>(),                                \
                  input.data_ptr<scalar_t>(),                              \
                  d,                                                       \
                  LIMIT));                                                 \
        });                                                                \
      });

void swiglustep_and_mul(
    torch::Tensor& out,    // [..., d]
    torch::Tensor& input,  // [..., 2 * d]
    double limit) {
  LAUNCH_SWIGLUSTEP_AND_MUL(vllm::swiglustep_and_mul, limit);
}
