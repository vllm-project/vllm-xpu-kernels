#include <sycl/sycl.hpp>
#include <cmath>
#include <algorithm>
#include "utils.h"
#include "dispatch_utils.h"

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

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
inline scalar_t compute(const scalar_t& x, const scalar_t& y) {
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
class act_kernel {
 public:
  act_kernel(scalar_t* __restrict__ out,          // [..., d]
             const scalar_t* __restrict__ input,  // [..., d]
             const int d)
      : out_(out), input_(input), d_(d) {}

  void operator() [[intel::reqd_sub_group_size(32)]] (
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

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
void act_and_mul_kernel(scalar_t* __restrict__ out,          // [..., d]
                        const scalar_t* __restrict__ input,  // [..., 2, d]
                        const int d, const sycl::nd_item<3>& item_ct1) {
  const int64_t token_idx = item_ct1.get_group(2);
  for (int64_t idx = item_ct1.get_local_id(2); idx < d;
       idx += item_ct1.get_local_range(2)) {
    const scalar_t x = input[token_idx * 2 * d + idx];
    const scalar_t y = input[token_idx * 2 * d + d + idx];
    out[token_idx * d + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
  }
}

template <typename scalar_t>
void call_silu_and_mul_kernel(torch::Tensor& out, torch::Tensor& input) {
  using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  // dpct::dim3 grid(num_tokens);
  // dpct::dim3 block(std::min(d, 1024));
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(d, 1024));
  if (num_tokens == 0) {
    return;
  }
  auto out_ptr = out.data_ptr<scalar_t>();
  auto input_ptr = input.data_ptr<scalar_t>();
  at::DeviceGuard device_guard(input.device());
  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
          act_and_mul_kernel<sycl_t, silu_kernel, true>(
              (sycl_t*)out_ptr, (sycl_t*)input_ptr, d, item_ct1);
        });
  });
}

}  // namespace vllm

void silu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "call_silu_and_mul_kernel",
      [&] { vllm::call_silu_and_mul_kernel<scalar_t>(out, input); });
}

// Launch element-wise activation kernel.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                                       \
  using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;                     \
  int d = input.size(-1);                                                      \
  int64_t num_tokens = input.numel() / input.size(-1);                         \
  sycl::range<3> grid(1, 1, num_tokens);                                       \
  sycl::range<3> block(1, 1, std::min(d, 1024));                               \
  if (num_tokens == 0) {                                                       \
    return;                                                                    \
  }                                                                            \
  auto out_ptr = out.data_ptr<scalar_t>();                                     \
  auto input_ptr = input.data_ptr<scalar_t>();                                 \
  at::DeviceGuard device_guard(input.device());                                \
  auto& queue = vllm::xpu::vllmGetQueue();                                     \
  queue.submit([&](sycl::handler& cgh) {                                       \
    cgh.parallel_for(sycl::nd_range<3>(grid * block, block),                   \
                     vllm::act_kernel<sycl_t, KERNEL>((sycl_t*)out_ptr,        \
                                                      (sycl_t*)input_ptr, d)); \
  });

void gelu_new(torch::Tensor& out,    // [..., d]
              torch::Tensor& input)  // [..., d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_new", [&] {
    LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel);
  });
}

void gelu_fast(torch::Tensor& out,    // [..., d]
               torch::Tensor& input)  // [..., d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_fast", [&] {
    LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel);
  });
}

void gelu_quick(torch::Tensor& out,    // [..., d]
                torch::Tensor& input)  // [..., d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_quick", [&] {
    LAUNCH_ACTIVATION_KERNEL(vllm::gelu_quick_kernel);
  });
}
