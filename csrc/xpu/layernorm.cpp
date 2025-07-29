#include <sycl/sycl.hpp>

#include <algorithm>
#include "utils.h"
#include "dispatch_utils.h"

namespace vllm {

template <typename scalar_t>
void rms_norm_kernel(scalar_t* __restrict__ out,          // [..., hidden_size]
                     const scalar_t* __restrict__ input,  // [..., hidden_size]
                     const int64_t input_stride,
                     const scalar_t* __restrict__ weight,  // [hidden_size]
                     const float epsilon, const int num_tokens,
                     const int hidden_size, const sycl::nd_item<3>& item_ct1,
                     float* s_variance) {
  float variance = 0.0f;

  for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
       idx += item_ct1.get_local_range(2)) {
    const float x = (float)input[item_ct1.get_group(2) * input_stride + idx];
    variance += x * x;
  }

  variance = sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_work_group<3>(), variance,
      sycl::plus<>());
  if (item_ct1.get_local_id(2) == 0) {
    *s_variance = sycl::rsqrt(variance / hidden_size + epsilon);
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);

  for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
       idx += item_ct1.get_local_range(2)) {
    float x = (float)input[item_ct1.get_group(2) * hidden_size + idx];
    out[item_ct1.get_group(2) * input_stride + idx] =
        ((scalar_t)(x * (*s_variance))) * weight[idx];
  }
}

template <typename scalar_t>
void call_rms_norm_kernel(torch::Tensor& out, torch::Tensor& input,
                          torch::Tensor& weight, float epsilon) {
  using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  int64_t input_stride = input.stride(-2);
  auto out_ptr = out.data_ptr<scalar_t>();
  auto input_ptr = input.data_ptr<scalar_t>();
  auto weight_ptr = weight.data_ptr<scalar_t>();
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(hidden_size, 1024));
  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
          rms_norm_kernel<sycl_t>((sycl_t*)out_ptr, (const sycl_t*)input_ptr,
                                  input_stride, (const sycl_t*)weight_ptr,
                                  epsilon, num_tokens, hidden_size, item_ct1,
                                  s_variance.get_pointer());
        });
  });
}

template <typename scalar_t>
void fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,     // [..., hidden_size]
    scalar_t* __restrict__ residual,  // [..., hidden_size]
    const int64_t input_stride,
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size,
    const sycl::nd_item<3>& item_ct1, float* s_variance) {
  float variance = 0.0f;

  for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
       idx += item_ct1.get_local_range(2)) {
    scalar_t z = (scalar_t)input[item_ct1.get_group(2) * input_stride + idx];
    z += residual[item_ct1.get_group(2) * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[item_ct1.get_group(2) * hidden_size + idx] = z;
  }

  variance = sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_work_group<3>(), variance,
      sycl::plus<>());
  if (item_ct1.get_local_id(2) == 0) {
    *s_variance = sycl::rsqrt(variance / hidden_size + epsilon);
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);

  for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
       idx += item_ct1.get_local_range(2)) {
    float x = (float)residual[item_ct1.get_group(2) * hidden_size + idx];
    input[item_ct1.get_group(2) * input_stride + idx] =
        ((scalar_t)(x * (*s_variance))) * weight[idx];
  }
}

template <typename scalar_t>
void call_fused_add_rms_norm_kernel(torch::Tensor& input,
                                    torch::Tensor& residual,
                                    torch::Tensor& weight, float epsilon) {
  using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  auto input_ptr = input.data_ptr<scalar_t>();
  auto residual_ptr = residual.data_ptr<scalar_t>();
  auto weight_ptr = weight.data_ptr<scalar_t>();
  int64_t input_stride = input.stride(-2);
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(hidden_size, 1024));
  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> shared_vals(sycl::range<1>(32), cgh);
    sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
          fused_add_rms_norm_kernel<sycl_t>(
              (sycl_t*)input_ptr, (sycl_t*)residual_ptr, input_stride,
              (const sycl_t*)weight_ptr, epsilon, num_tokens, hidden_size,
              item_ct1, s_variance.get_pointer());
        });
  });
}

}  // namespace vllm

void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
              double epsilon) {
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "call_rms_norm_kernel", [&] {
        vllm::call_rms_norm_kernel<scalar_t>(out, input, weight, epsilon);
      });
}

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        torch::Tensor& weight, double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(),
                               "call_fused_add_rms_norm_kernel", [&] {
                                 vllm::call_fused_add_rms_norm_kernel<scalar_t>(
                                     input, residual, weight, epsilon);
                               });
}