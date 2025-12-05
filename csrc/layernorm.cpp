#include <sycl/sycl.hpp>

#include <algorithm>
#include "utils.h"
#include "dispatch_utils.h"

namespace vllm {

template <typename scalar_t>
class rms_norm_kernel {
 public:
  rms_norm_kernel(
      scalar_t* out_,
      const scalar_t* input_,
      const int64_t input_stride_,
      const scalar_t* weight_,
      const float epsilon_,
      const int num_tokens_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_)
      : out(out_),
        input(input_),
        input_stride(input_stride_),
        weight(weight_),
        epsilon(epsilon_),
        num_tokens(num_tokens_),
        hidden_size(hidden_size_),
        s_variance(s_variance_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();
    float variance = 0.0f;

    for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
         idx += item_ct1.get_local_range(2)) {
      const float x = (float)input[item_ct1.get_group(2) * input_stride + idx];
      variance += x * x;
    }

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());
    if (item_ct1.get_local_id(2) == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size + epsilon);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
         idx += item_ct1.get_local_range(2)) {
      float x = (float)input[item_ct1.get_group(2) * input_stride + idx];
      out[item_ct1.get_group(2) * hidden_size + idx] =
          ((scalar_t)(x * (*s_variance_ptr))) * weight[idx];
    }
  }

 private:
  scalar_t* __restrict__ out;          // [..., hidden_size]
  const scalar_t* __restrict__ input;  // [..., hidden_size]
  const int64_t input_stride;
  const scalar_t* __restrict__ weight;  // [hidden_size]
  const float epsilon;
  const int num_tokens;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;
};

template <typename scalar_t>
void call_rms_norm_kernel(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
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
        vllm::rms_norm_kernel<sycl_t>(
            (sycl_t*)out_ptr,
            (const sycl_t*)input_ptr,
            input_stride,
            (const sycl_t*)weight_ptr,
            epsilon,
            num_tokens,
            hidden_size,
            s_variance));
  });
}

template <typename scalar_t>
class fused_add_rms_norm_kernel {
 public:
  fused_add_rms_norm_kernel(
      scalar_t* __restrict__ input_,     // [..., hidden_size]
      scalar_t* __restrict__ residual_,  // [..., hidden_size]
      const int64_t input_stride_,
      const scalar_t* __restrict__ weight_,  // [hidden_size]
      const float epsilon_,
      const int num_tokens_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_)
      : input(input_),
        residual(residual_),
        input_stride(input_stride_),
        weight(weight_),
        epsilon(epsilon_),
        num_tokens(num_tokens_),
        hidden_size(hidden_size_),
        s_variance(s_variance_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();
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
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());
    if (item_ct1.get_local_id(2) == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size + epsilon);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
         idx += item_ct1.get_local_range(2)) {
      float x = (float)residual[item_ct1.get_group(2) * hidden_size + idx];
      input[item_ct1.get_group(2) * input_stride + idx] =
          ((scalar_t)(x * (*s_variance_ptr))) * weight[idx];
    }
  }

 private:
  scalar_t* __restrict__ input;     // [..., hidden_size]
  scalar_t* __restrict__ residual;  // [..., hidden_size]
  const int64_t input_stride;
  const scalar_t* __restrict__ weight;  // [hidden_size]
  const float epsilon;
  const int num_tokens;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;  // local memory for variance
};

template <typename scalar_t>
void call_fused_add_rms_norm_kernel(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
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
    sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block),
        fused_add_rms_norm_kernel<sycl_t>(
            (sycl_t*)input_ptr,
            (sycl_t*)residual_ptr,
            input_stride,
            (const sycl_t*)weight_ptr,
            epsilon,
            num_tokens,
            hidden_size,
            s_variance));
  });
}

}  // namespace vllm

void rms_norm(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    double epsilon) {
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "call_rms_norm_kernel", [&] {
        vllm::call_rms_norm_kernel<scalar_t>(out, input, weight, epsilon);
      });
}

void fused_add_rms_norm(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "call_fused_add_rms_norm_kernel", [&] {
        vllm::call_fused_add_rms_norm_kernel<scalar_t>(
            input, residual, weight, epsilon);
      });
}