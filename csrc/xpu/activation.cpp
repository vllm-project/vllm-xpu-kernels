#include <sycl/sycl.hpp>

#include <algorithm>
#include "utils.h"
#include "dispatch_utils.h"

namespace vllm {

template <typename T>
inline T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + sycl::exp((float)-x)));
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
inline scalar_t compute(const scalar_t& x, const scalar_t& y) {
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d,
    const sycl::nd_item<3>& item_ct1) {
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
  //dpct::dim3 grid(num_tokens);                                           
  //dpct::dim3 block(std::min(d, 1024));                                   
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
             act_and_mul_kernel<sycl_t, silu_kernel, true>((sycl_t*)out_ptr, (sycl_t*)input_ptr, d, item_ct1);
       });
  });  



}

}  // namespace vllm

void silu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  VLLM_DISPATCH_FLOATING_TYPES(
    input.scalar_type(), "call_silu_and_mul_kernel", [&] {
           vllm::call_silu_and_mul_kernel<scalar_t>(out, input);
    }); 
}
