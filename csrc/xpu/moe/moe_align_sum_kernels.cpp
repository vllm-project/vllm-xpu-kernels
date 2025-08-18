#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>

#include <sycl/sycl.hpp>

#include "xpu/dispatch_utils.h"
#include "xpu/ops.h"
#include "xpu/utils.h"

namespace vllm {
namespace moe {

template <typename scalar_t, int TOPK>
class moe_sum_kernel {
 private:
  scalar_t* output;       // [..., d]
  const scalar_t* input;  // [..., topk, d]
  int d;

 public:
  moe_sum_kernel(scalar_t* output, const scalar_t* input, int d)
      : output(output), input(input), d(d) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t token_idx = item.get_group(0);
    for (int64_t idx = item.get_local_id(0); idx < d;
         idx += item.get_local_range(0)) {
      scalar_t x = 0.0;
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        x += input[token_idx * TOPK * d + k * d + idx];
      }
      output[token_idx * d + idx] = x;
    }
  }
};

}  // namespace moe
}  // namespace vllm

void moe_sum(torch::Tensor& input,   // [num_tokens, topk, hidden_size]
             torch::Tensor& output)  // [num_tokens, hidden_size]
{
  const int hidden_size = input.size(-1);
  const auto num_tokens = output.numel() / hidden_size;
  const int topk = input.size(1);

  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(std::min(hidden_size, 1024));
  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);
  auto& queue = vllm::xpu::vllmGetQueue();

  switch (topk) {
    case 2:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(sycl::nd_range<1>(grid * block, block),
                           vllm::moe::moe_sum_kernel<scalar_t, 2>(
                               output.data_ptr<scalar_t>(),
                               input.data_ptr<scalar_t>(), hidden_size));
        });
      });
      break;

    case 3:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(sycl::nd_range<1>(grid * block, block),
                           vllm::moe::moe_sum_kernel<scalar_t, 3>(
                               output.data_ptr<scalar_t>(),
                               input.data_ptr<scalar_t>(), hidden_size));
        });
      });
      break;

    case 4:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(sycl::nd_range<1>(grid * block, block),
                           vllm::moe::moe_sum_kernel<scalar_t, 4>(
                               output.data_ptr<scalar_t>(),
                               input.data_ptr<scalar_t>(), hidden_size));
        });
      });
      break;

    default:
      at::sum_out(output, input, 1);
      break;
  }
}
