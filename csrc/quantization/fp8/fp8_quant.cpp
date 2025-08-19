#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/xpu/XPUContext.h>

#include <sycl/sycl.hpp>

#include "dispatch_utils.h"
#include "ops.h"

#include "fp8_quant.h"
#include "quant_utils.h"
#include "utils.h"

namespace vllm {

template <typename scalar_t, typename fp8_type>
class scaled_fp8_quant_kernel {
 private:
  fp8_type* out;
  const scalar_t* input;
  const float* scale;
  int64_t num_elems;

 public:
  scaled_fp8_quant_kernel(fp8_type* out_, const scalar_t* input_,
                          const float* scale_, int64_t num_elems_)
      : out(out_), input(input_), scale(scale_), num_elems(num_elems_) {}
  void operator()(sycl::nd_item<1> item) const {
    int tid = item.get_global_linear_id();

    // Invert the scale so that we can use multiplications to avoid expensive
    // division.
    const float inverted_scale = 1.0f / (*scale);
    fp8::ConvertWithScaleOp<true, fp8_type> op{inverted_scale};
    fp8::scaled_convert_vec(input, out, num_elems, tid,
                            item.get_local_range(0) * item.get_group_range(0),
                            op);
  }
};

template <typename scalar_t, typename fp8_type>
class dynamic_per_token_scaled_fp8_quant_kernel {
 private:
  fp8_type* out;
  float* scale;
  scalar_t const* input;
  float const* scale_ub;
  const int hidden_size;

 public:
  dynamic_per_token_scaled_fp8_quant_kernel(fp8_type* out_, float* scale_,
                                            scalar_t const* input_,
                                            float const* scale_ub_,
                                            const int hidden_size_)
      : out(out_),
        scale(scale_),
        input(input_),
        scale_ub(scale_ub_),
        hidden_size(hidden_size_) {}

  void operator()(sycl::nd_item<1> item) const {
    int const tid = item.get_local_id(0);
    int const token_idx = item.get_group(0);

    // Use int64 to avoid overflowing an int32 when calculating this offset
    int64_t offset = static_cast<int64_t>(token_idx) * hidden_size;
    scalar_t const* token_input = &input[offset];
    fp8_type* token_output = &out[offset];

    // For vectorization, token_input and token_output pointers need to be
    // aligned at 8-byte and 4-byte addresses respectively.
    bool const can_vectorize = hidden_size % 4 == 0;

    float absmax_val = 0.0f;
    if (can_vectorize) {
      absmax_val = thread_max_vec(token_input, hidden_size, tid,
                                  item.get_local_range(0));
    } else {
      for (int i = tid; i < hidden_size; i += item.get_local_range(0)) {
        float const x = static_cast<float>(token_input[i]);
        absmax_val = sycl::max(absmax_val, sycl::fabs(x));
      }
    }

    float const block_absmax_val_maybe = sycl::reduce_over_group(
        item.get_group(), absmax_val, sycl::maximum<float>());
    // __shared__ float token_scale;
    auto& token_scale =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1]>(
            item.get_group());
    if (tid == 0) {
      if (scale_ub) {
        token_scale[0] = sycl::min(block_absmax_val_maybe, *scale_ub);
      } else {
        token_scale[0] = block_absmax_val_maybe;
      }
      // token scale computation
      token_scale[0] =
          sycl::max(token_scale[0] / fp8::quant_type_max_v<fp8_type>,
                    fp8::min_scaling_factor<fp8_type>::val());
      scale[token_idx] = token_scale[0];
    }
    group_barrier(item.get_group());

    // Note that we don't use inverted scales so we can match FBGemm impl.
    const float inverted_scale = 1.0f / (token_scale[0]);
    if (can_vectorize) {
      fp8::ConvertWithScaleOp<true, fp8_type> op{inverted_scale};
      fp8::scaled_convert_vec(token_input, token_output, hidden_size, tid,
                              item.get_local_range(0), op);
    } else {
      for (int i = tid; i < hidden_size; i += item.get_local_range(0)) {
        fp8::ConvertWithScaleOp<true, fp8_type> op{inverted_scale};
        op(token_output[i], token_input[i]);
      }
    }
  }
};

}  // namespace vllm

void static_scaled_fp8_quant(torch::Tensor& out,          // [..., d]
                             torch::Tensor const& input,  // [..., d]
                             torch::Tensor const& scale)  // [1]
{
  int64_t num_tokens = input.numel() / input.size(-1);
  int64_t num_elems = input.numel();
  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(1024);
  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);

  auto& queue = vllm::xpu::vllmGetQueue();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
              // Launch the kernel
              queue.submit([&](sycl::handler& cgh) {
                auto kernel = vllm::scaled_fp8_quant_kernel<scalar_t, fp8_t>(
                    out.data_ptr<fp8_t>(), input.data_ptr<scalar_t>(),
                    scale.data_ptr<float>(), num_elems);
                cgh.parallel_for(sycl::nd_range<1>(grid * block, block),
                                 kernel);
              });
            });
      });
}

void dynamic_scaled_fp8_quant(torch::Tensor& out,          // [..., d]
                              torch::Tensor const& input,  // [..., d]
                              torch::Tensor& scale)        // [1]
{
  int64_t num_tokens = input.numel() / input.size(-1);
  int64_t num_elems = input.numel();
  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(1024);
  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);

  auto& queue = vllm::xpu::vllmGetQueue();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
              // Launch the kernel
              queue.submit([&](sycl::handler& cgh) {
                auto max_reduce_kernel =
                    vllm::segmented_max_reduction<scalar_t, fp8_t>(
                        scale.data_ptr<float>(), input.data_ptr<scalar_t>(),
                        num_elems);
                cgh.parallel_for(sycl::nd_range<1>(grid * block, block),
                                 max_reduce_kernel);
              });
              queue.submit([&](sycl::handler& cgh) {
                auto kernel = vllm::scaled_fp8_quant_kernel<scalar_t, fp8_t>(
                    out.data_ptr<fp8_t>(), input.data_ptr<scalar_t>(),
                    scale.data_ptr<float>(), num_elems);
                cgh.parallel_for(sycl::nd_range<1>(grid * block, block),
                                 kernel);
              });
            });
      });
}

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out,          // [..., d]
    torch::Tensor const& input,  // [..., d]
    torch::Tensor& scales, std::optional<at::Tensor> const& scale_ub) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(std::min(hidden_size, 1024));

  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);

  auto& queue = vllm::xpu::vllmGetQueue();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "dynamic_per_token_scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(),
            "dynamic_per_token_scaled_fp8_quant_kernel_fp8_type", [&] {
              // Launch the kernel
              queue
                  .submit([&](sycl::handler& cgh) {
                    auto kernel =
                        vllm::dynamic_per_token_scaled_fp8_quant_kernel<
                            scalar_t, fp8_t>(
                            out.data_ptr<fp8_t>(), scales.data_ptr<float>(),
                            input.data_ptr<scalar_t>(),
                            scale_ub.has_value() ? scale_ub->data_ptr<float>()
                                                 : nullptr,
                            hidden_size);
                    cgh.parallel_for(sycl::nd_range<1>(grid * block, block),
                                     kernel);
                  })
                  .wait();
            });
      });
}
