// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <sycl/sycl.hpp>
#include <torch/all.h>
#include "dispatch_utils.h"
#include "utils.h"
#include "quantization/fp8/quant_utils.h"
#include "quantization/fused_kernels/quant_conversions.h"

namespace vllm {

template <typename scalar_t, typename out_t>
class silu_and_mul_per_block_quant_kernel {
 public:
  silu_and_mul_per_block_quant_kernel(
      out_t* __restrict__ out_,
      float* __restrict__ scales_,
      const scalar_t* __restrict__ input_,
      const float* scale_ub_,
      const int hidden_size_,
      const int num_groups_,
      const int group_size_,
      const int64_t scale_stride_token_,
      const int64_t scale_stride_group_)
      : out(out_),
        scales(scales_),
        input(input_),
        scale_ub(scale_ub_),
        hidden_size(hidden_size_),
        num_groups(num_groups_),
        group_size(group_size_),
        scale_stride_token(scale_stride_token_),
        scale_stride_group(scale_stride_group_) {}

  void operator()(sycl::nd_item<1> item) const {
    const int wg_id = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));

    const int token_idx = wg_id / num_groups;
    const int group_idx = wg_id % num_groups;
    const int group_start = group_idx * group_size;

    const scalar_t* token_gate =
        input + token_idx * hidden_size * 2 + group_start;
    const scalar_t* token_up = token_gate + hidden_size;
    out_t* token_output = out + token_idx * hidden_size + group_start;

    // Step 1: Each thread loads one element, computes SiLU(gate) * up
    const float gate_f = static_cast<float>(token_gate[tid]);
    const float up_f = static_cast<float>(token_up[tid]);
    const float silu_gate = gate_f / (1.0f + sycl::exp(-gate_f));
    const float result = silu_gate * up_f;

    // Step 2: Reduce to find group max
    const float group_max = sycl::reduce_over_group(
        item.get_group(), sycl::fabs(result), sycl::maximum<float>());

    // Step 3: Compute scale
    float group_scale;
    if constexpr (std::is_same_v<out_t, int8_t>) {
      group_scale =
          std::max(group_max / 127.0f, std::numeric_limits<float>::epsilon());
    } else {
      const float fp8_max = static_cast<float>(fp8::quant_type_max_v<out_t>);
      group_scale =
          sycl::max(group_max / fp8_max, fp8::min_scaling_factor<out_t>::val());
      if (scale_ub != nullptr) {
        group_scale = sycl::min(group_scale, *scale_ub);
        group_scale =
            sycl::max(group_scale, fp8::min_scaling_factor<out_t>::val());
      }
    }

    if (tid == 0) {
      const int64_t scale_idx =
          static_cast<int64_t>(token_idx) * scale_stride_token +
          static_cast<int64_t>(group_idx) * scale_stride_group;
      scales[scale_idx] = group_scale;
    }

    // Step 4: Quantize and write output
    token_output[tid] =
        vllm::ScaledQuant<out_t, false>::quant_fn(result, group_scale);
  }

 private:
  out_t* __restrict__ out;
  float* __restrict__ scales;
  const scalar_t* __restrict__ input;
  const float* scale_ub;
  const int hidden_size;
  const int num_groups;
  const int group_size;
  const int64_t scale_stride_token;
  const int64_t scale_stride_group;
};

}  // namespace vllm

void silu_and_mul_per_block_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor& scales,
    int64_t group_size,
    std::optional<torch::Tensor> scale_ub,
    bool is_scale_transposed) {
  TORCH_CHECK(
      out.dtype() == torch::kFloat8_e4m3fn ||
          out.dtype() == torch::kFloat8_e5m2 || out.dtype() == torch::kInt8,
      "out must be float8_e4m3fn, float8_e5m2, or int8");
  TORCH_CHECK(
      input.dtype() == torch::kFloat16 || input.dtype() == torch::kBFloat16,
      "input must be float16 or bfloat16");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(scales.dtype() == torch::kFloat32, "scales must be float32");
  TORCH_CHECK(
      group_size == 64 || group_size == 128,
      "group_size must be 64 or 128, got ",
      group_size);
  if (scale_ub.has_value()) {
    TORCH_CHECK(
        out.dtype() == torch::kFloat8_e4m3fn ||
            out.dtype() == torch::kFloat8_e5m2,
        "scale_ub is only supported for FP8 output");
  }

  const int64_t num_tokens = input.size(0);
  const int64_t hidden_size = out.size(-1);
  const int64_t num_groups = hidden_size / group_size;
  TORCH_CHECK(input.dim() == 2, "input must be 2-dimensional");
  TORCH_CHECK(out.dim() == 2, "out must be 2-dimensional");
  TORCH_CHECK(
      input.size(-1) == hidden_size * 2,
      "input last dim must be 2 * hidden_size");
  TORCH_CHECK(
      hidden_size % group_size == 0,
      "hidden_size must be divisible by group_size");
  TORCH_CHECK(
      out.size(0) == num_tokens,
      "out.size(0) must equal num_tokens (input.size(0))");
  TORCH_CHECK(
      scales.size(0) == num_tokens && scales.size(1) == num_groups,
      "scales must have logical shape [num_tokens, num_groups]");

  if (num_tokens == 0) return;

  const int64_t scale_stride_token =
      is_scale_transposed ? 1LL : static_cast<int64_t>(num_groups);
  const int64_t scale_stride_group = is_scale_transposed ? num_tokens : 1LL;

  const at::DeviceGuard device_guard(input.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  const float* scale_ub_ptr =
      scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr;

  const int64_t num_wgs = num_tokens * num_groups;
  const int64_t wg_size = group_size;

  VLLM_DISPATCH_HALF_TYPES(
      input.scalar_type(), "silu_and_mul_per_block_quant", [&] {
        using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        const auto* input_ptr =
            reinterpret_cast<const sycl_t*>(input.data_ptr<scalar_t>());
        float* scales_ptr = scales.data_ptr<float>();

        auto launch = [&](auto out_ptr) {
          using out_t = std::remove_pointer_t<decltype(out_ptr)>;
          queue.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<1>(num_wgs * wg_size, wg_size),
                vllm::silu_and_mul_per_block_quant_kernel<sycl_t, out_t>(
                    out_ptr,
                    scales_ptr,
                    input_ptr,
                    scale_ub_ptr,
                    static_cast<int>(hidden_size),
                    static_cast<int>(num_groups),
                    static_cast<int>(group_size),
                    scale_stride_token,
                    scale_stride_group));
          });
        };

        if (out.dtype() == torch::kFloat8_e4m3fn) {
          launch(out.data_ptr<at::Float8_e4m3fn>());
        } else if (out.dtype() == torch::kFloat8_e5m2) {
          launch(out.data_ptr<at::Float8_e5m2>());
        } else {
          launch(out.data_ptr<int8_t>());
        }
      });
}
