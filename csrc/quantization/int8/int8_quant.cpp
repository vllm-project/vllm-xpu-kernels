// SPDX-License-Identifier: Apache-2.0
// Dynamic per-token symmetric int8 activation quantization.
//
// One work-group per token row: pass 1 reduces |x| to the row absmax
// (reduce_over_group), every work-item derives the same scale, pass 2
// re-reads the row (cache-resident for typical hidden sizes, e.g. 17408 bf16 =
// 34 KiB) and writes int8. Math lives in int8_quant_math.h and matches vLLM's
// dynamic_per_token_int8_quant_ref (torch reference) exactly.
#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <sycl/sycl.hpp>

#include "dispatch_utils.h"
#include "ops.h"
#include "utils.h"

#include "quantization/int8/int8_quant_math.h"

namespace vllm {

template <typename scalar_t, typename scale_t, int VEC>
class dynamic_per_token_int8_quant_kernel {
 private:
  int8_t* out;
  scale_t* scale;
  scalar_t const* input;
  const int64_t in_row_stride;
  const int hidden_size;

  struct alignas(sizeof(scalar_t) * VEC) in_vec_t {
    scalar_t v[VEC];
  };
  struct alignas(VEC) out_vec_t {
    int8_t v[VEC];
  };

 public:
  dynamic_per_token_int8_quant_kernel(
      int8_t* out_,
      scale_t* scale_,
      scalar_t const* input_,
      int64_t in_row_stride_,
      int hidden_size_)
      : out(out_),
        scale(scale_),
        input(input_),
        in_row_stride(in_row_stride_),
        hidden_size(hidden_size_) {}

  void operator()(sycl::nd_item<1> item) const {
    const int tid = item.get_local_id(0);
    const int nthr = item.get_local_range(0);
    const int64_t row = item.get_group(0);

    scalar_t const* row_in = input + row * in_row_stride;
    int8_t* row_out = out + row * static_cast<int64_t>(hidden_size);

    float absmax = 0.0f;
    if constexpr (VEC > 1) {
      const int nvec = hidden_size / VEC;
      auto const* vin = reinterpret_cast<in_vec_t const*>(row_in);
      for (int i = tid; i < nvec; i += nthr) {
        in_vec_t x = vin[i];
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
          absmax =
              int8q::max_f(absmax, int8q::abs_f(static_cast<float>(x.v[j])));
        }
      }
    } else {
      for (int i = tid; i < hidden_size; i += nthr) {
        absmax =
            int8q::max_f(absmax, int8q::abs_f(static_cast<float>(row_in[i])));
      }
    }
    absmax = sycl::reduce_over_group(
        item.get_group(), absmax, sycl::maximum<float>());

    const float s = int8q::token_scale_from_absmax<scalar_t>(absmax);
    if (tid == 0) {
      scale[row] = static_cast<scale_t>(s);
    }

    if constexpr (VEC > 1) {
      const int nvec = hidden_size / VEC;
      auto const* vin = reinterpret_cast<in_vec_t const*>(row_in);
      auto* vout = reinterpret_cast<out_vec_t*>(row_out);
      for (int i = tid; i < nvec; i += nthr) {
        in_vec_t x = vin[i];
        out_vec_t q;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
          q.v[j] = int8q::quantize_one<scalar_t>(static_cast<float>(x.v[j]), s);
        }
        vout[i] = q;
      }
    } else {
      for (int i = tid; i < hidden_size; i += nthr) {
        row_out[i] =
            int8q::quantize_one<scalar_t>(static_cast<float>(row_in[i]), s);
      }
    }
  }
};

}  // namespace vllm

namespace {

template <typename scalar_t, typename scale_t, int VEC>
void launch_int8_quant(
    sycl::queue& queue,
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor& scales,
    int64_t num_tokens,
    int64_t in_row_stride,
    int hidden_size) {
  int wg = 256;
  const int work = VEC > 1 ? hidden_size / VEC : hidden_size;
  while (wg > 32 && wg / 2 >= work)
    wg /= 2;
  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(wg);
  queue.submit([&](sycl::handler& cgh) {
    auto kernel =
        vllm::dynamic_per_token_int8_quant_kernel<scalar_t, scale_t, VEC>(
            out.data_ptr<int8_t>(),
            reinterpret_cast<scale_t*>(scales.data_ptr()),
            reinterpret_cast<scalar_t const*>(input.data_ptr()),
            in_row_stride,
            hidden_size);
    cgh.parallel_for(sycl::nd_range<1>(grid * block, block), kernel);
  });
}

}  // namespace

// out:    int8, contiguous, same shape as input
// input:  [..., K] fp32/fp16/bf16; last dim contiguous; 2-D may carry a row
//         stride, higher ranks must be contiguous
// scales: numel == num_tokens, float32 or input.dtype; value is the scale
//         rounded to input.dtype (exactly what the reference returns)
void dynamic_per_token_scaled_int8_quant(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scales) {
  TORCH_CHECK(input.dim() >= 1, "int8 quant: input must have rank >= 1");
  TORCH_CHECK(
      out.scalar_type() == at::ScalarType::Char,
      "int8 quant: out must be int8");
  TORCH_CHECK(out.is_contiguous(), "int8 quant: out must be contiguous");
  TORCH_CHECK(
      out.sizes() == input.sizes(), "int8 quant: out/input shape mismatch");
  TORCH_CHECK(
      input.stride(-1) == 1, "int8 quant: input last dim must be contiguous");
  const int64_t hidden64 = input.size(-1);
  TORCH_CHECK(
      hidden64 > 0 && hidden64 <= INT32_MAX, "int8 quant: bad hidden size");
  const int hidden_size = static_cast<int>(hidden64);
  const int64_t num_tokens = input.numel() / hidden_size;
  int64_t in_row_stride = hidden_size;
  if (input.dim() == 2) {
    in_row_stride = input.stride(0);
  } else {
    TORCH_CHECK(
        input.is_contiguous(), "int8 quant: rank > 2 input must be contiguous");
  }
  TORCH_CHECK(
      scales.is_contiguous() && scales.numel() == num_tokens,
      "int8 quant: scales must be contiguous with one entry per token");
  const bool scale_f32 = scales.scalar_type() == at::ScalarType::Float;
  TORCH_CHECK(
      scale_f32 || scales.scalar_type() == input.scalar_type(),
      "int8 quant: scales dtype must be float32 or the input dtype");
  if (num_tokens == 0) return;

  const at::DeviceGuard device_guard(input.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "dynamic_per_token_scaled_int8_quant", [&] {
        constexpr int VEC = 16 / sizeof(scalar_t);
        const uintptr_t in_addr = reinterpret_cast<uintptr_t>(input.data_ptr());
        const uintptr_t out_addr = reinterpret_cast<uintptr_t>(out.data_ptr());
        const bool vec_ok = hidden_size % VEC == 0 &&
                            in_row_stride % VEC == 0 && in_addr % 16 == 0 &&
                            out_addr % VEC == 0;
        if (scale_f32) {
          if (vec_ok)
            launch_int8_quant<scalar_t, float, VEC>(
                queue,
                out,
                input,
                scales,
                num_tokens,
                in_row_stride,
                hidden_size);
          else
            launch_int8_quant<scalar_t, float, 1>(
                queue,
                out,
                input,
                scales,
                num_tokens,
                in_row_stride,
                hidden_size);
        } else {
          if (vec_ok)
            launch_int8_quant<scalar_t, scalar_t, VEC>(
                queue,
                out,
                input,
                scales,
                num_tokens,
                in_row_stride,
                hidden_size);
          else
            launch_int8_quant<scalar_t, scalar_t, 1>(
                queue,
                out,
                input,
                scales,
                num_tokens,
                in_row_stride,
                hidden_size);
        }
      });
}
