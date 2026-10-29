// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <sycl/sycl.hpp>
#include <torch/all.h>
#include <ATen/DeviceGuard.h>

#include <limits>

#include "dispatch_utils.h"
#include "ops.h"
#include "utils.h"
#include "quantization/fp4/mxfp4_quant.h"
#include "quantization/mx_quant_utils.h"
#include "quantization/swiglu_limit.h"
#include "quantization/utils.h"

namespace vllm {
// Input  [M, 2*H]   – concatenated (gate, up) activations
// Output [M, H/2]   – packed FP4 nibbles, two per byte (low=even, high=odd)
// Scales [M, H/32]  – float32 UE8M0 (power-of-two) scales
// One lane per 16-byte vector of the output.
template <typename scalar_t, bool COL_MAJOR, bool CLAMP = false>
class silu_and_mul_mxfp4_quant_kernel {
 public:
  silu_and_mul_mxfp4_quant_kernel(
      uint8_t* __restrict__ out_,
      float* __restrict__ scales_,
      const scalar_t* __restrict__ input_,
      const int num_vecs_,
      const int vecs_per_row_,
      const float epsilon_,
      const int num_tokens_,
      const float swiglu_limit_ = kSwigluLimitDisabled)
      : input(input_),
        num_vecs(num_vecs_),
        vecs_per_row(vecs_per_row_),
        out(out_),
        scales(scales_),
        epsilon(epsilon_),
        num_tokens(num_tokens_),
        swiglu_limit(swiglu_limit_) {}

  void operator()
      [[sycl::reqd_sub_group_size(32),
        sycl::reqd_work_group_size(mx::kWorkGroupSize)]] (
          sycl::nd_item<1> item) const {
    constexpr int VEC = mx::kVec;
    const int v = mx::flat_index(item);
    const bool active = v < num_vecs;
    const int token = v / vecs_per_row;
    const int col = v - token * vecs_per_row;

    float res[VEC];
    float absmax = 0.0f;
    if (active) {
      const auto* vin = reinterpret_cast<const vec_n_t<scalar_t, VEC>*>(input);
      const int gate = 2 * token * vecs_per_row + col;
      const vec_n_t<scalar_t, VEC> g = vin[gate];
      const vec_n_t<scalar_t, VEC> u = vin[gate + vecs_per_row];
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        float gf = static_cast<float>(g.val[k]);
        float uf = static_cast<float>(u.val[k]);
        if constexpr (CLAMP) {
          gf = swiglu_clamp_gate(gf, swiglu_limit);
          uf = swiglu_clamp_up(uf, swiglu_limit);
        }
        res[k] = (gf / (1.0f + sycl::exp(-gf))) * uf;
        absmax = sycl::fmax(absmax, sycl::fabs(res[k]));
      }
    }
    absmax = mx::group_absmax(item.get_sub_group(), absmax, mx::kLanesPerGroup);
    float inv_scale;
    const float y_s = mxfp4::mxfp4_pow2_scale(absmax, epsilon, inv_scale);
    const uint32_t o = mxfp4::mxfp4_pack8<float>(res, inv_scale);
    if (!active) return;

    if (col % mx::kLanesPerGroup == 0) {
      if constexpr (COL_MAJOR) {
        scales[(col / mx::kLanesPerGroup) * num_tokens + token] = y_s;
      } else {
        // Row-major: group v / 4.
        scales[v / mx::kLanesPerGroup] = y_s;
      }
    }
    reinterpret_cast<uint32_t*>(out)[v] = o;
  }

 private:
  const scalar_t* __restrict__ input;
  const int num_vecs;
  const int vecs_per_row;
  uint8_t* __restrict__ out;
  float* __restrict__ scales;
  const float epsilon;
  const int num_tokens;
  const float swiglu_limit;
};

}  // namespace vllm

void silu_and_mul_mxfp4_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor& scales,
    int64_t group_size,
    double eps,
    std::optional<double> swiglu_limit) {
  TORCH_CHECK(
      input.dtype() == torch::kFloat16 || input.dtype() == torch::kBFloat16,
      "input must be float16 or bfloat16");
  TORCH_CHECK(
      out.scalar_type() == at::ScalarType::Byte,
      "out must be uint8 (packed FP4)");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(scales.dtype() == torch::kFloat32, "scales must be float32");
  TORCH_CHECK(
      group_size == 32, "MXFP4 requires group_size == 32, got ", group_size);
  TORCH_CHECK(input.dim() == 2, "input must be 2-dimensional");
  TORCH_CHECK(out.dim() == 2, "out must be 2-dimensional");
  TORCH_CHECK(scales.dim() == 2, "scales must be 2-dimensional");

  const int64_t num_tokens = input.size(0);
  const int64_t hidden_size = input.size(-1) / 2;
  const int64_t num_groups = hidden_size / group_size;
  TORCH_CHECK(
      input.size(-1) == hidden_size * 2,
      "input last dim must be 2 * hidden_size");
  TORCH_CHECK(
      hidden_size % group_size == 0,
      "hidden_size must be divisible by group_size");
  TORCH_CHECK(
      out.size(0) == num_tokens && out.size(1) == hidden_size / 2,
      "out must have shape [num_tokens, hidden_size / 2]");
  TORCH_CHECK(
      scales.size(0) == num_tokens && scales.size(1) == num_groups,
      "scales must have logical shape [num_tokens, num_groups]");

  const float limit = vllm::swiglu_limit_value(swiglu_limit);
  if (num_tokens == 0) return;

  const bool is_scale_transposed = scales.stride(0) < scales.stride(1);

  const at::DeviceGuard device_guard(input.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  const int64_t num_vecs = num_tokens * hidden_size / vllm::mx::kVec;
  TORCH_CHECK(
      num_vecs <= std::numeric_limits<int>::max() / 2, "input is too large");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(out.data_ptr()) % 4 == 0,
      "input must be 16-byte and out 4-byte aligned");
  const int64_t wg_size = vllm::mx::kWorkGroupSize;
  const int64_t num_wgs = (num_vecs + wg_size - 1) / wg_size;

  VLLM_DISPATCH_HALF_TYPES(
      input.scalar_type(), "silu_and_mul_mxfp4_quant", [&] {
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        const auto* input_ptr =
            reinterpret_cast<const sycl_t*>(input.data_ptr<scalar_t>());
        float* scales_ptr = scales.data_ptr<float>();
        uint8_t* out_ptr = out.data_ptr<uint8_t>();

        auto launch = [&](auto col_major_tag) {
          vllm::dispatch_swiglu_limit(limit, [&](auto clamp_tag) {
            queue.submit([&](sycl::handler& h) {
              h.parallel_for(
                  sycl::nd_range<1>(num_wgs * wg_size, wg_size),
                  vllm::silu_and_mul_mxfp4_quant_kernel<
                      sycl_t,
                      decltype(col_major_tag)::value,
                      decltype(clamp_tag)::value>(
                      out_ptr,
                      scales_ptr,
                      input_ptr,
                      static_cast<int>(num_vecs),
                      static_cast<int>(hidden_size / vllm::mx::kVec),
                      static_cast<float>(eps),
                      static_cast<int>(num_tokens),
                      vllm::swiglu_limit_as<scalar_t>(limit)));
            });
          });
        };
        if (is_scale_transposed) {
          launch(std::true_type{});
        } else {
          launch(std::false_type{});
        }
      });
}
