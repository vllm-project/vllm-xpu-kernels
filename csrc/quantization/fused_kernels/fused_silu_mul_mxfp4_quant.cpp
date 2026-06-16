// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <sycl/sycl.hpp>
#include <torch/all.h>
#include <ATen/DeviceGuard.h>
#include "dispatch_utils.h"
#include "ops.h"
#include "utils.h"
#include "quantization/fp4/mxfp4_quant.h"
#include "quantization/utils.h"

namespace vllm {

// Input  [M, 2*H]   – concatenated (gate, up) activations
// Output [M, H/2]   – packed FP4 nibbles, two per byte (low=even, high=odd)
// Scales [M, H/32]  – float32 UE8M0 (power-of-two) scales
template <typename scalar_t>
class silu_and_mul_mxfp4_quant_kernel {
 public:
  silu_and_mul_mxfp4_quant_kernel(
      uint8_t* __restrict__ out_,
      float* __restrict__ scales_,
      const scalar_t* __restrict__ input_,
      const int hidden_size_,
      const int num_groups_,
      const int group_size_,
      const float epsilon_,
      const int64_t scale_stride_token_,
      const int64_t scale_stride_group_)
      : out(out_),
        scales(scales_),
        input(input_),
        hidden_size(hidden_size_),
        num_groups(num_groups_),
        group_size(group_size_),
        epsilon(epsilon_),
        scale_stride_token(scale_stride_token_),
        scale_stride_group(scale_stride_group_) {}

  void operator()
      [[sycl::reqd_sub_group_size(32)]] (sycl::nd_item<1> item) const {
    const int token_idx = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));
    const int local_range = static_cast<int>(item.get_local_range(0));
    auto sg = item.get_sub_group();
    const int lane = sg.get_local_id()[0];

    constexpr int VEC = 16 / sizeof(scalar_t);
    const int lanes_per_group = group_size / VEC;  // 32/8 = 4
    const int num_chunks = hidden_size / VEC;
    constexpr float FP4_MAX = vllm::mxfp4::FP4_MAX;

    const scalar_t* token_gate =
        input + static_cast<int64_t>(token_idx) * hidden_size * 2;
    const scalar_t* token_up = token_gate + hidden_size;
    uint8_t* token_output =
        out + static_cast<int64_t>(token_idx) * (hidden_size / 2);
    const auto* vgate =
        reinterpret_cast<const vec_n_t<scalar_t, VEC>*>(token_gate);
    const auto* vup = reinterpret_cast<const vec_n_t<scalar_t, VEC>*>(token_up);
    auto* vout = reinterpret_cast<vec_n_t<uint8_t, VEC / 2>*>(token_output);

    for (int base = 0; base < num_chunks; base += local_range) {
      const int c = base + tid;
      const bool active = c < num_chunks;

      float res[VEC];
      float lane_absmax = 0.0f;
      if (active) {
        vec_n_t<scalar_t, VEC> g = vgate[c];
        vec_n_t<scalar_t, VEC> u = vup[c];
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
          const float gf = static_cast<float>(g.val[k]);
          const float uf = static_cast<float>(u.val[k]);
          const float r = (gf / (1.0f + sycl::exp(-gf))) * uf;
          res[k] = r;
          lane_absmax = sycl::max(lane_absmax, sycl::fabs(r));
        }
      }

      float gmax = lane_absmax;
#pragma unroll
      for (int off = 1; off < lanes_per_group; off <<= 1) {
        gmax = sycl::max(gmax, sycl::permute_group_by_xor(sg, gmax, off));
      }

      float y_s = gmax / FP4_MAX;
      y_s = sycl::exp2(
          sycl::ceil(sycl::log2(sycl::fmax(sycl::fabs(y_s), epsilon))));

      if (active && (lane % lanes_per_group) == 0) {
        const int g_idx = c / lanes_per_group;
        const int64_t scale_idx =
            static_cast<int64_t>(token_idx) * scale_stride_token +
            static_cast<int64_t>(g_idx) * scale_stride_group;
        scales[scale_idx] = y_s;
      }

      if (active) {
        const float inv_scale = 1.0f / y_s;
        vec_n_t<uint8_t, VEC / 2> o;
#pragma unroll
        for (int k = 0; k < VEC; k += 2) {
          float s0 =
              sycl::fmax(-FP4_MAX, sycl::fmin(res[k] * inv_scale, FP4_MAX));
          float s1 =
              sycl::fmax(-FP4_MAX, sycl::fmin(res[k + 1] * inv_scale, FP4_MAX));
          uint8_t n0 = vllm::mxfp4::float_to_fp4_e2m1(s0);
          uint8_t n1 = vllm::mxfp4::float_to_fp4_e2m1(s1);
          o.val[k / 2] =
              static_cast<uint8_t>(((n1 & 0x0Fu) << 4) | (n0 & 0x0Fu));
        }
        vout[c] = o;
      }
    }
  }

 private:
  uint8_t* __restrict__ out;
  float* __restrict__ scales;
  const scalar_t* __restrict__ input;
  const int hidden_size;
  const int num_groups;
  const int group_size;
  const float epsilon;
  const int64_t scale_stride_token;
  const int64_t scale_stride_group;
};

}  // namespace vllm

void silu_and_mul_mxfp4_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor& scales,
    int64_t group_size,
    double eps) {
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

  if (num_tokens == 0) return;

  const bool is_scale_transposed = scales.stride(0) < scales.stride(1);
  const int64_t scale_stride_token =
      is_scale_transposed ? 1LL : static_cast<int64_t>(num_groups);
  const int64_t scale_stride_group = is_scale_transposed ? num_tokens : 1LL;

  const at::DeviceGuard device_guard(input.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  const int64_t num_wgs = num_tokens;
  const int64_t num_chunks = hidden_size / 8;  // 16 bytes / 2-byte half
  int64_t wg_size = std::min<int64_t>(((num_chunks + 31) / 32) * 32, 1024);
  if (wg_size < 32) wg_size = 32;

  VLLM_DISPATCH_HALF_TYPES(
      input.scalar_type(), "silu_and_mul_mxfp4_quant", [&] {
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        const auto* input_ptr =
            reinterpret_cast<const sycl_t*>(input.data_ptr<scalar_t>());
        float* scales_ptr = scales.data_ptr<float>();
        uint8_t* out_ptr = out.data_ptr<uint8_t>();

        queue.submit([&](sycl::handler& h) {
          h.parallel_for(
              sycl::nd_range<1>(num_wgs * wg_size, wg_size),
              vllm::silu_and_mul_mxfp4_quant_kernel<sycl_t>(
                  out_ptr,
                  scales_ptr,
                  input_ptr,
                  static_cast<int>(hidden_size),
                  static_cast<int>(num_groups),
                  static_cast<int>(group_size),
                  static_cast<float>(eps),
                  scale_stride_token,
                  scale_stride_group));
        });
      });
}
