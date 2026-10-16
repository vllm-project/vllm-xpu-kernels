// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <sycl/sycl.hpp>
#include <torch/all.h>

#include <type_traits>

#include "dispatch_utils.h"
#include "utils.h"
#include "quantization/fp8/quant_utils.h"
#include "quantization/fused_kernels/quant_conversions.h"
#include "quantization/moe_scale_layout.h"
#include "quantization/utils.h"

namespace vllm {

template <typename scalar_t, typename out_t, int CPT>
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
      const int64_t scale_stride_group_,
      const bool scale_ue8m0_,
      const MoEScaleLayout moe_layout_ = {})
      : out(out_),
        scales(scales_),
        input(input_),
        scale_ub(scale_ub_),
        hidden_size(hidden_size_),
        num_groups(num_groups_),
        group_size(group_size_),
        scale_stride_token(scale_stride_token_),
        scale_stride_group(scale_stride_group_),
        scale_ue8m0(scale_ue8m0_),
        moe_layout(moe_layout_) {}

  void operator()
      [[sycl::reqd_sub_group_size(32)]] (sycl::nd_item<1> item) const {
    const int token_idx = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));
    const int local_range = static_cast<int>(item.get_local_range(0));
    auto sg = item.get_sub_group();
    const int lane = sg.get_local_id()[0];

    constexpr int VEC = 16 / sizeof(scalar_t);
    const int chunks_per_group = group_size / VEC;
    const int lanes_per_group = chunks_per_group / CPT;
    const int num_chunks = hidden_size / VEC;
    const int threads_per_row = num_chunks / CPT;

    const scalar_t* token_gate =
        input + static_cast<int64_t>(token_idx) * hidden_size * 2;
    const scalar_t* token_up = token_gate + hidden_size;
    out_t* token_output = out + static_cast<int64_t>(token_idx) * hidden_size;
    const auto* vgate =
        reinterpret_cast<const vec_n_t<scalar_t, VEC>*>(token_gate);
    const auto* vup = reinterpret_cast<const vec_n_t<scalar_t, VEC>*>(token_up);
    auto* vout = reinterpret_cast<vec_n_t<out_t, VEC>*>(token_output);

    for (int base = 0; base < threads_per_row; base += local_range) {
      const int t = base + tid;
      const bool active = t < threads_per_row;
      const int c0 = t * CPT;

      float res[CPT][VEC];
      float lane_absmax = 0.0f;
      if (active) {
#pragma unroll
        for (int j = 0; j < CPT; ++j) {
          vec_n_t<scalar_t, VEC> g = vgate[c0 + j];
          vec_n_t<scalar_t, VEC> u = vup[c0 + j];
#pragma unroll
          for (int k = 0; k < VEC; ++k) {
            const float gf = static_cast<float>(g.val[k]);
            const float uf = static_cast<float>(u.val[k]);
            const float r = (gf / (1.0f + sycl::exp(-gf))) * uf;
            res[j][k] = r;
            lane_absmax = sycl::max(lane_absmax, sycl::fabs(r));
          }
        }
      }

      // Segmented max over the lanes_per_group consecutive lanes of a group.
      float gmax = lane_absmax;
#pragma unroll
      for (int off = 1; off < lanes_per_group; off <<= 1) {
        gmax = sycl::max(gmax, sycl::permute_group_by_xor(sg, gmax, off));
      }

      float group_scale;
      if constexpr (std::is_same_v<out_t, int8_t>) {
        group_scale =
            std::max(gmax / 127.0f, std::numeric_limits<float>::epsilon());
      } else {
        const float fp8_max = static_cast<float>(fp8::quant_type_max_v<out_t>);
        group_scale =
            sycl::max(gmax / fp8_max, fp8::min_scaling_factor<out_t>::val());
        if (scale_ub != nullptr) {
          group_scale = sycl::min(group_scale, *scale_ub);
          group_scale =
              sycl::max(group_scale, fp8::min_scaling_factor<out_t>::val());
        }
        if (scale_ue8m0) {
          group_scale = sycl::exp2(
              sycl::ceil(
                  sycl::log2(sycl::fmax(sycl::fabs(group_scale), 1e-10f))));
        }
      }

      if (active && (lanes_per_group <= 1 || (lane % lanes_per_group) == 0)) {
        const int g_idx = c0 / chunks_per_group;
        if (moe_layout.enabled()) {
          moe_layout.store(token_idx, g_idx, group_scale);
        } else {
          const int64_t scale_idx =
              static_cast<int64_t>(token_idx) * scale_stride_token +
              static_cast<int64_t>(g_idx) * scale_stride_group;
          scales[scale_idx] = group_scale;
        }
      }

      if (active) {
#pragma unroll
        for (int j = 0; j < CPT; ++j) {
          vec_n_t<out_t, VEC> o;
#pragma unroll
          for (int k = 0; k < VEC; ++k) {
            o.val[k] = vllm::ScaledQuant<out_t, false>::quant_fn(
                res[j][k], group_scale);
          }
          vout[c0 + j] = o;
        }
      }
    }
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
  const bool scale_ue8m0;
  const MoEScaleLayout moe_layout;
};

}  // namespace vllm

void silu_and_mul_per_block_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor& scales,
    int64_t group_size,
    std::optional<torch::Tensor> scale_ub,
    bool is_scale_transposed,
    bool scale_ue8m0,
    std::optional<torch::Tensor> expert_scale_desc) {
  TORCH_CHECK(
      out.dtype() == torch::kFloat8_e4m3fn ||
          out.dtype() == torch::kFloat8_e5m2 || out.dtype() == torch::kInt8,
      "out must be float8_e4m3fn, float8_e5m2, or int8");
  TORCH_CHECK(
      input.dtype() == torch::kFloat16 || input.dtype() == torch::kBFloat16,
      "input must be float16 or bfloat16");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(
      expert_scale_desc.has_value() || scales.dtype() == torch::kFloat32,
      "scales must be float32");
  TORCH_CHECK(
      !expert_scale_desc.has_value() || scale_ue8m0,
      "expert_scale_desc emits raw e8m0 scales, so scale_ue8m0 must be set");
  TORCH_CHECK(
      !expert_scale_desc.has_value() || !is_scale_transposed,
      "expert_scale_desc owns the scale layout, so is_scale_transposed must "
      "be false");
  TORCH_CHECK(
      group_size == 32 || group_size == 64 || group_size == 128,
      "group_size must be 32, 64, or 128, got ",
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
      scales.size(1) == num_groups &&
          (expert_scale_desc.has_value() ? scales.size(0) >= num_tokens
                                         : scales.size(0) == num_tokens),
      "scales must have logical shape [num_tokens, num_groups], or "
      "[padded_scale_rows, num_groups] with expert_scale_desc");

  if (num_tokens == 0) return;

  const auto moe_layout = vllm::make_moe_scale_layout(
      scales, expert_scale_desc, static_cast<int>(num_groups), num_tokens);

  const int64_t scale_stride_token =
      is_scale_transposed ? 1LL : static_cast<int64_t>(num_groups);
  const int64_t scale_stride_group = is_scale_transposed ? num_tokens : 1LL;

  const at::DeviceGuard device_guard(input.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  const float* scale_ub_ptr =
      scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr;

  const int64_t num_wgs = num_tokens;
  // The kernel is instantiated for half types only, so a 16-byte access is
  // always 8 elements wide.
  constexpr int vec_width = 8;
  const int64_t num_chunks = hidden_size / vec_width;
  const int64_t chunks_per_group = group_size / vec_width;
  const int eu_count = static_cast<int>(
      queue.get_device().get_info<sycl::info::device::max_compute_units>());

  VLLM_DISPATCH_HALF_TYPES(
      input.scalar_type(), "silu_and_mul_per_block_quant", [&] {
        using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        const auto* input_ptr =
            reinterpret_cast<const sycl_t*>(input.data_ptr<scalar_t>());
        float* scales_ptr =
            moe_layout.enabled() ? nullptr : scales.data_ptr<float>();

        auto launch = [&](auto out_ptr, auto cpt_tag) {
          using out_t = std::remove_pointer_t<decltype(out_ptr)>;
          constexpr int CPT = decltype(cpt_tag)::value;
          const int64_t wg_size =
              vllm::fused_quant_block_size(num_chunks / CPT, 1, false);
          queue.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<1>(num_wgs * wg_size, wg_size),
                vllm::silu_and_mul_per_block_quant_kernel<sycl_t, out_t, CPT>(
                    out_ptr,
                    scales_ptr,
                    input_ptr,
                    scale_ub_ptr,
                    static_cast<int>(hidden_size),
                    static_cast<int>(num_groups),
                    static_cast<int>(group_size),
                    scale_stride_token,
                    scale_stride_group,
                    scale_ue8m0,
                    moe_layout));
          });
        };

        auto dispatch = [&](auto out_ptr) {
          vllm::dispatch_quant_chunks(
              num_tokens,
              num_chunks,
              chunks_per_group,
              eu_count,
              [&](auto cpt_tag) { launch(out_ptr, cpt_tag); });
        };

        if (out.dtype() == torch::kFloat8_e4m3fn) {
          dispatch(out.data_ptr<at::Float8_e4m3fn>());
        } else if (out.dtype() == torch::kFloat8_e5m2) {
          dispatch(out.data_ptr<at::Float8_e5m2>());
        } else {
          dispatch(out.data_ptr<int8_t>());
        }
      });
}
