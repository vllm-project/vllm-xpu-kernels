// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Router GEMM for MiniMax-M3 (and DeepSeek-style) MoE routers:
//   out[m, n] = sum_k activation[m, k] * weight[n, k]
// activation (mat_a) is bf16 or fp32; weight (mat_b) is always fp32; output is
// always fp32. Shapes: M (tokens) <= 32, N (experts) e.g. 256, K (hidden) e.g.
// 3072. SYCL port of csrc/libtorch_stable/fp32_router_gemm.cu from
// vllm-project/vllm#45381 (m3_release): one sub-group per output column, each
// lane strides over K and accumulates one partial sum per token, then the
// sub-group reduces and lane 0 writes the column.

#include <sycl/sycl.hpp>
#include "utils.h"
#include <torch/all.h>

namespace vllm {

using bf16_t = sycl::ext::oneapi::bfloat16;

static constexpr int ROUTER_GEMM_SG_SIZE = 16;
static constexpr int ROUTER_GEMM_MAX_TOKENS = 32;

template <typename InputT>
class fp32_router_gemm_kernel {
 public:
  fp32_router_gemm_kernel(
      float* __restrict__ out,
      const InputT* __restrict__ mat_a,
      const float* __restrict__ mat_b,
      int64_t num_tokens,
      int64_t num_experts,
      int64_t hidden_dim)
      : out_(out),
        mat_a_(mat_a),
        mat_b_(mat_b),
        num_tokens_(num_tokens),
        num_experts_(num_experts),
        hidden_dim_(hidden_dim) {}

  [[sycl::reqd_sub_group_size(ROUTER_GEMM_SG_SIZE)]] void operator()(
      sycl::nd_item<2> item) const {
    const int n_idx = item.get_global_id(0);  // expert / output column
    if (n_idx >= num_experts_) return;
    auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int sg_size = sg.get_max_local_range()[0];

    const float* b_col = mat_b_ + n_idx * hidden_dim_;

    float acc[ROUTER_GEMM_MAX_TOKENS];
#pragma unroll
    for (int m = 0; m < ROUTER_GEMM_MAX_TOKENS; ++m) acc[m] = 0.0f;

    // Each lane handles a strided slice of the K dimension.
    for (int64_t k = lane; k < hidden_dim_; k += sg_size) {
      const float b = b_col[k];
      for (int64_t m = 0; m < num_tokens_; ++m) {
        const float a = static_cast<float>(mat_a_[m * hidden_dim_ + k]);
        acc[m] += a * b;
      }
    }

    for (int64_t m = 0; m < num_tokens_; ++m) {
      const float total =
          sycl::reduce_over_group(sg, acc[m], sycl::plus<float>());
      if (lane == 0) {
        out_[m * num_experts_ + n_idx] = total;
      }
    }
  }

 private:
  float* __restrict__ out_;
  const InputT* __restrict__ mat_a_;
  const float* __restrict__ mat_b_;
  int64_t num_tokens_;
  int64_t num_experts_;
  int64_t hidden_dim_;
};

template <typename InputT>
static void launch_fp32_router_gemm(
    torch::Tensor& out,
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t hidden_dim) {
  sycl::range<2> local(1, ROUTER_GEMM_SG_SIZE);
  sycl::range<2> global(num_experts, ROUTER_GEMM_SG_SIZE);

  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>(global, local),
        vllm::fp32_router_gemm_kernel<InputT>(
            out.data_ptr<float>(),
            reinterpret_cast<const InputT*>(mat_a.data_ptr()),
            mat_b.data_ptr<float>(),
            num_tokens,
            num_experts,
            hidden_dim));
  });
}

}  // namespace vllm

torch::Tensor fp32_router_gemm(
    const torch::Tensor& mat_a,  // [num_tokens, hidden] activation (bf16/fp32)
    const torch::Tensor& mat_b)  // [num_experts, hidden] weight (fp32)
{
  TORCH_CHECK(mat_a.dim() == 2, "mat_a must be 2-D");
  TORCH_CHECK(mat_b.dim() == 2, "mat_b must be 2-D");
  TORCH_CHECK(mat_a.is_contiguous(), "mat_a must be contiguous");
  TORCH_CHECK(mat_b.is_contiguous(), "mat_b must be contiguous");
  TORCH_CHECK(mat_b.dtype() == torch::kFloat32, "weight must be fp32");
  TORCH_CHECK(
      mat_a.size(1) == mat_b.size(1),
      "mat_a and mat_b must share the hidden (K) dimension");

  const int64_t num_tokens = mat_a.size(0);
  const int64_t hidden_dim = mat_a.size(1);
  const int64_t num_experts = mat_b.size(0);
  TORCH_CHECK(
      num_tokens <= vllm::ROUTER_GEMM_MAX_TOKENS,
      "fp32_router_gemm supports at most ",
      vllm::ROUTER_GEMM_MAX_TOKENS,
      " tokens (got ",
      num_tokens,
      ")");

  at::DeviceGuard dg(mat_a.device());
  auto out = torch::empty(
      {num_tokens, num_experts},
      mat_a.options().dtype(torch::kFloat32));

  if (mat_a.dtype() == torch::kBFloat16) {
    vllm::launch_fp32_router_gemm<vllm::bf16_t>(
        out, mat_a, mat_b, num_tokens, num_experts, hidden_dim);
  } else if (mat_a.dtype() == torch::kFloat32) {
    vllm::launch_fp32_router_gemm<float>(
        out, mat_a, mat_b, num_tokens, num_experts, hidden_dim);
  } else {
    TORCH_CHECK(false, "activation must be bf16 or fp32");
  }
  return out;
}
