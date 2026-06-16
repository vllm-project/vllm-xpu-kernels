// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Router GEMM for MiniMax-M3 (and DeepSeek-style) MoE routers:
//   out[m, n] = sum_k activation[m, k] * weight[n, k]
// activation (mat_a) is bf16 or fp32; weight (mat_b) is always fp32; output is
// always fp32. Shapes: M (tokens) <= 32, N (experts) e.g. 256, K (hidden) e.g.
// 3072. SYCL port of csrc/libtorch_stable/fp32_router_gemm.cu from
// vllm-project/vllm#45381 (m3_release).
//
// Parallelization: one work-group per output column (expert). The K dimension
// is split across ROUTER_GEMM_NSG sub-groups so that many more hardware threads
// run concurrently than the original "one sub-group per column" layout (which
// launched only `num_experts` threads and was ~80x off the memory-bound limit
// because it could not hide load latency). Each lane accumulates float4-vector
// partial sums over its K stripe; partials are reduced within each sub-group,
// staged in shared local memory, and summed across sub-groups before writing
// the column.

#include <sycl/sycl.hpp>
#include "utils.h"
#include <torch/all.h>

namespace vllm {

using bf16_t = sycl::ext::oneapi::bfloat16;

static constexpr int ROUTER_GEMM_SG_SIZE = 16;
static constexpr int ROUTER_GEMM_MAX_TOKENS = 32;
// Sub-groups per expert (K-splits). num_experts * NSG hardware threads run
// concurrently, which is what restores occupancy on BMG/Xe2.
static constexpr int ROUTER_GEMM_NSG = 4;
static constexpr int ROUTER_GEMM_VEC = 16;

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
    const int n_idx = item.get_group(0);  // expert / output column
    auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int wi = item.get_local_id(1);  // 0 .. NSG*SG_SIZE-1
    const int sg_id = wi / ROUTER_GEMM_SG_SIZE;
    constexpr int kThreads = ROUTER_GEMM_NSG * ROUTER_GEMM_SG_SIZE;
    constexpr int kVec = ROUTER_GEMM_VEC;

    const float* b_col = mat_b_ + n_idx * hidden_dim_;

    float acc[ROUTER_GEMM_MAX_TOKENS];
#pragma unroll
    for (int m = 0; m < ROUTER_GEMM_MAX_TOKENS; ++m) acc[m] = 0.0f;

    using AVecB = vllm::xpu::aligned_vec<float, kVec>;
    using AVecA = vllm::xpu::aligned_vec<InputT, kVec>;

    // Vectorized main loop: each thread strides over kVec-aligned chunks. Only
    // safe when every row/column starts kVec-aligned, i.e. hidden_dim % kVec==0
    // (mat_a/mat_b are contiguous with row stride == hidden_dim); otherwise fall
    // back to the scalar path below. Two strided chunks are processed per
    // iteration with independent partial sums (p0/p1) to break the acc[m]
    // dependency chain (ALUWR stall) and overlap two loads (SBID latency).
    const int64_t k_vec_end =
        (hidden_dim_ % kVec == 0) ? (hidden_dim_ / kVec) * kVec : 0;
    const int64_t stride1 = (int64_t)kThreads * kVec;
    int64_t k = (int64_t)wi * kVec;
    for (; k + stride1 < k_vec_end; k += 2 * stride1) {
      const AVecB bv0 = *reinterpret_cast<const AVecB*>(b_col + k);
      const AVecB bv1 = *reinterpret_cast<const AVecB*>(b_col + k + stride1);
      for (int64_t m = 0; m < num_tokens_; ++m) {
        const AVecA av0 =
            *reinterpret_cast<const AVecA*>(mat_a_ + m * hidden_dim_ + k);
        const AVecA av1 = *reinterpret_cast<const AVecA*>(
            mat_a_ + m * hidden_dim_ + k + stride1);
        float p0 = 0.0f, p1 = 0.0f;
#pragma unroll
        for (int v = 0; v < kVec; ++v) {
          p0 += static_cast<float>(av0.val[v]) * bv0.val[v];
          p1 += static_cast<float>(av1.val[v]) * bv1.val[v];
        }
        acc[m] += p0 + p1;
      }
    }
    for (; k < k_vec_end; k += stride1) {
      const AVecB bv = *reinterpret_cast<const AVecB*>(b_col + k);
      for (int64_t m = 0; m < num_tokens_; ++m) {
        const AVecA av =
            *reinterpret_cast<const AVecA*>(mat_a_ + m * hidden_dim_ + k);
        float p = 0.0f;
#pragma unroll
        for (int v = 0; v < kVec; ++v)
          p += static_cast<float>(av.val[v]) * bv.val[v];
        acc[m] += p;
      }
    }
    // Scalar tail for K not divisible by the vector width.
    for (int64_t kt = k_vec_end + wi; kt < hidden_dim_; kt += kThreads) {
      const float b = b_col[kt];
      for (int64_t m = 0; m < num_tokens_; ++m)
        acc[m] += static_cast<float>(mat_a_[m * hidden_dim_ + kt]) * b;
    }

    // Stage per-sub-group partials in SLM, then reduce across sub-groups.
    auto slm = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[ROUTER_GEMM_NSG * ROUTER_GEMM_MAX_TOKENS]>(item.get_group());
    for (int64_t m = 0; m < num_tokens_; ++m) {
      const float part = sycl::reduce_over_group(sg, acc[m], sycl::plus<float>());
      if (lane == 0) slm[sg_id * ROUTER_GEMM_MAX_TOKENS + m] = part;
    }
    sycl::group_barrier(item.get_group());

    if (sg_id == 0) {
      for (int64_t m = lane; m < num_tokens_; m += ROUTER_GEMM_SG_SIZE) {
        float total = 0.0f;
#pragma unroll
        for (int s = 0; s < ROUTER_GEMM_NSG; ++s)
          total += slm[s * ROUTER_GEMM_MAX_TOKENS + m];
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
  constexpr int kWG = ROUTER_GEMM_NSG * ROUTER_GEMM_SG_SIZE;
  sycl::range<2> local(1, kWG);
  sycl::range<2> global(num_experts, kWG);

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
