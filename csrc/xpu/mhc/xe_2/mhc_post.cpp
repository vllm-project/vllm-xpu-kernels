// DeepSeek V4 MHC post-processing kernel — optimized implementation.
//
// Computes:
//   out[tok, o, h] = sum_i(comb_res_mix[tok, i, o] * residual[tok, i, h])
//                  + post_layer_mix[tok, o] * x[tok, h]

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "utils.h"

#include "cutlass/device_kernel.h"
#include "cute/util/compat/device.hpp"
#include "cute/util/compat/dims.hpp"
#include "cute/util/compat/launch_policy.hpp"

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;
static constexpr int HC = 4;

using vllm::xpu::aligned_vec;

class MhcPostOptKernel {
 public:
  static constexpr int BLOCK_M = 1;
  static constexpr int VEC = 8;
  static constexpr int SG_SIZE = 16;
  static constexpr int WG_SIZE = 256;
  static constexpr int THREADS_PER_TOK = WG_SIZE / BLOCK_M;

  struct Params {
    const float* __restrict comb_res_mix;
    const bf16* __restrict residual;
    const float* __restrict post_layer_mix;
    const bf16* __restrict x;
    bf16* __restrict out;
    int N;
    int H;
  };

  CUTLASS_DEVICE
  void operator()(const Params& p, char*) const {
    using vec_bf16_t = aligned_vec<bf16, VEC>;

    const int wg_id = int(BlockIdxX());
    const int tid = int(ThreadIdxY());
    const int tok_base = wg_id * BLOCK_M;
    const int t = tid / THREADS_PER_TOK;
    const int local_k = tid % THREADS_PER_TOK;
    const int tok = tok_base + t;
    if (tok >= p.N) return;

    float a_reg[HC][HC];
    float c_reg[HC];
#pragma unroll
    for (int i = 0; i < HC; ++i) {
#pragma unroll
      for (int j = 0; j < HC; ++j)
        a_reg[i][j] = p.comb_res_mix[tok * HC * HC + i * HC + j];
      c_reg[i] = p.post_layer_mix[tok * HC + i];
    }

    for (int k = local_k * VEC; k < p.H; k += THREADS_PER_TOK * VEC) {
      auto xv = *reinterpret_cast<const vec_bf16_t*>(p.x + tok * p.H + k);
      float acc[HC][VEC];
#pragma unroll
      for (int o = 0; o < HC; ++o)
#pragma unroll
        for (int v = 0; v < VEC; ++v)
          acc[o][v] = c_reg[o] * float(xv.val[v]);

#pragma unroll
      for (int i = 0; i < HC; ++i) {
        auto rv = *reinterpret_cast<const vec_bf16_t*>(
            p.residual + (tok * HC + i) * p.H + k);
#pragma unroll
        for (int o = 0; o < HC; ++o) {
          float a_val = a_reg[i][o];
#pragma unroll
          for (int v = 0; v < VEC; ++v)
            acc[o][v] += a_val * float(rv.val[v]);
        }
      }

#pragma unroll
      for (int o = 0; o < HC; ++o) {
        vec_bf16_t ov;
#pragma unroll
        for (int v = 0; v < VEC; ++v)
          ov.val[v] = static_cast<bf16>(acc[o][v]);
        *reinterpret_cast<vec_bf16_t*>(p.out + (tok * HC + o) * p.H + k) = ov;
      }
    }
  }
};

}  // namespace

void launch_mhc_post_opt(
    sycl::queue& q,
    const float* __restrict comb_res_mix,
    const bf16* __restrict residual,
    const float* __restrict post_layer_mix,
    const bf16* __restrict x,
    bf16* __restrict out,
    int N,
    int H) {
  static constexpr int BLOCK_M = MhcPostOptKernel::BLOCK_M;
  static constexpr int WG_SIZE = MhcPostOptKernel::WG_SIZE;
  const size_t num_wgs = static_cast<size_t>((N + BLOCK_M - 1) / BLOCK_M);

  MhcPostOptKernel::Params params{
      comb_res_mix, residual, post_layer_mix, x, out, N, H};

  const auto sycl_block = compat::dim3(1, WG_SIZE, 1);
  const auto sycl_grid = compat::dim3(static_cast<unsigned int>(num_wgs), 1, 1);

  compat::experimental::launch_properties launch_props{
      sycl::ext::oneapi::experimental::work_group_scratch_size(0),
  };
  compat::experimental::kernel_properties kernel_props{
      sycl::ext::oneapi::experimental::sub_group_size<16>};
  compat::experimental::launch_policy policy{
      sycl_grid, sycl_block, launch_props, kernel_props};
  compat::experimental::
      launch<cutlass::device_kernel<MhcPostOptKernel>, MhcPostOptKernel>(
          policy, q, params);
}

at::Tensor mhc_post(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post_layer_mix,
    const at::Tensor& comb_res_mix) {
  TORCH_CHECK(
      x.is_xpu() && residual.is_xpu() && post_layer_mix.is_xpu() &&
          comb_res_mix.is_xpu(),
      "mhc_post: tensors must be on XPU");
  TORCH_CHECK(
      residual.scalar_type() == at::kBFloat16, "residual must be bfloat16");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(
      post_layer_mix.scalar_type() == at::kFloat,
      "post_layer_mix must be float32");
  TORCH_CHECK(
      comb_res_mix.scalar_type() == at::kFloat, "comb_res_mix must be float32");

  auto residual_c = residual.contiguous();
  auto x_c = x.contiguous();
  auto a_c = comb_res_mix.contiguous();
  auto c_c = post_layer_mix.squeeze(-1).contiguous();

  TORCH_CHECK(residual_c.dim() == 3, "mhc_post: residual must be [N, HC, H]");
  TORCH_CHECK(x_c.dim() == 2, "mhc_post: x must be [N, H]");
  TORCH_CHECK(
      c_c.dim() == 2, "mhc_post: post_layer_mix must be [N, HC] or [N, HC, 1]");
  TORCH_CHECK(a_c.dim() == 3, "mhc_post: comb_res_mix must be [N, HC, HC]");
  TORCH_CHECK(residual_c.size(0) == x_c.size(0), "mhc_post: batch mismatch");
  TORCH_CHECK(
      residual_c.size(0) == c_c.size(0),
      "mhc_post: post_layer_mix batch mismatch");
  TORCH_CHECK(
      residual_c.size(0) == a_c.size(0),
      "mhc_post: comb_res_mix batch mismatch");
  TORCH_CHECK(
      residual_c.size(1) == HC, "mhc_post: residual dim -2 must be HC=4");
  TORCH_CHECK(c_c.size(1) == HC, "mhc_post: post_layer_mix shape mismatch");
  TORCH_CHECK(
      a_c.size(1) == HC && a_c.size(2) == HC,
      "mhc_post: comb_res_mix shape mismatch");
  TORCH_CHECK(
      residual_c.size(2) == x_c.size(1), "mhc_post: hidden size mismatch");

  const int64_t N = residual_c.size(0);
  const int64_t H = residual_c.size(2);
  TORCH_CHECK(
      H % 8 == 0,
      "mhc_post: hidden_size must be a multiple of 8 for vectorized "
      "loads/stores");
  auto out = at::empty_like(residual_c);
  if (N == 0) return out;

  auto& q = vllm::xpu::vllmGetQueue();
  launch_mhc_post_opt(
      q,
      a_c.data_ptr<float>(),
      reinterpret_cast<const bf16*>(residual_c.data_ptr()),
      c_c.data_ptr<float>(),
      reinterpret_cast<const bf16*>(x_c.data_ptr()),
      reinterpret_cast<bf16*>(out.data_ptr()),
      static_cast<int>(N),
      static_cast<int>(H));

  return out;
}
