// DeepSeek V4 HC head fused kernel — optimized single-kernel implementation.

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

static inline float sigmoid(float x) {
  return 1.f / (1.f + sycl::native::exp(-x));
}

class HcHeadFusedOptKernel {
 public:
  static constexpr int BLOCK_M = 2;
  static constexpr int VEC_DOT = 8;
  static constexpr int VEC_OUT = 8;
  static constexpr int SG_SIZE = 16;
  static constexpr int WG_SIZE = 256;
  static constexpr int NUM_SG = WG_SIZE / SG_SIZE;
  static constexpr int NUM_RED = HC + 1;

  struct Params {
    const bf16* hs_flat;
    const float* fn;
    const float* hc_scale;
    const float* hc_base;
    bf16* out;
    int N;
    int H;
    int hc_h;
    float rms_eps;
    float hc_eps;
  };

  CUTLASS_DEVICE
  void operator()(const Params& p, char* smem_buf) const {
    using vec_bf16_dot_t = aligned_vec<bf16, VEC_DOT>;
    using vec_f32_dot_t = aligned_vec<float, VEC_DOT>;
    using vec_bf16_out_t = aligned_vec<bf16, VEC_OUT>;

    float* red_scratch = reinterpret_cast<float*>(smem_buf);
    float* pre_s = red_scratch + NUM_RED * NUM_SG * BLOCK_M;

    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int wg_id = int(BlockIdxX());
    const int tid = int(ThreadIdxY());
    auto sg = it.get_sub_group();
    const int sg_id = sg.get_group_id()[0];
    const int sg_lane = sg.get_local_id()[0];
    const int tok_base = wg_id * BLOCK_M;

    float local_mix[BLOCK_M][HC];
    float local_sq[BLOCK_M];
#pragma unroll
    for (int t = 0; t < BLOCK_M; ++t) {
      local_sq[t] = 0.f;
#pragma unroll
      for (int j = 0; j < HC; ++j)
        local_mix[t][j] = 0.f;
    }

    for (int k = tid * VEC_DOT; k < p.hc_h; k += WG_SIZE * VEC_DOT) {
      vec_f32_dot_t fn_tile[HC];
#pragma unroll
      for (int j = 0; j < HC; ++j)
        fn_tile[j] =
            *reinterpret_cast<const vec_f32_dot_t*>(p.fn + j * p.hc_h + k);

#pragma unroll
      for (int t = 0; t < BLOCK_M; ++t) {
        int tok = tok_base + t;
        if (tok >= p.N) break;
        auto xv = *reinterpret_cast<const vec_bf16_dot_t*>(
            p.hs_flat + tok * p.hc_h + k);
        float xf[VEC_DOT];
#pragma unroll
        for (int v = 0; v < VEC_DOT; ++v)
          xf[v] = float(xv.val[v]);
#pragma unroll
        for (int v = 0; v < VEC_DOT; ++v)
          local_sq[t] += xf[v] * xf[v];
#pragma unroll
        for (int j = 0; j < HC; ++j) {
          float acc = 0.f;
#pragma unroll
          for (int v = 0; v < VEC_DOT; ++v)
            acc += xf[v] * fn_tile[j].val[v];
          local_mix[t][j] += acc;
        }
      }
    }

#pragma unroll
    for (int t = 0; t < BLOCK_M; ++t) {
      int tok = tok_base + t;
      if (tok >= p.N) break;
      float* tok_scratch = &red_scratch[t * NUM_RED * NUM_SG];
      float sg_sq =
          sycl::reduce_over_group(sg, local_sq[t], sycl::plus<float>());
      if (sg_lane == 0) tok_scratch[sg_id * NUM_RED] = sg_sq;
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        float sg_mix =
            sycl::reduce_over_group(sg, local_mix[t][j], sycl::plus<float>());
        if (sg_lane == 0) tok_scratch[sg_id * NUM_RED + j + 1] = sg_mix;
      }
      sycl::group_barrier(it.get_group());

      if (sg_id == 0 && sg_lane < NUM_RED) {
        float s = 0.f;
#pragma unroll
        for (int si = 0; si < NUM_SG; ++si)
          s += tok_scratch[si * NUM_RED + sg_lane];
        tok_scratch[sg_lane] = s;
      }
      sycl::group_barrier(it.get_group());

      float rrms = sycl::native::rsqrt(
          tok_scratch[0] / static_cast<float>(p.hc_h) + p.rms_eps);
      if (tid < HC) {
        float mix_val = tok_scratch[tid + 1] * rrms;
        pre_s[t * HC + tid] =
            sigmoid(sycl::fma(mix_val, p.hc_scale[0], p.hc_base[tid])) +
            p.hc_eps;
      }
    }
    sycl::group_barrier(it.get_group());

#pragma unroll
    for (int t = 0; t < BLOCK_M; ++t) {
      int tok = tok_base + t;
      if (tok >= p.N) break;

      float pre_reg[HC];
#pragma unroll
      for (int m = 0; m < HC; ++m)
        pre_reg[m] = pre_s[t * HC + m];

      for (int k = tid * VEC_OUT; k < p.H; k += WG_SIZE * VEC_OUT) {
        float acc[VEC_OUT];
#pragma unroll
        for (int v = 0; v < VEC_OUT; ++v)
          acc[v] = 0.f;
#pragma unroll
        for (int m = 0; m < HC; ++m) {
          auto rv = *reinterpret_cast<const vec_bf16_out_t*>(
              p.hs_flat + (tok * HC + m) * p.H + k);
#pragma unroll
          for (int v = 0; v < VEC_OUT; ++v)
            acc[v] += pre_reg[m] * float(rv.val[v]);
        }
        vec_bf16_out_t ov;
#pragma unroll
        for (int v = 0; v < VEC_OUT; ++v)
          ov.val[v] = static_cast<bf16>(acc[v]);
        *reinterpret_cast<vec_bf16_out_t*>(p.out + tok * p.H + k) = ov;
      }
    }
  }
};

void launch_hc_head_fused_opt(
    sycl::queue& q,
    const bf16* hs_flat,
    const float* fn,
    const float* hc_scale,
    const float* hc_base,
    bf16* out,
    int N,
    int H,
    float rms_eps,
    float hc_eps) {
  static constexpr int BLOCK_M = HcHeadFusedOptKernel::BLOCK_M;
  static constexpr int WG_SIZE = HcHeadFusedOptKernel::WG_SIZE;
  const int hc_h = HC * H;

  const size_t num_wgs = static_cast<size_t>((N + BLOCK_M - 1) / BLOCK_M);

  HcHeadFusedOptKernel::Params params{
      hs_flat, fn, hc_scale, hc_base, out, N, H, hc_h, rms_eps, hc_eps};

  const int smem_size = static_cast<int>(
      (HcHeadFusedOptKernel::NUM_RED * HcHeadFusedOptKernel::NUM_SG *
           HcHeadFusedOptKernel::BLOCK_M +
       HC * HcHeadFusedOptKernel::BLOCK_M) *
      sizeof(float));

  const auto sycl_block = compat::dim3(1, WG_SIZE, 1);
  const auto sycl_grid = compat::dim3(static_cast<unsigned int>(num_wgs), 1, 1);

  compat::experimental::launch_properties launch_props{
      sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
  };
  compat::experimental::kernel_properties kernel_props{
      sycl::ext::oneapi::experimental::sub_group_size<16>};
  compat::experimental::launch_policy policy{
      sycl_grid, sycl_block, launch_props, kernel_props};
  compat::experimental::launch<
      cutlass::device_kernel<HcHeadFusedOptKernel>,
      HcHeadFusedOptKernel>(policy, q, params);
}

}  // namespace

void hc_head_fused(
    const at::Tensor& hs_flat,
    const at::Tensor& fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    at::Tensor& out,
    double rms_eps,
    double hc_eps) {
  TORCH_CHECK(
      hs_flat.is_xpu() && fn.is_xpu() && hc_scale.is_xpu() &&
          hc_base.is_xpu() && out.is_xpu(),
      "hc_head_fused: tensors must be on XPU");
  TORCH_CHECK(
      hs_flat.scalar_type() == at::kBFloat16, "hs_flat must be bfloat16");
  TORCH_CHECK(fn.scalar_type() == at::kFloat, "fn must be float32");
  TORCH_CHECK(hc_scale.scalar_type() == at::kFloat, "hc_scale must be float32");
  TORCH_CHECK(hc_base.scalar_type() == at::kFloat, "hc_base must be float32");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be bfloat16");
  TORCH_CHECK(out.is_contiguous(), "hc_head_fused: out must be contiguous");

  const int64_t N = hs_flat.size(0);
  const int64_t H = hs_flat.size(2);
  TORCH_CHECK(
      H % 8 == 0,
      "hc_head_fused: hidden_size must be a multiple of 8 for vectorized "
      "loads/stores");

  TORCH_CHECK(hs_flat.dim() == 3, "hc_head_fused: hs_flat must be [N, HC, H]");
  TORCH_CHECK(
      hs_flat.size(1) == HC, "hc_head_fused: hs_flat dim1 must be HC=4");
  TORCH_CHECK(
      fn.size(0) == HC && fn.size(1) == HC * H,
      "hc_head_fused: fn shape mismatch");
  TORCH_CHECK(
      hc_scale.numel() == 1, "hc_head_fused: hc_scale must have 1 element");
  TORCH_CHECK(hc_base.numel() == HC, "hc_head_fused: hc_base size mismatch");
  TORCH_CHECK(
      out.dim() == 2 && out.size(0) == N && out.size(1) == H,
      "hc_head_fused: out shape mismatch");

  if (N == 0) return;

  auto hs_c = hs_flat.contiguous();
  auto fn_c = fn.contiguous();
  auto hc_scale_c = hc_scale.contiguous();
  auto hc_base_c = hc_base.contiguous();
  auto& q = vllm::xpu::vllmGetQueue();
  launch_hc_head_fused_opt(
      q,
      reinterpret_cast<const bf16*>(hs_c.data_ptr()),
      fn_c.data_ptr<float>(),
      hc_scale_c.data_ptr<float>(),
      hc_base_c.data_ptr<float>(),
      reinterpret_cast<bf16*>(out.data_ptr()),
      static_cast<int>(N),
      static_cast<int>(H),
      static_cast<float>(rms_eps),
      static_cast<float>(hc_eps));
}
