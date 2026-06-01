#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>

#include <cstdint>
#include "utils.h"
namespace vllm {

using bf16 = sycl::ext::oneapi::bfloat16;

static constexpr int HC = 4;
static constexpr int SG = 16;
static constexpr int WG_THREADS = 128;
static constexpr int VEC = 8;
static constexpr int N_SG = WG_THREADS / SG;

template <int N>
static inline sycl::vec<bf16, N> vload_bf16(const bf16* p) {
  return *reinterpret_cast<const sycl::vec<bf16, N>*>(p);
}
template <int N>
static inline void vstore_bf16(bf16* p, const sycl::vec<bf16, N>& v) {
  *reinterpret_cast<sycl::vec<bf16, N>*>(p) = v;
}

static inline float fast_sigmoid(float x) {
  return 1.f / (1.f + sycl::native::exp(-x));
}

template <int COUNT, int N_SG_V, int SG_SIZE = 32>
static inline void wg_reduce_sum_batch(
    sycl::nd_item<1>& it,
    const float (&partials)[COUNT],
    float* scratch,
    float (&results)[COUNT]) {
  auto sg = it.get_sub_group();
  const int sg_id = sg.get_group_id()[0];
  const int sg_lid = sg.get_local_id()[0];

#pragma unroll
  for (int j = 0; j < COUNT; ++j) {
    float s = sycl::reduce_over_group(sg, partials[j], sycl::plus<float>());
    if (sg_lid == 0) scratch[sg_id * COUNT + j] = s;
  }
  sycl::group_barrier(it.get_group());

  if (sg_id == 0) {
    for (int j = sg_lid; j < COUNT; j += SG_SIZE) {
      float t = 0.f;
#pragma unroll
      for (int s = 0; s < N_SG_V; ++s)
        t += scratch[s * COUNT + j];
      scratch[j] = t;
    }
  }
  sycl::group_barrier(it.get_group());

#pragma unroll
  for (int j = 0; j < COUNT; ++j)
    results[j] = scratch[j];
}

class HcHeadKernel;

static sycl::event launch_hc_head_fused(
    sycl::queue& q,
    const bf16* residual,
    const float* fn,
    const float* hc_scale,
    const float* hc_base,
    bf16* out,
    int N,
    int H,
    float rms_eps,
    float hc_eps) {
  const int hc_h = HC * H;

  sycl::range<1> global(static_cast<size_t>(N) * WG_THREADS);
  sycl::range<1> local(WG_THREADS);

  constexpr int HC_RED = HC + 1;

  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> red_scratch(HC_RED * N_SG, h);
    sycl::local_accessor<float, 1> pre_s(HC, h);

    h.parallel_for<HcHeadKernel>(
        sycl::nd_range<1>(global, local),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
          const int tok = it.get_group(0);
          const int tid = it.get_local_id(0);

          float local_mix[HC];
#pragma unroll
          for (int j = 0; j < HC; ++j)
            local_mix[j] = 0.f;
          float local_sq = 0.f;

          const bf16* x_row = residual + tok * hc_h;
          for (int k = tid * VEC; k < hc_h; k += WG_THREADS * VEC) {
            auto xv = vload_bf16<VEC>(x_row + k);
            float xf[VEC];
#pragma unroll
            for (int v = 0; v < VEC; ++v) {
              xf[v] = float(xv[v]);
              local_sq += xf[v] * xf[v];
            }
#pragma unroll
            for (int j = 0; j < HC; ++j) {
              const float* fr = fn + j * hc_h + k;
              float acc = 0.f;
#pragma unroll
              for (int v = 0; v < VEC; ++v)
                acc += xf[v] * fr[v];
              local_mix[j] += acc;
            }
          }

          float partials[HC_RED];
          partials[0] = local_sq;
#pragma unroll
          for (int j = 0; j < HC; ++j)
            partials[j + 1] = local_mix[j];

          float reduced[HC_RED];
          wg_reduce_sum_batch<HC_RED, N_SG, SG>(
              it, partials, &red_scratch[0], reduced);

          float rrms = sycl::native::rsqrt(reduced[0] / hc_h + rms_eps);
          if (tid == 0) {
#pragma unroll
            for (int j = 0; j < HC; ++j) {
              pre_s[j] = fast_sigmoid(
                             reduced[j + 1] * rrms * hc_scale[0] + hc_base[j]) +
                         hc_eps;
            }
          }
          sycl::group_barrier(it.get_group());

          float pre_reg[HC];
#pragma unroll
          for (int m = 0; m < HC; ++m)
            pre_reg[m] = pre_s[m];

          for (int k = tid * VEC; k < H; k += WG_THREADS * VEC) {
            float acc[VEC];
#pragma unroll
            for (int v = 0; v < VEC; ++v)
              acc[v] = 0.f;

#pragma unroll
            for (int m = 0; m < HC; ++m) {
              auto rv = vload_bf16<VEC>(residual + (tok * HC + m) * H + k);
#pragma unroll
              for (int v = 0; v < VEC; ++v)
                acc[v] += pre_reg[m] * float(rv[v]);
            }
            sycl::vec<bf16, VEC> ov;
#pragma unroll
            for (int v = 0; v < VEC; ++v)
              ov[v] = bf16(acc[v]);
            vstore_bf16<VEC>(out + tok * H + k, ov);
          }
        });
  });
}

}  // namespace vllm

using namespace vllm;

void hc_head_fused(
    const at::Tensor& hs_flat,
    const at::Tensor& fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    at::Tensor& out,
    int64_t hidden_size,
    double rms_eps,
    double hc_eps,
    int64_t hc_mult) {
  TORCH_CHECK(hs_flat.is_xpu(), "hc_head_fused: tensors must be on XPU");
  TORCH_CHECK(
      hs_flat.scalar_type() == at::kBFloat16, "hs_flat must be bfloat16");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be bfloat16");

  const int64_t N = hs_flat.size(0);
  if (N == 0) return;

  auto hs_c = hs_flat.contiguous();
  auto fn_c = fn.contiguous();

  auto& q = vllm::xpu::vllmGetQueue();
  launch_hc_head_fused(
      q,
      reinterpret_cast<const bf16*>(hs_c.data_ptr()),
      fn_c.data_ptr<float>(),
      hc_scale.data_ptr<float>(),
      hc_base.data_ptr<float>(),
      reinterpret_cast<bf16*>(out.data_ptr()),
      static_cast<int>(N),
      static_cast<int>(hidden_size),
      static_cast<float>(rms_eps),
      static_cast<float>(hc_eps));
}
