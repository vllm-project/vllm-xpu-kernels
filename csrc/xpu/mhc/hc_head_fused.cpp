// hc_head_fused.cpp — HC Head Fused kernel (XPU/SYCL)
//
// A lightweight variant of mhc_pre that handles the case where only the
// pre-mixing weight is needed (no post or Sinkhorn combination output).
//
// For each input token this kernel:
//   1. Accumulates per-thread partials: squared-norm + HC projection
//   dot-products.
//   2. Work-group reduces and RMS-normalises the dot-products.
//   3. Thread 0 applies sigmoid to produce HC pre-mixing weights.
//   4. All threads blend the HC per-head residual slices using those weights.
//
// Tensor layout:
//   residual  : [N, HC, H]  (bfloat16)
//   proj_mat  : [HC, HC*H]  (float32)
//   out       : [N, H]      (bfloat16)

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

// Number of heads / channels in MHC.
static constexpr int HC = 4;

// Work-group / vectorisation parameters.
static constexpr int SG_SIZE = 16;
static constexpr int WG_SIZE = 128;
static constexpr int VEC_WIDTH = 8;
static constexpr int N_SG = WG_SIZE / SG_SIZE;

static inline float fast_sigmoid(float x) {
  return 1.f / (1.f + sycl::native::exp(-x));
}

// ---------------------------------------------------------------------------
// Work-group reduction: sum COUNT values across all sub-groups.
// (See mhc_pre.cpp for full documentation.)
// ---------------------------------------------------------------------------
template <int COUNT, int N_SG_V, int SG_SZ = 32>
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
    float sg_sum =
        sycl::reduce_over_group(sg, partials[j], sycl::plus<float>());
    if (sg_lid == 0) scratch[sg_id * COUNT + j] = sg_sum;
  }
  sycl::group_barrier(it.get_group());

  if (sg_id == 0) {
    for (int j = sg_lid; j < COUNT; j += SG_SZ) {
      float total = 0.f;
#pragma unroll
      for (int s = 0; s < N_SG_V; ++s) {
        total += scratch[s * COUNT + j];
      }
      scratch[j] = total;
    }
  }
  sycl::group_barrier(it.get_group());

#pragma unroll
  for (int j = 0; j < COUNT; ++j) {
    results[j] = scratch[j];
  }
}

// ---------------------------------------------------------------------------
// Kernel launch
// ---------------------------------------------------------------------------

class HcHeadKernel {
 public:
  HcHeadKernel(
      const bf16* residual_,
      const float* proj_mat_,
      const float* hc_scale_,
      const float* hc_bias_,
      bf16* out_,
      int H_,
      float rms_eps_,
      float hc_eps_,
      sycl::local_accessor<float, 1> reduce_scratch_,
      sycl::local_accessor<float, 1> pre_weights_s_)
      : residual(residual_),
        proj_mat(proj_mat_),
        hc_scale(hc_scale_),
        hc_bias(hc_bias_),
        out(out_),
        H(H_),
        rms_eps(rms_eps_),
        hc_eps(hc_eps_),
        reduce_scratch(reduce_scratch_),
        pre_weights_s(pre_weights_s_) {}

  [[sycl::reqd_sub_group_size(SG_SIZE)]]
  void operator()(sycl::nd_item<1> it) const {
    const int total_head_dim = HC * H;
    const int tok = it.get_group(0);
    const int tid = it.get_local_id(0);

    // ----------------------------------------------------------------
    // Phase 1: Accumulate squared-norm and HC projection dot-products.
    // ----------------------------------------------------------------
    float proj_partial[HC];
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      proj_partial[j] = 0.f;
    }
    float sq_partial = 0.f;

    const bf16* res_row = residual + tok * total_head_dim;
    for (int k = tid * VEC_WIDTH; k < total_head_dim;
         k += WG_SIZE * VEC_WIDTH) {
      auto res_vec = vload_bf16<VEC_WIDTH>(res_row + k);
      float res_f[VEC_WIDTH];
#pragma unroll
      for (int v = 0; v < VEC_WIDTH; ++v) {
        res_f[v] = float(res_vec[v]);
        sq_partial += res_f[v] * res_f[v];
      }
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        auto proj_vec = vload_f32<VEC_WIDTH>(proj_mat + j * total_head_dim + k);
        float acc = 0.f;
#pragma unroll
        for (int v = 0; v < VEC_WIDTH; ++v) {
          acc += res_f[v] * proj_vec[v];
        }
        proj_partial[j] += acc;
      }
    }

    // ----------------------------------------------------------------
    // Phase 2: Work-group reduce and RMS-normalise.
    // ----------------------------------------------------------------
    constexpr int RED_COUNT = HC + 1;
    float partials[RED_COUNT];
    partials[0] = sq_partial;
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      partials[j + 1] = proj_partial[j];
    }

    float reduced[RED_COUNT];
    wg_reduce_sum_batch<RED_COUNT, N_SG, SG_SIZE>(
        it, partials, &reduce_scratch[0], reduced);

    float rrms = sycl::native::rsqrt(
        reduced[0] / static_cast<float>(total_head_dim) + rms_eps);
    if (tid == 0) {
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        pre_weights_s[j] =
            fast_sigmoid(reduced[j + 1] * rrms * hc_scale[0] + hc_bias[j]) +
            hc_eps;
      }
    }
    sycl::group_barrier(it.get_group());

    // ----------------------------------------------------------------
    // Phase 3: Blend HC head residuals using pre-mixing weights.
    // ----------------------------------------------------------------
    float pre_weights[HC];
#pragma unroll
    for (int m = 0; m < HC; ++m) {
      pre_weights[m] = pre_weights_s[m];
    }

    for (int k = tid * VEC_WIDTH; k < H; k += WG_SIZE * VEC_WIDTH) {
      float acc[VEC_WIDTH];
#pragma unroll
      for (int v = 0; v < VEC_WIDTH; ++v) {
        acc[v] = 0.f;
      }

#pragma unroll
      for (int m = 0; m < HC; ++m) {
        auto head_vec =
            vload_bf16<VEC_WIDTH>(residual + (tok * HC + m) * H + k);
#pragma unroll
        for (int v = 0; v < VEC_WIDTH; ++v) {
          acc[v] += pre_weights[m] * float(head_vec[v]);
        }
      }

      sycl::vec<bf16, VEC_WIDTH> out_vec;
#pragma unroll
      for (int v = 0; v < VEC_WIDTH; ++v) {
        out_vec[v] = bf16(acc[v]);
      }
      vstore_bf16<VEC_WIDTH>(out + tok * H + k, out_vec);
    }
  }

 private:
  const bf16* residual;
  const float* proj_mat;
  const float* hc_scale;
  const float* hc_bias;
  bf16* out;
  int H;
  float rms_eps;
  float hc_eps;
  sycl::local_accessor<float, 1> reduce_scratch;
  sycl::local_accessor<float, 1> pre_weights_s;
};

static sycl::event launch_hc_head_fused(
    sycl::queue& q,
    const bf16* residual,   // [N, HC, H] input residual
    const float* proj_mat,  // [HC, HC*H] projection matrix
    const float* hc_scale,  // [1] sigmoid scale
    const float* hc_bias,   // [HC] sigmoid bias per head
    bf16* out,              // [N, H] blended output
    int N,                  // number of tokens
    int H,                  // hidden dimension per head
    float rms_eps,          // RMS normalisation epsilon
    float hc_eps) {         // epsilon added to pre-mixing weight after sigmoid

  sycl::range<1> global(static_cast<size_t>(N) * WG_SIZE);
  sycl::range<1> local(WG_SIZE);

  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> reduce_scratch((HC + 1) * N_SG, h);
    // HC pre-mixing weights written by thread 0 and read by all threads.
    sycl::local_accessor<float, 1> pre_weights_s(HC, h);

    h.parallel_for(
        sycl::nd_range<1>(global, local),
        HcHeadKernel(
            residual,
            proj_mat,
            hc_scale,
            hc_bias,
            out,
            H,
            rms_eps,
            hc_eps,
            reduce_scratch,
            pre_weights_s));
  });  // q.submit
}

}  // namespace vllm

// ---------------------------------------------------------------------------
// PyTorch/ATen entry point
// ---------------------------------------------------------------------------

using namespace vllm;

// hc_head_fused — fused HC head blending op exposed to PyTorch.
//
// Args:
//   hs_flat    : [N, HC*H] bfloat16  — flattened residual input
//   proj_mat   : [HC, HC*H] float32  — projection matrix
//   hc_scale   : [1] float32          — sigmoid scale
//   hc_bias    : [HC] float32         — sigmoid bias per head
//   out        : [N, H] bfloat16      — output buffer (written in-place)
//   hidden_size: H (per-head hidden dimension)
//   rms_eps    : RMS normalisation epsilon
//   hc_eps     : epsilon added to pre-mixing weight after sigmoid
//   hc_mult    : unused (reserved)
void hc_head_fused(
    const at::Tensor& hs_flat,
    const at::Tensor& proj_mat,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_bias,
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
  auto proj_mat_c = proj_mat.contiguous();

  auto& q = vllm::xpu::vllmGetQueue();
  launch_hc_head_fused(
      q,
      reinterpret_cast<const bf16*>(hs_c.data_ptr()),
      proj_mat_c.data_ptr<float>(),
      hc_scale.data_ptr<float>(),
      hc_bias.data_ptr<float>(),
      reinterpret_cast<bf16*>(out.data_ptr()),
      static_cast<int>(N),
      static_cast<int>(hidden_size),
      static_cast<float>(rms_eps),
      static_cast<float>(hc_eps));
}
