// mhc_post.cpp — MHC (Multi-Head Channel) Post-processing kernel (XPU/SYCL)
//
// For each token this kernel reconstructs HC new residual heads from:
//   - layer_out  [N, H]      (bf16)  — output of the transformer sub-layer
//   - residual   [N, HC, H]  (bf16)  — original residual heads
//   - post_mix   [N, HC]     (f32)   — per-head post-layer scale (from mhc_pre)
//   - comb_mix   [N, HC, HC] (f32)   — Sinkhorn combination matrix (from
//   mhc_pre)
//
// The computation per output head `o` and feature index `k` is:
//   out[tok, o, k] = post_mix[tok, o] * layer_out[tok, k]
//                  + sum_i( comb_mix[tok, i, o] * residual[tok, i, k] )

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

// ---------------------------------------------------------------------------
// Kernel launch
// ---------------------------------------------------------------------------

class MhcPostKernel {
 public:
  static constexpr int kVecWidth = 8;
  static constexpr int kWGSize = 256;

  MhcPostKernel(
      const float* comb_mix_,
      const bf16* residual_,
      const float* post_mix_,
      const bf16* layer_out_,
      bf16* out_,
      int H_,
      sycl::local_accessor<float, 1> shared_coeffs_)
      : comb_mix(comb_mix_),
        residual(residual_),
        post_mix(post_mix_),
        layer_out(layer_out_),
        out(out_),
        H(H_),
        shared_coeffs(shared_coeffs_) {}

  [[sycl::reqd_sub_group_size(16)]]
  void operator()(sycl::nd_item<1> it) const {
    const int tok = it.get_group(0);
    const int lid = it.get_local_id(0);

    // ----------------------------------------------------------------
    // Phase 1: Thread 0 loads per-token comb_mix and post_mix into
    //          shared memory so all threads can reuse them.
    //          HC*HC + HC = 20 floats total.
    // ----------------------------------------------------------------
    if (lid == 0) {
// comb_mix row: HC*HC floats loaded as HC × float4
#pragma unroll
      for (int i = 0; i < HC; ++i) {
        auto row = vload_f32<HC>(comb_mix + tok * HC * HC + i * HC);
#pragma unroll
        for (int j = 0; j < HC; ++j) {
          shared_coeffs[i * HC + j] = row[j];
        }
      }
      // post_mix row: HC floats loaded as float4
      auto pm = vload_f32<HC>(post_mix + tok * HC);
#pragma unroll
      for (int i = 0; i < HC; ++i) {
        shared_coeffs[HC * HC + i] = pm[i];
      }
    }
    sycl::group_barrier(it.get_group());

    float comb_reg[HC][HC];
    float post_reg[HC];
#pragma unroll
    for (int i = 0; i < HC; ++i) {
      post_reg[i] = shared_coeffs[HC * HC + i];
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        comb_reg[i][j] = shared_coeffs[i * HC + j];
      }
    }

    // ----------------------------------------------------------------
    // Phase 2: Vectorised loop over feature dimension.
    // ----------------------------------------------------------------
    for (int k = lid * kVecWidth; k < H; k += kWGSize * kVecWidth) {
      if (k + kVecWidth <= H) {
        auto layer_vec = vload_bf16<kVecWidth>(layer_out + tok * H + k);

        float acc[HC][kVecWidth];
#pragma unroll
        for (int o = 0; o < HC; ++o) {
#pragma unroll
          for (int v = 0; v < kVecWidth; ++v) {
            acc[o][v] = post_reg[o] * float(layer_vec[v]);
          }
        }

#pragma unroll
        for (int i = 0; i < HC; ++i) {
          auto res_vec =
              vload_bf16<kVecWidth>(residual + (tok * HC + i) * H + k);
#pragma unroll
          for (int o = 0; o < HC; ++o) {
            float c_val = comb_reg[i][o];
#pragma unroll
            for (int v = 0; v < kVecWidth; ++v) {
              acc[o][v] += c_val * float(res_vec[v]);
            }
          }
        }

#pragma unroll
        for (int o = 0; o < HC; ++o) {
          sycl::vec<bf16, kVecWidth> out_vec;
#pragma unroll
          for (int v = 0; v < kVecWidth; ++v) {
            out_vec[v] = bf16(acc[o][v]);
          }
          vstore_bf16<kVecWidth>(out + (tok * HC + o) * H + k, out_vec);
        }
      } else {
        for (int kk = k; kk < H; ++kk) {
          float lo_val = float(layer_out[tok * H + kk]);
          float acc[HC];
#pragma unroll
          for (int o = 0; o < HC; ++o) {
            acc[o] = post_reg[o] * lo_val;
          }
#pragma unroll
          for (int i = 0; i < HC; ++i) {
            float res_val = float(residual[(tok * HC + i) * H + kk]);
#pragma unroll
            for (int o = 0; o < HC; ++o) {
              acc[o] += comb_reg[i][o] * res_val;
            }
          }
#pragma unroll
          for (int o = 0; o < HC; ++o) {
            out[(tok * HC + o) * H + kk] = bf16(acc[o]);
          }
        }
      }
    }
  }

 private:
  const float* comb_mix;
  const bf16* residual;
  const float* post_mix;
  const bf16* layer_out;
  bf16* out;
  int H;
  sycl::local_accessor<float, 1> shared_coeffs;
};

static sycl::event launch_mhc_post(
    sycl::queue& q,
    const float* __restrict comb_mix,  // [N, HC, HC] combination matrix
    const bf16* __restrict residual,   // [N, HC, H]  input residual heads
    const float* __restrict post_mix,  // [N, HC]     post-layer scale per head
    const bf16* __restrict layer_out,  // [N, H]      transformer sub-layer
                                       // output
    bf16* __restrict out,  // [N, HC, H]  new residual heads (output)
    int N,                 // number of tokens
    int H) {               // hidden dimension per head
  sycl::range<1> global(static_cast<size_t>(N) * MhcPostKernel::kWGSize);
  sycl::range<1> local(MhcPostKernel::kWGSize);

  return q.submit([&](sycl::handler& h) {
    // Shared memory layout: [comb_mix: HC*HC floats | post_mix: HC floats]
    sycl::local_accessor<float, 1> shared_coeffs(HC * HC + HC, h);

    h.parallel_for(
        sycl::nd_range<1>(global, local),
        MhcPostKernel(
            comb_mix, residual, post_mix, layer_out, out, H, shared_coeffs));
  });  // q.submit
}

}  // namespace vllm

// ---------------------------------------------------------------------------
// PyTorch/ATen entry point
// ---------------------------------------------------------------------------

using namespace vllm;

// mhc_post — MHC post-processing op exposed to PyTorch.
//
// Args:
//   layer_out     : [*, H]     bf16  — transformer sub-layer output
//   residual      : [*, HC, H] bf16  — original residual heads
//   post_mix      : [*, HC, 1] f32   — per-head post-layer scale (from mhc_pre)
//   comb_mix      : [*, HC, HC] f32  — Sinkhorn combination matrix (from
//   mhc_pre)
//
// Returns: new_residual [*, HC, H] bf16
at::Tensor mhc_post(
    const at::Tensor& layer_out,
    const at::Tensor& residual,
    const at::Tensor& post_mix,
    const at::Tensor& comb_mix) {
  TORCH_CHECK(residual.is_xpu(), "mhc_post: tensors must be on XPU");
  TORCH_CHECK(
      residual.scalar_type() == at::kBFloat16, "residual must be bfloat16");
  TORCH_CHECK(
      layer_out.scalar_type() == at::kBFloat16, "layer_out must be bfloat16");
  TORCH_CHECK(post_mix.scalar_type() == at::kFloat, "post_mix must be float32");
  TORCH_CHECK(comb_mix.scalar_type() == at::kFloat, "comb_mix must be float32");

  auto residual_c = residual.contiguous();
  auto layer_out_c = layer_out.contiguous();
  auto comb_mix_c = comb_mix.contiguous();
  auto post_mix_c = post_mix.squeeze(-1).contiguous();

  const int64_t H = residual_c.size(-1);
  const int64_t N = residual_c.size(0);

  auto out = at::empty_like(residual_c);
  if (N == 0) return out;

  auto& q = vllm::xpu::vllmGetQueue();
  launch_mhc_post(
      q,
      comb_mix_c.data_ptr<float>(),
      reinterpret_cast<const bf16*>(residual_c.data_ptr()),
      post_mix_c.data_ptr<float>(),
      reinterpret_cast<const bf16*>(layer_out_c.data_ptr()),
      reinterpret_cast<bf16*>(out.data_ptr()),
      static_cast<int>(N),
      static_cast<int>(H));

  return out;
}
