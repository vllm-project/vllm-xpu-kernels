// mhc_pre.cpp — MHC (Multi-Head Channel) Pre-processing kernel (XPU/SYCL)
//
// For each input token this kernel performs the following steps:
//   1. Accumulate per-thread partial sums:
//        - squared-norm of the flattened residual (for RMS normalisation)
//        - HC3 feature-norm dot-products: dot(x, proj_mat[i]) for i in [0, HC3)
//   2. Work-group reduce the partials, then RMS-normalise the dot-products.
//   3. (sub-group 0 only) Compute per-head mixing weights:
//        - pre_weight[m]  = sigmoid(proj[m]  * scale[0] + bias[m])       +
//        pre_eps
//        - post_weight[m] = sigmoid(proj[HC+m] * scale[1] + bias[HC+m])  *
//        post_mult
//      Write post_weight to the `post_mix` output.
//   4. (sub-group 0 only) Compute HC×HC Sinkhorn-normalised combination weights
//      from the remaining HC² projections and write them to `comb_mix`.
//   5. Blend the HC per-head residual slices using pre_weight to produce
//   `layer_input`.
//
// Tensor layout:
//   residual    : [N, HC, H]   (bfloat16)
//   proj_mat    : [HC3, HC*H]  (float32)   — HC3 = 2*HC + HC*HC
//   post_mix    : [N, HC]      (float32)
//   comb_mix    : [N, HC, HC]  (float32)
//   layer_input : [N, H]       (bfloat16)

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

static inline float fast_sigmoid(float x) {
  return 1.f / (1.f + sycl::native::exp(-x));
}

// ---------------------------------------------------------------------------
// Work-group reduction: sum COUNT values across all sub-groups.
//
// Each thread contributes `partials[0..COUNT)`.  After the call every thread
// reads the work-group-wide sums from `results[0..COUNT)`.
// `scratch` must have at least N_SG * COUNT floats of shared storage.
// ---------------------------------------------------------------------------
template <int COUNT, int N_SG, int SG_SIZE = 32>
static inline void wg_reduce_sum_batch(
    sycl::nd_item<1>& it,
    const float (&partials)[COUNT],  // per-thread inputs
    float* scratch,                  // shared memory scratch [N_SG * COUNT]
    float (&results)[COUNT]) {       // work-group-wide sums (output)
  auto sg = it.get_sub_group();
  const int sg_id = sg.get_group_id()[0];
  const int sg_lid = sg.get_local_id()[0];

// Step 1: reduce within each sub-group and write one partial per sub-group.
#pragma unroll
  for (int j = 0; j < COUNT; ++j) {
    float sg_sum =
        sycl::reduce_over_group(sg, partials[j], sycl::plus<float>());
    if (sg_lid == 0) scratch[sg_id * COUNT + j] = sg_sum;
  }
  sycl::group_barrier(it.get_group());

  // Step 2: sub-group 0 accumulates the per-sub-group partial sums.
  if (sg_id == 0) {
    for (int j = sg_lid; j < COUNT; j += SG_SIZE) {
      float total = 0.f;
#pragma unroll
      for (int s = 0; s < N_SG; ++s) {
        total += scratch[s * COUNT + j];
      }
      scratch[j] = total;  // overwrite with final sum
    }
  }
  sycl::group_barrier(it.get_group());

// Step 3: broadcast results to every thread.
#pragma unroll
  for (int j = 0; j < COUNT; ++j) {
    results[j] = scratch[j];
  }
}

// ---------------------------------------------------------------------------
// Kernel launch
// ---------------------------------------------------------------------------

class MhcPreKernel {
 public:
  static constexpr int kTokensPerWG = 2;
  static constexpr int kSGSize = 16;
  static constexpr int kWGSize = 256;
  static constexpr int kVecWidth = 8;
  static constexpr int kNumSubGroups = kWGSize / kSGSize;
  static constexpr int kHC3 = HC * (2 + HC);
  static constexpr int kRedCount = kHC3 + 1;
  static constexpr int kProjBatch = 4;

  MhcPreKernel(
      const bf16* residual_,
      const float* proj_mat_,
      const float* hc_scale_,
      const float* hc_bias_,
      float* post_mix_,
      float* comb_mix_,
      bf16* layer_input_,
      int N_,
      int H_,
      float rms_eps_,
      float pre_eps_,
      float sinkhorn_eps_,
      float post_mult_,
      int sinkhorn_repeat_,
      sycl::local_accessor<float, 1> shared_mixes_,
      sycl::local_accessor<float, 1> reduce_scratch_)
      : residual(residual_),
        proj_mat(proj_mat_),
        hc_scale(hc_scale_),
        hc_bias(hc_bias_),
        post_mix(post_mix_),
        comb_mix(comb_mix_),
        layer_input(layer_input_),
        N(N_),
        H(H_),
        rms_eps(rms_eps_),
        pre_eps(pre_eps_),
        sinkhorn_eps(sinkhorn_eps_),
        post_mult(post_mult_),
        sinkhorn_repeat(sinkhorn_repeat_),
        shared_mixes(shared_mixes_),
        reduce_scratch(reduce_scratch_) {}

  [[sycl::reqd_sub_group_size(kSGSize)]]
  void operator()(sycl::nd_item<1> it) const {
    const int total_head_dim = HC * H;
    const int wg_id = it.get_group(0);
    const int tid = it.get_local_id(0);
    auto sg = it.get_sub_group();
    const int tok_base = wg_id * kTokensPerWG;

    // ----------------------------------------------------------------
    // Phase 1: Accumulate squared-norm and HC3 projection dot-products
    //          over the flattened residual [tok, HC, H].
    // ----------------------------------------------------------------

    // proj_partial[t][j] = dot(residual[tok_base+t], proj_mat[j])
    float proj_partial[kTokensPerWG][kHC3];
    // sq_partial[t] = sum_i residual[tok_base+t][i]^2
    float sq_partial[kTokensPerWG];

#pragma unroll
    for (int t = 0; t < kTokensPerWG; ++t) {
      sq_partial[t] = 0.f;
#pragma unroll
      for (int j = 0; j < kHC3; ++j) {
        proj_partial[t][j] = 0.f;
      }
    }

    // Process projection rows in batches of kProjBatch to amortise
    // register pressure.  Split into FULL_BATCHES + optional REMAINDER.
    constexpr int FULL_BATCHES = kHC3 / kProjBatch;
    constexpr int REMAINDER = kHC3 % kProjBatch;

    // Batch 0: also accumulates the squared-norm (avoids an extra pass).
    for (int k = tid * kVecWidth; k < total_head_dim;
         k += kWGSize * kVecWidth) {
      // Load kProjBatch projection vectors (rows 0..kProjBatch-1).
      float proj_chunk[kProjBatch][kVecWidth];
#pragma unroll
      for (int j = 0; j < kProjBatch; ++j) {
        auto pv = vload_f32<kVecWidth>(proj_mat + j * total_head_dim + k);
#pragma unroll
        for (int v = 0; v < kVecWidth; ++v) {
          proj_chunk[j][v] = pv[v];
        }
      }

#pragma unroll
      for (int t = 0; t < kTokensPerWG; ++t) {
        int tok = tok_base + t;
        if (tok >= N) break;

        // Load residual elements for this token.
        auto res_vec =
            vload_bf16<kVecWidth>(residual + tok * total_head_dim + k);
        float res_f[kVecWidth];
#pragma unroll
        for (int v = 0; v < kVecWidth; ++v) {
          res_f[v] = float(res_vec[v]);
        }

// Accumulate squared-norm.
#pragma unroll
        for (int v = 0; v < kVecWidth; ++v) {
          sq_partial[t] += res_f[v] * res_f[v];
        }
// Accumulate dot-products for rows in this batch.
#pragma unroll
        for (int j = 0; j < kProjBatch; ++j) {
          float acc = 0.f;
#pragma unroll
          for (int v = 0; v < kVecWidth; ++v) {
            acc += res_f[v] * proj_chunk[j][v];
          }
          proj_partial[t][j] += acc;
        }
      }
    }

// Remaining full batches (batch 1..FULL_BATCHES-1).
#pragma unroll
    for (int b = 1; b < FULL_BATCHES; ++b) {
      for (int k = tid * kVecWidth; k < total_head_dim;
           k += kWGSize * kVecWidth) {
        float proj_chunk[kProjBatch][kVecWidth];
#pragma unroll
        for (int j = 0; j < kProjBatch; ++j) {
          auto pv = vload_f32<kVecWidth>(
              proj_mat + (b * kProjBatch + j) * total_head_dim + k);
#pragma unroll
          for (int v = 0; v < kVecWidth; ++v) {
            proj_chunk[j][v] = pv[v];
          }
        }

#pragma unroll
        for (int t = 0; t < kTokensPerWG; ++t) {
          int tok = tok_base + t;
          if (tok >= N) break;

          auto res_vec =
              vload_bf16<kVecWidth>(residual + tok * total_head_dim + k);
          float res_f[kVecWidth];
#pragma unroll
          for (int v = 0; v < kVecWidth; ++v) {
            res_f[v] = float(res_vec[v]);
          }

#pragma unroll
          for (int j = 0; j < kProjBatch; ++j) {
            float acc = 0.f;
#pragma unroll
            for (int v = 0; v < kVecWidth; ++v) {
              acc += res_f[v] * proj_chunk[j][v];
            }
            proj_partial[t][b * kProjBatch + j] += acc;
          }
        }
      }
    }

    // Tail batch (rows FULL_BATCHES*kProjBatch .. kHC3-1).
    if constexpr (REMAINDER > 0) {
      constexpr int REM_START = FULL_BATCHES * kProjBatch;
      for (int k = tid * kVecWidth; k < total_head_dim;
           k += kWGSize * kVecWidth) {
        float proj_chunk[REMAINDER][kVecWidth];
#pragma unroll
        for (int j = 0; j < REMAINDER; ++j) {
          auto pv = vload_f32<kVecWidth>(
              proj_mat + (REM_START + j) * total_head_dim + k);
#pragma unroll
          for (int v = 0; v < kVecWidth; ++v) {
            proj_chunk[j][v] = pv[v];
          }
        }

#pragma unroll
        for (int t = 0; t < kTokensPerWG; ++t) {
          int tok = tok_base + t;
          if (tok >= N) break;

          auto res_vec =
              vload_bf16<kVecWidth>(residual + tok * total_head_dim + k);
          float res_f[kVecWidth];
#pragma unroll
          for (int v = 0; v < kVecWidth; ++v) {
            res_f[v] = float(res_vec[v]);
          }

#pragma unroll
          for (int j = 0; j < REMAINDER; ++j) {
            float acc = 0.f;
#pragma unroll
            for (int v = 0; v < kVecWidth; ++v) {
              acc += res_f[v] * proj_chunk[j][v];
            }
            proj_partial[t][REM_START + j] += acc;
          }
        }
      }
    }

    // ----------------------------------------------------------------
    // Phase 2: Work-group reduce and RMS-normalise.
    // ----------------------------------------------------------------

#pragma unroll
    for (int t = 0; t < kTokensPerWG; ++t) {
      int tok = tok_base + t;
      if (tok >= N) break;

      // Pack sq_partial and proj_partial into a single reduction array.
      float partials[kRedCount];
      partials[0] = sq_partial[t];
#pragma unroll
      for (int j = 0; j < kHC3; ++j) {
        partials[j + 1] = proj_partial[t][j];
      }

      float reduced[kRedCount];
      wg_reduce_sum_batch<kRedCount, kNumSubGroups, kSGSize>(
          it,
          partials,
          &reduce_scratch[t * kRedCount * kNumSubGroups],
          reduced);

      // RMS scale factor: rsqrt( mean(x^2) + eps )
      float rrms = sycl::native::rsqrt(
          reduced[0] / static_cast<float>(total_head_dim) + rms_eps);

      // Write RMS-normalised projections to shared memory.
      if (tid < kHC3) {
        shared_mixes[t * kHC3 + tid] = reduced[tid + 1] * rrms;
      }
    }

    sycl::group_barrier(it.get_group());

    // ----------------------------------------------------------------
    // Phases 3 & 4: Compute mixing weights.
    // Only sub-group 0 participates; each lane handles one element.
    // ----------------------------------------------------------------

    if (sg.get_group_id()[0] == 0) {
      const int lane = sg.get_local_id()[0];

      // ---- Phase 3: Pre / post sigmoid mixing weights ----
      // Lanes 0..kTokensPerWG*HC-1 each handle one (token, head) pair.
      int tok_in_sg = lane / HC;
      int head_idx = lane % HC;

      if (tok_in_sg < kTokensPerWG) {
        int tok = tok_base + tok_in_sg;
        if (tok < N) {
          float pre_weight = fast_sigmoid(
                                 sycl::fma(
                                     shared_mixes[tok_in_sg * kHC3 + head_idx],
                                     hc_scale[0],
                                     hc_bias[head_idx])) +
                             pre_eps;
          shared_mixes[tok_in_sg * kHC3 + head_idx] = pre_weight;

          float post_weight =
              fast_sigmoid(
                  sycl::fma(
                      shared_mixes[tok_in_sg * kHC3 + head_idx + HC],
                      hc_scale[1],
                      hc_bias[head_idx + HC])) *
              post_mult;
          post_mix[tok * HC + head_idx] = post_weight;
        }
      }
      sycl::group_barrier(sg);

      // ---- Phase 4: Sinkhorn-normalised combination weights ----
      constexpr int COMB_LANES = HC * HC;
      constexpr int TOKS_PER_ROUND = kSGSize / COMB_LANES;
      constexpr int NUM_COMB_ROUNDS =
          (kTokensPerWG + TOKS_PER_ROUND - 1) / TOKS_PER_ROUND;

#pragma unroll
      for (int round = 0; round < NUM_COMB_ROUNDS; ++round) {
        int tok_in_round = lane / COMB_LANES;
        int comb_idx = lane % COMB_LANES;
        int t = round * TOKS_PER_ROUND + tok_in_round;

        if (t < kTokensPerWG && tok_in_round < TOKS_PER_ROUND) {
          int tok = tok_base + t;
          if (tok < N) {
            float weight = sycl::fma(
                shared_mixes[t * kHC3 + comb_idx + 2 * HC],
                hc_scale[2],
                hc_bias[comb_idx + 2 * HC]);

            float row_max = weight;
#pragma unroll
            for (int off = 1; off < HC; off <<= 1) {
              row_max = sycl::max(
                  row_max, sycl::permute_group_by_xor(sg, row_max, off));
            }
            weight = sycl::native::exp(weight - row_max);

            float row_sum = weight;
#pragma unroll
            for (int off = 1; off < HC; off <<= 1) {
              row_sum += sycl::permute_group_by_xor(sg, row_sum, off);
            }
            weight *= sycl::native::recip(row_sum);
            weight += sinkhorn_eps;

            for (int iter = 0; iter < sinkhorn_repeat; ++iter) {
              float col_sum = weight;
#pragma unroll
              for (int off = HC; off < COMB_LANES; off <<= 1) {
                col_sum += sycl::permute_group_by_xor(sg, col_sum, off);
              }
              weight *= sycl::native::recip(col_sum + sinkhorn_eps);

              if (iter < sinkhorn_repeat - 1) {
                float row_sum2 = weight;
#pragma unroll
                for (int off = 1; off < HC; off <<= 1) {
                  row_sum2 += sycl::permute_group_by_xor(sg, row_sum2, off);
                }
                weight *= sycl::native::recip(row_sum2 + sinkhorn_eps);
              }
            }
            comb_mix[tok * HC * HC + comb_idx] = weight;
          }
        }

        if (round < NUM_COMB_ROUNDS - 1) sycl::group_barrier(sg);
      }
    }

    sycl::group_barrier(it.get_group());

    // ----------------------------------------------------------------
    // Phase 5: Blend HC head residuals using pre-mixing weights.
    //   layer_input[tok][k] = sum_m pre_weight[m] * residual[tok, m, k]
    // ----------------------------------------------------------------

#pragma unroll
    for (int t = 0; t < kTokensPerWG; ++t) {
      int tok = tok_base + t;
      if (tok >= N) break;

      float pre_weights[HC];
#pragma unroll
      for (int m = 0; m < HC; ++m) {
        pre_weights[m] = shared_mixes[t * kHC3 + m];
      }

      for (int k = tid * kVecWidth; k < H; k += kWGSize * kVecWidth) {
        float acc[kVecWidth];
#pragma unroll
        for (int v = 0; v < kVecWidth; ++v) {
          acc[v] = 0.f;
        }

#pragma unroll
        for (int m = 0; m < HC; ++m) {
          auto head_vec =
              vload_bf16<kVecWidth>(residual + (tok * HC + m) * H + k);
#pragma unroll
          for (int v = 0; v < kVecWidth; ++v) {
            acc[v] += pre_weights[m] * float(head_vec[v]);
          }
        }

        sycl::vec<bf16, kVecWidth> out_vec;
#pragma unroll
        for (int v = 0; v < kVecWidth; ++v) {
          out_vec[v] = bf16(acc[v]);
        }
        vstore_bf16<kVecWidth>(layer_input + tok * H + k, out_vec);
      }
    }
  }

 private:
  const bf16* residual;
  const float* proj_mat;
  const float* hc_scale;
  const float* hc_bias;
  float* post_mix;
  float* comb_mix;
  bf16* layer_input;
  int N;
  int H;
  float rms_eps;
  float pre_eps;
  float sinkhorn_eps;
  float post_mult;
  int sinkhorn_repeat;
  sycl::local_accessor<float, 1> shared_mixes;
  sycl::local_accessor<float, 1> reduce_scratch;
};

static sycl::event launch_mhc_pre(
    sycl::queue& q,
    const bf16* residual,   // [N, HC, H] input residual
    const float* proj_mat,  // [HC3, HC*H] projection matrix
    const float* hc_scale,  // [3] per-group sigmoid scale
    const float* hc_bias,   // [HC3] per-element sigmoid bias
    float* post_mix,        // [N, HC] post-sigmoid mixing weights (output)
    float* comb_mix,        // [N, HC*HC] Sinkhorn combination weights (output)
    bf16* layer_input,      // [N, H] blended layer input (output)
    int N,                  // number of tokens
    int H,                  // hidden dimension per head
    float rms_eps,          // epsilon for RMS normalisation
    float pre_eps,          // epsilon added to pre-mixing weights
    float sinkhorn_eps,     // epsilon for Sinkhorn normalisation
    float post_mult,        // scalar multiplier applied to post-mixing weights
    int sinkhorn_repeat) {  // number of Sinkhorn normalisation iterations

  sycl::range<1> global(
      static_cast<size_t>(
          (N + MhcPreKernel::kTokensPerWG - 1) / MhcPreKernel::kTokensPerWG) *
      MhcPreKernel::kWGSize);
  sycl::range<1> local(MhcPreKernel::kWGSize);

  return q.submit([&](sycl::handler& h) {
    // Shared memory: HC3 mix values per token (written after reduction)
    sycl::local_accessor<float, 1> shared_mixes(
        MhcPreKernel::kHC3 * MhcPreKernel::kTokensPerWG, h);
    // Shared scratch for the work-group reduction
    sycl::local_accessor<float, 1> reduce_scratch(
        MhcPreKernel::kRedCount * MhcPreKernel::kNumSubGroups *
            MhcPreKernel::kTokensPerWG,
        h);

    h.parallel_for(
        sycl::nd_range<1>(global, local),
        MhcPreKernel(
            residual,
            proj_mat,
            hc_scale,
            hc_bias,
            post_mix,
            comb_mix,
            layer_input,
            N,
            H,
            rms_eps,
            pre_eps,
            sinkhorn_eps,
            post_mult,
            sinkhorn_repeat,
            shared_mixes,
            reduce_scratch));
  });  // q.submit
}

}  // namespace vllm

// ---------------------------------------------------------------------------
// PyTorch/ATen entry point
// ---------------------------------------------------------------------------

using namespace vllm;

// mhc_pre — MHC pre-processing op exposed to PyTorch.
//
// Args:
//   residual         : [*, HC, H] bfloat16
//   feat_proj        : [HC3, HC*H] float32   (feature-norm projection matrix)
//   hc_scale         : [3] float32            (per-group sigmoid scale)
//   hc_bias          : [HC3] float32          (per-element sigmoid bias)
//   rms_eps          : RMS normalisation epsilon
//   hc_pre_eps       : epsilon added to pre-mixing weights after sigmoid
//   hc_sinkhorn_eps  : Sinkhorn normalisation epsilon
//   hc_post_mult     : scalar multiplier for post-mixing weights
//   sinkhorn_repeat  : number of Sinkhorn iterations
//
// Returns: (post_mix, comb_mix, layer_input)
std::tuple<at::Tensor, at::Tensor, at::Tensor> mhc_pre(
    const at::Tensor& residual,
    const at::Tensor& feat_proj,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_bias,
    double rms_eps,
    double hc_pre_eps,
    double hc_sinkhorn_eps,
    double hc_post_mult,
    int64_t sinkhorn_repeat) {
  TORCH_CHECK(
      residual.is_xpu() && feat_proj.is_xpu(),
      "mhc_pre: tensors must be on XPU");
  TORCH_CHECK(
      residual.scalar_type() == at::kBFloat16, "residual must be bfloat16");
  TORCH_CHECK(
      feat_proj.scalar_type() == at::kFloat, "feat_proj must be float32");
  TORCH_CHECK(hc_scale.scalar_type() == at::kFloat, "hc_scale must be float32");
  TORCH_CHECK(hc_bias.scalar_type() == at::kFloat, "hc_bias must be float32");

  auto residual_c = residual.contiguous();
  auto feat_proj_c = feat_proj.contiguous();

  const int64_t H = residual_c.size(-1);
  const int64_t HC2 = HC * HC;       // combination matrix size
  const int64_t HC3 = HC * 2 + HC2;  // total projection count

  TORCH_CHECK(
      feat_proj_c.size(0) == HC3 && feat_proj_c.size(1) == HC * H,
      "feat_proj shape mismatch: expected [HC3, HC*H]");
  TORCH_CHECK(hc_scale.numel() == 3, "hc_scale must have 3 elements");
  TORCH_CHECK(hc_bias.numel() == HC3, "hc_bias must have HC3 elements");

  // Flatten all batch dimensions: [*, HC, H] -> [N, HC, H]
  auto outer_shape = residual_c.sizes().slice(0, residual_c.dim() - 2).vec();
  auto residual_flat = residual_c.view({-1, HC, H});
  const int64_t N = residual_flat.size(0);

  auto opts_f32 = residual_c.options().dtype(at::kFloat);
  auto opts_bf16 = residual_c.options().dtype(at::kBFloat16);

  auto post_mix = at::empty({N, HC}, opts_f32);
  auto comb_mix = at::empty({N, HC2}, opts_f32);
  auto layer_input = at::empty({N, H}, opts_bf16);

  // Helper: build an output shape by appending trailing dims to the batch dims.
  auto make_shape = [&](std::initializer_list<int64_t> trailing) {
    auto s = outer_shape;
    for (auto d : trailing)
      s.push_back(d);
    return s;
  };

  if (N == 0) {
    return {
        post_mix.view(make_shape({HC, 1})),
        comb_mix.view(make_shape({HC, HC})),
        layer_input.view(make_shape({H}))};
  }

  auto& q = vllm::xpu::vllmGetQueue();
  launch_mhc_pre(
      q,
      reinterpret_cast<const bf16*>(residual_flat.data_ptr()),
      feat_proj_c.data_ptr<float>(),
      hc_scale.data_ptr<float>(),
      hc_bias.data_ptr<float>(),
      post_mix.data_ptr<float>(),
      comb_mix.data_ptr<float>(),
      reinterpret_cast<bf16*>(layer_input.data_ptr()),
      static_cast<int>(N),
      static_cast<int>(H),
      static_cast<float>(rms_eps),
      static_cast<float>(hc_pre_eps),
      static_cast<float>(hc_sinkhorn_eps),
      static_cast<float>(hc_post_mult),
      static_cast<int>(sinkhorn_repeat));

  return {
      post_mix.view(make_shape({HC, 1})),
      comb_mix.view(make_shape({HC, HC})),
      layer_input.view(make_shape({H}))};
}
