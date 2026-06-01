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

template <int N>
static inline sycl::vec<bf16, N> vload_bf16(const bf16* p) {
  return *reinterpret_cast<const sycl::vec<bf16, N>*>(p);
}
template <int N>
static inline void vstore_bf16(bf16* p, const sycl::vec<bf16, N>& v) {
  *reinterpret_cast<sycl::vec<bf16, N>*>(p) = v;
}
template <int N>
static inline sycl::vec<float, N> vload_f32(const float* p) {
  return *reinterpret_cast<const sycl::vec<float, N>*>(p);
}

static inline float fast_sigmoid(float x) {
  return 1.f / (1.f + sycl::native::exp(-x));
}

template <int COUNT, int N_SG, int SG_SIZE = 32>
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
      for (int s = 0; s < N_SG; ++s)
        t += scratch[s * COUNT + j];
      scratch[j] = t;
    }
  }
  sycl::group_barrier(it.get_group());

#pragma unroll
  for (int j = 0; j < COUNT; ++j)
    results[j] = scratch[j];
}

class MhcPreKernel;

static sycl::event launch_mhc_pre(
    sycl::queue& q,
    const bf16* residual,
    const float* fn,
    const float* hc_scale,
    const float* hc_base,
    float* post_mix,
    float* comb_mix,
    bf16* layer_input,
    int N,
    int H,
    float rms_eps,
    float pre_eps,
    float sk_eps,
    float post_mult,
    int sk_repeat) {
  static constexpr int TOKENS_PER_WG = 2;
  static constexpr int SG_PRE = 16;
  static constexpr int WG_THREADS_PRE = 256;
  static constexpr int VEC_PRE = 8;
  static constexpr int N_SG_PRE = WG_THREADS_PRE / SG_PRE;
  constexpr int HC3 = HC * (2 + HC);
  constexpr int RED_COUNT = HC3 + 1;
  constexpr int FN_BATCH = 4;
  const int hc_h = HC * H;

  sycl::range<1> global(
      static_cast<size_t>((N + TOKENS_PER_WG - 1) / TOKENS_PER_WG) *
      WG_THREADS_PRE);
  sycl::range<1> local(WG_THREADS_PRE);

  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> mixes_s(HC3 * TOKENS_PER_WG, h);
    sycl::local_accessor<float, 1> red_scratch(
        RED_COUNT * N_SG_PRE * TOKENS_PER_WG, h);

    h.parallel_for<MhcPreKernel>(
        sycl::nd_range<1>(global, local),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_PRE)]]
        {
          const int wg_id = it.get_group(0);
          const int tid = it.get_local_id(0);
          auto sg = it.get_sub_group();
          const int tok_base = wg_id * TOKENS_PER_WG;

          float local_mix[TOKENS_PER_WG][HC3];
          float local_sq[TOKENS_PER_WG];
#pragma unroll
          for (int t = 0; t < TOKENS_PER_WG; ++t) {
            local_sq[t] = 0.f;
#pragma unroll
            for (int j = 0; j < HC3; ++j)
              local_mix[t][j] = 0.f;
          }

          constexpr int FULL_BATCHES = HC3 / FN_BATCH;
          constexpr int REMAINDER = HC3 % FN_BATCH;

          for (int k = tid * VEC_PRE; k < hc_h; k += WG_THREADS_PRE * VEC_PRE) {
            float fnf[FN_BATCH][VEC_PRE];
#pragma unroll
            for (int j = 0; j < FN_BATCH; ++j) {
              auto fnv = vload_f32<VEC_PRE>(fn + j * hc_h + k);
#pragma unroll
              for (int v = 0; v < VEC_PRE; ++v)
                fnf[j][v] = fnv[v];
            }

#pragma unroll
            for (int t = 0; t < TOKENS_PER_WG; ++t) {
              int tok = tok_base + t;
              if (tok >= N) break;

              auto xv = vload_bf16<VEC_PRE>(residual + tok * hc_h + k);
              float xf[VEC_PRE];
#pragma unroll
              for (int v = 0; v < VEC_PRE; ++v)
                xf[v] = float(xv[v]);

#pragma unroll
              for (int v = 0; v < VEC_PRE; ++v)
                local_sq[t] += xf[v] * xf[v];

#pragma unroll
              for (int j = 0; j < FN_BATCH; ++j) {
                float acc = 0.f;
#pragma unroll
                for (int v = 0; v < VEC_PRE; ++v)
                  acc += xf[v] * fnf[j][v];
                local_mix[t][j] += acc;
              }
            }
          }

#pragma unroll
          for (int b = 1; b < FULL_BATCHES; ++b) {
            for (int k = tid * VEC_PRE; k < hc_h;
                 k += WG_THREADS_PRE * VEC_PRE) {
              float fnf[FN_BATCH][VEC_PRE];
#pragma unroll
              for (int j = 0; j < FN_BATCH; ++j) {
                auto fnv =
                    vload_f32<VEC_PRE>(fn + (b * FN_BATCH + j) * hc_h + k);
#pragma unroll
                for (int v = 0; v < VEC_PRE; ++v)
                  fnf[j][v] = fnv[v];
              }

#pragma unroll
              for (int t = 0; t < TOKENS_PER_WG; ++t) {
                int tok = tok_base + t;
                if (tok >= N) break;

                auto xv = vload_bf16<VEC_PRE>(residual + tok * hc_h + k);
                float xf[VEC_PRE];
#pragma unroll
                for (int v = 0; v < VEC_PRE; ++v)
                  xf[v] = float(xv[v]);

#pragma unroll
                for (int j = 0; j < FN_BATCH; ++j) {
                  float acc = 0.f;
#pragma unroll
                  for (int v = 0; v < VEC_PRE; ++v)
                    acc += xf[v] * fnf[j][v];
                  local_mix[t][b * FN_BATCH + j] += acc;
                }
              }
            }
          }

          if constexpr (REMAINDER > 0) {
            constexpr int REM_START = FULL_BATCHES * FN_BATCH;
            for (int k = tid * VEC_PRE; k < hc_h;
                 k += WG_THREADS_PRE * VEC_PRE) {
              float fnf[REMAINDER][VEC_PRE];
#pragma unroll
              for (int j = 0; j < REMAINDER; ++j) {
                auto fnv = vload_f32<VEC_PRE>(fn + (REM_START + j) * hc_h + k);
#pragma unroll
                for (int v = 0; v < VEC_PRE; ++v)
                  fnf[j][v] = fnv[v];
              }

#pragma unroll
              for (int t = 0; t < TOKENS_PER_WG; ++t) {
                int tok = tok_base + t;
                if (tok >= N) break;

                auto xv = vload_bf16<VEC_PRE>(residual + tok * hc_h + k);
                float xf[VEC_PRE];
#pragma unroll
                for (int v = 0; v < VEC_PRE; ++v)
                  xf[v] = float(xv[v]);

#pragma unroll
                for (int j = 0; j < REMAINDER; ++j) {
                  float acc = 0.f;
#pragma unroll
                  for (int v = 0; v < VEC_PRE; ++v)
                    acc += xf[v] * fnf[j][v];
                  local_mix[t][REM_START + j] += acc;
                }
              }
            }
          }

#pragma unroll
          for (int t = 0; t < TOKENS_PER_WG; ++t) {
            int tok = tok_base + t;
            if (tok >= N) break;

            float partials[RED_COUNT];
            partials[0] = local_sq[t];
#pragma unroll
            for (int j = 0; j < HC3; ++j)
              partials[j + 1] = local_mix[t][j];

            float reduced[RED_COUNT];
            wg_reduce_sum_batch<RED_COUNT, N_SG_PRE, SG_PRE>(
                it, partials, &red_scratch[t * RED_COUNT * N_SG_PRE], reduced);

            float rrms = sycl::native::rsqrt(reduced[0] / hc_h + rms_eps);

            if (tid < HC3) mixes_s[t * HC3 + tid] = reduced[tid + 1] * rrms;
          }

          sycl::group_barrier(it.get_group());

          if (sg.get_group_id()[0] == 0) {
            const int lane = sg.get_local_id()[0];

            int token_pre = lane / HC;
            int m_pre = lane % HC;

            if (token_pre < TOKENS_PER_WG) {
              int tok = tok_base + token_pre;
              if (tok < N) {
                float pv = fast_sigmoid(
                               sycl::fma(
                                   mixes_s[token_pre * HC3 + m_pre],
                                   hc_scale[0],
                                   hc_base[m_pre])) +
                           pre_eps;
                mixes_s[token_pre * HC3 + m_pre] = pv;

                float v = fast_sigmoid(
                              sycl::fma(
                                  mixes_s[token_pre * HC3 + m_pre + HC],
                                  hc_scale[1],
                                  hc_base[m_pre + HC])) *
                          post_mult;
                post_mix[tok * HC + m_pre] = v;
              }
            }
            sycl::group_barrier(sg);

            constexpr int COMB_LANES = HC * HC;
            constexpr int TOKS_PER_ROUND = SG_PRE / COMB_LANES;
            constexpr int NUM_COMB_ROUNDS =
                (TOKENS_PER_WG + TOKS_PER_ROUND - 1) / TOKS_PER_ROUND;

#pragma unroll
            for (int round = 0; round < NUM_COMB_ROUNDS; ++round) {
              int my_tok_in_round = lane / COMB_LANES;
              int my_idx = lane % COMB_LANES;
              int t = round * TOKS_PER_ROUND + my_tok_in_round;

              if (t < TOKENS_PER_WG && my_tok_in_round < TOKS_PER_ROUND) {
                int tok = tok_base + t;
                if (tok < N) {
                  float v = sycl::fma(
                      mixes_s[t * HC3 + my_idx + 2 * HC],
                      hc_scale[2],
                      hc_base[my_idx + 2 * HC]);

                  float vmax = v;
#pragma unroll
                  for (int off = 1; off < HC; off <<= 1)
                    vmax = sycl::max(
                        vmax, sycl::permute_group_by_xor(sg, vmax, off));

                  v = sycl::native::exp(v - vmax);

                  float rsum = v;
#pragma unroll
                  for (int off = 1; off < HC; off <<= 1)
                    rsum += sycl::permute_group_by_xor(sg, rsum, off);
                  v *= sycl::native::recip(rsum);
                  v += sk_eps;

#pragma unroll
                  for (int it_sk = 0; it_sk < sk_repeat; ++it_sk) {
                    float cs = v;
#pragma unroll
                    for (int off = HC; off < COMB_LANES; off <<= 1)
                      cs += sycl::permute_group_by_xor(sg, cs, off);
                    v *= sycl::native::recip(cs + sk_eps);

                    if (it_sk < sk_repeat - 1) {
                      float rs = v;
#pragma unroll
                      for (int off = 1; off < HC; off <<= 1)
                        rs += sycl::permute_group_by_xor(sg, rs, off);
                      v *= sycl::native::recip(rs + sk_eps);
                    }
                  }
                  comb_mix[tok * HC * HC + my_idx] = v;
                }
              }

              if (round < NUM_COMB_ROUNDS - 1) sycl::group_barrier(sg);
            }
          }

          sycl::group_barrier(it.get_group());

#pragma unroll
          for (int t = 0; t < TOKENS_PER_WG; ++t) {
            int tok = tok_base + t;
            if (tok >= N) break;

            float pre_reg[HC];
#pragma unroll
            for (int m = 0; m < HC; ++m)
              pre_reg[m] = mixes_s[t * HC3 + m];

            for (int k = tid * VEC_PRE; k < H; k += WG_THREADS_PRE * VEC_PRE) {
              float acc[VEC_PRE];
#pragma unroll
              for (int v = 0; v < VEC_PRE; ++v)
                acc[v] = 0.f;

#pragma unroll
              for (int m = 0; m < HC; ++m) {
                auto rv =
                    vload_bf16<VEC_PRE>(residual + (tok * HC + m) * H + k);
#pragma unroll
                for (int v = 0; v < VEC_PRE; ++v)
                  acc[v] += pre_reg[m] * float(rv[v]);
              }

              sycl::vec<bf16, VEC_PRE> ov;
#pragma unroll
              for (int v = 0; v < VEC_PRE; ++v)
                ov[v] = bf16(acc[v]);
              vstore_bf16<VEC_PRE>(layer_input + tok * H + k, ov);
            }
          }
        });
  });
}

}  // namespace vllm

using namespace vllm;

std::tuple<at::Tensor, at::Tensor, at::Tensor> mhc_pre(
    const at::Tensor& residual,
    const at::Tensor& fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    double rms_eps,
    double hc_pre_eps,
    double hc_sinkhorn_eps,
    double hc_post_mult_value,
    int64_t sinkhorn_repeat) {
  TORCH_CHECK(
      residual.is_xpu() && fn.is_xpu(), "mhc_pre: tensors must be on XPU");
  TORCH_CHECK(
      residual.scalar_type() == at::kBFloat16, "residual must be bfloat16");
  TORCH_CHECK(fn.scalar_type() == at::kFloat, "fn must be float32");
  TORCH_CHECK(hc_scale.scalar_type() == at::kFloat, "hc_scale must be float32");
  TORCH_CHECK(hc_base.scalar_type() == at::kFloat, "hc_base must be float32");

  auto residual_c = residual.contiguous();
  auto fn_c = fn.contiguous();

  const int64_t H = residual_c.size(-1);
  const int64_t HC2 = HC * HC;
  const int64_t HC3 = HC * 2 + HC2;

  TORCH_CHECK(
      fn_c.size(0) == HC3 && fn_c.size(1) == HC * H, "fn shape mismatch");
  TORCH_CHECK(hc_scale.numel() == 3, "hc_scale must have 3 elements");
  TORCH_CHECK(hc_base.numel() == HC3, "hc_base must have HC3 elements");

  auto outer_shape = residual_c.sizes().slice(0, residual_c.dim() - 2).vec();
  auto residual_flat = residual_c.view({-1, HC, H});
  const int64_t N = residual_flat.size(0);

  auto opts_f32 = residual_c.options().dtype(at::kFloat);
  auto opts_bf16 = residual_c.options().dtype(at::kBFloat16);

  auto post_mix = at::empty({N, HC}, opts_f32);
  auto comb_mix = at::empty({N, HC2}, opts_f32);
  auto layer_input = at::empty({N, H}, opts_bf16);

  if (N == 0) {
    std::vector<int64_t> ps = outer_shape;
    ps.push_back(HC);
    ps.push_back(1);
    std::vector<int64_t> cs = outer_shape;
    cs.push_back(HC);
    cs.push_back(HC);
    std::vector<int64_t> ls = outer_shape;
    ls.push_back(H);
    return {post_mix.view(ps), comb_mix.view(cs), layer_input.view(ls)};
  }

  auto& q = vllm::xpu::vllmGetQueue();
  launch_mhc_pre(
      q,
      reinterpret_cast<const bf16*>(residual_flat.data_ptr()),
      fn_c.data_ptr<float>(),
      hc_scale.data_ptr<float>(),
      hc_base.data_ptr<float>(),
      post_mix.data_ptr<float>(),
      comb_mix.data_ptr<float>(),
      reinterpret_cast<bf16*>(layer_input.data_ptr()),
      static_cast<int>(N),
      static_cast<int>(H),
      static_cast<float>(rms_eps),
      static_cast<float>(hc_pre_eps),
      static_cast<float>(hc_sinkhorn_eps),
      static_cast<float>(hc_post_mult_value),
      static_cast<int>(sinkhorn_repeat));

  std::vector<int64_t> ps = outer_shape;
  ps.push_back(HC);
  ps.push_back(1);
  std::vector<int64_t> cs = outer_shape;
  cs.push_back(HC);
  cs.push_back(HC);
  std::vector<int64_t> ls = outer_shape;
  ls.push_back(H);
  return {post_mix.view(ps), comb_mix.view(cs), layer_input.view(ls)};
}
