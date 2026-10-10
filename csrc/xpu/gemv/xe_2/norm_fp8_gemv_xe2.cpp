// SPDX-License-Identifier: Apache-2.0
//
// RMSNorm-prologue + FP8 GEMV fusions (decode, one launch each):
//
// gated_rmsnorm_fp8_gemv  (GDN out_proj; vLLM RMSNormGated, norm_before_gate)
//     y[m, h, :] = fp16( x * rsqrt(mean_D(x^2) + eps) * w_norm * silu(z) )
//     out        = fp16( y.view(M, H*D) @ (W * s)^T )
//
// resadd_rmsnorm_fp8_gemv[2]  (Gemma-style fused_add_rms_norm + qkv / qkvz+ba)
//     t            = x.float() + residual.float()
//     residual_out = fp16(t)
//     y            = fp16( t * rsqrt(mean_K(t^2) + eps) * (w_norm + 1) )
//     out_i        = fp16( y @ (W_i * s_i)^T )
//
// y never touches memory: every sub-group forms the normalized activations
// of the K chunks it multiplies directly in the DPAS A-operand layout (see
// fp8_gemv.hpp) and feeds them to the shared GEMV body.
//
// M == 1, K == 2048 (the dispatched case) uses register-resident prologues:
// each sub-group fetches the x / z / residual / norm-weight pieces of its own
// chunks with a few 2D block loads (issued with the first two weight chunks
// in flight), keeps them in registers and builds all A operands from
// registers.
//   * gated: head statistics come from the sub-group's own data (it also
//     loads the other half of each of its heads), silu(z) * w is computed in
//     the prologue; no SLM, no barrier. K is split 8 ways when that still
//     fits one wave (out_proj 2048 rows), else 4.
//   * resadd: sum((x+r)^2) over K needs the whole row -> one SLM reduction
//     and one barrier per work-group; work-group 0 writes residual_out.
// Other shapes (M <= 8, other K / D) use a generic, slower path that reloads
// the activations per chunk (correct, not tuned; supported() is M == 1 only).
//
// The graph-level *_fp8_gemm entries take the fused GEMV for M == 1 only;
// other inputs run one standalone SYCL RMSNorm kernel + fp8_gemm_w8a16.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

#include <ATen/DeviceGuard.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>

#include <cmath>
#include <optional>
#include <tuple>

#include "utils.h"
#include "fp8_gemv.hpp"
#include "../fp8_gemv_interface.h"

namespace vllm::fp8_gemv {

namespace syclex = sycl::ext::oneapi::experimental;

template <typename Scalar>
static inline uint16_t scalar_bits(float v) {
  return sycl::bit_cast<uint16_t>(Scalar(v));
}
template <typename Scalar>
static inline float lo_f(uint32_t d) {
  return float(sycl::bit_cast<Scalar>(uint16_t(d & 0xFFFFu)));
}
template <typename Scalar>
static inline float hi_f(uint32_t d) {
  return float(sycl::bit_cast<Scalar>(uint16_t(d >> 16)));
}
static inline float silu(float v) { return v / (1.f + sycl::exp(-v)); }
// prologue-side silu (hidden behind the weight preloads / barrier)
static inline float silu_fast(float v) {
  return v / (1.f + sycl::native::exp(-v));
}

// --------------------------------------------------- gated RMSNorm A -----
// x, z: [M, H, D] fp16 with strides (s_m, s_h, 1); w_norm [D] fp16.
template <int MP, typename Scalar>
struct GatedNormA {
  const Scalar* x;
  const Scalar* z;
  const Scalar* wn;
  int64_t xs_m, xs_h, zs_m, zs_h;
  int D;
  int M;
  float eps;

  inline void
  operator()(const sycl::sub_group& sg, int kc, int, AOps<MP>& A) const {
    const int lane = sg.get_local_linear_id();
    const int h = kc / D, d0 = kc - h * D;  // chunk lies inside one head
    const float inv_d = 1.f / float(D);
#pragma unroll
    for (int m = 0; m < MP; ++m) {
      if (m >= M) {
#pragma unroll
        for (int t = 0; t < kChunkSteps; ++t) {
          vset<MP>(A.e[t], m, short(0));
          vset<MP>(A.o[t], m, short(0));
        }
        continue;
      }
      const Scalar* xh = x + m * xs_m + h * xs_h;
      const Scalar* zh = z + m * zs_m + h * zs_h;
      // head statistics: lane covers 16 B vectors lane, lane + 16, ...
      float ss = 0.f;
      for (int v = lane; v < D / 8; v += kSg) {
        const sycl::vec<uint32_t, 4> q4 =
            reinterpret_cast<const sycl::vec<uint32_t, 4>*>(xh)[v];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float a = lo_f<Scalar>(q4[q]), b = hi_f<Scalar>(q4[q]);
          ss = sycl::fma(a, a, ss);
          ss = sycl::fma(b, b, ss);
        }
      }
      ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
      const float rstd = sycl::rsqrt(ss * inv_d + eps);
#pragma unroll
      for (int t = 0; t < kChunkSteps; ++t) {
        const int d = d0 + t * kStepK + 2 * lane;  // even element of the pair
        const uint32_t xv = *reinterpret_cast<const uint32_t*>(xh + d);
        const uint32_t zv = *reinterpret_cast<const uint32_t*>(zh + d);
        const uint32_t wv = *reinterpret_cast<const uint32_t*>(wn + d);
        const float y0 =
            lo_f<Scalar>(xv) * rstd * lo_f<Scalar>(wv) * silu(lo_f<Scalar>(zv));
        const float y1 =
            hi_f<Scalar>(xv) * rstd * hi_f<Scalar>(wv) * silu(hi_f<Scalar>(zv));
        vset<MP>(A.e[t], m, short(scalar_bits<Scalar>(y0)));
        vset<MP>(A.o[t], m, short(scalar_bits<Scalar>(y1)));
      }
    }
  }
};

// ------------------------------------------ residual-add Gemma RMSNorm A -----
// x, r: [M, K] fp16 (row strides xs, rs); w_norm [K] fp16; rstd[m] from the
// work-group prologue.
template <int MP, typename Scalar>
struct ResAddNormA {
  const Scalar* x;
  const Scalar* r;
  const Scalar* wn;
  int64_t xs, rs;
  const float* rstd;  // SLM, [MP]
  int M;

  inline void
  operator()(const sycl::sub_group& sg, int kc, int, AOps<MP>& A) const {
    const int lane = sg.get_local_linear_id();
#pragma unroll
    for (int t = 0; t < kChunkSteps; ++t) {
      const int k = kc + t * kStepK + 2 * lane;
      const uint32_t wv = *reinterpret_cast<const uint32_t*>(wn + k);
      const float w0 = lo_f<Scalar>(wv) + 1.f, w1 = hi_f<Scalar>(wv) + 1.f;
#pragma unroll
      for (int m = 0; m < MP; ++m) {
        if (m < M) {
          const uint32_t xv =
              *reinterpret_cast<const uint32_t*>(x + m * xs + k);
          const uint32_t rv =
              *reinterpret_cast<const uint32_t*>(r + m * rs + k);
          const float rs_m = rstd[m];
          const float y0 = (lo_f<Scalar>(xv) + lo_f<Scalar>(rv)) * rs_m * w0;
          const float y1 = (hi_f<Scalar>(xv) + hi_f<Scalar>(rv)) * rs_m * w1;
          vset<MP>(A.e[t], m, short(scalar_bits<Scalar>(y0)));
          vset<MP>(A.o[t], m, short(scalar_bits<Scalar>(y1)));
        } else {
          vset<MP>(A.e[t], m, short(0));
          vset<MP>(A.o[t], m, short(0));
        }
      }
    }
  }
};

// ------------------------------------------------------------ kernels -----
// Weight chunks put in flight before a prologue (1..4 measured: 2 is best,
// 3+ spills with 128 GRF).
constexpr int NPRE = 2;

template <int MP, int KS, int RG, typename Scalar>
struct GatedNormGemvKernel {
  GatedNormA<MP, Scalar> ga;
  Seg<Scalar> seg;
  int K;

  [[sycl::reqd_sub_group_size(kSg)]] void
  operator()(sycl::nd_item<1> item) const {
#ifdef __SYCL_DEVICE_ONLY__
    auto* part = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[RG * KS * MP * kSg]>(item.get_group());
    const u32x8 none[1][kChunkSteps] = {};
    gemv_body<MP, KS, RG>(
        item,
        seg,
        int(item.get_group(0)) * RG * kRowBlock,
        ga,
        part,
        ga.M,
        K,
        none,
        false);
#endif
  }
};

template <int MP, int KS, int RG, int NSEG, typename Scalar>
struct ResAddNormGemvKernel {
  const Scalar* x;
  const Scalar* r;
  const Scalar* wn;
  Scalar* r_out;
  int64_t xs, rs, ros;
  int M, K;
  float eps;
  Seg<Scalar> seg0, seg1;

  [[sycl::reqd_sub_group_size(kSg)]] void
  operator()(sycl::nd_item<1> item) const {
#ifdef __SYCL_DEVICE_ONLY__
    constexpr int NSG = RG * KS;
    auto* part = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[NSG * MP * kSg]>(item.get_group());
    auto* stat = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[NSG * MP + MP]>(item.get_group());
    auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int sgid = sg.get_group_linear_id();
    const int ks = sgid % KS, rg = sgid / KS;
    const int g = int(item.get_group(0));
    const bool second = NSEG == 2 && g >= seg1.wg_begin;
    const Seg<Scalar>& seg = second ? seg1 : seg0;
    const int wg_row0 = (g - seg.wg_begin) * RG * kRowBlock;
    const int row0 = wg_row0 + rg * kRowBlock;

    // 1. put the first weight chunk in flight before the reduction
    u32x8 wpre[NPRE][kChunkSteps];
    const bool pre = preload_w<NPRE, KS>(seg, row0, ks, K, wpre);

    // 2. sum((x + r)^2) per token, split over the work-group's sub-groups
    //    (16 B vectors, 4 independent loads per lane in flight);
    //    work-group 0 also writes residual_out = fp16(x + r).
    using u32x4 = sycl::vec<uint32_t, 4>;
    const bool write_res = g == 0;
    const int KV = K / 8, per = (KV + NSG - 1) / NSG;  // 16 B vectors
    const int v_begin = sgid * per, v_end = sycl::min(KV, v_begin + per);
  #pragma unroll
    for (int m = 0; m < MP; ++m) {
      float ss = 0.f;
      if (m < M) {
        const u32x4* xp = reinterpret_cast<const u32x4*>(x + m * xs);
        const u32x4* rp = reinterpret_cast<const u32x4*>(r + m * rs);
        u32x4* op = reinterpret_cast<u32x4*>(r_out + m * ros);
        for (int v0 = v_begin; v0 < v_end; v0 += 4 * kSg) {
          u32x4 xv[4], rv[4];
  #pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int v = v0 + i * kSg + lane;
            if (v < v_end) {
              xv[i] = xp[v];
              rv[i] = rp[v];
            } else {
              xv[i] = 0u;
              rv[i] = 0u;
            }
          }
  #pragma unroll
          for (int i = 0; i < 4; ++i) {
            u32x4 o;
  #pragma unroll
            for (int q = 0; q < 4; ++q) {
              const float t0 = lo_f<Scalar>(xv[i][q]) + lo_f<Scalar>(rv[i][q]);
              const float t1 = hi_f<Scalar>(xv[i][q]) + hi_f<Scalar>(rv[i][q]);
              ss = sycl::fma(t0, t0, ss);
              ss = sycl::fma(t1, t1, ss);
              o[q] = uint32_t(scalar_bits<Scalar>(t0)) |
                     (uint32_t(scalar_bits<Scalar>(t1)) << 16);
            }
            const int v = v0 + i * kSg + lane;
            if (write_res && v < v_end) op[v] = o;
          }
        }
      }
      ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
      if (lane == 0) stat[sgid * MP + m] = ss;
    }
    sycl::group_barrier(item.get_group());
    if (sgid == 0 && lane < MP) {
      float ss = 0.f;
      for (int q = 0; q < NSG; ++q)
        ss += stat[q * MP + lane];
      stat[NSG * MP + lane] = sycl::rsqrt(ss / float(K) + eps);
    }
    sycl::group_barrier(item.get_group());

    const ResAddNormA<MP, Scalar> la{x, r, wn, xs, rs, stat + NSG * MP, M};
    gemv_body<MP, KS, RG, ResAddNormA<MP, Scalar>, NPRE>(
        item, seg, wg_row0, la, part, M, K, wpre, pre);
#endif
  }
};

// ------------------------------------------- M = 1 register-resident path
// ----- For M == 1 and K == CPS * KS * 64, sub-group ks owns chunks ks + i * KS
// (i < CPS), which sit at a constant stride in memory. It fetches the
// activations of exactly those chunks with 2D block loads (surface = CPS
// rows of one 128 B chunk, pitch = chunk stride; one message per 32-K step
// and tensor), in the DPAS A-lane layout, keeps them in registers across the
// statistics reduction and forms the A operands from registers. Activation
// loads are issued before the weight preloads (16 send tokens per thread),
// otherwise they queue behind DRAM latency.
template <int CPS, typename Scalar>
static inline void load_chunks(
    const Scalar* base,   // first chunk of this sub-group
    int64_t pitch_elems,  // element stride between its chunks
    uint32_t (&d)[CPS][kChunkSteps]) {
#ifdef __SYCL_DEVICE_ONLY__
  uint32_t v[kChunkSteps][CPS];
  #pragma unroll
  for (int t = 0; t < kChunkSteps; ++t) {
    cute::intel::coord_t co;
    co[0] = t * kSg;  // dwords
    co[1] = 0;
    cute::detail::XeSubgroup2DBlockLoad<4, 16, CPS, 1>{}(
        base, kChunkK * 2, CPS, int(pitch_elems * 2), co, &v[t][0]);
  }
  #pragma unroll
  for (int i = 0; i < CPS; ++i)
  #pragma unroll
    for (int t = 0; t < kChunkSteps; ++t)
      d[i][t] = v[t][i];
#endif
}

template <int CPS, typename Scalar>
static inline void store_chunks(
    Scalar* base, int64_t pitch_elems, const uint32_t (&d)[CPS][kChunkSteps]) {
#ifdef __SYCL_DEVICE_ONLY__
  uint32_t v[kChunkSteps][CPS];
  #pragma unroll
  for (int i = 0; i < CPS; ++i)
  #pragma unroll
    for (int t = 0; t < kChunkSteps; ++t)
      v[t][i] = d[i][t];
  #pragma unroll
  for (int t = 0; t < kChunkSteps; ++t) {
    cute::intel::coord_t co;
    co[0] = t * kSg;
    co[1] = 0;
    cute::detail::XeSubgroup2DBlockStore<4, 16, CPS, 1>{}(
        base, kChunkK * 2, CPS, int(pitch_elems * 2), co, &v[t][0]);
  }
#endif
}

template <int CPS>
struct RegA {
  uint32_t xd[CPS][kChunkSteps];  // fp16 pairs of x
  uint32_t gd[CPS][kChunkSteps];  // gate z / residual r
  uint32_t wd[CPS][kChunkSteps];  // norm weight (resadd; gated uses wd[0])
  float rstd[CPS];
};
// gated: w_norm * silu(z) per element, precomputed in the prologue
template <int CPS>
struct GateF {
  float g[CPS][kChunkSteps][2];
};

template <int CPS, typename Scalar>
struct ResAddRegA {
  const RegA<CPS>* d;
  inline void
  operator()(const sycl::sub_group&, int, int it, AOps<1>& A) const {
#pragma unroll
    for (int t = 0; t < kChunkSteps; ++t) {
      const uint32_t xv = d->xd[it][t], rv = d->gd[it][t], wv = d->wd[it][t];
      const float y0 = (lo_f<Scalar>(xv) + lo_f<Scalar>(rv)) * d->rstd[0] *
                       (lo_f<Scalar>(wv) + 1.f);
      const float y1 = (hi_f<Scalar>(xv) + hi_f<Scalar>(rv)) * d->rstd[0] *
                       (hi_f<Scalar>(wv) + 1.f);
      A.e[t] = short(scalar_bits<Scalar>(y0));
      A.o[t] = short(scalar_bits<Scalar>(y1));
    }
  }
};

template <int CPS, typename Scalar>
struct GatedRegA {
  const RegA<CPS>* d;
  const GateF<CPS>* gf;
  inline void
  operator()(const sycl::sub_group&, int, int it, AOps<1>& A) const {
#pragma unroll
    for (int t = 0; t < kChunkSteps; ++t) {
      const uint32_t xv = d->xd[it][t];
      const float y0 = lo_f<Scalar>(xv) * d->rstd[it] * gf->g[it][t][0];
      const float y1 = hi_f<Scalar>(xv) * d->rstd[it] * gf->g[it][t][1];
      A.e[t] = short(scalar_bits<Scalar>(y0));
      A.o[t] = short(scalar_bits<Scalar>(y1));
    }
  }
};

template <int KS, int NSEG, int CPS, typename Scalar>
struct ResAddNormGemvRegKernel {
  const Scalar* x;
  const Scalar* r;
  const Scalar* wn;
  Scalar* r_out;
  int K;
  float eps;
  Seg<Scalar> seg0, seg1;

  [[sycl::reqd_sub_group_size(kSg)]] void
  operator()(sycl::nd_item<1> item) const {
#ifdef __SYCL_DEVICE_ONLY__
    auto* part =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[KS * kSg]>(
            item.get_group());
    auto* stat =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[KS]>(
            item.get_group());
    auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int ks = sg.get_group_linear_id();
    const int g = int(item.get_group(0));
    const bool second = NSEG == 2 && g >= seg1.wg_begin;
    const Seg<Scalar>& seg = second ? seg1 : seg0;
    const int row0 = (g - seg.wg_begin) * kRowBlock;

    constexpr int64_t kPitch = int64_t(KS) * kChunkK;  // between own chunks
    const int64_t e0 = int64_t(ks) * kChunkK;
    RegA<CPS> d;
    load_chunks<CPS>(x + e0, kPitch, d.xd);
    load_chunks<CPS>(r + e0, kPitch, d.gd);
    load_chunks<CPS>(wn + e0, kPitch, d.wd);
    u32x8 wpre[NPRE][kChunkSteps];
    const bool pre = preload_w<NPRE, KS>(seg, row0, ks, K, wpre);

    float ss = 0.f;
    uint32_t td[CPS][kChunkSteps];
  #pragma unroll
    for (int i = 0; i < CPS; ++i)
  #pragma unroll
      for (int t = 0; t < kChunkSteps; ++t) {
        const float t0 = lo_f<Scalar>(d.xd[i][t]) + lo_f<Scalar>(d.gd[i][t]);
        const float t1 = hi_f<Scalar>(d.xd[i][t]) + hi_f<Scalar>(d.gd[i][t]);
        ss = sycl::fma(t0, t0, ss);
        ss = sycl::fma(t1, t1, ss);
        td[i][t] = uint32_t(scalar_bits<Scalar>(t0)) |
                   (uint32_t(scalar_bits<Scalar>(t1)) << 16);
      }
    if (g == 0) store_chunks<CPS>(r_out + e0, kPitch, td);  // residual_out
    ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
    if (lane == 0) stat[ks] = ss;
    sycl::group_barrier(item.get_group());
    float tot = 0.f;
  #pragma unroll
    for (int q = 0; q < KS; ++q)
      tot += stat[q];
    d.rstd[0] = sycl::rsqrt(tot / float(K) + eps);

    const ResAddRegA<CPS, Scalar> la{&d};
    gemv_body<1, KS, 1, ResAddRegA<CPS, Scalar>, NPRE, CPS>(
        item, seg, row0, la, part, 1, K, wpre, pre);
#endif
  }
};

// gated: x, z [1, H, D] (head strides xs_h, zs_h), CPH = D / 64 chunks per
// head with CPH | KS, so all chunks of sub-group ks share d0 and sit
// KS / CPH heads apart. The sub-group also loads the other CPH - 1 chunks of
// each of its heads (x only), so head statistics need no cross-sub-group
// exchange (no SLM, no barrier).
template <int KS, int CPS, int CPH, typename Scalar>
struct GatedNormGemvRegKernel {
  const Scalar* x;
  const Scalar* z;
  const Scalar* wn;
  int64_t xs_h, zs_h;
  int K;
  float eps;
  Seg<Scalar> seg;

  [[sycl::reqd_sub_group_size(kSg)]] void
  operator()(sycl::nd_item<1> item) const {
#ifdef __SYCL_DEVICE_ONLY__
    static_assert(
        KS % CPH == 0, "a head must not straddle sub-groups' strides");
    constexpr int D = CPH * kChunkK;
    constexpr int HSTEP = KS / CPH;  // heads between own chunks
    auto* part =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[KS * kSg]>(
            item.get_group());
    auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int ks = sg.get_group_linear_id();
    const int row0 = int(item.get_group(0)) * kRowBlock;

    const int h0 = ks / CPH, q_own = ks % CPH, d0 = q_own * kChunkK;
    RegA<CPS> d;
    load_chunks<CPS>(x + h0 * xs_h + d0, HSTEP * xs_h, d.xd);
    load_chunks<CPS>(z + h0 * zs_h + d0, HSTEP * zs_h, d.gd);
    uint32_t xo[CPH > 1 ? CPH - 1 : 1][CPS][kChunkSteps];
  #pragma unroll
    for (int q = 0, o = 0; q < CPH; ++q) {
      if (q == q_own) continue;
      load_chunks<CPS>(x + h0 * xs_h + q * kChunkK, HSTEP * xs_h, xo[o]);
      ++o;
    }
  #pragma unroll
    for (int t = 0; t < kChunkSteps; ++t)
      d.wd[0][t] = reinterpret_cast<const uint32_t*>(wn + d0)[t * kSg + lane];
    u32x8 wpre[NPRE][kChunkSteps];
    const bool pre = preload_w<NPRE, KS>(seg, row0, ks, K, wpre);

    GateF<CPS> gf;
  #pragma unroll
    for (int i = 0; i < CPS; ++i) {
      float ss = 0.f;
  #pragma unroll
      for (int t = 0; t < kChunkSteps; ++t) {
        float a = lo_f<Scalar>(d.xd[i][t]), b = hi_f<Scalar>(d.xd[i][t]);
        ss = sycl::fma(a, a, ss);
        ss = sycl::fma(b, b, ss);
  #pragma unroll
        for (int o = 0; o < CPH - 1; ++o) {
          a = lo_f<Scalar>(xo[o][i][t]);
          b = hi_f<Scalar>(xo[o][i][t]);
          ss = sycl::fma(a, a, ss);
          ss = sycl::fma(b, b, ss);
        }
        gf.g[i][t][0] =
            lo_f<Scalar>(d.wd[0][t]) * silu_fast(lo_f<Scalar>(d.gd[i][t]));
        gf.g[i][t][1] =
            hi_f<Scalar>(d.wd[0][t]) * silu_fast(hi_f<Scalar>(d.gd[i][t]));
      }
      ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
      d.rstd[i] = sycl::rsqrt(ss * (1.f / float(D)) + eps);
    }

    const GatedRegA<CPS, Scalar> la{&d, &gf};
    gemv_body<1, KS, 1, GatedRegA<CPS, Scalar>, NPRE, CPS>(
        item, seg, row0, la, part, 1, K, wpre, pre);
#endif
  }
};

// --------------------------------------------------------- host side -----
static inline bool aligned(const void* p, uintptr_t a) {
  return (reinterpret_cast<uintptr_t>(p) & (a - 1)) == 0;
}

static bool w_ok(const torch::Tensor& w, const torch::Tensor& x, int64_t K) {
  return w.device() == x.device() && w.dim() == 2 &&
         w.scalar_type() == at::ScalarType::Float8_e4m3fn && w.size(1) == K &&
         w.size(0) >= 1 && w.size(0) < (int64_t(1) << 24) && w.stride(1) == 1 &&
         w.stride(0) == K && aligned(w.data_ptr(), 64) && K % kChunkK == 0 &&
         K < (int64_t(1) << 24);
}
static bool s_ok(const torch::Tensor& s, const torch::Tensor& x) {
  return s.device() == x.device() && s.scalar_type() == at::kFloat &&
         s.numel() == 1;
}
static void check_w(
    const torch::Tensor& w,
    const torch::Tensor& s,
    const torch::Tensor& x,
    int64_t K,
    const char* name) {
  TORCH_CHECK(
      w_ok(w, x, K),
      "norm_fp8_gemv: ",
      name,
      " must be a 64 B aligned row-major float8_e4m3fn [N, K=",
      K,
      "] tensor on the input device (K % 64 == 0), got ",
      w.sizes(),
      " ",
      w.scalar_type(),
      " strides ",
      w.strides());
  TORCH_CHECK(
      s_ok(s, x),
      "norm_fp8_gemv: scale for ",
      name,
      " must be a 1-element float32 tensor on the input device");
}

// [M, H, D] or [M, H*D] fp16 activation; returns (s_m, s_h).
static bool heads_ok(
    const torch::Tensor& t,
    int64_t M,
    int64_t H,
    int64_t D,
    int64_t& s_m,
    int64_t& s_h) {
  if (!t.is_xpu() ||
      (t.scalar_type() != at::kHalf && t.scalar_type() != at::kBFloat16))
    return false;
  if (t.dim() == 3) {
    if (t.size(0) != M || t.size(1) != H || t.size(2) != D || t.stride(2) != 1)
      return false;
    s_m = t.stride(0);
    s_h = t.stride(1);
  } else if (t.dim() == 2) {
    if (t.size(0) != M || t.size(1) != H * D || t.stride(1) != 1) return false;
    s_m = t.stride(0);
    s_h = D;
  } else {
    return false;
  }
  // 16 B vector loads of head rows, dword loads of fp16 pairs
  return aligned(t.data_ptr(), 16) && s_m % 8 == 0 && s_h % 8 == 0;
}

static bool is_bmg_cached(int dev) {
  static thread_local int last_dev = -1;
  static thread_local bool last_ok = false;
  if (dev != last_dev) {
    last_ok = vllm::xpu::is_bmg(dev);
    last_dev = dev;
  }
  return last_ok;
}

// ---- gated -----------------------------------------------------------------
static bool gated_shapes(
    const torch::Tensor& x,
    const torch::Tensor& z,
    const torch::Tensor& wn,
    int64_t& M,
    int64_t& H,
    int64_t& D,
    int64_t xs[2],
    int64_t zs[2]) {
  if (!x.is_xpu() || (x.dim() != 2 && x.dim() != 3) || wn.dim() != 1)
    return false;
  D = wn.size(0);
  if (D <= 0 || D % kChunkK != 0) return false;
  M = x.size(0);
  H = x.dim() == 3 ? x.size(1) : x.size(1) / D;
  if (x.dim() == 2 && x.size(1) % D != 0) return false;
  if (M < 1 || M > kMaxM || H < 1) return false;
  return heads_ok(x, M, H, D, xs[0], xs[1]) &&
         heads_ok(z, M, H, D, zs[0], zs[1]) &&
         z.scalar_type() == x.scalar_type() && z.device() == x.device() &&
         wn.device() == x.device() && wn.scalar_type() == x.scalar_type() &&
         wn.stride(0) == 1 && aligned(wn.data_ptr(), 4);
}

bool gated_rmsnorm_fp8_gemv_supported(
    const torch::Tensor& x,
    const torch::Tensor& z,
    const torch::Tensor& norm_weight,
    const torch::Tensor& w,
    const std::optional<torch::Tensor>& scale) {
  int64_t M, H, D, xs[2], zs[2];
  if (!gated_shapes(x, z, norm_weight, M, H, D, xs, zs)) return false;
  if (M != 1) return false;  // dispatch policy: M == 1 only
  if (!scale.has_value() || !w_ok(w, x, H * D) || !s_ok(*scale, x))
    return false;
  return is_bmg_cached(x.get_device());
}

template <typename Scalar>
torch::Tensor gated_rmsnorm_fp8_gemv_impl(
    const torch::Tensor& x,
    const torch::Tensor& z,
    const torch::Tensor& norm_weight,
    double eps,
    const torch::Tensor& w,
    const torch::Tensor& scale) {
  int64_t M, H, D, xs[2], zs[2];
  TORCH_CHECK(
      gated_shapes(x, z, norm_weight, M, H, D, xs, zs),
      "gated_rmsnorm_fp8_gemv: x, z must be matching fp16/bf16 [M<=8, H, D] "
      "(or [M, H*D]) "
      "with contiguous last dim, strides multiple of 8 and 16 B alignment; "
      "norm_weight "
      "fp16/bf16 [D] contiguous with D % 64 == 0 (got x ",
      x.sizes(),
      " z ",
      z.sizes(),
      " norm_weight ",
      norm_weight.sizes(),
      ")");
  const int64_t K = H * D;
  check_w(w, scale, x, K, "w");
  const at::DeviceGuard guard(x.device());
  const int64_t N = w.size(0);
  auto out = at::empty({M, N}, x.options());
  auto& q = vllm::xpu::vllmGetQueue();
  Seg<Scalar> seg{
      reinterpret_cast<const uint8_t*>(w.data_ptr()),
      scale.data_ptr<float>(),
      reinterpret_cast<Scalar*>(out.data_ptr()),
      int(N),
      int(N),
      0};
  auto launch = [&](auto mp_tag) {
    constexpr int MP = decltype(mp_tag)::value;
    constexpr int KS = 4, RG = 1;
    GatedNormA<MP, Scalar> ga{
        reinterpret_cast<const Scalar*>(x.data_ptr()),
        reinterpret_cast<const Scalar*>(z.data_ptr()),
        reinterpret_cast<const Scalar*>(norm_weight.data_ptr()),
        xs[0],
        xs[1],
        zs[0],
        zs[1],
        int(D),
        int(M),
        float(eps)};
    const int wgs = int((N + RG * kRowBlock - 1) / (RG * kRowBlock));
    syclex::nd_launch(
        q,
        sycl::nd_range<1>(size_t(wgs) * RG * KS * kSg, RG * KS * kSg),
        GatedNormGemvKernel<MP, KS, RG, Scalar>{ga, seg, int(K)});
  };
  // M = 1, K = 2048 register-resident path. K split: 8 sub-groups per row
  // block while that still fits one wave (<= 2048 sub-groups, e.g. the
  // 2048-row out_proj: 11.1 -> 10.1 us), else 4.
  const int64_t row_blocks = (N + kRowBlock - 1) / kRowBlock;
  const bool reg_path =
      M == 1 && K == 2048 && (D == 64 || D == 128 || D == 256) &&
      aligned(x.data_ptr(), 64) && aligned(z.data_ptr(), 64) &&
      (xs[1] * 2) % 64 == 0 && (zs[1] * 2) % 64 == 0;
  if (reg_path) {
    auto go = [&](auto ks_tag, auto cph_tag) {
      constexpr int KS = decltype(ks_tag)::value;
      constexpr int CPH = decltype(cph_tag)::value;
      constexpr int CPS = 2048 / (KS * kChunkK);
      syclex::nd_launch(
          q,
          sycl::nd_range<1>(size_t(row_blocks) * KS * kSg, KS * kSg),
          GatedNormGemvRegKernel<KS, CPS, CPH, Scalar>{
              reinterpret_cast<const Scalar*>(x.data_ptr()),
              reinterpret_cast<const Scalar*>(z.data_ptr()),
              reinterpret_cast<const Scalar*>(norm_weight.data_ptr()),
              xs[1],
              zs[1],
              int(K),
              float(eps),
              seg});
    };
    auto by_d = [&](auto ks_tag) {
      if (D == 64)
        go(ks_tag, std::integral_constant<int, 1>{});
      else if (D == 128)
        go(ks_tag, std::integral_constant<int, 2>{});
      else
        go(ks_tag, std::integral_constant<int, 4>{});
    };
    if (row_blocks * 8 <= 2048)
      by_d(std::integral_constant<int, 8>{});
    else
      by_d(std::integral_constant<int, 4>{});
  } else if (M == 1) {
    launch(std::integral_constant<int, 1>{});
  } else if (M == 2) {
    launch(std::integral_constant<int, 2>{});
  } else if (M <= 4) {
    launch(std::integral_constant<int, 4>{});
  } else {
    launch(std::integral_constant<int, 8>{});
  }
  return out;
}

torch::Tensor gated_rmsnorm_fp8_gemv(
    const torch::Tensor& x,
    const torch::Tensor& z,
    const torch::Tensor& norm_weight,
    double eps,
    const torch::Tensor& w,
    const torch::Tensor& scale) {
  if (x.scalar_type() == at::kHalf)
    return gated_rmsnorm_fp8_gemv_impl<sycl::half>(
        x, z, norm_weight, eps, w, scale);
  return gated_rmsnorm_fp8_gemv_impl<sycl::ext::oneapi::bfloat16>(
      x, z, norm_weight, eps, w, scale);
}

// ---- residual add + Gemma RMSNorm ------------------------------------------
static bool resadd_shapes(
    const torch::Tensor& x,
    const torch::Tensor& r,
    const torch::Tensor& wn,
    int64_t& M,
    int64_t& K) {
  if (!x.is_xpu() || x.dim() != 2 || r.dim() != 2 || wn.dim() != 1)
    return false;
  M = x.size(0);
  K = x.size(1);
  return M >= 1 && M <= kMaxM && K % kChunkK == 0 &&
         (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) &&
         r.scalar_type() == x.scalar_type() &&
         wn.scalar_type() == x.scalar_type() && r.sizes() == x.sizes() &&
         wn.size(0) == K && x.stride(1) == 1 && r.stride(1) == 1 &&
         wn.stride(0) == 1 && x.stride(0) % 8 == 0 && r.stride(0) % 8 == 0 &&
         aligned(x.data_ptr(), 16) && aligned(r.data_ptr(), 16) &&
         aligned(wn.data_ptr(), 4) && r.device() == x.device() &&
         wn.device() == x.device();
}

bool resadd_rmsnorm_fp8_gemv_supported(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& norm_weight,
    const torch::Tensor& w,
    const std::optional<torch::Tensor>& scale) {
  int64_t M, K;
  if (!resadd_shapes(x, residual, norm_weight, M, K)) return false;
  if (M != 1) return false;  // dispatch policy: M == 1 only
  if (!scale.has_value() || !w_ok(w, x, K) || !s_ok(*scale, x)) return false;
  return is_bmg_cached(x.get_device());
}

template <int NSEG, typename Scalar>
static void launch_resadd(
    sycl::queue& q,
    const torch::Tensor& x,
    const torch::Tensor& r,
    const torch::Tensor& wn,
    torch::Tensor& r_out,
    double eps,
    int64_t M,
    int64_t K,
    Seg<Scalar> s0,
    Seg<Scalar> s1) {
  constexpr int KS = 4, RG = 1;
  static_assert(RG == 1, "register path assumes one row block per group");
  if (NSEG == 2 && s1.N < s0.N) std::swap(s0, s1);  // small weight first
  constexpr int rows = RG * kRowBlock;
  const int w0 = (s0.N + rows - 1) / rows;
  s0.wg_begin = 0;
  s1.wg_begin = w0;
  const int total = w0 + (NSEG == 2 ? (s1.N + rows - 1) / rows : 0);
  auto go = [&](auto mp_tag) {
    constexpr int MP = decltype(mp_tag)::value;
    syclex::nd_launch(
        q,
        sycl::nd_range<1>(size_t(total) * RG * KS * kSg, RG * KS * kSg),
        ResAddNormGemvKernel<MP, KS, RG, NSEG, Scalar>{
            reinterpret_cast<const Scalar*>(x.data_ptr()),
            reinterpret_cast<const Scalar*>(r.data_ptr()),
            reinterpret_cast<const Scalar*>(wn.data_ptr()),
            reinterpret_cast<Scalar*>(r_out.data_ptr()),
            x.stride(0),
            r.stride(0),
            r_out.stride(0),
            int(M),
            int(K),
            float(eps),
            s0,
            s1});
  };
  constexpr int kCps = 8;
  if (M == 1 && K == kCps * KS * kChunkK && aligned(x.data_ptr(), 64) &&
      aligned(r.data_ptr(), 64) && aligned(wn.data_ptr(), 64)) {
    syclex::nd_launch(
        q,
        sycl::nd_range<1>(size_t(total) * RG * KS * kSg, RG * KS * kSg),
        ResAddNormGemvRegKernel<KS, NSEG, kCps, Scalar>{
            reinterpret_cast<const Scalar*>(x.data_ptr()),
            reinterpret_cast<const Scalar*>(r.data_ptr()),
            reinterpret_cast<const Scalar*>(wn.data_ptr()),
            reinterpret_cast<Scalar*>(r_out.data_ptr()),
            int(K),
            float(eps),
            s0,
            s1});
    return;
  }
  if (M == 1)
    go(std::integral_constant<int, 1>{});
  else if (M == 2)
    go(std::integral_constant<int, 2>{});
  else if (M <= 4)
    go(std::integral_constant<int, 4>{});
  else
    go(std::integral_constant<int, 8>{});
}

static void check_resadd(
    const torch::Tensor& x,
    const torch::Tensor& r,
    const torch::Tensor& wn,
    int64_t& M,
    int64_t& K) {
  TORCH_CHECK(
      resadd_shapes(x, r, wn, M, K),
      "resadd_rmsnorm_fp8_gemv: x, residual must be matching fp16/bf16 [M<=8, "
      "K] (same "
      "shape, contiguous last dim, row stride multiple of 8, 16 B aligned), "
      "norm_weight "
      "fp16/bf16 [K] contiguous, K % 64 == 0 (got x ",
      x.sizes(),
      " residual ",
      r.sizes(),
      " norm_weight ",
      wn.sizes(),
      ")");
}

template <typename Scalar>
std::tuple<torch::Tensor, torch::Tensor> resadd_rmsnorm_fp8_gemv_impl(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& norm_weight,
    double eps,
    const torch::Tensor& w,
    const torch::Tensor& scale) {
  int64_t M, K;
  check_resadd(x, residual, norm_weight, M, K);
  check_w(w, scale, x, K, "w");
  const at::DeviceGuard guard(x.device());
  const int64_t N = w.size(0);
  auto out = at::empty({M, N}, x.options());
  auto r_out = at::empty({M, K}, x.options());
  auto& q = vllm::xpu::vllmGetQueue();
  Seg<Scalar> s0{
      reinterpret_cast<const uint8_t*>(w.data_ptr()),
      scale.data_ptr<float>(),
      reinterpret_cast<Scalar*>(out.data_ptr()),
      int(N),
      int(N),
      0};
  launch_resadd<1>(q, x, residual, norm_weight, r_out, eps, M, K, s0, s0);
  return {out, r_out};
}

std::tuple<torch::Tensor, torch::Tensor> resadd_rmsnorm_fp8_gemv(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& norm_weight,
    double eps,
    const torch::Tensor& w,
    const torch::Tensor& scale) {
  if (x.scalar_type() == at::kHalf)
    return resadd_rmsnorm_fp8_gemv_impl<sycl::half>(
        x, residual, norm_weight, eps, w, scale);
  return resadd_rmsnorm_fp8_gemv_impl<sycl::ext::oneapi::bfloat16>(
      x, residual, norm_weight, eps, w, scale);
}

template <typename Scalar>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
resadd_rmsnorm_fp8_gemv2_impl(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& norm_weight,
    double eps,
    const torch::Tensor& w1,
    const torch::Tensor& scale1,
    const torch::Tensor& w2,
    const torch::Tensor& scale2) {
  int64_t M, K;
  check_resadd(x, residual, norm_weight, M, K);
  check_w(w1, scale1, x, K, "w1");
  check_w(w2, scale2, x, K, "w2");
  const at::DeviceGuard guard(x.device());
  const int64_t N1 = w1.size(0), N2 = w2.size(0);
  auto out1 = at::empty({M, N1}, x.options());
  auto out2 = at::empty({M, N2}, x.options());
  auto r_out = at::empty({M, K}, x.options());
  auto& q = vllm::xpu::vllmGetQueue();
  Seg<Scalar> s0{
      reinterpret_cast<const uint8_t*>(w1.data_ptr()),
      scale1.data_ptr<float>(),
      reinterpret_cast<Scalar*>(out1.data_ptr()),
      int(N1),
      int(N1),
      0};
  Seg<Scalar> s1{
      reinterpret_cast<const uint8_t*>(w2.data_ptr()),
      scale2.data_ptr<float>(),
      reinterpret_cast<Scalar*>(out2.data_ptr()),
      int(N2),
      int(N2),
      0};
  launch_resadd<2>(q, x, residual, norm_weight, r_out, eps, M, K, s0, s1);
  return {out1, out2, r_out};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
resadd_rmsnorm_fp8_gemv2(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& norm_weight,
    double eps,
    const torch::Tensor& w1,
    const torch::Tensor& scale1,
    const torch::Tensor& w2,
    const torch::Tensor& scale2) {
  if (x.scalar_type() == at::kHalf)
    return resadd_rmsnorm_fp8_gemv2_impl<sycl::half>(
        x, residual, norm_weight, eps, w1, scale1, w2, scale2);
  return resadd_rmsnorm_fp8_gemv2_impl<sycl::ext::oneapi::bfloat16>(
      x, residual, norm_weight, eps, w1, scale1, w2, scale2);
}

// ---------------------------------------------------------------------------
// Standalone RMSNorm kernels for the unfused path (M > 1): one launch per
// norm instead of an eager ATen op chain, whose per-op host cost dominated
// multi-row decode. Same math as the fused prologues (fp32, one work-group
// per row / per (row, head)); fp16 in and out.
// ---------------------------------------------------------------------------
constexpr int kNormWg = 256;

template <typename Scalar>
struct GatedNormKernel {
  const Scalar* x;
  const Scalar* z;
  const Scalar* wn;
  Scalar* y;  // [M, H * D] contiguous
  int64_t xs_m, xs_h, zs_m, zs_h;
  int H, D;
  float eps;

  void operator()(sycl::nd_item<1> it) const {
    const int g = it.get_group(0), lid = it.get_local_id(0);
    const int m = g / H, h = g - m * H;
    const Scalar* xh = x + m * xs_m + h * xs_h;
    const Scalar* zh = z + m * zs_m + h * zs_h;
    Scalar* yh = y + int64_t(g) * D;
    float ss = 0.f;
    for (int d = lid; d < D; d += kNormWg) {
      const float v = float(xh[d]);
      ss = sycl::fma(v, v, ss);
    }
    ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
    const float rstd = sycl::rsqrt(ss / float(D) + eps);
    for (int d = lid; d < D; d += kNormWg)
      yh[d] = Scalar(float(xh[d]) * rstd * float(wn[d]) * silu(float(zh[d])));
  }
};

template <typename Scalar>
struct ResAddNormKernel {
  const Scalar* x;
  const Scalar* r;
  const Scalar* wn;
  Scalar* y;      // [M, K] contiguous
  Scalar* r_out;  // [M, K] contiguous
  int64_t xs, rs;
  int K;
  float eps;

  void operator()(sycl::nd_item<1> it) const {
    const int m = it.get_group(0), lid = it.get_local_id(0);
    const Scalar* xm = x + m * xs;
    const Scalar* rm = r + m * rs;
    float ss = 0.f;
    for (int k = lid; k < K; k += kNormWg) {
      const float t = float(xm[k]) + float(rm[k]);
      r_out[int64_t(m) * K + k] = Scalar(t);
      ss = sycl::fma(t, t, ss);
    }
    ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
    const float rstd = sycl::rsqrt(ss / float(K) + eps);
    for (int k = lid; k < K; k += kNormWg) {
      const float t = float(xm[k]) + float(rm[k]);
      y[int64_t(m) * K + k] = Scalar(t * rstd * (float(wn[k]) + 1.f));
    }
  }
};

// fp16 [M, H, D] or [M, H*D] with contiguous last dim; returns (s_m, s_h).
static bool norm_heads_ok(
    const torch::Tensor& t,
    const torch::Tensor& ref,
    int64_t H,
    int64_t D,
    int64_t& s_m,
    int64_t& s_h) {
  if (!t.is_xpu() ||
      (t.scalar_type() != at::kHalf && t.scalar_type() != at::kBFloat16) ||
      (t.device() != ref.device() || t.scalar_type() != ref.scalar_type()))
    return false;
  if (t.dim() == 3 && t.size(1) == H && t.size(2) == D && t.stride(2) == 1) {
    s_m = t.stride(0);
    s_h = t.stride(1);
    return true;
  }
  if (t.dim() == 2 && t.size(1) == H * D && t.stride(1) == 1) {
    s_m = t.stride(0);
    s_h = D;
    return true;
  }
  return false;
}

// Returns y [M, H*D], or nullopt if the inputs need the generic ATen path.
template <typename Scalar>
static std::optional<torch::Tensor> gated_rmsnorm_kernel_impl(
    const torch::Tensor& x,
    const torch::Tensor& z,
    const torch::Tensor& wn,
    double eps) {
  if (wn.dim() != 1 || wn.scalar_type() != x.scalar_type() ||
      wn.stride(0) != 1 || wn.device() != x.device() ||
      (x.dim() != 2 && x.dim() != 3) || z.sizes() != x.sizes())
    return std::nullopt;
  const int64_t D = wn.size(0), M = x.size(0);
  if (D < 1 || (x.dim() == 2 && x.size(1) % D != 0)) return std::nullopt;
  const int64_t H = x.dim() == 3 ? x.size(1) : x.size(1) / D;
  int64_t xs_m, xs_h, zs_m, zs_h;
  if (H < 1 || !norm_heads_ok(x, x, H, D, xs_m, xs_h) ||
      !norm_heads_ok(z, x, H, D, zs_m, zs_h) || M * H >= (int64_t(1) << 31))
    return std::nullopt;
  const at::DeviceGuard guard(x.device());
  auto y = at::empty({M, H * D}, x.options());
  if (M == 0) return y;
  syclex::nd_launch(
      vllm::xpu::vllmGetQueue(),
      sycl::nd_range<1>(size_t(M * H) * kNormWg, kNormWg),
      GatedNormKernel<Scalar>{
          reinterpret_cast<const Scalar*>(x.data_ptr()),
          reinterpret_cast<const Scalar*>(z.data_ptr()),
          reinterpret_cast<const Scalar*>(wn.data_ptr()),
          reinterpret_cast<Scalar*>(y.data_ptr()),
          xs_m,
          xs_h,
          zs_m,
          zs_h,
          int(H),
          int(D),
          float(eps)});
  return y;
}

static std::optional<torch::Tensor> gated_rmsnorm_kernel(
    const torch::Tensor& x,
    const torch::Tensor& z,
    const torch::Tensor& wn,
    double eps) {
  if (x.scalar_type() == at::kHalf)
    return gated_rmsnorm_kernel_impl<sycl::half>(x, z, wn, eps);
  return gated_rmsnorm_kernel_impl<sycl::ext::oneapi::bfloat16>(x, z, wn, eps);
}

// Returns (y, residual_out) [M, K], or nullopt for the generic ATen path.
template <typename Scalar>
static std::optional<std::tuple<torch::Tensor, torch::Tensor>>
resadd_rmsnorm_kernel_impl(
    const torch::Tensor& x,
    const torch::Tensor& r,
    const torch::Tensor& wn,
    double eps) {
  if (!x.is_xpu() || x.dim() != 2 || r.sizes() != x.sizes() ||
      (x.scalar_type() != at::kHalf && x.scalar_type() != at::kBFloat16) ||
      r.scalar_type() != x.scalar_type() ||
      wn.scalar_type() != x.scalar_type() || wn.dim() != 1 ||
      wn.size(0) != x.size(1) || x.stride(1) != 1 || r.stride(1) != 1 ||
      wn.stride(0) != 1 || r.device() != x.device() ||
      wn.device() != x.device() || x.size(0) >= (int64_t(1) << 31))
    return std::nullopt;
  const int64_t M = x.size(0), K = x.size(1);
  const at::DeviceGuard guard(x.device());
  auto y = at::empty({M, K}, x.options());
  auto r_out = at::empty({M, K}, x.options());
  if (M > 0 && K > 0)
    syclex::nd_launch(
        vllm::xpu::vllmGetQueue(),
        sycl::nd_range<1>(size_t(M) * kNormWg, kNormWg),
        ResAddNormKernel<Scalar>{
            reinterpret_cast<const Scalar*>(x.data_ptr()),
            reinterpret_cast<const Scalar*>(r.data_ptr()),
            reinterpret_cast<const Scalar*>(wn.data_ptr()),
            reinterpret_cast<Scalar*>(y.data_ptr()),
            reinterpret_cast<Scalar*>(r_out.data_ptr()),
            x.stride(0),
            r.stride(0),
            int(K),
            float(eps)});
  return std::make_tuple(y, r_out);
}

static std::optional<std::tuple<torch::Tensor, torch::Tensor>>
resadd_rmsnorm_kernel(
    const torch::Tensor& x,
    const torch::Tensor& r,
    const torch::Tensor& wn,
    double eps) {
  if (x.scalar_type() == at::kHalf)
    return resadd_rmsnorm_kernel_impl<sycl::half>(x, r, wn, eps);
  return resadd_rmsnorm_kernel_impl<sycl::ext::oneapi::bfloat16>(x, r, wn, eps);
}

// ---------------------------------------------------------------------------
// Graph-level entries: fused kernel when supported (decode M == 1), otherwise
// the unfused norm (one SYCL kernel above, ATen for other dtypes / layouts)
// followed by fp8_gemm_w8a16. B*_kn are the [K, N] transposed views that
// fp8_gemm_w8a16 takes.
// ---------------------------------------------------------------------------
static torch::Tensor call_fp8_gemm_w8a16(
    const torch::Tensor& a, const torch::Tensor& b_kn, const torch::Tensor& s) {
  using Sig = torch::Tensor(
      const torch::Tensor&,
      const torch::Tensor&,
      const std::optional<torch::Tensor>&,
      const std::optional<torch::Tensor>&);
  static auto gemm = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("_xpu_C::fp8_gemm_w8a16", "")
                         .typed<Sig>();
  return gemm.call(a, b_kn, s, std::nullopt);
}

torch::Tensor gated_rmsnorm_fp8_gemm(
    const torch::Tensor& x,
    const torch::Tensor& z,
    const torch::Tensor& norm_weight,
    double eps,
    const torch::Tensor& b_kn,
    const torch::Tensor& scale) {
  if (b_kn.dim() == 2 &&
      gated_rmsnorm_fp8_gemv_supported(x, z, norm_weight, b_kn.t(), scale))
    return gated_rmsnorm_fp8_gemv(x, z, norm_weight, eps, b_kn.t(), scale);
  if (auto y = gated_rmsnorm_kernel(x, z, norm_weight, eps))
    return call_fp8_gemm_w8a16(*y, b_kn, scale);
  // RMSNormGated (norm_before_gate): per head of D = norm_weight.numel().
  const int64_t D = norm_weight.numel();
  auto xf = x.reshape({-1, D}).to(at::kFloat);
  auto zf = z.reshape({-1, D}).to(at::kFloat);
  auto y = xf * at::rsqrt(xf.pow(2).mean(-1, true) + eps) *
           norm_weight.to(at::kFloat) * at::silu(zf);
  auto rows = x.size(0);
  return call_fp8_gemm_w8a16(
      y.to(x.scalar_type()).reshape({rows, -1}), b_kn, scale);
}

static std::tuple<torch::Tensor, torch::Tensor> resadd_rmsnorm_unfused(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& norm_weight,
    double eps) {
  if (auto yr = resadd_rmsnorm_kernel(x, residual, norm_weight, eps))
    return *yr;
  auto t = x.to(at::kFloat) + residual.to(at::kFloat);
  auto y = t * at::rsqrt(t.pow(2).mean(-1, true) + eps) *
           (norm_weight.to(at::kFloat) + 1.0);
  return {y.to(x.scalar_type()), t.to(x.scalar_type())};
}

std::tuple<torch::Tensor, torch::Tensor> resadd_rmsnorm_fp8_gemm(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& norm_weight,
    double eps,
    const torch::Tensor& b_kn,
    const torch::Tensor& scale) {
  if (b_kn.dim() == 2 && resadd_rmsnorm_fp8_gemv_supported(
                             x, residual, norm_weight, b_kn.t(), scale))
    return resadd_rmsnorm_fp8_gemv(
        x, residual, norm_weight, eps, b_kn.t(), scale);
  auto [y, r_out] = resadd_rmsnorm_unfused(x, residual, norm_weight, eps);
  return {call_fp8_gemm_w8a16(y, b_kn, scale), r_out};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
resadd_rmsnorm_fp8_gemm_pair(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& norm_weight,
    double eps,
    const torch::Tensor& b1_kn,
    const torch::Tensor& scale1,
    const torch::Tensor& b2_kn,
    const torch::Tensor& scale2) {
  if (b1_kn.dim() == 2 && b2_kn.dim() == 2 &&
      resadd_rmsnorm_fp8_gemv_supported(
          x, residual, norm_weight, b1_kn.t(), scale1) &&
      resadd_rmsnorm_fp8_gemv_supported(
          x, residual, norm_weight, b2_kn.t(), scale2))
    return resadd_rmsnorm_fp8_gemv2(
        x, residual, norm_weight, eps, b1_kn.t(), scale1, b2_kn.t(), scale2);
  auto [y, r_out] = resadd_rmsnorm_unfused(x, residual, norm_weight, eps);
  return {
      call_fp8_gemm_w8a16(y, b1_kn, scale1),
      call_fp8_gemm_w8a16(y, b2_kn, scale2),
      r_out};
}

}  // namespace vllm::fp8_gemv
