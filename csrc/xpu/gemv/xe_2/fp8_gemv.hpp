// SPDX-License-Identifier: Apache-2.0
//
// Small-M (M = 1..8) FP8-E4M3 x FP16/BF16 GEMV for Xe2 (BMG), fp32
// accumulation:
//
//     out[m, n] = input_dtype( scale * sum_k a[m, k] * w[n, k] )
//
// w is row-major [N, K] FP8-E4M3 with ONE fp32 per-tensor scale, a is FP16 or
// BF16. The same core serves fused variants whose activation operand `a` is
// produced by an in-kernel prologue (RMSNorm / gated RMSNorm) into SLM.
//
// Design (plain SYCL + the SYCL-TLA/CuTe Xe DPAS and 2D-block-IO atoms):
//   * one sub-group (16 lanes) owns a 16-row block of W and a set of K
//     "chunks" (64 K each); KS sub-groups split K for the same 16 rows and
//     combine their partial sums through SLM at the end (no global scratch);
//     a work-group holds RG such row blocks (RG * KS sub-groups);
//   * W is read with 2D block TRANSPOSED loads (16 rows x 8 dwords per
//     message): lane j receives 32 contiguous bytes of row n0 + j, which is
//     exactly the DPAS B-operand (VNNI) register layout once the FP8 bytes
//     are widened to FP16/BF16 pairs -- one message per 512 B, no gathers;
//   * FP8 -> FP16/BF16 uses exact bit arithmetic at value * 2^-8, including
//     subnormals; the 2^8 is folded into the final scale;
//   * the byte pairing produced by that decode ({0,2} and {1,3} of each
//     word) is matched by splitting every 32-K step into an "even k" and an
//     "odd k" DPAS: the A operand (activations, M rows padded to MP in
//     {1,2,4,8} = DPAS repeat count) takes each low / high 16-bit element;
//     no shuffles are needed on either operand;
//   * FP16/BF16 DPAS with FP32 accumulation does the math, so the cost per
//     weight byte is almost independent of M.
#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <type_traits>

#include <cute/tensor.hpp>
#include <cute/arch/copy_xe_legacy.hpp>
#include <cute/arch/mma_xe_legacy.hpp>

#include <cstdint>

namespace vllm::fp8_gemv {

constexpr int kSg = 16;         // sub-group size
constexpr int kRowBlock = 16;   // W rows per sub-group (= DPAS N)
constexpr int kStepK = 32;      // K per step (one even + one odd DPAS)
constexpr int kChunkSteps = 2;  // steps per chunk
constexpr int kChunkK = kStepK * kChunkSteps;  // 64
constexpr int kMaxM = 8;
constexpr float kE4M3DecodeScale = 256.f;  // undo the 2^-8 of the bit decode

using u32x8 = sycl::vec<uint32_t, 8>;

// ---------------------------------------------------------------- DPAS -----
template <int MP, typename Scalar = sycl::half>
struct Dpas;
template <typename Scalar>
struct Dpas<8, Scalar> {
  using A = cute::intel::short8;
  using C = cute::intel::float8;
  static inline void mma(C& c, const A& a, const cute::intel::int8& b) {
    if constexpr (std::is_same_v<Scalar, sycl::half>)
      cute::XE_8x16x16_F32F16F16F32_TT::fma(c, a, b, c);
    else
      cute::XE_8x16x16_F32BF16BF16F32_TT::fma(c, a, b, c);
  }
};
template <typename Scalar>
struct Dpas<4, Scalar> {
  using A = cute::intel::short4;
  using C = cute::intel::float4;
  static inline void mma(C& c, const A& a, const cute::intel::int8& b) {
    if constexpr (std::is_same_v<Scalar, sycl::half>)
      cute::XE_4x16x16_F32F16F16F32_TT::fma(c, a, b, c);
    else
      cute::XE_4x16x16_F32BF16BF16F32_TT::fma(c, a, b, c);
  }
};
template <typename Scalar>
struct Dpas<2, Scalar> {
  using A = cute::intel::short2;
  using C = cute::intel::float2;
  static inline void mma(C& c, const A& a, const cute::intel::int8& b) {
    if constexpr (std::is_same_v<Scalar, sycl::half>)
      cute::XE_2x16x16_F32F16F16F32_TT::fma(c, a, b, c);
    else
      cute::XE_2x16x16_F32BF16BF16F32_TT::fma(c, a, b, c);
  }
};
template <typename Scalar>
struct Dpas<1, Scalar> {
  using A = short;
  using C = float;
  static inline void mma(C& c, const A& a, const cute::intel::int8& b) {
    if constexpr (std::is_same_v<Scalar, sycl::half>)
      cute::XE_1x16x16_F32F16F16F32_TT::fma(c, a, b, c);
    else
      cute::XE_1x16x16_F32BF16BF16F32_TT::fma(c, a, b, c);
  }
};

template <int MP, class V, class T>
static inline void vset(V& v, int i, T x) {
  if constexpr (MP == 1)
    v = x;
  else
    v[i] = x;
}
template <int MP, class V>
static inline float vget(const V& v, int i) {
  if constexpr (MP == 1)
    return v;
  else
    return v[i];
}

// FP16 bits -> BF16 bits with the hardware conversions (exact here: the
// FP16 values decoded from E4M3 have at most 4 significant bits).
static inline uint32_t h2_bf2(uint32_t h) {
  const float lo = float(sycl::bit_cast<sycl::half>(uint16_t(h & 0xffffu)));
  const float hi = float(sycl::bit_cast<sycl::half>(uint16_t(h >> 16)));
  return uint32_t(sycl::bit_cast<uint16_t>(sycl::ext::oneapi::bfloat16(lo))) |
         (uint32_t(sycl::bit_cast<uint16_t>(sycl::ext::oneapi::bfloat16(hi)))
          << 16);
}

// Four E4M3 bytes -> 16-bit pairs: even = {0, 2}, odd = {1, 3}, as FP16 (or
// BF16) values at value * 2^-8. The FP16 form is exact bit arithmetic,
// subnormals included; BF16 converts that FP16 form.
template <typename Scalar = sycl::half>
static inline uint32_t e4m3_even(uint32_t v) {
  const uint32_t h = ((v & 0x00800080u) << 8) | ((v & 0x007F007Fu) << 7);
  if constexpr (std::is_same_v<Scalar, sycl::half>)
    return h;
  else
    return h2_bf2(h);
}
template <typename Scalar = sycl::half>
static inline uint32_t e4m3_odd(uint32_t v) {
  const uint32_t h = (v & 0x80008000u) | ((v & 0x7F007F00u) >> 1);
  if constexpr (std::is_same_v<Scalar, sycl::half>)
    return h;
  else
    return h2_bf2(h);
}

// DPAS A operands of one chunk (even / odd k of each 32-K step).
template <int MP>
struct AOps {
  typename Dpas<MP>::A e[kChunkSteps];
  typename Dpas<MP>::A o[kChunkSteps];
};

template <int MP>
static inline void
a_from_dwords(const uint32_t (&ab)[kChunkSteps][MP], AOps<MP>& A) {
#pragma unroll
  for (int t = 0; t < kChunkSteps; ++t)
#pragma unroll
    for (int m = 0; m < MP; ++m) {
      vset<MP>(A.e[t], m, short(ab[t][m] & 0xFFFFu));
      vset<MP>(A.o[t], m, short(ab[t][m] >> 16));
    }
}

// ------------------------------------------------------------- loads -----
// W chunk: per step one transposed 2D block load of 16 rows x 8 dwords;
// lane j receives row row0 + j. Rows >= N (surface height) read as zero.
static inline void load_w_chunk(
    const uint8_t* w,
    int K,
    int N,
    int row0,
    int kc,
    u32x8 (&wr)[kChunkSteps]) {
#ifdef __SYCL_DEVICE_ONLY__
  #pragma unroll
  for (int t = 0; t < kChunkSteps; ++t) {
    cute::intel::coord_t co;
    co[0] = (kc + t * kStepK) / 4;  // in dwords
    co[1] = row0;
    cute::detail::XeSubgroup2DBlockLoadTranspose<4, 8, 16, 1>{}(
        w, K, N, K, co, &wr[t]);
  }
#endif
}

// Activation chunk from global: per step one 2D load of MP rows x 16
// dwords; lane j of row m = a[m, kc + 32 t + 2 j .. + 1]. Rows >= M read 0.
template <int MP, typename Scalar>
static inline void load_a_global(
    const Scalar* a, int K, int M, int pitch_bytes, int kc, AOps<MP>& A) {
#ifdef __SYCL_DEVICE_ONLY__
  uint32_t ab[kChunkSteps][MP];
  #pragma unroll
  for (int t = 0; t < kChunkSteps; ++t) {
    cute::intel::coord_t co;
    co[0] = (kc + t * kStepK) / 2;  // in dwords
    co[1] = 0;
    cute::detail::XeSubgroup2DBlockLoad<4, 16, MP, 1>{}(
        a, K * 2, M, pitch_bytes, co, &ab[t][0]);
  }
  a_from_dwords<MP>(ab, A);
#endif
}

// SLM activation layout ("A image"): dword index (t * MP + m) * 16 + j holds
// the fp16 pair a[m, 32 t + 2 j], a[m, 32 t + 2 j + 1] -- the same register
// image a 2D block load of MP rows x 16 dwords produces; rows m >= M are
// zero. Size: MP * K / 2 dwords. A sub-group block read of MP dwords per
// lane fetches one step.
template <int MP>
static inline int slm_a_dword(int m, int k) {
  return ((k / kStepK) * MP + m) * kSg + (k % kStepK) / 2;
}

template <int MP>
static inline void load_a_slm(
    const sycl::sub_group& sg, const uint32_t* slm_a, int kc, AOps<MP>& A) {
  uint32_t ab[kChunkSteps][MP];
#pragma unroll
  for (int t = 0; t < kChunkSteps; ++t) {
    const uint32_t* ptr = slm_a + ((kc / kStepK + t) * MP) * kSg;
    auto lp = sycl::address_space_cast<
        sycl::access::address_space::local_space,
        sycl::access::decorated::yes>(const_cast<uint32_t*>(ptr));
    if constexpr (MP == 1) {
      ab[t][0] = sg.load(lp);
    } else {
      const sycl::vec<uint32_t, MP> u = sg.load<MP>(lp);
#pragma unroll
      for (int m = 0; m < MP; ++m)
        ab[t][m] = u[m];
    }
  }
  a_from_dwords<MP>(ab, A);
}

// A-operand producers for gemv_body.
template <int MP, typename Scalar = sycl::half>
struct GlobalA {
  const Scalar* a;
  int K, M, pitch_bytes;
  inline void
  operator()(const sycl::sub_group&, int kc, int, AOps<MP>& A) const {
    load_a_global<MP>(a, K, M, pitch_bytes, kc, A);
  }
};
template <int MP>
struct SlmA {
  const uint32_t* slm_a;
  inline void
  operator()(const sycl::sub_group& sg, int kc, int, AOps<MP>& A) const {
    load_a_slm<MP>(sg, slm_a, kc, A);
  }
};

// ----------------------------------------------------------- compute -----
template <int MP, typename Scalar>
static inline void compute_chunk(
    const u32x8 (&wr)[kChunkSteps],
    const AOps<MP>& A,
    typename Dpas<MP>::C& acc_even,
    typename Dpas<MP>::C& acc_odd) {
#ifdef __SYCL_DEVICE_ONLY__
  #pragma unroll
  for (int t = 0; t < kChunkSteps; ++t) {
    cute::intel::int8 be, bo;
  #pragma unroll
    for (int i = 0; i < 8; ++i) {
      be[i] = int(e4m3_even<Scalar>(wr[t][i]));
      bo[i] = int(e4m3_odd<Scalar>(wr[t][i]));
    }
    Dpas<MP, Scalar>::mma(acc_even, A.e[t], be);
    Dpas<MP, Scalar>::mma(acc_odd, A.o[t], bo);
  }
#endif
}

// One weight matrix handled by a contiguous range of work-groups.
template <typename Scalar = sycl::half>
struct Seg {
  const uint8_t* w;    // [N, K] row-major E4M3
  const float* scale;  // [1]
  Scalar* out;         // [M, N], row stride out_stride (elements)
  int N;
  int out_stride;
  int wg_begin;  // first work-group of this segment
};

// K-chunk ownership: iteration i of K-split sub-group ks handles chunk
// ks + i * KS (interleaved: the KS sub-groups of a row block read adjacent
// 64 B pieces of the same rows at the same time; measured ~1 us faster on
// 4608x2048 than contiguous K ranges per sub-group).
template <int KS>
static inline int chunk_of(int ks, int i) {
  return ks + i * KS;
}

// ----------------------------------------------------- main GEMV body -----
// Called by every sub-group of a work-group of RG * KS sub-groups (sub-group
// id = rg * KS + ks); the work-group owns rows [wg_row0, wg_row0 + 16 RG) of
// `seg`, sub-group (rg, ks) owns rows wg_row0 + 16 rg + [0, 16) and chunks
// chunk_of<KS>(ks, i). `slm_part` must hold RG * KS * MP * 16 floats.
// `load_a(sg, kc, i, AOps&)` produces the DPAS A operands of chunk kc
// (= iteration i of this sub-group): global 2D loads, the SLM A image, or a
// normalization prologue.
// PRE > 0: the caller already issued the weight loads of the first PRE
// chunks into wpre (lets them overlap a prologue); see preload_w().
// CPS > 0: compile-time chunks per sub-group (K == CPS * KS * 64); the loop
// is fully unrolled so A producers may keep per-chunk data in registers.
template <
    int MP,
    int KS,
    int RG,
    class ALoad,
    int PRE = 1,
    int CPS = 0,
    typename Scalar = sycl::half>
static inline void gemv_body(
    const sycl::nd_item<1>& item,
    const Seg<Scalar>& seg,
    int wg_row0,
    const ALoad& load_a_fn,
    float* slm_part,
    int M,
    int K,
    const u32x8 (&wpre)[PRE][kChunkSteps],
    bool w_preloaded) {
  using D = Dpas<MP, Scalar>;
  auto sg = item.get_sub_group();
  const int lane = sg.get_local_linear_id();
  const int sgid = sg.get_group_linear_id();
  const int ks = sgid % KS, rg = sgid / KS;
  const int row0 = wg_row0 + rg * kRowBlock;
  const bool active = row0 < seg.N;

  typename D::C acc_e, acc_o;
#pragma unroll
  for (int i = 0; i < MP; ++i) {
    vset<MP>(acc_e, i, 0.f);
    vset<MP>(acc_o, i, 0.f);
  }
  auto step = [&](int it, int c, const u32x8(&wr)[kChunkSteps]) {
    AOps<MP> A;
    load_a_fn(sg, c * kChunkK, it, A);
    compute_chunk<MP, Scalar>(wr, A, acc_e, acc_o);
  };
  if (active) {
    if constexpr (CPS > 0) {
#pragma unroll
      for (int it = 0; it < CPS; ++it) {
        const int c = chunk_of<KS>(ks, it);
        if (it < PRE && w_preloaded) {
          step(it, c, wpre[it < PRE ? it : 0]);
        } else {
          u32x8 wr[kChunkSteps];
          load_w_chunk(seg.w, K, seg.N, row0, c * kChunkK, wr);
          step(it, c, wr);
        }
      }
    } else {
      const int nchunks = K / kChunkK;
      int it = 0;
      if (w_preloaded) {
#pragma unroll
        for (int p = 0; p < PRE; ++p) {
          const int c = chunk_of<KS>(ks, it);
          if (c < nchunks) {
            step(it, c, wpre[p]);
            ++it;
          }
        }
      }
      for (;; ++it) {
        const int c = chunk_of<KS>(ks, it);
        if (c >= nchunks) break;
        u32x8 wr[kChunkSteps];
        load_w_chunk(seg.w, K, seg.N, row0, c * kChunkK, wr);
        step(it, c, wr);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < MP; ++i)
    vset<MP>(acc_e, i, vget<MP>(acc_e, i) + vget<MP>(acc_o, i));

  const float s = seg.scale[0] * kE4M3DecodeScale;
  if constexpr (KS == 1 && RG == 1) {
    const int n = row0 + lane;
    if (active && n < seg.N) {
#pragma unroll
      for (int i = 0; i < MP; ++i)
        if (i < M)
          seg.out[int64_t(i) * seg.out_stride + n] =
              Scalar(vget<MP>(acc_e, i) * s);
    }
  } else {
    // partials: [rg][ks][i][lane]
#pragma unroll
    for (int i = 0; i < MP; ++i)
      slm_part[((rg * KS + ks) * MP + i) * kSg + lane] = vget<MP>(acc_e, i);
    sycl::group_barrier(item.get_group());
    if constexpr (RG == 1) {
      // sub-group ks reduces tokens ks, ks + KS, ... of the 16 rows.
      const int n = row0 + lane;
      if (active && n < seg.N) {
        for (int i = ks; i < M; i += KS) {
          float v = 0.f;
#pragma unroll
          for (int q = 0; q < KS; ++q)
            v += slm_part[(q * MP + i) * kSg + lane];
          seg.out[int64_t(i) * seg.out_stride + n] = Scalar(v * s);
        }
      }
    } else {
      // Packed epilogue: sub-group g writes token rows i = g, g + RG*KS, ...
      // lane j owns rows wg_row0 + RG j .. + RG - 1 (one RG*2-byte store),
      // so every store instruction writes RG * 32 contiguous bytes.
      const int r_first = lane * RG;
      for (int i = sgid; i < M; i += RG * KS) {
        float v[RG];
#pragma unroll
        for (int u = 0; u < RG; ++u) {
          const int r = r_first + u, rgi = r / kRowBlock, l = r % kRowBlock;
          float acc = 0.f;
#pragma unroll
          for (int q = 0; q < KS; ++q)
            acc += slm_part[((rgi * KS + q) * MP + i) * kSg + l];
          v[u] = acc * s;
        }
        const int n0 = wg_row0 + r_first;
        Scalar* o = seg.out + int64_t(i) * seg.out_stride + n0;
        if (n0 + RG <= seg.N && (seg.out_stride % RG) == 0) {
          sycl::vec<uint16_t, RG> hv;
#pragma unroll
          for (int u = 0; u < RG; ++u)
            hv[u] = sycl::bit_cast<uint16_t>(Scalar(v[u]));
          *reinterpret_cast<sycl::vec<uint16_t, RG>*>(o) = hv;
        } else {
#pragma unroll
          for (int u = 0; u < RG; ++u)
            if (n0 + u < seg.N) o[u] = Scalar(v[u]);
        }
      }
    }
  }
}

// Issue the weight loads of this sub-group's first PRE chunks for
// gemv_body(..., wpre, true); returns false if the rows are out of range.
template <int PRE, int KS, typename Scalar>
static inline bool preload_w(
    const Seg<Scalar>& seg,
    int row0,
    int ks,
    int K,
    u32x8 (&wpre)[PRE][kChunkSteps]) {
  if (row0 >= seg.N) return false;
  const int nchunks = K / kChunkK;
#pragma unroll
  for (int p = 0; p < PRE; ++p) {
    const int c = chunk_of<KS>(ks, p);
    if (c < nchunks) load_w_chunk(seg.w, K, seg.N, row0, c * kChunkK, wpre[p]);
  }
  return true;
}

// Stage a[M, K] (fp16, row pitch in elements) into the SLM A image; rows
// M..MP-1 are zero-filled. Work-group cooperative; caller barriers.
// Each work-item moves 4 consecutive dwords (8 fp16, 16 B) per iteration.
template <int MP, typename Scalar>
static inline void stage_a_slm(
    const sycl::nd_item<1>& item,
    const Scalar* a,
    int64_t pitch,
    int M,
    int K,
    uint32_t* slm_a) {
  const int lid = item.get_local_linear_id();
  const int nthr = item.get_local_range(0);
  const int K8 = K / 8;  // 16-byte vectors per row
  for (int idx = lid; idx < MP * K8; idx += nthr) {
    const int m = idx / K8, k = (idx - m * K8) * 8;
    sycl::vec<uint32_t, 4> v(0u);
    if (m < M)
      v = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(a + m * pitch + k);
    *reinterpret_cast<sycl::vec<uint32_t, 4>*>(slm_a + slm_a_dword<MP>(m, k)) =
        v;
  }
}

}  // namespace vllm::fp8_gemv
