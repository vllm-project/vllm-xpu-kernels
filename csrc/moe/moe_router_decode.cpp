#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>

#include <ATen/DeviceGuard.h>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "../utils.h"
#include "moe_ops.h"

namespace vllm {
namespace moe {

// MoE router for small-M decode (M <= 16 tokens), ONE kernel launch:
//   [norm variant]  s = x + residual; residual_out = dt(s);
//                   normed_out = dt(s * rsqrt(mean(s^2) + eps) * (1 + g))
//   (dt = the activation dtype, fp16 or bf16)
//   logits = dt(a @ W^T)   (a and W of the same dtype, fp32 accumulate)
//   softmax over E, top-k, optional renormalization (== topk_softmax).
//
// GEMV: a work-group (32 sub-groups) owns kExpPerWg = 8 experts and up to
//   kMaxRows = 4 token rows (M > 4: multiple row groups of work-groups). A
//   sub-group computes kR = 4 experts x one 128-wide K block; kKSplit = 16
//   sub-groups cover K = 2048. All accesses to 128-element blocks are
//   sub-group block loads/stores (striped: lane l owns elements l + 16 j).
//   The W loads are issued first; x (or x / residual) is loaded once per
//   block and reused for the 4 experts. Partials: sub-group reduce, packed
//   into lanes, SLM, summed by one work-item per (expert, row).
//   E = 256: 32 work-groups (one per Xe core) per row group.
// Top-k: the last work-group to finish (device-scope ticket that resets
//   itself) reads the M x E logits back, one sub-group per token. Keys are
//   order-preserving FP16/BF16 logit bits << 16 | (0xffff - expert) (ties ->
//   lowest id). top_k <= 8: each lane sorts its keys into a top-8 list and 4
//   xor-shuffle bitonic merges leave the sub-group's top-8 in every lane;
//   top_k <= 16: repeated sub-group arg-max. Softmax only for the picks
//   (plus the full denominator if !renormalize).
// Norm variant: the sub-group owning K block b loads x / residual / norm_w
//   of block b together with W; the kKSplit partial sums of squares of a row
//   meet in SLM (one barrier), then it normalizes in registers and runs the
//   GEMV (for K <= 2048 nothing is read twice). With >= 3 rows per
//   work-group the two expert groups split the rows and exchange the
//   normed rows through SLM (second barrier, half the x / residual traffic).
//   Outputs are in-place safe (normed_out may alias x, residual_out may
//   alias residual) without any spin-wait: a work-group only reads the rows
//   of its row group and bumps that row group's `reads_done` counter once
//   all its reads have completed; the work-group completing the count writes
//   the row group's normed_out (from SLM) / residual_out, after its ticket
//   so the stores stay off the critical path.
// Constraints: M <= 16, E <= 512, top_k <= 16, K % 128 == 0 (norm variant:
//   K <= 6144), 16-byte aligned rows. Launches on one queue must not overlap
//   (in-order queue; the counters are per queue).
constexpr int kRouterSg = 16;
constexpr int kSgPerWg = 32;  // sub-groups per work-group
constexpr int kKSplit = 16;   // sub-groups sharing an expert (K blocks)
constexpr int kR = 4;         // experts per sub-group
constexpr int kEGroups = kSgPerWg / kKSplit;
constexpr int kExpPerWg = kR * kEGroups;
constexpr int kWgSize = kRouterSg * kSgPerWg;
constexpr int kMaxRows = 4;  // token rows per work-group
constexpr int kRouterMaxM = 16;
constexpr int kRouterMaxEpl = 32;      // up to 512 experts
constexpr int kBlk = kRouterSg * 8;    // K elements per sub-group block
constexpr int kRouterNormMaxK = 6144;  // norm variant: MR x K fp16 in SLM
static_assert(kSgPerWg % kKSplit == 0, "");
static_assert(kSgPerWg >= kRouterMaxM, "top-k: one sub-group per token");
static_assert(kSgPerWg >= kMaxRows, "norm: one sub-group per row");
static_assert(kExpPerWg * kMaxRows <= kWgSize, "");

// 8 fp16 values per lane. (A plain array rather than sycl::vec: writes to
// sycl::vec elements were kept in private memory by the compiler.)
template <typename Scalar>
struct alignas(16) Scalar8 {
  Scalar h[8];
  Scalar& operator[](int j) { return h[j]; }
  const Scalar& operator[](int j) const { return h[j]; }
};

namespace syclex = sycl::ext::oneapi::experimental;

// Sub-group block load / store of 128 contiguous matching fp16/bf16 (16-byte
// aligned).
template <typename Scalar>
static inline Scalar8<Scalar>
sg_load8(const sycl::sub_group& sg, const Scalar* p) {
  Scalar8<Scalar> v;
  syclex::group_load(
      sg,
      sycl::address_space_cast<
          sycl::access::address_space::global_space,
          sycl::access::decorated::yes>(p)
          .get_decorated(),
      sycl::span<Scalar, 8>(v.h),
      syclex::properties{
          syclex::data_placement_striped,
          syclex::full_group,
          syclex::alignment<16>});
  return v;
}

template <typename Scalar>
static inline void
sg_store8(const sycl::sub_group& sg, Scalar* p, const Scalar8<Scalar>& v) {
  syclex::group_store(
      sg,
      sycl::span<const Scalar, 8>(v.h),
      sycl::address_space_cast<
          sycl::access::address_space::global_space,
          sycl::access::decorated::yes>(p)
          .get_decorated(),
      syclex::properties{
          syclex::data_placement_striped,
          syclex::full_group,
          syclex::alignment<16>});
}

// Per-queue device scratch: self-resetting counters (one cache line each).
constexpr int kMaxRowGroups = (kRouterMaxM + kMaxRows - 1) / kMaxRows;
struct RouterSync {
  unsigned int ticket;  // work-groups done
  unsigned int pad0[15];
  struct {
    unsigned int n;  // norm: work-groups of a row group done reading x / r
    unsigned int pad[15];
  } reads_done[kMaxRowGroups];
};

template <typename Scalar>
struct RouterArgs {
  const Scalar* x;
  int64_t x_stride;
  const Scalar* w;
  float* logits;
  float* topk_weights;
  int32_t* topk_ids;
  RouterSync* sync;
  int M, E, K, top_k;
  int nwg_e;  // work-groups per row group
  bool renormalize;
  // norm variant only
  const Scalar* residual;
  int64_t res_stride;
  const Scalar* norm_w;
  Scalar* normed_out;
  int64_t normed_stride;
  Scalar* residual_out;
  int64_t res_out_stride;
  float eps;
};

template <typename T>
using router_atomic = sycl::atomic_ref<
    T,
    sycl::memory_order::acq_rel,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>;

template <typename Scalar>
static inline uint32_t router_key(float logit, int e) {
  const uint16_t h = sycl::bit_cast<uint16_t>(static_cast<Scalar>(logit));
  const uint16_t o = (h & 0x8000) ? static_cast<uint16_t>(~h)
                                  : static_cast<uint16_t>(h | 0x8000);
  return (static_cast<uint32_t>(o) << 16) | static_cast<uint32_t>(0xffff - e);
}

template <typename Scalar>
static inline float router_key_logit(uint32_t key) {
  const uint16_t o = static_cast<uint16_t>(key >> 16);
  const uint16_t h = (o & 0x8000) ? static_cast<uint16_t>(o & 0x7fff)
                                  : static_cast<uint16_t>(~o);
  return static_cast<float>(sycl::bit_cast<Scalar>(h));
}

template <typename Scalar>
static inline float
router_dot8(const Scalar8<Scalar>& u, const Scalar8<Scalar>& v, float acc) {
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    acc = sycl::fma(static_cast<float>(u[j]), static_cast<float>(v[j]), acc);
  }
  return acc;
}

template <typename Scalar>
static inline float
router_sumsq8(const Scalar8<Scalar>& xv, const Scalar8<Scalar>& rv, float acc) {
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float s = static_cast<float>(xv[j]) + static_cast<float>(rv[j]);
    acc = sycl::fma(s, s, acc);
  }
  return acc;
}

// normed = input_dtype(((x + r) * rs) * (1 + g)), FP32 math
template <typename Scalar>
static inline Scalar8<Scalar> router_norm8(
    const Scalar8<Scalar>& xv,
    const Scalar8<Scalar>& rv,
    const Scalar8<Scalar>& gv,
    float rs) {
  Scalar8<Scalar> o;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float s = static_cast<float>(xv[j]) + static_cast<float>(rv[j]);
    o[j] = static_cast<Scalar>((s * rs) * (static_cast<float>(gv[j]) + 1.0f));
  }
  return o;
}

template <typename Scalar>
static inline Scalar8<Scalar>
router_add8(const Scalar8<Scalar>& xv, const Scalar8<Scalar>& rv) {
  Scalar8<Scalar> o;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    o[j] = static_cast<Scalar>(
        static_cast<float>(xv[j]) + static_cast<float>(rv[j]));
  }
  return o;
}

// ---- top-8 selection network on 32-bit keys (descending) ----
static inline void router_ce(uint32_t& a, uint32_t& b) {  // a >= b after
  const uint32_t hi = sycl::max(a, b), lo = sycl::min(a, b);
  a = hi;
  b = lo;
}

// Sorts a bitonic sequence of 8 keys into descending order.
static inline void router_bitonic_clean8(uint32_t (&v)[8]) {
#pragma unroll
  for (int d = 4; d >= 1; d /= 2) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if ((i & d) == 0) router_ce(v[i], v[i + d]);
    }
  }
}

// Sorts 8 keys descending (bitonic sort network, 6 layers).
static inline void router_sort8(uint32_t (&v)[8]) {
#pragma unroll
  for (int size = 2; size <= 8; size *= 2) {
#pragma unroll
    for (int d = size / 2; d >= 1; d /= 2) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int j = i ^ d;
        if (j > i) {
          if ((i & size) == 0) {
            router_ce(v[i], v[j]);
          } else {
            router_ce(v[j], v[i]);
          }
        }
      }
    }
  }
}

// a <- top-8 of (a, b), both sorted descending.
static inline void router_merge8(uint32_t (&a)[8], const uint32_t (&b)[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i)
    a[i] = sycl::max(a[i], b[7 - i]);
  router_bitonic_clean8(a);
}

// SLM (floats): [partials: kEGroups x kKSplit x kPartStride]
//               [sum of squares: kKSplit x MR] [flags]
//               [norm: normed rows, MR x K fp16]
template <int MR>
struct RouterSlm {
  static constexpr int kN = kR * MR;  // (expert, row) values per sub-group
  static constexpr int kChunks = (kN + kRouterSg - 1) / kRouterSg;
  static constexpr int kPartStride = kChunks * kRouterSg;
  static constexpr int kPart = kEGroups * kKSplit * kPartStride;
  static constexpr int kSsq = kKSplit * MR;
  static constexpr int kFloats = (kPart + kSsq + 4 + 3) / 4 * 4;  // 16B units
  static int floats(bool norm, int K) {
    return kFloats + (norm ? MR * K / 2 : 0);
  }
};

// MR: rows per work-group (compile time); this work-group's rows are
// [row0, row0 + mloc), mloc <= MR. EPL: logits per lane in the top-k.
template <int MR, int EPL, bool NORM, typename Scalar>
class router_kernel {
  // norm variant: split the rows between the expert groups
  static constexpr bool kSplitRows = MR >= 3;

 public:
  router_kernel(RouterArgs<Scalar> a, sycl::local_accessor<float, 1> slm)
      : a(a), slm(slm) {}

  [[sycl::reqd_sub_group_size(kRouterSg)]] void
  operator()(sycl::nd_item<1> item) const {
    using Slm = RouterSlm<MR>;
    auto sg = item.get_sub_group();
    auto wg = item.get_group();
    const int lane = sg.get_local_linear_id();
    const int sgid = sg.get_group_linear_id();
    const int tid = item.get_local_linear_id();
    const int gid = item.get_group(0);
    const int nwg = item.get_group_range(0);
    const int gid_e = gid % a.nwg_e;  // expert slice
    const int rg = gid / a.nwg_e;     // row group
    const int row0 = rg * MR;
    const int mloc = sycl::min(MR, a.M - row0);
    const int E = a.E, K = a.K;
    const int nblk = K / kBlk;
    float* s_part = &slm[0];
    float* s_ssq = s_part + Slm::kPart;
    // norm: normed rows [MR][nblk][16 lanes][8], after the fixed part
    [[maybe_unused]] Scalar* s_nrm =
        reinterpret_cast<Scalar*>(s_part + Slm::kFloats);
    int* s_flag = reinterpret_cast<int*>(s_ssq + Slm::kSsq);

    // ---- W loads for the first K block go out before anything else ----
    const int eg = sgid / kKSplit;
    const int ks = sgid % kKSplit;
    const int e0 = gid_e * kExpPerWg + eg * kR;
    const Scalar* wr[kR];
#pragma unroll
    for (int r = 0; r < kR; ++r) {
      const int e = e0 + r < E ? e0 + r : E - 1;  // clamp; result dropped
      wr[r] = a.w + static_cast<int64_t>(e) * K;
    }
    Scalar8<Scalar> wv[kR];
    auto load_w = [&](int b) {
      // (index clamp instead of an aggregate ?:, which goes to memory)
      const int bb = b < nblk ? b : 0;
#pragma unroll
      for (int r = 0; r < kR; ++r)
        wv[r] = sg_load8(sg, wr[r] + bb * kBlk);
    };
    load_w(ks);

    // Row m of this work-group (rows past mloc re-read row0; dropped).
    auto row = [&](int m) { return row0 + (m < mloc ? m : 0); };
    auto load_rows = [&](const Scalar* base,
                         int64_t stride,
                         int k,
                         Scalar8<Scalar>(&out)[MR]) {
#pragma unroll
      for (int m = 0; m < MR; ++m) {
        out[m] = sg_load8(sg, base + row(m) * stride + k);
      }
    };

    float acc[kR][MR];
#pragma unroll
    for (int r = 0; r < kR; ++r) {
#pragma unroll
      for (int m = 0; m < MR; ++m)
        acc[r][m] = 0.f;
    }
    auto gemv_block = [&](const Scalar8<Scalar>(&xb)[MR]) {
#pragma unroll
      for (int m = 0; m < MR; ++m) {
#pragma unroll
        for (int r = 0; r < kR; ++r) {
          acc[r][m] = router_dot8(wv[r], xb[m], acc[r][m]);
        }
      }
    };

    [[maybe_unused]] bool last_reader = false;  // (tid 0 only)
    if constexpr (NORM) {
      // x / residual / norm_w of this sub-group's first K block (the only
      // one if K <= 2048) are loaded together with W and stay in registers.
      // MR >= 3: rows are split between the expert groups (group e owns rows
      // e, e + kEGroups, ...): each group reads and normalizes only its own
      // rows and passes them to the other group through SLM (one more
      // barrier, half the x / residual traffic). MR <= 2: every group
      // normalizes all rows itself. Group 0 (or the owner) also keeps the
      // normed rows in SLM for the output writer.
      constexpr bool kSplit = kSplitRows;
      constexpr int kOwnRows = kSplit ? (MR + kEGroups - 1) / kEGroups : MR;
      auto own = [&](int i) { return kSplit ? eg + i * kEGroups : i; };
      const bool has0 = ks < nblk;
      const int k0 = (has0 ? ks : 0) * kBlk;
      Scalar8<Scalar> xv0[kOwnRows], rv0[kOwnRows];
#pragma unroll
      for (int i = 0; i < kOwnRows; ++i) {
        if (own(i) < MR) {  // (uniform per sub-group)
          xv0[i] = sg_load8(sg, a.x + row(own(i)) * a.x_stride + k0);
          rv0[i] = sg_load8(sg, a.residual + row(own(i)) * a.res_stride + k0);
        }
      }
      const Scalar8<Scalar> gv0 = sg_load8(sg, a.norm_w + k0);
      // Pass 1: this sub-group's share of sum((x + r)^2) of its rows.
      float ss[kOwnRows];
#pragma unroll
      for (int i = 0; i < kOwnRows; ++i) {
        ss[i] = own(i) < MR && has0 ? router_sumsq8(xv0[i], rv0[i], 0.f) : 0.f;
      }
      for (int b = ks + kKSplit; b < nblk; b += kKSplit) {  // K > 2048
#pragma unroll
        for (int i = 0; i < kOwnRows; ++i) {
          if (own(i) < MR) {
            const int k = b * kBlk;
            const Scalar8<Scalar> xv =
                sg_load8(sg, a.x + row(own(i)) * a.x_stride + k);
            const Scalar8<Scalar> rv =
                sg_load8(sg, a.residual + row(own(i)) * a.res_stride + k);
            ss[i] = router_sumsq8(xv, rv, ss[i]);
          }
        }
      }
      // s_ssq[ks][m]: the kKSplit sub-groups of a group cover all of K.
      float packed_ss = 0.f;
#pragma unroll
      for (int i = 0; i < kOwnRows; ++i) {
        const float t = sycl::reduce_over_group(sg, ss[i], sycl::plus<float>());
        packed_ss = lane == i ? t : packed_ss;
      }
      if ((kSplit || eg == 0) && lane < kOwnRows && own(lane) < MR) {
        s_ssq[ks * MR + own(lane)] = packed_ss;
      }
      sycl::group_barrier(wg);
      if (!kSplit && nblk <= kKSplit && tid == 0) {
        // Every x / residual read of this work-group has completed (the
        // values are consumed). The result is only needed after the GEMV.
        last_reader =
            router_atomic<unsigned int>(a.sync->reads_done[rg].n)
                .fetch_add(1u) == static_cast<unsigned int>(a.nwg_e - 1);
      }
      Scalar8<Scalar>* s_nv =
          reinterpret_cast<Scalar8<Scalar>*>(s_nrm);  // [MR][nblk][16]
      const bool keeper = kSplit || eg == 0;
      float rs[kOwnRows];
#pragma unroll
      for (int i = 0; i < kOwnRows; ++i) {
        const int m = own(i) < MR ? own(i) : 0;
        const float t = lane < kKSplit ? s_ssq[lane * MR + m] : 0.f;
        rs[i] = sycl::rsqrt(
            sycl::reduce_over_group(sg, t, sycl::plus<float>()) /
                static_cast<float>(K) +
            a.eps);
      }
      // Pass 2: normalize the own rows (MR <= 2: and run the GEMV on them).
      for (int b = ks; b < nblk; b += kKSplit) {
        const int k = b * kBlk;
        Scalar8<Scalar> nv[kOwnRows];
        if (b == ks) {
#pragma unroll
          for (int i = 0; i < kOwnRows; ++i) {
            nv[i] = router_norm8(xv0[i], rv0[i], gv0, rs[i]);
          }
        } else {  // K > 2048
          load_w(b);
          const Scalar8<Scalar> gv = sg_load8(sg, a.norm_w + k);
#pragma unroll
          for (int i = 0; i < kOwnRows; ++i) {
            if (own(i) < MR) {
              const Scalar8<Scalar> xv =
                  sg_load8(sg, a.x + row(own(i)) * a.x_stride + k);
              const Scalar8<Scalar> rv =
                  sg_load8(sg, a.residual + row(own(i)) * a.res_stride + k);
              nv[i] = router_norm8(xv, rv, gv, rs[i]);
            }
          }
        }
#pragma unroll
        for (int i = 0; i < kOwnRows; ++i) {
          const int m = own(i);
          if (m < MR) {
            if (keeper) s_nv[(m * nblk + b) * kRouterSg + lane] = nv[i];
            if constexpr (!kSplit) {
#pragma unroll
              for (int r = 0; r < kR; ++r) {
                acc[r][m] = router_dot8(wv[r], nv[i], acc[r][m]);
              }
            }
          }
        }
      }
      if constexpr (kSplit) {
        sycl::group_barrier(wg);
        if (tid == 0) {  // all x / residual reads are done now
          last_reader =
              router_atomic<unsigned int>(a.sync->reads_done[rg].n)
                  .fetch_add(1u) == static_cast<unsigned int>(a.nwg_e - 1);
        }
        // GEMV on all rows from SLM (W of block ks is still in registers
        // unless K > 2048)
        for (int b = ks; b < nblk; b += kKSplit) {
          if (nblk > kKSplit) load_w(b);
          Scalar8<Scalar> nv[MR];
#pragma unroll
          for (int m = 0; m < MR; ++m) {
            nv[m] = s_nv[((row(m) - row0) * nblk + b) * kRouterSg + lane];
          }
          gemv_block(nv);
        }
      }
    } else {
      for (int b = ks; b < nblk; b += kKSplit) {
        if (b != ks) load_w(b);
        Scalar8<Scalar> xv[MR];
        load_rows(a.x, a.x_stride, b * kBlk, xv);
        gemv_block(xv);
      }
    }

    // ---- partials: sub-group reduce; value i = r * MR + m goes to lane
    // i % 16 of chunk i / 16 (one SLM store per chunk) ----
    float packed[Slm::kChunks];
#pragma unroll
    for (int c = 0; c < Slm::kChunks; ++c)
      packed[c] = 0.f;
#pragma unroll
    for (int r = 0; r < kR; ++r) {
#pragma unroll
      for (int m = 0; m < MR; ++m) {
        const int i = r * MR + m;
        const float t =
            sycl::reduce_over_group(sg, acc[r][m], sycl::plus<float>());
        packed[i / kRouterSg] =
            lane == i % kRouterSg ? t : packed[i / kRouterSg];
      }
    }
#pragma unroll
    for (int c = 0; c < Slm::kChunks; ++c) {
      if (c * kRouterSg + lane < Slm::kN) {
        s_part[(eg * kKSplit + ks) * Slm::kPartStride + c * kRouterSg + lane] =
            packed[c];
      }
    }
    if constexpr (NORM) {
      if (tid == 0) {
        if (!kSplitRows && nblk > kKSplit) {  // K > 2048: pass 2 read x / r
          last_reader =
              router_atomic<unsigned int>(a.sync->reads_done[rg].n)
                  .fetch_add(1u) == static_cast<unsigned int>(a.nwg_e - 1);
        }
        s_flag[0] = last_reader;
      }
    }
    sycl::group_barrier(wg);
    if (tid < kExpPerWg * MR) {
      const int egl = tid / Slm::kN, i = tid % Slm::kN;
      const int eo = gid_e * kExpPerWg + egl * kR + i / MR;
      const int m = i % MR;
      if (eo < E && m < mloc) {
        float t = 0.f;
#pragma unroll
        for (int q = 0; q < kKSplit; ++q) {
          t += s_part[(egl * kKSplit + q) * Slm::kPartStride + i];
        }
        a.logits[(row0 + m) * E + eo] =
            static_cast<float>(static_cast<Scalar>(t));
      }
    }

    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
    sycl::group_barrier(wg);
    if (tid == 0) {
      router_atomic<unsigned int> ticket(a.sync->ticket);
      const bool last =
          ticket.fetch_add(1u) == static_cast<unsigned int>(nwg - 1);
      s_flag[1] = last;
      if (last) {
        // Every work-group has passed both counters: reset them. (One
        // work-item only: atomics on one address serialize.)
        ticket.store(0u, sycl::memory_order::relaxed);
        if constexpr (NORM) {
          for (int i = 0; i < kMaxRowGroups; ++i) {
            router_atomic<unsigned int>(a.sync->reads_done[i].n)
                .store(0u, sycl::memory_order::relaxed);
          }
        }
      }
    }
    sycl::group_barrier(wg);
    const int M = a.M;

    // ---- norm: the work-group that completed its row group's reads_done
    // count writes the row group's outputs (nobody reads these rows any
    // more). After the ticket: nothing in this kernel reads the outputs, so
    // the ticket's release fence need not wait for these stores. In the
    // last work-group, sub-groups < M run the top-k meanwhile.
    if constexpr (NORM) {
      const int first = s_flag[1] ? M : 0;
      if (s_flag[0] && sgid >= first) {
        for (int p = sgid - first; p < nblk * mloc; p += kSgPerWg - first) {
          const int b = p % nblk, m = p / nblk, k = b * kBlk;
          const Scalar8<Scalar> nv = reinterpret_cast<const Scalar8<Scalar>*>(
              s_nrm)[(m * nblk + b) * kRouterSg + lane];
          const Scalar8<Scalar> xv =
              sg_load8(sg, a.x + (row0 + m) * a.x_stride + k);
          const Scalar8<Scalar> rv =
              sg_load8(sg, a.residual + (row0 + m) * a.res_stride + k);
          sg_store8(sg, a.normed_out + (row0 + m) * a.normed_stride + k, nv);
          sg_store8(
              sg,
              a.residual_out + (row0 + m) * a.res_out_stride + k,
              router_add8(xv, rv));
        }
      }
    }
    if constexpr (NORM) {
      // (sub-groups with output stores in flight must not run the fence)
      if (!s_flag[1] || sgid >= M) return;
      sycl::atomic_fence(
          sycl::memory_order::acquire, sycl::memory_scope::device);
    } else {
      if (!s_flag[1]) return;
      sycl::atomic_fence(
          sycl::memory_order::acquire, sycl::memory_scope::device);
      if (sgid >= M) return;
    }

    // ---- the last work-group does the top-k ----

    const int t = sgid;
    const float* l = a.logits + static_cast<int64_t>(t) * E;
    // All loads first (clamped index, no branches), then the keys.
    float v[EPL];
#pragma unroll
    for (int i = 0; i < EPL; ++i) {
      const int ee = lane + i * kRouterSg;
      v[i] = l[ee < E ? ee : E - 1];
    }
    uint32_t key[EPL];
#pragma unroll
    for (int i = 0; i < EPL; ++i) {
      const int ee = lane + i * kRouterSg;
      key[i] = ee < E ? router_key<Scalar>(v[i], ee) : 0u;
    }
    uint32_t sel = 0;  // lane j ends up with pick j
    if (a.top_k <= 8) {
      // Per lane: top-8 of its EPL keys; then 4 xor-shuffle merge levels
      // leave the sub-group's top-8 (sorted) in every lane.
      uint32_t best[8];
#pragma unroll
      for (int i = 0; i < 8; ++i)
        best[i] = key[i];
      router_sort8(best);
#pragma unroll
      for (int g0 = 8; g0 < EPL; g0 += 8) {
        uint32_t nxt[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
          nxt[i] = key[g0 + i];
        router_sort8(nxt);
        router_merge8(best, nxt);
      }
#pragma unroll
      for (int d = 1; d < kRouterSg; d *= 2) {
        uint32_t other[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          other[i] = sycl::permute_group_by_xor(sg, best[i], d);
        }
        router_merge8(best, other);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i)
        sel = lane == i ? best[i] : sel;
    } else {
      // Repeated arg-max: tree max over the lane's slots, one sub-group
      // max, then the owning slot is cleared.
      for (int j = 0; j < a.top_k; ++j) {
        uint32_t tr[EPL];
#pragma unroll
        for (int i = 0; i < EPL; ++i)
          tr[i] = key[i];
#pragma unroll
        for (int w = EPL / 2; w >= 1; w /= 2) {
#pragma unroll
          for (int i = 0; i < w; ++i)
            tr[i] = sycl::max(tr[i], tr[i + w]);
        }
        const uint32_t g =
            sycl::reduce_over_group(sg, tr[0], sycl::maximum<uint32_t>());
#pragma unroll
        for (int i = 0; i < EPL; ++i)
          key[i] = key[i] == g ? 0u : key[i];
        sel = lane == j ? g : sel;
      }
    }
    // Lane 0 holds the largest logit.
    const float mx =
        router_key_logit<Scalar>(sycl::group_broadcast(sg, sel, 0));
    const float p =
        lane < a.top_k ? sycl::exp(router_key_logit<Scalar>(sel) - mx) : 0.f;
    float denom;
    if (a.renormalize) {
      denom = sycl::reduce_over_group(sg, p, sycl::plus<float>());
    } else {
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < EPL; ++i) {
        const int ee = lane + i * kRouterSg;
        sum += ee < E ? sycl::exp(v[i] - mx) : 0.f;
      }
      denom = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
    }
    if (lane < a.top_k) {
      a.topk_weights[t * a.top_k + lane] = p / denom;
      a.topk_ids[t * a.top_k + lane] = 0xffff - static_cast<int>(sel & 0xffff);
    }
  }

 private:
  RouterArgs<Scalar> a;
  sycl::local_accessor<float, 1> slm;
};

// Device scratch per queue (per stream): allocated once, never freed.
static RouterSync* router_sync(sycl::queue& q) {
  static std::mutex mu;
  static std::unordered_map<const sycl::queue*, RouterSync*> syncs;
  std::lock_guard<std::mutex> lock(mu);
  RouterSync*& p = syncs[&q];
  if (!p) {
    p = sycl::malloc_device<RouterSync>(1, q);
    TORCH_CHECK(p, "router: scratch allocation failed");
    q.memset(p, 0, sizeof(RouterSync)).wait();
  }
  return p;
}

template <int MR, int EPL, bool NORM, typename Scalar>
static void router_launch(sycl::queue& q, RouterArgs<Scalar> a) {
  a.nwg_e = (a.E + kExpPerWg - 1) / kExpPerWg;
  const int nrg = (a.M + MR - 1) / MR;
  const int64_t num_wg = static_cast<int64_t>(a.nwg_e) * nrg;
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> slm(
        sycl::range<1>(RouterSlm<MR>::floats(NORM, a.K)), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(num_wg * kWgSize, kWgSize),
        router_kernel<MR, EPL, NORM, Scalar>(a, slm));
  });
}

template <int MR, bool NORM, typename Scalar>
static void router_dispatch_e(sycl::queue& q, const RouterArgs<Scalar>& a) {
  if (a.E <= 8 * kRouterSg) {
    router_launch<MR, 8, NORM>(q, a);
  } else if (a.E <= 16 * kRouterSg) {
    router_launch<MR, 16, NORM>(q, a);
  } else {
    router_launch<MR, 32, NORM>(q, a);
  }
}

// M <= 4: one row group of M rows; M > 4: multiple row groups of <= 4 rows.
template <bool NORM, typename Scalar>
static void router_dispatch(sycl::queue& q, const RouterArgs<Scalar>& a) {
  switch (a.M) {
    case 1:
      router_dispatch_e<1, NORM>(q, a);
      break;
    case 2:
      router_dispatch_e<2, NORM>(q, a);
      break;
    case 3:
      router_dispatch_e<3, NORM>(q, a);
      break;
    case 4:
      router_dispatch_e<4, NORM>(q, a);
      break;
    case 5:
    case 6:
      router_dispatch_e<3, NORM>(q, a);
      break;
    default:
      router_dispatch_e<4, NORM>(q, a);
      break;
  }
}

static inline bool router_aligned16(const void* p) {
  return reinterpret_cast<uintptr_t>(p) % 16 == 0;
}

// Checks shared by both entry points.
static void router_check_common(
    const char* name,
    const torch::Tensor& x,
    const torch::Tensor& router_weight,
    const torch::Tensor& logits,
    const torch::Tensor& topk_weights,
    const torch::Tensor& topk_ids) {
  TORCH_CHECK(
      x.dim() == 2 &&
          (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) &&
          x.stride(1) == 1,
      name,
      ": x must be fp16/bf16 [M, K] with unit inner stride");
  TORCH_CHECK(
      router_weight.dim() == 2 &&
          router_weight.scalar_type() == x.scalar_type() &&
          router_weight.is_contiguous(),
      name,
      ": router_weight must be contiguous matching fp16/bf16 [E, K]");
  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  const int64_t E = router_weight.size(0);
  const int64_t top_k = topk_ids.size(-1);
  TORCH_CHECK(router_weight.size(1) == K, name, ": K mismatch");
  TORCH_CHECK(M <= kRouterMaxM, name, ": at most 16 tokens");
  TORCH_CHECK(K % kBlk == 0, name, ": K must be a multiple of 128");
  TORCH_CHECK(
      E <= kRouterSg * kRouterMaxEpl && top_k >= 1 && top_k <= kRouterSg &&
          top_k <= E,
      name,
      ": unsupported num_experts / top_k");
  TORCH_CHECK(
      router_aligned16(router_weight.data_ptr()) &&
          router_aligned16(x.data_ptr()) && x.stride(0) % 8 == 0,
      name,
      ": x and router_weight must be 16-byte aligned");
  TORCH_CHECK(
      logits.scalar_type() == at::kFloat && logits.is_contiguous() &&
      logits.numel() >= M * E);
  TORCH_CHECK(
      topk_weights.scalar_type() == at::kFloat &&
      topk_weights.is_contiguous() && topk_weights.numel() == M * top_k);
  TORCH_CHECK(
      topk_ids.scalar_type() == at::kInt && topk_ids.is_contiguous() &&
      topk_ids.numel() == M * top_k);
}

template <typename Scalar>
static RouterArgs<Scalar> router_make_args(
    const torch::Tensor& x,
    const torch::Tensor& router_weight,
    torch::Tensor& logits,
    torch::Tensor& topk_weights,
    torch::Tensor& topk_ids,
    bool renormalize,
    sycl::queue& queue) {
  RouterArgs<Scalar> a{};
  a.x = reinterpret_cast<const Scalar*>(x.data_ptr());
  a.x_stride = x.stride(0);
  a.w = reinterpret_cast<const Scalar*>(router_weight.data_ptr());
  a.logits = logits.data_ptr<float>();
  a.topk_weights = topk_weights.data_ptr<float>();
  a.topk_ids = topk_ids.data_ptr<int32_t>();
  a.sync = router_sync(queue);
  a.M = static_cast<int>(x.size(0));
  a.E = static_cast<int>(router_weight.size(0));
  a.K = static_cast<int>(x.size(1));
  a.top_k = static_cast<int>(topk_ids.size(-1));
  a.renormalize = renormalize;
  return a;
}

}  // namespace moe
}  // namespace vllm

void router_gemv_topk_softmax(
    const torch::Tensor& x,              // [M, K] fp16
    const torch::Tensor& router_weight,  // [E, K] fp16
    torch::Tensor& logits,               // [M, E] fp32 (scratch / output)
    torch::Tensor& topk_weights,         // [M, top_k] fp32
    torch::Tensor& topk_ids,             // [M, top_k] int32
    bool renormalize) {
  using namespace vllm::moe;
  const at::DeviceGuard device_guard(x.device());
  router_check_common(
      "router_gemv_topk_softmax",
      x,
      router_weight,
      logits,
      topk_weights,
      topk_ids);
  const int64_t M = x.size(0);
  if (M == 0) return;
  auto& queue = vllm::xpu::vllmGetQueue();
  auto launch = [&](auto scalar) {
    using Scalar = decltype(scalar);
    const RouterArgs<Scalar> a = router_make_args<Scalar>(
        x, router_weight, logits, topk_weights, topk_ids, renormalize, queue);
    router_dispatch<false>(queue, a);
  };
  if (x.scalar_type() == at::kHalf)
    launch(sycl::half{});
  else
    launch(sycl::ext::oneapi::bfloat16{});
}

void router_resadd_norm_gemv_topk_softmax(
    const torch::Tensor& x,            // [M, K] fp16
    const torch::Tensor& residual,     // [M, K] fp16
    const torch::Tensor& norm_weight,  // [K] fp16, applied as (1 + w)
    double eps,
    const torch::Tensor& router_weight,  // [E, K] fp16
    torch::Tensor& logits,               // [M, E] fp32 (scratch / output)
    torch::Tensor& topk_weights,         // [M, top_k] fp32
    torch::Tensor& topk_ids,             // [M, top_k] int32
    torch::Tensor& normed_out,           // [M, K] fp16 (may alias x)
    torch::Tensor& residual_out,         // [M, K] fp16 (may alias residual)
    bool renormalize) {
  using namespace vllm::moe;
  const char* name = "router_resadd_norm_gemv_topk_softmax";
  const at::DeviceGuard device_guard(x.device());
  router_check_common(name, x, router_weight, logits, topk_weights, topk_ids);
  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  auto check_act = [&](const torch::Tensor& t, const char* what) {
    TORCH_CHECK(
        t.dim() == 2 && t.size(0) == M && t.size(1) == K &&
            t.scalar_type() == x.scalar_type() && t.stride(1) == 1 &&
            t.stride(0) % 8 == 0 && router_aligned16(t.data_ptr()),
        name,
        ": ",
        what,
        " must be fp16/bf16 [M, K], unit inner stride, 16-byte aligned rows");
  };
  check_act(residual, "residual");
  check_act(normed_out, "normed_out");
  check_act(residual_out, "residual_out");
  TORCH_CHECK(
      K <= kRouterNormMaxK,
      name,
      ": K <= ",
      kRouterNormMaxK,
      " (normed rows are staged in SLM)");
  TORCH_CHECK(
      norm_weight.dim() == 1 && norm_weight.size(0) == K &&
          norm_weight.scalar_type() == x.scalar_type() &&
          norm_weight.is_contiguous() &&
          router_aligned16(norm_weight.data_ptr()),
      name,
      ": norm_weight must be contiguous 16-byte aligned matching fp16/bf16 "
      "[K]");
  if (M == 0) return;
  auto& queue = vllm::xpu::vllmGetQueue();
  auto launch = [&](auto scalar) {
    using Scalar = decltype(scalar);
    RouterArgs<Scalar> a = router_make_args<Scalar>(
        x, router_weight, logits, topk_weights, topk_ids, renormalize, queue);
    a.residual = reinterpret_cast<const Scalar*>(residual.data_ptr());
    a.res_stride = residual.stride(0);
    a.norm_w = reinterpret_cast<const Scalar*>(norm_weight.data_ptr());
    a.normed_out = reinterpret_cast<Scalar*>(normed_out.data_ptr());
    a.normed_stride = normed_out.stride(0);
    a.residual_out = reinterpret_cast<Scalar*>(residual_out.data_ptr());
    a.res_out_stride = residual_out.stride(0);
    a.eps = static_cast<float>(eps);
    router_dispatch<true>(queue, a);
  };
  if (x.scalar_type() == at::kHalf)
    launch(sycl::half{});
  else
    launch(sycl::ext::oneapi::bfloat16{});
}
