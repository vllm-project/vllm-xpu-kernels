#include <sycl/sycl.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

#include "../utils.h"
#include "../dispatch_utils.h"

// ============================================================================
// MoE top-k routing kernels (XPU)
//
// The file is organized in four layers:
//
//   1. Scoring policies (SoftmaxScoring / SigmoidScoring)
//      Own everything specific to a scoring function:
//        - which value participates in top-k selection ("selection score")
//        - which row-wide statistics it needs (row max M, normalizer Z)
//        - how a winner's output weight is derived from its score
//        - what the register engine caches per element
//      All softmax/sigmoid knowledge lives here and only here.
//
//   2. Selection engines (RegisterTopK / ChunkedTopK / FallbackTopK)
//      Generic top-k kernels over selection scores, parameterized on a
//      policy. They differ only in data residency (full register cache /
//      per-chunk running maxima / streaming) and contain no scoring-function
//      branches; every policy hook is constexpr-guarded, so hooks that do
//      not apply compile away.
//
//      In particular, the softmax / no-bias / renormalize fast path is NOT a
//      special kernel: it is simply SoftmaxScoring<T, false>, whose selection
//      score is the raw logit (softmax is monotonic), whose row max is the
//      top-1 winner, and which needs no Z. The engines then naturally select
//      on raw logits and exp() only the k winners ("top-k first, softmax
//      after"); the final 1/sum_k rescale cancels the global normalizer.
//
//   3. Routing (launch_fast_specialized)
//      Picks an engine from policy traits + problem size.
//
//   4. Dispatch
//      Instantiates (dtype x index type x scoring x bias x experts x topk).
// ============================================================================

namespace vllm {
namespace moe {
enum class ScoringFunc {
  SOFTMAX = 0,
  SIGMOID = 1,
};

namespace topk {

constexpr int kSgSize = 16;
constexpr int kSSgSize = 16;
constexpr int kWgSize = 64;
constexpr int kMaxTopK = 8;
constexpr int kTargetValuesPerLane = 64;
constexpr int kMidTargetValuesPerLane = 16;
constexpr int kSmallTargetValuesPerLane = 4;
constexpr int kMinLanes = 4;
constexpr int kTargetChunkSize = 8;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// ---------------------------------------------------------------------------
// Machine model (Xe). The routing rules below are expressed in these units
// instead of bare problem-size thresholds, so they stay meaningful across
// devices.
// ---------------------------------------------------------------------------
struct ArchModel {
  // Register budget per work-item, in the units of register_state_dwords()
  // (see routing below). IGC caps a SIMD16 kernel at 128 regs/thread, but
  // how that maps to per-item state depends on the GRF width (512-bit on
  // Xe-HPC, 256-bit on Xe-HPG/Xe2/Xe3) and on how well the compiler packs
  // sub-dword values — so the cap is calibrated per arch family from
  // observed spill points:
  //   PVC (64B GRF)      : clean at 112 (576 fp16); 448 fp32 = 132 excluded.
  //   BMG/CRI (32B GRF)  : 576 fp16/bf16 = 112 spills ~2-5 regs; 108 clean.
  // AOT device compilations define __SYCL_TARGET_INTEL_GPU_<TARGET>__ per
  // pass, so each arch in a fat binary gets its own pruning. If the macro is
  // missing (older toolchain / JIT), we fall back to the conservative cap.
#if defined(__SYCL_TARGET_INTEL_GPU_PVC__) || \
    defined(__SYCL_TARGET_INTEL_GPU_PVC_VG__)
  static constexpr int kBudgetDWords = 128;
#else
  static constexpr int kBudgetDWords = 108;
#endif

  // Saturation: one "wave" of work-items = EUs x resident threads x SIMD.
  static constexpr int kThreadsPerEu = 8;
};

// Profitability crossovers, in machine waves (calibrated on PVC via the old
// element thresholds at N=256; the wave unit keeps them device-relative):
//  - far below one wave the grid is latency-bound and the register engine's
//    short critical path wins;
//  - below ~2 waves the chunked engine uses more lanes per row;
//  - beyond that, fewer lanes per row (more rows in flight) wins.
constexpr float kRegisterMaxWaves = 0.1f;  // ~= old 196608 elems @ N=256
constexpr float kMidMaxWaves = 2.0f;       // ~= old 4194304 elems @ N=256

//**************************helper Functions****************************

inline float ieee_div(float a, float b) {
  volatile float denom = b;
  return a / denom;
}

// Fast sigmoid, consistent with the native::exp choice of these kernels.
inline float sigmoid_fast(float x) {
  return 1.0f / (1.0f + sycl::native::exp(-x));
}

template <typename T>
inline float sigmoid_typed(T x) {
  const float value = 1.0f / (1.0f + sycl::exp(-static_cast<float>(x)));
  return static_cast<float>(static_cast<T>(value));
}

template <typename T, int W>
struct alignas(sizeof(T) * W) WideVec {
  T v[W];
};

template <typename T>
using T4 = WideVec<T, 4>;

template <typename T>
inline T4<T> ld4(const T* __restrict__ p) {
  return *reinterpret_cast<const T4<T>*>(p);
}

// Feasible lane counts are powers of two (xor-shuffle reductions) that leave
// each lane a whole number of float4s. Among them, pick the one whose
// per-lane work is closest to the target — NOT the largest power of two below
// N/target, which needlessly serializes (e.g. N=448, target 64: 448/64 = 7,
// pow2 floor gives 4 lanes at 112 values/lane, but 8 lanes at 56 values/lane
// is both closer to the target and measurably faster). Ties keep fewer lanes
// (less shuffle-reduction overhead).
constexpr int lanes_for_vpl(int num_experts, int target_values_per_lane) {
  int best = 1;
  int best_err = 0x7fffffff;
  bool found = false;
  for (int lanes = kMinLanes; lanes <= kSgSize; lanes <<= 1) {
    if (num_experts % (lanes * 4) != 0) continue;
    const int vpl = num_experts / lanes;
    const int err = vpl > target_values_per_lane ? vpl - target_values_per_lane
                                                 : target_values_per_lane - vpl;
    if (err < best_err) {
      best = lanes;
      best_err = err;
      found = true;
    }
  }
  if (found) return best;
  // Tiny rows cannot afford kMinLanes; take the largest feasible lane count.
  for (int lanes = kMinLanes >> 1; lanes >= 1; lanes >>= 1)
    if (num_experts % (lanes * 4) == 0) return lanes;
  return 1;
}

constexpr int lanes_for(int num_experts) {
  return lanes_for_vpl(num_experts, kTargetValuesPerLane);
}

constexpr int lanes_for_mid(int num_experts) {
  return lanes_for_vpl(num_experts, kMidTargetValuesPerLane);
}

constexpr int small_lanes_for(int num_experts) {
  return lanes_for_vpl(num_experts, kSmallTargetValuesPerLane);
}

constexpr int groups_per_chunk(int groups_per_lane) {
  int chunk_groups = kTargetChunkSize / 4;
  if (chunk_groups < 1) chunk_groups = 1;
  while (chunk_groups > 1 && (groups_per_lane % chunk_groups != 0 ||
                              groups_per_lane / chunk_groups < 2))
    chunk_groups >>= 1;
  return chunk_groups;
}

template <int LANES>
inline void lane_group_argmax(const sycl::sub_group& sg, float& s, int& i) {
#pragma unroll
  for (int m = 1; m < LANES; m <<= 1) {
    const float os = sycl::permute_group_by_xor(sg, s, static_cast<size_t>(m));
    const int oi = sycl::permute_group_by_xor(sg, i, static_cast<size_t>(m));
    const bool take = (os > s) | ((os == s) & (oi < i));
    s = take ? os : s;
    i = take ? oi : i;
  }
}

template <int LANES>
inline float lane_group_max(const sycl::sub_group& sg, float s) {
#pragma unroll
  for (int m = 1; m < LANES; m <<= 1) {
    const float os = sycl::permute_group_by_xor(sg, s, static_cast<size_t>(m));
    s = sycl::fmax(s, os);
  }
  return s;
}

template <int LANES>
inline float lane_group_sum(const sycl::sub_group& sg, float s) {
#pragma unroll
  for (int m = 1; m < LANES; m <<= 1) {
    s += sycl::permute_group_by_xor(sg, s, static_cast<size_t>(m));
  }
  return s;
}

constexpr int store_width(int topk) {
  return topk % 4 == 0 ? 4 : (topk % 2 == 0 ? 2 : 1);
}

inline void store_row(
    float* __restrict__ weights,
    int* __restrict__ indices,
    int* __restrict__ source_rows,
    int64_t token,
    int64_t num_tokens,
    const float* __restrict__ num,
    const int* __restrict__ wid,
    float scale,
    bool is_pad,
    int topk) {
  float* __restrict__ wo = weights + token * topk;
  int* __restrict__ io = indices + token * topk;
  int* __restrict__ so = source_rows + token * topk;
  for (int k = 0; k < topk; ++k) {
    wo[k] = is_pad ? 0.0f : num[k] * scale;
    io[k] = is_pad ? -1 : wid[k];
    so[k] = k * num_tokens + token;
  }
}

// Vectorized variant for compile-time topk: W-wide stores per array.
template <int TOPK, int W>
inline void store_row_vec(
    float* __restrict__ weights,
    int* __restrict__ indices,
    int* __restrict__ source_rows,
    int64_t token,
    int64_t num_tokens,
    const float* __restrict__ num,
    const int* __restrict__ wid,
    float scale,
    bool is_pad) {
  float* __restrict__ wo = weights + token * TOPK;
  int* __restrict__ io = indices + token * TOPK;
  int* __restrict__ so = source_rows + token * TOPK;
#pragma unroll
  for (int b = 0; b < TOPK; b += W) {
    WideVec<float, W> tw;
    WideVec<int, W> ti;
#pragma unroll
    for (int e = 0; e < W; ++e) {
      const int k = b + e;
      tw.v[e] = is_pad ? 0.0f : num[k] * scale;
      ti.v[e] = is_pad ? -1 : wid[k];
      so[k] = k * num_tokens + token;
    }
    *reinterpret_cast<WideVec<float, W>*>(wo + b) = tw;
    *reinterpret_cast<WideVec<int, W>*>(io + b) = ti;
  }
}

template <int TOPK>
inline void store_row_dispatch(
    float* __restrict__ weights,
    int* __restrict__ indices,
    int* __restrict__ source_rows,
    int64_t token,
    int64_t num_tokens,
    const float* __restrict__ num,
    const int* __restrict__ wid,
    float scale,
    bool is_pad,
    int topk) {
  if constexpr (TOPK == 0) {
    store_row(
        weights,
        indices,
        source_rows,
        token,
        num_tokens,
        num,
        wid,
        scale,
        is_pad,
        topk);
  } else {
    store_row_vec<TOPK, store_width(TOPK)>(
        weights,
        indices,
        source_rows,
        token,
        num_tokens,
        num,
        wid,
        scale,
        is_pad);
  }
}

//**************************Layer 1: scoring
// policies****************************
//
// Policy interface consumed by the selection engines:
//
//   using InputT / CacheT     element type / register-engine cache type
//   kUsesRowStats             row-wide state (M, Z) exists at all
//   kNeedsRowNorm             selection score needs M and 1/Z up front
//                             (engines must run a normalization pre-pass
//                             before forming selection maxima)
//   kRowMaxIsWinner           row max M is the top-1 winner itself, so no
//                             separate max pass is ever needed
//   needs_full_z()            runtime: output weights are true softmax probs
//                             (renormalize == false), so the full-row Z is
//                             accumulated once M is known
//   select(x, idx)            selection score of a raw element
//   to_cache(x, idx)          per-element value for the register cache (the
//                             cache doubles as the selection domain)
//   materialize_score(x, idx) rewrite a cached raw logit into its normalized
//                             selection score in place (kNeedsRowNorm only)
//   norm_accum(x)             exp(x - M) contribution to Z
//   winner_weight(ws, wi)     winner weight from its selection score
//   weight_from_raw(raw, wi)  winner weight from its raw logit (streaming
//                             fallback; avoids score - bias cancellation)
//   output_scale()            normalizer folded into the one-shot store scale
//                             (1/Z for softmax-no-bias, 1 otherwise)
//   kRegisterWhenFits         routing hint: register engine wins whenever the
//                             cache fits without spilling
//   kRegisterWhenSmall        routing hint: register engine wins only for
//                             latency-bound (sub-wave) grids

// Row-statistics interface for policies that need none.
struct NoRowStats {
  static constexpr bool kUsesRowStats = false;
  static constexpr bool kNeedsRowNorm = false;
  static constexpr bool kRowMaxIsWinner = false;

  bool needs_full_z() const { return false; }
  void set_row_max(float) {}
  void set_inv_z(float) {}
  float output_scale() const { return 1.0f; }
};

// ---------------------------------------------------------------------------
// Scoring policy: sigmoid
//
//   selection score : sigmoid(x) rounded to the input type (+ bias)
//   row statistics  : none — the score is purely elementwise
//   output weight   : sigmoid(x) of the winner
//
// RAW_SELECT (register engine only, float input, no bias): cache the raw
// logits and select on them directly — sigmoid is strictly monotonic, so the
// top-k set is unchanged and sigmoid is evaluated only for the k winners.
// ---------------------------------------------------------------------------
template <typename T, bool HAS_BIAS, bool RAW_SELECT>
struct SigmoidScoring : NoRowStats {
  using InputT = T;
  // Biased scores are cached as float (the prob is recoverable as
  // score - bias); unbiased caches keep the input type.
  using CacheT = std::conditional_t<HAS_BIAS, float, T>;

  // An elementwise score is cheap to cache and caching avoids recomputing
  // the scoring function in every pick, so the register engine wins whenever
  // the cache fits without spilling (see register_engine_fits).
  static constexpr bool kRegisterWhenFits = true;
  static constexpr bool kRegisterWhenSmall = false;

  const float* __restrict__ bias;
  bool renormalize;

  SigmoidScoring(const float* b, bool renorm) : bias(b), renormalize(renorm) {}

  float select(T x, int idx) const {
    return sigmoid_typed(x) + (HAS_BIAS ? bias[idx] : 0.0f);
  }

  CacheT to_cache(T x, int idx) const {
    if constexpr (HAS_BIAS) {
      return sigmoid_typed(x) + bias[idx];
    } else if constexpr (RAW_SELECT) {
      return x;
    } else {
      return static_cast<CacheT>(sigmoid_typed(x));
    }
  }

  float winner_weight(float ws, int wi) const {
    if constexpr (HAS_BIAS) {
      return ws - bias[wi];
    } else if constexpr (RAW_SELECT) {
      return sigmoid_typed(static_cast<T>(ws));  // ws is the raw logit
    } else {
      return ws;  // ws is already the sigmoid value
    }
  }

  float weight_from_raw(T raw, int /*wi*/) const { return sigmoid_typed(raw); }
};

// ---------------------------------------------------------------------------
// Scoring policy: softmax
//
//   HAS_BIAS (bias-augmented routing):
//     selection score : exp(x - M) / Z + bias   -> needs row max M and Z
//     output weight   : exp(x_w - M) / Z        (= score - bias)
//
//   no bias:
//     selection score : raw logit (softmax is monotonic in x)
//     row max M       : the top-1 winner itself — no max pass needed
//     output weight   : exp(x_w - M), with the normalizer folded into the
//       one-shot store scale (see output_scale):
//       - renormalize == true : no Z at all; the final 1/sum_k rescale makes
//         the weights exact softmax-over-winners ("top-k first, softmax
//         after")
//       - renormalize == false: the full-row Z is accumulated once the first
//         winner fixes M, and the store scale is rsf / Z
// ---------------------------------------------------------------------------
template <typename T, bool HAS_BIAS>
struct SoftmaxScoring {
  using InputT = T;
  using CacheT = T;  // the register engine caches raw logits

  static constexpr bool kUsesRowStats = true;
  static constexpr bool kNeedsRowNorm = HAS_BIAS;     // selection needs M, 1/Z
  static constexpr bool kRowMaxIsWinner = !HAS_BIAS;  // M is the top-1 logit

  // Caching raw logits is just a load, and with bias the cache must be
  // rewritten into probabilities, so the register engine only pays off
  // without bias, and only for latency-bound (sub-wave) grids.
  static constexpr bool kRegisterWhenFits = false;
  static constexpr bool kRegisterWhenSmall = !HAS_BIAS;

  const float* __restrict__ bias;
  bool renormalize;
  float row_max = 0.0f;
  float inv_z = 1.0f;  // 1/Z once known

  SoftmaxScoring(const float* b, bool renorm) : bias(b), renormalize(renorm) {}

  bool needs_full_z() const { return !HAS_BIAS && !renormalize; }

  void set_row_max(float m) { row_max = m; }
  void set_inv_z(float iz) { inv_z = iz; }

  float norm_accum(float x) const { return sycl::native::exp(x - row_max); }
  float norm_score(float x, int idx) const {
    return sycl::native::exp(x - row_max) * inv_z + bias[idx];
  }

  float select(T x, int idx) const {
    if constexpr (HAS_BIAS) {
      return norm_score(static_cast<float>(x), idx);
    } else {
      return static_cast<float>(x);  // monotonic with the softmax prob
    }
  }

  CacheT to_cache(T x, int /*idx*/) const { return x; }
  CacheT materialize_score(float x, int idx) const {
    return static_cast<CacheT>(norm_score(x, idx));
  }

  // Softmax weights are exp(x - M); for the no-bias policy the 1/Z factor is
  // folded into the one-shot store scale instead of multiplying every pick.
  float winner_weight(float ws, int wi) const {
    if constexpr (HAS_BIAS) {
      return ws - bias[wi];  // softmax prob at the winner
    } else {
      return sycl::native::exp(ws - row_max);
    }
  }

  float weight_from_raw(T raw, int /*wi*/) const {
    if constexpr (HAS_BIAS) {
      return sycl::native::exp(static_cast<float>(raw) - row_max) * inv_z;
    } else {
      return sycl::native::exp(static_cast<float>(raw) - row_max);
    }
  }

  // Multiplier folded into the final store scale.
  float output_scale() const { return HAS_BIAS ? 1.0f : inv_z; }
};

// Policy used by the chunked / fallback engines.
template <ScoringFunc SF, typename T, bool HAS_BIAS>
using KernelPolicy = std::conditional_t<
    SF == ScoringFunc::SOFTMAX,
    SoftmaxScoring<T, HAS_BIAS>,
    SigmoidScoring<T, HAS_BIAS, /*RAW_SELECT=*/false>>;

// Policy used by the register engine, which may select on raw sigmoid logits
// (float input, no bias).
template <ScoringFunc SF, typename T, bool HAS_BIAS>
using RegisterKernelPolicy = std::conditional_t<
    SF == ScoringFunc::SOFTMAX,
    SoftmaxScoring<T, HAS_BIAS>,
    SigmoidScoring<T, HAS_BIAS, !HAS_BIAS && std::is_same_v<T, float>>>;

//**************************Layer 2: selection
// engines****************************

// ---------------------------------------------------------------------------
// Chunked engine: keeps one running selection maximum per chunk of the
// per-lane slice (cs/ci), and rescans only the winner's chunk after each
// pick. Suited to large rows; reads the row from global memory.
// ---------------------------------------------------------------------------
template <int N, typename Policy, int LANES, int TOPK = 0>
struct ChunkedTopK {
  static_assert(TOPK == 0 || TOPK == 4 || TOPK == 8);
  using InputT = typename Policy::InputT;
  static constexpr int kVpl = N / LANES;
  static constexpr int kG = kVpl / 4;
  static constexpr int kGpc = groups_per_chunk(kG);
  static constexpr int kChunks = kG / kGpc;
  static_assert(
      kVpl % 4 == 0, "per-lane slice must be a whole number of float4");
  static_assert(kChunks >= 1, "empty chunk cache");

  const InputT* __restrict__ gating;
  float* __restrict__ weights;
  int* __restrict__ indices;
  int* __restrict__ source_rows;
  const bool* __restrict__ is_padding;
  Policy policy;
  const double routed_scaling_factor;
  int64_t num_tokens;
  int runtime_topk;

  static int lane_of(const sycl::sub_group& sg) {
    return static_cast<int>(sg.get_local_id()[0]) & (LANES > 1 ? LANES - 1 : 0);
  }

  // Full-row normalizer Z = sum(exp(logit - M)) reduced over the lane group;
  // returns 1/Z. Only instantiated for policies whose weights need it.
  float compute_inv_z(
      const InputT* __restrict__ row,
      const sycl::sub_group& sg,
      const Policy& scoring) const {
    float zs = 0.0f;
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
#pragma unroll
      for (int g = 0; g < kGpc; ++g) {
        const int i0 = ((c * kGpc + g) * LANES + lane_of(sg)) * 4;
        const T4<InputT> x = ld4(row + i0);
#pragma unroll
        for (int e = 0; e < 4; ++e)
          zs += scoring.norm_accum(static_cast<float>(x.v[e]));
      }
    }
    if constexpr (LANES > 1) zs = lane_group_sum<LANES>(sg, zs);
    return ieee_div(1.0f, zs);
  }

  [[sycl::reqd_sub_group_size(kSgSize)]] void
  operator()(sycl::nd_item<1> it) const {
    const int topk = TOPK == 0 ? runtime_topk : TOPK;
    const sycl::sub_group sg = it.get_sub_group();
    const int64_t gid = static_cast<int64_t>(it.get_global_linear_id());
    const int lane = LANES == 1 ? 0 : static_cast<int>(gid & (LANES - 1));
    const int64_t token = LANES == 1 ? gid : gid / LANES;
    const bool active = token < num_tokens;
    const InputT* __restrict__ row = gating + (active ? token : 0) * N;

    Policy policy = this->policy;

    // Pass A: per-chunk selection maxima cs/ci. When the selection score
    // needs row statistics (kNeedsRowNorm), this pass tracks raw maxima
    // instead and the selection maxima are formed in pass B once M and 1/Z
    // are known.
    float cs[kChunks];
    int ci[kChunks];
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      float bs = kNegInf;
      int bi = N;
#pragma unroll
      for (int g = 0; g < kGpc; ++g) {
        const int i0 = ((c * kGpc + g) * LANES + lane) * 4;
        const T4<InputT> x = ld4(row + i0);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float sel = Policy::kNeedsRowNorm
                                ? static_cast<float>(x.v[e])
                                : policy.select(x.v[e], i0 + e);
          const bool take = sel > bs;
          bs = take ? sel : bs;
          bi = take ? i0 + e : bi;
        }
      }
      cs[c] = bs;
      ci[c] = bi;
    }

    if constexpr (Policy::kNeedsRowNorm) {
      // cs currently holds raw chunk maxima: reduce to the row max M.
      float rm = cs[0];
#pragma unroll
      for (int c = 1; c < kChunks; ++c)
        rm = sycl::fmax(rm, cs[c]);
      if constexpr (LANES > 1) rm = lane_group_max<LANES>(sg, rm);
      policy.set_row_max(rm);
      policy.set_inv_z(compute_inv_z(row, sg, policy));
      // Pass B: selection maxima over the normalized scores.
#pragma unroll
      for (int c = 0; c < kChunks; ++c) {
        float bs = kNegInf;
        int bi = N;
#pragma unroll
        for (int g = 0; g < kGpc; ++g) {
          const int i0 = ((c * kGpc + g) * LANES + lane) * 4;
          const T4<InputT> x = ld4(row + i0);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float sel =
                policy.norm_score(static_cast<float>(x.v[e]), i0 + e);
            const bool take = sel > bs;
            bs = take ? sel : bs;
            bi = take ? i0 + e : bi;
          }
        }
        cs[c] = bs;
        ci[c] = bi;
      }
    }

    float num[TOPK == 0 ? kMaxTopK : TOPK];
    int wid[TOPK == 0 ? kMaxTopK : TOPK];
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < topk; ++k) {
      float ws = cs[0];
      int wi = ci[0];
      int bc = 0;
#pragma unroll
      for (int c = 1; c < kChunks; ++c) {
        const bool take = cs[c] > ws;
        ws = take ? cs[c] : ws;
        wi = take ? ci[c] : wi;
        bc = take ? c : bc;
      }
      if constexpr (LANES > 1) lane_group_argmax<LANES>(sg, ws, wi);

      // Policies whose selection is monotonic in the raw logit get the row
      // max for free from the first winner; accumulate Z only if the weights
      // are true (unrenormalized) softmax probs.
      if constexpr (Policy::kRowMaxIsWinner) {
        if (k == 0) {
          policy.set_row_max(ws);
          if (policy.needs_full_z())
            policy.set_inv_z(compute_inv_z(row, sg, policy));
        }
      }

      const float w = policy.winner_weight(ws, wi);
      num[k] = w;
      sum += num[k];
      wid[k] = wi;

      // Rescan the winner's chunk for the next-best score below (ws, wi).
      // Branchless: iteration order is ascending idx, so on a tie the first
      // (lowest-index) candidate is kept.
      const bool mine = LANES == 1 || (((wi >> 2) & (LANES - 1)) == lane);
      if (mine) {
        float ns = kNegInf;
        int ni = N;
#pragma unroll
        for (int g = 0; g < kGpc; ++g) {
          const int i0 = ((bc * kGpc + g) * LANES + lane) * 4;
          const T4<InputT> x = ld4(row + i0);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int idx = i0 + e;
            const float sel = policy.select(x.v[e], idx);
            const bool below = (sel < ws) | ((sel == ws) & (idx > wi));
            const bool take = below & ((sel > ns) | ((sel == ns) & (idx < ni)));
            ns = take ? sel : ns;
            ni = take ? idx : ni;
          }
        }
#pragma unroll
        for (int c = 0; c < kChunks; ++c) {
          const bool hit = c == bc;
          cs[c] = hit ? ns : cs[c];
          ci[c] = hit ? ni : ci[c];
        }
      }
    }

    if (active & (lane == 0)) {
      const bool is_pad = is_padding != nullptr && is_padding[token];
      float scale =
          static_cast<float>(routed_scaling_factor) * policy.output_scale();
      if (policy.renormalize) {
        const float denom = sum > 0.0f ? sum : 1.0f;
        scale = ieee_div(scale, denom);
      }
      store_row_dispatch<TOPK>(
          weights,
          indices,
          source_rows,
          token,
          num_tokens,
          num,
          wid,
          scale,
          is_pad,
          topk);
    }
  }
};

// ---------------------------------------------------------------------------
// Register engine: caches the whole per-lane slice in registers (the cache
// doubles as the selection domain) and iterates a full argmax per pick,
// evicting each winner. Suited to small rows.
// ---------------------------------------------------------------------------
template <int N, typename Policy, int LANES, int TOPK = 0>
struct RegisterTopK {
  static_assert(TOPK == 0 || TOPK == 4 || TOPK == 8);
  using InputT = typename Policy::InputT;
  using CacheT = typename Policy::CacheT;
  static constexpr int kVpl = N / LANES;
  static constexpr int kG = kVpl / 4;

  static_assert(kG >= 1, "small kernel needs at least one float4 per lane");

  const InputT* __restrict__ gating;
  float* __restrict__ weights;
  int* __restrict__ indices;
  int* __restrict__ source_rows;
  const bool* __restrict__ is_padding;
  Policy policy;
  const double routed_scaling_factor;
  int64_t num_tokens;
  int runtime_topk;

  [[sycl::reqd_sub_group_size(kSSgSize)]] void
  operator()(sycl::nd_item<1> it) const {
    const int topk = TOPK == 0 ? runtime_topk : TOPK;
    const sycl::sub_group sg = it.get_sub_group();
    const int64_t gid = static_cast<int64_t>(it.get_global_linear_id());
    const int lane = LANES == 1 ? 0 : static_cast<int>(gid & (LANES - 1));
    const int64_t token = LANES == 1 ? gid : gid / LANES;
    const bool active = token < num_tokens;
    const InputT* __restrict__ row = gating + (active ? token : 0) * N;

    Policy policy = this->policy;

    // Fill the register cache. What is cached (raw logits, rounded scores,
    // biased scores) is the policy's CacheT/to_cache decision.
    T4<CacheT> v[kG];
#pragma unroll
    for (int g = 0; g < kG; ++g) {
      const int i0 = (g * LANES + lane) * 4;
      const T4<InputT> x = ld4(row + i0);
#pragma unroll
      for (int e = 0; e < 4; ++e)
        v[g].v[e] = policy.to_cache(x.v[e], i0 + e);
    }

    // Normalized selection scores are formed in place from the cached raw
    // logits. Once selection scores are formed, the original probability is
    // recoverable as score - bias, so a second register array would only
    // increase GRF pressure.
    if constexpr (Policy::kNeedsRowNorm) {
      float rm = static_cast<float>(v[0].v[0]);
#pragma unroll
      for (int g = 0; g < kG; ++g)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          rm = sycl::fmax(rm, static_cast<float>(v[g].v[e]));
      if constexpr (LANES > 1) rm = lane_group_max<LANES>(sg, rm);
      policy.set_row_max(rm);
      float zs = 0.0f;
#pragma unroll
      for (int g = 0; g < kG; ++g)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          zs += policy.norm_accum(static_cast<float>(v[g].v[e]));
      if constexpr (LANES > 1) zs = lane_group_sum<LANES>(sg, zs);
      policy.set_inv_z(ieee_div(1.0f, zs));
#pragma unroll
      for (int g = 0; g < kG; ++g)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          v[g].v[e] = policy.materialize_score(
              static_cast<float>(v[g].v[e]), (g * LANES + lane) * 4 + e);
    }

    float num[TOPK == 0 ? kMaxTopK : TOPK];
    int wid[TOPK == 0 ? kMaxTopK : TOPK];
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < topk; ++k) {
      float acc[4];
#pragma unroll
      for (int e = 0; e < 4; ++e)
        acc[e] = static_cast<float>(v[0].v[e]);
#pragma unroll
      for (int g = 1; g < kG; ++g) {
#pragma unroll
        for (int e = 0; e < 4; ++e)
          acc[e] = sycl::fmax(acc[e], static_cast<float>(v[g].v[e]));
      }
      acc[0] = sycl::fmax(acc[0], acc[2]);
      acc[1] = sycl::fmax(acc[1], acc[3]);
      float ws = sycl::fmax(acc[0], acc[1]);

      int iacc[4];
#pragma unroll
      for (int e = 0; e < 4; ++e)
        iacc[e] = static_cast<float>(v[0].v[e]) == ws ? lane * 4 + e : N;
#pragma unroll
      for (int g = 1; g < kG; ++g) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int idx = (g * LANES + lane) * 4 + e;
          iacc[e] = static_cast<float>(v[g].v[e]) == ws
                        ? sycl::min(iacc[e], idx)
                        : iacc[e];
        }
      }
      iacc[0] = sycl::min(iacc[0], iacc[2]);
      iacc[1] = sycl::min(iacc[1], iacc[3]);
      int wi = sycl::min(iacc[0], iacc[1]);

      if constexpr (LANES > 1) lane_group_argmax<LANES>(sg, ws, wi);

      if constexpr (Policy::kRowMaxIsWinner) {
        if (k == 0) {
          policy.set_row_max(ws);
          if (policy.needs_full_z()) {
            // Full-row Z from the register copy.
            float zs = 0.0f;
#pragma unroll
            for (int g = 0; g < kG; ++g)
#pragma unroll
              for (int e = 0; e < 4; ++e)
                zs += policy.norm_accum(static_cast<float>(v[g].v[e]));
            if constexpr (LANES > 1) zs = lane_group_sum<LANES>(sg, zs);
            policy.set_inv_z(ieee_div(1.0f, zs));
          }
        }
      }

      const float w = policy.winner_weight(ws, wi);
      num[k] = w;
      sum += num[k];
      wid[k] = wi;

#pragma unroll
      for (int g = 0; g < kG; ++g) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int idx = (g * LANES + lane) * 4 + e;
          v[g].v[e] = idx == wi ? static_cast<CacheT>(kNegInf) : v[g].v[e];
        }
      }
    }

    if (active & (lane == 0)) {
      const bool is_pad = is_padding != nullptr && is_padding[token];
      float scale =
          static_cast<float>(routed_scaling_factor) * policy.output_scale();
      if (policy.renormalize) {
        const float denom = sum > 0.0f ? sum : 1.0f;
        scale = ieee_div(scale, denom);
      }
      store_row_dispatch<TOPK>(
          weights,
          indices,
          source_rows,
          token,
          num_tokens,
          num,
          wid,
          scale,
          is_pad,
          topk);
    }
  }
};

// ---------------------------------------------------------------------------
// Fallback engine: streams the row from global memory once per pick (full
// sub-group per token). Handles any N and any index type.
// ---------------------------------------------------------------------------
template <int N, typename IndexType, typename Policy>
struct FallbackTopK {
  using InputT = typename Policy::InputT;

  const InputT* __restrict__ gating;
  float* __restrict__ weights;
  IndexType* __restrict__ indices;
  int* __restrict__ source_rows;
  const bool* __restrict__ is_padding;
  Policy policy;
  const double routed_scaling_factor;
  int64_t num_tokens;
  int topk;

  [[sycl::reqd_sub_group_size(kSgSize)]] void
  operator()(sycl::nd_item<1> it) const {
    const sycl::sub_group sg = it.get_sub_group();
    const int64_t gid = static_cast<int64_t>(it.get_global_linear_id());
    const int lane = static_cast<int>(gid & (kSgSize - 1));
    const int64_t token = gid / kSgSize;
    const bool active = token < num_tokens;
    const bool is_pad = is_padding != nullptr && active && is_padding[token];
    const InputT* row = gating + (active ? token : 0) * N;

    Policy policy = this->policy;

    // Row statistics up front: the streaming engine revisits the row for
    // every pick, so M (and Z when needed) are computed once here.
    if constexpr (Policy::kUsesRowStats) {
      float local_max = kNegInf;
      for (int i = lane; i < N; i += kSgSize)
        local_max = sycl::fmax(local_max, static_cast<float>(row[i]));
      policy.set_row_max(lane_group_max<kSgSize>(sg, local_max));

      if constexpr (Policy::kNeedsRowNorm) {
        float local_sum = 0.0f;
        for (int i = lane; i < N; i += kSgSize)
          local_sum += policy.norm_accum(static_cast<float>(row[i]));
        policy.set_inv_z(
            ieee_div(1.0f, lane_group_sum<kSgSize>(sg, local_sum)));
      } else if (policy.needs_full_z()) {
        float local_sum = 0.0f;
        for (int i = lane; i < N; i += kSgSize)
          local_sum += policy.norm_accum(static_cast<float>(row[i]));
        policy.set_inv_z(
            ieee_div(1.0f, lane_group_sum<kSgSize>(sg, local_sum)));
      }
    }

    float sum = 0.0f;
    float previous_score = std::numeric_limits<float>::infinity();
    int previous_index = -1;
    for (int k = 0; k < topk; ++k) {
      float local_score = kNegInf;
      int local_index = N;
      for (int i = lane; i < N; i += kSgSize) {
        const float score = policy.select(row[i], i);
        const bool after_previous =
            k == 0 || score < previous_score ||
            ((score == previous_score) && i > previous_index);
        const bool take =
            after_previous && (score > local_score ||
                               ((score == local_score) && i < local_index));
        local_score = take ? score : local_score;
        local_index = take ? i : local_index;
      }

      lane_group_argmax<kSgSize>(sg, local_score, local_index);
      // Weight from the raw logit: avoids the score - bias cancellation.
      const float weight =
          policy.weight_from_raw(row[local_index], local_index);
      sum += weight;

      if (active && lane == 0) {
        const int64_t out = token * topk + k;
        weights[out] = is_pad ? 0.0f : weight;
        indices[out] = is_pad ? static_cast<IndexType>(-1)
                              : static_cast<IndexType>(local_index);
        source_rows[out] = static_cast<int64_t>(k) * num_tokens + token;
      }
      previous_score = local_score;
      previous_index = local_index;
    }

    if (active && lane == 0) {
      float scale =
          static_cast<float>(routed_scaling_factor) * policy.output_scale();
      if (policy.renormalize) scale = ieee_div(scale, sum > 0.0f ? sum : 1.0f);
      if (!is_pad) {
        for (int k = 0; k < topk; ++k)
          weights[token * topk + k] *= scale;
      }
    }
  }
};

//**************************Layer 3: routing****************************

// ---------------------------------------------------------------------------
// Register-engine feasibility (compile time).
//
// The register engine holds the whole per-lane slice in registers; if the
// resident state exceeds the per-work-item register budget, the kernel spills
// to local memory and is strictly slower than the chunked engine (observed:
// N=448/576 with a float cache spill on PVC). Resident state is dominated by
// the selection cache (plus ~1x of it in unroll/argmax temporaries), the
// top-k outputs and a fixed cost for addressing/policy/misc. kTempFactor and
// kFixedDWords are calibrated so the observed spill points fall on the right
// side — re-calibrate when the compiler/arch changes, or check the IGC
// register-usage dump.
// ---------------------------------------------------------------------------
template <int N, int LANES, typename Policy, int TOPK>
constexpr int register_state_dwords() {
  using CacheT = typename Policy::CacheT;
  constexpr int kCache = (N / LANES) * static_cast<int>(sizeof(CacheT)) / 4;
  constexpr int kOut = 2 * (TOPK == 0 ? kMaxTopK : TOPK);  // num[] + wid[]
  constexpr int kArgmax = 8;                               // acc[4] + iacc[4]
  constexpr int kNorm = Policy::kNeedsRowNorm ? 8 : 0;     // M/Z + materialize
  constexpr int kTempFactor = 2;
  constexpr int kFixedDWords = 52;
  return kCache * kTempFactor + kOut + kArgmax + kNorm + kFixedDWords;
}

template <int N, int LANES, typename Policy, int TOPK>
constexpr bool register_engine_fits() {
  return register_state_dwords<N, LANES, Policy, TOPK>() <=
         ArchModel::kBudgetDWords;
}

// One wave of work-items: every EU resident thread runs one sub-group.
// (Assumes a single device; the value is cached on first use.)
inline int64_t work_items_per_wave(const sycl::queue& q) {
  static const int64_t items = [&] {
    const int eus =
        q.get_device().get_info<sycl::info::device::max_compute_units>();
    return static_cast<int64_t>(eus) * ArchModel::kThreadsPerEu * kSSgSize;
  }();
  return items;
}

template <
    int N,
    typename InputT,
    int LANES,
    ScoringFunc SF,
    bool HAS_BIAS,
    int TOPK = 0>
void launch_chunked(
    sycl::queue& q,
    const InputT* gating,
    float* weights,
    int* indices,
    int* source_rows,
    const bool* is_padding,
    const float* bias,
    const bool renormalize,
    const double routed_scaling_factor,
    int64_t num_tokens,
    int topk) {
  using Policy = KernelPolicy<SF, InputT, HAS_BIAS>;
  constexpr int kTokensPerWg = kWgSize / LANES;
  const size_t groups =
      static_cast<size_t>((num_tokens + kTokensPerWg - 1) / kTokensPerWg);
  q.parallel_for(
      sycl::nd_range<1>{
          sycl::range<1>{groups * kWgSize}, sycl::range<1>{kWgSize}},
      ChunkedTopK<N, Policy, LANES, TOPK>{
          gating,
          weights,
          indices,
          source_rows,
          is_padding,
          Policy{bias, renormalize},
          routed_scaling_factor,
          num_tokens,
          topk});
}

template <int N, typename InputT, ScoringFunc SF, bool HAS_BIAS, int TOPK>
void launch_register(
    sycl::queue& q,
    const InputT* gating,
    float* weights,
    int* indices,
    int* source_rows,
    const bool* is_padding,
    const float* bias,
    const bool renormalize,
    const double routed_scaling_factor,
    int64_t num_tokens,
    int topk) {
  using Policy = RegisterKernelPolicy<SF, InputT, HAS_BIAS>;
  constexpr int kSLanes = small_lanes_for(N);
  constexpr int kSTokensPerWg = kWgSize / kSLanes;
  const size_t groups =
      static_cast<size_t>((num_tokens + kSTokensPerWg - 1) / kSTokensPerWg);
  q.parallel_for(
      sycl::nd_range<1>{
          sycl::range<1>{groups * kWgSize}, sycl::range<1>{kWgSize}},
      RegisterTopK<N, Policy, kSLanes, TOPK>{
          gating,
          weights,
          indices,
          source_rows,
          is_padding,
          Policy{bias, renormalize},
          routed_scaling_factor,
          num_tokens,
          topk});
}

template <int N, typename InputT, ScoringFunc SF, bool HAS_BIAS, int TOPK>
void launch_fast_specialized(
    sycl::queue& q,
    const InputT* gating,
    float* weights,
    int* indices,
    int* source_rows,
    const bool* is_padding,
    const float* bias,
    const bool renormalize,
    const double routed_scaling_factor,
    int64_t num_tokens,
    int topk) {
  using Traits = KernelPolicy<SF, InputT, HAS_BIAS>;
  using RegPolicy = RegisterKernelPolicy<SF, InputT, HAS_BIAS>;
  constexpr int kSLanes = small_lanes_for(N);

  // Feasibility is compile-time: a spilling register kernel is strictly
  // slower than the chunked engine, so it is never instantiated.
  constexpr bool kFits = register_engine_fits<N, kSLanes, RegPolicy, TOPK>();

  // Profitability is runtime, in machine waves:
  //  - kRegisterWhenFits (sigmoid): cached scores avoid recomputing the
  //    scoring function in every pick, so registers win whenever they fit;
  //  - kRegisterWhenSmall (softmax, no bias): registers only pay off when the
  //    grid is latency-bound (a fraction of one wave).
  if constexpr (
      (Traits::kRegisterWhenFits || Traits::kRegisterWhenSmall) && kFits) {
    const int64_t items = num_tokens * kSLanes;
    if (Traits::kRegisterWhenFits ||
        items <=
            static_cast<int64_t>(kRegisterMaxWaves * work_items_per_wave(q))) {
      launch_register<N, InputT, SF, HAS_BIAS, TOPK>(
          q,
          gating,
          weights,
          indices,
          source_rows,
          is_padding,
          bias,
          renormalize,
          routed_scaling_factor,
          num_tokens,
          topk);
      return;
    }
  }

  // Chunked engine: more lanes per row while the grid is small, fewer lanes
  // (more rows in flight) once the machine is saturated.
  constexpr int kMidLanes = lanes_for_mid(N);
  const int64_t mid_items = num_tokens * kMidLanes;
  if (mid_items <=
      static_cast<int64_t>(kMidMaxWaves * work_items_per_wave(q))) {
    launch_chunked<N, InputT, kMidLanes, SF, HAS_BIAS, TOPK>(
        q,
        gating,
        weights,
        indices,
        source_rows,
        is_padding,
        bias,
        renormalize,
        routed_scaling_factor,
        num_tokens,
        topk);
    return;
  }

  launch_chunked<N, InputT, lanes_for(N), SF, HAS_BIAS, TOPK>(
      q,
      gating,
      weights,
      indices,
      source_rows,
      is_padding,
      bias,
      renormalize,
      routed_scaling_factor,
      num_tokens,
      topk);
}

template <int N, typename InputT, ScoringFunc SF, bool HAS_BIAS>
void launch_fast(
    sycl::queue& q,
    const InputT* gating,
    float* weights,
    int* indices,
    int* source_rows,
    const bool* is_padding,
    const float* bias,
    const bool renormalize,
    const double routed_scaling_factor,
    int64_t num_tokens,
    int topk) {
  const auto launch = [&](auto topk_constant) {
    launch_fast_specialized<
        N,
        InputT,
        SF,
        HAS_BIAS,
        decltype(topk_constant)::value>(
        q,
        gating,
        weights,
        indices,
        source_rows,
        is_padding,
        bias,
        renormalize,
        routed_scaling_factor,
        num_tokens,
        topk);
  };

  switch (topk) {
    case 4:
      launch(std::integral_constant<int, 4>{});
      break;
    case 8:
      launch(std::integral_constant<int, 8>{});
      break;
    default:
      launch(std::integral_constant<int, 0>{});
      break;
  }
}

template <
    int N,
    typename InputT,
    typename IndexType,
    ScoringFunc SF,
    bool HAS_BIAS>
void launch_static(
    sycl::queue& q,
    const InputT* gating,
    float* weights,
    IndexType* indices,
    int* source_rows,
    const bool* is_padding,
    const float* bias,
    bool renormalize,
    double routed_scaling_factor,
    int64_t num_tokens,
    int topk) {
  using Policy = KernelPolicy<SF, InputT, HAS_BIAS>;
  constexpr int kTokensPerWg = kWgSize / kSgSize;
  const size_t groups =
      static_cast<size_t>((num_tokens + kTokensPerWg - 1) / kTokensPerWg);
  q.parallel_for(
      sycl::nd_range<1>{
          sycl::range<1>{groups * kWgSize}, sycl::range<1>{kWgSize}},
      FallbackTopK<N, IndexType, Policy>{
          gating,
          weights,
          indices,
          source_rows,
          is_padding,
          Policy{bias, renormalize},
          routed_scaling_factor,
          num_tokens,
          topk});
}

//**************************Layer 4: dispatch****************************

template <typename InputT, typename IndexType, ScoringFunc SF, bool HAS_BIAS>
bool dispatch_static_experts(
    sycl::queue& q,
    const InputT* gating,
    float* weights,
    IndexType* indices,
    int* source_rows,
    const bool* is_padding,
    const float* bias,
    bool renormalize,
    double routed_scaling_factor,
    int64_t num_tokens,
    int num_experts,
    int topk) {
#define LAUNCH_STATIC(N)                             \
  launch_static<N, InputT, IndexType, SF, HAS_BIAS>( \
      q,                                             \
      gating,                                        \
      weights,                                       \
      indices,                                       \
      source_rows,                                   \
      is_padding,                                    \
      bias,                                          \
      renormalize,                                   \
      routed_scaling_factor,                         \
      num_tokens,                                    \
      topk);                                         \
  return true

  switch (num_experts) {
    case 1:
      LAUNCH_STATIC(1);
    case 2:
      LAUNCH_STATIC(2);
    case 4:
      LAUNCH_STATIC(4);
    case 8:
      LAUNCH_STATIC(8);
    case 16:
      LAUNCH_STATIC(16);
    case 32:
      LAUNCH_STATIC(32);
    case 64:
      LAUNCH_STATIC(64);
    case 128:
      LAUNCH_STATIC(128);
    case 192:
      LAUNCH_STATIC(192);
    case 256:
      LAUNCH_STATIC(256);
    case 320:
      LAUNCH_STATIC(320);
    case 384:
      LAUNCH_STATIC(384);
    case 448:
      LAUNCH_STATIC(448);
    case 512:
      LAUNCH_STATIC(512);
    case 576:
      LAUNCH_STATIC(576);
    case 1024:
      LAUNCH_STATIC(1024);
    default:
      return false;
  }

#undef LAUNCH_STATIC
}

template <typename InputT, typename IndexType, ScoringFunc SF, bool HAS_BIAS>
bool dispatch_experts_topk(
    sycl::queue& q,
    const InputT* gating,
    float* weights,
    IndexType* indices,
    int* source_rows,
    const bool* is_padding,
    const float* bias,
    const bool renormalize,
    const double routed_scaling_factor,
    int64_t num_tokens,
    int num_experts,
    int topk) {
  if (topk > kMaxTopK) return false;

  const bool aligned = (reinterpret_cast<uintptr_t>(gating) % 16 == 0) &&
                       (reinterpret_cast<uintptr_t>(weights) % 16 == 0) &&
                       (reinterpret_cast<uintptr_t>(indices) % 16 == 0);
  if (!aligned) return false;

#define LAUNCH_FAST(N)                  \
  launch_fast<N, InputT, SF, HAS_BIAS>( \
      q,                                \
      gating,                           \
      weights,                          \
      indices,                          \
      source_rows,                      \
      is_padding,                       \
      bias,                             \
      renormalize,                      \
      routed_scaling_factor,            \
      num_tokens,                       \
      topk);                            \
  return true

  switch (num_experts) {
    case 4:
      LAUNCH_FAST(4);
    case 8:
      LAUNCH_FAST(8);
    case 16:
      LAUNCH_FAST(16);
    case 32:
      LAUNCH_FAST(32);
    case 64:
      LAUNCH_FAST(64);
    case 128:
      LAUNCH_FAST(128);
    case 192:
      LAUNCH_FAST(192);
    case 256:
      LAUNCH_FAST(256);
    case 320:
      LAUNCH_FAST(320);
    case 384:
      LAUNCH_FAST(384);
    case 448:
      LAUNCH_FAST(448);
    case 512:
      LAUNCH_FAST(512);
    case 576:
      LAUNCH_FAST(576);
    case 1024:
      LAUNCH_FAST(1024);
    default:
      return false;
  }

#undef LAUNCH_FAST
}

template <typename InputT, typename IndexType, ScoringFunc SF, bool HAS_BIAS>
void dispatch_topk_all(
    sycl::queue& q,
    const InputT* gating,
    float* weights,
    IndexType* indices,
    int* source_rows,
    const bool* is_padding,
    const float* bias,
    bool renormalize,
    double routed_scaling_factor,
    int64_t num_tokens,
    int num_experts,
    int topk) {
  if constexpr (std::is_same_v<IndexType, int>) {
    const bool aligned = (reinterpret_cast<uintptr_t>(gating) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(weights) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(indices) % 16 == 0);
    if (aligned && dispatch_experts_topk<InputT, IndexType, SF, HAS_BIAS>(
                       q,
                       gating,
                       weights,
                       indices,
                       source_rows,
                       is_padding,
                       bias,
                       renormalize,
                       routed_scaling_factor,
                       num_tokens,
                       num_experts,
                       topk)) {
      return;
    }
  }

  const bool launched =
      dispatch_static_experts<InputT, IndexType, SF, HAS_BIAS>(
          q,
          gating,
          weights,
          indices,
          source_rows,
          is_padding,
          bias,
          renormalize,
          routed_scaling_factor,
          num_tokens,
          num_experts,
          topk);
  TORCH_CHECK(launched, "topk: unsupported num_experts: ", num_experts);
}

}  // namespace topk

}  // namespace moe
}  // namespace vllm

template <typename InputT, typename IndexType, vllm::moe::ScoringFunc SF>
static void dispatch_topk_typed(
    sycl::queue& queue,
    const InputT* gating,
    float* weights,
    IndexType* indices,
    int* source_rows,
    const bool* is_padding,
    const float* bias,
    bool renormalize,
    double routed_scaling_factor,
    int64_t num_tokens,
    int num_experts,
    int topk) {
  if (bias != nullptr) {
    vllm::moe::topk::dispatch_topk_all<InputT, IndexType, SF, true>(
        queue,
        gating,
        weights,
        indices,
        source_rows,
        is_padding,
        bias,
        renormalize,
        routed_scaling_factor,
        num_tokens,
        num_experts,
        topk);
  } else {
    vllm::moe::topk::dispatch_topk_all<InputT, IndexType, SF, false>(
        queue,
        gating,
        weights,
        indices,
        source_rows,
        is_padding,
        nullptr,
        renormalize,
        routed_scaling_factor,
        num_tokens,
        num_experts,
        topk);
  }
}

// check function
static void check_is_padding(
    const std::optional<torch::Tensor>& is_padding, int64_t num_tokens) {
  if (!is_padding.has_value()) {
    return;
  }
  const torch::Tensor& is_padding_tensor = is_padding.value();
  TORCH_CHECK(
      is_padding_tensor.scalar_type() == at::ScalarType::Bool,
      "is_padding tensor must be bool");
  TORCH_CHECK(is_padding_tensor.dim() == 1, "is_padding tensor must be 1D");
  TORCH_CHECK(
      is_padding_tensor.size(0) == num_tokens,
      "is_padding size mismatch, expected: ",
      num_tokens);
  TORCH_CHECK(
      is_padding_tensor.is_contiguous(),
      "is_padding tensor must be contiguous");
}

static void check_topk_inputs(
    const char* op,
    const torch::Tensor& topk_weights,
    const torch::Tensor& topk_indices,
    const torch::Tensor& token_expert_indices,
    const torch::Tensor& gating_output,
    const std::optional<torch::Tensor>& bias,
    int64_t num_tokens,
    int num_experts,
    int topk) {
  TORCH_CHECK(
      gating_output.dim() >= 1 && gating_output.is_contiguous(),
      op,
      ": gating_output must be contiguous");
  TORCH_CHECK(
      topk_weights.scalar_type() == torch::kFloat &&
          topk_weights.is_contiguous(),
      op,
      ": topk_weights must be contiguous float32");
  TORCH_CHECK(
      token_expert_indices.scalar_type() == torch::kInt &&
          token_expert_indices.is_contiguous(),
      op,
      ": token_expert_indices must be contiguous int32");
  TORCH_CHECK(
      topk_indices.is_contiguous(), op, ": topk_indices must be contiguous");
  TORCH_CHECK(
      topk_weights.numel() == num_tokens * topk &&
          topk_indices.numel() == num_tokens * topk &&
          token_expert_indices.numel() == num_tokens * topk,
      op,
      ": output shape mismatch");
  TORCH_CHECK(
      topk > 0 && topk < num_experts,
      op,
      ": topk must be smaller than num_experts");
  if (bias.has_value()) {
    TORCH_CHECK(
        bias->scalar_type() == torch::kFloat && bias->dim() == 1 &&
            bias->size(0) == num_experts && bias->is_contiguous(),
        op,
        ": bias must be contiguous float32 [num_experts]");
  }
}
// Distribute according to the input dtype
template <vllm::moe::ScoringFunc SF>
static void dispatch_topk_inputs(
    sycl::queue& queue,
    const torch::Tensor& gating_output,
    torch::Tensor& topk_weights,
    torch::Tensor& topk_indices,
    torch::Tensor& token_expert_indices,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& is_padding,
    bool renormalize,
    double routed_scaling_factor,
    int64_t num_tokens,
    int num_experts,
    int topk) {
  const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;
  const bool* padding_ptr =
      is_padding.has_value() ? is_padding->data_ptr<bool>() : nullptr;
  float* weights_ptr = topk_weights.data_ptr<float>();
  int* source_rows_ptr = token_expert_indices.data_ptr<int>();

  const void* gating_ptr = gating_output.const_data_ptr();
  void* indices_ptr = topk_indices.mutable_data_ptr();

#define DISPATCH_INDEX(INDEX_T)                     \
  dispatch_topk_typed<INPUT_T, INDEX_T, SF>(        \
      queue,                                        \
      reinterpret_cast<const INPUT_T*>(gating_ptr), \
      weights_ptr,                                  \
      reinterpret_cast<INDEX_T*>(indices_ptr),      \
      source_rows_ptr,                              \
      padding_ptr,                                  \
      bias_ptr,                                     \
      renormalize,                                  \
      routed_scaling_factor,                        \
      num_tokens,                                   \
      num_experts,                                  \
      topk)

#define DISPATCH_INPUT(INDEX_T)                                \
  if (gating_output.scalar_type() == torch::kFloat) {          \
    using INPUT_T = float;                                     \
    DISPATCH_INDEX(INDEX_T);                                   \
  } else if (gating_output.scalar_type() == torch::kFloat16) { \
    using INPUT_T = sycl::half;                                \
    DISPATCH_INDEX(INDEX_T);                                   \
  } else {                                                     \
    using INPUT_T = sycl::ext::oneapi::bfloat16;               \
    DISPATCH_INDEX(INDEX_T);                                   \
  }

  if (topk_indices.scalar_type() == torch::kInt32) {
    DISPATCH_INPUT(int32_t);
  } else if (topk_indices.scalar_type() == torch::kUInt32) {
    DISPATCH_INPUT(uint32_t);
  } else {
    DISPATCH_INPUT(int64_t);
  }

#undef DISPATCH_INPUT
#undef DISPATCH_INDEX
}

void topk_softmax(
    torch::Tensor& topk_weights,          // [num_tokens, topk]
    torch::Tensor& topk_indices,          // [num_tokens, topk]
    torch::Tensor& token_expert_indices,  // [num_tokens, topk]
    torch::Tensor& gating_output,         // [num_tokens, num_experts]
    const bool renormalize,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> is_padding) {
  constexpr double routed_scaling_factor = 1.0;
  const int num_experts = gating_output.size(-1);
  const auto num_tokens = gating_output.numel() / num_experts;
  const int topk = topk_weights.size(-1);
  check_is_padding(is_padding, num_tokens);
  check_topk_inputs(
      "topk_softmax",
      topk_weights,
      topk_indices,
      token_expert_indices,
      gating_output,
      bias,
      num_tokens,
      num_experts,
      topk);
  const at::DeviceGuard device_guard(gating_output.device());
  auto& queue = vllm::xpu::vllmGetQueue();
  dispatch_topk_inputs<vllm::moe::ScoringFunc::SOFTMAX>(
      queue,
      gating_output,
      topk_weights,
      topk_indices,
      token_expert_indices,
      bias,
      is_padding,
      renormalize,
      routed_scaling_factor,
      num_tokens,
      num_experts,
      topk);
}

void topk_sigmoid(
    torch::Tensor& topk_weights,          // [num_tokens, topk]
    torch::Tensor& topk_indices,          // [num_tokens, topk]
    torch::Tensor& token_expert_indices,  // [num_tokens, topk]
    torch::Tensor& gating_output,         // [num_tokens, num_experts]
    const bool renormalize,
    std::optional<torch::Tensor> bias,
    const double routed_scaling_factor,
    std::optional<torch::Tensor> is_padding) {
  const int num_experts = gating_output.size(-1);
  const auto num_tokens = gating_output.numel() / num_experts;
  const int topk = topk_weights.size(-1);
  check_is_padding(is_padding, num_tokens);
  check_topk_inputs(
      "topk_sigmoid",
      topk_weights,
      topk_indices,
      token_expert_indices,
      gating_output,
      bias,
      num_tokens,
      num_experts,
      topk);

  const at::DeviceGuard device_guard(gating_output.device());
  auto& queue = vllm::xpu::vllmGetQueue();
  dispatch_topk_inputs<vllm::moe::ScoringFunc::SIGMOID>(
      queue,
      gating_output,
      topk_weights,
      topk_indices,
      token_expert_indices,
      bias,
      is_padding,
      renormalize,
      routed_scaling_factor,
      num_tokens,
      num_experts,
      topk);
}