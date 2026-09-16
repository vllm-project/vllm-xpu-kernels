#include <sycl/sycl.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

#include "../utils.h"
#include "../dispatch_utils.h"

#if defined(__clang__)
  #pragma clang diagnostic ignored "-Wpass-failed"
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
  #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#ifndef MST_SG
  #define MST_SG 16
#endif
#ifndef MST_WG
  #define MST_WG 64
#endif
#ifndef MST_VPL
  #define MST_VPL 64
#endif
#ifndef MST_MIN_LANES
  #define MST_MIN_LANES 4
#endif
#ifndef MST_S
  #define MST_S 8
#endif
#ifndef MST_SSG
  #define MST_SSG 16
#endif
#ifndef MST_SMALL_T
  #define MST_SMALL_T 196608
#endif
#ifndef MST_MID_T
  #define MST_MID_T 4194304
#endif
#ifndef MST_VPL_MID
  #define MST_VPL_MID 16
#endif

namespace vllm {
namespace moe {
enum class ScoringFunc {
  SOFTMAX = 0,
  SIGMOID = 1,
};

namespace topk {

constexpr int kSgSize = MST_SG;
constexpr int kSSgSize = MST_SSG;
constexpr int kWgSize = MST_WG;
constexpr int kMaxTopK = 8;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

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

constexpr int pow2_floor(int l) {
  int p = 1;
  while (p * 2 <= l)
    p *= 2;
  return p;
}

constexpr int lanes_for_vpl(int n, int vpl) {
  int lanes = n / vpl;

  if (lanes < MST_MIN_LANES) lanes = MST_MIN_LANES;
  if (lanes > kSgSize) lanes = kSgSize;
  if (lanes > n / 4) lanes = n / 4;

  lanes = pow2_floor(lanes);

  while (lanes > 1 && (n / lanes) % 4 != 0)
    lanes >>= 1;

  return lanes;
}

constexpr int lanes_for(int n) { return lanes_for_vpl(n, MST_VPL); }
constexpr int lanes_for_mid(int n) { return lanes_for_vpl(n, MST_VPL_MID); }

constexpr int small_lanes_for(int n) {
  int l = n / 4;
  if (l > kSSgSize) l = kSSgSize;
  if (l < 1) l = 1;
  return pow2_floor(l);
}

constexpr int groups_per_chunk(int kg) {
  int gpc = MST_S / 4;
  if (gpc < 1) gpc = 1;
  while (gpc > 1 && (kg % gpc != 0 || kg / gpc < 2))
    gpc >>= 1;
  return gpc;
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

// Selection-value helper: what score participates in the argmax for element
// `val` (raw logit) at column `idx`.
//   SOFTMAX, no bias : raw logit (monotonic with the softmax prob)
//   SOFTMAX, bias    : softmax prob + bias  (needs row_max and 1/Z)
//   SIGMOID, no bias : raw logit (monotonic with sigmoid)
//   SIGMOID, bias    : sigmoid(logit) + bias
template <ScoringFunc SF, bool HAS_BIAS, typename T>
static inline float selection_value(
    T val, int idx, float row_max, float prob_scale, const float* bias) {
  if constexpr (SF == ScoringFunc::SIGMOID) {
    return sigmoid_typed(val) + (HAS_BIAS ? bias[idx] : 0.0f);
  } else if constexpr (SF == ScoringFunc::SOFTMAX && HAS_BIAS) {
    return sycl::native::exp(static_cast<float>(val) - row_max) * prob_scale +
           bias[idx];
  } else {
    return static_cast<float>(val);
  }
}

// Weight written for the consensus winner (ws attained at wi).
template <ScoringFunc SF, bool HAS_BIAS>
static inline float winner_weight(float ws, float bias_wi) {
  if constexpr (SF == ScoringFunc::SIGMOID && !HAS_BIAS) {
    return ws;
  } else if constexpr (HAS_BIAS) {
    return ws - bias_wi;  // softmax prob resp. sigmoid value at wi
  } else {
    return ws;  // caller applies exp(ws - row_max) [* 1/Z]
  }
}

template <int N, typename InputT, int LANES, ScoringFunc SF, bool HAS_BIAS>
struct MoeSoftmaxTopk {
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
  const float* __restrict__ bias;
  const bool renormalize;
  const double routed_scaling_factor;
  int64_t num_tokens;
  int topk;

  // Full-row normalizer Z = sum(exp(logit - row_max)) reduced over the lane
  // group; returns 1/Z. Only called when the true softmax prob is needed.
  float compute_inv_z(
      const InputT* __restrict__ row,
      const sycl::sub_group& sg,
      float row_max) const {
    float zs = 0.0f;
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
#pragma unroll
      for (int g = 0; g < kGpc; ++g) {
        const int i0 = ((c * kGpc + g) * LANES + lane_of(sg)) * 4;
        const T4<InputT> x = ld4(row + i0);
#pragma unroll
        for (int e = 0; e < 4; ++e)
          zs += sycl::native::exp(x.v[e] - row_max);
      }
    }
    if constexpr (LANES > 1) zs = lane_group_sum<LANES>(sg, zs);
    return ieee_div(1.0f, zs);
  }

  static int lane_of(const sycl::sub_group& sg) {
    return static_cast<int>(sg.get_local_id()[0]) & (LANES > 1 ? LANES - 1 : 0);
  }

  [[sycl::reqd_sub_group_size(kSgSize)]] void
  operator()(sycl::nd_item<1> it) const {
    const sycl::sub_group sg = it.get_sub_group();
    const int64_t gid = static_cast<int64_t>(it.get_global_linear_id());
    const int lane = LANES == 1 ? 0 : static_cast<int>(gid & (LANES - 1));
    const int64_t token = LANES == 1 ? gid : gid / LANES;
    const bool active = token < num_tokens;
    const InputT* __restrict__ row = gating + (active ? token : 0) * N;

    // Pass A: per-chunk selection maxes cs[c]/ci[c]. For SOFTMAX+bias the
    // selection value needs the row normalizers, so this pass only tracks the
    // raw chunk maxes rcs[c] and the selection maxes are recomputed in a third
    // pass once M and 1/Z are known.
    float cs[kChunks];
    int ci[kChunks];
    float rcs[kChunks];
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      float bs = kNegInf;
      int bi = N;
      float rs = kNegInf;
#pragma unroll
      for (int g = 0; g < kGpc; ++g) {
        const int i0 = ((c * kGpc + g) * LANES + lane) * 4;
        const T4<InputT> x = ld4(row + i0);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float val = static_cast<float>(x.v[e]);
          float sel = val;
          if constexpr (SF == ScoringFunc::SIGMOID)
            sel = sigmoid_typed(x.v[e]) + (HAS_BIAS ? bias[i0 + e] : 0.0f);
          const bool take = sel > bs;
          bs = take ? sel : bs;
          bi = take ? i0 + e : bi;
          if constexpr (SF == ScoringFunc::SOFTMAX && HAS_BIAS)
            rs = sycl::fmax(rs, val);
        }
      }
      cs[c] = bs;
      ci[c] = bi;
      rcs[c] = rs;
    }

    float row_max = 0.0f;
    float prob_scale = 1.0f;  // 1/Z when true softmax probs are needed
    if constexpr (SF == ScoringFunc::SOFTMAX && HAS_BIAS) {
      float rm = rcs[0];
#pragma unroll
      for (int c = 1; c < kChunks; ++c)
        rm = sycl::fmax(rm, rcs[c]);
      if constexpr (LANES > 1) rm = lane_group_max<LANES>(sg, rm);
      row_max = rm;
      // Pass B: Z for the prob selection values.
      prob_scale = compute_inv_z(row, sg, row_max);
      // Pass C: selection maxes over softmax prob + bias.
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
                sycl::native::exp(x.v[e] - row_max) * prob_scale + bias[i0 + e];
            const bool take = sel > bs;
            bs = take ? sel : bs;
            bi = take ? i0 + e : bi;
          }
        }
        cs[c] = bs;
        ci[c] = bi;
      }
    }

    float num[kMaxTopK];
    int wid[kMaxTopK];
    float sum = 0.0f;
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

      float w = winner_weight<SF, HAS_BIAS>(ws, HAS_BIAS ? bias[wi] : 0.0f);
      if constexpr (SF == ScoringFunc::SOFTMAX && !HAS_BIAS) {
        if (k == 0) row_max = ws;
        if (k == 0 && !renormalize)
          prob_scale = compute_inv_z(row, sg, row_max);
        w = sycl::native::exp(ws - row_max) * prob_scale;
      }
      num[k] = w;
      sum += num[k];
      wid[k] = wi;

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
            if (idx == wi) continue;  // exclude the winner by identity
            const float sel = selection_value<SF, HAS_BIAS>(
                x.v[e], idx, row_max, prob_scale, bias);
            const bool take = sel > ns;
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
      float scale = static_cast<float>(routed_scaling_factor);
      if (renormalize) {
        const float denom = sum > 0.0f ? sum : 1.0f;
        scale = ieee_div(scale, denom);
      }
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
    }
  }
};

template <int N, typename InputT, int LANES, ScoringFunc SF, bool HAS_BIAS>
struct MoeSoftmaxTopkSmall {
  static constexpr int kVpl = N / LANES;
  static constexpr int kG = kVpl / 4;

  static_assert(kG >= 1, "small kernel needs at least one float4 per lane");

  const InputT* __restrict__ gating;
  float* __restrict__ weights;
  int* __restrict__ indices;
  int* __restrict__ source_rows;
  const bool* __restrict__ is_padding;
  const float* __restrict__ bias;
  const bool renormalize;
  const double routed_scaling_factor;
  int64_t num_tokens;
  int topk;

  [[sycl::reqd_sub_group_size(kSSgSize)]] void
  operator()(sycl::nd_item<1> it) const {
    const sycl::sub_group sg = it.get_sub_group();
    const int64_t gid = static_cast<int64_t>(it.get_global_linear_id());
    const int lane = LANES == 1 ? 0 : static_cast<int>(gid & (LANES - 1));
    const int64_t token = LANES == 1 ? gid : gid / LANES;
    const bool active = token < num_tokens;
    const InputT* __restrict__ row = gating + (active ? token : 0) * N;

    T4<InputT> v[kG];
#pragma unroll
    for (int g = 0; g < kG; ++g)
      v[g] = ld4(row + (g * LANES + lane) * 4);

    if constexpr (SF == ScoringFunc::SIGMOID && !HAS_BIAS) {
#pragma unroll
      for (int g = 0; g < kG; ++g)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          v[g].v[e] = static_cast<InputT>(sigmoid_typed(v[g].v[e]));
    }

    // Materialize biased selection values in place. Once selection scores are
    // formed, the original probability is recoverable as score - bias, so a
    // second register array only increases GRF pressure.
    if constexpr (HAS_BIAS) {
      if constexpr (SF == ScoringFunc::SOFTMAX) {
        float rm = v[0].v[0];
#pragma unroll
        for (int g = 0; g < kG; ++g)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            rm = sycl::fmax(rm, static_cast<float>(v[g].v[e]));
        if constexpr (LANES > 1) rm = lane_group_max<LANES>(sg, rm);
        float zs = 0.0f;
#pragma unroll
        for (int g = 0; g < kG; ++g)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            zs += sycl::native::exp(static_cast<float>(v[g].v[e]) - rm);
        if constexpr (LANES > 1) zs = lane_group_sum<LANES>(sg, zs);
        const float iz = ieee_div(1.0f, zs);
#pragma unroll
        for (int g = 0; g < kG; ++g)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            v[g].v[e] = static_cast<InputT>(
                sycl::native::exp(static_cast<float>(v[g].v[e]) - rm) * iz +
                bias[(g * LANES + lane) * 4 + e]);
      } else {
#pragma unroll
        for (int g = 0; g < kG; ++g)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            v[g].v[e] = static_cast<InputT>(
                sigmoid_typed(v[g].v[e]) + bias[(g * LANES + lane) * 4 + e]);
      }
    }

    float num[kMaxTopK];
    int wid[kMaxTopK];
    float row_max = 0.0f;
    float prob_scale = 1.0f;  // 1/Z when renormalize is false
    float sum = 0.0f;
    for (int k = 0; k < topk; ++k) {
      float acc[4];
#pragma unroll
      for (int e = 0; e < 4; ++e)
        acc[e] = v[0].v[e];
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

      float w = winner_weight<SF, HAS_BIAS>(ws, HAS_BIAS ? bias[wi] : 0.0f);
      if constexpr (SF == ScoringFunc::SOFTMAX && !HAS_BIAS) {
        if (k == 0) row_max = ws;
        if (k == 0 && !renormalize) {
          // Full-row Z from the register copy.
          float zs = 0.0f;
#pragma unroll
          for (int g = 0; g < kG; ++g)
#pragma unroll
            for (int e = 0; e < 4; ++e)
              zs += sycl::native::exp(static_cast<float>(v[g].v[e]) - row_max);
          if constexpr (LANES > 1) zs = lane_group_sum<LANES>(sg, zs);
          prob_scale = ieee_div(1.0f, zs);
        }
        w = sycl::native::exp(ws - row_max) * prob_scale;
      }
      num[k] = w;
      sum += num[k];
      wid[k] = wi;

#pragma unroll
      for (int g = 0; g < kG; ++g) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int idx = (g * LANES + lane) * 4 + e;
          if (idx == wi) v[g].v[e] = kNegInf;
        }
      }
    }

    if (active & (lane == 0)) {
      const bool is_pad = is_padding != nullptr && is_padding[token];
      float scale = static_cast<float>(routed_scaling_factor);
      if (renormalize) {
        const float denom = sum > 0.0f ? sum : 1.0f;
        scale = ieee_div(scale, denom);
      }
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
    }
  }
};

template <int N, typename InputT, int LANES, ScoringFunc SF, bool HAS_BIAS>
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
  constexpr int kTokensPerWg = kWgSize / LANES;
  const size_t groups =
      static_cast<size_t>((num_tokens + kTokensPerWg - 1) / kTokensPerWg);
  q.parallel_for(
      sycl::nd_range<1>{
          sycl::range<1>{groups * kWgSize}, sycl::range<1>{kWgSize}},
      MoeSoftmaxTopk<N, InputT, LANES, SF, HAS_BIAS>{
          gating,
          weights,
          indices,
          source_rows,
          is_padding,
          bias,
          renormalize,
          routed_scaling_factor,
          num_tokens,
          topk});
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
  const int64_t elems = num_tokens * N;

  if constexpr (!HAS_BIAS) {
    if (elems <= MST_SMALL_T) {
      constexpr int kSLanes = small_lanes_for(N);
      constexpr int kSTokensPerWg = kWgSize / kSLanes;
      const size_t groups =
          static_cast<size_t>((num_tokens + kSTokensPerWg - 1) / kSTokensPerWg);
      q.parallel_for(
          sycl::nd_range<1>{
              sycl::range<1>{groups * kWgSize}, sycl::range<1>{kWgSize}},
          MoeSoftmaxTopkSmall<N, InputT, kSLanes, SF, HAS_BIAS>{
              gating,
              weights,
              indices,
              source_rows,
              is_padding,
              bias,
              renormalize,
              routed_scaling_factor,
              num_tokens,
              topk});
      return;
    }
  }

  if (elems <= MST_MID_T) {
    launch_chunked<N, InputT, lanes_for_mid(N), SF, HAS_BIAS>(
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

  launch_chunked<N, InputT, lanes_for(N), SF, HAS_BIAS>(
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

template <
    int N,
    typename InputT,
    typename IndexT,
    ScoringFunc SF,
    bool HAS_BIAS>
struct MoeTopkStatic {
  const InputT* __restrict__ gating;
  float* __restrict__ weights;
  IndexT* __restrict__ indices;
  int* __restrict__ source_rows;
  const bool* __restrict__ is_padding;
  const float* __restrict__ bias;
  const bool renormalize;
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

    float row_max = 0.0f;
    float prob_scale = 1.0f;
    if constexpr (SF == ScoringFunc::SOFTMAX) {
      float local_max = kNegInf;
      for (int i = lane; i < N; i += kSgSize)
        local_max = sycl::fmax(local_max, static_cast<float>(row[i]));
      row_max = lane_group_max<kSgSize>(sg, local_max);

      if constexpr (HAS_BIAS) {
        float local_sum = 0.0f;
        for (int i = lane; i < N; i += kSgSize)
          local_sum += sycl::native::exp(static_cast<float>(row[i]) - row_max);
        prob_scale = ieee_div(1.0f, lane_group_sum<kSgSize>(sg, local_sum));
      } else if (!renormalize) {
        float local_sum = 0.0f;
        for (int i = lane; i < N; i += kSgSize)
          local_sum += sycl::native::exp(static_cast<float>(row[i]) - row_max);
        prob_scale = ieee_div(1.0f, lane_group_sum<kSgSize>(sg, local_sum));
      }
    }

    float sum = 0.0f;
    float previous_score = std::numeric_limits<float>::infinity();
    int previous_index = -1;
    for (int k = 0; k < topk; ++k) {
      float local_score = kNegInf;
      int local_index = N;
      for (int i = lane; i < N; i += kSgSize) {
        const float value = static_cast<float>(row[i]);
        float score;
        if constexpr (SF == ScoringFunc::SIGMOID) {
          const float sigmoid_value = sigmoid_typed(row[i]);
          score = sigmoid_value + (HAS_BIAS ? bias[i] : 0.0f);
        } else {
          score = selection_value<SF, HAS_BIAS>(
              value, i, row_max, prob_scale, bias);
        }
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
      const float value = static_cast<float>(row[local_index]);
      float weight;
      if constexpr (SF == ScoringFunc::SOFTMAX) {
        weight = sycl::native::exp(value - row_max) * prob_scale;
      } else {
        weight = sigmoid_typed(row[local_index]);
      }
      sum += weight;

      if (active && lane == 0) {
        const int64_t out = token * topk + k;
        weights[out] = is_pad ? 0.0f : weight;
        indices[out] =
            is_pad ? static_cast<IndexT>(-1) : static_cast<IndexT>(local_index);
        source_rows[out] = static_cast<int64_t>(k) * num_tokens + token;
      }
      previous_score = local_score;
      previous_index = local_index;
    }

    if (active && lane == 0) {
      float scale = static_cast<float>(routed_scaling_factor);
      if (renormalize) scale = ieee_div(scale, sum > 0.0f ? sum : 1.0f);
      if (!is_pad) {
        for (int k = 0; k < topk; ++k)
          weights[token * topk + k] *= scale;
      }
    }
  }
};

template <
    int N,
    typename InputT,
    typename IndexT,
    ScoringFunc SF,
    bool HAS_BIAS>
void launch_static(
    sycl::queue& q,
    const InputT* gating,
    float* weights,
    IndexT* indices,
    int* source_rows,
    const bool* is_padding,
    const float* bias,
    bool renormalize,
    double routed_scaling_factor,
    int64_t num_tokens,
    int topk) {
  constexpr int kTokensPerWg = kWgSize / kSgSize;
  const size_t groups =
      static_cast<size_t>((num_tokens + kTokensPerWg - 1) / kTokensPerWg);
  q.parallel_for(
      sycl::nd_range<1>{
          sycl::range<1>{groups * kWgSize}, sycl::range<1>{kWgSize}},
      MoeTopkStatic<N, InputT, IndexT, SF, HAS_BIAS>{
          gating,
          weights,
          indices,
          source_rows,
          is_padding,
          bias,
          renormalize,
          routed_scaling_factor,
          num_tokens,
          topk});
}

template <typename InputT, typename IndexT, ScoringFunc SF, bool HAS_BIAS>
bool dispatch_static_experts(
    sycl::queue& q,
    const InputT* gating,
    float* weights,
    IndexT* indices,
    int* source_rows,
    const bool* is_padding,
    const float* bias,
    bool renormalize,
    double routed_scaling_factor,
    int64_t num_tokens,
    int num_experts,
    int topk) {
#define LAUNCH_STATIC(N)                          \
  launch_static<N, InputT, IndexT, SF, HAS_BIAS>( \
      q,                                          \
      gating,                                     \
      weights,                                    \
      indices,                                    \
      source_rows,                                \
      is_padding,                                 \
      bias,                                       \
      renormalize,                                \
      routed_scaling_factor,                      \
      num_tokens,                                 \
      topk);                                      \
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

template <typename InputT, typename IndexT, ScoringFunc SF, bool HAS_BIAS>
bool dispatch_experts_topk(
    sycl::queue& q,
    const InputT* gating,
    float* weights,
    IndexT* indices,
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

template <typename InputT, typename IndexT, ScoringFunc SF, bool HAS_BIAS>
void dispatch_topk_all(
    sycl::queue& q,
    const InputT* gating,
    float* weights,
    IndexT* indices,
    int* source_rows,
    const bool* is_padding,
    const float* bias,
    bool renormalize,
    double routed_scaling_factor,
    int64_t num_tokens,
    int num_experts,
    int topk) {
  if constexpr (std::is_same_v<IndexT, int>) {
    const bool aligned = (reinterpret_cast<uintptr_t>(gating) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(weights) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(indices) % 16 == 0);
    if (aligned && dispatch_experts_topk<InputT, IndexT, SF, HAS_BIAS>(
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

  const bool launched = dispatch_static_experts<InputT, IndexT, SF, HAS_BIAS>(
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

template <typename InputT, typename IndexT, vllm::moe::ScoringFunc SF>
static void dispatch_topk_typed(
    sycl::queue& queue,
    const InputT* gating,
    float* weights,
    IndexT* indices,
    int* source_rows,
    const bool* is_padding,
    const float* bias,
    bool renormalize,
    double routed_scaling_factor,
    int64_t num_tokens,
    int num_experts,
    int topk) {
  if (bias != nullptr) {
    vllm::moe::topk::dispatch_topk_all<InputT, IndexT, SF, true>(
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
    vllm::moe::topk::dispatch_topk_all<InputT, IndexT, SF, false>(
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