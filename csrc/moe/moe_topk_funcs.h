#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <type_traits>

namespace vllm {
namespace moe {
namespace reduce_topk {

static constexpr int kWARP_SIZE = 32;

// ---------------------------------------------------------------------------
// Monotonic bit twiddle for IEEE float, mirrors cub::FloatTraits<float>.
// Maps float -> uint32 such that unsigned ordering matches float ordering.
// ---------------------------------------------------------------------------
[[intel::device_indirectly_callable]] inline uint32_t
twiddle_in_float(float v) {
  uint32_t k = sycl::bit_cast<uint32_t>(v);
  uint32_t mask = (k & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
  return k ^ mask;
}

[[intel::device_indirectly_callable]] inline float
twiddle_out_float(uint32_t k) {
  uint32_t mask = (k & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
  uint32_t b = k ^ mask;
  return sycl::bit_cast<float>(b);
}

// ---------------------------------------------------------------------------
// TopKRedType: packs (value, index) into a single unsigned integer so that a
// warp-wide max reduction yields the largest value, breaking ties toward the
// smaller index. Only the `float` value type is required by the callers in
// this translation unit (scores are pre-converted to float).
// ---------------------------------------------------------------------------
template <typename T_>
struct TopKRedType {
  using T = T_;
  static_assert(
      std::is_same_v<T, float>, "SYCL reduce_topk only implemented for float");

  using TypeCmp = uint64_t;
  static constexpr int kMoveBits = 32;
  static constexpr int kMaxIdx = 65535;
  TypeCmp compVal;

  [[intel::device_indirectly_callable]] static inline TypeCmp
  makeCmpVal(T val, int32_t idx = 0) {
    uint32_t valueBits = twiddle_in_float(val);
    TypeCmp compactTmp = static_cast<TypeCmp>(valueBits);
    // Use 65535 minus idx to give higher priority to smaller indices.
    compactTmp = (compactTmp << kMoveBits) |
                 (0xFFFFu & static_cast<uint32_t>(kMaxIdx - idx));
    return compactTmp;
  }

  [[intel::device_indirectly_callable]] static inline void
  unpack(T& value, int32_t& index, TypeCmp cmp) {
    index = kMaxIdx - static_cast<int32_t>(cmp & 0xFFFF);
    uint32_t valueBits = static_cast<uint32_t>(cmp >> kMoveBits);
    value = twiddle_out_float(valueBits);
  }

  TopKRedType() = default;

  [[intel::device_indirectly_callable]] TopKRedType(T val, int32_t idx)
      : compVal(makeCmpVal(val, idx)) {}

  [[intel::device_indirectly_callable]] operator TypeCmp() const noexcept {
    return compVal;
  }

  // Warp-wide max of the packed (value, index). Implemented with two 32-bit
  // sub_group reductions (max float value, then min index among the winners)
  // instead of a single 64-bit reduce_over_group: 64-bit sub_group reductions
  // hang / are unsupported on this Intel GPU runtime, whereas 32-bit float/int
  // reductions are the proven pattern used elsewhere in this project.
  [[intel::device_indirectly_callable]] inline TypeCmp
  reduce(sycl::sub_group const& sg) {
    float value;
    int32_t index;
    unpack(value, index, compVal);
    float const maxValue =
        sycl::reduce_over_group(sg, value, sycl::maximum<float>());
    int32_t const minIndex = sycl::reduce_over_group(
        sg, value == maxValue ? index : 0x7FFFFFFF, sycl::minimum<int>());
    return makeCmpVal(maxValue, minIndex);
  }
};

// ---------------------------------------------------------------------------
// Compile-time register sort used to pre-sort each lane's candidate array
// before the warp reduction (pure register ops, no shuffles). A descending
// bubble sort on the packed comparison value; N is a small compile-time
// constant so the loops fully unroll.
// ---------------------------------------------------------------------------
template <int N, typename RedType>
struct Sort {
  static_assert(N > 0 && N <= 64, "Sort only supports N in range [1, 64]");

  [[intel::device_indirectly_callable]] static void run(RedType* topK) {
#pragma unroll
    for (int i = 0; i < N - 1; ++i) {
#pragma unroll
      for (int j = 0; j < N - 1; ++j) {
        if (topK[j].compVal < topK[j + 1].compVal) {
          auto tmp = topK[j].compVal;
          topK[j].compVal = topK[j + 1].compVal;
          topK[j + 1].compVal = tmp;
        }
      }
    }
  }
};

template <typename RedType>
struct Sort<1, RedType> {
  [[intel::device_indirectly_callable]] static void run(RedType*) {}
};

// ---------------------------------------------------------------------------
// reduceTopK: single scalar value per lane -> top-`actualK` across the warp.
// ---------------------------------------------------------------------------
template <int K, typename Type>
[[intel::device_indirectly_callable]] inline void reduceTopK(
    sycl::sub_group const& sg,
    Type (&out)[K],
    int32_t (&outIdx)[K],
    Type value,
    int32_t idx,
    Type const minValue,
    int actualK = K) {
  static_assert(K > 0, "Top K must have K > 0");
  static_assert(K < kWARP_SIZE, "Top K must have K < kWARP_SIZE");
  using RedType = TopKRedType<Type>;
  RedType topK{value, idx};
  typename RedType::TypeCmp packedMax{};
  for (int kk = 0; kk < actualK; ++kk) {
    topK = kk > 0 && packedMax == topK.compVal ? RedType{minValue, idx} : topK;
    packedMax = topK.reduce(sg);
    RedType::unpack(out[kk], outIdx[kk], packedMax);
  }
}

// ---------------------------------------------------------------------------
// reduceTopK: `N` candidates per lane -> top-`actualK` across the warp.
// ---------------------------------------------------------------------------
template <int K, typename Type, int N>
[[intel::device_indirectly_callable]] inline void reduceTopK(
    sycl::sub_group const& sg,
    Type (&out)[K],
    int32_t (&outIdx)[K],
    Type (&value)[N],
    int32_t (&idx)[N],
    Type const minValue,
    int actualK = K) {
  static_assert(K > 0, "Top K must have K > 0");
  static_assert(K <= kWARP_SIZE, "Top K must have K <= kWARP_SIZE");
  static_assert(N > 0, "Top K must have N > 0");
  static_assert(
      N <= 64,
      "Only support candidates number less than or equal to "
      "64*32=2048");
  using RedType = TopKRedType<Type>;
  RedType topK[N];
#pragma unroll
  for (int nn = 0; nn < N; ++nn) {
    topK[nn] = RedType{value[nn], idx[nn]};
  }

  Sort<N, RedType>::run(topK);

  typename RedType::TypeCmp packedMax{};
  for (int kk = 0; kk < actualK; ++kk) {
    bool update = kk > 0 && packedMax == topK[0].compVal;
#pragma unroll
    for (int nn = 0; nn < N; ++nn) {
      topK[nn] = update && nn == N - 1 ? RedType{minValue, idx[nn]}
                 : update              ? topK[nn + 1]
                                       : topK[nn];
    }
    packedMax = topK[0].reduce(sg);
    RedType::unpack(out[kk], outIdx[kk], packedMax);
  }
}

// ---------------------------------------------------------------------------
// Lane-owned top-K specialization range (hierarchical per-lane result path).
// ---------------------------------------------------------------------------
template <
    int NumExperts,
    int NumTopExperts,
    int MinExperts,
    int MaxExperts,
    int MinTopExperts,
    int MaxTopExperts>
struct LaneOwnedTopKRange {
  static_assert(MinExperts > 0 && MinExperts <= MaxExperts);
  static_assert(MinTopExperts > 0 && MinTopExperts <= MaxTopExperts);
  static constexpr bool kEnabled =
      NumExperts >= MinExperts && NumExperts <= MaxExperts &&
      NumTopExperts >= MinTopExperts && NumTopExperts <= MaxTopExperts;
};

static constexpr int kHIGH_EXPERT_LANE_OWNED_TOPK_MIN_EXPERTS = 512;
static constexpr int kHIGH_EXPERT_LANE_OWNED_TOPK_MAX_EXPERTS = 1024;
static constexpr int kHIGH_EXPERT_LANE_OWNED_TOPK_MIN_TOP_EXPERTS = 9;
static constexpr int kHIGH_EXPERT_LANE_OWNED_TOPK_MAX_TOP_EXPERTS = 16;

template <int NumExperts, int NumTopExperts>
using HighExpertLaneOwnedTopKRange = LaneOwnedTopKRange<
    NumExperts,
    NumTopExperts,
    kHIGH_EXPERT_LANE_OWNED_TOPK_MIN_EXPERTS,
    kHIGH_EXPERT_LANE_OWNED_TOPK_MAX_EXPERTS,
    kHIGH_EXPERT_LANE_OWNED_TOPK_MIN_TOP_EXPERTS,
    kHIGH_EXPERT_LANE_OWNED_TOPK_MAX_TOP_EXPERTS>;

// ---------------------------------------------------------------------------
// reduceTopKForLane: like reduceTopK (array variant) but each lane keeps only
// its own rank-`laneIdx` winner (used by the hierarchical single-group path).
// ---------------------------------------------------------------------------
template <int K, typename Type, int N>
[[intel::device_indirectly_callable]] inline void reduceTopKForLane(
    sycl::sub_group const& sg,
    Type& out,
    int32_t& outIdx,
    Type (&value)[N],
    int32_t (&idx)[N],
    Type const minValue,
    int32_t laneIdx) {
  static_assert(K > 0, "Top K must have K > 0");
  static_assert(K <= kWARP_SIZE, "Top K must have K <= kWARP_SIZE");
  static_assert(N > 0, "Top K must have N > 0");
  static_assert(
      N <= 64,
      "Only support candidates number less than or equal to "
      "64*32=2048");
  using RedType = TopKRedType<Type>;
  RedType topK[N];
#pragma unroll
  for (int nn = 0; nn < N; ++nn) {
    topK[nn] = RedType{value[nn], idx[nn]};
  }

  Sort<N, RedType>::run(topK);

  typename RedType::TypeCmp packedMax{};
  typename RedType::TypeCmp lanePacked{};
#pragma unroll
  for (int kk = 0; kk < K; ++kk) {
    bool update = kk > 0 && packedMax == topK[0].compVal;
#pragma unroll
    for (int nn = 0; nn < N; ++nn) {
      topK[nn] = update && nn == N - 1 ? RedType{minValue, idx[nn]}
                 : update              ? topK[nn + 1]
                                       : topK[nn];
    }
    packedMax = topK[0].reduce(sg);
    if (laneIdx == kk) {
      lanePacked = packedMax;
    }
  }

  if (laneIdx < K) {
    RedType::unpack(out, outIdx, lanePacked);
  } else {
    out = minValue;
    outIdx = -1;
  }
}

}  // namespace reduce_topk
}  // namespace moe
}  // namespace vllm
