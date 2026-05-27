#include <torch/all.h>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sycl/sycl.hpp>
#include "../dispatch_utils.h"
#include "../utils.h"
namespace vllm {
namespace moe {

constexpr unsigned FULL_WARP_MASK = 0xffffffff;
static constexpr int WARP_SIZE = 32;
static constexpr int NumNemotronExperts = 512;
static constexpr int NumKimiK2Experts = 384;
static constexpr int NumDeepseekExperts = 256;
static constexpr int MaxSupportedExpertCount =
    std::max({NumNemotronExperts, NumKimiK2Experts, NumDeepseekExperts});
static constexpr int MaxNumExpertsUnit = 128;
static constexpr int NumTopGroupScores = 2;
static constexpr int DefaultMaxNumTopExperts = 8;
static constexpr int MaxSupportedTopExperts = 22;
static constexpr int MaxNumTopGroups = 4;
static constexpr int MaxReduceTopK = 32;

enum ScoringFunc : int { SCORING_NONE = 0, SCORING_SIGMOID = 1 };

template <typename T, typename BiasT, typename IdxT, ScoringFunc SF>
class VllmGroupedTopKFusedKernel;

template <
    typename T,
    typename BiasT,
    typename IdxT,
    ScoringFunc SF,
    int MaxNumExperts,
    bool MultiGroups,
    int MaxNumTopExperts = DefaultMaxNumTopExperts>
class VllmGroupedTopKFusedSmallExpertCountKernel;

template <typename T_OUT, typename T_IN>
inline T_OUT sycl_cast(T_IN val) {
  return static_cast<T_OUT>(val);
}

template <typename T>
inline T neg_inf() {
  T out;
  xpu::from_float(out, -std::numeric_limits<float>::infinity());
  return out;
}

template <typename T>
inline bool is_finite(const T val) {
  return std::isfinite(xpu::to_float(val));
}

inline float sigmoid_accurate(float x) {
  return 1.f / (1.f + sycl::native::exp(-x));
}

template <typename T>
inline T apply_sigmoid(T val) {
  float f = xpu::to_float(val);
  T out;
  xpu::from_float(out, sigmoid_accurate(f));
  return out;
}

template <ScoringFunc SF, typename T>
inline T apply_scoring(T val) {
  if constexpr (SF == SCORING_NONE) {
    return val;
  } else if constexpr (SF == SCORING_SIGMOID) {
    return apply_sigmoid(val);
  }
}

namespace reduce_topk {

template <int N_IN, typename T, typename IdxT>
inline void reduceTopK(
    sycl::sub_group subgroup,
    T* out_val,
    IdxT* out_idx,
    const T* in_vals,
    const IdxT* in_idxs,
    T min_val,
    int topk) {
  constexpr IdxT invalid_idx = std::numeric_limits<IdxT>::max();
  bool selected[N_IN] = {false};

  for (int k = 0; k < topk; ++k) {
    T local_best_val = min_val;
    IdxT local_best_idx = invalid_idx;
    int local_best_pos = -1;

#pragma unroll
    for (int i = 0; i < N_IN; ++i) {
      if (selected[i]) {
        continue;
      }
      T cand_val = in_vals[i];
      IdxT cand_idx = in_idxs[i];
      if ((cand_val > local_best_val) ||
          ((cand_val == local_best_val) && (cand_idx < local_best_idx))) {
        local_best_val = cand_val;
        local_best_idx = cand_idx;
        local_best_pos = i;
      }
    }
    float local_best_val_tmp = xpu::to_float(local_best_val);
    float warp_best_val_tmp = sycl::reduce_over_group(
        subgroup, local_best_val_tmp, sycl::maximum<float>());

    T warp_best_val = static_cast<T>(warp_best_val_tmp);
    IdxT warp_best_idx = invalid_idx;

    if (local_best_pos != -1 && local_best_val == warp_best_val) {
      warp_best_idx = local_best_idx;
    }
    warp_best_idx =
        sycl::reduce_over_group(subgroup, warp_best_idx, sycl::minimum<IdxT>());

    bool found = (warp_best_idx != invalid_idx);
    if (found) {
      int insert_pos = k;
      bool still_shifting = true;
#pragma unroll
      for (int shift = 0; shift < MaxReduceTopK - 1; ++shift) {
        int prev_pos = k - shift - 1;
        bool active = shift < k;
        bool should_shift = active && still_shifting &&
                            out_val[prev_pos] == warp_best_val &&
                            out_idx[prev_pos] > warp_best_idx;
        if (should_shift) {
          out_val[prev_pos + 1] = out_val[prev_pos];
          out_idx[prev_pos + 1] = out_idx[prev_pos];
          insert_pos = prev_pos;
        }
        still_shifting = still_shifting && should_shift;
      }
      out_val[insert_pos] = warp_best_val;
      out_idx[insert_pos] = warp_best_idx;
    } else {
      out_val[k] = min_val;
      out_idx[k] = 0;
    }

    if (found && local_best_pos != -1 && local_best_val == warp_best_val &&
        local_best_idx == warp_best_idx) {
      selected[local_best_pos] = true;
    }
  }
}

template <typename T, typename IdxT>
inline void reduceTopK(
    sycl::sub_group subgroup,
    T* out_val,
    IdxT* out_idx,
    T val,
    IdxT idx,
    T min_val,
    int topk) {
  T in_vals[1] = {val};
  IdxT in_idxs[1] = {idx};
  reduceTopK<1>(subgroup, out_val, out_idx, in_vals, in_idxs, min_val, topk);
}

}  // namespace reduce_topk

template <
    typename T,
    typename BiasT,
    typename IdxT,
    ScoringFunc SF,
    int MaxNumExperts,
    bool MultiGroups,
    int MaxNumTopExperts = DefaultMaxNumTopExperts>
SYCL_EXTERNAL inline void grouped_topk_fused_small_expert_count_kernel(
    T* scores,
    float* topkValues,
    IdxT* topkIndices,
    BiasT const* routingBias,
    int64_t const numTokens,
    int64_t const numGroup,
    int64_t const topkGroup,
    int64_t const topk,
    int64_t const numExperts,
    int64_t const numExpertsPerGroup,
    bool const renormalize,
    double const routedScalingFactor,
    sycl::nd_item<1> item) {
  constexpr float invalidScoreFloat = -std::numeric_limits<float>::infinity();

  int threadIdx = item.get_local_id(0);
  int blockIdx = item.get_group(0);
  if constexpr (MultiGroups) {
    if (blockIdx >= numTokens) return;
  }
  int localSize = item.get_local_range(0);
  bool has_bias = (routingBias != nullptr);

  int laneIdx = threadIdx % WARP_SIZE;
  int warpIdx = threadIdx / WARP_SIZE;

  topkValues += blockIdx * topk;
  topkIndices += blockIdx * topk;

  if constexpr (MultiGroups) {
    auto subgroup = item.get_sub_group();
    T* scoresToken = scores + static_cast<int64_t>(blockIdx) * numExperts;
    T selectedGroupScores[WARP_SIZE];
    int32_t selectedGroupIdx[WARP_SIZE];

    T groupScore = neg_inf<T>();
    if (laneIdx < numGroup) {
      int32_t groupOffset = laneIdx * numExpertsPerGroup;
      T largest = neg_inf<T>();
      T secondLargest = neg_inf<T>();

      for (int32_t i = 0; i < numExpertsPerGroup; ++i) {
        T value = apply_scoring<SF>(scoresToken[groupOffset + i]);
        if (has_bias) {
          value = value + sycl_cast<T, BiasT>(routingBias[groupOffset + i]);
        }
        if (value > largest) {
          secondLargest = largest;
          largest = value;
        } else if (value > secondLargest) {
          secondLargest = value;
        }
      }
      groupScore = has_bias ? largest + secondLargest : largest;
    }

    reduce_topk::reduceTopK(
        subgroup,
        selectedGroupScores,
        selectedGroupIdx,
        groupScore,
        laneIdx,
        neg_inf<T>(),
        static_cast<int>(topkGroup));

    bool proceed = false;

    if (topkGroup > 0) {
      proceed = (selectedGroupScores[topkGroup - 1] != neg_inf<T>());
    }

    if (!proceed) {
      for (int i = laneIdx; i < topk; i += WARP_SIZE) {
        topkIndices[i] = static_cast<IdxT>(i);
        topkValues[i] = 1.0f / static_cast<float>(topk);
      }
      return;
    }

    constexpr int MaxExpertCandidatesPerLane = NumDeepseekExperts / WARP_SIZE;
    T localCandidateScores[MaxExpertCandidatesPerLane];
    IdxT localCandidateIdx[MaxExpertCandidatesPerLane];
    T selectedExpertScores[DefaultMaxNumTopExperts];
    IdxT selectedExpertIdx[DefaultMaxNumTopExperts];

    for (int i = 0; i < MaxExpertCandidatesPerLane; ++i) {
      localCandidateScores[i] = neg_inf<T>();
      localCandidateIdx[i] = 0;
    }

    int32_t totalCandidates = topkGroup * numExpertsPerGroup;
    for (int32_t candidate = laneIdx; candidate < totalCandidates;
         candidate += WARP_SIZE) {
      int32_t localSlot = candidate / WARP_SIZE;
      int32_t selectedGroup = candidate / numExpertsPerGroup;
      int32_t expertInGroup = candidate % numExpertsPerGroup;
      int32_t gid = selectedGroupIdx[selectedGroup];
      int32_t idx = gid * numExpertsPerGroup + expertInGroup;
      T candidateScore = neg_inf<T>();

      T input = scoresToken[idx];
      if (is_finite(input)) {
        T score = apply_scoring<SF>(input);
        candidateScore = score;
        if (has_bias) {
          candidateScore =
              candidateScore + sycl_cast<T, BiasT>(routingBias[idx]);
        }
      }

      localCandidateScores[localSlot] = candidateScore;
      localCandidateIdx[localSlot] = static_cast<IdxT>(idx);
    }

    reduce_topk::reduceTopK<MaxExpertCandidatesPerLane>(
        subgroup,
        selectedExpertScores,
        selectedExpertIdx,
        localCandidateScores,
        localCandidateIdx,
        neg_inf<T>(),
        static_cast<int>(topk));

    for (int i = 1; i < topk; ++i) {
      T score = selectedExpertScores[i];
      IdxT idx = selectedExpertIdx[i];
      int j = i;
      while (j > 0 && ((selectedExpertScores[j - 1] < score) ||
                       ((selectedExpertScores[j - 1] == score) &&
                        (selectedExpertIdx[j - 1] > idx)))) {
        selectedExpertScores[j] = selectedExpertScores[j - 1];
        selectedExpertIdx[j] = selectedExpertIdx[j - 1];
        --j;
      }
      selectedExpertScores[j] = score;
      selectedExpertIdx[j] = idx;
    }

    float laneUnbiased = 0.0f;
    IdxT laneIdxOut = 0;
    if (laneIdx < topk) {
      laneIdxOut = selectedExpertIdx[laneIdx];
      T in = scoresToken[static_cast<int32_t>(laneIdxOut)];
      laneUnbiased = xpu::to_float(apply_scoring<SF>(in));
    }

    float scale = static_cast<float>(routedScalingFactor);
    if (renormalize) {
      // Match baseline precision: sum and divide in T precision
      T laneScoreT = static_cast<T>(laneUnbiased);
      T topkSumT = static_cast<T>(0);
      for (int i = 0; i < static_cast<int>(topk); ++i) {
        T val = sycl::select_from_group(subgroup, laneScoreT, i);
        topkSumT = topkSumT + val;
      }
      if (laneIdx < topk) {
        T normalizedT = laneScoreT / topkSumT;
        laneUnbiased = xpu::to_float(normalizedT);
      }
    }

    if (laneIdx < topk) {
      topkIndices[laneIdx] = laneIdxOut;
      topkValues[laneIdx] = laneUnbiased * scale;
    }
    return;
  } else {
    // Single-group path: select top-k from all experts using single warp.
    constexpr int ExpertsPerLane = MaxNumExperts / WARP_SIZE;
    auto subgroup = item.get_sub_group();
    T* scoresToken = scores + static_cast<int64_t>(blockIdx) * numExperts;

    if constexpr (MaxNumExperts <= NumKimiK2Experts) {
      // Direct path: <=384 experts, fits in registers
      T localBiasedScores[ExpertsPerLane];
      IdxT localIdx[ExpertsPerLane];

      for (int i = 0; i < ExpertsPerLane; ++i) {
        int expertId = i * WARP_SIZE + laneIdx;
        if (expertId < numExperts) {
          T raw = scoresToken[expertId];
          T scored = apply_scoring<SF>(raw);
          localBiasedScores[i] =
              has_bias ? (scored + sycl_cast<T, BiasT>(routingBias[expertId]))
                       : scored;
        } else {
          localBiasedScores[i] = neg_inf<T>();
        }
        localIdx[i] = static_cast<IdxT>(expertId);
      }

      T selectedScores[MaxNumTopExperts];
      IdxT selectedIdx[MaxNumTopExperts];
      reduce_topk::reduceTopK<ExpertsPerLane>(
          subgroup,
          selectedScores,
          selectedIdx,
          localBiasedScores,
          localIdx,
          neg_inf<T>(),
          static_cast<int>(topk));

      float unbiasedArr[DefaultMaxNumTopExperts];
      for (int i = 0; i < static_cast<int>(topk); ++i) {
        IdxT idx_i = selectedIdx[i];
        T raw = scoresToken[static_cast<int32_t>(idx_i)];
        T scored = apply_scoring<SF>(raw);
        unbiasedArr[i] = xpu::to_float(scored);
      }

      float laneUnbiased = 0.0f;
      IdxT laneIdxOut = 0;
      if (laneIdx < topk) {
        laneIdxOut = selectedIdx[laneIdx];
        laneUnbiased = unbiasedArr[laneIdx];
      }

      float scale = static_cast<float>(routedScalingFactor);
      if (renormalize) {
        float topk_sum = 0.0f;
        for (int i = 0; i < static_cast<int>(topk); ++i) {
          topk_sum += unbiasedArr[i];
        }
        if (laneIdx < topk) {
          laneUnbiased /= topk_sum;
        }
      }

      if (laneIdx < topk) {
        topkValues[laneIdx] = laneUnbiased * scale;
        topkIndices[laneIdx] = laneIdxOut;
      }
    } else {
      // Large expert path (>256): Each lane streams through its experts,
      // maintaining a local top-k via insertion sort, then iteratively
      // selects global top-k across the warp. Minimal register pressure.
      constexpr int LocalTopK = DefaultMaxNumTopExperts;  // 8
      auto subgroup = item.get_sub_group();
      T* scoresToken2 = scores + static_cast<int64_t>(blockIdx) * numExperts;

      // Each lane maintains its local top-k sorted descending
      float laneTopVal[LocalTopK];
      IdxT laneTopIdx[LocalTopK];
      for (int k = 0; k < LocalTopK; ++k) {
        laneTopVal[k] = -std::numeric_limits<float>::infinity();
        laneTopIdx[k] = std::numeric_limits<IdxT>::max();
      }

      // Stream through all experts assigned to this lane
      for (int i = 0; i < ExpertsPerLane; ++i) {
        int expertId = i * WARP_SIZE + laneIdx;
        if (expertId >= numExperts) break;

        T raw = scoresToken2[expertId];
        T scored = apply_scoring<SF>(raw);
        float val =
            has_bias ? xpu::to_float(
                           scored + sycl_cast<T, BiasT>(routingBias[expertId]))
                     : xpu::to_float(scored);
        IdxT idx = static_cast<IdxT>(expertId);

        // Check if this beats the worst in our local top-k
        if (val > laneTopVal[LocalTopK - 1] ||
            (val == laneTopVal[LocalTopK - 1] &&
             idx < laneTopIdx[LocalTopK - 1])) {
          // Insert into sorted position
          int pos = LocalTopK - 1;
          for (int j = LocalTopK - 2; j >= 0; --j) {
            if (val > laneTopVal[j] ||
                (val == laneTopVal[j] && idx < laneTopIdx[j])) {
              laneTopVal[j + 1] = laneTopVal[j];
              laneTopIdx[j + 1] = laneTopIdx[j];
              pos = j;
            } else {
              break;
            }
          }
          laneTopVal[pos] = val;
          laneTopIdx[pos] = idx;
        }
      }

      // Now iteratively select global top-k from the warp.
      // Each lane offers its current best (top of local sorted list).
      // The warp finds the global max, the winning lane pops it.
      IdxT globalTopIdx[DefaultMaxNumTopExperts];
      int lanePtr = 0;  // points to next candidate in laneTopVal

      for (int k = 0; k < static_cast<int>(topk); ++k) {
        float myVal = (lanePtr < LocalTopK)
                          ? laneTopVal[lanePtr]
                          : -std::numeric_limits<float>::infinity();
        IdxT myIdx = (lanePtr < LocalTopK) ? laneTopIdx[lanePtr]
                                           : std::numeric_limits<IdxT>::max();

        // Find the best value across all lanes
        float bestVal =
            sycl::reduce_over_group(subgroup, myVal, sycl::maximum<float>());

        // Among lanes that have bestVal, pick smallest idx
        IdxT candidateIdx =
            (myVal == bestVal) ? myIdx : std::numeric_limits<IdxT>::max();
        IdxT bestIdx = sycl::reduce_over_group(
            subgroup, candidateIdx, sycl::minimum<IdxT>());

        globalTopIdx[k] = bestIdx;

        // The winning lane advances its pointer
        if (myIdx == bestIdx && myVal == bestVal) {
          lanePtr++;
        }
      }

      // Re-read unbiased scores for winners
      float unbiasedArr[DefaultMaxNumTopExperts];
      for (int i = 0; i < static_cast<int>(topk); ++i) {
        T raw = scoresToken2[static_cast<int32_t>(globalTopIdx[i])];
        T scored = apply_scoring<SF>(raw);
        unbiasedArr[i] = xpu::to_float(scored);
      }

      float laneUnbiased = 0.0f;
      IdxT laneIdxOut = 0;
      if (laneIdx < topk) {
        laneIdxOut = globalTopIdx[laneIdx];
        laneUnbiased = unbiasedArr[laneIdx];
      }

      float scale = static_cast<float>(routedScalingFactor);
      if (renormalize) {
        float topk_sum = 0.0f;
        for (int i = 0; i < static_cast<int>(topk); ++i) {
          topk_sum += unbiasedArr[i];
        }
        if (laneIdx < topk) {
          laneUnbiased /= topk_sum;
        }
      }

      if (laneIdx < topk) {
        topkValues[laneIdx] = laneUnbiased * scale;
        topkIndices[laneIdx] = laneIdxOut;
      }
    }
  }  // end if constexpr (!MultiGroups)
}

template <typename T, typename BiasT, typename IdxT, ScoringFunc SF>
void invokeNoAuxTc(
    T* scores,
    float* topk_values,
    IdxT* topk_indices,
    BiasT const* bias,
    int64_t const num_tokens,
    int64_t const num_experts,
    int64_t const n_group,
    int64_t const topk_group,
    int64_t const topk,
    bool const renormalize,
    double const routed_scaling_factor,
    bool enable_pdl,
    sycl::queue& queue) {
  int64_t experts_per_group = num_experts / n_group;
  bool is_single_group =
      (n_group == 1) && (topk_group == 1) &&
      (num_experts <= MaxSupportedExpertCount) &&
      (topk <= DefaultMaxNumTopExperts || topk == MaxSupportedTopExperts);

#define LAUNCH_SMALL_KERNEL(                                            \
    MAX_EXPERTS, MultiGroups, MAX_TOP_EXPERTS, NUM_THREADS)             \
  do {                                                                  \
    size_t local_size = static_cast<size_t>(NUM_THREADS);               \
    size_t global_size = static_cast<size_t>(num_tokens) * local_size;  \
    queue.submit([&](sycl::handler& cgh) {                              \
      cgh.parallel_for<VllmGroupedTopKFusedSmallExpertCountKernel<      \
          T,                                                            \
          BiasT,                                                        \
          IdxT,                                                         \
          SF,                                                           \
          MAX_EXPERTS,                                                  \
          MultiGroups,                                                  \
          MAX_TOP_EXPERTS>>(                                            \
          sycl::nd_range<1>(                                            \
              sycl::range<1>(global_size), sycl::range<1>(local_size)), \
          [=](sycl::nd_item<1> item)                                    \
              [[sycl::reqd_sub_group_size(WARP_SIZE)]] {                \
                grouped_topk_fused_small_expert_count_kernel<           \
                    T,                                                  \
                    BiasT,                                              \
                    IdxT,                                               \
                    SF,                                                 \
                    MAX_EXPERTS,                                        \
                    MultiGroups,                                        \
                    MAX_TOP_EXPERTS>(                                   \
                    scores,                                             \
                    topk_values,                                        \
                    topk_indices,                                       \
                    bias,                                               \
                    num_tokens,                                         \
                    n_group,                                            \
                    topk_group,                                         \
                    topk,                                               \
                    num_experts,                                        \
                    experts_per_group,                                  \
                    renormalize,                                        \
                    routed_scaling_factor,                              \
                    item);                                              \
              });                                                       \
    });                                                                 \
  } while (0)

  if (is_single_group) {
    if (num_experts == NumNemotronExperts && n_group == 1 &&
        topk == MaxSupportedTopExperts) {
      LAUNCH_SMALL_KERNEL(
          NumNemotronExperts, false, MaxSupportedTopExperts, WARP_SIZE);
    } else if (
        num_experts > NumKimiK2Experts &&
        num_experts <= MaxSupportedExpertCount) {
      LAUNCH_SMALL_KERNEL(
          MaxSupportedExpertCount, false, DefaultMaxNumTopExperts, WARP_SIZE);
    } else if (
        num_experts > MaxNumExpertsUnit && num_experts <= NumKimiK2Experts) {
      LAUNCH_SMALL_KERNEL(
          NumKimiK2Experts, false, DefaultMaxNumTopExperts, WARP_SIZE);
    } else {
      LAUNCH_SMALL_KERNEL(
          MaxNumExpertsUnit, false, DefaultMaxNumTopExperts, WARP_SIZE);
    }
  } else {
    LAUNCH_SMALL_KERNEL(
        NumDeepseekExperts, true, DefaultMaxNumTopExperts, WARP_SIZE);
  }

#undef LAUNCH_SMALL_KERNEL
}

#define INSTANTIATE_NOAUX_TC(T, BiasT, IdxT, SF)   \
  template void invokeNoAuxTc<T, BiasT, IdxT, SF>( \
      T * scores,                                  \
      float* topk_values,                          \
      IdxT* topk_indices,                          \
      BiasT const* bias,                           \
      int64_t const num_tokens,                    \
      int64_t const num_experts,                   \
      int64_t const n_group,                       \
      int64_t const topk_group,                    \
      int64_t const topk,                          \
      bool const renormalize,                      \
      double const routed_scaling_factor,          \
      bool enable_pdl,                             \
      sycl::queue& queue);

INSTANTIATE_NOAUX_TC(float, float, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(float, sycl::half, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(
    float, sycl::ext::oneapi::bfloat16, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(sycl::half, float, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(sycl::half, sycl::half, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(
    sycl::half, sycl::ext::oneapi::bfloat16, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(
    sycl::ext::oneapi::bfloat16, float, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(
    sycl::ext::oneapi::bfloat16, sycl::half, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(
    sycl::ext::oneapi::bfloat16,
    sycl::ext::oneapi::bfloat16,
    int32_t,
    SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(float, float, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(float, sycl::half, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(float, sycl::ext::oneapi::bfloat16, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(sycl::half, float, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(sycl::half, sycl::half, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(
    sycl::half, sycl::ext::oneapi::bfloat16, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(sycl::ext::oneapi::bfloat16, float, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(
    sycl::ext::oneapi::bfloat16, sycl::half, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(
    sycl::ext::oneapi::bfloat16,
    sycl::ext::oneapi::bfloat16,
    int32_t,
    SCORING_NONE);
}  // end namespace moe
}  // namespace vllm

std::tuple<torch::Tensor, torch::Tensor> fused_grouped_topk(
    torch::Tensor const& hidden_states,
    torch::Tensor const& gating_output,
    int64_t const n_topk,
    bool const renormalize,
    int64_t const n_expert_group,
    int64_t const n_topk_group,
    c10::string_view const scoring_func,
    double const routed_scaling_factor,
    c10::optional<torch::Tensor> const& bias) {
  auto data_type = gating_output.scalar_type();
  bool has_bias = bias.has_value() && bias->defined();
  auto bias_type = has_bias ? bias->scalar_type() : torch::kFloat32;
  auto input_size = gating_output.sizes();
  int64_t num_tokens = input_size[0];
  int64_t num_experts = input_size[1];
  int64_t n_group = n_expert_group;
  int64_t topk_group = n_topk_group;
  int64_t topk = n_topk;

  TORCH_CHECK(
      hidden_states.sizes()[0] == gating_output.sizes()[0],
      "Number of tokens mismatch");
  TORCH_CHECK(input_size.size() == 2, "gating_output must be a 2D Tensor");
  TORCH_CHECK(n_group > 0, "n_group must be positive");
  TORCH_CHECK(topk > 0, "topk must be positive");
  TORCH_CHECK(topk_group > 0, "topk_group must be positive");
  TORCH_CHECK(topk_group <= n_group, "topk_group must be <= n_group");
  TORCH_CHECK(
      num_experts % n_group == 0, "num_experts should be divisible by n_group");
  TORCH_CHECK(
      n_group <= 32, "n_group should be smaller than or equal to 32 for now");
  TORCH_CHECK(topk <= 32, "topk should be smaller than or equal to 32 for now");
  TORCH_CHECK(
      topk <= topk_group * (num_experts / n_group),
      "topk must be <= topk_group * (num_experts / n_group)");
  TORCH_CHECK(
      scoring_func == "sigmoid" || scoring_func == "softmax",
      "Unsupported scoring_func: ",
      scoring_func);
  auto const sf = (scoring_func == "sigmoid") ? vllm::moe::SCORING_SIGMOID
                                              : vllm::moe::SCORING_NONE;

  torch::Tensor topk_values = torch::empty(
      {num_tokens, topk},
      torch::dtype(torch::kFloat32).device(gating_output.device()));
  torch::Tensor topk_indices = torch::empty(
      {num_tokens, topk},
      torch::dtype(torch::kInt32).device(gating_output.device()));

  // auto device_idx = gating_output.device().index();
  auto device = gating_output.device();
  auto& stream = vllm::xpu::vllmGetQueue(device.index());

#define LAUNCH_KERNEL_SF(T, BiasT, IdxT)                                      \
  do {                                                                        \
    switch (sf) {                                                             \
      case vllm::moe::SCORING_NONE:                                           \
        vllm::moe::invokeNoAuxTc<T, BiasT, IdxT, vllm::moe::SCORING_NONE>(    \
            reinterpret_cast<T*>(gating_output.mutable_data_ptr()),           \
            reinterpret_cast<float*>(topk_values.mutable_data_ptr()),         \
            reinterpret_cast<IdxT*>(topk_indices.mutable_data_ptr()),         \
            (has_bias ? reinterpret_cast<BiasT const*>(bias->data_ptr())      \
                      : nullptr),                                             \
            num_tokens,                                                       \
            num_experts,                                                      \
            n_group,                                                          \
            topk_group,                                                       \
            topk,                                                             \
            renormalize,                                                      \
            routed_scaling_factor,                                            \
            false,                                                            \
            stream);                                                          \
        break;                                                                \
      case vllm::moe::SCORING_SIGMOID:                                        \
        vllm::moe::invokeNoAuxTc<T, BiasT, IdxT, vllm::moe::SCORING_SIGMOID>( \
            reinterpret_cast<T*>(gating_output.mutable_data_ptr()),           \
            reinterpret_cast<float*>(topk_values.mutable_data_ptr()),         \
            reinterpret_cast<IdxT*>(topk_indices.mutable_data_ptr()),         \
            (has_bias ? reinterpret_cast<BiasT const*>(bias->data_ptr())      \
                      : nullptr),                                             \
            num_tokens,                                                       \
            num_experts,                                                      \
            n_group,                                                          \
            topk_group,                                                       \
            topk,                                                             \
            renormalize,                                                      \
            routed_scaling_factor,                                            \
            false,                                                            \
            stream);                                                          \
        break;                                                                \
      default:                                                                \
        throw std::invalid_argument("Unsupported scoring_func");              \
        break;                                                                \
    }                                                                         \
  } while (0)

#define LAUNCH_KERNEL(T, IdxT)                                         \
  do {                                                                 \
    switch (bias_type) {                                               \
      case torch::kFloat16:                                            \
        LAUNCH_KERNEL_SF(T, sycl::half, IdxT);                         \
        break;                                                         \
      case torch::kFloat32:                                            \
        LAUNCH_KERNEL_SF(T, float, IdxT);                              \
        break;                                                         \
      case torch::kBFloat16:                                           \
        LAUNCH_KERNEL_SF(T, sycl::ext::oneapi::bfloat16, IdxT);        \
        break;                                                         \
      default:                                                         \
        throw std::invalid_argument(                                   \
            "Invalid bias dtype, only supports float16, float32, and " \
            "bfloat16");                                               \
        break;                                                         \
    }                                                                  \
  } while (0)

  switch (data_type) {
    case torch::kFloat16:
      LAUNCH_KERNEL(sycl::half, int32_t);
      break;
    case torch::kFloat32:
      LAUNCH_KERNEL(float, int32_t);
      break;
    case torch::kBFloat16:
      LAUNCH_KERNEL(sycl::ext::oneapi::bfloat16, int32_t);
      break;
    default:
      throw std::invalid_argument(
          "Invalid dtype, only supports float16, float32, and bfloat16");
      break;
  }
#undef LAUNCH_KERNEL
#undef LAUNCH_KERNEL_SF
  return {topk_values, topk_indices};
}