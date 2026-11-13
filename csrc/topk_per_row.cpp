// Adapted from
// https://github.com/vllm-project/vllm/blob/main/csrc/sampler.cu#L646-L728

#include <sycl/sycl.hpp>

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>

#include "dispatch_utils.h"
#include "utils.h"

namespace vllm {

static constexpr int SUBGROUP_SIZE = 32;

// The number of slots for the final pass.
static constexpr int kNumFinalItems = 2048;

// The number of bins in the histogram.
static constexpr int kNumBins = 2048;

// The structure to store the final items (for the final pass).
struct FinalItems {
  // Shared memory to store the indices for the final pass.
  int indices[kNumFinalItems];
  // Shared memory to store the logits for the final pass.
  float logits[kNumFinalItems];
};

struct Histogram {
  int data[kNumBins];
  int tmp[SUBGROUP_SIZE + 1];  // used in prefix sum computation
};

// Struct to hold all static sized shared memory objects
struct SharedStates {
  // Shared memory to compute the block sort.
  union {
    FinalItems items;
    Histogram histo;
  } smemFinal;
  // Shared memory to store the threshold bin.
  int smemThresholdBinIdx[1];
  // Shared memory counter to register the candidates for the final phase.
  int smemFinalDstIdx[1];
  // Shared memory to determine if the threshold bin fits in the final items.
  int smemFinalBinSize[1];
  // Shared memory to keep track of the top-k values found so far by the
  // previous iterations
  int smemFoundTopKValues[1];
};

template <typename scalar_t>
struct alignas(8) vec4_t {
  scalar_t x;
  scalar_t y;
  scalar_t z;
  scalar_t w;
};

template <typename T>
T atomicAdd(T* address, T val) {
  sycl::atomic_ref<
      T,
      sycl::memory_order::relaxed,
      sycl::memory_scope::device,
      sycl::access::address_space::local_space>
      atom_val(*address);
  return atom_val.fetch_add(val);
}

inline auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = sycl::bit_cast<uint32_t>(x);
  return (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
}

inline auto convert_to_uint16(sycl::half hx) -> uint16_t {
  uint16_t bits = sycl::bit_cast<uint16_t>(hx);
  return (bits & 0x8000) ? bits : ~bits & 0x7fff;
}

template <int step>
static inline uint32_t extractBinIdx(float x) {
  if constexpr (step == 0) {
    sycl::half hx = static_cast<sycl::half>(x);
    uint16_t bits = convert_to_uint16(hx);
    // the highest 11 bits, so total 2048 bins
    return bits >> 5;
  } else {
    uint32_t bits = convert_to_uint32(x);

    if constexpr (step == 1) {
      // the highest 11 bits, so total 2048 bins
      return bits >> 21;
    } else if constexpr (step == 2) {
      // the middle 11 bits, so total 2048 bins
      return (bits >> 10) & 0x7ff;
    } else if constexpr (step == 3) {
      // the lowest 10 bits, so total 1024 bins
      return bits & 0x3ff;
    }
  }
}

template <int shift>
static inline bool isPartialMatch(float x, uint32_t pattern) {
  if constexpr (shift == 0) {
    return true;
  }
  uint32_t bits = convert_to_uint32(x);
  return (bits ^ pattern) >> shift == 0;
}

template <typename T, typename idxT, typename Func>
void vectorized_process(
    size_t thread_rank, size_t num_threads, const T* in, idxT len, Func f) {
  using WideT = vec4_t<float>;
  if constexpr (sizeof(T) >= sizeof(WideT)) {
    // in this path, no vectorized load is performed, just process as normal
    for (idxT i = thread_rank; i < len; i += num_threads) {
      f(in[i], i);
    }
  } else {
    static_assert(sizeof(WideT) % sizeof(T) == 0);
    constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);

    // This union is to allow to conveniently access individual elements after
    // vectorized load
    // TODO: it's UB
    union {
      WideT scalar;
      T array[items_per_scalar];
    } wide;

    int skip_cnt =
        (reinterpret_cast<size_t>(in) % sizeof(WideT))
            ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) /
               sizeof(T))
            : 0;
    if (skip_cnt > len) {
      skip_cnt = len;
    }
    const WideT* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
    const idxT len_cast = (len - skip_cnt) / items_per_scalar;

    for (idxT i = thread_rank; i < len_cast; i += num_threads) {
      wide.scalar = in_cast[i];
      const idxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
      for (int j = 0; j < items_per_scalar; ++j) {
        f(wide.array[j], real_i + j);
      }
    }

    static_assert(SUBGROUP_SIZE >= items_per_scalar);
    // and because items_per_scalar > skip_cnt, SUBGROUP_SIZE > skip_cnt
    // no need to use loop
    if (thread_rank < skip_cnt) {
      f(in[thread_rank], thread_rank);
    }
    // because len_cast = (len - skip_cnt) / items_per_scalar,
    // len_cast * items_per_scalar + items_per_scalar > len - skip_cnt;
    // and so
    // len - (skip_cnt + len_cast * items_per_scalar) < items_per_scalar <=
    // SUBGROUP_SIZE no need to use loop
    const idxT remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
    if (remain_i < len) {
      f(in[remain_i], remain_i);
    }
  }
}

template <
    int step,
    int kNumThreadsPerBlock,
    int kNumBins,
    int kNumFinalItems,
    bool multipleBlocksPerRow,
    bool mergeBlocks,
    typename SmemFinalType,
    typename SmemOutputType>
bool processHistogramStep(
    const sycl::nd_item<3>& item,
    const int* indices,
    const float* logits,
    int rowEnd,
    uint32_t& logitPattern,
    int& thresholdBinIdx,
    SmemOutputType* smemOutput,
    int* smemThresholdBinIdx,
    int* smemFinalDstIdx,
    int* smemFinalBinSize,
    int* smemFoundTopKValues,
    SmemFinalType& smemFinal,
    int stride1,
    int rowStart,
    int topK) {
  sycl::group group = item.get_group();
  auto sg = item.get_sub_group();
  auto threadIdx_x = item.get_local_id(0);
  auto lane_id = threadIdx_x % SUBGROUP_SIZE;
  auto sg_id = threadIdx_x / SUBGROUP_SIZE;
  auto sg_num = (kNumThreadsPerBlock + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE;

  // Clear the histogram.
  // The histogram size is [kNumBins], and the threads access it in this layout:
#pragma unroll
  for (int idx = threadIdx_x; idx < kNumBins; idx += kNumThreadsPerBlock) {
    smemFinal.histo.data[idx] = 0;
  }

  // Make sure the histogram is ready.
  sycl::group_barrier(group);

  // Update pattern
  constexpr auto patternShift = step < 2 ? 0 : step == 2 ? 21 : 10;
  if constexpr (step == 2) {
    logitPattern = static_cast<uint32_t>(thresholdBinIdx & 0x7ff)
                   << patternShift;
  } else if constexpr (step == 3) {
    logitPattern |= static_cast<uint32_t>(thresholdBinIdx & 0x7ff)
                    << patternShift;
  }

  auto distributeToBins = [&](float logit, int /* idx */ = 0) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      atomicAdd(&smemFinal.histo.data[binIdx], 1);
    }
  };

  // Distribute the elements to the histogram bins.
  if (stride1 == 1) {
    vectorized_process(
        threadIdx_x,
        kNumThreadsPerBlock,
        logits + rowStart,
        rowEnd - rowStart,
        distributeToBins);
  } else {
    for (int idx = rowStart + threadIdx_x; idx < rowEnd;
         idx += kNumThreadsPerBlock) {
      float logit = logits[idx * stride1];
      distributeToBins(logit, idx);
    }
  }
  // Make sure the histogram is ready.
  sycl::group_barrier(group);

  // Reads the value of the starting position in the smemOutput array
  int lastValue = smemFoundTopKValues[0];

  for (int round = 0; round < kNumBins / kNumThreadsPerBlock; round++) {
    // Read the values from SMEM.
    int idx = threadIdx_x + kNumThreadsPerBlock * round;
    int binCount{0};
    binCount = smemFinal.histo.data[idx];

    // Make sure each thread has read its value.
    sycl::group_barrier(group);

    // Compute the prefix sum.
    int prefixSum{0}, totalSum{0};
    constexpr bool USE_CUB = false;
    if constexpr (USE_CUB) {
      // No CUB lib in sycl, so we implement our own prefix sum
    } else {
      // subgroup-level prefix sum
      int subgroupLocalInclusivePrefixSum = binCount;
      for (int offset = 1; offset < SUBGROUP_SIZE; offset *= 2) {
        int val = sycl::shift_group_right(
            sg, subgroupLocalInclusivePrefixSum, offset);
        if (lane_id >= offset) {
          subgroupLocalInclusivePrefixSum += val;
        }
      }
      int subgroupLocalExclusivePrefixSum =
          subgroupLocalInclusivePrefixSum - binCount;

      // the last lane in each subgroup writes the subgroup level total sum to
      // smem
      int subgroupLocalTotalSum = 0;
      if (lane_id == SUBGROUP_SIZE - 1) {
        subgroupLocalTotalSum = subgroupLocalInclusivePrefixSum;
        smemFinal.histo.tmp[sg_id] = subgroupLocalTotalSum;
      }

      // the first thread in the block computes the prefix sum of subgroup sums
      sycl::group_barrier(group);
      if (sg_id == 0 && lane_id == 0) {
        int sum = 0;
        for (int i = 0; i < sg_num; i++) {
          int val = smemFinal.histo.tmp[i];
          smemFinal.histo.tmp[i] = sum;
          sum += val;
        }
        smemFinal.histo.tmp[sg_num] = sum;
      }

      // add the subgroup prefix sum to each thread's local exclusive prefix sum
      sycl::group_barrier(group);
      int subgroupPrefixSum = smemFinal.histo.tmp[sg_id];
      prefixSum = subgroupPrefixSum + subgroupLocalExclusivePrefixSum;

      totalSum = smemFinal.histo.tmp[sg_num];
    }

    // Update the histogram with the prefix sums.
    prefixSum += lastValue;  // lastValue is the prefix sum of last thread in
                             // the previous round
    totalSum += lastValue;
    smemFinal.histo.data[idx] = prefixSum;

    // Make sure the data is in shared memory.
    sycl::group_barrier(group);

    // Find the last valid bin.
    bool foundThreshold = false;
    if (prefixSum < topK) {
      int nextPrefixSum = threadIdx_x == kNumThreadsPerBlock - 1
                              ? totalSum
                              : smemFinal.histo.data[idx + 1];

      if (nextPrefixSum >= topK) {
        smemThresholdBinIdx[0] = idx;
        smemFinalBinSize[0] = nextPrefixSum - prefixSum;
        foundThreshold = true;
      }
    }

    // Early exit: if any thread found the threshold, we can skip remaining
    // rounds
    if (sycl::any_of_group(group, foundThreshold)) {
      break;
    }

    lastValue = totalSum;
  }

  // Make sure the data is in shared memory.
  sycl::group_barrier(group);

  // The threshold bin.
  thresholdBinIdx = smemThresholdBinIdx[0];

  auto processBins = [&](float logit, int idx) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      if (binIdx < thresholdBinIdx) {
        // The element is part of the top-k selection
        int dstIdx = atomicAdd(&smemFoundTopKValues[0], 1);

        if constexpr (mergeBlocks) {
          smemOutput[dstIdx] = indices[idx];
        } else if constexpr (multipleBlocksPerRow) {
          smemOutput[dstIdx] = idx + rowStart;
          reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
        } else {
          smemOutput[dstIdx] = idx;
        }
      }
      if constexpr (step < 3) {
        // Only fill the final items for sorting if the threshold bin fits
        if (binIdx == thresholdBinIdx &&
            smemFinalBinSize[0] <= kNumFinalItems) {
          int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
          smemFinal.items.logits[dstIdx] = logit;
          if constexpr (mergeBlocks) {
            smemFinal.items.indices[dstIdx] = indices[idx];
          } else if constexpr (multipleBlocksPerRow) {
            smemFinal.items.indices[dstIdx] = idx + rowStart;
          } else {
            smemFinal.items.indices[dstIdx] = idx;
          }
        }
      } else {
        if (binIdx == thresholdBinIdx) {
          // The elements in the threshold bin share the same 32 bits at step 3
          int dstIdx = atomicAdd(&smemFinal.histo.data[binIdx], 1);
          if (dstIdx < topK) {
            if constexpr (mergeBlocks) {
              smemOutput[dstIdx] = indices[idx];
            } else if constexpr (multipleBlocksPerRow) {
              smemOutput[dstIdx] = idx + rowStart;
              reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
            } else {
              smemOutput[dstIdx] = idx;
            }
          }
        }
      }
    }
  };

  if (stride1 == 1) {
    vectorized_process(
        threadIdx_x,
        kNumThreadsPerBlock,
        logits + rowStart,
        rowEnd - rowStart,
        processBins);
  } else {
    for (int idx = rowStart + threadIdx_x; idx < rowEnd;
         idx += kNumThreadsPerBlock) {
      float logit = logits[idx * stride1];
      processBins(logit, idx);
    }
  }

  // Make sure the elements are in shared memory.
  sycl::group_barrier(group);

  // Check if we should continue to next step
  return smemFinalBinSize[0] > kNumFinalItems;
}

template <
    int kNumThreadsPerBlock,
    int kNumBins,
    bool useRadixSort,
    bool multipleBlocksPerRow = false,
    bool mergeBlocks = false>
static void topKPerRowJob(
    const sycl::nd_item<3>& item,
    sycl::local_accessor<char, 1> slm,
    const int* indices,
    const float* logits,
    int rowStart,
    int rowEnd,
    int* outIndices,
    float* outLogits,
    int stride1,
    int topK) {
  sycl::group group = item.get_group();
  auto sg = item.get_sub_group();
  auto threadIdx_x = item.get_local_id(0);

  char* smem_buf =
      slm.template get_multi_ptr<sycl::access::decorated::no>().get();

  int sharedStatesOffset = topK * sizeof(int32_t);
  if constexpr (multipleBlocksPerRow) {
    sharedStatesOffset += topK * sizeof(float);
  }

  int32_t* smemOutput = reinterpret_cast<int32_t*>(smem_buf);
  SharedStates* sharedStates =
      reinterpret_cast<SharedStates*>(smem_buf + sharedStatesOffset);

  // Shared memory references
  auto& smemFinal = sharedStates->smemFinal;
  int* smemThresholdBinIdx = sharedStates->smemThresholdBinIdx;
  int* smemFinalDstIdx = sharedStates->smemFinalDstIdx;
  int* smemFinalBinSize = sharedStates->smemFinalBinSize;
  int* smemFoundTopKValues = sharedStates->smemFoundTopKValues;

  // The length of the row.
  int rowLen = rowEnd - rowStart;

  // Shortcut if the length of the row is smaller than Top-K. Indices are not
  // sorted by their corresponding logit.
  if (rowLen <= topK) {
    for (int rowIt = threadIdx_x; rowIt < rowLen;
         rowIt += kNumThreadsPerBlock) {
      if constexpr (multipleBlocksPerRow) {
        outIndices[rowIt] = rowIt + rowStart;
        outLogits[rowIt] = logits[rowIt + rowStart];
      } else {
        outIndices[rowIt] = rowIt;
      }
    }
    for (int rowIt = rowLen + threadIdx_x; rowIt < topK;
         rowIt += kNumThreadsPerBlock) {
      outIndices[rowIt] = -1;
      if constexpr (multipleBlocksPerRow) {
        outLogits[rowIt] = -std::numeric_limits<float>::max();
      }
    }

    return;
  }
  // Initialize values
  if (threadIdx_x == 0) {
    smemFinalDstIdx[0] = 0;
    smemFoundTopKValues[0] = 0;
  }
  sycl::group_barrier(group);
  int thresholdBinIdx = -1;
  uint32_t logitPattern = 0;

  // Step 0: Process first 11 bits of half representation
  bool continueToNextStep = processHistogramStep<
      0,
      kNumThreadsPerBlock,
      kNumBins,
      kNumFinalItems,
      multipleBlocksPerRow,
      mergeBlocks>(
      item,
      indices,
      logits,
      rowEnd,
      logitPattern,
      thresholdBinIdx,
      smemOutput,
      smemThresholdBinIdx,
      smemFinalDstIdx,
      smemFinalBinSize,
      smemFoundTopKValues,
      smemFinal,
      stride1,
      rowStart,
      topK);

  if (continueToNextStep) {
    // Step 1: Process next 11 bits
    continueToNextStep = processHistogramStep<
        1,
        kNumThreadsPerBlock,
        kNumBins,
        kNumFinalItems,
        multipleBlocksPerRow,
        mergeBlocks>(
        item,
        indices,
        logits,
        rowEnd,
        logitPattern,
        thresholdBinIdx,
        smemOutput,
        smemThresholdBinIdx,
        smemFinalDstIdx,
        smemFinalBinSize,
        smemFoundTopKValues,
        smemFinal,
        stride1,
        rowStart,
        topK);
  }

  if (continueToNextStep) {
    // Step 2: Process next 11 bits
    continueToNextStep = processHistogramStep<
        2,
        kNumThreadsPerBlock,
        kNumBins,
        kNumFinalItems,
        multipleBlocksPerRow,
        mergeBlocks>(
        item,
        indices,
        logits,
        rowEnd,
        logitPattern,
        thresholdBinIdx,
        smemOutput,
        smemThresholdBinIdx,
        smemFinalDstIdx,
        smemFinalBinSize,
        smemFoundTopKValues,
        smemFinal,
        stride1,
        rowStart,
        topK);
  }

  if (continueToNextStep) {
    // Step 3: Process last 10 bits
    processHistogramStep<
        3,
        kNumThreadsPerBlock,
        kNumBins,
        kNumFinalItems,
        multipleBlocksPerRow,
        mergeBlocks>(
        item,
        indices,
        logits,
        rowEnd,
        logitPattern,
        thresholdBinIdx,
        smemOutput,
        smemThresholdBinIdx,
        smemFinalDstIdx,
        smemFinalBinSize,
        smemFoundTopKValues,
        smemFinal,
        stride1,
        rowStart,
        topK);
  }

  if (!continueToNextStep) {
    // The histogram did not proceed to the final 10 bits, therefore we need to
    // sort the final items The logits of the elements to be sorted in the final
    // pass.
    if constexpr (useRadixSort) {
      static_assert(!useRadixSort, "Radix sort is not implemented yet");
      // TODO
    } else {
      // Sorting with insertion sort
      auto baseIdx = smemFoundTopKValues[0];
      for (int i = threadIdx_x; i < smemFinalDstIdx[0];
           i += kNumThreadsPerBlock) {
        int outIndex = 0;
        auto logit = smemFinal.items.logits[i];
        for (int j = 0; j < smemFinalDstIdx[0]; j++) {
          auto otherLogit = smemFinal.items.logits[j];
          if (logit < otherLogit || (logit == otherLogit && i < j)) {
            outIndex++;
          }
        }
        // Store if outIndex is in bounds
        auto remainedTopK = topK - baseIdx;
        if (outIndex < remainedTopK) {
          smemOutput[outIndex + baseIdx] = smemFinal.items.indices[i];
          if constexpr (multipleBlocksPerRow) {
            reinterpret_cast<float*>(smemOutput + topK)[outIndex + baseIdx] =
                smemFinal.items.logits[i];
          }
        }
      }
    }
    sycl::group_barrier(group);
  }

  // Store to global memory.
  for (int i = threadIdx_x; i < topK; i += kNumThreadsPerBlock) {
    if constexpr (multipleBlocksPerRow) {
      outIndices[i] = smemOutput[i];
      outLogits[i] = reinterpret_cast<float*>(smemOutput + topK)[i];
    } else {
      if (stride1 == 1) {
        // stride1 == 1 will use vectorized_process, which indexes already skip
        // the rowStart.
        outIndices[i] = smemOutput[i];
      } else {
        outIndices[i] = smemOutput[i] - rowStart;
      }
    }
  }
}

template <int kNumThreadsPerBlock, bool useRadixSort>
class top_k_per_row_prefill_kernel {
 public:
  top_k_per_row_prefill_kernel(
      sycl::local_accessor<char, 1>& slm,
      const int32_t* indices,
      const float* logits,
      const int32_t* rowStart,
      const int32_t* rowEnd,
      const int64_t stride0,
      const int64_t stride1,
      const int64_t topK,
      const int64_t offsetIndex)
      : slm_(slm),
        indices_(indices),
        logits_(logits),
        rowStart_(rowStart),
        rowEnd_(rowEnd),
        stride0_(stride0),
        stride1_(stride1),
        topK_(topK),
        offsetIndex_(offsetIndex) {}
  void operator() [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] (
      const sycl::nd_item<3>& item) const {
    int64_t group_idx = item.get_group(0);
    int64_t local_idx = item.get_local_id(0);
    int local_range = item.get_local_range(0);

    auto blockIdx_x = group_idx;

    // The row computed by this block.
    int rowIdx = blockIdx_x + offsetIndex_;

    // The range of logits within the row.
    int rowStart = rowStart_[rowIdx];
    int rowEnd = rowEnd_[rowIdx];

    // Local pointers to this block
    const int32_t* outIndices = indices_ + static_cast<int64_t>(rowIdx) * topK_;
    const float* logits = logits_ + static_cast<int64_t>(rowIdx) * stride0_;

    // Launch the top-k per row job
    topKPerRowJob<kNumThreadsPerBlock, kNumBins, useRadixSort, false, false>(
        item,
        slm_,
        /*indices=*/nullptr,
        logits,
        rowStart,
        rowEnd,
        const_cast<int32_t*>(outIndices),
        /*outLogits=*/nullptr,
        stride1_,
        topK_);
  }

 private:
  sycl::local_accessor<char, 1> slm_;
  const int32_t* indices_;   // [num_tokens, topK]
  const float* logits_;      // [num_tokens, num_max_logits]
  const int32_t* rowStart_;  // [num_tokens]
  const int32_t* rowEnd_;    // [num_tokens]
  const int64_t stride0_;
  const int64_t stride1_;
  const int64_t topK_;
  const int64_t offsetIndex_;
};

template <
    int kNumThreadsPerBlock,
    bool useRadixSort,
    bool multipleBlocksPerRow = false,
    bool mergeBlocks = false>
class top_k_per_row_decode_kernel {
 public:
  top_k_per_row_decode_kernel(
      sycl::local_accessor<char, 1>& slm,
      const float* logits,
      const int32_t* seqLens,
      int32_t* outIndices,
      int64_t stride0,
      int64_t stride1,
      int64_t topK,
      int64_t next_n,
      float* outLogits = nullptr,
      const int numBlocksToMerge = 0,
      const int32_t* indices = nullptr)
      : slm_(slm),
        logits_(logits),
        seqLens_(seqLens),
        outIndices_(outIndices),
        stride0_(stride0),
        stride1_(stride1),
        topK_(topK),
        next_n_(next_n),
        outLogits_(outLogits),
        numBlocksToMerge_(numBlocksToMerge),
        indices_(indices) {}

  void operator() [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] (
      const sycl::nd_item<3>& item) const {
    auto blockIdx_x = item.get_group(0);
    auto blockIdx_y = item.get_group(1);
    auto gridDim_y = item.get_group_range(1);

    // The row computed by this block.
    int rowIdx = blockIdx_x;

    // The range of logits within the row.
    int rowStart = 0;
    int32_t seq_len = seqLens_[rowIdx / next_n_];
    int rowEnd = seq_len - next_n_ + (rowIdx % next_n_) + 1;

    const float* logits = logits_;
    const int* indices = indices_;
    int* outIndices = outIndices_;
    float* outLogits = outLogits_;
    // Local pointers to this block
    if constexpr (!multipleBlocksPerRow && !mergeBlocks) {
      outIndices += static_cast<int64_t>(rowIdx) * topK_;
    } else if constexpr (multipleBlocksPerRow) {
      const auto blockSize = rowEnd / gridDim_y;  // 16384 / 2 = 8192
      rowStart = blockSize * blockIdx_y;          // 8192 * 1 = 8192
      rowEnd = gridDim_y == blockIdx_y + 1 ? rowEnd : rowStart + blockSize;
      outIndices +=
          static_cast<int64_t>(rowIdx) * gridDim_y * topK_ + blockIdx_y * topK_;
      outLogits +=
          static_cast<int64_t>(rowIdx) * gridDim_y * topK_ + blockIdx_y * topK_;
    } else if constexpr (mergeBlocks) {
      rowEnd = numBlocksToMerge_ * topK_;
      indices += static_cast<int64_t>(rowIdx) * numBlocksToMerge_ * topK_;
      outIndices += static_cast<int64_t>(rowIdx) * topK_;
    }
    logits += static_cast<int64_t>(rowIdx) * stride0_;

    topKPerRowJob<
        kNumThreadsPerBlock,
        kNumBins,
        useRadixSort,
        multipleBlocksPerRow,
        mergeBlocks>(
        item,
        slm_,
        indices,
        logits,
        rowStart,
        rowEnd,
        outIndices,
        outLogits,
        stride1_,
        topK_);
  }

 private:
  sycl::local_accessor<char, 1> slm_;
  const float* logits_;
  const int32_t* seqLens_;
  int32_t* outIndices_;
  int64_t stride0_;
  int64_t stride1_;
  int64_t topK_;
  int64_t next_n_;
  float* outLogits_;
  const int numBlocksToMerge_;
  const int32_t* indices_;
};

// Delegate-based selection for wide rows (Intel BMG): one work-group per row.
// top_k_per_row_decode/prefill send a call here, instead of to the radix
// kernels above, where the dispatch rule at the end of this namespace says it
// is measurably faster (see wideRowLayout).
//
// Rows of at least minDelegateLen(topK) columns run the delegate select: from
// an SLM copy of the row if it fits the stage (StagedRow), else straight from
// global memory (GlobalRow). Shorter rows, and the rare row on which the select
// gives up (more candidates than slots, e.g. very tie-heavy rows), run
// exactRowJob from the same SLM arena - exact either way.
//
// Delegate select. Split the row into groups of G floats and call each group's
// largest key its delegate. Let t be the topK-th largest delegate: topK groups
// hold a key >= t each, so t is at most the row's topK-th largest key, and
// every key > t lies in one of the fewer than topK groups whose delegate is >
// t. So the row's top topK are its keys > t from those groups - the candidates
// - plus, when there are fewer than topK of them, keys == t. With G ~ n /
// (kGroupDiv * topK) the only passes over the whole row are the copy into SLM
// and one max per group; the radix selects run over the ~3 * topK delegates and
// the candidates, not over n keys.
//
// Keys are uint32 in the order of the radix kernels' (+NaN above +inf, -NaN
// below -inf, +0 above -0), and every comparison is on keys or on the float
// bits, so NaN, signed zeros and infinities rank as they do there.

// Forced inlining of the hot helpers into the kernel body.
#define TOPK_ALWAYS_INLINE inline __attribute__((always_inline))

namespace delegate_topk {

// Exact single-work-group selection. It runs the rows the delegate select
// below does not take (short rows, rows on which it gives up). It selects the
// topK largest of a row of more than topK columns with unit column stride: a
// one-pass histogram (topKPerRowReduce) and, for the rare row whose threshold
// fp16 bin is too large for it, a split of that bin (processRefineStep, then
// processTieSelect for ties). Names that also exist in the radix kernels above
// (kNumFinalItems, Histogram, SharedStates) differ here, hence the namespace.

// Threads per work-group, one work-group per row.
static constexpr int kNumThreads = 512;
static constexpr int kNumSubGroups = kNumThreads / SUBGROUP_SIZE;

// The number of slots for the final pass. This buffer is the largest part of
// the job's shared memory; rows whose threshold bin does not fit are split
// further.
static constexpr int kNumFinalItems = 512;

// Elements each thread keeps in registers on the reduction path. 8 covers every
// row up to 8 * kNumThreads = 4096 elements; longer rows re-read in chunks.
static constexpr int kFastMaxRegItems = 8;

// The reduction path hands the threshold bin's elements to a pairwise ranking,
// which is O(n^2), so it gives up above this many of them.
static constexpr int kFastMaxCandidates = 128;

// Rows at least this long may hand the radix select up to half as many
// candidates as the histogram has slots, rather than kFastMaxCandidates, and
// keep them in the histogram: giving up costs them two more reads of a row that
// no longer fits the last level cache, which is more than ranking the larger
// set.
static constexpr int kLongRowMinLen = 16384;

// Rows at least this long read the row as 16-byte vectors.
static constexpr int kVecRowMinLen = 262144;
// The longest row whose pass two reserves candidates once per chunk.
static constexpr int kCandBatchMaxRowLen = 131072;

// The structure to store the final items (for the final pass).
struct FinalItems {
  // Shared memory to store the indices for the final pass.
  int indices[kNumFinalItems];
  // Shared memory to store the logits for the final pass.
  float logits[kNumFinalItems];
};

// The reduce path keeps this many copies of every bin, interleaved, and lane l
// of a subgroup counts into copy l % kHistCopies.
static constexpr int kHistCopies = 4;

// What a row long enough for the large candidate buffer keeps instead. The 32
// lanes of a subgroup then share kLongHistCopies counters per bin rather than
// kHistCopies, so a row whose elements crowd into one fp16 bin serialises half
// as many same-address atomics: Xe has no same-address coalescing for subgroup
// atomics. A shorter row keeps kHistCopies, so that it does not pay the wider
// clear and scan of the histogram over a row that is too short to hide them.
static constexpr int kLongHistCopies = 7;

struct Histogram {
  // 16-byte aligned so the reduce path can sweep a thread's bins as one vector.
  alignas(16) int data[kNumBins * kLongHistCopies];
  int tmp[SUBGROUP_SIZE + 1];  // used in prefix sum computation
};

struct SmemFinal {
  FinalItems items;
  Histogram histo;
};

// Struct to hold all static sized shared memory objects
struct SharedStates {
  SmemFinal smemFinal;
  // Shared memory to store the threshold bin.
  int smemThresholdBinIdx;
  // Shared memory counter to register the candidates for the final phase.
  int smemFinalDstIdx;
  // Shared memory to determine if the threshold bin fits in the final items.
  int smemFinalBinSize;
  // Shared memory to keep track of the top-k values found so far.
  int smemFoundTopKValues;
  // Output slot counter for the elements processRefineStep() accepts on their
  // fp16 bin alone.
  int smemRefineDstIdx;
};

constexpr size_t alignUp(size_t size, size_t alignment) {
  return ((size + alignment - 1) / alignment) * alignment;
}

// The topK output slots come first, then the shared state.
constexpr size_t sharedStatesOffset(size_t topK) {
  return alignUp(topK * sizeof(int32_t), alignof(SharedStates));
}

constexpr size_t sharedMemorySize(size_t topK) {
  return sharedStatesOffset(topK) + sizeof(SharedStates);
}

// Not recursive: a SYCL kernel may not call a recursive function, even one that
// only ever runs at compile time.
static constexpr int constexprLog2(int n) {
  int log = 0;
  while (n > 1) {
    n /= 2;
    ++log;
  }
  return log;
}

// The bin index for the reduction path below. Same monotone half key as
// extractBinIdx<0>(), but only the highest constexprLog2(kNumFastBins) bits, so
// that one bin can be owned by one thread.
template <int kNumFastBins>
static inline uint32_t extractFastBinIdx(float x) {
  static_assert((kNumFastBins & (kNumFastBins - 1)) == 0, "power of two");
  static_assert(kNumFastBins <= (1 << 16));
  sycl::half hx = static_cast<sycl::half>(x);
  return convert_to_uint16(hx) >> (16 - constexprLog2(kNumFastBins));
}

// The kBinsPerThread bins a thread owns are consecutive, so lane l of a
// subgroup addresses dword kBinsPerThread * l and all 32 lanes of a plain
// scalar sweep land on one bank in kBinsPerThread rounds. These move the whole
// set as one 16-byte message instead. The reduce path sweeps its bins three
// times: to clear them, to read the counts and to write the prefix sums back.
template <int kN>
static inline void binsLoad(const int* base, int (&out)[kN]) {
  if constexpr (kN % 4 == 0) {
#pragma unroll
    for (int j = 0; j < kN; j += 4) {
      const sycl::vec<int, 4> v =
          *reinterpret_cast<const sycl::vec<int, 4>*>(base + j);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        out[j + i] = v[i];
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < kN; ++i) {
      out[i] = base[i];
    }
  }
}

template <int kN>
static inline void binsStore(int* base, const int (&in)[kN]) {
  if constexpr (kN % 4 == 0) {
#pragma unroll
    for (int j = 0; j < kN; j += 4) {
      sycl::vec<int, 4> v;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        v[i] = in[j + i];
      }
      *reinterpret_cast<sycl::vec<int, 4>*>(base + j) = v;
    }
  } else {
#pragma unroll
    for (int i = 0; i < kN; ++i) {
      base[i] = in[i];
    }
  }
}

// Exclusive scan of one value per thread over the whole work-group: a subgroup
// ladder, then one subgroup scans the subgroup totals. Two barriers, ~12
// shuffles, no atomics. tmp needs kNumSubGroups + 1 ints of SLM; the last slot
// carries the block total, which every thread needs to keep a running output
// cursor without a further reduction.
static inline int blockExclusiveScan(
    const sycl::group<3>& group,
    const sycl::sub_group& sg,
    int lane_id,
    int sg_id,
    int* tmp,
    int val,
    int& total) {
  static_assert(
      kNumSubGroups <= SUBGROUP_SIZE, "the totals are scanned by one ladder");

  int inclusive = val;
  for (int offset = 1; offset < SUBGROUP_SIZE; offset *= 2) {
    int other = sycl::shift_group_right(sg, inclusive, offset);
    if (lane_id >= offset) {
      inclusive += other;
    }
  }
  if (lane_id == SUBGROUP_SIZE - 1) {
    tmp[sg_id] = inclusive;
  }
  sycl::group_barrier(group);

  if (sg_id == 0) {
    int v = lane_id < kNumSubGroups ? tmp[lane_id] : 0;
    int inc = v;
    for (int offset = 1; offset < SUBGROUP_SIZE; offset *= 2) {
      int other = sycl::shift_group_right(sg, inc, offset);
      if (lane_id >= offset) {
        inc += other;
      }
    }
    if (lane_id < kNumSubGroups) {
      tmp[lane_id] = inc - v;  // exclusive
      if (lane_id == kNumSubGroups - 1) {
        tmp[kNumSubGroups] = inc;  // the block total
      }
    }
  }
  sycl::group_barrier(group);

  total = tmp[kNumSubGroups];
  return tmp[sg_id] + inclusive - val;
}

// vectorized_process() with kDepth vector loads in flight per thread instead of
// one. Its loop body waits on its own load, and a pass that does little per
// element is then bound by the load latency: with one vector per thread a Xe
// core has 32KB in flight, about two thirds of what the bandwidth needs.
template <int kDepth, typename Func>
static inline void vectorizedProcessDeep(
    int thread_rank,
    int num_threads,
    const float* in,
    int len,
    int rotate,
    Func f) {
  const int misalign = reinterpret_cast<uintptr_t>(in) % sizeof(sycl::float4);
  const int skip = sycl::min(
      misalign
          ? static_cast<int>((sizeof(sycl::float4) - misalign) / sizeof(float))
          : 0,
      len);
  const sycl::float4* in4 = reinterpret_cast<const sycl::float4*>(in + skip);
  const int len4 = (len - skip) / 4;
  const int blockLen = kDepth * num_threads;
  const int numBlocks = (len4 + blockLen - 1) / blockLen;
  const int first = numBlocks > 0 ? rotate % numBlocks : 0;
  for (int b = 0; b < numBlocks; ++b) {
    int blk = b + first;
    blk = blk < numBlocks ? blk : blk - numBlocks;
    const int i = blk * blockLen + thread_rank;
    sycl::float4 w[kDepth];
#pragma unroll
    for (int d = 0; d < kDepth; ++d) {
      if (i + d * num_threads < len4) {
        w[d] = in4[i + d * num_threads];
      }
    }
#pragma unroll
    for (int d = 0; d < kDepth; ++d) {
      if (i + d * num_threads < len4) {
        const int real = skip + (i + d * num_threads) * 4;
        f(w[d].x(), real);
        f(w[d].y(), real + 1);
        f(w[d].z(), real + 2);
        f(w[d].w(), real + 3);
      }
    }
  }
  if (thread_rank < skip) {
    f(in[thread_rank], thread_rank);
  }
  const int remain = skip + len4 * 4 + thread_rank;
  if (remain < len) {
    f(in[remain], remain);
  }
}

// Slack, in fp32 key units, left below the first processRefineStep() bucket and
// above the last: one fp16 ulp, for the values that round onto the bin.
static constexpr uint32_t kRefineSlack = 1u << 13;

// Width, as a shift of the fp32 key, of one processRefineStep() bucket. An fp16
// bin of 2^kFastShift halves of one exponent is 2^(kFastShift + 13) consecutive
// fp32 keys, plus the slack at either end, so the pass reaches
// (2^(kFastShift + 13) + 2 * kRefineSlack) >> refineShift buckets: 1088 at
// kLongRefineShift + 2 and 4352 at kLongRefineShift, of the
// kNumBins * kLongHistCopies ints the histogram it borrows holds. A long row
// takes the finer one, so that a row packed into one fp16 bin splits fine
// enough for its threshold bucket to fit the final items; a short row takes the
// coarse one, because clearing the difference costs more than the extra reads
// it saves. The shift is a plain argument, not a template one, so that the two
// choices share one copy of the pass in the kernel.
static constexpr int kLongRefineShift = 6;

// Buckets processRefineStep() splits its fp16 bin into at this shift, rounded
// up to what its scan splits evenly over the threads. Clearing and scanning
// only these, rather than the whole borrowed histogram, is what keeps the finer
// resolution affordable.
static constexpr int refineNumBuckets(int refineShift) {
  constexpr int kFastShift = 16 - constexprLog2(kNumBins);
  const int reached =
      ((1 << (kFastShift + 13)) + 2 * static_cast<int>(kRefineSlack)) >>
      refineShift;
  return ((reached + kNumThreads - 1) / kNumThreads) * kNumThreads;
}

// Whether processRefineStep() can split this fp16 bin. The bin shares the top
// 11 bits of the fp16 key, and with them one exponent. Exponent 0 (zero and
// subnormals) and 31 (inf and NaN) are excluded: everything below 2^-24 rounds
// to the zero bin and everything above 65520 to the inf one, so those bins
// cover far more than 2^18 fp32 keys.
template <int kNumFastBins>
static inline bool fastBinIsRefinable(int bin) {
  const uint32_t key16 = static_cast<uint32_t>(bin)
                         << (16 - constexprLog2(kNumFastBins));
  const uint32_t bits16 = (key16 & 0x8000) ? key16 : (~key16 & 0x7fff);
  const uint32_t exp16 = (bits16 >> 10) & 0x1f;
  return exp16 != 0 && exp16 != 0x1f;
}

// The fp32 key of the first bucket of processRefineStep() for this fp16 bin:
// the key of the bin's largest value (its smallest key, since keys descend with
// the value), minus the slack for values that round onto it.
template <int kNumFastBins>
static inline uint32_t refineBase(int bin) {
  const uint32_t key16 = static_cast<uint32_t>(bin)
                         << (16 - constexprLog2(kNumFastBins));
  const uint16_t bits16 =
      static_cast<uint16_t>((key16 & 0x8000) ? key16 : (~key16 & 0x7fff));
  const float first = static_cast<float>(sycl::bit_cast<sycl::half>(bits16));
  return convert_to_uint32(first) - kRefineSlack;
}

// Takes over from the reduction path when it gave up on a threshold bin that
// was too large, and splits that same bin again on its fp32 bits.
//
// It keeps what the reduction path already established, the threshold fp16 bin
// fastBin and the `accepted` elements in the bins before it, and histograms
// only that bin's elements, a few fp32 keys per bucket. The elements then
// spread over ~1000 buckets, so the new threshold bucket holds a handful and
// the final rank is over those. Two reads of the row, histogram and emit, after
// the reduction path's one.
//
// Returns false, having written nothing but shared state, if even the threshold
// bucket does not fit kNumFinalItems (a row of mostly exact duplicates). The
// caller then finishes the row with processTieSelect().
static bool processRefineStep(
    const sycl::nd_item<3>& item,
    const float* logits,
    int rowLen,
    int fastBin,
    int refineShift,
    int accepted,
    int32_t* smemOutput,
    int* smemThresholdBinIdx,
    int* smemFinalDstIdx,
    int* smemFinalBinSize,
    int* smemFoundTopKValues,
    int* smemRefineDstIdx,
    SmemFinal& smemFinal,
    int topK) {
  sycl::group group = item.get_group();
  auto sg = item.get_sub_group();
  auto threadIdx_x = item.get_local_id(0);
  auto lane_id = threadIdx_x % SUBGROUP_SIZE;
  auto sg_id = threadIdx_x / SUBGROUP_SIZE;
  auto sg_num = (kNumThreads + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE;

  // Buckets this pass splits fastBin into, out of the ints of the borrowed
  // histogram, which is kLongHistCopies times what the fp16 pass needed. A row
  // packed into one fp16 bin then splits fine enough for its threshold bucket
  // to fit the final items, instead of giving up here and paying the whole
  // fp32 chain (six more reads of the row).
  const int numBuckets = refineNumBuckets(refineShift);
  static_assert(
      refineNumBuckets(kLongRefineShift) <= kNumBins * kLongHistCopies,
      "the buckets of the finest shift fit the histogram this borrows");
  const uint32_t base = refineBase<kNumBins>(fastBin);
  // Only called on elements of fastBin, which lie in the first numBuckets
  // buckets. The clamp is there for memory safety only.
  auto bucketOf = [&](float logit) {
    const uint32_t b = (convert_to_uint32(logit) - base) >> refineShift;
    return static_cast<int>(
        sycl::min(b, static_cast<uint32_t>(numBuckets - 1)));
  };

  for (int idx = threadIdx_x; idx < numBuckets; idx += kNumThreads) {
    smemFinal.histo.data[idx] = 0;
  }
  if (threadIdx_x == 0) {
    smemRefineDstIdx[0] = 0;
  }
  sycl::group_barrier(group);

  auto distributeToBins = [&](float logit, int /* idx */) {
    if (static_cast<int>(extractFastBinIdx<kNumBins>(logit)) == fastBin) {
      atomicAdd(&smemFinal.histo.data[bucketOf(logit)], 1);
    }
  };
  vectorizedProcessDeep<4>(
      threadIdx_x,
      kNumThreads,
      logits,
      rowLen,
      static_cast<int>(item.get_group_linear_id()),
      distributeToBins);
  sycl::group_barrier(group);

  // A block-wide exclusive scan over the buckets, based at the count of
  // elements the fp16 bins before fastBin already accepted.
  // refineNumBuckets() rounds up to a multiple of the threads, so the buckets
  // split evenly.
  const int binsPerThread = numBuckets / kNumThreads;
  static_assert(kNumThreads <= SUBGROUP_SIZE * SUBGROUP_SIZE);
  const int binBase = threadIdx_x * binsPerThread;

  int threadSum = 0;
  for (int j = 0; j < binsPerThread; ++j) {
    threadSum += smemFinal.histo.data[binBase + j];
  }
  int sgInclusive = threadSum;
  for (int offset = 1; offset < SUBGROUP_SIZE; offset *= 2) {
    int val = sycl::shift_group_right(sg, sgInclusive, offset);
    if (lane_id >= offset) {
      sgInclusive += val;
    }
  }
  if (lane_id == SUBGROUP_SIZE - 1) {
    smemFinal.histo.tmp[sg_id] = sgInclusive;
  }
  sycl::group_barrier(group);
  if (sg_id == 0) {
    int val = lane_id < sg_num ? smemFinal.histo.tmp[lane_id] : 0;
    int inclusive = val;
    for (int offset = 1; offset < SUBGROUP_SIZE; offset *= 2) {
      int other = sycl::shift_group_right(sg, inclusive, offset);
      if (lane_id >= offset) {
        inclusive += other;
      }
    }
    if (lane_id < sg_num) {
      smemFinal.histo.tmp[lane_id] = inclusive - val;
    }
  }
  sycl::group_barrier(group);

  // accepted < topK <= accepted + |fastBin|, so exactly one bucket brackets
  // topK.
  int prefixSum =
      accepted + smemFinal.histo.tmp[sg_id] + (sgInclusive - threadSum);
  for (int j = 0; j < binsPerThread; ++j) {
    const int binCount = smemFinal.histo.data[binBase + j];
    smemFinal.histo.data[binBase + j] = prefixSum;
    const int nextPrefixSum = prefixSum + binCount;
    if (prefixSum < topK && nextPrefixSum >= topK) {
      smemThresholdBinIdx[0] = binBase + j;
      smemFinalBinSize[0] = binCount;
      smemFoundTopKValues[0] = prefixSum;
    }
    prefixSum = nextPrefixSum;
  }
  sycl::group_barrier(group);

  if (smemFinalBinSize[0] > kNumFinalItems) {
    return false;
  }
  const int thresholdBucket = smemThresholdBinIdx[0];

  // Elements of the bins before fastBin take the first `accepted` slots, in no
  // particular order, from one counter: at most topK atomics per row, against
  // the ~s_kv elements this pass reads. Elements of fastBin take their bucket's
  // slots, and those of the threshold bucket go to the final items.
  auto processBins = [&](float logit, int idx) {
    const int bin = static_cast<int>(extractFastBinIdx<kNumBins>(logit));
    if (bin < fastBin) {
      smemOutput[atomicAdd(&smemRefineDstIdx[0], 1)] = idx;
    } else if (bin == fastBin) {
      const int bucket = bucketOf(logit);
      if (bucket < thresholdBucket) {
        smemOutput[atomicAdd(&smemFinal.histo.data[bucket], 1)] = idx;
      } else if (bucket == thresholdBucket) {
        const int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
        smemFinal.items.logits[dstIdx] = logit;
        smemFinal.items.indices[dstIdx] = idx;
      }
    }
  };
  vectorizedProcessDeep<4>(
      threadIdx_x,
      kNumThreads,
      logits,
      rowLen,
      static_cast<int>(item.get_group_linear_id()),
      processBins);
  sycl::group_barrier(group);
  return true;
}

// Digit width of processTieSelect(): 4096 buckets, 8 per thread at 512
// threads, in the ints of the histogram it borrows.
static constexpr int kTieSelectDigitBits = 12;

// Takes over a row whose fp16 threshold bin neither the reduction path nor
// processRefineStep() could split: exact duplicates, or values that all round
// into the zero (or inf) fp16 bin, which processRefineStep() cannot split at
// all.
//
// It ranks only fastBin's elements, on their fp32 key relative to the smallest
// one, and only on as many bits as their spread needs:
//
//   * One read writes the elements of the bins before fastBin, fills the rest
//     of the output with fastBin's elements as if they were all tied, and takes
//     their minimum and maximum key. If those are equal, which is the case for
//     a row of ties, that read was the answer.
//   * Otherwise one histogram read per kTieSelectDigitBits digit of the
//     spread, most significant first, each over the elements that share the
//     digits chosen so far, stopping once the chosen bucket is taken whole or
//     the remaining keys are exact ties.
//   * Then one emit read rewrites fastBin's share of the output: the elements
//     ranked strictly above the final bucket and `need` of those in it.
//
// Histogram increments and slot claims are aggregated per subgroup: the lanes
// that share the lowest active lane's bucket add as one, so a tie costs one
// atomic per subgroup instead of 32 to the same address.
static void processTieSelect(
    const sycl::nd_item<3>& item,
    const float* rowBase,
    int rowLen,
    int fastBin,
    int accepted,
    int* outIndices,
    int* hist,
    int* scanTmp,
    int* selVars,
    int topK) {
  constexpr int kItems = 4;
  constexpr int kNumDigitBuckets = 1 << kTieSelectDigitBits;
  constexpr int kBucketsPerThread = kNumDigitBuckets / kNumThreads;
  static_assert(
      kNumDigitBuckets % kNumThreads == 0,
      "the buckets split evenly over the threads");
  static_assert(
      kNumDigitBuckets <= kNumBins * kLongHistCopies,
      "the buckets fit the histogram this borrows");

  sycl::group group = item.get_group();
  auto sg = item.get_sub_group();
  const int tid = static_cast<int>(item.get_local_id(0));
  const int lane = tid % SUBGROUP_SIZE;
  const int sgId = tid / SUBGROUP_SIZE;

  int* inCursor = selVars + 3;
  int* tieCursor = selVars + 4;
  if (tid == 0) {
    *inCursor = 0;
    *tieCursor = 0;
  }
  sycl::group_barrier(group);

  // Every thread runs the same number of iterations, so the subgroup
  // operations inside fn see all of their lanes. Element k of a thread's chunk
  // is kNumThreads apart from element k + 1, so each load is one
  // coalesced run across the subgroup.
  constexpr int kPerIteration = kNumThreads * kItems;
  const int numIterations = (rowLen + kPerIteration - 1) / kPerIteration;
  auto walk = [&](auto&& fn) {
    for (int it = 0; it < numIterations; ++it) {
      const int base = it * kPerIteration + tid;
      float v[kItems];
#pragma unroll
      for (int k = 0; k < kItems; ++k) {
        const int j = base + k * kNumThreads;
        v[k] = j < rowLen ? rowBase[static_cast<int64_t>(j)] : 0.f;
      }
#pragma unroll
      for (int k = 0; k < kItems; ++k) {
        const int j = base + k * kNumThreads;
        fn(v[k], j, j < rowLen);
      }
    }
  };
  auto inFastBin = [&](float x) {
    return static_cast<int>(extractFastBinIdx<kNumBins>(x)) == fastBin;
  };
  // Subgroup-uniform: counters[b] += 1 for every active lane.
  auto aggregatedInc = [&](int* counters, bool active, int b) {
    if (!sycl::any_of_group(sg, active)) {
      return;
    }
    const int leader = sycl::reduce_over_group(
        sg, active ? lane : SUBGROUP_SIZE, sycl::minimum<int>());
    const int leaderBucket = sycl::select_from_group(sg, b, leader);
    const bool match = active && b == leaderBucket;
    const int n = sycl::reduce_over_group(sg, match ? 1 : 0, sycl::plus<int>());
    if (lane == leader) {
      atomicAdd(&counters[leaderBucket], n);
    } else if (active && !match) {
      atomicAdd(&counters[b], 1);
    }
  };
  // Subgroup-uniform: a slot from *cursor for every active lane. end is the
  // cursor after this subgroup's claim, the same in every lane.
  auto claim = [&](int* cursor, bool active, int& end) {
    const int flag = active ? 1 : 0;
    const int n = sycl::reduce_over_group(sg, flag, sycl::plus<int>());
    if (n == 0) {
      return 0;
    }
    const int rank =
        sycl::exclusive_scan_over_group(sg, flag, sycl::plus<int>());
    int first = 0;
    if (lane == 0) {
      first = atomicAdd(cursor, n);
    }
    first = sycl::group_broadcast(sg, first, 0);
    end = first + n;
    return first + rank;
  };

  sycl::group_barrier(group);

  // Pass one: the elements of the bins before fastBin take slots [0, accepted),
  // and those of fastBin fill [accepted, topK) as if they were all tied, while
  // their minimum and maximum key are taken. If they are tied, which is what
  // brings a row here, that is already the answer; otherwise the passes below
  // rank them and the final one rewrites [accepted, topK).
  const int need0 = topK - accepted;
  uint32_t lo = 0xffffffffu;
  uint32_t hi = 0;
  bool tiesOpen = true;
  walk([&](float x, int j, bool valid) {
    bool in = false;
    bool tie = false;
    if (valid) {
      const int bin = static_cast<int>(extractFastBinIdx<kNumBins>(x));
      in = bin < fastBin;
      if (bin == fastBin) {
        const uint32_t key = convert_to_uint32(x);
        lo = sycl::min(lo, key);
        hi = sycl::max(hi, key);
        tie = tiesOpen;
      }
    }
    int end = 0;
    const int inSlot = claim(inCursor, in, end);
    if (in) {
      outIndices[inSlot] = j;
    }
    end = 0;
    const int tieSlot = claim(tieCursor, tie, end);
    if (tie && tieSlot < need0) {
      outIndices[accepted + tieSlot] = j;
    }
    tiesOpen = tiesOpen && end < need0;
  });
  const uint32_t minKey =
      sycl::reduce_over_group(group, lo, sycl::minimum<uint32_t>());
  const uint32_t range =
      sycl::reduce_over_group(group, hi, sycl::maximum<uint32_t>()) - minKey;
  if (range == 0) {
    sycl::group_barrier(group);
    return;
  }
  // Every claim of pass one is done.
  sycl::group_barrier(group);
  if (tid == 0) {
    *inCursor = accepted;
    *tieCursor = 0;
  }

  // The fastBin elements still in play are those whose key offset, shifted
  // right by `shift`, equals `prefix`; `need` of them complete the top-k. The
  // barrier of the first histogram pass publishes the reset cursors.
  int need = need0;
  int shift = 0;
  uint32_t prefix = 0;
  {
    const int rangeBits = 32 - static_cast<int>(sycl::clz(range));
    int hiShift = 32;
    shift = sycl::max(0, rangeBits - kTieSelectDigitBits);
    while (true) {
      int* myBuckets = hist + tid * kBucketsPerThread;
      int counts[kBucketsPerThread] = {};
      binsStore(myBuckets, counts);
      sycl::group_barrier(group);

      const uint32_t digitMask =
          hiShift == 32 ? static_cast<uint32_t>(kNumDigitBuckets - 1)
                        : (1u << (hiShift - shift)) - 1;
      walk([&](float x, int /* j */, bool valid) {
        bool active = valid && inFastBin(x);
        uint32_t off = 0;
        if (active) {
          off = convert_to_uint32(x) - minKey;
          active = hiShift == 32 || (off >> hiShift) == prefix;
        }
        aggregatedInc(
            hist, active, static_cast<int>((off >> shift) & digitMask));
      });
      sycl::group_barrier(group);

      binsLoad(myBuckets, counts);
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kBucketsPerThread; ++i) {
        sum += counts[i];
      }
      int total;
      int before =
          blockExclusiveScan(group, sg, lane, sgId, scanTmp, sum, total);
      // The candidates number at least need, so one bucket brackets it.
      if (before < need && before + sum >= need) {
#pragma unroll
        for (int i = 0; i < kBucketsPerThread; ++i) {
          if (before < need && before + counts[i] >= need) {
            selVars[0] = tid * kBucketsPerThread + i;
            selVars[1] = before;
            selVars[2] = counts[i];
          }
          before += counts[i];
        }
      }
      sycl::group_barrier(group);

      const uint32_t bucket = static_cast<uint32_t>(selVars[0]);
      prefix = hiShift == 32 ? bucket : (prefix << (hiShift - shift)) | bucket;
      need -= selVars[1];
      const int count = selVars[2];
      hiShift = shift;
      if (count == need || shift == 0) {
        break;
      }
      shift = sycl::max(0, shift - kTieSelectDigitBits);
      // selVars is rewritten by the next scan only after the barriers of its
      // histogram pass, which every thread reaches after the reads above.
    }
  }

  // The elements of fastBin below the final bucket take slots
  // [accepted, topK - need), and `need` of the final bucket's take the rest.
  // Once a subgroup sees the tie slots filled it stops claiming them.
  const int tieBase = topK - need;
  tiesOpen = true;
  walk([&](float x, int j, bool valid) {
    bool in = false;
    bool tie = false;
    if (valid && inFastBin(x)) {
      const uint32_t top = (convert_to_uint32(x) - minKey) >> shift;
      in = top < prefix;
      tie = tiesOpen && top == prefix;
    }
    int end = 0;
    const int inSlot = claim(inCursor, in, end);
    if (in) {
      outIndices[inSlot] = j;
    }
    end = 0;
    const int tieSlot = claim(tieCursor, tie, end);
    if (tie && tieSlot < need) {
      outIndices[tieBase + tieSlot] = j;
    }
    tiesOpen = tiesOpen && end < need;
  });
  sycl::group_barrier(group);
}

// One-pass, reduction-based selection. Returns false without touching global
// memory if the row is not one it can finish, and the caller splits the
// threshold bin further.
//
//   * The bins are split evenly over the threads, so clearing the histogram is
//     kBinsPerThread stores, the prefix sum is one value per thread plus a
//     local walk, and the threshold bin needs neither a search nor a broadcast:
//     the thread whose own prefix brackets topK is the one that owns it.
//
//   * Nothing is staged in shared memory. Slots come from the per-bin prefix
//     sums and per-subgroup packing, so the accepted indices go straight to
//     global memory. The output order is unspecified.
//
//   * A row of up to kFastMaxRegItems * kNumThreads elements is read once; each
//     thread keeps its elements in registers between the histogram and the
//     scatter (kRegResident). Longer rows re-read in chunks.
//
//   * The elements sharing the threshold bin are compacted and ranked pairwise
//     (or, when there are many, by a radix select) over the few elements of
//     that bin.
template <int kCopies, bool kRegResident>
static bool topKPerRowReduce(
    const sycl::nd_item<3>& item,
    const float* rowBase,
    int rowLen,
    int* outIndices,
    int* smemBins,
    int* smemTmp,
    int* itemIndices,
    float* itemLogits,
    int* smemThresholdBinIdx,
    int* smemFinalBinSize,
    int* smemFoundTopKValues,
    int topK) {
  constexpr int kBinsPerThread = kNumBins / kNumThreads;
  static_assert(
      kBinsPerThread * kNumThreads == kNumBins,
      "the bins are split evenly over the threads");
  static_assert(
      kFastMaxCandidates <= kNumFinalItems, "shares the final-items slots");
  static_assert(
      kNumBins * kCopies <= kNumBins * kLongHistCopies,
      "the bins and their copies fit the histogram");

  sycl::group group = item.get_group();
  auto sg = item.get_sub_group();
  const int threadIdx_x = item.get_local_id(0);
  const int lane_id = threadIdx_x % SUBGROUP_SIZE;
  const int sg_id = threadIdx_x / SUBGROUP_SIZE;
  const int binBase = threadIdx_x * kBinsPerThread;
  // Output and candidate cursors, in the scan scratch above the
  // kNumSubGroups + 1 slots blockExclusiveScan() uses.
  constexpr int kCandCursor = SUBGROUP_SIZE - 8;
  constexpr int kAcceptCursor = kCandCursor + 4;
  static_assert(kNumSubGroups + 1 <= kCandCursor);
  int* acceptCursor = smemTmp + kAcceptCursor;
  int* candCursor = smemTmp + kCandCursor;
  const int histCopy = lane_id % kCopies;

  constexpr int kChunk = kFastMaxRegItems * kNumThreads;
  constexpr int kNumReg = kRegResident ? kFastMaxRegItems : 1;
  float regVal[kNumReg];
  // The bin of regVal[i], kept from pass one so pass two classifies with an
  // integer compare instead of extracting the bin from the float again.
  // Elements past the end of the row get a bin one past the last, which no
  // threshold bin can match, so they are rejected without a separate bounds
  // test.
  int regBin[kNumReg];

  // The element this thread keeps in register slot i. A vec4 layout
  // (four consecutive elements per thread, one 16-byte load) was measured here
  // too: it left the read time unchanged, so the row read is bandwidth bound
  // rather than request bound, and the plain strided order is kept because it
  // costs a few instructions less further down.
  auto regOffset = [&](int i) { return threadIdx_x + i * kNumThreads; };

  // How many of this thread's kFastMaxRegItems slots hold an element of the
  // row. Every loop below stops there instead of running the full
  // kFastMaxRegItems and masking, because the slots are handed out with a
  // stride of kNumThreads: the 32 lanes of a subgroup agree on this count to
  // within one, so a short row really does leave the loop early.
  const int regItems =
      kRegResident ? (rowLen - threadIdx_x + kNumThreads - 1) / kNumThreads
                   : kFastMaxRegItems;

  // The chunked passes read a unit-stride row as 16-byte vectors, two per
  // thread per chunk: the first `head` elements up to 16-byte alignment are
  // taken on their own. With only a few long rows in flight a work-group has
  // to keep its own reads streaming, and a vec4 load moves four times as much
  // per message as the strided scalar ones.
  static_assert(kRegResident || kFastMaxRegItems % 4 == 0);
  const bool vecLoads = !kRegResident && rowLen >= kVecRowMinLen;
  const int head =
      vecLoads
          ? sycl::min(
                static_cast<int>(
                    (16 - reinterpret_cast<uintptr_t>(rowBase) % 16) % 16 / 4),
                rowLen)
          : 0;
  const int bodyLen = rowLen - head;
  // The chunked passes start each row at a chunk picked by its work-group.
  // Rows are laid out at a fixed pitch, often a power of two, and rows that
  // all start their read at the same offset then hit the same memory channels
  // at the same time: at 128 rows of 2^20 that doubled the time of a pass.
  const int numChunks = (bodyLen + kChunk - 1) / kChunk;
  const int firstChunk =
      static_cast<int>(item.get_group_linear_id() % numChunks);
  auto chunkAt = [&](int c) {
    const int k = c + firstChunk;
    return (k < numChunks ? k : k - numChunks) * kChunk;
  };
  auto loadChunk = [&](int chunk,
                       float (&val)[kFastMaxRegItems],
                       int (&off)[kFastMaxRegItems]) {
    if (vecLoads) {
      const float* body = rowBase + head;
#pragma unroll
      for (int j = 0; j < kFastMaxRegItems / 4; ++j) {
        const int o = chunk + (threadIdx_x + j * kNumThreads) * 4;
        if (o + 3 < bodyLen) {
          const sycl::float4 w =
              *reinterpret_cast<const sycl::float4*>(body + o);
          val[4 * j] = w.x();
          val[4 * j + 1] = w.y();
          val[4 * j + 2] = w.z();
          val[4 * j + 3] = w.w();
        } else {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            val[4 * j + e] = o + e < bodyLen ? body[o + e] : 0.f;
          }
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          off[4 * j + e] = o + e < bodyLen ? head + o + e : rowLen;
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < kFastMaxRegItems; ++i) {
        off[i] = chunk + threadIdx_x + i * kNumThreads;
        val[i] = off[i] < rowLen ? rowBase[off[i]] : 0.f;
        off[i] = off[i] < rowLen ? off[i] : rowLen;
      }
    }
  };

  {
    int zero[kBinsPerThread * kCopies] = {};
    binsStore<kBinsPerThread * kCopies>(smemBins + binBase * kCopies, zero);
  }
  sycl::group_barrier(group);

  // Pass one over the row: the histogram. These atomics are spread over
  // kNumBins addresses, ~one collision per subgroup, and are not the
  // contended ones the scatter below removes.
  //
  // Reading the row before the barrier that publishes the cleared histogram, so
  // that the barrier waits behind loads already in flight, was measured and
  // lost: the threads then reach their loads at different times and the row
  // reads stop arriving as one front.
  if constexpr (kRegResident) {
    // Both loops run over the whole register set before the next one starts:
    // folding the atomicAdd() into the fill would make every iteration wait on
    // its own load, so a row would cost one memory latency per element instead
    // of one in total.
#pragma unroll
    for (int i = 0; i < kFastMaxRegItems; ++i) {
      if (i >= regItems) {
        break;
      }
      regVal[i] = rowBase[regOffset(i)];
    }
#pragma unroll
    for (int i = 0; i < kFastMaxRegItems; ++i) {
      if (i >= regItems) {
        break;
      }
      regBin[i] = extractFastBinIdx<kNumBins>(regVal[i]);
      atomicAdd(&smemBins[regBin[i] * kCopies + histCopy], 1);
    }
  } else {
    // Same two-loop shape as the register-resident case above, one chunk at a
    // time: a chunk is staged in registers before any of its atomics, so a
    // chunk costs one memory latency instead of one per element. Walking the
    // row with the atomic folded into the load, here and in pass two, was
    // measured slower.
    if (threadIdx_x < head) {
      atomicAdd(
          &smemBins
              [extractFastBinIdx<kNumBins>(rowBase[threadIdx_x]) * kCopies +
               histCopy],
          1);
    }
    // Clustered logits put most of a row into a few bins, and the lanes of
    // one atomic that hit the same address serialise even across the copies:
    // at 2730 rows of 49152 that doubled this pass. So each subgroup samples
    // one element per lane and picks the window of kHotBins consecutive bins
    // that covers most of the sample, and if that is most of it, counts the
    // elements of the window in registers, kHotBits per bin in one 64-bit
    // word, with one atomic per bin at the end of the row. The counts are
    // exact whatever the sample; it only decides which bins skip the atomics.
    constexpr int kHotBins = 4;
    constexpr int kHotBits = 16;
    constexpr int kHotMinLanes = 16;
    int hotLo = 0;
    bool hotLoop = false;
    {
      const int s = threadIdx_x < rowLen ? threadIdx_x : 0;
      const int sample =
          static_cast<int>(extractFastBinIdx<kNumBins>(rowBase[s]));
      const int lead = sycl::group_broadcast(sg, sample, 0);
      int best = -1;
#pragma unroll
      for (int w = -(kHotBins - 1); w <= kHotBins - 1; ++w) {
        const int lo = sycl::clamp(lead + w, 0, kNumBins - kHotBins);
        uint32_t mask;
        sycl::ext::oneapi::group_ballot(
            sg, static_cast<uint32_t>(sample - lo) < kHotBins)
            .extract_bits(mask);
        const int covered = static_cast<int>(sycl::popcount(mask));
        if (covered > best) {
          best = covered;
          hotLo = lo;
        }
      }
      // A lane counts at most rowLen / kNumThreads elements per bin.
      hotLoop = best >= kHotMinLanes &&
                rowLen / kNumThreads < (1 << kHotBits) - kFastMaxRegItems;
    }
    if (hotLoop) {
      uint64_t hotCount = 0;
      for (int c = 0; c < numChunks; ++c) {
        const int chunk = chunkAt(c);
        float chunkVal[kFastMaxRegItems];
        int chunkOff[kFastMaxRegItems];
        loadChunk(chunk, chunkVal, chunkOff);
#pragma unroll
        for (int i = 0; i < kFastMaxRegItems; ++i) {
          const int bin =
              chunkOff[i] < rowLen
                  ? static_cast<int>(extractFastBinIdx<kNumBins>(chunkVal[i]))
                  : -1;
          const uint32_t d = static_cast<uint32_t>(bin - hotLo);
          if (d < kHotBins) {
            hotCount += uint64_t{1} << (d * kHotBits);
          } else if (bin >= 0) {
            atomicAdd(&smemBins[bin * kCopies + histCopy], 1);
          }
        }
      }
#pragma unroll
      for (int k = 0; k < kHotBins; ++k) {
        const int count = static_cast<int>(
            (hotCount >> (k * kHotBits)) & ((uint64_t{1} << kHotBits) - 1));
        const int total = sycl::reduce_over_group(sg, count, sycl::plus<int>());
        if (lane_id == 0 && total > 0) {
          atomicAdd(&smemBins[(hotLo + k) * kCopies], total);
        }
      }
    } else {
      for (int c = 0; c < numChunks; ++c) {
        const int chunk = chunkAt(c);
        float chunkVal[kFastMaxRegItems];
        int chunkOff[kFastMaxRegItems];
        loadChunk(chunk, chunkVal, chunkOff);
#pragma unroll
        for (int i = 0; i < kFastMaxRegItems; ++i) {
          if (chunkOff[i] < rowLen) {
            atomicAdd(
                &smemBins
                    [extractFastBinIdx<kNumBins>(chunkVal[i]) * kCopies +
                     histCopy],
                1);
          }
        }
      }
    }
  }
  sycl::group_barrier(group);

  // The prefix sum, and with it the threshold bin: the one bin whose exclusive
  // prefix is below topK and whose inclusive prefix is not. It is unique, an
  // empty bin can never be it, and because the prefix is non-decreasing every
  // earlier bin is fully accepted and every later one fully rejected. So that
  // one index classifies the whole row, and each bin's exclusive prefix is
  // where its elements belong in the output; both go back into the histogram
  // slots for pass two to use.
  int threadSum = 0;
  int binCount[kBinsPerThread];
  {
    int raw[kBinsPerThread * kCopies];
    binsLoad<kBinsPerThread * kCopies>(smemBins + binBase * kCopies, raw);
#pragma unroll
    for (int b = 0; b < kBinsPerThread; ++b) {
      binCount[b] = 0;
#pragma unroll
      for (int c = 0; c < kCopies; ++c) {
        binCount[b] += raw[b * kCopies + c];
      }
    }
  }
#pragma unroll
  for (int b = 0; b < kBinsPerThread; ++b) {
    threadSum += binCount[b];
  }
  // The row total the scan hands back is what the multi-step path needs to keep
  // a running output cursor; this path takes its cursor from the threshold
  // bin's own prefix instead, so it is written and never read.
  int rowTotal = 0;
  int prefix = blockExclusiveScan(
      group, sg, lane_id, sg_id, smemTmp, threadSum, rowTotal);
#pragma unroll
  for (int b = 0; b < kBinsPerThread; ++b) {
    const int inclusive = prefix + binCount[b];
    if (prefix < topK && topK <= inclusive) {
      smemThresholdBinIdx[0] = binBase + b;
      smemFinalBinSize[0] = binCount[b];
      smemFoundTopKValues[0] = prefix;
      *acceptCursor = 0;
      *candCursor = prefix;
    }
    prefix = inclusive;
  }
  sycl::group_barrier(group);

  // A large candidate set lives in the histogram, which nothing reads after
  // the scan above.
  // The items buffer it replaces then holds the radix select below.
  int* const selHist = itemIndices;
  int* const selVars = reinterpret_cast<int*>(itemLogits);
  const bool bigCandidates = !kRegResident && rowLen >= kLongRowMinLen;
  if (bigCandidates) {
    itemLogits = reinterpret_cast<float*>(smemBins);
    itemIndices = smemBins + kNumBins * kCopies / 2;
  }

  // rowLen > topK, so some bin is always the threshold bin. Bail out only when
  // its elements would not fit the compacted candidate buffer: nothing has been
  // written to global memory yet, so the caller can start over.
  if (smemFinalBinSize[0] >
      (bigCandidates ? kNumBins * kCopies / 2 : kFastMaxCandidates)) {
    return false;
  }

  // Pass two. Elements below the threshold bin go straight to global memory, in
  // descending bin order rather than index order. Elements of the threshold bin
  // land in the candidate buffer instead, ready for the rank below. Everything
  // above it has nowhere to go.
  const int thresholdBin = smemThresholdBinIdx[0];
  const int candBase = smemFoundTopKValues[0];
  // Skipping a thread whose elements are all rejected was measured here and
  // lost: the test is per thread but the hardware is per subgroup, which walks
  // the slots either way. Only a subgroup-uniform predicate can skip work here.
  //
  // Accepted elements are packed per subgroup: a ballot per register slot gives
  // each lane its place among the subgroup's accepted elements, and one atomic
  // per subgroup reserves the whole run. Taking the slot from the element's
  // own bin instead was one shared memory atomic per element, and on real
  // logits the accepted elements crowd into a few fp16 bins, so the lanes of
  // one atomic hit the same few addresses and serialise; the ranks are
  // consecutive, so the stores coalesce as well.
  const uint32_t lowerLanes = (1u << lane_id) - 1u;
  auto ballot = [&](bool pred) {
    uint32_t bits;
    sycl::ext::oneapi::group_ballot(sg, pred).extract_bits(bits);
    return bits;
  };
  // Candidates are packed the same way. They all share one bin by definition,
  // so a counter bumped per element is the most contended atomic of all.
  auto emitCandidates = [&](bool pred, int off, float logit) {
    const uint32_t mask = ballot(pred);
    if (mask == 0) {
      return;
    }
    int base = 0;
    if (lane_id == 0) {
      base = atomicAdd(candCursor, static_cast<int>(sycl::popcount(mask)));
    }
    base = sycl::group_broadcast(sg, base, 0) - candBase;
    if (pred) {
      const int slot = base + sycl::popcount(mask & lowerLanes);
      itemLogits[slot] = logit;
      itemIndices[slot] = off;
    }
  };
  auto reserve = [&](int count) {
    int base = 0;
    if (lane_id == 0 && count > 0) {
      base = atomicAdd(acceptCursor, count);
    }
    return sycl::group_broadcast(sg, base, 0);
  };
  if constexpr (kRegResident) {
    // The subgroup's lane 0 holds the most slots, so its count bounds the
    // loop for every lane and the ballots see the whole subgroup.
    const int sgItems = sycl::group_broadcast(sg, regItems, 0);
    uint32_t accMask[kFastMaxRegItems];
    int sgAccepted = 0;
#pragma unroll
    for (int i = 0; i < kFastMaxRegItems; ++i) {
      if (i >= sgItems) {
        break;
      }
      const bool live = i < regItems;
      accMask[i] = ballot(live && regBin[i] < thresholdBin);
      sgAccepted += sycl::popcount(accMask[i]);
      if (live && regBin[i] == thresholdBin) {
        const int slot = atomicAdd(candCursor, 1) - candBase;
        itemLogits[slot] = regVal[i];
        itemIndices[slot] = regOffset(i);
      }
    }
    int base = reserve(sgAccepted);
#pragma unroll
    for (int i = 0; i < kFastMaxRegItems; ++i) {
      if (i >= sgItems) {
        break;
      }
      if ((accMask[i] >> lane_id) & 1u) {
        outIndices[base + sycl::popcount(accMask[i] & lowerLanes)] =
            regOffset(i);
      }
      base += sycl::popcount(accMask[i]);
    }
  } else {
    // The chunk is staged in registers before any of its atomics, for the same
    // reason as in pass one.
    // The head is one more slot, live on the lanes that hold a head element.
    // head is uniform, so every lane takes part in its ballots.
    if (head > 0) {
      const bool live = threadIdx_x < head;
      const float x = live ? rowBase[threadIdx_x] : 0.f;
      const int bin =
          live ? static_cast<int>(extractFastBinIdx<kNumBins>(x)) : kNumBins;
      const uint32_t mask = ballot(bin < thresholdBin);
      const int base = reserve(sycl::popcount(mask));
      if (bin < thresholdBin) {
        outIndices[base + sycl::popcount(mask & lowerLanes)] = threadIdx_x;
      }
      emitCandidates(bin == thresholdBin, threadIdx_x, x);
    }
    // Backwards, so that it starts on the end of the row, which pass one read
    // last and the last level cache still holds.
    // Rows longer than kCandBatchMaxRowLen keep one candidate atomic per slot:
    // batching them was 1.016 slower at 512 rows of 163840 and 1.069 at
    // 262144 (topK=512), growing with the row, against 1.004 at 131072 and
    // 0.94-0.97 at 49152-98304 (topK=2048). The
    // two loops are whole, as a branch per slot cost the batched loop its gain.
    if (rowLen > kCandBatchMaxRowLen) {
      for (int c = numChunks - 1; c >= 0; --c) {
        const int chunk = chunkAt(c);
        float chunkVal[kFastMaxRegItems];
        int chunkOff[kFastMaxRegItems];
        loadChunk(chunk, chunkVal, chunkOff);
        uint32_t accMask[kFastMaxRegItems];
        int sgAccepted = 0;
#pragma unroll
        for (int i = 0; i < kFastMaxRegItems; ++i) {
          // Elements past the end of the row get a bin one past the last, which
          // no threshold bin can match.
          const int bin =
              chunkOff[i] < rowLen
                  ? static_cast<int>(extractFastBinIdx<kNumBins>(chunkVal[i]))
                  : kNumBins;
          accMask[i] = ballot(bin < thresholdBin);
          sgAccepted += sycl::popcount(accMask[i]);
          emitCandidates(bin == thresholdBin, chunkOff[i], chunkVal[i]);
        }
        int base = reserve(sgAccepted);
#pragma unroll
        for (int i = 0; i < kFastMaxRegItems; ++i) {
          if ((accMask[i] >> lane_id) & 1u) {
            outIndices[base + sycl::popcount(accMask[i] & lowerLanes)] =
                chunkOff[i];
          }
          base += sycl::popcount(accMask[i]);
        }
      }
    } else {
      for (int c = numChunks - 1; c >= 0; --c) {
        const int chunk = chunkAt(c);
        float chunkVal[kFastMaxRegItems];
        int chunkOff[kFastMaxRegItems];
        loadChunk(chunk, chunkVal, chunkOff);
        uint32_t accMask[kFastMaxRegItems];
        uint32_t candMask[kFastMaxRegItems];
        int sgAccepted = 0;
        int sgCands = 0;
#pragma unroll
        for (int i = 0; i < kFastMaxRegItems; ++i) {
          // Elements past the end of the row get a bin one past the last, which
          // no threshold bin can match.
          const int bin =
              chunkOff[i] < rowLen
                  ? static_cast<int>(extractFastBinIdx<kNumBins>(chunkVal[i]))
                  : kNumBins;
          accMask[i] = ballot(bin < thresholdBin);
          sgAccepted += sycl::popcount(accMask[i]);
          candMask[i] = ballot(bin == thresholdBin);
          sgCands += sycl::popcount(candMask[i]);
        }
        int base = reserve(sgAccepted);
#pragma unroll
        for (int i = 0; i < kFastMaxRegItems; ++i) {
          if ((accMask[i] >> lane_id) & 1u) {
            outIndices[base + sycl::popcount(accMask[i] & lowerLanes)] =
                chunkOff[i];
          }
          base += sycl::popcount(accMask[i]);
        }
        // The candidates of the whole chunk take one atomic, like the accepted
        // ones: on clustered logits most slots hold candidates, and one atomic
        // per slot on the one cursor every subgroup shares serialised them.
        if (sgCands > 0) {
          int cand = 0;
          if (lane_id == 0) {
            cand = atomicAdd(candCursor, sgCands);
          }
          cand = sycl::group_broadcast(sg, cand, 0) - candBase;
#pragma unroll
          for (int i = 0; i < kFastMaxRegItems; ++i) {
            if ((candMask[i] >> lane_id) & 1u) {
              const int slot = cand + sycl::popcount(candMask[i] & lowerLanes);
              itemLogits[slot] = chunkVal[i];
              itemIndices[slot] = chunkOff[i];
            }
            cand += sycl::popcount(candMask[i]);
          }
        }
      }
    }
  }
  sycl::group_barrier(group);

  // The threshold bin, ranked pairwise on the exact float values, ties by
  // index.
  const int numCandidates = smemFinalBinSize[0];
  const int accepted = smemFoundTopKValues[0];
  const int remainedTopK = topK - accepted;
  // A subgroup per candidate, and the whole subgroup walks the candidate list,
  // so the quadratic work is spread over all the threads.
  // A long row's candidate set can hold thousands of elements, and ranking
  // those pairwise is quadratic: at 32 rows of 65536 it doubled the kernel. It
  // is cut down by a radix select instead, kSelBits of the exact float key per
  // pass. The keys are taken as offsets from the smallest one, and the shared
  // fp16 bin keeps those to about 19 bits, so that is two or three passes over
  // the candidates. The ranks within the selected set are not needed: any order
  // of the top-k is a valid output.
  if (numCandidates > kFastMaxCandidates) {
    constexpr int kSelBins = kNumThreads;
    constexpr int kSelBits = constexprLog2(kSelBins);
    static_assert(
        kSelBins <= kNumFinalItems, "the histogram fits the items buffer");
    int* lessCursor = selVars + 3;
    int* tieCursor = selVars + 4;
    if (threadIdx_x == 0) {
      *lessCursor = 0;
      *tieCursor = 0;
    }
    uint32_t lo = 0xffffffffu;
    uint32_t hi = 0;
    for (int i = threadIdx_x; i < numCandidates; i += kNumThreads) {
      const uint32_t key = convert_to_uint32(itemLogits[i]);
      lo = sycl::min(lo, key);
      hi = sycl::max(hi, key);
    }
    const uint32_t minKey =
        sycl::reduce_over_group(group, lo, sycl::minimum<uint32_t>());
    const uint32_t range =
        sycl::reduce_over_group(group, hi, sycl::maximum<uint32_t>()) - minKey;
    // Publishes the cleared cursors when no pass below runs.
    sycl::group_barrier(group);
    auto offsetOf = [&](int i) -> uint64_t {
      return convert_to_uint32(itemLogits[i]) - minKey;
    };
    // Offset bits below `shift` are still open. `prefix` is the threshold's
    // offset above them, and `need` how many of the elements that share it
    // are still to be taken.
    int shift = range == 0 ? 0 : 32 - static_cast<int>(sycl::clz(range));
    uint64_t prefix = 0;
    int need = remainedTopK;
    while (shift > 0) {
      const int low = sycl::max(shift - kSelBits, 0);
      const uint32_t digitMask = (1u << (shift - low)) - 1u;
      selHist[threadIdx_x] = 0;
      sycl::group_barrier(group);
      for (int i = threadIdx_x; i < numCandidates; i += kNumThreads) {
        const uint64_t off = offsetOf(i);
        if ((off >> shift) == prefix) {
          atomicAdd(&selHist[(off >> low) & digitMask], 1);
        }
      }
      sycl::group_barrier(group);
      const int count = selHist[threadIdx_x];
      int total = 0;
      const int before =
          blockExclusiveScan(group, sg, lane_id, sg_id, smemTmp, count, total);
      if (before < need && need <= before + count) {
        selVars[0] = threadIdx_x;
        selVars[1] = before;
        selVars[2] = count;
      }
      sycl::group_barrier(group);
      prefix = (prefix << (shift - low)) | static_cast<uint32_t>(selVars[0]);
      need -= selVars[1];
      shift = low;
      // The whole digit is taken: the bits below it cannot split it.
      if (need == selVars[2]) {
        break;
      }
    }
    // Everything below the threshold prefix is in, and `need` of the
    // elements on it. Packed per subgroup like the emits of pass two.
    const int taken = remainedTopK - need;
    for (int i = threadIdx_x; i - lane_id < numCandidates; i += kNumThreads) {
      const bool live = i < numCandidates;
      const uint64_t top = live ? offsetOf(i) >> shift : ~uint64_t{0};
      const int index = live ? itemIndices[i] : 0;
      uint32_t mask = ballot(top < prefix);
      if (mask != 0) {
        int base = 0;
        if (lane_id == 0) {
          base = atomicAdd(lessCursor, static_cast<int>(sycl::popcount(mask)));
        }
        base = sycl::group_broadcast(sg, base, 0);
        if (top < prefix) {
          outIndices[accepted + base + sycl::popcount(mask & lowerLanes)] =
              index;
        }
      }
      mask = ballot(top == prefix);
      if (mask != 0) {
        int base = 0;
        if (lane_id == 0) {
          base = atomicAdd(tieCursor, static_cast<int>(sycl::popcount(mask)));
        }
        base = sycl::group_broadcast(sg, base, 0);
        const int slot = base + sycl::popcount(mask & lowerLanes);
        if (top == prefix && slot < need) {
          outIndices[accepted + taken + slot] = index;
        }
      }
    }
    return true;
  }
  for (int i = sg_id; i < numCandidates; i += kNumSubGroups) {
    const float logit = itemLogits[i];
    int outIndex = 0;
    for (int j = lane_id; j < numCandidates; j += SUBGROUP_SIZE) {
      const float otherLogit = itemLogits[j];
      if (logit < otherLogit || (logit == otherLogit && i < j)) {
        ++outIndex;
      }
    }
    outIndex = sycl::reduce_over_group(sg, outIndex, sycl::plus<int>());
    if (lane_id == 0 && outIndex < remainedTopK) {
      outIndices[accepted + outIndex] = itemIndices[i];
    }
  }
  return true;
}

// The exact job: selects the topK largest of the rowLen > topK columns at `row`
// (unit column stride) and writes their column indices to outIndices. slm is
// the work-group's arena, used from offset 0.
static void exactRowJob(
    const sycl::nd_item<3>& item,
    sycl::local_accessor<sycl::float4, 1> slm,
    const float* row,
    int rowLen,
    int* outIndices,
    int topK) {
  sycl::group group = item.get_group();
  auto threadIdx_x = item.get_local_id(0);

  char* smem_buf = reinterpret_cast<char*>(
      slm.template get_multi_ptr<sycl::access::decorated::no>().get());

  int32_t* smemOutput = reinterpret_cast<int32_t*>(smem_buf);
  SharedStates* sharedStates =
      reinterpret_cast<SharedStates*>(smem_buf + sharedStatesOffset(topK));

  // Shared memory references
  auto& smemFinal = sharedStates->smemFinal;
  int* smemThresholdBinIdx = &sharedStates->smemThresholdBinIdx;
  int* smemFinalDstIdx = &sharedStates->smemFinalDstIdx;
  int* smemFinalBinSize = &sharedStates->smemFinalBinSize;
  int* smemFoundTopKValues = &sharedStates->smemFoundTopKValues;

  // The reduction path writes its own output, or returns false having written
  // nothing. Three shapes of the same pass: the register-resident one for a row
  // that fits kFastMaxRegItems slots per thread, and for a longer row the
  // chunked one, with the extra histogram copies once the row is long enough to
  // hide their clear and scan.
  bool done;
  if (rowLen <= kFastMaxRegItems * kNumThreads) {
    done = topKPerRowReduce<kHistCopies, /*kRegResident=*/true>(
        item,
        row,
        rowLen,
        outIndices,
        smemFinal.histo.data,
        smemFinal.histo.tmp,
        smemFinal.items.indices,
        smemFinal.items.logits,
        smemThresholdBinIdx,
        smemFinalBinSize,
        smemFoundTopKValues,
        topK);
  } else if (rowLen >= kLongRowMinLen) {
    done = topKPerRowReduce<kLongHistCopies, /*kRegResident=*/false>(
        item,
        row,
        rowLen,
        outIndices,
        smemFinal.histo.data,
        smemFinal.histo.tmp,
        smemFinal.items.indices,
        smemFinal.items.logits,
        smemThresholdBinIdx,
        smemFinalBinSize,
        smemFoundTopKValues,
        topK);
  } else {
    done = topKPerRowReduce<kHistCopies, /*kRegResident=*/false>(
        item,
        row,
        rowLen,
        outIndices,
        smemFinal.histo.data,
        smemFinal.histo.tmp,
        smemFinal.items.indices,
        smemFinal.items.logits,
        smemThresholdBinIdx,
        smemFinalBinSize,
        smemFoundTopKValues,
        topK);
  }
  if (done) {
    return;
  }
  // Everything the reduction path touched is either re-initialised below or
  // cleared by the refine step before it is read.
  sycl::group_barrier(group);
  // The bin it gave up on is read before anything below rewrites it.
  const int refineBin = smemThresholdBinIdx[0];
  const int refineAccepted = smemFoundTopKValues[0];
  // Every thread holds its copy before thread 0 resets them below.
  sycl::group_barrier(group);

  // Initialize values
  if (threadIdx_x == 0) {
    smemFinalDstIdx[0] = 0;
    smemFoundTopKValues[0] = 0;
  }
  sycl::group_barrier(group);

  // Split the reduction path's threshold bin further if its exponent allows it.
  // Same split as the reduction path above: only a long row is worth the finer
  // buckets, and only a long row can have given up with them. Should even its
  // threshold bucket be too large, processTieSelect() finishes the row.
  const bool refined =
      fastBinIsRefinable<kNumBins>(refineBin) &&
      processRefineStep(
          item,
          row,
          rowLen,
          refineBin,
          rowLen >= kLongRowMinLen ? kLongRefineShift : kLongRefineShift + 2,
          refineAccepted,
          smemOutput,
          smemThresholdBinIdx,
          smemFinalDstIdx,
          smemFinalBinSize,
          smemFoundTopKValues,
          &sharedStates->smemRefineDstIdx,
          smemFinal,
          topK);
  if (!refined) {
    processTieSelect(
        item,
        row,
        rowLen,
        refineBin,
        refineAccepted,
        outIndices,
        smemFinal.histo.data,
        smemFinal.histo.tmp,
        smemFinal.items.indices,
        topK);
    return;
  }

  // Rank the threshold bucket's elements pairwise, ties by index.
  const auto baseIdx = smemFoundTopKValues[0];
  for (int i = threadIdx_x; i < smemFinalDstIdx[0]; i += kNumThreads) {
    int outIndex = 0;
    auto logit = smemFinal.items.logits[i];
    for (int j = 0; j < smemFinalDstIdx[0]; j++) {
      auto otherLogit = smemFinal.items.logits[j];
      if (logit < otherLogit || (logit == otherLogit && i < j)) {
        outIndex++;
      }
    }
    // Store if outIndex is in bounds
    auto remainedTopK = topK - baseIdx;
    if (outIndex < remainedTopK) {
      smemOutput[outIndex + baseIdx] = smemFinal.items.indices[i];
    }
  }
  sycl::group_barrier(group);

  // Store to global memory.
  for (int i = threadIdx_x; i < topK; i += kNumThreads) {
    outIndices[i] = smemOutput[i];
  }
}

// Work-groups a core keeps resident: one (a 512-wide group per Xe core; two do
// not fit the SLM arena of topK 2048 and gained nothing). kSlmReserve is the
// SLM left unused per work-group when the arena is sized against the core's
// SLM.
static constexpr size_t kSlmReserve = 4096;

// Largest topK the kernel can run (the dispatch rule takes fewer: topKTuned).
static constexpr int64_t kMaxTopK = 4096;

// Group size G = n / (kGroupDiv * topK), rounded down to a power of two and
// clamped to [kGMin, kGMax]: ~3 * topK delegates, which leaves 2.1-3.1 * topK
// keys >= t on randn and clustered data.
// kGMin: one float4 per lane and group; kGMax bounds the candidates
// ((topK - 1) * G) and the G keys a lane holds in registers.
static constexpr int kGroupDiv = 3;
static constexpr int kGMin = 4;
static constexpr int kGMax = 16;
// The copies move whole float4s between kCopyAlign-aligned addresses: a row
// starts h = (address % kCopyAlign) / 4 floats into its first line, at most
// kMaxShift, and the stage holds it shifted by h. The first kLineFloats floats
// of the stage (the head) are loaded one by one when h > 0.
static constexpr int kCopyAlign = 64;
static constexpr int kLineFloats = kCopyAlign / static_cast<int>(sizeof(float));
static constexpr int kMaxShift = kLineFloats - 1;

// Rows from minDelegateLen(topK) columns on run the delegate select, shorter
// ones the job. kMinLenFactor * topK columns give at least 2 * topK delegates
// at G = kGMin (1.2-1.3 * topK candidates on randn rows). Below
// kMinDelegateCols columns the job is faster still (topK 512: rows of 4096 to
// 6143 columns lost 5 % on B70), and rows from
// kGMin * kGroupDiv * topK columns on always have enough delegates. A
// delegate row is never
// shorter than the stage head (topK 1: 16, not 12 columns), so it never ends
// inside it.
static constexpr int kMinLenFactor = 2 * kGMin;
static constexpr int64_t kMinDelegateCols = 6144;
static constexpr int64_t minDelegateLen(int64_t topK) {
  const int64_t len = kMinLenFactor * topK;
  const int64_t atLeast = len > kMinDelegateCols ? len : kMinDelegateCols;
  const int64_t always = kGMin * kGroupDiv * topK;
  const int64_t shortest = atLeast < always ? atLeast : always;
  return shortest > kLineFloats ? shortest : kLineFloats;
}

// Candidate slots by default (8 B each of SLM): 4096 keeps the arena small
// enough that rows up to 262144 columns still fit the global select on BMG's
// 128 KB SLM. A row with more candidates falls back to the job (exact, slower).
static constexpr int kCandCap = 4096;
// Most candidate slots the global select takes when SLM allows (key + position
// = 8 B each).
static constexpr int64_t kGlobalCandCapMax = 16384;

// Radix selects: 11-bit digits (2048 bins: 13-bit digits made the find step
// bank conflicted), at most three passes; a selected bin of at most kSmallSet
// keys is ranked in one subgroup instead.
static constexpr int kRadixBits = 11;
static constexpr int kHistBins = 1 << kRadixBits;
static constexpr int kSmallSet = SUBGROUP_SIZE;

// Stage lengths are multiples of a chunk of the largest group, 32 lanes x
// kGMax floats, and the stage has one such chunk of slack for the shift that
// aligns the rows and for padding the last chunk.
static constexpr int kStageAlign = SUBGROUP_SIZE * kGMax;
// SLM loads per lane that the passes over the stage and the delegates
// (delegates, group list, candidates) keep in flight: they are latency bound,
// and the compiler does not overlap the loads of different loop iterations.
// Unrolling the radix and emit passes as well measured slower.
static constexpr int kLoadsInFlight = 4;
// The same for the pass that streams a row from global memory (rows past the
// stage cap): LSC bandwidth is bytes in flight over the latency. The pass over
// the listed groups keeps kLoadsInFlight: 8 there (32 keys a lane) spill.
static constexpr int kGlobalLoads = 8;
// Widest launch the delegate select takes (measured 1.2-2.8x up to here; from
// 327680 on it loses for most shapes, 524288 down to 0.25x): the global
// select's delegates (n / kGMax words) must fit the arena. Wider launches use
// the radix kernels.
static constexpr int64_t kMaxWidth = 262144;

// Stage padding: the bit pattern of the smallest key, 0.
static constexpr uint32_t kPadBits = 0xffffffffu;

// Words of the misc block of the arena.
enum MiscSlot : int {
  kSel = 0,         // find(): selected bin, -1 if none
  kSelAbove,        // keys in the bins above it
  kSelCount,        // keys in it
  kSmallN,          // keys collected for the small-set finish
  kGroupN,          // group list entries
  kCandN,           // candidates (may exceed the slots)
  kTieN,            // ties appended after the candidates
  kEqTaken,         // emit: keys == T taken
  kOutN,            // emit: output slots taken
  kSubTotals = 32,  // find(): one count per subgroup
  kMaxKey = 64,     // largest delegate
  kLowKey,          // a lower bound of t
  kSmall = 96,      // kSmallSet keys of the small-set finish
  kMiscInts = 128
};
static_assert(kSubTotals + kNumSubGroups <= kMaxKey, "misc slots overlap");
static_assert(kSmall + kSmallSet <= kMiscInts, "misc slots overlap");
static_assert(kHistBins % kNumThreads == 0, "find() splits the bins evenly");
static_assert(kNumSubGroups <= SUBGROUP_SIZE, "find() scans one per lane");

// Byte offsets (64 B aligned) of a select's state in the arena.
struct SelectOffsets {
  int delegates;   // uint32 per group, padded to a multiple of 128
  int groups;      // int32 [topK]: groups whose delegate is > t
  int candidates;  // [candCap] of candidates and ties: StagedRow int32 stage
                   // positions, GlobalRow uint32 keys
  int positions;   // GlobalRow: int32 [candCap] stage positions
  int histogram;   // int32 [kHistBins]
  int misc;        // int32 [kMiscInts]
};

// The SLM arena of a work-group, set on the host. Every offset is a multiple of
// 64 B, and the arena itself is allocated as float4s, so the 16 B vector
// accesses are aligned by construction. The staged select keeps the
// stage at offset 0, the global one its delegates; exactRowJob uses the
// arena from offset 0. One row per work-group: the three never meet.
struct ArenaLayout {
  int stageCap;          // longest row the staged select takes, 0 for none
  int globalCap;         // longest row the global select takes, 0 for none
  int candCap;           // candidate slots of the staged select
  int globalCandCap;     // candidate slots of the global select
  SelectOffsets staged;  // StagedRow
  SelectOffsets global;  // GlobalRow
  size_t bytes;          // the whole arena
};

template <typename T>
using SlmPtr = typename sycl::multi_ptr<
    T,
    sycl::access::address_space::local_space,
    sycl::access::decorated::yes>::pointer;

template <typename T>
using SlmAtomic = sycl::atomic_ref<
    T,
    sycl::memory_order::relaxed,
    sycl::memory_scope::work_group,
    sycl::access::address_space::local_space>;

static TOPK_ALWAYS_INLINE uint32_t ballot(const sycl::sub_group& sg, bool p) {
  uint32_t bits = 0;
  sycl::ext::oneapi::group_ballot(sg, p).extract_bits(bits);
  return bits;
}

static TOPK_ALWAYS_INLINE int popc(uint32_t m) {
  return static_cast<int>(sycl::popcount(m));
}

static TOPK_ALWAYS_INLINE int bitWidth(uint32_t v) {
  return v ? 32 - static_cast<int>(sycl::clz(v)) : 0;
}

static constexpr int divUp(int a, int b) { return (a + b - 1) / b; }
static constexpr int roundUp(int a, int b) { return divUp(a, b) * b; }

// The key of float bits (larger float, larger key).
static TOPK_ALWAYS_INLINE uint32_t keyOf(uint32_t bits) {
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// The float bits of a key.
static TOPK_ALWAYS_INLINE uint32_t bitsOf(uint32_t key) {
  return (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
}

// The largest key of a group from the signed max and min of its float bits:
// non-negative floats order as their bits do, and the negative ones, all
// below them, in reverse.
static TOPK_ALWAYS_INLINE uint32_t groupKey(int32_t smax, int32_t smin) {
  return smax >= 0 ? (static_cast<uint32_t>(smax) | 0x80000000u)
                   : ~static_cast<uint32_t>(smin);
}

// "key > t" and "key == t" on float bits, one compare each. As signed
// integers the bits order like the floats for t >= +0; for t <= -0 the larger
// floats are the unsigned bit patterns below t's. The flip turns either into
// an unsigned compare.
struct Threshold {
  uint32_t bits;
  uint32_t flip;
  uint32_t ref;
  TOPK_ALWAYS_INLINE explicit Threshold(uint32_t key)
      : bits(bitsOf(key)),
        flip((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u),
        ref(bits ^ flip) {}
  TOPK_ALWAYS_INLINE bool above(uint32_t b) const { return (b ^ flip) > ref; }
  TOPK_ALWAYS_INLINE bool equal(uint32_t b) const { return b == bits; }
};

// The keys a radix pass counts: lo <= key <= lo + last, digit
// (key - lo) >> shift.
struct Digit {
  uint32_t lo;
  uint32_t last;
  int shift;
  // Keys whose low shift + width bits are free above lo. Capped at the largest
  // key, so that keys below lo cannot wrap around into the range.
  TOPK_ALWAYS_INLINE Digit(uint32_t lo_, int shift_, int width)
      : lo(lo_), shift(shift_) {
    const int bits = shift_ + width;
    const uint32_t span = bits >= 32 ? 0xffffffffu : (1u << bits) - 1u;
    last = sycl::min(span, 0xffffffffu - lo_);
  }
  TOPK_ALWAYS_INLINE bool test(uint32_t key, int& bin) const {
    const uint32_t rel = key - lo;
    bin = static_cast<int>(rel >> shift);
    return rel <= last;
  }
};

// The pow2 group size of a staged row of n >= minDelegateLen(topK) columns.
static TOPK_ALWAYS_INLINE int groupSize(int n, int topK) {
  const int ratio = n / (kGroupDiv * topK);
  int g = kGMin;
  while (2 * g <= ratio && g < kGMax)
    g *= 2;
  return g;
}

// A lower bound of t from one lane's delegates, in the order it computes
// them: the lane cuts its delegates of numChunks chunks into S = ceil(topK /
// WG) runs, and the WG * S >= topK run maxima are distinct delegates, so their
// minimum is at most t. Valid only when every run has a delegate (bounded()).
struct RunBound {
  const int runs;
  const int runLen;
  int inRun = 0;
  int run = 0;
  uint32_t runMax = 0;
  uint32_t low = 0xffffffffu;
  TOPK_ALWAYS_INLINE RunBound(int numChunks, int sgid, int topK)
      : runs(divUp(topK, kNumThreads)),
        runLen(sycl::max(1, divUp(numChunks - sgid, kNumSubGroups) / runs)) {}
  static TOPK_ALWAYS_INLINE bool bounded(int numChunks, int topK) {
    return numChunks / kNumSubGroups >= 2 * divUp(topK, kNumThreads);
  }
  TOPK_ALWAYS_INLINE void add(uint32_t d) {
    runMax = sycl::max(runMax, d);
    if (++inRun == runLen && run < runs - 1) {
      low = sycl::min(low, runMax);
      runMax = 0;
      inRun = 0;
      ++run;
    }
  }
  TOPK_ALWAYS_INLINE uint32_t finish() const { return sycl::min(low, runMax); }
};

// The parts of the delegate select that do not read the row, run by the
// whole work-group: the radix selects of t (over the delegates) and of T
// (over the candidates), the group list and the output. StagedRow (row in SLM)
// and GlobalRow (row in global memory) add the passes over the row. Every
// branch on row data is uniform across the work-group, and every exit is taken
// before the first output write: run() returns false (the caller then runs the
// job on the row) if the candidates overflow their slots or if a select finds
// no bin, which the bounds rule out.
//
// Both paths see the row as the stage holds it: stage[p] = row[p - h] for h <=
// p < h + n and the key-0 pattern elsewhere, h = (row address % kCopyAlign) /
// 4, so that float4 v of the stage is a 16 B-aligned float4 of the row (and the
// copies move whole float4s between 64 B-aligned addresses). Chunk c is stage
// floats [32 G c, 32 G (c + 1)); group j = 32 c + l is the G floats of float4s
// 8 G c + 32 q + l, q < G / 4, so that the lanes of a subgroup read a chunk
// conflict-free from SLM and coalesced from global memory. Candidates and ties
// are stage positions; output index = position - h.
class DelegateSelect {
 protected:
  TOPK_ALWAYS_INLINE DelegateSelect(
      const sycl::nd_item<3>& item,
      SlmPtr<char> arena,
      const SelectOffsets& at,
      const float* row,
      int n,
      int topK,
      int candCap,
      int32_t* out)
      : item_(item),
        sg_(item.get_sub_group()),
        lid_(static_cast<int>(item.get_local_id(0))),
        sgid_(static_cast<int>(sg_.get_group_linear_id())),
        lane_(static_cast<int>(sg_.get_local_linear_id())),
        lower_((1u << lane_) - 1u),
        src_(reinterpret_cast<const uint32_t*>(row)),
        h_(static_cast<int>(
            (reinterpret_cast<uintptr_t>(row) & (kCopyAlign - 1)) /
            sizeof(float))),
        n_(n),
        topK_(topK),
        candCap_(candCap),
        out_(out),
        delegates_(reinterpret_cast<SlmPtr<uint32_t>>(arena + at.delegates)),
        groups_(reinterpret_cast<SlmPtr<int32_t>>(arena + at.groups)),
        hist_(reinterpret_cast<SlmPtr<int32_t>>(arena + at.histogram)),
        misc_(reinterpret_cast<SlmPtr<int32_t>>(arena + at.misc)),
        miscU_(reinterpret_cast<SlmPtr<uint32_t>>(arena + at.misc)) {}

  template <typename T, typename P>
  static TOPK_ALWAYS_INLINE sycl::vec<T, 4> load4(P p, int v) {
    return *reinterpret_cast<SlmPtr<const sycl::vec<T, 4>>>(p + 4 * v);
  }

  // First float4 of group j, and the stage position of its u-th float.
  template <int G>
  static TOPK_ALWAYS_INLINE int groupFloat4(int j) {
    return (j >> 5) * 8 * G + (j & 31);
  }
  static TOPK_ALWAYS_INLINE int stagePos(int v, int u) {
    return 4 * (v + (u >> 2) * SUBGROUP_SIZE) + (u & 3);
  }

  // The delegate of a group from its kQ float4s of float bits.
  template <int kQ>
  static TOPK_ALWAYS_INLINE uint32_t
  groupDelegate(const sycl::vec<int32_t, 4> (&x)[kQ]) {
    sycl::vec<int32_t, 4> hi = x[0];
    sycl::vec<int32_t, 4> lo = x[0];
#pragma unroll
    for (int q = 1; q < kQ; ++q) {
      hi = sycl::max(hi, x[q]);
      lo = sycl::min(lo, x[q]);
    }
    return groupKey(
        sycl::max(sycl::max(hi.x(), hi.y()), sycl::max(hi.z(), hi.w())),
        sycl::min(sycl::min(lo.x(), lo.y()), sycl::min(lo.z(), lo.w())));
  }

  // After the delegate pass: pads the delegates with key 0 to delegateWords
  // and publishes their maximum and the lower bound of t (0 if unbounded).
  // The select state must be initialised and visible.
  TOPK_ALWAYS_INLINE void publishDelegates(
      int numDelegates,
      int delegateWords,
      uint32_t rowMax,
      uint32_t low) const {
    for (int j = numDelegates + lid_; j < delegateWords; j += kNumThreads)
      delegates_[j] = 0;
    rowMax = sycl::reduce_over_group(sg_, rowMax, sycl::maximum<uint32_t>());
    low = sycl::reduce_over_group(sg_, low, sycl::minimum<uint32_t>());
    if (lane_ == 0) {
      SlmAtomic<uint32_t>(miscU_[kMaxKey]).fetch_max(rowMax);
      SlmAtomic<uint32_t>(miscU_[kLowKey]).fetch_min(low);
    }
    sycl::group_barrier(item_.get_group());
  }

  // t, the topK-th largest delegate, over the delegates >= the lower bound.
  TOPK_ALWAYS_INLINE bool delegateThreshold(
      int delegateWords, uint32_t maxKey, uint32_t& t, int& need) const {
    const uint32_t lowKey = miscU_[kLowKey];
    const auto pass = [&](const Digit& d, bool collect) {
      for (int b = sgid_ * SUBGROUP_SIZE; b < delegateWords / 4;
           b += kNumThreads) {
        const auto v = load4<uint32_t>(delegates_, b + lane_);
        const uint32_t key[4] = {v.x(), v.y(), v.z(), v.w()};
        int bin[4];
        bool in[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          in[u] = d.test(key[u], bin[u]);
          if (collect) addSmall(key[u] - lowKey, in[u]);
        }
        if (!collect) histAdd4(bin, in);
      }
    };
    return select(lowKey, bitWidth(maxKey - lowKey), pass, t, need);
  }

  // T, the topK-th largest of numCands > topK candidates (all keys > t).
  // cand(i, pos) returns candidate i's key and sets its stage position.
  template <typename Cand>
  TOPK_ALWAYS_INLINE bool candidateThreshold(
      int numCands,
      uint32_t t,
      uint32_t maxKey,
      const Cand& cand,
      uint32_t& T,
      int& need) const {
    const auto pass = [&](const Digit& d, bool collect) {
      for (int b = sgid_ * SUBGROUP_SIZE; b < numCands; b += kNumThreads) {
        const int i = b + lane_;
        const bool valid = i < numCands;
        int pos = 0;
        const uint32_t key = cand(valid ? i : 0, pos);
        int bin = 0;
        const bool in = valid && d.test(key, bin);
        if (collect)
          addSmall(key - t, in);
        else
          histAdd(bin, in);
      }
    };
    return select(t, bitWidth(maxKey - t), pass, T, need);
  }

  TOPK_ALWAYS_INLINE void addSmall(uint32_t rel, bool in) const {
    if (in) {
      const int p = SlmAtomic<int32_t>(misc_[kSmallN]).fetch_add(1);
      if (p < kSmallSet) miscU_[kSmall + p] = rel;
    }
  }

  TOPK_ALWAYS_INLINE void histAdd(int bin, bool in) const {
    if (in) SlmAtomic<int32_t>(hist_[bin]).fetch_add(1);
  }

  // Four keys per lane; when all 128 of the subgroup hit one bin (tie-heavy
  // rows), one atomic adds them.
  TOPK_ALWAYS_INLINE void
  histAdd4(const int (&bin)[4], const bool (&in)[4]) const {
    const bool same = in[0] && in[1] && in[2] && in[3] && bin[0] == bin[1] &&
                      bin[1] == bin[2] && bin[2] == bin[3];
    const int bin0 = sycl::group_broadcast(sg_, bin[0], 0);
    if (sycl::all_of_group(sg_, same && bin[0] == bin0)) {
      if (lane_ == 0)
        SlmAtomic<int32_t>(hist_[bin0]).fetch_add(4 * SUBGROUP_SIZE);
      return;
    }
#pragma unroll
    for (int u = 0; u < 4; ++u)
      histAdd(bin[u], in[u]);
  }

  // With the histogram of numBins bins complete: the bin, counted from the
  // top, in which the count reaches need, the count above it and in it.
  // Clears the histogram for the next pass. False if no bin qualifies.
  TOPK_ALWAYS_INLINE bool
  find(int numBins, int need, int& bin, int& above, int& count) const {
    constexpr int kPerThread = kHistBins / kNumThreads;
    if (lid_ == 0) misc_[kSel] = -1;
    int c[kPerThread];
    int sum = 0;
#pragma unroll
    for (int u = 0; u < kPerThread; ++u) {
      const int q = lid_ * kPerThread + u;
      c[u] = q < numBins ? hist_[numBins - 1 - q] : 0;
      sum += c[u];
    }
    const int incl = sycl::inclusive_scan_over_group(sg_, sum, sycl::plus<>());
    if (lane_ == SUBGROUP_SIZE - 1) misc_[kSubTotals + sgid_] = incl;
    sycl::group_barrier(item_.get_group());
    const int total = lane_ < kNumSubGroups ? misc_[kSubTotals + lane_] : 0;
    const int prior =
        sycl::exclusive_scan_over_group(sg_, total, sycl::plus<>());
    int a = sycl::group_broadcast(sg_, prior, sgid_) + incl - sum;
#pragma unroll
    for (int u = 0; u < kPerThread; ++u) {
      const int q = lid_ * kPerThread + u;
      if (q < numBins && a < need && a + c[u] >= need) {
        misc_[kSel] = numBins - 1 - q;
        misc_[kSelAbove] = a;
        misc_[kSelCount] = c[u];
        misc_[kSmallN] = 0;
      }
      a += c[u];
    }
#pragma unroll
    for (int u = 0; u < kPerThread; ++u) {
      const int q = lid_ * kPerThread + u;
      if (q < numBins && c[u]) hist_[numBins - 1 - q] = 0;
    }
    sycl::group_barrier(item_.get_group());
    bin = misc_[kSel];
    above = misc_[kSelAbove];
    count = misc_[kSelCount];
    return bin >= 0;
  }

  // Rank the `count` keys of the selected bin (collected as key - base) in
  // subgroup 0: prefix becomes the need-th largest of them.
  TOPK_ALWAYS_INLINE bool
  rankSmall(int count, int& need, uint32_t& prefix) const {
    if (sgid_ == 0) {
      const uint32_t mine = lane_ < count ? miscU_[kSmall + lane_] : 0u;
      int gt = 0;
      int eq = 0;
      for (int j = 0; j < count; ++j) {
        const uint32_t other = sycl::select_from_group(sg_, mine, j);
        gt += other > mine ? 1 : 0;
        eq += other == mine ? 1 : 0;
      }
      const uint32_t hits =
          ballot(sg_, lane_ < count && gt < need && gt + eq >= need);
      const int first = hits ? static_cast<int>(sycl::ctz(hits)) : 0;
      const int gtFirst = sycl::select_from_group(sg_, gt, first);
      if (lane_ == 0) {
        misc_[kSel] = hits ? first : -1;
        misc_[kSelAbove] = gtFirst;
      }
    }
    sycl::group_barrier(item_.get_group());
    const int sel = misc_[kSel];
    if (sel < 0 || misc_[kSmallN] != count) return false;
    need -= misc_[kSelAbove];
    prefix = miscU_[kSmall + sel];
    return true;
  }

  // The topK-th largest key of a set holding at least topK keys >= base, all
  // below base + 2^bits: key, and need = how many keys == key to take (the
  // other topK - need are > key). pass(digit, collect) adds the digit of every
  // key in the digit's range to the histogram, or with collect appends their
  // key - base to the small set. The histogram is zero on entry and exit.
  template <typename Pass>
  TOPK_ALWAYS_INLINE bool
  select(uint32_t base, int bits, const Pass& pass, uint32_t& key, int& need)
      const {
    uint32_t prefix = 0;
    int done = 0;
    need = topK_;
#pragma unroll
    for (int p = 0; p < 3; ++p) {
      if (done < bits) {
        const int width = sycl::min(kRadixBits, bits - done);
        const int shift = bits - done - width;
        pass(Digit(base + prefix, shift, width), false);
        sycl::group_barrier(item_.get_group());
        int bin = 0;
        int above = 0;
        int count = 0;
        if (!find(1 << width, need, bin, above, count)) return false;
        need -= above;
        prefix |= static_cast<uint32_t>(bin) << shift;
        done += width;
        if (done < bits && count <= kSmallSet) {
          pass(Digit(base + prefix, 0, bits - done), true);
          sycl::group_barrier(item_.get_group());
          if (!rankSmall(count, need, prefix)) return false;
          done = bits;
        }
      }
    }
    key = base + prefix;
    return true;
  }

  // The group list: every group whose delegate is > t (fewer than topK). Per
  // subgroup a count pass over its chunks, one atomic, a write pass: an atomic
  // per chunk would put its latency on the loop's dependency chain.
  TOPK_ALWAYS_INLINE void groupList(int numChunks, uint32_t t) const {
    constexpr int kU = kLoadsInFlight;
    constexpr int kStep = kU * kNumSubGroups;
    int mine = 0;
    for (int c0 = sgid_; c0 < numChunks; c0 += kStep) {
      uint32_t d[kU];
      delegatesOf(c0, numChunks, d);
#pragma unroll
      for (int u = 0; u < kU; ++u)
        mine += (c0 + u * kNumSubGroups < numChunks && d[u] > t) ? 1 : 0;
    }
    const int total = sycl::reduce_over_group(sg_, mine, sycl::plus<>());
    int slot = 0;
    if (lane_ == 0 && total)
      slot = SlmAtomic<int32_t>(misc_[kGroupN]).fetch_add(total);
    slot = sycl::group_broadcast(sg_, slot, 0);
    for (int c0 = sgid_; c0 < numChunks; c0 += kStep) {
      uint32_t d[kU];
      delegatesOf(c0, numChunks, d);
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const int c = c0 + u * kNumSubGroups;
        const bool above = c < numChunks && d[u] > t;
        const uint32_t mask = ballot(sg_, above);
        const int w = slot + popc(mask & lower_);
        if (above && w < topK_) groups_[w] = c * SUBGROUP_SIZE + lane_;
        slot += popc(mask);
      }
    }
  }

  // This lane's delegates of chunks c0 + u * kNumSubGroups (clamped to the
  // last chunk; the caller masks those).
  TOPK_ALWAYS_INLINE void
  delegatesOf(int c0, int numChunks, uint32_t (&d)[kLoadsInFlight]) const {
#pragma unroll
    for (int u = 0; u < kLoadsInFlight; ++u)
      d[u] = delegates_
          [sycl::min(c0 + u * kNumSubGroups, numChunks - 1) * SUBGROUP_SIZE +
           lane_];
  }

  // More than topK candidates: write those > T and `need` of those == T.
  // Per subgroup a count pass and one atomic each for its share of the == T
  // quota and for its output slots, then the write pass. cand as in
  // candidateThreshold.
  template <typename Cand>
  TOPK_ALWAYS_INLINE void
  emitSelected(int numCands, uint32_t T, int need, const Cand& cand) const {
    int gt = 0;
    int eq = 0;
    for (int b = sgid_ * SUBGROUP_SIZE; b < numCands; b += kNumThreads) {
      const int i = b + lane_;
      const bool valid = i < numCands;
      int pos = 0;
      const uint32_t key = cand(valid ? i : 0, pos);
      gt += (valid && key > T) ? 1 : 0;
      eq += (valid && key == T) ? 1 : 0;
    }
    const int sgGt = sycl::reduce_over_group(sg_, gt, sycl::plus<>());
    const int sgEq = sycl::reduce_over_group(sg_, eq, sycl::plus<>());
    int eqBase = 0;
    if (lane_ == 0 && sgEq)
      eqBase = SlmAtomic<int32_t>(misc_[kEqTaken]).fetch_add(sgEq);
    eqBase = sycl::group_broadcast(sg_, eqBase, 0);
    const int quota = sycl::clamp(need - eqBase, 0, sgEq);
    int o = 0;
    if (lane_ == 0 && sgGt + quota)
      o = SlmAtomic<int32_t>(misc_[kOutN]).fetch_add(sgGt + quota);
    o = sycl::group_broadcast(sg_, o, 0);
    int eqSeen = 0;
    for (int b = sgid_ * SUBGROUP_SIZE; b < numCands; b += kNumThreads) {
      const int i = b + lane_;
      const bool valid = i < numCands;
      int pos = 0;
      const uint32_t key = cand(valid ? i : 0, pos);
      const bool isEq = valid && key == T;
      const uint32_t eqMask = ballot(sg_, isEq);
      const bool take = (valid && key > T) ||
                        (isEq && eqSeen + popc(eqMask & lower_) < quota);
      const uint32_t takeMask = ballot(sg_, take);
      if (take) out_[o + popc(takeMask & lower_)] = pos - h_;
      o += popc(takeMask);
      eqSeen += popc(eqMask);
    }
  }

  // At most topK candidates, ties included: the first topK positions.
  TOPK_ALWAYS_INLINE void emitPositions(SlmPtr<int32_t> positions) const {
    for (int i = lid_; i < topK_; i += kNumThreads)
      out_[i] = positions[i] - h_;
  }

  const sycl::nd_item<3>& item_;
  const sycl::sub_group sg_;
  const int lid_;
  const int sgid_;
  const int lane_;
  const uint32_t lower_;  // lanes below this one
  const uint32_t* src_;   // the row's float bits from its start
  const int h_;           // stage shift
  const int n_;
  const int topK_;
  const int candCap_;
  int32_t* out_;
  SlmPtr<uint32_t> delegates_;
  SlmPtr<int32_t> groups_;
  SlmPtr<int32_t> hist_;
  SlmPtr<int32_t> misc_;
  SlmPtr<uint32_t> miscU_;
};

// A row of up to the stage cap: copied into SLM once, and every pass reads it
// there.
class StagedRow : public DelegateSelect {
 public:
  TOPK_ALWAYS_INLINE StagedRow(
      const sycl::nd_item<3>& item,
      SlmPtr<char> arena,
      const ArenaLayout& layout,
      const float* row,
      int n,
      int topK,
      int32_t* out)
      : DelegateSelect(
            item, arena, layout.staged, row, n, topK, layout.candCap, out),
        stage_(reinterpret_cast<SlmPtr<uint32_t>>(arena)),
        cands_(
            reinterpret_cast<SlmPtr<int32_t>>(
                arena + layout.staged.candidates)) {}

  template <int G>
  TOPK_ALWAYS_INLINE bool run() const {
    const int nPad = roundUp(h_ + n_, SUBGROUP_SIZE * G);
    const int numChunks = nPad / (SUBGROUP_SIZE * G);
    const int numDelegates = numChunks * SUBGROUP_SIZE;
    const int delegateWords = roundUp(numDelegates, 128);
    stage(nPad);
    delegates<G>(numChunks, numDelegates, delegateWords);
    const uint32_t maxKey = miscU_[kMaxKey];
    uint32_t t = 0;
    int need = 0;
    if (!delegateThreshold(delegateWords, maxKey, t, need)) return false;

    groupList(numChunks, t);
    sycl::group_barrier(item_.get_group());
    const int numGroups = misc_[kGroupN];
    if (numGroups >= topK_) return false;
    const Threshold th(t);
    candidates<G>(numGroups, th);
    sycl::group_barrier(item_.get_group());
    const int numCands = misc_[kCandN];
    if (numCands > candCap_) return false;

    const auto cand = [&](int i, int& pos) {
      pos = cands_[i];
      return keyOf(stage_[pos]);
    };
    if (numCands > topK_) {
      uint32_t T = 0;
      if (!candidateThreshold(numCands, t, maxKey, cand, T, need)) return false;
      emitSelected(numCands, T, need, cand);
    } else {
      if (numCands < topK_) {
        if (!ties<G>(numDelegates, t, th, numCands)) return false;
        sycl::group_barrier(item_.get_group());
      }
      emitPositions(cands_);
    }
    return true;
  }

 private:
  // Fill the stage up to nPad and clear the select state; returns once every
  // thread can read the stage. The up to kMaxShift floats before the first
  // kCopyAlign-aligned one and the up to 3 after the last whole float4 are
  // loaded one by one, so nothing outside [row, row + n) is read. The row is at
  // least kLineFloats long (minDelegateLen), so it ends after the head.
  TOPK_ALWAYS_INLINE void stage(int nPad) const {
    // Float4s: stage[0, kLineFloats) is the head.
    const int first = h_ ? kLineFloats / 4 : 0;
    const int end = (h_ + n_) >> 2;
    for (int v = first + lid_; v < end; v += kNumThreads) {
      const sycl::vec<uint32_t, 4> x =
          *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(src_ + (4 * v - h_));
      stage_[4 * v] = x[0];
      stage_[4 * v + 1] = x[1];
      stage_[4 * v + 2] = x[2];
      stage_[4 * v + 3] = x[3];
    }
    if (lid_ < 4 * first) stage_[lid_] = lid_ < h_ ? kPadBits : src_[lid_ - h_];
    for (int p = 4 * end + lid_; p < nPad; p += kNumThreads)
      stage_[p] = p < h_ + n_ ? src_[p - h_] : kPadBits;
    for (int i = lid_; i < kHistBins; i += kNumThreads)
      hist_[i] = 0;
    if (lid_ < kMiscInts) misc_[lid_] = lid_ == kLowKey ? -1 : 0;
    sycl::group_barrier(item_.get_group());
  }

  // The delegates (key 0 past numDelegates), their maximum and a lower bound
  // of t (RunBound).
  template <int G>
  TOPK_ALWAYS_INLINE void
  delegates(int numChunks, int numDelegates, int delegateWords) const {
    constexpr int kQ = G / 4;
    RunBound bound(numChunks, sgid_, topK_);
    uint32_t rowMax = 0;
    // kU of the subgroup's chunks per iteration, all their loads issued first.
    constexpr int kU = kLoadsInFlight / kQ > 1 ? kLoadsInFlight / kQ : 1;
    for (int c0 = sgid_; c0 < numChunks; c0 += kU * kNumSubGroups) {
      sycl::vec<int32_t, 4> x[kU][kQ];
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const int c = sycl::min(c0 + u * kNumSubGroups, numChunks - 1);
#pragma unroll
        for (int q = 0; q < kQ; ++q)
          x[u][q] =
              load4<int32_t>(stage_, c * 8 * G + q * SUBGROUP_SIZE + lane_);
      }
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const int c = c0 + u * kNumSubGroups;
        if (c < numChunks) {
          const uint32_t d = groupDelegate<kQ>(x[u]);
          delegates_[c * SUBGROUP_SIZE + lane_] = d;
          rowMax = sycl::max(rowMax, d);
          bound.add(d);
        }
      }
    }
    publishDelegates(
        numDelegates,
        delegateWords,
        rowMax,
        RunBound::bounded(numChunks, topK_) ? bound.finish() : 0u);
  }

  // The candidates, every key > t of the listed groups: a count pass, one
  // atomic per subgroup, a write pass. Writes past candCap_ are dropped; the
  // count is kept in full.
  template <int G>
  TOPK_ALWAYS_INLINE void candidates(int numGroups, const Threshold& th) const {
    constexpr int kQ = G / 4;
    // kU listed groups per lane and iteration, their loads issued first.
    constexpr int kU = kLoadsInFlight / kQ > 1 ? kLoadsInFlight / kQ : 1;
    constexpr int kStep = kU * kNumThreads;
    int mine = 0;
    for (int b0 = sgid_ * SUBGROUP_SIZE; b0 < numGroups; b0 += kStep) {
      uint32_t e[kU][G];
      int v[kU];
      groupKeys<G, kU>(b0, numGroups, e, v);
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const bool valid = b0 + u * kNumThreads + lane_ < numGroups;
#pragma unroll
        for (int k = 0; k < G; ++k)
          mine += (valid && th.above(e[u][k])) ? 1 : 0;
      }
    }
    const int total = sycl::reduce_over_group(sg_, mine, sycl::plus<>());
    int slot = 0;
    if (lane_ == 0 && total)
      slot = SlmAtomic<int32_t>(misc_[kCandN]).fetch_add(total);
    slot = sycl::group_broadcast(sg_, slot, 0);
    for (int b0 = sgid_ * SUBGROUP_SIZE; b0 < numGroups; b0 += kStep) {
      uint32_t e[kU][G];
      int v[kU];
      groupKeys<G, kU>(b0, numGroups, e, v);
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const bool valid = b0 + u * kNumThreads + lane_ < numGroups;
        int cnt = 0;
#pragma unroll
        for (int k = 0; k < G; ++k)
          cnt += (valid && th.above(e[u][k])) ? 1 : 0;
        const int off =
            sycl::exclusive_scan_over_group(sg_, cnt, sycl::plus<>());
        int w = slot + off;
#pragma unroll
        for (int k = 0; k < G; ++k) {
          if (valid && th.above(e[u][k])) {
            if (w < candCap_) cands_[w] = stagePos(v[u], k);
            ++w;
          }
        }
        slot += sycl::group_broadcast(sg_, off + cnt, SUBGROUP_SIZE - 1);
      }
    }
  }

  // The float bits of listed groups b0 + u * WG + lane, u < kU, and their
  // first float4 (entries past numGroups > 0 read the last one; the caller
  // masks them).
  template <int G, int kU>
  TOPK_ALWAYS_INLINE void
  groupKeys(int b0, int numGroups, uint32_t (&e)[kU][G], int (&v)[kU]) const {
    constexpr int kQ = G / 4;
#pragma unroll
    for (int u = 0; u < kU; ++u)
      v[u] = groupFloat4<G>(
          groups_[sycl::min(b0 + u * kNumThreads + lane_, numGroups - 1)]);
#pragma unroll
    for (int u = 0; u < kU; ++u) {
#pragma unroll
      for (int q = 0; q < kQ; ++q) {
        const auto x = load4<uint32_t>(stage_, v[u] + q * SUBGROUP_SIZE);
        e[u][4 * q] = x.x();
        e[u][4 * q + 1] = x.y();
        e[u][4 * q + 2] = x.z();
        e[u][4 * q + 3] = x.w();
      }
    }
  }

  // Fewer than topK candidates: append topK - numCands keys == t from the
  // groups whose delegate is >= t (no other holds one), stopping once the
  // work-group has seen enough. Padding has key 0, so t == 0 would match it:
  // only positions inside the row count.
  template <int G>
  TOPK_ALWAYS_INLINE bool
  ties(int numDelegates, uint32_t t, const Threshold& th, int numCands) const {
    constexpr int kQ = G / 4;
    const int needed = topK_ - numCands;
    int have = 0;
    for (int b = 0; b < numDelegates && have < needed; b += kNumThreads) {
      const int j = sycl::min(b + lid_, numDelegates - 1);
      const bool valid = b + lid_ < numDelegates && delegates_[j] >= t;
      const int v = groupFloat4<G>(j);
      uint32_t e[G];
#pragma unroll
      for (int q = 0; q < kQ; ++q) {
        const auto x = load4<uint32_t>(stage_, v + q * SUBGROUP_SIZE);
        e[4 * q] = x.x();
        e[4 * q + 1] = x.y();
        e[4 * q + 2] = x.z();
        e[4 * q + 3] = x.w();
      }
      bool tie[G];
      int cnt = 0;
#pragma unroll
      for (int u = 0; u < G; ++u) {
        const int p = stagePos(v, u);
        tie[u] = valid && th.equal(e[u]) && p >= h_ && p < h_ + n_;
        cnt += tie[u] ? 1 : 0;
      }
      const int off = sycl::exclusive_scan_over_group(sg_, cnt, sycl::plus<>());
      const int total =
          sycl::group_broadcast(sg_, off + cnt, SUBGROUP_SIZE - 1);
      if (total) {
        int base = 0;
        if (lane_ == 0)
          base = SlmAtomic<int32_t>(misc_[kTieN]).fetch_add(total);
        int w = numCands + sycl::group_broadcast(sg_, base, 0) + off;
#pragma unroll
        for (int u = 0; u < G; ++u) {
          if (tie[u]) {
            if (w < topK_) cands_[w] = stagePos(v, u);
            ++w;
          }
        }
      }
      have += sycl::reduce_over_group(item_.get_group(), cnt, sycl::plus<>());
    }
    return have >= needed;
  }

  SlmPtr<uint32_t> stage_;
  SlmPtr<int32_t> cands_;
};

// A row longer than the stage cap, read from global memory: the delegate
// pass streams it with kGlobalLoads float4 loads per lane in flight, and the
// later passes re-read only the listed groups (and, for ties, the groups whose
// delegate is t), which the first pass has just brought into the caches.
// Candidates are (key, position) pairs in SLM, so the select of T and the
// output do not read the row again. Whole float4s inside the row are single
// 16 B loads; the at most 3 floats before the first one and the at most 3
// after the last one are loaded one by one, and stage positions outside the
// row read as the key-0 pattern, so nothing outside [row, row + n) is read.
//
// G follows the row length as in the staged select (groupSize). A row with
// more candidates than slots (e.g. very tie-heavy rows) falls back to the job.
class GlobalRow : public DelegateSelect {
 public:
  TOPK_ALWAYS_INLINE GlobalRow(
      const sycl::nd_item<3>& item,
      SlmPtr<char> arena,
      const ArenaLayout& layout,
      const float* row,
      int n,
      int topK,
      int32_t* out)
      : DelegateSelect(
            item,
            arena,
            layout.global,
            row,
            n,
            topK,
            layout.globalCandCap,
            out),
        candKeys_(
            reinterpret_cast<SlmPtr<uint32_t>>(
                arena + layout.global.candidates)),
        candPos_(
            reinterpret_cast<SlmPtr<int32_t>>(
                arena + layout.global.positions)) {}

  template <int G>
  TOPK_ALWAYS_INLINE bool run() const {
    const int nPad = roundUp(h_ + n_, SUBGROUP_SIZE * G);
    const int numChunks = nPad / (SUBGROUP_SIZE * G);
    const int numDelegates = numChunks * SUBGROUP_SIZE;
    const int delegateWords = roundUp(numDelegates, 128);
    for (int i = lid_; i < kHistBins; i += kNumThreads)
      hist_[i] = 0;
    if (lid_ < kMiscInts) misc_[lid_] = lid_ == kLowKey ? -1 : 0;
    delegates<G>(numChunks, numDelegates, delegateWords);
    const uint32_t maxKey = miscU_[kMaxKey];
    uint32_t t = 0;
    int need = 0;
    if (!delegateThreshold(delegateWords, maxKey, t, need)) return false;

    groupList(numChunks, t);
    sycl::group_barrier(item_.get_group());
    const int numGroups = misc_[kGroupN];
    if (numGroups >= topK_) return false;
    const Threshold th(t);
    candidates<G>(numGroups, th);
    sycl::group_barrier(item_.get_group());
    const int numCands = misc_[kCandN];
    if (numCands > candCap_) return false;

    const auto cand = [&](int i, int& pos) {
      pos = candPos_[i];
      return candKeys_[i];
    };
    if (numCands > topK_) {
      uint32_t T = 0;
      if (!candidateThreshold(numCands, t, maxKey, cand, T, need)) return false;
      emitSelected(numCands, T, need, cand);
    } else {
      if (numCands < topK_) {
        if (!ties<G>(numDelegates, t, th, numCands)) return false;
        sycl::group_barrier(item_.get_group());
      }
      emitPositions(candPos_);
    }
    return true;
  }

 private:
  // Float4 v of the stage, which lies inside the row.
  TOPK_ALWAYS_INLINE sycl::vec<int32_t, 4> inside4(int v) const {
    return *reinterpret_cast<const sycl::vec<int32_t, 4>*>(src_ + (4 * v - h_));
  }

  // Float4 v of the stage, anywhere.
  TOPK_ALWAYS_INLINE sycl::vec<int32_t, 4> stage4(int v) const {
    const int p = 4 * v - h_;  // row index of its first float
    if (p >= 0 && p + 4 <= n_) return inside4(v);
    const auto at = [&](int i) {
      return i >= 0 && i < n_ ? static_cast<int32_t>(src_[i])
                              : static_cast<int32_t>(kPadBits);
    };
    return sycl::vec<int32_t, 4>(at(p), at(p + 1), at(p + 2), at(p + 3));
  }

  // The G float bits of group j's first float4 v.
  template <int G>
  TOPK_ALWAYS_INLINE void groupBits(int v, uint32_t (&e)[G]) const {
#pragma unroll
    for (int q = 0; q < G / 4; ++q) {
      const auto x = stage4(v + q * SUBGROUP_SIZE);
      e[4 * q] = static_cast<uint32_t>(x.x());
      e[4 * q + 1] = static_cast<uint32_t>(x.y());
      e[4 * q + 2] = static_cast<uint32_t>(x.z());
      e[4 * q + 3] = static_cast<uint32_t>(x.w());
    }
  }

  // The delegates as on the staged path. Chunks [firstInside, endInside) lie
  // inside the row; an iteration whose chunks all do loads them unchecked.
  template <int G>
  TOPK_ALWAYS_INLINE void
  delegates(int numChunks, int numDelegates, int delegateWords) const {
    constexpr int kQ = G / 4;
    constexpr int kU = kGlobalLoads / kQ > 1 ? kGlobalLoads / kQ : 1;
    const int firstInside = h_ ? 1 : 0;
    const int endInside = (h_ + n_) / (SUBGROUP_SIZE * G);
    RunBound bound(numChunks, sgid_, topK_);
    uint32_t rowMax = 0;
    const auto keep = [&](int c, uint32_t d) {
      delegates_[c * SUBGROUP_SIZE + lane_] = d;
      rowMax = sycl::max(rowMax, d);
      bound.add(d);
    };
    for (int c0 = sgid_; c0 < numChunks; c0 += kU * kNumSubGroups) {
      if (c0 >= firstInside && c0 + (kU - 1) * kNumSubGroups < endInside) {
        // One address per lane, the kU * kQ loads at constant offsets.
        const auto* p = reinterpret_cast<const sycl::vec<int32_t, 4>*>(
            src_ + (4 * (c0 * 8 * G + lane_) - h_));
        sycl::vec<int32_t, 4> x[kU][kQ];
#pragma unroll
        for (int u = 0; u < kU; ++u) {
#pragma unroll
          for (int q = 0; q < kQ; ++q)
            x[u][q] = p[u * kNumSubGroups * 8 * G + q * SUBGROUP_SIZE];
        }
#pragma unroll
        for (int u = 0; u < kU; ++u)
          keep(c0 + u * kNumSubGroups, groupDelegate<kQ>(x[u]));
      } else {
        // The first and the last chunks, one at a time.
#pragma unroll
        for (int u = 0; u < kU; ++u) {
          const int c = c0 + u * kNumSubGroups;
          if (c < numChunks) {
            sycl::vec<int32_t, 4> x[kQ];
#pragma unroll
            for (int q = 0; q < kQ; ++q)
              x[q] = stage4(c * 8 * G + q * SUBGROUP_SIZE + lane_);
            keep(c, groupDelegate<kQ>(x));
          }
        }
      }
    }
    // The select state cleared by run() is visible from here on.
    sycl::group_barrier(item_.get_group());
    publishDelegates(
        numDelegates,
        delegateWords,
        rowMax,
        RunBound::bounded(numChunks, topK_) ? bound.finish() : 0u);
  }

  // The candidates, every key > t of the listed groups, as (key, position):
  // one pass, one atomic per subgroup and iteration (a second pass would read
  // the groups from global memory again). Writes past candCap_ are dropped;
  // the count is kept in full.
  template <int G>
  TOPK_ALWAYS_INLINE void candidates(int numGroups, const Threshold& th) const {
    constexpr int kQ = G / 4;
    constexpr int kU = kLoadsInFlight / kQ > 1 ? kLoadsInFlight / kQ : 1;
    for (int b0 = sgid_ * SUBGROUP_SIZE; b0 < numGroups;
         b0 += kU * kNumThreads) {
      uint32_t e[kU][G];
      int v[kU];
#pragma unroll
      for (int u = 0; u < kU; ++u)
        v[u] = groupFloat4<G>(
            groups_[sycl::min(b0 + u * kNumThreads + lane_, numGroups - 1)]);
#pragma unroll
      for (int u = 0; u < kU; ++u)
        groupBits<G>(v[u], e[u]);
      int cnt = 0;
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const bool valid = b0 + u * kNumThreads + lane_ < numGroups;
#pragma unroll
        for (int k = 0; k < G; ++k)
          cnt += (valid && th.above(e[u][k])) ? 1 : 0;
      }
      const int off = sycl::exclusive_scan_over_group(sg_, cnt, sycl::plus<>());
      const int total =
          sycl::group_broadcast(sg_, off + cnt, SUBGROUP_SIZE - 1);
      int base = 0;
      if (lane_ == 0 && total)
        base = SlmAtomic<int32_t>(misc_[kCandN]).fetch_add(total);
      int w = sycl::group_broadcast(sg_, base, 0) + off;
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const bool valid = b0 + u * kNumThreads + lane_ < numGroups;
#pragma unroll
        for (int k = 0; k < G; ++k) {
          if (valid && th.above(e[u][k])) {
            if (w < candCap_) {
              candKeys_[w] = keyOf(e[u][k]);
              candPos_[w] = stagePos(v[u], k);
            }
            ++w;
          }
        }
      }
    }
  }

  // As StagedRow::ties, reading only the groups whose delegate is >= t.
  template <int G>
  TOPK_ALWAYS_INLINE bool
  ties(int numDelegates, uint32_t t, const Threshold& th, int numCands) const {
    const int needed = topK_ - numCands;
    int have = 0;
    for (int b = 0; b < numDelegates && have < needed; b += kNumThreads) {
      const int j = sycl::min(b + lid_, numDelegates - 1);
      const bool valid = b + lid_ < numDelegates && delegates_[j] >= t;
      const int v = groupFloat4<G>(j);
      uint32_t e[G];
      if (valid) {
        groupBits<G>(v, e);
      } else {
#pragma unroll
        for (int u = 0; u < G; ++u)
          e[u] = kPadBits;
      }
      bool tie[G];
      int cnt = 0;
#pragma unroll
      for (int u = 0; u < G; ++u) {
        const int p = stagePos(v, u);
        tie[u] = valid && th.equal(e[u]) && p >= h_ && p < h_ + n_;
        cnt += tie[u] ? 1 : 0;
      }
      const int off = sycl::exclusive_scan_over_group(sg_, cnt, sycl::plus<>());
      const int total =
          sycl::group_broadcast(sg_, off + cnt, SUBGROUP_SIZE - 1);
      if (total) {
        int base = 0;
        if (lane_ == 0)
          base = SlmAtomic<int32_t>(misc_[kTieN]).fetch_add(total);
        int w = numCands + sycl::group_broadcast(sg_, base, 0) + off;
#pragma unroll
        for (int u = 0; u < G; ++u) {
          if (tie[u]) {
            if (w < topK_) candPos_[w] = stagePos(v, u);
            ++w;
          }
        }
      }
      have += sycl::reduce_over_group(item_.get_group(), cnt, sycl::plus<>());
    }
    return have >= needed;
  }

  SlmPtr<uint32_t> candKeys_;
  SlmPtr<int32_t> candPos_;
};

// SLM one work-group may take: the core's SLM (one work-group resident),
// rounded down to a power of two, less kSlmReserve.
static size_t arenaBudget(size_t localMemBytes) {
  const size_t share = localMemBytes;
  size_t pow2 = 1;
  while (pow2 <= share / 2)
    pow2 *= 2;
  return pow2 > kSlmReserve ? pow2 - kSlmReserve : 0;
}

// Delegate words for staged rows of up to `width` columns: for each group size
// the delegates of its longest row (one per lane and chunk of 32 G floats,
// the stage shift included), padded to 128.
static int64_t delegateWords(int64_t width, int64_t topK) {
  int64_t words = 0;
  for (int64_t g = kGMin; g <= kGMax; g *= 2) {
    // G = g from g * kGroupDiv * topK columns (kGMin: from minDelegateLen) to
    // twice that.
    const int64_t from = g * kGroupDiv * topK;
    const int64_t shortest = g == kGMin ? minDelegateLen(topK) : from;
    const int64_t longest = g == kGMax ? width : std::min(width, 2 * from - 1);
    if (longest >= shortest) {
      const int64_t chunk = SUBGROUP_SIZE * g;
      words = std::max(
          words, (longest + kMaxShift + chunk - 1) / chunk * SUBGROUP_SIZE);
    }
  }
  return (words + 127) / 128 * 128;
}

// The staged select's layout for rows of up to stageCap columns; returns its
// size in bytes.
static size_t stagedLayout(
    int64_t stageCap, int64_t topK, int64_t candCap, SelectOffsets& at) {
  size_t offset =
      alignUp(sizeof(uint32_t) * (stageCap + kStageAlign), size_t{64});
  at.delegates = static_cast<int>(offset);
  offset += alignUp(sizeof(uint32_t) * delegateWords(stageCap, topK), 64);
  at.groups = static_cast<int>(offset);
  offset += alignUp(sizeof(int32_t) * topK, 64);
  at.candidates = static_cast<int>(offset);
  offset += alignUp(sizeof(int32_t) * candCap, 64);
  at.positions = 0;
  at.histogram = static_cast<int>(offset);
  offset += sizeof(int32_t) * kHistBins;
  at.misc = static_cast<int>(offset);
  offset += sizeof(int32_t) * kMiscInts;
  return offset;
}

// The global select's layout for rows of up to `width` columns (delegates from
// offset 0); returns its size in bytes.
static size_t
globalLayout(int64_t width, int64_t topK, int64_t candCap, SelectOffsets& at) {
  const int64_t words = delegateWords(width, topK);
  size_t offset = 0;
  at.delegates = 0;
  offset += alignUp(sizeof(uint32_t) * words, size_t{64});
  at.groups = static_cast<int>(offset);
  offset += alignUp(sizeof(int32_t) * topK, 64);
  at.candidates = static_cast<int>(offset);
  offset += alignUp(sizeof(uint32_t) * candCap, 64);
  at.positions = static_cast<int>(offset);
  offset += alignUp(sizeof(int32_t) * candCap, 64);
  at.histogram = static_cast<int>(offset);
  offset += sizeof(int32_t) * kHistBins;
  at.misc = static_cast<int>(offset);
  offset += sizeof(int32_t) * kMiscInts;
  return offset;
}

// The longest stage, a multiple of kStageAlign within the budget, whose staged
// layout fits the budget; 0 if that is shorter than the shortest staged row.
// Every term of stagedLayout grows with the stage, so a launch's stage cap is
// the smaller of this and its width rounded up. Searched once per (budget,
// topK, slots) and cached: launches differ in width.
static int64_t longestStage(int64_t topK, int64_t slots, size_t budget) {
  static std::mutex mutex;
  static std::map<std::tuple<size_t, int64_t, int64_t>, int64_t> cache;
  std::lock_guard<std::mutex> lock(mutex);
  const auto key = std::make_tuple(budget, topK, slots);
  const auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  const int64_t shortest = minDelegateLen(topK);
  int64_t cap = static_cast<int64_t>(budget / sizeof(uint32_t)) / kStageAlign *
                kStageAlign;
  SelectOffsets at{};
  while (cap >= shortest && stagedLayout(cap, topK, slots, at) > budget)
    cap -= kStageAlign;
  if (cap < shortest) cap = 0;
  cache.emplace(key, cap);
  return cap;
}

// The arena of a launch of rows up to `width` (<= kMaxWidth) columns:
// the staged select takes rows of up to stageCap columns, the longest multiple
// of kStageAlign (at most the width rounded up) whose stage fits the budget,
// 0 if that is shorter than the shortest staged row; the global select takes
// the longer ones if its state for the width fits too (globalCap = width,
// else 0: those rows run the job).
static ArenaLayout arenaLayout(int64_t width, int64_t topK, size_t budget) {
  int64_t slots = std::min<int64_t>(kCandCap, (topK - 1) * kGMax + 1);
  // More slots than SLM: the stage will not fit (and candCap stays an int).
  slots = std::min<int64_t>(std::max(slots, topK), budget / 4 + 1);
  const int64_t shortest = minDelegateLen(topK);
  int64_t cap = std::min<int64_t>(
      (width + kStageAlign - 1) / kStageAlign * kStageAlign,
      longestStage(topK, slots, budget));
  ArenaLayout layout{};
  size_t bytes = 0;
  if (cap >= shortest)
    bytes = stagedLayout(cap, topK, slots, layout.staged);
  else
    cap = 0;
  layout.stageCap = static_cast<int>(cap);
  layout.candCap = static_cast<int>(slots);
  if (width > cap && width >= shortest) {
    // The global select has no stage to size: it takes every candidate slot
    // that still fits (clustered rows - the topK largest in a few groups -
    // carry ~5k candidates, past the 4096 of the staged select).
    int64_t gslots = slots;
    SelectOffsets probe{};
    const size_t base = globalLayout(width, topK, 0, probe);
    if (base < budget) {
      // 8 B a slot (key + position), whole 128s.
      const int64_t fit = static_cast<int64_t>((budget - base) / 8) / 128 * 128;
      gslots = std::max(slots, std::min(kGlobalCandCapMax, fit));
    }
    const size_t globalBytes = globalLayout(width, topK, gslots, layout.global);
    layout.globalCandCap = static_cast<int>(gslots);
    if (globalBytes <= budget) {
      layout.globalCap = static_cast<int>(width);
      bytes = std::max(bytes, globalBytes);
    } else {
      layout.global = SelectOffsets{};
    }
  }
  layout.bytes = std::max(bytes, sharedMemorySize(static_cast<size_t>(topK)));
  return layout;
}

// One work-group per row. Decode rows cover [0, len) with len from seqLens as
// in top_k_per_row_decode_kernel; prefill rows [rowStarts[r], rowEnds[r]).
// Output indices are relative to the row's start, -1 past its length.
template <int kThreads, bool kPrefill>
class top_k_per_row_delegate_kernel {
  static_assert(kThreads == kNumThreads, "kernel threads == job threads");

 public:
  top_k_per_row_delegate_kernel(
      sycl::local_accessor<sycl::float4, 1>& arena,
      const ArenaLayout& layout,
      const float* logits,
      int64_t width,
      const int32_t* seqLens,
      const int32_t* rowStarts,
      const int32_t* rowEnds,
      int32_t* indices,
      int64_t stride0,
      int64_t topK,
      int64_t nextN)
      : arena_(arena),
        layout_(layout),
        logits_(logits),
        width_(width),
        seqLens_(seqLens),
        rowStarts_(rowStarts),
        rowEnds_(rowEnds),
        indices_(indices),
        stride0_(stride0),
        topK_(topK),
        nextN_(nextN) {}

  TOPK_ALWAYS_INLINE void operator()
      [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] (
          const sycl::nd_item<3>& item) const {
    const int64_t rowIdx = item.get_group(0);
    int rowStart = 0;
    int rowEnd = 0;
    if constexpr (kPrefill) {
      rowStart = rowStarts_[rowIdx];
      rowEnd = rowEnds_[rowIdx];
    } else {
      // Clamp as in CUDA: padded rows with seq_len < next_n have no columns.
      const int32_t seqLen = seqLens_[rowIdx / nextN_];
      rowEnd = sycl::max(0, int(seqLen - nextN_ + (rowIdx % nextN_) + 1));
    }
    const int topK = static_cast<int>(topK_);
    // Bounds outside [0, width) are clipped to the row; rowEnd <= rowStart is
    // an empty row.
    rowStart = sycl::max(rowStart, 0);
    rowEnd = sycl::min(rowEnd, static_cast<int>(width_));
    const int n = sycl::max(rowEnd - rowStart, 0);
    const float* row = logits_ + rowIdx * stride0_;
    int32_t* out = indices_ + rowIdx * topK_;
    const int lid = static_cast<int>(item.get_local_id(0));
    if (n <= topK) {
      for (int i = lid; i < topK; i += kThreads)
        out[i] = i < n ? i : -1;
      return;
    }
    if (n >= minDelegateLen(topK) &&
        (n <= layout_.stageCap || n <= layout_.globalCap)) {
      const auto arena = reinterpret_cast<SlmPtr<char>>(
          arena_.template get_multi_ptr<sycl::access::decorated::yes>().get());
      bool done = false;
      if (n <= layout_.stageCap) {
        const StagedRow staged(
            item, arena, layout_, row + rowStart, n, topK, out);
        switch (groupSize(n, topK)) {
          case 4:
            done = staged.run<4>();
            break;
          case 8:
            done = staged.run<8>();
            break;
          default:
            done = staged.run<16>();
            break;
        }
      } else {
        const GlobalRow global(
            item, arena, layout_, row + rowStart, n, topK, out);
        switch (groupSize(n, topK)) {
          case 4:
            done = global.run<4>();
            break;
          case 8:
            done = global.run<8>();
            break;
          default:
            done = global.run<16>();
            break;
        }
      }
      if (done) return;
      // The job reuses the arena.
      sycl::group_barrier(item.get_group());
    }
    // The host only launches for stride1 == 1.
    exactRowJob(item, arena_, row + rowStart, n, out, topK);
  }

 private:
  sycl::local_accessor<sycl::float4, 1> arena_;
  ArenaLayout layout_;
  const float* logits_;
  int64_t width_;             // columns of logits
  const int32_t* seqLens_;    // decode
  const int32_t* rowStarts_;  // prefill
  const int32_t* rowEnds_;    // prefill
  int32_t* indices_;
  int64_t stride0_;
  int64_t topK_;
  int64_t nextN_;
};

// Launches the kernel for rows that wideRowLayout() accepted. rowA is the
// seqLens (decode) or rowStarts (prefill) vector, rowB the rowEnds (prefill).
// Returns false if the runtime rejected the launch at submission, in which case
// nothing ran and the caller uses the radix kernels. Errors the device reports
// later, while the kernel runs, are not caught here: they surface as for any
// other kernel.
template <bool kPrefill>
static bool launch(
    const torch::Tensor& logits,
    const torch::Tensor& rowA,
    const torch::Tensor& rowB,
    torch::Tensor& indices,
    int64_t numRows,
    int64_t stride0,
    int64_t topK,
    int64_t nextN,
    const ArenaLayout& layout) {
  const at::DeviceGuard device_guard(logits.device());
  auto& queue = ::vllm::xpu::vllmGetQueue(logits.device().index());
  const float* logitsPtr = logits.data_ptr<float>();
  const int32_t* seqLens = kPrefill ? nullptr : rowA.data_ptr<int32_t>();
  const int32_t* rowStarts = kPrefill ? rowA.data_ptr<int32_t>() : nullptr;
  const int32_t* rowEnds = kPrefill ? rowB.data_ptr<int32_t>() : nullptr;
  int32_t* indicesPtr = indices.data_ptr<int32_t>();
  sycl::range<3> grid(numRows, 1, 1);
  sycl::range<3> block(kNumThreads, 1, 1);
  try {
    queue.submit([&](sycl::handler& cgh) {
      // layout.bytes is a multiple of 16: whole 64 B blocks, or the job's
      // state.
      sycl::local_accessor<sycl::float4, 1> arena(
          sycl::range<1>((layout.bytes + 15) / 16), cgh);
      const top_k_per_row_delegate_kernel<kNumThreads, kPrefill> kernel(
          arena,
          layout,
          logitsPtr,
          logits.size(1),
          seqLens,
          rowStarts,
          rowEnds,
          indicesPtr,
          stride0,
          topK,
          nextN);
      cgh.parallel_for(sycl::nd_range<3>(grid * block, block), kernel);
    });
  } catch (const sycl::exception& e) {
    TORCH_WARN_ONCE(
        "top_k_per_row: delegate-based launch rejected (",
        e.what(),
        "), using the radix kernels");
    return false;
  }
  return true;
}

// The layout the kernel assumes: float32 rows with unit column stride, int32
// [numRows, topK] contiguous output on the same device.
static bool legalCommon(
    const torch::Tensor& logits,
    const torch::Tensor& indices,
    int64_t numRows,
    int64_t stride0,
    int64_t stride1,
    int64_t topK) {
  if (numRows < 1 || topK < 1 || topK > kMaxTopK || stride1 != 1) return false;
  if (!logits.defined() || !logits.is_xpu() ||
      logits.scalar_type() != at::kFloat || logits.dim() != 2)
    return false;
  if (logits.size(0) != numRows || logits.size(1) < 1 ||
      logits.stride(0) != stride0 || stride0 < 0 || logits.stride(1) != 1)
    return false;
  if (!indices.defined() || indices.device() != logits.device() ||
      indices.scalar_type() != at::kInt || indices.dim() != 2 ||
      indices.size(0) != numRows || indices.size(1) != topK ||
      !indices.is_contiguous())
    return false;
  return true;
}

// An int32 vector of n entries on the logits' device.
static bool
legalRowVector(const torch::Tensor& v, const torch::Tensor& logits, int64_t n) {
  return v.defined() && v.device() == logits.device() &&
         v.scalar_type() == at::kInt && v.dim() == 1 && v.size(0) == n &&
         v.is_contiguous();
}

// Dispatch rule, measured on B70 (BMG-G31, 32 Xe cores) against the radix
// kernels of this file over decode rows {1..256} x widths {4k..512k} x topK
// {512, 1024, 2048}, prefill q {16..256} x widths {4k..128k}, decode launches
// with per-row lengths (uniform, short, bimodal) and rows of other value
// distributions (clustered topK, ties, narrow, constant, sorted): the delegate
// select is taken only where it was >= ~1.0x on every one of them, with the
// crossover rounded up.
//   rows <= xeCores (one work-group per row: a single wave)
//     and minWidthFor(topK) <= width <= 262144, topK in {512, 1024, 2048}
// Wide launches of more rows than cores stay on the radix kernels even though
// full-length rows win there (1.2-1.9x): the rule sees only the padded width,
// and a launch of many *short* rows (seq_len <= 8k in a 128k-wide logits
// tensor) runs the exact job here at one work-group per core (the arena is
// sized for the widest row, ~100 KB of the 128 KB SLM), where the radix kernel
// fits ~5 work-groups per core: 0.6-0.85x. With a single wave the same short
// rows cost the same as there (1.00x).
// Only BMG-G31 devices (B70) are measured, so only they take it. Only the topK
// values it was tuned at (thresholds, candidate slots and the minimum widths
// are tuned per value).
static bool topKTuned(int64_t topK) {
  return topK == 512 || topK == 1024 || topK == 2048;
}

// The narrowest launch taken, by topK. Clustered rows (the topK largest in one
// block of the row, e.g. a recent-token bonus) carry ~2x the usual candidates
// here, and the radix kernel finishes them in one histogram step: below these
// widths the delegate select loses 0.7-0.9x on them where it wins only
// 1.05-1.4x on smooth rows. At/above them it is >= 0.97x on every row type
// measured.
// Loose upper bound on the rows tested before the device is known (BMG-G31
// has 32 Xe cores); the device's own count is checked after it.
static constexpr int64_t kMaxRowsBound = 128;

static int64_t minWidthFor(int64_t topK) {
  return topK <= 1024 ? 32768 : std::max<int64_t>(45056, 22 * topK);
}

// What the rule needs of a device, decided once per device index: it is a
// BMG-G31 that can run the work-group size and has its Xe-core count and SLM
// size available. Read without a lock after the first call.
struct DeviceCaps {
  bool use = false;
  int xeCores = 0;
  size_t localMemBytes = 0;
};

static const DeviceCaps* capsFor(int deviceIndex) {
  constexpr int kMaxDevices = 64;
  static DeviceCaps caps[kMaxDevices];
  static std::once_flag once[kMaxDevices];
  if (deviceIndex < 0 || deviceIndex >= kMaxDevices) return nullptr;
  std::call_once(once[deviceIndex], [&] {
    namespace intel_info = sycl::ext::intel::info::device;
    const sycl::device device = c10::xpu::get_raw_device(deviceIndex);
    DeviceCaps& c = caps[deviceIndex];
    if (::vllm::xpu::is_bmg_g31(deviceIndex) &&
        device.has(sycl::aspect::ext_intel_gpu_slices) &&
        device.has(sycl::aspect::ext_intel_gpu_subslices_per_slice) &&
        device.get_info<sycl::info::device::max_work_group_size>() >=
            static_cast<size_t>(kNumThreads)) {
      c.xeCores = static_cast<int>(device.get_info<intel_info::gpu_slices>()) *
                  static_cast<int>(
                      device.get_info<intel_info::gpu_subslices_per_slice>());
      c.localMemBytes = device.get_info<sycl::info::device::local_mem_size>();
      c.use = c.xeCores > 0;
    }
  });
  return caps[deviceIndex].use ? &caps[deviceIndex] : nullptr;
}

// The arena of a (width, topK) launch on a device, cached: decode repeats the
// same launch every step. Widths are rounded up to the stage chunk, so the
// cache holds at most 3 * kMaxWidth / kStageAlign entries per device however
// the widths of a prefill vary; a layout for the rounded width serves every
// narrower row.
struct Plan {
  bool use;  // the arena fits the SLM budget
  ArenaLayout layout;
};

static Plan
planFor(int deviceIndex, const DeviceCaps& caps, int64_t width, int64_t topK) {
  static std::mutex mutex;
  static std::map<std::tuple<int, int64_t, int64_t>, Plan> cache;
  const int64_t rounded = (width + kStageAlign - 1) / kStageAlign * kStageAlign;
  std::lock_guard<std::mutex> lock(mutex);
  const auto key = std::make_tuple(deviceIndex, rounded, topK);
  const auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  Plan plan{};
  const size_t budget = arenaBudget(caps.localMemBytes);
  plan.layout = arenaLayout(rounded, topK, budget);
  plan.use = plan.layout.bytes <= budget;
  cache.emplace(key, plan);
  return plan;
}

// The arena layout if the dispatch rule sends this call to the delegate select,
// nothing if the radix kernels run it. rowA is seqLens (decode) or rowStarts
// (prefill), rowB rowEnds (prefill); nextN is 1 for prefill. The tests are
// ordered by cost, since the time spent here is added to every call that keeps
// the radix kernels: integers first, then the device, then the tensor layout,
// and only then the cached arena lookup.
template <bool kPrefill>
static std::optional<ArenaLayout> wideRowLayout(
    const torch::Tensor& logits,
    const torch::Tensor& rowA,
    const torch::Tensor& rowB,
    const torch::Tensor& indices,
    int64_t numRows,
    int64_t stride0,
    int64_t stride1,
    int64_t topK,
    int64_t nextN) {
  if (!topKTuned(topK) || numRows < 1 || numRows > kMaxRowsBound ||
      !logits.defined() || logits.dim() != 2)
    return std::nullopt;
  const int64_t width = logits.size(1);
  if (width < minWidthFor(topK) || width > kMaxWidth) return std::nullopt;
  // The device is queried only for a tensor that is on one, and its decision
  // is cached per device.
  if (!logits.is_xpu()) return std::nullopt;
  const int deviceIndex = logits.device().index();
  const DeviceCaps* caps = capsFor(deviceIndex);
  if (caps == nullptr || numRows > caps->xeCores) return std::nullopt;
  if (!legalCommon(logits, indices, numRows, stride0, stride1, topK))
    return std::nullopt;
  if constexpr (kPrefill) {
    if (!legalRowVector(rowA, logits, numRows) ||
        !legalRowVector(rowB, logits, numRows))
      return std::nullopt;
  } else {
    // Only 1D seqLens: the radix kernel reads a 2D (B, next_n) tensor as flat
    // 1D with the real next_n, so a 2D one stays with it to keep both
    // agreeing on every row's length.
    if (nextN < 1 || numRows % nextN != 0 ||
        !legalRowVector(rowA, logits, numRows / nextN))
      return std::nullopt;
  }
  const Plan plan = planFor(deviceIndex, *caps, width, topK);
  if (!plan.use) return std::nullopt;
  return plan.layout;
}

}  // namespace delegate_topk

#undef TOPK_ALWAYS_INLINE

}  // namespace vllm

void top_k_per_row_decode(
    const torch::Tensor& logits,
    int64_t next_n,
    const torch::Tensor& seqLens,
    torch::Tensor& indices,
    int64_t numRows,
    int64_t stride0,
    int64_t stride1,
    int64_t topK) {
  TORCH_CHECK(logits.size(0) == numRows);
  TORCH_CHECK(logits.stride(0) == stride0);
  TORCH_CHECK(logits.stride(1) == stride1);

  TORCH_CHECK(seqLens.size(0) == numRows / next_n);

  TORCH_CHECK(indices.size(0) == numRows);
  TORCH_CHECK(indices.size(1) == topK);

  // Wide rows on BMG: delegate-based selection.
  if (const auto layout = vllm::delegate_topk::wideRowLayout<false>(
          logits,
          seqLens,
          seqLens,
          indices,
          numRows,
          stride0,
          stride1,
          topK,
          next_n)) {
    if (vllm::delegate_topk::launch<false>(
            logits,
            seqLens,
            seqLens,
            indices,
            numRows,
            stride0,
            topK,
            next_n,
            *layout)) {
      return;
    }
  }

  constexpr int kSortingAlgorithmThreshold = 12288;
  constexpr int kSplitWorkThreshold = 200 * 1000;
  constexpr int kNumThreadsPerBlock = 512;
  auto& queue = vllm::xpu::vllmGetQueue();
  const at::DeviceGuard device_guard(logits.device());

  const auto numColumns = logits.size(1);

  if (numColumns < kSortingAlgorithmThreshold) {
    // Use insertion sort
    sycl::range<3> grid(numRows, 1, 1);
    sycl::range<3> block(kNumThreadsPerBlock, 1, 1);
    size_t dynamic_smem_in_bytes =
        sizeof(vllm::SharedStates) + topK * sizeof(int32_t);
    queue.submit([&](sycl::handler& cgh) {
      // SLM allocation
      sycl::local_accessor<char, 1> slm(
          sycl::range<1>(dynamic_smem_in_bytes), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          vllm::top_k_per_row_decode_kernel<kNumThreadsPerBlock, false>(
              slm,
              logits.data_ptr<float>(),
              seqLens.data_ptr<int>(),
              indices.data_ptr<int>(),
              stride0,
              stride1,
              topK,
              next_n));
    });
  } else if (numColumns < kSplitWorkThreshold) {
    // From this threshold, use radix sort instead
    sycl::range<3> grid(numRows, 1, 1);
    sycl::range<3> block(kNumThreadsPerBlock, 1, 1);
    size_t dynamic_smem_in_bytes =
        sizeof(vllm::SharedStates) + topK * sizeof(int32_t);
    queue.submit([&](sycl::handler& cgh) {
      // SLM allocation
      sycl::local_accessor<char, 1> slm(
          sycl::range<1>(dynamic_smem_in_bytes), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          vllm::top_k_per_row_decode_kernel<
              kNumThreadsPerBlock,
              false /*TODO true*/>(
              slm,
              logits.data_ptr<float>(),
              seqLens.data_ptr<int>(),
              indices.data_ptr<int>(),
              stride0,
              stride1,
              topK,
              next_n));
    });
  } else {
    // Long sequences are run in two steps
    constexpr auto multipleBlocksPerRowConfig = 10;

    const auto outIndicesAux = torch::empty(
        {numRows, multipleBlocksPerRowConfig, topK},
        torch::dtype(torch::kInt32).device(logits.device()));
    const auto outLogitsAux = torch::empty(
        {numRows, multipleBlocksPerRowConfig, topK},
        torch::dtype(torch::kFloat).device(logits.device()));

    // Step 1: each row is processed by multiple blocks
    sycl::range<3> grid_step1(numRows, multipleBlocksPerRowConfig, 1);
    sycl::range<3> block_step1(kNumThreadsPerBlock, 1, 1);
    size_t dynamic_smem_in_bytes_step1 = sizeof(vllm::SharedStates) +
                                         topK * sizeof(int32_t) +
                                         topK * sizeof(float);
    queue.submit([&](sycl::handler& cgh) {
      // SLM allocation
      sycl::local_accessor<char, 1> slm(
          sycl::range<1>(dynamic_smem_in_bytes_step1), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(grid_step1 * block_step1, block_step1),
          vllm::top_k_per_row_decode_kernel<
              kNumThreadsPerBlock,
              false /*TODO true*/,
              true>(
              slm,
              logits.data_ptr<float>(),
              seqLens.data_ptr<int>(),
              outIndicesAux.data_ptr<int>(),
              stride0,
              stride1,
              topK,
              next_n,
              outLogitsAux.data_ptr<float>()));
    });

    // Step 2: merge the results from multiple blocks
    constexpr int kNumThreadsPerBlockMerge = 1024;
    sycl::range<3> grid_step2(numRows, 1, 1);
    sycl::range<3> block_step2(kNumThreadsPerBlockMerge, 1, 1);
    size_t dynamic_smem_in_bytes_step2 =
        sizeof(vllm::SharedStates) + topK * sizeof(int32_t);
    queue.submit([&](sycl::handler& cgh) {
      // SLM allocation
      sycl::local_accessor<char, 1> slm(
          sycl::range<1>(dynamic_smem_in_bytes_step2), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(grid_step2 * block_step2, block_step2),
          vllm::top_k_per_row_decode_kernel<
              kNumThreadsPerBlockMerge,
              false /*TODO true*/,
              false,
              true>(
              slm,
              outLogitsAux.data_ptr<float>(),
              seqLens.data_ptr<int32_t>(),
              indices.data_ptr<int32_t>(),
              outLogitsAux.stride(0),
              1,
              topK,
              next_n,
              /*outLogits=*/nullptr,
              multipleBlocksPerRowConfig,
              outIndicesAux.data_ptr<int32_t>()));
    });
  }
  return;
}

void top_k_per_row_prefill(
    const torch::Tensor& logits,
    const torch::Tensor& rowStarts,
    const torch::Tensor& rowEnds,
    torch::Tensor& indices,
    int64_t numRows,
    int64_t stride0,
    int64_t stride1,
    int64_t topK) {
  TORCH_CHECK(logits.size(0) == numRows);
  TORCH_CHECK(logits.stride(0) == stride0);
  TORCH_CHECK(logits.stride(1) == stride1);

  TORCH_CHECK(rowStarts.size(0) == numRows);
  TORCH_CHECK(rowEnds.size(0) == numRows);

  TORCH_CHECK(indices.size(0) == numRows);
  TORCH_CHECK(indices.size(1) == topK);

  // Wide rows on BMG: delegate-based selection.
  if (const auto layout = vllm::delegate_topk::wideRowLayout<true>(
          logits,
          rowStarts,
          rowEnds,
          indices,
          numRows,
          stride0,
          stride1,
          topK,
          1)) {
    if (vllm::delegate_topk::launch<true>(
            logits,
            rowStarts,
            rowEnds,
            indices,
            numRows,
            stride0,
            topK,
            1,
            *layout)) {
      return;
    }
  }

  constexpr int kSortingAlgorithmThreshold = 12288;
  constexpr int kNumThreadsPerBlock = 512;
  auto& queue = vllm::xpu::vllmGetQueue();
  const at::DeviceGuard device_guard(logits.device());

  int numInsertionBlocks =
      std::min(static_cast<int>(numRows), kSortingAlgorithmThreshold);

  size_t dynamic_smem_in_bytes =
      sizeof(vllm::SharedStates) + topK * sizeof(int32_t);

  // For small input sizes, we only launch insertion sort kernels.
  sycl::range<3> grid(numInsertionBlocks, 1, 1);
  sycl::range<3> block(kNumThreadsPerBlock, 1, 1);
  queue.submit([&](sycl::handler& cgh) {
    // SLM allocation
    sycl::local_accessor<char, 1> slm(
        sycl::range<1>(dynamic_smem_in_bytes), cgh);
    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block),
        vllm::top_k_per_row_prefill_kernel<kNumThreadsPerBlock, false>(
            slm,
            indices.data_ptr<int32_t>(),
            logits.data_ptr<float>(),
            rowStarts.data_ptr<int32_t>(),
            rowEnds.data_ptr<int32_t>(),
            stride0,
            stride1,
            topK,
            0));
  });

  // For large input sizes, we launch radix sort kernels for the remaining
  if (numRows > kSortingAlgorithmThreshold) {
    int numRadixBlocks = numRows - kSortingAlgorithmThreshold;
    sycl::range<3> grid_radix(numRadixBlocks, 1, 1);
    sycl::range<3> block_radix(kNumThreadsPerBlock, 1, 1);
    queue.submit([&](sycl::handler& cgh) {
      // SLM allocation
      sycl::local_accessor<char, 1> slm(
          sycl::range<1>(dynamic_smem_in_bytes), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(grid_radix * block_radix, block_radix),
          vllm::top_k_per_row_prefill_kernel<kNumThreadsPerBlock, false>(
              slm,
              indices.data_ptr<int32_t>(),
              logits.data_ptr<float>(),
              rowStarts.data_ptr<int32_t>(),
              rowEnds.data_ptr<int32_t>(),
              stride0,
              stride1,
              topK,
              kSortingAlgorithmThreshold));
    });
  }
}
