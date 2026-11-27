// Adapted from
// https://github.com/vllm-project/vllm/blob/main/csrc/sampler.cu#L646-L728

#include <sycl/sycl.hpp>

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>

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
// top_k_per_row_decode/prefill route a call here where tryRun's dispatch rule
// says it is faster. Rows of at least minDelegateLen(topK) columns run the
// delegate select; shorter rows, and a row on which the select gives up (more
// candidates than slots), run the upstream job (topKPerRowJob) from the same
// SLM arena, so they behave as in the radix kernels.
//
// Delegate select. Split the row into groups of G floats; a group's largest key
// is its delegate. Let t be the topK-th largest delegate: topK groups hold a
// key >= t, so t <= the row's topK-th largest key, and every key > t lies in
// one of the fewer than topK groups whose delegate is > t. The top topK are the
// keys > t of those groups (the candidates), plus keys == t if there are fewer
// than topK. With G ~ n / (kGroupDiv * topK) the only whole-row passes are the
// first pass and one max per group.
//
// Keys are uint32 in the order of the radix kernels' (+NaN above +inf, -NaN
// below -inf, +0 above -0), so NaN, signed zeros and infinities rank as there.

#define TOPK_ALWAYS_INLINE inline __attribute__((always_inline))

namespace delegate_topk {

static constexpr int kNumThreads = 512;
static constexpr int kNumSubGroups = kNumThreads / SUBGROUP_SIZE;

constexpr size_t sharedMemorySize(size_t topK) {
  const size_t offset = (topK * sizeof(int32_t) + alignof(SharedStates) - 1) /
                        alignof(SharedStates) * alignof(SharedStates);
  return offset + sizeof(SharedStates);
}

// SLM left unused per work-group (a launch with only 4096 B spare is rejected).
// One group is resident per core.
static constexpr size_t kSlmReserve = 8192;

// Group size G = n / (kGroupDiv * topK) rounded down to a power of two, in
// [kGMin, kGMax]; kGMax bounds the candidates, (topK - 1) * G.
static constexpr int kGroupDiv = 3;
static constexpr int kGMin = 4;
static constexpr int kGMax = 16;
// Loads are float4s at 64 B aligned addresses: a row starts h = (address % 64)
// / 4 floats into its first line of kLineFloats floats (loaded singly if h >
// 0).
static constexpr int kLineFloats = 16;

// Rows from minDelegateLen(topK) columns run the delegate select, shorter ones
// the job: 2 * kGMin * topK columns give 2 * topK delegates at G = kGMin. A
// delegate row is never shorter than the head.
static constexpr int64_t minDelegateLen(int64_t topK) {
  const int64_t minCols = std::max<int64_t>(2 * kGMin * topK, 6144);
  return std::max<int64_t>(
      std::min<int64_t>(minCols, kGMin * kGroupDiv * topK), kLineFloats);
}

// Radix select: 11-bit digits, at most three passes; a bin of at most kSmallSet
// keys is ranked in one subgroup.
static constexpr int kRadixBits = 11;
static constexpr int kHistBins = 1 << kRadixBits;
static constexpr int kSmallSet = SUBGROUP_SIZE;
static constexpr size_t kArenaAlign = 64;

// SLM loads per lane kept in flight over the delegates (latency bound).
static constexpr int kLoadsInFlight = 4;
// Same for the pass streaming the row from global memory.
static constexpr int kGlobalLoads = 8;
// Widest launch taken: the delegates (n / kGMax words) must fit the arena.
static constexpr int64_t kMaxWidth = 262144;

// Padding past the row: the bit pattern of the smallest key, 0.
static constexpr uint32_t kPadBits = 0xffffffffu;

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

// The SLM arena, set on the host: delegates at 0, then these offsets, multiples
// of 64 B from a 64 B aligned base. The select and the job use it in turn.
struct ArenaLayout {
  int groups;      // int32 [topK]: groups whose delegate is > t
  int candidates;  // uint32 [candCap] keys
  int positions;   // int32 [candCap] stage positions
  int histogram;   // int32 [kHistBins]
  int misc;        // int32 [kMiscInts]
  int candCap;     // candidate slots
  int maxRow;      // longest row the delegate select takes
  size_t bytes;    // the whole arena
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

template <typename T>
static constexpr T divUp(T a, T b) {
  return (a + b - 1) / b;
}

static TOPK_ALWAYS_INLINE uint32_t keyOf(uint32_t bits) {
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

static TOPK_ALWAYS_INLINE uint32_t bitsOf(uint32_t key) {
  return (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
}

// Largest key of a group from the signed max and min of its float bits:
// non-negative floats order as their bits, the negative ones in reverse.
static TOPK_ALWAYS_INLINE uint32_t groupKey(int32_t smax, int32_t smin) {
  return smax >= 0 ? (static_cast<uint32_t>(smax) | 0x80000000u)
                   : ~static_cast<uint32_t>(smin);
}

// key > t and key == t on float bits, one compare each: as signed ints the bits
// order like floats for t >= +0; for t <= -0 the larger floats are the unsigned
// patterns below t's. The flip makes either an unsigned compare.
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

struct Digit {
  uint32_t lo;
  uint32_t last;
  int shift;
  // Keys whose low shift + width bits are free above lo, capped so that keys
  // below lo cannot wrap around into the range.
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

static TOPK_ALWAYS_INLINE int groupSize(int n, int topK) {
  const int ratio = n / (kGroupDiv * topK);
  int g = kGMin;
  while (2 * g <= ratio && g < kGMax)
    g *= 2;
  return g;
}

// Lower bound of t from one lane's delegates: it cuts its numChunks delegates
// into S = ceil(topK / WG) runs; the WG * S >= topK run maxima are distinct
// delegates, so their minimum is <= t. Valid only when every run has a delegate
// (bounded()).
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

// The delegate select over a row in global memory, by the whole work-group:
// the delegate pass streams the row; later passes re-read only the listed
// groups (and, for ties, those with delegate t), which are still in cache.
// Candidates are (key, position) pairs in SLM. Nothing outside the row is read.
// Branches on row data are uniform, and every exit precedes the first output
// write: run() returns false (the caller runs the job) if candidates overflow
// their slots or a select finds no bin (which the bounds rule out).
//
// The row is seen shifted: position p holds row[p - h] for h <= p < h + n and
// the key-0 pattern elsewhere, so float4 v of the shifted row is 16 B aligned.
// Chunk c is positions [32 G c, 32 G (c + 1)); group j = 32 c + l is the G
// floats of float4s 8 G c + 32 q + l, q < G / 4, so a subgroup reads a chunk
// coalesced. Candidates and ties are positions; output index = position - h.
class GlobalRow {
 public:
  TOPK_ALWAYS_INLINE GlobalRow(
      const sycl::nd_item<3>& item,
      SlmPtr<char> arena,
      const ArenaLayout& at,
      const float* row,
      int n,
      int topK,
      int32_t* out)
      : item_(item),
        sg_(item.get_sub_group()),
        lid_(static_cast<int>(item.get_local_id(0))),
        sgid_(static_cast<int>(sg_.get_group_linear_id())),
        lane_(static_cast<int>(sg_.get_local_linear_id())),
        lower_((1u << lane_) - 1u),
        src_(reinterpret_cast<const uint32_t*>(row)),
        h_(static_cast<int>(
            (reinterpret_cast<uintptr_t>(row) & (kArenaAlign - 1)) /
            sizeof(float))),
        n_(n),
        topK_(topK),
        candCap_(at.candCap),
        out_(out),
        delegates_(reinterpret_cast<SlmPtr<uint32_t>>(arena)),
        groups_(reinterpret_cast<SlmPtr<int32_t>>(arena + at.groups)),
        hist_(reinterpret_cast<SlmPtr<int32_t>>(arena + at.histogram)),
        misc_(reinterpret_cast<SlmPtr<int32_t>>(arena + at.misc)),
        miscU_(reinterpret_cast<SlmPtr<uint32_t>>(arena + at.misc)),
        candKeys_(reinterpret_cast<SlmPtr<uint32_t>>(arena + at.candidates)),
        candPos_(reinterpret_cast<SlmPtr<int32_t>>(arena + at.positions)) {}

  template <int G>
  TOPK_ALWAYS_INLINE bool run() const {
    const int numChunks = divUp(h_ + n_, SUBGROUP_SIZE * G);
    const int numDelegates = numChunks * SUBGROUP_SIZE;
    const int delegateWords = divUp(numDelegates, 128) * 128;
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
      for (int i = lid_; i < topK_; i += kNumThreads)
        out_[i] = candPos_[i] - h_;
    }
    return true;
  }

 private:
  template <typename T, typename P>
  static TOPK_ALWAYS_INLINE sycl::vec<T, 4> load4(P p, int v) {
    return *reinterpret_cast<SlmPtr<const sycl::vec<T, 4>>>(p + 4 * v);
  }

  template <int G>
  static TOPK_ALWAYS_INLINE int groupFloat4(int j) {
    return (j >> 5) * 8 * G + (j & 31);
  }
  static TOPK_ALWAYS_INLINE int stagePos(int v, int u) {
    return 4 * (v + (u >> 2) * SUBGROUP_SIZE) + (u & 3);
  }

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

  // Pads the delegates with key 0 to delegateWords and publishes their maximum
  // and the lower bound of t (0 if unbounded).
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

  // When all keys of the subgroup hit one bin (ties), one atomic adds them.
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

  // The bin, from the top, in which the count reaches need, and the counts
  // above and in it. Clears the histogram. False if none.
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
  // below base + 2^bits, and how many keys == it to take. pass(digit, collect)
  // adds the digit of every key in range to the histogram, or with collect
  // appends key - base to the small set.
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

  // The group list: every group whose delegate is > t. Per subgroup a count
  // pass, one atomic, a write pass.
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

  // This lane's delegates of chunks c0 + u * kNumSubGroups (clamped).
  TOPK_ALWAYS_INLINE void
  delegatesOf(int c0, int numChunks, uint32_t (&d)[kLoadsInFlight]) const {
#pragma unroll
    for (int u = 0; u < kLoadsInFlight; ++u)
      d[u] = delegates_
          [sycl::min(c0 + u * kNumSubGroups, numChunks - 1) * SUBGROUP_SIZE +
           lane_];
  }

  // More than topK candidates: write those > T and `need` of those == T.
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

  TOPK_ALWAYS_INLINE sycl::vec<int32_t, 4> stage4(int v) const {
    const int p = 4 * v - h_;  // row index of its first float
    if (p >= 0 && p + 4 <= n_)
      return *reinterpret_cast<const sycl::vec<int32_t, 4>*>(src_ + p);
    const auto at = [&](int i) {
      return i >= 0 && i < n_ ? static_cast<int32_t>(src_[i])
                              : static_cast<int32_t>(kPadBits);
    };
    return sycl::vec<int32_t, 4>(at(p), at(p + 1), at(p + 2), at(p + 3));
  }

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

  // The delegates. Chunks [firstInside, endInside) lie inside the row and load
  // unchecked.
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
    sycl::group_barrier(item_.get_group());
    publishDelegates(
        numDelegates,
        delegateWords,
        rowMax,
        RunBound::bounded(numChunks, topK_) ? bound.finish() : 0u);
  }

  // Candidates, every key > t of the listed groups, in one pass. Writes past
  // candCap_ are dropped; the count is kept.
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

  // Fewer than topK candidates: keys == t fill the rest.
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

// SLM one work-group may take: the core's, rounded down to a power of two, less
// kSlmReserve.
static size_t arenaBudget(size_t localMemBytes) {
  const size_t share = localMemBytes;
  size_t pow2 = 1;
  while (pow2 <= share / 2)
    pow2 *= 2;
  return pow2 > kSlmReserve ? pow2 - kSlmReserve : 0;
}

static constexpr size_t alignUp(size_t size, size_t alignment) {
  return divUp(size, alignment) * alignment;
}

// Arena of rows up to `width` columns with candCap candidate slots. Delegates:
// one per lane and chunk of 32 G floats; for G < kGMax a row has fewer than 2 *
// kGroupDiv * topK * G floats.
static ArenaLayout
layoutFor(int64_t width, int64_t topK, int64_t candCap, size_t budget) {
  const int64_t chunks =
      std::max<int64_t>(
          2 * kGroupDiv * topK / SUBGROUP_SIZE,
          (width + kLineFloats - 1) / (SUBGROUP_SIZE * kGMax)) +
      2;
  ArenaLayout a{};
  size_t at = alignUp(sizeof(uint32_t) * chunks * SUBGROUP_SIZE, 512);
  a.groups = static_cast<int>(at);
  at += alignUp(sizeof(int32_t) * topK, 64);
  a.candidates = static_cast<int>(at);
  at += alignUp(sizeof(uint32_t) * candCap, 64);
  a.positions = static_cast<int>(at);
  at += alignUp(sizeof(int32_t) * candCap, 64);
  a.histogram = static_cast<int>(at);
  at += sizeof(int32_t) * kHistBins;
  a.misc = static_cast<int>(at);
  at += sizeof(int32_t) * kMiscInts;
  a.candCap = static_cast<int>(candCap);
  a.bytes = std::max(at, sharedMemorySize(static_cast<size_t>(topK)));
  a.maxRow = at <= budget ? static_cast<int>(width) : 0;
  return a;
}

// Takes every candidate slot that still fits (clustered rows carry ~5k).
static ArenaLayout arenaLayout(int64_t width, int64_t topK, size_t budget) {
  // 4096 slots (8 B each) keep the arena small enough for 262144 columns.
  int64_t slots = std::min<int64_t>(4096, (topK - 1) * kGMax + 1);
  slots = std::min<int64_t>(std::max(slots, topK), budget / 4 + 1);
  const size_t base = layoutFor(width, topK, 0, budget).bytes;
  if (base < budget) {
    const int64_t fit = static_cast<int64_t>((budget - base) / 8) / 128 * 128;
    slots = std::max(slots, std::min<int64_t>(16384, fit));
  }
  return layoutFor(width, topK, slots, budget);
}

// One work-group per row. Decode rows cover [0, len) with len from seqLens as
// in top_k_per_row_decode_kernel; prefill rows [rowStarts[r], rowEnds[r]).
// Output indices are relative to the row's start, -1 past its length.
template <bool kPrefill>
struct top_k_per_row_delegate_kernel {
  sycl::local_accessor<char, 1> arena_;
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
    rowStart = sycl::max(rowStart, 0);
    rowEnd = sycl::min(rowEnd, static_cast<int>(width_));
    const int n = sycl::max(rowEnd - rowStart, 0);
    const float* row = logits_ + rowIdx * stride0_;
    int32_t* out = indices_ + rowIdx * topK_;
    if (n >= minDelegateLen(topK) && n <= layout_.maxRow) {
      // The arena base is 64 B aligned unless the runtime says otherwise.
      auto base = reinterpret_cast<SlmPtr<char>>(
          arena_.template get_multi_ptr<sycl::access::decorated::yes>().get());
      const auto skew =
          reinterpret_cast<uintptr_t>(
              arena_.template get_multi_ptr<sycl::access::decorated::no>()
                  .get()) &
          (kArenaAlign - 1);
      const auto arena = base + (skew ? kArenaAlign - skew : 0);
      const GlobalRow global(
          item, arena, layout_, row + rowStart, n, topK, out);
      bool done = false;
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
      if (done) return;
      sycl::group_barrier(item.get_group());
    }
    // Short rows, and rows the select gave up on: the upstream job (stride 1).
    topKPerRowJob<kNumThreads, kNumBins, false>(
        item,
        arena_,
        nullptr,
        row,
        rowStart,
        rowStart + n,
        out,
        nullptr,
        1,
        topK);
  }
};

// Dispatch rule, measured on B70 (BMG-G31) over decode and prefill launches and
// many value distributions: taken only where it was >= ~1.0x on all of them.
// That is BMG-G31 only, topK in {512, 1024, 2048}, rows <= xeCores (one
// work-group per row; many short rows in a wide tensor lose to the radix
// kernels' ~5 groups per core) and minWidth(topK) <= width <= kMaxWidth (below,
// clustered rows lose: the radix kernel finishes them in one histogram step).
struct DeviceCaps {
  int xeCores = 0;  // 0: the select is not taken on this device
  size_t localMemBytes = 0;
};

// Decided once per device index; read without a lock after the first call.
static const DeviceCaps& capsFor(int deviceIndex) {
  constexpr int kMaxDevices = 64;
  static DeviceCaps caps[kMaxDevices];
  static std::once_flag once[kMaxDevices];
  static const DeviceCaps none;
  if (deviceIndex < 0 || deviceIndex >= kMaxDevices) return none;
  std::call_once(once[deviceIndex], [&] {
    namespace intel_info = sycl::ext::intel::info::device;
    const sycl::device device = c10::xpu::get_raw_device(deviceIndex);
    if (::vllm::xpu::is_bmg_g31(deviceIndex) &&
        device.has(sycl::aspect::ext_intel_gpu_slices) &&
        device.has(sycl::aspect::ext_intel_gpu_subslices_per_slice) &&
        device.get_info<sycl::info::device::max_work_group_size>() >=
            static_cast<size_t>(kNumThreads)) {
      caps[deviceIndex].xeCores =
          static_cast<int>(device.get_info<intel_info::gpu_slices>()) *
          static_cast<int>(
              device.get_info<intel_info::gpu_subslices_per_slice>());
      caps[deviceIndex].localMemBytes =
          device.get_info<sycl::info::device::local_mem_size>();
    }
  });
  return caps[deviceIndex];
}

// Runs the call with the delegate select if the rule takes it, else (or if the
// runtime rejected the launch) returns false and nothing ran. rowA is seqLens
// or rowStarts, rowB seqLens or rowEnds. The caller has checked the sizes and
// strides against each other. Only 1D seqLens: the radix kernel reads a 2D one
// as flat 1D, so it stays there to keep both agreeing on every row.
template <bool kPrefill>
static bool tryRun(
    const torch::Tensor& logits,
    const torch::Tensor& rowA,
    const torch::Tensor& rowB,
    torch::Tensor& indices,
    int64_t numRows,
    int64_t stride0,
    int64_t topK,
    int64_t nextN) {
  const bool tuned = topK == 512 || topK == 1024 || topK == 2048;
  if (!tuned || numRows < 1 || nextN < 1 || numRows % nextN != 0 ||
      logits.dim() != 2 || !logits.is_xpu() ||
      logits.scalar_type() != at::kFloat || logits.stride(1) != 1 ||
      stride0 < 0)
    return false;
  const int64_t width = logits.size(1);
  const DeviceCaps& caps = capsFor(logits.device().index());
  if (width < (topK <= 1024 ? 32768 : std::max<int64_t>(45056, 22 * topK)) ||
      width > kMaxWidth || numRows > caps.xeCores)
    return false;
  if (indices.device() != logits.device() ||
      indices.scalar_type() != at::kInt || indices.dim() != 2 ||
      !indices.is_contiguous())
    return false;
  for (const torch::Tensor* v : {&rowA, &rowB}) {
    if (v->device() != logits.device() || v->scalar_type() != at::kInt ||
        v->dim() != 1 || !v->is_contiguous())
      return false;
  }
  const ArenaLayout layout =
      arenaLayout(width, topK, arenaBudget(caps.localMemBytes));
  if (layout.maxRow == 0) return false;

  const at::DeviceGuard device_guard(logits.device());
  auto& queue = ::vllm::xpu::vllmGetQueue(logits.device().index());
  const int32_t* rowAPtr = rowA.data_ptr<int32_t>();
  const int32_t* rowBPtr = rowB.data_ptr<int32_t>();
  sycl::range<3> grid(numRows, 1, 1);
  sycl::range<3> block(kNumThreads, 1, 1);
  try {
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<char, 1> arena(
          sycl::range<1>(layout.bytes + kArenaAlign), cgh);
      const top_k_per_row_delegate_kernel<kPrefill> kernel{
          arena,
          layout,
          logits.data_ptr<float>(),
          width,
          kPrefill ? nullptr : rowAPtr,
          kPrefill ? rowAPtr : nullptr,
          kPrefill ? rowBPtr : nullptr,
          indices.data_ptr<int32_t>(),
          stride0,
          topK,
          nextN};
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
  if (vllm::delegate_topk::tryRun<false>(
          logits, seqLens, seqLens, indices, numRows, stride0, topK, next_n))
    return;

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
  if (vllm::delegate_topk::tryRun<true>(
          logits, rowStarts, rowEnds, indices, numRows, stride0, topK, 1))
    return;

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
