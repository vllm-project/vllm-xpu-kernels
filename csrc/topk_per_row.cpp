// Adapated from
// https://github.com/vllm-project/vllm/blob/main/csrc/sampler.cu#L646-L728

#include <sycl/sycl.hpp>

#include <algorithm>
#include <iostream>
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
