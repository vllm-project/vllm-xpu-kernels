// Adapted from
// https://github.com/vllm-project/vllm/blob/main/csrc/sampler.cu#L646-L728

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <iostream>
#include <map>
#include <mutex>
#include <string>

#include "dispatch_utils.h"
#include "utils.h"
#include "sycl_launch.h"

namespace vllm {

static constexpr int SUBGROUP_SIZE = 32;

// The number of slots for the final pass. This buffer is the largest part of a
// work-group's shared memory, and shared memory is what decides how many
// work-groups a core can keep resident, which is in turn what hides the latency
// of every stage: at topK=128 the six stages of a row cost 8 to 15 us each, all
// six within a factor of two of one another, which is the signature of a
// pipeline that has nothing else to run while it waits rather than of one stage
// being expensive. 512 slots leave 12.6KB per work-group instead of 24.7KB.
// Rows whose threshold bin does not fit take one more radix step, which is only
// correctness-neutral because step 3 writes the last bin's elements without
// using this buffer at all.
static constexpr int kNumFinalItems = 512;

// The number of bins in the histogram.
static constexpr int kNumBins = 2048;

// Rows per work-group in the grouped prefill kernel. One work-group takes this
// many consecutive rows, which in a causal prefill means rows of nearly equal
// length, and splits its threads between them. Two measurements motivate it,
// both on the S=3500 triangle with the split drawn at rowLen > topK:
//
//   * At topK=128 the 3372 rows that need a selection cost 75.2 us of the 75.5,
//     and the pure read and write of the same rows is 23.1. The gap is a per
//     row fixed cost of ~16 ns, five work-group barriers deep, and only ~4 rows
//     are in flight per Xe core to hide it. Sharing one barrier between
//     kRowsPerGroup rows divides that cost and multiplies the rows in flight by
//     the same factor.
//
//   * At topK=2048 the 2048 write-only rows cost 15.1 us alone and the 1452
//   rows
//     that need a selection cost 49.1 alone, but together they cost 66.7: the
//     two add up, so none of the write is hidden behind the selection. Grouped
//     rows write kRowsPerGroup * topK words as one contiguous run.
static constexpr int kRowsPerGroup = 4;

template <int kThreads>
static constexpr int kReduceCursorBase = std::max(
    SUBGROUP_SIZE - kRowsPerGroup,
    kThreads / SUBGROUP_SIZE + 2 * kRowsPerGroup);

template <int kThreads>
static constexpr int kReduceScratchSlots =
    kReduceCursorBase<kThreads> + kRowsPerGroup;

// The reduction path (topKPerRowReduce) keeps this many of the row's elements
// per thread in registers, so it can read the row once. 8 covers every row up
// to 8 * 512 = 4096 elements, which is where DSA prefill rows sit; longer rows
// use the same code with a chunked re-read. Elements each thread keeps in
// registers. kFastMaxRegItems * the block size has to cover the longest row
// this path accepts, and both ends of that trade were measured: 256 threads
// with 16 items each lost 20 us of 71 on the S=3500 triangle, and even the bare
// read and write of level 1 lost 11, because half as many threads on a row
// means half as many loads in flight to hide the latency.
static constexpr int kFastMaxRegItems = 8;
static constexpr int kMediumRowCapacity = 10240;

// The reduction path hands the threshold bin's elements to a pairwise ranking,
// which is O(n^2), so it gives up above this many of them and lets the
// multi-step path take the row. The threshold bin holds 44 elements on average
// and 93 at worst over the S=3500 causal triangle; it only overflows this when
// the row's logits are clustered far more tightly than that.
static constexpr int kFastMaxCandidates = 128;

// Above this the accepted and candidate counters no longer share one int in the
// scan that allocates output slots.
static constexpr int kFastMaxTopK = 32768;

// Rows at least this long may hand the pairwise ranking up to half as many
// candidates as the histogram has slots, rather than kFastMaxCandidates, and
// keep them in the histogram: giving up costs them two more reads of a row
// that no longer fits the last level cache, which is more than ranking the
// larger set.
static constexpr int kLongRowMinLen = 16384;

// Rows at least this long read the row as 16-byte vectors.
static constexpr int kVecRowMinLen = 262144;
// The longest row whose pass two reserves candidates once per chunk.
static constexpr int kCandBatchMaxRowLen = 131072;

// The 27 MiB of output for the S=3500 triangle would fit the 32 MB last level
// cache, but the 16 MB of logits streaming past it evicts it, and the two
// streams together then run at the DRAM read+write roof. Asking the row read
// not to keep its lines, with sycl_ext_intel_cache_controls read_hint of
// streaming or uncached at L2 and at L3, was measured on all three settings and
// changed nothing. The structure to store the final items (for the final pass).
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
// as many same-address atomics: Intel Xe has no same-address coalescing for
// subgroup atomics, and on logits quantised to a few fp16 bins (26k of a 49152
// row in one bin) pass one is where the time goes. A shorter row keeps
// kHistCopies, so that it does not pay the wider clear and scan of the
// histogram over a row that is too short to hide them.
static constexpr int kLongHistCopies = 7;

struct Histogram {
  // 16-byte aligned so the reduce path can sweep a thread's bins as one vector.
  alignas(16) int data[kNumBins * kLongHistCopies];
  int tmp[SUBGROUP_SIZE + 1];  // used in prefix sum computation
};

// The histogram and the final-pass items used to share one allocation through
// a union. They cannot any more: processBins() now takes an output slot by
// incrementing its own bin's prefix sum, which lives in histo.data, in the same
// loop that fills items. That costs kNumBins * 4 = 8KB more SLM per
// work-group. On this device the work-group limit at 512 threads is thread
// capacity, not the 384KB SLM pool, so it is free.
struct SmemFinal {
  FinalItems items;
  Histogram histo;
};

// Struct to hold all static sized shared memory objects
struct SharedStates {
  // Shared memory to compute the block sort.
  SmemFinal smemFinal;
  // One slot per row of a group, so the rows sharing a work-group in the
  // grouped kernel each get their own. Slot 0 is the one every other path uses.
  // Shared memory to store the threshold bin.
  int smemThresholdBinIdx[kRowsPerGroup];
  // Shared memory counter to register the candidates for the final phase.
  int smemFinalDstIdx[1];
  // Shared memory to determine if the threshold bin fits in the final items.
  int smemFinalBinSize[kRowsPerGroup];
  // Shared memory to keep track of the top-k values found so far by the
  // previous iterations
  int smemFoundTopKValues[kRowsPerGroup];
  // Set by any row of a group that gave up, so the whole work-group can agree
  // to rerun those rows one at a time on the multi-step path.
  int smemGroupBail[1];
  // Output slot counter for the elements processRefineStep() accepts on their
  // fp16 bin alone.
  int smemRefineDstIdx[1];
};

constexpr size_t alignUp(size_t size, size_t alignment) {
  return ((size + alignment - 1) / alignment) * alignment;
}

constexpr size_t sharedStatesOffset(size_t topK, bool withLogits = false) {
  size_t offset = topK * sizeof(int32_t);
  if (withLogits) {
    offset += topK * sizeof(float);
  }
  return alignUp(offset, alignof(SharedStates));
}

constexpr size_t sharedMemorySize(size_t topK, bool withLogits = false) {
  return sharedStatesOffset(topK, withLogits) + sizeof(SharedStates);
}

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
// times, to clear them, to read the counts and to write the prefix sums back,
// and that traffic was 14.3 us of 74.5 at topK=128.
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
// ladder, then one subgroup scans the sg_num subgroup totals. Two barriers,
// ~12 shuffles, no atomics. tmp needs SUBGROUP_SIZE + 1 ints of SLM; the last
// slot carries the block total, which every thread needs to keep a running
// output cursor without a further reduction.
//
// Trading the second barrier for a repeat of the small scan in every subgroup
// was measured and lost 15 us of 71 on the S=3500 triangle: sycl group scans
// and a broadcast from a runtime lane cost more than the barrier they save.
// kThreadsPerRow makes the scan segmented: the carry stops at every multiple of
// kThreadsPerRow, so one call scans kNumThreadsPerBlock / kThreadsPerRow
// independent rows at once and they share both barriers. sg_id is the block
// wide subgroup index, and total comes back as the caller's own row's total.
template <int kNumThreadsPerBlock, int kThreadsPerRow = kNumThreadsPerBlock>
static inline int blockExclusiveScan(
    const sycl::group<3>& group,
    const sycl::sub_group& sg,
    int lane_id,
    int sg_id,
    int* tmp,
    int val,
    int& total) {
  constexpr int sg_num = kNumThreadsPerBlock / SUBGROUP_SIZE;
  constexpr int sg_per_row = kThreadsPerRow / SUBGROUP_SIZE;
  static_assert(
      sg_num <= SUBGROUP_SIZE, "the totals are scanned by one ladder");
  static_assert(sg_per_row >= 1, "a row keeps whole subgroups");
  static_assert(sg_num % sg_per_row == 0, "the rows split the block evenly");

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
    int v = lane_id < sg_num ? tmp[lane_id] : 0;
    int inc = v;
    for (int offset = 1; offset < SUBGROUP_SIZE; offset *= 2) {
      int other = sycl::shift_group_right(sg, inc, offset);
      if (lane_id >= offset &&
          lane_id / sg_per_row == (lane_id - offset) / sg_per_row) {
        inc += other;
      }
    }
    if (lane_id < sg_num) {
      tmp[lane_id] = inc - v;  // exclusive within the row
      if (lane_id % sg_per_row == sg_per_row - 1) {
        tmp[sg_num + lane_id / sg_per_row] = inc;  // that row's total
      }
    }
  }
  sycl::group_barrier(group);

  total = tmp[sg_num + sg_id / sg_per_row];
  return tmp[sg_id] + inclusive - val;
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

  // How many top-k values the previous steps already committed. It is the base
  // of this step's prefix sums and the first free slot in smemOutput.
  //
  // Step 1 is the exception and must start from zero. Its patternShift is 0, so
  // isPartialMatch() is vacuously true and the step-1 histogram counts the
  // whole row again, not just the elements of step 0's threshold bin. Step 1
  // therefore re-derives the accepted set from scratch, and adding step 0's
  // count on top of a histogram that already contains those elements counts
  // them twice: the threshold bin comes out too early and every element step 0
  // accepted is written to a second slot as well. probe_multistep.py shows the
  // effect (2048 - F unique indices out of topK), and with F large enough the
  // duplicate writes run past smemOutput's topK slots into the shared state
  // behind it. The repo's own worst case, "10LSBits", pins every logit to the
  // same top 22 bits, which forces F == 0 at every step and hides this.
  int lastValue = step == 1 ? 0 : smemFoundTopKValues[0];

  // One block-wide exclusive scan over all kNumBins bins, kBinsPerThread
  // consecutive bins per thread.
  //
  // This replaces a loop of kNumBins / kNumThreadsPerBlock rounds that re-read
  // and re-wrote shared memory every round, serialised a sg_num-element scan of
  // the subgroup totals on lane 0 while the other threads waited at a barrier,
  // and needed a neighbour lookup (plus another barrier) to see the next bin's
  // prefix sum. It gave up an early exit once the threshold bin was found, but
  // the threshold bin is only in an early round for small topK, and the fixed
  // cost of this loop is what dominates the kernel at DSA row lengths: for
  // 3500-element rows it is ~60% of the runtime.
  constexpr int kBinsPerThread = kNumBins / kNumThreadsPerBlock;
  static_assert(kBinsPerThread * kNumThreadsPerBlock == kNumBins);
  // The scan of the subgroup totals below is done by one subgroup in a single
  // ladder, so there must be no more subgroups than lanes.
  static_assert(kNumThreadsPerBlock <= SUBGROUP_SIZE * SUBGROUP_SIZE);
  const int binBase = threadIdx_x * kBinsPerThread;

  // The counts are read twice from SLM rather than held in a register array:
  // keeping kBinsPerThread of them live across the scan pushed the kernel from
  // 64 to 96 GRF per thread, and the occupancy that costs is worth more than
  // the re-read (each thread only ever touches its own bins, so the second read
  // needs no barrier).
  int threadSum = 0;
  for (int j = 0; j < kBinsPerThread; ++j) {
    threadSum += smemFinal.histo.data[binBase + j];
  }

  // Subgroup-level inclusive scan of the per-thread sums.
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

  // Scan the subgroup totals with the whole first subgroup rather than walking
  // them one at a time on a single lane.
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
      smemFinal.histo.tmp[lane_id] = inclusive - val;  // exclusive
    }
  }
  sycl::group_barrier(group);

  // Write the prefix sums back and pick the threshold bin. Exactly one bin
  // satisfies prefix < topK <= prefix + count (an empty bin cannot, since its
  // two bounds are equal), so the owning thread can write the shared state with
  // no reduction and no neighbour lookup. smemFoundTopKValues is set here
  // instead of being accumulated by atomics in processBins(): its value is the
  // threshold bin's prefix sum by construction. The read of that same location
  // above is separated from this write by the two barriers of the scan.
  int prefixSum =
      lastValue + smemFinal.histo.tmp[sg_id] + (sgInclusive - threadSum);
  for (int j = 0; j < kBinsPerThread; ++j) {
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

  // Make sure the prefix sums and the threshold bin are in shared memory.
  sycl::group_barrier(group);

  // The threshold bin.
  thresholdBinIdx = smemThresholdBinIdx[0];

  auto processBins = [&](float logit, int idx) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      if (binIdx < thresholdBinIdx) {
        // The element is part of the top-k selection. Its slot comes from its
        // own bin's prefix sum, so the atomics spread over the bins below the
        // threshold instead of serialising on one counter, which every one of
        // the topK accepted elements in the row used to hit. Step 3 below
        // already allocated its slots this way. Side effect: the output ends up
        // grouped by bin, so it is close to sorted by logit.
        int dstIdx = atomicAdd(&smemFinal.histo.data[binIdx], 1);

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
template <int kNumThreadsPerBlock, int kNumBins>
static constexpr int refineNumBuckets(int refineShift) {
  constexpr int kFastShift = 16 - constexprLog2(kNumBins);
  const int reached =
      ((1 << (kFastShift + 13)) + 2 * static_cast<int>(kRefineSlack)) >>
      refineShift;
  return ((reached + kNumThreadsPerBlock - 1) / kNumThreadsPerBlock) *
         kNumThreadsPerBlock;
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
// This replaces steps 0 and 1 of the chain above for such rows, and both were
// wasted work there. Step 0 rebuilds the reduction path's histogram bit for bit
// (the same top 11 bits of the fp16 key). Step 1 then starts over on the top 11
// bits of the fp32 key, which are sign, 8 exponent bits and only 2 mantissa
// bits, so its bins are 8 fp16 bins wide and its threshold bin is about 8 times
// too large again. At s_kv=49152 the fp16 threshold bin holds ~910 elements, so
// neither fits and the row goes on to step 2: seven reads of the row where
// CUDA, whose final buffer is 2048 slots, makes two.
//
// Instead this keeps what the reduction path already established, the threshold
// fp16 bin fastBin and the `accepted` elements in the bins before it, and
// histograms only that bin's elements, 256 fp32 keys per bucket. Its ~910
// elements then spread over ~1000 buckets, so the new threshold bucket holds a
// handful and the final rank is over those rather than over ~910. Two reads of
// the row, histogram and emit, after the reduction path's one.
//
// Returns false, having written nothing but shared state, if even the threshold
// bucket does not fit kNumFinalItems (a row of mostly exact duplicates). The
// caller then finishes the row with processTieSelect().
template <
    int kNumThreadsPerBlock,
    int kNumBins,
    int kNumFinalItems,
    typename SmemFinalType>
static bool processRefineStep(
    const sycl::nd_item<3>& item,
    const float* logits,
    int rowStart,
    int rowEnd,
    int fastBin,
    int refineShift,
    int accepted,
    int32_t* smemOutput,
    int* smemThresholdBinIdx,
    int* smemFinalDstIdx,
    int* smemFinalBinSize,
    int* smemFoundTopKValues,
    int* smemRefineDstIdx,
    SmemFinalType& smemFinal,
    int stride1,
    int topK) {
  sycl::group group = item.get_group();
  auto sg = item.get_sub_group();
  auto threadIdx_x = item.get_local_id(0);
  auto lane_id = threadIdx_x % SUBGROUP_SIZE;
  auto sg_id = threadIdx_x / SUBGROUP_SIZE;
  auto sg_num = (kNumThreadsPerBlock + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE;

  // Buckets this pass splits fastBin into, out of the ints of the borrowed
  // histogram, which is kLongHistCopies times what the fp16 pass needed. A row
  // packed into one fp16 bin then splits fine enough for its threshold bucket
  // to fit the final items, instead of giving up here and paying the whole
  // fp32 chain (six more reads of the row).
  const int numBuckets =
      refineNumBuckets<kNumThreadsPerBlock, kNumBins>(refineShift);
  static_assert(
      refineNumBuckets<kNumThreadsPerBlock, kNumBins>(kLongRefineShift) <=
          kNumBins * kLongHistCopies,
      "the buckets of the finest shift fit the histogram this borrows");
  const uint32_t base = refineBase<kNumBins>(fastBin);
  // Only called on elements of fastBin, which lie in the first numBuckets
  // buckets. The clamp is there for memory safety only.
  auto bucketOf = [&](float logit) {
    const uint32_t b = (convert_to_uint32(logit) - base) >> refineShift;
    return static_cast<int>(
        sycl::min(b, static_cast<uint32_t>(numBuckets - 1)));
  };

  for (int idx = threadIdx_x; idx < numBuckets; idx += kNumThreadsPerBlock) {
    smemFinal.histo.data[idx] = 0;
  }
  if (threadIdx_x == 0) {
    smemRefineDstIdx[0] = 0;
  }
  sycl::group_barrier(group);

  auto distributeToBins = [&](float logit, int /* idx */ = 0) {
    if (static_cast<int>(extractFastBinIdx<kNumBins>(logit)) == fastBin) {
      atomicAdd(&smemFinal.histo.data[bucketOf(logit)], 1);
    }
  };
  if (stride1 == 1) {
    vectorizedProcessDeep<4>(
        threadIdx_x,
        kNumThreadsPerBlock,
        logits + rowStart,
        rowEnd - rowStart,
        static_cast<int>(item.get_group_linear_id()),
        distributeToBins);
  } else {
    for (int idx = rowStart + threadIdx_x; idx < rowEnd;
         idx += kNumThreadsPerBlock) {
      distributeToBins(logits[idx * stride1], idx);
    }
  }
  sycl::group_barrier(group);

  // The same block-wide exclusive scan as processHistogramStep(), based at the
  // count of elements the fp16 bins before fastBin already accepted.
  // refineNumBuckets() rounds up to a multiple of the threads, so the buckets
  // split evenly.
  const int binsPerThread = numBuckets / kNumThreadsPerBlock;
  static_assert(kNumThreadsPerBlock <= SUBGROUP_SIZE * SUBGROUP_SIZE);
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
  if (stride1 == 1) {
    vectorizedProcessDeep<4>(
        threadIdx_x,
        kNumThreadsPerBlock,
        logits + rowStart,
        rowEnd - rowStart,
        static_cast<int>(item.get_group_linear_id()),
        processBins);
  } else {
    for (int idx = rowStart + threadIdx_x; idx < rowEnd;
         idx += kNumThreadsPerBlock) {
      processBins(logits[idx * stride1], idx);
    }
  }
  sycl::group_barrier(group);
  return true;
}

// Digit width of processTieSelect(): 4096 buckets, 8 per thread at 512
// threads, in the ints of the histogram it borrows.
static constexpr int kTieSelectDigitBits = 12;

// Takes over a single-block row whose fp16 threshold bin neither the reduction
// path nor processRefineStep() could split: exact duplicates, or values that
// all round into the zero (or inf) fp16 bin, which processRefineStep() cannot
// split at all. Those rows used to go through steps 1 to 3, seven or more reads
// of the row, each of whose histogram passes funnels the whole tie into one
// SLM address one atomic at a time. At s_q=2730, s_kv=49152, topK=2048 a
// prefill with every logit 0.0 (dummy weights) took 3.0 ms against 0.93 ms on
// spread logits; this takes it to 1.33 ms.
//
// Instead this ranks only fastBin's elements, on their fp32 key relative to the
// smallest one, and only on as many bits as their spread needs:
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
template <
    int kNumThreadsPerBlock,
    int kNumBins,
    int kDigitBits = kTieSelectDigitBits>
static void processTieSelect(
    const sycl::nd_item<3>& item,
    const float* rowBase,
    int rowLen,
    int stride1,
    int fastBin,
    int accepted,
    int* outIndices,
    int* hist,
    int* scanTmp,
    int* selVars,
    int topK) {
  constexpr int kItems = 4;
  constexpr int kNumDigitBuckets = 1 << kDigitBits;
  constexpr int kBucketsPerThread = kNumDigitBuckets / kNumThreadsPerBlock;
  static_assert(
      kNumDigitBuckets % kNumThreadsPerBlock == 0,
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
  // is kNumThreadsPerBlock apart from element k + 1, so each load is one
  // coalesced run across the subgroup.
  constexpr int kPerIteration = kNumThreadsPerBlock * kItems;
  const int numIterations = (rowLen + kPerIteration - 1) / kPerIteration;
  auto walk = [&](auto&& fn) {
    for (int it = 0; it < numIterations; ++it) {
      const int base = it * kPerIteration + tid;
      float v[kItems];
#pragma unroll
      for (int k = 0; k < kItems; ++k) {
        const int j = base + k * kNumThreadsPerBlock;
        v[k] = j < rowLen ? rowBase[static_cast<int64_t>(j) * stride1] : 0.f;
      }
#pragma unroll
      for (int k = 0; k < kItems; ++k) {
        const int j = base + k * kNumThreadsPerBlock;
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
    shift = sycl::max(0, rangeBits - kDigitBits);
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
      int before = blockExclusiveScan<kNumThreadsPerBlock>(
          group, sg, lane, sgId, scanTmp, sum, total);
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
      shift = sycl::max(0, shift - kDigitBits);
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
// memory if the row is not one it can finish, and the caller falls back to the
// processHistogramStep() chain above.
//
// What it does differently from that chain, and why:
//
//   * The bins are split evenly over the threads, so clearing the histogram is
//     kBinsPerThread stores, the prefix sum is one value per thread plus a
//     local walk, and the threshold bin needs neither a search nor a broadcast:
//     the thread whose own prefix brackets topK is the one that owns it. 512
//     bins (exactly one per thread) were measured against 2048: the histogram
//     and the scan cost the same either way, but the threshold bin then holds
//     44 elements on average instead of 12 over the S=3500 causal triangle, and
//     the final rank below is quadratic in that count. So the wider histogram
//     wins, since everything after this point scales with that count and 44 is
//     still tiny.
//
//   * Nothing is staged in smemOutput. Slots come from one work-group scan of
//     the per-thread accepted counts, so the ~2048 atomicAdd()s that every
//     accepted element in the row used to aim at a single counter become zero
//     atomics, and the accepted indices go straight to global memory. The
//     output order is unspecified, so no ordering is lost.
//
//   * The row is read once. Each thread keeps its own elements in registers
//     between the histogram and the scatter (kRegResident), so the scatter
//     needs no second load. Rows too long for that re-read in chunks, which is
//     still the two passes the other path always does.
//
//   * The elements sharing the threshold bin are compacted by the same scan and
//     ranked pairwise, exactly as the final pass of the other path does, but
//     over the ~44 of them rather than up to kNumFinalItems.
// kThreadsPerRow is how many of the block's threads work on this row: the whole
// block when one row owns the work-group, or a slice of it when kRowsPerGroup
// rows share one. Everything below is row local except the two barriers of the
// scan and the barriers between stages, which stay block wide and are exactly
// what the rows of a group share. kNumFastBins, kMaxCandidates and the items
// buffer are the per row slices the caller hands over.
template <
    int kNumThreadsPerBlock,
    int kThreadsPerRow,
    int kNumFastBins,
    int kCopies,
    int kMaxRegItems,
    int kMaxCandidates,
    bool kRegResident,
    int kRowCapacity = std::numeric_limits<int>::max()>
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
    int* smemGroupBail,
    int stride1,
    int topK) {
  // True when this row shares its work-group with others, which changes only
  // how giving up works: see the bail-out below.
  constexpr bool kGrouped = kThreadsPerRow != kNumThreadsPerBlock;
  constexpr int kBinsPerThread = kNumFastBins / kThreadsPerRow;
  static_assert(
      kBinsPerThread * kThreadsPerRow == kNumFastBins,
      "the bins are split evenly over the threads");
  static_assert(
      kMaxCandidates <= kNumFinalItems, "shares the final-items slots");
  static_assert(
      kNumFastBins * kCopies <= kNumBins * kLongHistCopies,
      "the bins and their copies fit the histogram");
  static_assert(
      kThreadsPerRow % SUBGROUP_SIZE == 0, "a row keeps whole subgroups");

  sycl::group group = item.get_group();
  auto sg = item.get_sub_group();
  // Row local thread id: the rows of a group take contiguous slices of the
  // block, so a subgroup never straddles two rows.
  const int threadIdx_x = item.get_local_id(0) % kThreadsPerRow;
  const int lane_id = threadIdx_x % SUBGROUP_SIZE;
  const int sg_id = threadIdx_x / SUBGROUP_SIZE;
  const int sg_id_block = item.get_local_id(0) / SUBGROUP_SIZE;
  const int binBase = threadIdx_x * kBinsPerThread;
  // Output cursor of the accepted elements, one per row of the group, in scan
  // scratch that blockExclusiveScan() leaves alone (it uses the first
  // kNumThreadsPerBlock / SUBGROUP_SIZE + kRowsPerGroup slots).
  // A 1024-thread group occupies all 33 scan slots. The scratch-size formula
  // leaves the emission counters disjoint from that scan.
  constexpr int kCursorBase = kReduceCursorBase<kNumThreadsPerBlock>;
  static_assert(
      kNumThreadsPerBlock / SUBGROUP_SIZE + kRowsPerGroup <=
      kCursorBase - kRowsPerGroup);
  int* acceptCursor =
      smemTmp + kCursorBase + item.get_local_id(0) / kThreadsPerRow;
  int* candCursor = acceptCursor - kRowsPerGroup;
  const int histCopy = lane_id % kCopies;

  constexpr int kChunk = kMaxRegItems * kThreadsPerRow;
  constexpr int kNumReg = kRegResident ? kMaxRegItems : 1;
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
  auto regOffset = [&](int i) { return threadIdx_x + i * kThreadsPerRow; };

  // How many of this thread's kMaxRegItems slots hold an element of the row.
  // Every loop below stops there instead of running the full kMaxRegItems and
  // masking, because the slots are handed out with a stride of kThreadsPerRow:
  // the 32 lanes of a subgroup agree on this count to within one, so a short
  // row really does leave the loop early instead of paying for kMaxRegItems
  // predicated iterations. At topK=128 the full triangle spends 4096 slot
  // visits on rows of a few hundred elements, which was 26 of the 95 us in the
  // histogram alone.
  const int regItems =
      kRegResident
          ? (rowLen - threadIdx_x + kThreadsPerRow - 1) / kThreadsPerRow
          : kMaxRegItems;

  // The chunked passes read a unit-stride row as 16-byte vectors, two per
  // thread per chunk: the first `head` elements up to 16-byte alignment are
  // taken on their own. With only a few long rows in flight a work-group has
  // to keep its own reads streaming, and a vec4 load moves four times as much
  // per message as the strided scalar ones.
  static_assert(kRegResident || kMaxRegItems % 4 == 0);
  const bool vecLoads = !kRegResident && stride1 == 1 &&
                        kRowCapacity >= kVecRowMinLen &&
                        rowLen >= kVecRowMinLen;
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
  auto loadChunk =
      [&](int chunk, float (&val)[kMaxRegItems], int (&off)[kMaxRegItems]) {
        if (vecLoads) {
          const float* body = rowBase + head;
#pragma unroll
          for (int j = 0; j < kMaxRegItems / 4; ++j) {
            const int o = chunk + (threadIdx_x + j * kThreadsPerRow) * 4;
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
        } else if (stride1 == 1) {
#pragma unroll
          for (int i = 0; i < kMaxRegItems; ++i) {
            off[i] = chunk + threadIdx_x + i * kThreadsPerRow;
            val[i] = off[i] < rowLen ? rowBase[off[i]] : 0.f;
            off[i] = off[i] < rowLen ? off[i] : rowLen;
          }
        } else {
#pragma unroll
          for (int i = 0; i < kMaxRegItems; ++i) {
            off[i] = chunk + threadIdx_x + i * kThreadsPerRow;
            val[i] = off[i] < rowLen ? rowBase[off[i] * stride1] : 0.f;
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
  // kNumFastBins addresses, ~one collision per subgroup, and are not the
  // contended ones the scatter below removes.
  //
  // Reading the row before the barrier that publishes the cleared histogram, so
  // that the barrier waits behind loads already in flight, was measured and
  // lost 3 us of 71: the threads then reach their loads at different times and
  // the row reads stop arriving as one front.
  if constexpr (kRegResident) {
    // Both loops run over the whole register set before the next one starts.
    // Folding the atomicAdd() into the fill makes every iteration wait on its
    // own load, so a row costs one memory latency per element instead of one in
    // total; that alone was enough to make this path slower than the two-pass
    // one it replaces. Measured: 38.8 of the 96 us this kernel spent on the
    // S=3500 triangle went into the fill. Reading the row is the largest single
    // cost of this kernel, so the unit stride case gets its own loop. With a
    // runtime stride the address of lane l is base + l * stride1 * 4, which the
    // compiler can only lower to a per lane gather; with the stride known to be
    // one the 32 lanes of a load cover 128 consecutive bytes and it lowers to a
    // linear load instead.
    auto fillFrom = [&](auto row) {
#pragma unroll
      for (int i = 0; i < kMaxRegItems; ++i) {
        if (i >= regItems) {
          break;
        }
        regVal[i] = row[regOffset(i)];
      }
    };
    if (stride1 == 1) {
      fillFrom(rowBase);
    } else {
#pragma unroll
      for (int i = 0; i < kMaxRegItems; ++i) {
        if (i >= regItems) {
          break;
        }
        regVal[i] = rowBase[regOffset(i) * stride1];
      }
    }
#pragma unroll
    for (int i = 0; i < kMaxRegItems; ++i) {
      if (i >= regItems) {
        break;
      }
      regBin[i] = extractFastBinIdx<kNumFastBins>(regVal[i]);
      atomicAdd(&smemBins[regBin[i] * kCopies + histCopy], 1);
    }
  } else {
    // Same two-loop shape as the register-resident case above, one chunk at a
    // time: a chunk is staged in registers before any of its atomics, so a
    // chunk costs one memory latency instead of one per element. This path
    // takes the same care as the register-resident one for the same reason.
    // Walking the row with the atomic folded into the load, here and in pass
    // two, was measured on a variant that ran the whole S=3500 triangle through
    // this path and cost 4.6 us of 75.0 at topK=128, 9.6 of 92.5 at 512
    // and 24.1 of 99.5 at 2048.
    if (threadIdx_x < head) {
      atomicAdd(
          &smemBins
              [extractFastBinIdx<kNumFastBins>(rowBase[threadIdx_x]) * kCopies +
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
      const int sample = static_cast<int>(
          extractFastBinIdx<kNumFastBins>(rowBase[s * stride1]));
      const int lead = sycl::group_broadcast(sg, sample, 0);
      int best = -1;
#pragma unroll
      for (int w = -(kHotBins - 1); w <= kHotBins - 1; ++w) {
        const int lo = sycl::clamp(lead + w, 0, kNumFastBins - kHotBins);
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
      // A lane counts at most rowLen / kThreadsPerRow elements per bin.
      hotLoop = best >= kHotMinLanes &&
                rowLen / kThreadsPerRow < (1 << kHotBits) - kMaxRegItems;
    }
    if (hotLoop) {
      uint64_t hotCount = 0;
      for (int c = 0; c < numChunks; ++c) {
        const int chunk = chunkAt(c);
        float chunkVal[kMaxRegItems];
        int chunkOff[kMaxRegItems];
        loadChunk(chunk, chunkVal, chunkOff);
#pragma unroll
        for (int i = 0; i < kMaxRegItems; ++i) {
          const int bin =
              chunkOff[i] < rowLen
                  ? static_cast<int>(
                        extractFastBinIdx<kNumFastBins>(chunkVal[i]))
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
        float chunkVal[kMaxRegItems];
        int chunkOff[kMaxRegItems];
        loadChunk(chunk, chunkVal, chunkOff);
#pragma unroll
        for (int i = 0; i < kMaxRegItems; ++i) {
          if (chunkOff[i] < rowLen) {
            atomicAdd(
                &smemBins
                    [extractFastBinIdx<kNumFastBins>(chunkVal[i]) * kCopies +
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
  int prefix = blockExclusiveScan<kNumThreadsPerBlock, kThreadsPerRow>(
      group, sg, lane_id, sg_id_block, smemTmp, threadSum, rowTotal);
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
  const bool bigCandidates = !kGrouped && !kRegResident &&
                             kRowCapacity >= kLongRowMinLen &&
                             rowLen >= kLongRowMinLen;
  if (bigCandidates) {
    itemLogits = reinterpret_cast<float*>(smemBins);
    itemIndices = smemBins + kNumFastBins * kCopies / 2;
  }

  // rowLen > topK, so some bin is always the threshold bin. Bail out only when
  // its elements would not fit the compacted candidate buffer: nothing has been
  // written to global memory yet, so the caller can start over.
  //
  // When several rows share the work-group one row cannot return on its own:
  // the rows behind it would wait forever at the barrier below. So the group
  // agrees first, at the cost of one more barrier, and all of its rows start
  // over together. smemGroupBail was cleared by the caller, before the barrier
  // that published the cleared histogram.
  if constexpr (kGrouped) {
    if (smemFinalBinSize[0] > kMaxCandidates && threadIdx_x == 0) {
      smemGroupBail[0] = 1;
    }
    sycl::group_barrier(group);
    if (smemGroupBail[0] != 0) {
      return false;
    }
  } else if (
      smemFinalBinSize[0] >
      (bigCandidates ? kNumFastBins * kCopies / 2 : kMaxCandidates)) {
    return false;
  }

  // Pass two. The slot an element gets is its bin's first output slot plus its
  // rank inside that bin, and the rank comes from bumping the bin's own
  // counter, which is the prefix the scan just left in shared memory. So there
  // is no second scan, no per chunk barrier and no cross lane traffic at all:
  // one shared memory atomic per element, spread over kNumFastBins addresses
  // that hold ~2 elements each. The block wide scan this replaces cost 11.2 us
  // of 84.7 on the S=3500 triangle, against ~1 us for the whole first pass of
  // the same atomics.
  //
  // Elements below the threshold bin go straight to global memory, in
  // descending bin order rather than index order. Elements of the threshold bin
  // land in the candidate buffer instead, at their rank inside the bin, ready
  // for the rank below. Everything above it has nowhere to go.
  const int thresholdBin = smemThresholdBinIdx[0];
  const int candBase = smemFoundTopKValues[0];
  // Skipping a thread whose elements are all rejected was measured here and
  // lost 1.9 us of 74.5. The test is per thread but the hardware is per
  // subgroup: at topK=128 the threshold sits near the row's 87th percentile, so
  // all 64 elements of a subgroup are rejected with probability 0.87^64, and
  // the subgroup walks the slots either way. Only a subgroup-uniform predicate
  // can skip work here.
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
    uint32_t accMask[kMaxRegItems];
    int sgAccepted = 0;
#pragma unroll
    for (int i = 0; i < kMaxRegItems; ++i) {
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
    for (int i = 0; i < kMaxRegItems; ++i) {
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
      const int bin = live
                          ? static_cast<int>(extractFastBinIdx<kNumFastBins>(x))
                          : kNumFastBins;
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
    if (kRowCapacity > kCandBatchMaxRowLen && rowLen > kCandBatchMaxRowLen) {
      for (int c = numChunks - 1; c >= 0; --c) {
        const int chunk = chunkAt(c);
        float chunkVal[kMaxRegItems];
        int chunkOff[kMaxRegItems];
        loadChunk(chunk, chunkVal, chunkOff);
        uint32_t accMask[kMaxRegItems];
        int sgAccepted = 0;
#pragma unroll
        for (int i = 0; i < kMaxRegItems; ++i) {
          // Elements past the end of the row get a bin one past the last, which
          // no threshold bin can match.
          const int bin =
              chunkOff[i] < rowLen
                  ? static_cast<int>(
                        extractFastBinIdx<kNumFastBins>(chunkVal[i]))
                  : kNumFastBins;
          accMask[i] = ballot(bin < thresholdBin);
          sgAccepted += sycl::popcount(accMask[i]);
          emitCandidates(bin == thresholdBin, chunkOff[i], chunkVal[i]);
        }
        int base = reserve(sgAccepted);
#pragma unroll
        for (int i = 0; i < kMaxRegItems; ++i) {
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
        float chunkVal[kMaxRegItems];
        int chunkOff[kMaxRegItems];
        loadChunk(chunk, chunkVal, chunkOff);
        uint32_t accMask[kMaxRegItems];
        uint32_t candMask[kMaxRegItems];
        int sgAccepted = 0;
        int sgCands = 0;
#pragma unroll
        for (int i = 0; i < kMaxRegItems; ++i) {
          // Elements past the end of the row get a bin one past the last, which
          // no threshold bin can match.
          const int bin =
              chunkOff[i] < rowLen
                  ? static_cast<int>(
                        extractFastBinIdx<kNumFastBins>(chunkVal[i]))
                  : kNumFastBins;
          accMask[i] = ballot(bin < thresholdBin);
          sgAccepted += sycl::popcount(accMask[i]);
          candMask[i] = ballot(bin == thresholdBin);
          sgCands += sycl::popcount(candMask[i]);
        }
        int base = reserve(sgAccepted);
#pragma unroll
        for (int i = 0; i < kMaxRegItems; ++i) {
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
          for (int i = 0; i < kMaxRegItems; ++i) {
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

  // The threshold bin, ranked pairwise on the exact float values. Same
  // comparison and same index tie-break as the other path's final pass, so the
  // set of values selected is identical.
  const int numCandidates = smemFinalBinSize[0];
  const int accepted = smemFoundTopKValues[0];
  const int remainedTopK = topK - accepted;
  // A subgroup per candidate, and the whole subgroup walks the candidate list,
  // so the quadratic work is spread over all the threads instead of over the
  // numCandidates of them that own a candidate. It matters most where the
  // candidate set is widest: at topK=128 the threshold bin covers a decade of
  // the value range and holds tens of elements, and this stage was 10.3 us
  // of 82.
  // A long row's candidate set can hold thousands of elements, and ranking
  // those pairwise is quadratic: at 32 rows of 65536 it doubled the kernel. It
  // is cut down by a radix select instead, kSelBits of the exact float key per
  // pass. The keys are taken as offsets from the smallest one, and the shared
  // fp16 bin keeps those to about 19 bits, so that is two or three passes over
  // the candidates. The ranks within the selected set are not needed: any order
  // of the top-k is a valid output.
  if constexpr (!kGrouped && !kRegResident && kRowCapacity >= kLongRowMinLen) {
    if (numCandidates > kMaxCandidates) {
      constexpr int kSelBins = kThreadsPerRow;
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
      for (int i = threadIdx_x; i < numCandidates; i += kThreadsPerRow) {
        const uint32_t key = convert_to_uint32(itemLogits[i]);
        lo = sycl::min(lo, key);
        hi = sycl::max(hi, key);
      }
      const uint32_t minKey =
          sycl::reduce_over_group(group, lo, sycl::minimum<uint32_t>());
      const uint32_t range =
          sycl::reduce_over_group(group, hi, sycl::maximum<uint32_t>()) -
          minKey;
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
        for (int i = threadIdx_x; i < numCandidates; i += kThreadsPerRow) {
          const uint64_t off = offsetOf(i);
          if ((off >> shift) == prefix) {
            atomicAdd(&selHist[(off >> low) & digitMask], 1);
          }
        }
        sycl::group_barrier(group);
        const int count = selHist[threadIdx_x];
        int total = 0;
        const int before =
            blockExclusiveScan<kNumThreadsPerBlock, kThreadsPerRow>(
                group, sg, lane_id, sg_id_block, smemTmp, count, total);
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
      for (int i = threadIdx_x; i - lane_id < numCandidates;
           i += kThreadsPerRow) {
        const bool live = i < numCandidates;
        const uint64_t top = live ? offsetOf(i) >> shift : ~uint64_t{0};
        const int index = live ? itemIndices[i] : 0;
        uint32_t mask = ballot(top < prefix);
        if (mask != 0) {
          int base = 0;
          if (lane_id == 0) {
            base =
                atomicAdd(lessCursor, static_cast<int>(sycl::popcount(mask)));
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
  }
  constexpr int kNumSubgroups = kThreadsPerRow / SUBGROUP_SIZE;
  for (int i = sg_id; i < numCandidates; i += kNumSubgroups) {
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

  size_t sharedStatesOffsetBytes =
      sharedStatesOffset(topK, multipleBlocksPerRow);

  int32_t* smemOutput = reinterpret_cast<int32_t*>(smem_buf);
  SharedStates* sharedStates =
      reinterpret_cast<SharedStates*>(smem_buf + sharedStatesOffsetBytes);

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

  // The threshold fp16 bin the reduction path gave up on, and the number of
  // elements in the bins before it, for processRefineStep(). -1 if it did not
  // run.
  int refineBin = -1;
  int refineAccepted = 0;

  // The reduction path handles the plain single-block case, which is every
  // prefill row and every decode row short enough not to be split. It writes
  // its own output, or returns false having written nothing.
  if constexpr (!multipleBlocksPerRow && !mergeBlocks) {
    if (topK <= kFastMaxTopK) {
      const float* rowBase = logits + static_cast<int64_t>(rowStart) * stride1;
      // Three shapes of the same pass: the register-resident one for a row
      // that fits kFastMaxRegItems slots per thread, and for a longer row the
      // chunked one, with the extra histogram copies once the row is long
      // enough to hide their clear and scan.
      bool done;
      if (rowLen <= kFastMaxRegItems * kNumThreadsPerBlock) {
        done = topKPerRowReduce<
            kNumThreadsPerBlock,
            kNumThreadsPerBlock,
            kNumBins,
            kHistCopies,
            kFastMaxRegItems,
            kFastMaxCandidates,
            /*kRegResident=*/true>(
            item,
            rowBase,
            rowLen,
            outIndices,
            smemFinal.histo.data,
            smemFinal.histo.tmp,
            smemFinal.items.indices,
            smemFinal.items.logits,
            smemThresholdBinIdx,
            smemFinalBinSize,
            smemFoundTopKValues,
            sharedStates->smemGroupBail,
            stride1,
            topK);
      } else if (rowLen >= kLongRowMinLen) {
        done = topKPerRowReduce<
            kNumThreadsPerBlock,
            kNumThreadsPerBlock,
            kNumBins,
            kLongHistCopies,
            kFastMaxRegItems,
            kFastMaxCandidates,
            /*kRegResident=*/false>(
            item,
            rowBase,
            rowLen,
            outIndices,
            smemFinal.histo.data,
            smemFinal.histo.tmp,
            smemFinal.items.indices,
            smemFinal.items.logits,
            smemThresholdBinIdx,
            smemFinalBinSize,
            smemFoundTopKValues,
            sharedStates->smemGroupBail,
            stride1,
            topK);
      } else {
        done = topKPerRowReduce<
            kNumThreadsPerBlock,
            kNumThreadsPerBlock,
            kNumBins,
            kHistCopies,
            kFastMaxRegItems,
            kFastMaxCandidates,
            /*kRegResident=*/false>(
            item,
            rowBase,
            rowLen,
            outIndices,
            smemFinal.histo.data,
            smemFinal.histo.tmp,
            smemFinal.items.indices,
            smemFinal.items.logits,
            smemThresholdBinIdx,
            smemFinalBinSize,
            smemFoundTopKValues,
            sharedStates->smemGroupBail,
            stride1,
            topK);
      }
      if (done) {
        return;
      }
      // Everything the reduction path touched is either re-initialised below or
      // cleared by processHistogramStep() before it is read.
      sycl::group_barrier(group);
      // The bin it gave up on is read before anything below rewrites it.
      refineBin = smemThresholdBinIdx[0];
      refineAccepted = smemFoundTopKValues[0];
      // Every thread holds its copy before thread 0 resets them below.
      sycl::group_barrier(group);
    }
  }

  // Initialize values
  if (threadIdx_x == 0) {
    smemFinalDstIdx[0] = 0;
    smemFoundTopKValues[0] = 0;
  }
  sycl::group_barrier(group);
  int thresholdBinIdx = -1;
  uint32_t logitPattern = 0;

  // Split the reduction path's threshold bin further if its exponent allows it.
  // Should even its threshold bucket be too large, processTieSelect() finishes
  // the row, so a row the reduction path gave up on never reaches the steps.
  bool refined = false;
  if constexpr (!multipleBlocksPerRow && !mergeBlocks) {
    if (refineBin >= 0 && fastBinIsRefinable<kNumBins>(refineBin)) {
      // Same split as the reduction path above: only a long row is worth the
      // finer buckets, and only a long row can have given up with them.
      refined =
          processRefineStep<kNumThreadsPerBlock, kNumBins, kNumFinalItems>(
              item,
              logits,
              rowStart,
              rowEnd,
              refineBin,
              rowLen >= kLongRowMinLen ? kLongRefineShift
                                       : kLongRefineShift + 2,
              refineAccepted,
              smemOutput,
              smemThresholdBinIdx,
              smemFinalDstIdx,
              smemFinalBinSize,
              smemFoundTopKValues,
              sharedStates->smemRefineDstIdx,
              smemFinal,
              stride1,
              topK);
    }
    // Neither could split the bin: finish the row on the fp32 keys of that bin
    // alone rather than on steps 1 to 3, which re-derive everything from the
    // whole row and funnel a tie into one SLM address per pass.
    if (refineBin >= 0 && !refined) {
      processTieSelect<kNumThreadsPerBlock, kNumBins>(
          item,
          logits + static_cast<int64_t>(rowStart) * stride1,
          rowLen,
          stride1,
          refineBin,
          refineAccepted,
          outIndices,
          smemFinal.histo.data,
          smemFinal.histo.tmp,
          smemFinal.items.indices,
          topK);
      return;
    }
  }

  // Step 0: Process first 11 bits of half representation
  bool continueToNextStep = !refined && processHistogramStep<
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

// kRows of a group's rows at a time, each on its own kNumThreadsPerBlock /
// kRows threads, its own slice of the histogram and its own slice of the
// candidate buffer. Every barrier is shared by the kRows rows, so the per row
// fixed cost of the selection is divided by kRows, and kRows times as many rows
// are in flight per core to hide what is left. Bins per thread stay at four
// however the block is split, because the histogram is sliced by the same
// factor as the threads, so the bins hold the same ~2 elements each as they do
// for a whole-block row.
//
// Returns false if any row of any sub-batch gave up; that decision is group
// wide, so all of the group's rows are then left for the caller to redo one at
// a time.
template <int kNumThreadsPerBlock, int kRows, int kGroupRows>
static bool topKPerRowGroupReduce(
    const sycl::nd_item<3>& item,
    SharedStates* sharedStates,
    const float* logits,
    const int32_t* rowStartArr,
    const int32_t* rowEndArr,
    int32_t* indices,
    int firstRow,
    int64_t stride0,
    int64_t stride1,
    int topK) {
  constexpr int kThreadsPerRow = kNumThreadsPerBlock / kRows;
  constexpr int kBins = kNumBins / kRows;
  static_assert(kGroupRows % kRows == 0, "the sub-batches cover the group");
  static_assert(
      kFastMaxCandidates * kRows <= kNumFinalItems,
      "every row of a sub-batch gets its own candidate slots");

  const int local_id = item.get_local_id(0);
  const int slot = local_id / kThreadsPerRow;
  auto& smemFinal = sharedStates->smemFinal;

  for (int base = 0; base < kGroupRows; base += kRows) {
    const int row = firstRow + base + slot;
    const int rowStart = rowStartArr[row];
    const int rowLen = rowEndArr[row] - rowStart;
    const float* rowBase = logits + static_cast<int64_t>(row) * stride0 +
                           static_cast<int64_t>(rowStart) * stride1;
    if (local_id == 0) {
      sharedStates->smemGroupBail[0] = 0;
    }
    const bool done = topKPerRowReduce<
        kNumThreadsPerBlock,
        kThreadsPerRow,
        kBins,
        kHistCopies,
        kFastMaxRegItems,
        kFastMaxCandidates,
        /*kRegResident=*/true>(
        item,
        rowBase,
        rowLen,
        indices + static_cast<int64_t>(row) * topK,
        smemFinal.histo.data + slot * kBins * kHistCopies,
        smemFinal.histo.tmp,
        smemFinal.items.indices + slot * kFastMaxCandidates,
        smemFinal.items.logits + slot * kFastMaxCandidates,
        sharedStates->smemThresholdBinIdx + slot,
        sharedStates->smemFinalBinSize + slot,
        sharedStates->smemFoundTopKValues + slot,
        sharedStates->smemGroupBail,
        stride1,
        topK);
    // Uniform over the work-group: every row of a sub-batch returns the same.
    if (!done) {
      return false;
    }
    if (base + kRows < kGroupRows) {
      // The next sub-batch clears the histogram and refills the candidate slots
      // this one is still reading.
      sycl::group_barrier(item.get_group());
    }
  }
  return true;
}

// Prefill, kRowsPerGroup consecutive rows per work-group. Consecutive is what
// makes it work: in a causal prefill those rows differ in length by less than
// kRowsPerGroup, so they always agree on which path to take and how many
// threads each of them can afford, and the group never has to straddle two
// cases.
template <int kNumThreadsPerBlock>
class top_k_per_row_prefill_group_kernel {
 public:
  top_k_per_row_prefill_group_kernel(
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

  // The same pairing as the single-row kernel, one level up, but taken by one
  // work-group instead of two: a work-group does a group of short rows and the
  // mirrored group of long ones, so the output writes of the short rows sit
  // next to the selection of the long ones and, more importantly, the two
  // groups add up to the same total length for every pair. Leaving the pairing
  // to the hardware only worked while there were enough work-groups per core to
  // average the imbalance out; owning both ends makes every work-group's share
  // of the triangle equal by construction, which is what lets kRowsPerGroup
  // grow.
  void operator() [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] (
      const sycl::nd_item<3>& item) const {
    const int64_t numBlocks = item.get_group_range(0);
    const int64_t blockIdx_x = item.get_group(0);
    const int64_t block = (blockIdx_x & 1) ? numBlocks - 1 - (blockIdx_x >> 1)
                                           : (blockIdx_x >> 1);
    processGroup(item, block);
  }

 private:
  void processGroup(const sycl::nd_item<3>& item, int64_t block) const {
    const int local_id = item.get_local_id(0);
    const int topK = static_cast<int>(topK_);
    const int firstRow = static_cast<int>(block) * kRowsPerGroup +
                         static_cast<int>(offsetIndex_);

    int minLen = std::numeric_limits<int>::max();
    int maxLen = 0;
#pragma unroll
    for (int r = 0; r < kRowsPerGroup; ++r) {
      const int len = rowEnd_[firstRow + r] - rowStart_[firstRow + r];
      minLen = std::min(minLen, len);
      maxLen = std::max(maxLen, len);
    }

    char* smem_buf =
        slm_.template get_multi_ptr<sycl::access::decorated::no>().get();
    SharedStates* sharedStates =
        reinterpret_cast<SharedStates*>(smem_buf + sharedStatesOffset(topK));

    if (maxLen <= topK) {
      // Every row is the shortcut. The group's rows are contiguous in the
      // output, so the whole block walks them back to back: one run of
      // kRowsPerGroup * topK words instead of that many separate work-groups
      // each writing topK.
      int32_t* out = const_cast<int32_t*>(indices_) +
                     static_cast<int64_t>(firstRow) * topK;
#pragma unroll
      for (int r = 0; r < kRowsPerGroup; ++r) {
        const int rowLen = rowEnd_[firstRow + r] - rowStart_[firstRow + r];
        for (int i = local_id; i < topK; i += kNumThreadsPerBlock) {
          out[i] = i < rowLen ? i : -1;
        }
        out += topK;
      }
      return;
    }

    if (minLen > topK && topK <= kFastMaxTopK) {
      // The widest split whose rows still fit in registers: eight elements per
      // thread over kNumThreadsPerBlock / kRows threads.
      bool done = false;
      if (maxLen <= kFastMaxRegItems * (kNumThreadsPerBlock / 4)) {
        done = topKPerRowGroupReduce<kNumThreadsPerBlock, 4, kRowsPerGroup>(
            item,
            sharedStates,
            logits_,
            rowStart_,
            rowEnd_,
            const_cast<int32_t*>(indices_),
            firstRow,
            stride0_,
            stride1_,
            topK);
      } else if (maxLen <= kFastMaxRegItems * (kNumThreadsPerBlock / 2)) {
        done = topKPerRowGroupReduce<kNumThreadsPerBlock, 2, kRowsPerGroup>(
            item,
            sharedStates,
            logits_,
            rowStart_,
            rowEnd_,
            const_cast<int32_t*>(indices_),
            firstRow,
            stride0_,
            stride1_,
            topK);
      } else if (maxLen <= kFastMaxRegItems * kNumThreadsPerBlock) {
        done = topKPerRowGroupReduce<kNumThreadsPerBlock, 1, kRowsPerGroup>(
            item,
            sharedStates,
            logits_,
            rowStart_,
            rowEnd_,
            const_cast<int32_t*>(indices_),
            firstRow,
            stride0_,
            stride1_,
            topK);
      }
      if (done) {
        return;
      }
      sycl::group_barrier(item.get_group());
    }

    // Rows of mixed kinds, rows too long for registers, or a group that gave
    // up: the whole block takes them one at a time, exactly as the single-row
    // kernel does. Rewriting a row that already succeeded is harmless, the
    // output is the same. The barrier that opens every job orders the previous
    // one's reads of shared memory before this one's writes.
    for (int r = 0; r < kRowsPerGroup; ++r) {
      const int rowIdx = firstRow + r;
      topKPerRowJob<kNumThreadsPerBlock, kNumBins, false, false, false>(
          item,
          slm_,
          /*indices=*/nullptr,
          logits_ + static_cast<int64_t>(rowIdx) * stride0_,
          rowStart_[rowIdx],
          rowEnd_[rowIdx],
          const_cast<int32_t*>(indices_) + static_cast<int64_t>(rowIdx) * topK,
          /*outLogits=*/nullptr,
          stride1_,
          topK);
      sycl::group_barrier(item.get_group());
    }
  }

  sycl::local_accessor<char, 1> slm_;
  const int32_t* indices_;
  const float* logits_;
  const int32_t* rowStart_;
  const int32_t* rowEnd_;
  const int64_t stride0_;
  const int64_t stride1_;
  const int64_t topK_;
  const int64_t offsetIndex_;
};

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

    // The row computed by this block, and with it the schedule. One block is
    // one row and the blocks are dispatched in order, so the plain mapping
    // would run every cheap row first and every expensive one last: in a causal
    // prefill row r holds r + 1 elements. Pairing the shortest row with the
    // longest instead, 0, n-1, 1, n-2, ..., puts a row that is nearly all
    // output write next to a row that is nearly all selection, so the blocks
    // resident together hold a mix of the two. Worth 5 us of 71 on the S=3500
    // triangle at topK=2048, and 6 of those 71 go to the bare read and write,
    // so it buys bus efficiency rather than arithmetic.
    const int64_t numBlocks = item.get_group_range(0);
    const int64_t block = (blockIdx_x & 1) ? numBlocks - 1 - (blockIdx_x >> 1)
                                           : (blockIdx_x >> 1);
    int rowIdx = block + offsetIndex_;

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

// ---------------------------------------------------------------------------
// Split radix select, for decode rows long enough that one work-group per row
// leaves the machine idle.
//
// One work-group per row spends almost all of its time waiting rather than
// reading: a whole 50000-element row can be read and histogrammed in 4 us, yet
// the single-group path takes 38 us. So a row is split over several groups
// here, and they cooperate inside one kernel launch through a device-wide
// barrier - a bounded spin on a global counter, measured at about 0.1 us a
// round.
//
// The select is an exact radix select over the monotone uint32 form of the
// float, in three digits of 11, 11 and 10 bits. Together they cover the whole
// key, so after the last step every element still in play has the same key and
// therefore the same value, and any subset of them completes the answer, so no
// value comparison is needed.
//
// A normal row takes two passes: one histograms step 0's digit, and the next
// emits the elements above its threshold bin and collects the ones inside it
// into a candidate buffer, which one group then finishes in SLM. A row whose
// threshold bin overflows the buffer takes a pass for step 1 and collects from
// there; only a row of near-identical values takes a pass per digit.
//
// Two hazards govern how the shared state is touched, and both cost correctness
// rather than speed when missed. A work-group's L1 is not coherent with another
// group's, so a histogram another group wrote is read with atomic loads and
// cleared with atomic stores - the increments being atomic is not enough, since
// a plain store of zero can linger in L1 and land after another group's
// increment. And a group's leader may only announce arrival at the barrier once
// the whole group's writes are done and released, or another group reads a
// half-built histogram.

// Digit widths of the three radix steps, high bits first: 11 + 11 + 10 = 32.
static constexpr int kSplitBins0 = 2048;
static constexpr int kSplitBins1 = 2048;
static constexpr int kSplitBins2 = 1024;
static constexpr int kSplitShift0 = 21;
static constexpr int kSplitShift1 = 10;
// The three global histograms of a row, laid end to end.
static constexpr int kSplitHistInts = kSplitBins0 + kSplitBins1 + kSplitBins2;
// Barrier rounds: one after the state is cleared, one after step 0's histogram,
// one after the second pass over the row, and one more for the rows that need a
// third. None of them is reached twice.
static constexpr int kSplitRounds = 4;
// How many elements of a threshold bin the candidate buffer holds. A row keeps
// the elements of step 0's threshold bin when they fit, and of step 1's when
// they do not; only a row whose values nearly all share their top 22 bits
// overflows both and needs a pass per digit.
static constexpr int kSplitCandCap = 8192;
// A row shorter than this is left to one group running the established
// single-group job: the split's barrier rounds and tail cost a roughly fixed
// time, which grows with the number of rows, so a row has to be long enough
// before splitting it pays. The logits width says nothing about the row -
// decode rows are as wide as the longest context the model serves, whatever the
// context is - so the kernel checks each row's own length. Measured against
// main, the split first wins at about 24k elements for one or two rows, 32k for
// up to eight and 48k for up to sixteen.
static inline int splitMinRowLen(int64_t numRows) {
  if (numRows <= 2) return 24576;
  if (numRows <= 8) return 32768;
  return 49152;
}
// SLM ints of the split path: eight words of scratch, then either a pass's
// histogram copies or - in the one-group tail, which builds a single-copy
// histogram - that histogram and the keys of the candidate buffer. The two
// never live at once, so they share the words, which keeps the group well
// under 64 KiB.
static constexpr int kSplitSlmInts =
    8 + std::max(kSplitBins0 * kHistCopies, kSplitBins1 + kSplitCandCap);
static_assert(
    kSplitBins2 <= kSplitBins1, "the tail's histograms fit below its keys");
static_assert(kSplitSlmInts * sizeof(int32_t) <= 64 * 1024, "64 KiB per group");
// The fewest elements a group of a split row gets. A row with fewer than
// splitBlocks times this leaves the groups past its length idle, so that a row
// much shorter than the logits width is not spread thinner than it is worth.
static constexpr int kSplitMinSlice = 2048;
// Elements a thread reads per iteration of a pass over the row, issued
// together. One load per iteration leaves the pass waiting on one memory round
// trip at a time and costs about half the bandwidth.
static constexpr int kSplitUnroll = 4;
// Per-row workspace: the histograms, the barrier counters, the three cursors,
// the count of groups that left an abandoned row, and the candidate buffer as
// keys followed by indices.
static constexpr int kSplitCursorOff = kSplitHistInts + kSplitRounds;
static constexpr int kSplitTieOff = kSplitCursorOff + 1;
static constexpr int kSplitCandCursorOff = kSplitTieOff + 1;
static constexpr int kSplitLeftOff = kSplitCandCursorOff + 1;
static constexpr int kSplitCandOff = kSplitLeftOff + 1;
static constexpr int kSplitRowStateInts = kSplitCandOff + 2 * kSplitCandCap;
// Bits of the key above each step's digit.
static constexpr uint32_t kSplitMask1 = 0xFFE00000u;
static constexpr uint32_t kSplitMask2 = 0xFFFFFC00u;
// A spin gives up rather than hanging the device if the groups of a row are
// not co-resident, and the row is then redone by one group (see
// splitRowBarrier()). The host only picks this path for group counts that were
// measured to be co-resident, so this is a backstop, not a normal outcome.
// A million spins is about 0.2 s on this device, some thousand times the
// longest wait a barrier normally sees.
static constexpr long kSplitSpinLimit = 1000 * 1000;
// Marks a barrier counter whose round was abandoned.
static constexpr int32_t kSplitAbandoned = 1 << 30;

using SplitGlobalAtomic = sycl::atomic_ref<
    int32_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>;
using SplitLocalAtomic = sycl::atomic_ref<
    int32_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::work_group,
    sycl::access::address_space::local_space>;

// Floats ordered as unsigned integers: negatives inverted, non-negatives with
// the sign bit set, so a larger float gives a larger key.
static inline uint32_t splitKeyOf(float x) {
  const uint32_t u = sycl::bit_cast<uint32_t>(x);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Words another group filled are read and written atomically: a work-group's L1
// is not coherent, so a plain load may be answered from this group's own copy,
// and a plain store of zero can linger there and land after someone else's
// increment.
static inline int32_t splitLoad(const int32_t* p, int i) {
  return SplitGlobalAtomic(const_cast<int32_t&>(p[i])).load();
}
static inline void splitStore(int32_t* p, int i, int32_t v) {
  SplitGlobalAtomic(p[i]).store(v);
}

// A device-wide barrier over the splitBlocks groups of one row.
//
// Counters come back to zero without any host-side clearing, and without the
// kernel having to take the same rounds every time - which it does not, since
// the slower paths below take rounds the fast path never reaches. Every launch
// clears rounds 1 and up before arriving at round 0, which is safe because no
// group can increment those until it has passed round 0; and round 1's last
// arrival clears round 0, which is safe because by then no group can still be
// spinning on it.
//
// `clearRound` is the counter this barrier's last arrival zeroes, or -1.
//
// Returns false if the row's barrier was abandoned. A spin that gives up marks
// the round's counter - only while the round is still short, so a complete
// round is never taken back - and every group arriving at the round after that
// fails too. No group of the row gets past the round then, and every one of
// them comes back false from it.
static inline bool splitRowBarrier(
    const sycl::nd_item<3>& item,
    int32_t* counters,
    int round,
    int clearRound,
    int splitBlocks,
    int32_t* smemFlag) {
  auto group = item.get_group();
  // Every work-item's writes to the row's state must be visible to the other
  // groups by the time they see this group arrive: a device-scope group
  // barrier orders them before the leader's release, and the leader's acquire
  // before everything the group reads next.
  sycl::group_barrier(group, sycl::memory_scope::device);
  if (item.get_local_id(0) == 0) {
    SplitGlobalAtomic arrived(counters[round]);
    int seen = arrived.fetch_add(1, sycl::memory_order::acq_rel) + 1;
    if (seen == splitBlocks && clearRound >= 0)
      SplitGlobalAtomic(counters[clearRound]).store(0);
    long spins = 0;
    bool gaveUp = false;
    while (seen < splitBlocks) {
      if (++spins < kSplitSpinLimit) {
        seen = arrived.load(sycl::memory_order::acquire);
      } else if (arrived.compare_exchange_strong(
                     seen,
                     seen | kSplitAbandoned,
                     sycl::memory_order::acq_rel)) {
        gaveUp = true;
        break;
      }
    }
    smemFlag[0] = gaveUp || (seen & kSplitAbandoned) != 0;
  }
  sycl::group_barrier(group, sycl::memory_scope::device);
  return smemFlag[0] == 0;
}

// The bin where the running count from the top of the histogram reaches
// `remaining`, the count of elements strictly above that bin, and the bin's own
// count. Every thread takes an equal share of the bins from the top, one scan
// over the group finds the thread that owns the crossing, and that thread walks
// its own few bins. A serial walk of all the bins by one thread costs 7 us and
// was the whole cost of an earlier version of this kernel.
//
// The bin's own count is published here because the thread that found the bin
// has just read it. Having every thread of the row read that one word instead -
// one dependent atomic load of a single address, from all of the row's groups
// at once - costs 10 us on a 50k row and 26 us on a 300k one.
//
// `kFromGlobal` picks between the row's global histogram, whose words another
// group filled, and one that stays in this group's SLM.
//
// Called by the whole group; leaves the bin in smem[0], the count above it in
// smem[1] and the bin's own count in smem[2].
template <int kNumBinsAtStep, int kNumThreadsPerBlock, bool kFromGlobal>
static inline void splitFindThreshold(
    const sycl::nd_item<3>& item,
    const int32_t* hist,
    int remaining,
    int32_t* smem) {
  constexpr int kBinsPerThread = kNumBinsAtStep / kNumThreadsPerBlock;
  auto group = item.get_group();
  const int threadIdx_x = item.get_local_id(0);
  if (threadIdx_x == 0) {
    smem[0] = 0;
    smem[1] = 0;
    smem[2] = 0;
  }
  sycl::group_barrier(group);
  const auto binCount = [&](int bin) {
    if constexpr (kFromGlobal) return splitLoad(hist, bin);
    return hist[bin];
  };
  const int highest = kNumBinsAtStep - kBinsPerThread * threadIdx_x - 1;
  int mine = 0;
  for (int j = 0; j < kBinsPerThread; ++j)
    mine += binCount(highest - j);
  const int prefix =
      sycl::inclusive_scan_over_group(group, mine, sycl::plus<int>());
  if (prefix >= remaining && prefix - mine < remaining) {
    int above = prefix - mine;
    int bin = highest;
    int count = 0;
    for (int j = 0; j < kBinsPerThread; ++j, --bin) {
      count = binCount(bin);
      if (above + count >= remaining) break;
      above += count;
    }
    smem[0] = bin;
    smem[1] = above;
    smem[2] = count;
  }
  sycl::group_barrier(group);
}

// Reserve room for `mine` of this lane's elements on a cursor, taking a single
// atomic for the whole subgroup, and return this lane's first slot - or -1 if
// no lane of the subgroup has anything. A cursor is one word, so an atomic per
// element serialises a thousand of them per row.
//
// Every lane must reach this, including the ones with nothing to write: the
// scans are subgroup collectives. That is why the passes below run a fixed
// iteration count with a predicate rather than exiting the loop per lane.
static inline int
splitReserve(const sycl::sub_group& sg, int32_t* cursor, int mine) {
  const int rank = sycl::exclusive_scan_over_group(sg, mine, sycl::plus<int>());
  const int total = sycl::reduce_over_group(sg, mine, sycl::plus<int>());
  if (total == 0) return -1;
  int base = 0;
  if (sg.get_local_id() == 0)
    base = SplitGlobalAtomic(*cursor).fetch_add(total);
  base = sycl::group_broadcast(sg, base, 0);
  return base + rank;
}

template <int kNumThreadsPerBlock>
class top_k_per_row_decode_split_kernel {
 public:
  top_k_per_row_decode_split_kernel(
      sycl::local_accessor<char, 1>& slm,
      const float* logits,
      const int32_t* seqLens,
      int32_t* outIndices,
      int32_t* workspace,
      int64_t stride0,
      int64_t stride1,
      int64_t topK,
      int64_t next_n,
      int splitBlocks,
      int minRowLen)
      : slm_(slm),
        logits_(logits),
        seqLens_(seqLens),
        outIndices_(outIndices),
        workspace_(workspace),
        stride0_(stride0),
        stride1_(stride1),
        topK_(topK),
        next_n_(next_n),
        splitBlocks_(splitBlocks),
        minRowLen_(minRowLen) {}

  // The compiler's own choice of register file for this kernel makes the
  // single-group job of a short row about 3% slower than in the established
  // kernel; the large one brings it back and speeds up the split a little.
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::grf_size<256>};
  }

  void operator() [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] (
      const sycl::nd_item<3>& item) const {
    // Rows vary fastest in the launch order, so the groups of slice 0 - the
    // only ones busy in a short row - are adjacent, as in the established
    // kernel, rather than one in every splitBlocks.
    const int rowIdx = item.get_group(1);
    const int sliceIdx = item.get_group(0);
    const int threadIdx_x = item.get_local_id(0);
    auto group = item.get_group();
    auto sg = item.get_sub_group();
    const int topK = topK_;

    const int32_t seqLen = seqLens_[rowIdx / next_n_];
    const int rowLen = seqLen - next_n_ + (rowIdx % next_n_) + 1;
    const float* rowBase = logits_ + static_cast<int64_t>(rowIdx) * stride0_;
    int32_t* outIndices = outIndices_ + static_cast<int64_t>(rowIdx) * topK;

    // Eight words of scratch; then a histogram of one step, in interleaved
    // copies so that the threads of a group spread their atomics over separate
    // words. A group that selects in SLM keeps a single copy and puts the keys
    // after it, over the copies it does not use.
    int32_t* smem = reinterpret_cast<int32_t*>(&slm_[0]);
    int32_t* smemHist = smem + 8;
    int32_t* smemKeys = smemHist + kSplitBins1;

    // Rows no longer than topK are the whole answer. One group writes it, so
    // this row needs no barriers at all - and since the barriers are per row,
    // the other rows are unaffected.
    if (rowLen <= topK) {
      if (sliceIdx == 0) {
        for (int i = threadIdx_x; i < rowLen; i += kNumThreadsPerBlock)
          outIndices[i] = i;
        for (int i = rowLen + threadIdx_x; i < topK; i += kNumThreadsPerBlock)
          outIndices[i] = -1;
      }
      return;
    }

    // A short row is one group's work, done by the same job the established
    // path runs, and the row's other groups have nothing to do.
    if (rowLen < minRowLen_) {
      if (sliceIdx == 0)
        topKPerRowJob<kNumThreadsPerBlock, kNumBins, false>(
            item,
            slm_,
            nullptr,
            rowBase,
            0,
            rowLen,
            outIndices,
            nullptr,
            stride1_,
            topK);
      return;
    }

    // Groups past the row's own length leave, and the rest split it evenly.
    const int splitBlocks =
        sycl::min(splitBlocks_, (rowLen + kSplitMinSlice - 1) / kSplitMinSlice);
    if (sliceIdx >= splitBlocks) return;

    int32_t* rowState =
        workspace_ + static_cast<int64_t>(rowIdx) * kSplitRowStateInts;
    int32_t* hist0 = rowState;
    int32_t* hist1 = rowState + kSplitBins0;
    int32_t* hist2 = rowState + kSplitBins0 + kSplitBins1;
    int32_t* barriers = rowState + kSplitHistInts;
    int32_t* cursor = rowState + kSplitCursorOff;
    int32_t* tieCursor = rowState + kSplitTieOff;
    int32_t* candCursor = rowState + kSplitCandCursorOff;
    int32_t* candKeys = rowState + kSplitCandOff;
    int32_t* candIndices = candKeys + kSplitCandCap;

    // A row whose barrier was abandoned is redone from scratch, by the last of
    // its groups to leave and with the established single-group job: by then
    // no other group of the row can still write its output. Every group of
    // such a row leaves through here, so the count reaches splitBlocks; and the
    // last one also readies the counters for the next launch, since no group
    // is left to touch them.
    const auto barrier = [&](int round, int clearRound) {
      if (splitRowBarrier(
              item, barriers, round, clearRound, splitBlocks, smem + 3))
        return true;
      if (threadIdx_x == 0) {
        SplitGlobalAtomic left(rowState[kSplitLeftOff]);
        const bool last =
            left.fetch_add(1, sycl::memory_order::acq_rel) == splitBlocks - 1;
        if (last) {
          left.store(0);
          for (int r = 0; r < kSplitRounds; ++r)
            splitStore(barriers, r, 0);
        }
        smem[4] = last;
      }
      sycl::group_barrier(group, sycl::memory_scope::device);
      const bool last = smem[4] != 0;
      sycl::group_barrier(group);
      if (last)
        topKPerRowJob<kNumThreadsPerBlock, kNumBins, false>(
            item,
            slm_,
            nullptr,
            rowBase,
            0,
            rowLen,
            outIndices,
            nullptr,
            stride1_,
            topK);
      return false;
    };

    const int slice = (rowLen + splitBlocks - 1) / splitBlocks;
    const int begin = sliceIdx * slice;
    const int end = sycl::min(begin + slice, rowLen);
    constexpr int kPerIteration = kNumThreadsPerBlock * kSplitUnroll;
    const int iterations =
        (sycl::max(end - begin, 0) + kPerIteration - 1) / kPerIteration;
    const auto indexOf = [&](int t, int u) {
      return begin + t * kPerIteration + u * kNumThreadsPerBlock + threadIdx_x;
    };
    // Loads are unconditional, clamped into the slice, so that all of an
    // iteration's loads issue together; the caller masks the clamped ones.
    const auto keyAt = [&](int i) {
      return splitKeyOf(rowBase[(i < end ? i : end - 1) * stride1_]);
    };

    // Clear this launch's global state with atomic stores: these same words are
    // incremented atomically below, and a plain store can land afterwards.
    for (int i = threadIdx_x; i < kSplitHistInts; i += kNumThreadsPerBlock)
      if (i % splitBlocks == sliceIdx) splitStore(rowState, i, 0);
    if (threadIdx_x == 0 && sliceIdx == 0) {
      splitStore(cursor, 0, 0);
      splitStore(tieCursor, 0, 0);
      splitStore(candCursor, 0, 0);
      for (int r = 1; r < kSplitRounds; ++r)
        splitStore(barriers, r, 0);
    }
    if (!barrier(0, -1)) return;

    const int copy = threadIdx_x % kHistCopies;

    // Step 0: histogram the slice by the top 11 bits of the key.
    clearSmemHistogram(item, smemHist, kSplitBins0);
    for (int t = 0; t < iterations; ++t) {
      uint32_t key[kSplitUnroll];
#pragma unroll
      for (int u = 0; u < kSplitUnroll; ++u)
        key[u] = keyAt(indexOf(t, u));
#pragma unroll
      for (int u = 0; u < kSplitUnroll; ++u)
        if (indexOf(t, u) < end)
          SplitLocalAtomic(
              smemHist[(key[u] >> kSplitShift0) * kHistCopies + copy])
              .fetch_add(1);
    }
    pushSmemHistogram(item, smemHist, hist0, kSplitBins0);
    if (!barrier(1, 0)) return;
    splitFindThreshold<kSplitBins0, kNumThreadsPerBlock, true>(
        item, hist0, topK, smem);
    const int bin0 = smem[0];
    const int above0 = smem[1];
    const int inBin0 = smem[2];
    const uint32_t prefix1 = static_cast<uint32_t>(bin0) << kSplitShift0;

    // One pass over the slice that emits the elements in play above a step's
    // threshold bin - up to `limit` on the row's cursor - and collects the ones
    // inside it, key and index, into the candidate buffer. Elements in play are
    // those whose bits above the step's digit are `prefix` under `mask`.
    const auto collect =
        [&](uint32_t mask, uint32_t prefix, int shift, int bin, int limit) {
          for (int t = 0; t < iterations; ++t) {
            uint32_t key[kSplitUnroll];
            int digit[kSplitUnroll];
#pragma unroll
            for (int u = 0; u < kSplitUnroll; ++u) {
              key[u] = keyAt(indexOf(t, u));
              digit[u] =
                  indexOf(t, u) < end && (key[u] & mask) == prefix
                      ? static_cast<int>((key[u] >> shift) & (kSplitBins1 - 1))
                      : -1;
            }
            int above = 0;
            int inside = 0;
#pragma unroll
            for (int u = 0; u < kSplitUnroll; ++u) {
              above += digit[u] > bin ? 1 : 0;
              inside += digit[u] == bin ? 1 : 0;
            }
            int slot = splitReserve(sg, cursor, above);
            if (slot >= 0) {
#pragma unroll
              for (int u = 0; u < kSplitUnroll; ++u)
                if (digit[u] > bin) {
                  if (slot < limit) outIndices[slot] = indexOf(t, u);
                  ++slot;
                }
            }
            int kept = splitReserve(sg, candCursor, inside);
            if (kept >= 0) {
#pragma unroll
              for (int u = 0; u < kSplitUnroll; ++u)
                if (digit[u] == bin) {
                  if (kept < kSplitCandCap) {
                    splitStore(candKeys, kept, static_cast<int32_t>(key[u]));
                    splitStore(candIndices, kept, indexOf(t, u));
                  }
                  ++kept;
                }
            }
          }
        };

    if (inBin0 <= kSplitCandCap) {
      // The elements still in play after step 0 are only the ones in its
      // threshold bin, about a thousand for a normal row, so this is the last
      // pass over the row and the last device-wide barrier.
      collect(0u, 0u, kSplitShift0, bin0, above0);
      if (!barrier(2, -1)) return;
      // The rest of the row is one group's work, so the others are done.
      if (sliceIdx != 0) return;
      const int remaining = topK - above0;
      // Digit 1, over the buffer. A thousand candidates do not contend for the
      // SLM histogram the way fifty thousand elements do, so this uses a single
      // copy: nothing to fold, and a quarter of the words to clear.
      for (int i = threadIdx_x; i < inBin0; i += kNumThreadsPerBlock)
        smemKeys[i] = splitLoad(candKeys, i);
      clearSmemBins(item, smemHist, kSplitBins1);
      for (int i = threadIdx_x; i < inBin0; i += kNumThreadsPerBlock) {
        const uint32_t key = static_cast<uint32_t>(smemKeys[i]);
        SplitLocalAtomic(smemHist[(key >> kSplitShift1) & (kSplitBins1 - 1)])
            .fetch_add(1);
      }
      splitFindThreshold<kSplitBins1, kNumThreadsPerBlock, false>(
          item, smemHist, remaining, smem);
      const int bin1 = smem[0];
      const int above1 = smem[1];

      // Emit the candidates above that bin and histogram the last digit of the
      // ones inside it, in one pass. The cursor is in SLM now: this group owns
      // every slot that is left.
      clearSmemBins(item, smemHist, kSplitBins2);
      if (threadIdx_x == 0) smem[5] = above0;
      sycl::group_barrier(group);
      for (int i = threadIdx_x; i < inBin0; i += kNumThreadsPerBlock) {
        const uint32_t key = static_cast<uint32_t>(smemKeys[i]);
        const int digit = (key >> kSplitShift1) & (kSplitBins1 - 1);
        if (digit > bin1) {
          const int slot = SplitLocalAtomic(smem[5]).fetch_add(1);
          if (slot < topK) outIndices[slot] = splitLoad(candIndices, i);
        } else if (digit == bin1) {
          SplitLocalAtomic(smemHist[key & (kSplitBins2 - 1)]).fetch_add(1);
        }
      }
      sycl::group_barrier(group);
      splitFindThreshold<kSplitBins2, kNumThreadsPerBlock, false>(
          item, smemHist, remaining - above1, smem);
      const int bin2 = smem[0];
      const int above2 = smem[1];

      // Everything left at this last bin shares a whole 32-bit key, so it
      // shares a value and any of it completes the answer - but only after the
      // elements strictly above it have their slots. Two cursors, so the
      // equal-key fill cannot take a slot a genuinely larger element owns.
      const int base2 = above0 + above1;
      if (threadIdx_x == 0) {
        smem[5] = base2;
        smem[6] = base2 + above2;
      }
      sycl::group_barrier(group);
      const uint32_t pattern2 =
          (static_cast<uint32_t>(bin0) << 11) | static_cast<uint32_t>(bin1);
      for (int i = threadIdx_x; i < inBin0; i += kNumThreadsPerBlock) {
        const uint32_t key = static_cast<uint32_t>(smemKeys[i]);
        if ((key >> kSplitShift1) != pattern2) continue;
        const int digit = key & (kSplitBins2 - 1);
        int slot = -1;
        if (digit > bin2) {
          slot = SplitLocalAtomic(smem[5]).fetch_add(1);
          if (slot >= base2 + above2) slot = -1;
        } else if (digit == bin2) {
          slot = SplitLocalAtomic(smem[6]).fetch_add(1);
          if (slot >= topK) slot = -1;
        }
        if (slot >= 0) outIndices[slot] = splitLoad(candIndices, i);
      }
      return;
    }

    // Step 0's threshold bin does not fit the buffer - a row whose winners sit
    // in one power of two, as a boosted band of logits does. Step 1 takes a
    // pass of its own: emit what is above step 0's threshold bin and histogram
    // what is inside it by the next 11 bits.
    clearSmemHistogram(item, smemHist, kSplitBins1);
    for (int t = 0; t < iterations; ++t) {
      uint32_t key[kSplitUnroll];
      int digit[kSplitUnroll];
#pragma unroll
      for (int u = 0; u < kSplitUnroll; ++u) {
        key[u] = keyAt(indexOf(t, u));
        digit[u] =
            indexOf(t, u) < end ? static_cast<int>(key[u] >> kSplitShift0) : -1;
      }
      int above = 0;
#pragma unroll
      for (int u = 0; u < kSplitUnroll; ++u)
        above += digit[u] > bin0 ? 1 : 0;
      int slot = splitReserve(sg, cursor, above);
#pragma unroll
      for (int u = 0; u < kSplitUnroll; ++u) {
        if (digit[u] > bin0) {
          if (slot < above0) outIndices[slot] = indexOf(t, u);
          ++slot;
        } else if (digit[u] == bin0) {
          const int next = (key[u] >> kSplitShift1) & (kSplitBins1 - 1);
          SplitLocalAtomic(smemHist[next * kHistCopies + copy]).fetch_add(1);
        }
      }
    }
    pushSmemHistogram(item, smemHist, hist1, kSplitBins1);
    if (!barrier(2, -1)) return;
    splitFindThreshold<kSplitBins1, kNumThreadsPerBlock, true>(
        item, hist1, topK - above0, smem);
    const int bin1 = smem[0];
    const int above1 = smem[1];
    const int inBin1 = smem[2];
    const uint32_t prefix2 =
        prefix1 | (static_cast<uint32_t>(bin1) << kSplitShift1);

    if (inBin1 <= kSplitCandCap) {
      // Step 1's bin is 2048 times narrower, so it fits for anything short of
      // a row of near-identical values; finish as above from there.
      collect(kSplitMask1, prefix1, kSplitShift1, bin1, above0 + above1);
      if (!barrier(3, -1)) return;
      if (sliceIdx != 0) return;
      // The last digit, over the buffer, with the same two-cursor rule.
      for (int i = threadIdx_x; i < inBin1; i += kNumThreadsPerBlock)
        smemKeys[i] = splitLoad(candKeys, i);
      clearSmemBins(item, smemHist, kSplitBins2);
      for (int i = threadIdx_x; i < inBin1; i += kNumThreadsPerBlock) {
        const uint32_t key = static_cast<uint32_t>(smemKeys[i]);
        SplitLocalAtomic(smemHist[key & (kSplitBins2 - 1)]).fetch_add(1);
      }
      splitFindThreshold<kSplitBins2, kNumThreadsPerBlock, false>(
          item, smemHist, topK - above0 - above1, smem);
      const int bin2 = smem[0];
      const int above2 = smem[1];
      const int base2 = above0 + above1;
      if (threadIdx_x == 0) {
        smem[5] = base2;
        smem[6] = base2 + above2;
      }
      sycl::group_barrier(group);
      for (int i = threadIdx_x; i < inBin1; i += kNumThreadsPerBlock) {
        const int digit = smemKeys[i] & (kSplitBins2 - 1);
        int slot = -1;
        if (digit > bin2) {
          slot = SplitLocalAtomic(smem[5]).fetch_add(1);
          if (slot >= base2 + above2) slot = -1;
        } else if (digit == bin2) {
          slot = SplitLocalAtomic(smem[6]).fetch_add(1);
          if (slot >= topK) slot = -1;
        }
        if (slot >= 0) outIndices[slot] = splitLoad(candIndices, i);
      }
      return;
    }

    // More than the buffer holds share their top 22 bits: a pass per digit.
    clearSmemHistogram(item, smemHist, kSplitBins2);
    for (int i = begin + threadIdx_x; i < end; i += kNumThreadsPerBlock) {
      const uint32_t key = splitKeyOf(rowBase[i * stride1_]);
      if ((key & kSplitMask1) != prefix1) continue;
      const int digit = (key >> kSplitShift1) & (kSplitBins1 - 1);
      if (digit > bin1) {
        emit(cursor, outIndices, i, above0 + above1);
      } else if (digit == bin1) {
        const int next = key & (kSplitBins2 - 1);
        SplitLocalAtomic(smemHist[next * kHistCopies + copy]).fetch_add(1);
      }
    }
    pushSmemHistogram(item, smemHist, hist2, kSplitBins2);
    if (!barrier(3, -1)) return;
    splitFindThreshold<kSplitBins2, kNumThreadsPerBlock, true>(
        item, hist2, topK - above0 - above1, smem);
    const int bin2 = smem[0];
    const int above2 = smem[1];

    // Everything left at the last bin shares a whole 32-bit key, so it shares
    // a value and any of it completes the answer - but only after the elements
    // strictly above it have their slots. Two cursors, so the equal-key fill
    // cannot take a slot a genuinely larger element owns.
    const int base2 = above0 + above1 + above2;
    for (int i = begin + threadIdx_x; i < end; i += kNumThreadsPerBlock) {
      const uint32_t key = splitKeyOf(rowBase[i * stride1_]);
      if ((key & kSplitMask2) != prefix2) continue;
      const int digit = key & (kSplitBins2 - 1);
      if (digit > bin2) {
        emit(cursor, outIndices, i, base2);
      } else if (digit == bin2) {
        const int slot = base2 + SplitGlobalAtomic(*tieCursor).fetch_add(1);
        if (slot < topK) outIndices[slot] = i;
      }
    }
  }

 private:
  static inline void
  emit(int32_t* cursor, int32_t* outIndices, int index, int limit) {
    const int slot = SplitGlobalAtomic(*cursor).fetch_add(1);
    if (slot < limit) outIndices[slot] = index;
  }

  static inline void clearSmemHistogram(
      const sycl::nd_item<3>& item, int32_t* smemHist, int numBins) {
    for (int i = item.get_local_id(0); i < numBins * kHistCopies;
         i += kNumThreadsPerBlock)
      smemHist[i] = 0;
    sycl::group_barrier(item.get_group());
  }

  // The single-copy form, for the histograms the tail builds out of the buffer.
  static inline void
  clearSmemBins(const sycl::nd_item<3>& item, int32_t* smemHist, int numBins) {
    for (int i = item.get_local_id(0); i < numBins; i += kNumThreadsPerBlock)
      smemHist[i] = 0;
    sycl::group_barrier(item.get_group());
  }

  // Fold the interleaved copies together and add the non-empty bins into the
  // row's global histogram. Empty bins are skipped, which is most of them.
  static inline void pushSmemHistogram(
      const sycl::nd_item<3>& item,
      const int32_t* smemHist,
      int32_t* hist,
      int numBins) {
    sycl::group_barrier(item.get_group());
    for (int bin = item.get_local_id(0); bin < numBins;
         bin += kNumThreadsPerBlock) {
      int count = 0;
      for (int j = 0; j < kHistCopies; ++j)
        count += smemHist[bin * kHistCopies + j];
      if (count) SplitGlobalAtomic(hist[bin]).fetch_add(count);
    }
  }

  sycl::local_accessor<char, 1> slm_;
  const float* logits_;
  const int32_t* seqLens_;
  int32_t* outIndices_;
  int32_t* workspace_;
  int64_t stride0_;
  int64_t stride1_;
  int64_t topK_;
  int64_t next_n_;
  int splitBlocks_;
  int minRowLen_;
};

// The workspace the split path needs, one buffer per device and stream so that
// concurrent streams cannot share barrier counters. Zeroed once: the kernel
// clears the histograms itself, the candidate buffer is written before it is
// read, and the barrier counters return to zero on their own as each round's
// last arrival clears the round before it - or, in a row whose barrier was
// abandoned, as the last group to leave clears them all.
static int32_t* splitRowWorkspace(const at::Device& device, int64_t numRows) {
  static std::mutex mutex;
  static std::map<std::pair<int, int64_t>, torch::Tensor> buffers;
  const auto stream = c10::xpu::getCurrentXPUStream(device.index());
  std::lock_guard<std::mutex> lock(mutex);
  auto& buffer = buffers[{device.index(), stream.id()}];
  if (!buffer.defined() || buffer.numel() < numRows * kSplitRowStateInts) {
    buffer = torch::zeros(
        {std::max<int64_t>(numRows, 32) * kSplitRowStateInts},
        torch::dtype(torch::kInt32).device(device));
  }
  return buffer.data_ptr<int32_t>();
}

// How many groups to split a decode row over, or 1 for the established paths.
//
// A row has to be long enough that the barrier rounds and the tail cost less
// than the established path's single pass - see splitMinRowLen. Measured best
// counts: about thirty two groups in total for rows of 50k (1, 2, 4 and 8 of
// them); slices of about 16k for longer rows, up to sixty four groups in total
// (1x300k, 4x256k, 8x128k, 16x128k, 8x512k); and never fewer than four a row.
// The width is only an upper bound on the row's real length - the kernel leaves
// idle the groups a shorter row does not need.
static inline int splitBlocksForDecode(int64_t numRows, int64_t numColumns) {
  constexpr int kMinTotalBlocks = 32;
  constexpr int kMaxTotalBlocks = 64;
  constexpr int kLongSliceLen = 16384;
  constexpr int kMinBlocksPerRow = 4;
  // Past this many rows one group per row already fills the machine, and the
  // established path is flat in the row count while this one is not.
  constexpr int kMaxSplitRows = 16;
  if (numRows <= 0 || numRows > kMaxSplitRows ||
      numColumns < splitMinRowLen(numRows))
    return 1;
  const int64_t wanted =
      std::max<int64_t>(kMinTotalBlocks / numRows, numColumns / kLongSliceLen);
  const int64_t blocks = std::max<int64_t>(
      std::min<int64_t>(wanted, kMaxTotalBlocks / numRows), kMinBlocksPerRow);
  return static_cast<int>(blocks);
}

// Medium rows need considerably less scratch than the long-row fallback.
// Keep their histogram and candidate storage local to this launch so resident
// work-groups can hide the latency of histogram updates and output stores.
template <bool kDecode>
struct MediumTopKScratch {
  static constexpr int kThreads = kDecode ? 1024 : 128;
  static constexpr int kBins = kDecode ? 4096 : 2048;
  static constexpr int kStagedMaxTopK = 512;
  // Threshold bin, bin size, accepted count, and two fallback cursors.
  static constexpr int kSelectionVars = 5;
  // Decode uses a finer first histogram to reduce candidate ranking work.
  // The exact fallback uses 11-bit digits within the same storage.
  alignas(16) int bins[kBins];
  int tmp[kReduceScratchSlots<kThreads>];
  int indices[kFastMaxCandidates];
  float values[kFastMaxCandidates];
  int vars[kSelectionVars];
  alignas(16) int output[kDecode ? kStagedMaxTopK : 1];
};

template <bool kDecode, bool kStaged>
struct MediumTopKKernel {
  using Scratch = MediumTopKScratch<kDecode>;
  static constexpr int kThreads = Scratch::kThreads;
  static constexpr int kItems =
      kDecode ? (kMediumRowCapacity + kThreads - 1) / kThreads : 8;
  static_assert(!kDecode || kItems * kThreads >= kMediumRowCapacity);
  const float* logits;
  const int* starts;
  const int* ends;
  int* output;
  int64_t stride0;
  int stride1;
  int topK;
  int nextN;

  void operator() [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] (
      const sycl::nd_item<3>& item) const {
    const int row = item.get_group(0);
    const int tid = item.get_local_id(0);
    const int begin = kDecode ? 0 : starts[row];
    const int end =
        kDecode ? ends[row / nextN] - nextN + row % nextN + 1 : ends[row];
    const int length = end - begin;
    const float* input = logits + row * stride0 + int64_t(begin) * stride1;
    int* out = output + int64_t(row) * topK;
    if (length <= topK) {
      for (int i = tid; i < topK; i += kThreads)
        out[i] = i < length ? i : -1;
      return;
    }
    auto& s = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        MediumTopKScratch<kDecode>>(item.get_group());
    constexpr int kBins = Scratch::kBins;
    // Coalesce the small decode result into one output store per lane.
    int* selected = kStaged ? s.output : out;
    // Host dispatch guarantees this row bound. Excluding the long-row paths
    // here also keeps their register requirements out of the common kernel.
    if (topKPerRowReduce<
            kThreads,
            kThreads,
            kBins,
            1,
            kItems,
            kFastMaxCandidates,
            kDecode,
            kMediumRowCapacity>(
            item,
            input,
            length,
            selected,
            s.bins,
            s.tmp,
            s.indices,
            s.values,
            s.vars,
            s.vars + 1,
            s.vars + 2,
            s.vars + 3,
            stride1,
            topK)) {
      if constexpr (kStaged) {
        sycl::group_barrier(item.get_group());
        for (int i = tid; i < topK; i += kThreads)
          out[i] = s.output[i];
      }
      return;
    }

    // Reuse the general kernel's exact tie/clustered-value fallback. A failed
    // histogram selection has published the threshold bin and accepted count
    // but has not written any indices yet.
    const int threshold = s.vars[0];
    const int accepted = s.vars[2];
    processTieSelect<kThreads, kBins, 11>(
        item,
        input,
        length,
        stride1,
        threshold,
        accepted,
        out,
        s.bins,
        s.tmp,
        s.vars,
        topK);
  }
};

template <bool kDecode, bool kStaged = false>
void launchMediumTopK(
    sycl::queue& queue,
    const float* logits,
    const int* starts,
    const int* ends,
    int* output,
    int rows,
    int64_t stride0,
    int stride1,
    int topK,
    int nextN = 1) {
  constexpr int kThreads = MediumTopKKernel<kDecode, kStaged>::kThreads;
  vllm::xpu::sycl_kernel_submit(
      queue,
      sycl::nd_range<3>(
          sycl::range<3>(size_t(rows) * kThreads, 1, 1),
          sycl::range<3>(kThreads, 1, 1)),
      MediumTopKKernel<kDecode, kStaged>{
          logits, starts, ends, output, stride0, stride1, topK, nextN});
}

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

  // Nothing to select, and an empty launch is an error.
  if (numRows == 0 || topK == 0) return;

  constexpr int kSortingAlgorithmThreshold = 12288;
  constexpr int kSplitWorkThreshold = 200 * 1000;
  constexpr int kNumThreadsPerBlock = 512;
  auto& queue = vllm::xpu::vllmGetQueue();
  const at::DeviceGuard device_guard(logits.device());

  const auto numColumns = logits.size(1);
  if (numColumns > 4096 && numColumns <= vllm::kMediumRowCapacity && topK > 0 &&
      topK <= 2048) {
    if (topK <= vllm::MediumTopKScratch<true>::kStagedMaxTopK) {
      vllm::launchMediumTopK<true, true>(
          queue,
          logits.data_ptr<float>(),
          nullptr,
          seqLens.data_ptr<int>(),
          indices.data_ptr<int>(),
          numRows,
          stride0,
          stride1,
          topK,
          next_n);
    } else {
      vllm::launchMediumTopK<true>(
          queue,
          logits.data_ptr<float>(),
          nullptr,
          seqLens.data_ptr<int>(),
          indices.data_ptr<int>(),
          numRows,
          stride0,
          stride1,
          topK,
          next_n);
    }
    return;
  }

  // Rows long enough that one work-group per row leaves the machine idle are
  // split over several groups that cooperate inside a single launch.
  const int splitBlocks = vllm::splitBlocksForDecode(numRows, numColumns);
  if (splitBlocks > 1) {
    int32_t* workspace = vllm::splitRowWorkspace(logits.device(), numRows);
    sycl::range<3> grid(splitBlocks, numRows, 1);
    sycl::range<3> block(kNumThreadsPerBlock, 1, 1);
    queue.submit([&](sycl::handler& cgh) {
      // The split path's SLM, or the single-group job's for a short row.
      sycl::local_accessor<char, 1> slm(
          sycl::range<1>(std::max(
              vllm::kSplitSlmInts * sizeof(int32_t),
              vllm::sharedMemorySize(topK))),
          cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          vllm::top_k_per_row_decode_split_kernel<kNumThreadsPerBlock>(
              slm,
              logits.data_ptr<float>(),
              seqLens.data_ptr<int>(),
              indices.data_ptr<int>(),
              workspace,
              stride0,
              stride1,
              topK,
              next_n,
              splitBlocks,
              vllm::splitMinRowLen(numRows)));
    });
    return;
  }

  if (numColumns < kSortingAlgorithmThreshold) {
    // Use insertion sort
    sycl::range<3> grid(numRows, 1, 1);
    sycl::range<3> block(kNumThreadsPerBlock, 1, 1);
    size_t dynamic_smem_in_bytes = vllm::sharedMemorySize(topK);
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
    size_t dynamic_smem_in_bytes = vllm::sharedMemorySize(topK);
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
    size_t dynamic_smem_in_bytes_step1 = vllm::sharedMemorySize(topK, true);
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
    size_t dynamic_smem_in_bytes_step2 = vllm::sharedMemorySize(topK);
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

  if (numRows == 0 || topK == 0) return;

  constexpr int kSortingAlgorithmThreshold = 12288;
  constexpr int kNumThreadsPerBlock = 512;
  auto& queue = vllm::xpu::vllmGetQueue();
  const at::DeviceGuard device_guard(logits.device());

  if (numRows > 0 && logits.size(1) > 4096 &&
      logits.size(1) <= vllm::kMediumRowCapacity && topK > 0 && topK <= 2048) {
    vllm::launchMediumTopK<false>(
        queue,
        logits.data_ptr<float>(),
        rowStarts.data_ptr<int>(),
        rowEnds.data_ptr<int>(),
        indices.data_ptr<int>(),
        numRows,
        stride0,
        stride1,
        topK);
    return;
  }

  int numInsertionBlocks =
      std::min(static_cast<int>(numRows), kSortingAlgorithmThreshold);
  int64_t rowOffset = 0;

  size_t dynamic_smem_in_bytes = vllm::sharedMemorySize(topK);

  // Whole groups of kRowsPerGroup rows go to the grouped kernel, the last few
  // rows to the single-row one.
  int numGroups = 0;
  if (numGroups > 0) {
    sycl::range<3> grid_group(numGroups, 1, 1);
    sycl::range<3> block_group(kNumThreadsPerBlock, 1, 1);
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<char, 1> slm(
          sycl::range<1>(dynamic_smem_in_bytes), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(grid_group * block_group, block_group),
          vllm::top_k_per_row_prefill_group_kernel<kNumThreadsPerBlock>(
              slm,
              indices.data_ptr<int32_t>(),
              logits.data_ptr<float>(),
              rowStarts.data_ptr<int32_t>(),
              rowEnds.data_ptr<int32_t>(),
              stride0,
              stride1,
              topK,
              rowOffset));
    });
    const int grouped_rows = numGroups * vllm::kRowsPerGroup;
    rowOffset += grouped_rows;
    numInsertionBlocks -= grouped_rows;
  }

  // For small input sizes, we only launch insertion sort kernels.
  sycl::range<3> grid(std::max(numInsertionBlocks, 1), 1, 1);
  sycl::range<3> block(kNumThreadsPerBlock, 1, 1);
  if (numInsertionBlocks > 0) {
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
              rowOffset));
    });
  }

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
