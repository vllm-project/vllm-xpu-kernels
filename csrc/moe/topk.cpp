#include <sycl/sycl.hpp>

#include <cstdint>
#include <limits>

#include "../utils.h"
#include "../dispatch_utils.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
static constexpr int WARP_SIZE = 32;

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

static inline float sigmoid(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

template <typename T>
static inline float sigmoid_typed(T x) {
  const float val = sigmoid(static_cast<float>(x));
  return static_cast<float>(static_cast<T>(val));
}
// ====================== Softmax things ===============================
// We have our own implementation of softmax here so we can support transposing
// the output in the softmax kernel when we extend this module to support
// expert-choice routing.
template <int TPB, typename InputdType>
class MoeSoftmax {
 public:
  MoeSoftmax(
      sycl::local_accessor<float, 1>& slm,
      const InputdType* input,
      const bool* finished,
      float* output,
      const int num_cols)
      : slm(slm),
        input(input),
        finished(finished),
        output(output),
        num_cols(num_cols) {}

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    void* slm_ptr = static_cast<void*>(
        slm.template get_multi_ptr<sycl::access::decorated::no>().get());

    float* normalizing_factor = reinterpret_cast<float*>(slm_ptr);
    float* float_max = normalizing_factor + 1;

    auto group = item.get_group();
    auto local_id_x = item.get_local_id(0);
    auto group_id_x = item.get_group(0);

    const int thread_row_offset = group_id_x * num_cols;

    float threadData(INFINITY * -1);

    // Don't touch finished rows.
    if ((finished != nullptr) && finished[group_id_x]) {
      return;
    }

    for (int ii = local_id_x; ii < num_cols; ii += TPB) {
      const int idx = thread_row_offset + ii;
      threadData = MAX(static_cast<float>(input[idx]), threadData);
    }

    const float maxElem =
        sycl::reduce_over_group(group, threadData, sycl::maximum<float>());
    if (local_id_x == 0) {
      *float_max = maxElem;
    }
    sycl::group_barrier(item.get_group());

    threadData = 0;

    for (int ii = local_id_x; ii < num_cols; ii += TPB) {
      const int idx = thread_row_offset + ii;
      threadData += sycl::exp((static_cast<float>(input[idx]) - *float_max));
    }

    const auto Z = sycl::reduce_over_group(group, threadData, sycl::plus<>());

    if (local_id_x == 0) {
      *normalizing_factor = 1.f / Z;
    }
    sycl::group_barrier(item.get_group());

    for (int ii = local_id_x; ii < num_cols; ii += TPB) {
      const int idx = thread_row_offset + ii;
      const float val =
          sycl::exp((static_cast<float>(input[idx]) - (*float_max))) *
          (*normalizing_factor);
      output[idx] = val;
    }
  }

 private:
  sycl::local_accessor<float, 1> slm;
  const InputdType* input;
  const bool* finished;
  float* output;
  const int num_cols;
};

template <int TPB, typename InputdType>
class MoeSigmoid {
 public:
  MoeSigmoid(
      const InputdType* input,
      const bool* finished,
      float* output,
      const int num_cols)
      : input(input), finished(finished), output(output), num_cols(num_cols) {}

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    auto local_id_x = item.get_local_id(0);
    auto group_id_x = item.get_group(0);

    const int thread_row_offset = group_id_x * num_cols;

    // Don't touch finished rows.
    if ((finished != nullptr) && finished[group_id_x]) {
      return;
    }

    for (int ii = local_id_x; ii < num_cols; ii += TPB) {
      const int idx = thread_row_offset + ii;
      output[idx] = sigmoid_typed(input[idx]);
    }
  }

 private:
  const InputdType* input;
  const bool* finished;
  float* output;
  const int num_cols;
};

template <int TPB, typename IndType>
class MoeTopK {
 public:
  MoeTopK(
      const float* inputs_after_softmax,
      const bool* finished,
      float* output,
      IndType* indices,
      int* source_rows,
      const int num_experts,
      const int k,
      const int start_expert,
      const int end_expert,
      const bool renormalize,
      const float* bias,
      const double routed_scaling_factor,
      const bool* is_padding)
      : inputs_after_softmax(inputs_after_softmax),
        finished(finished),
        output(output),
        indices(indices),
        source_rows(source_rows),
        num_experts(num_experts),
        k(k),
        start_expert(start_expert),
        end_expert(end_expert),
        renormalize(renormalize),
        bias(bias),
        routed_scaling_factor(routed_scaling_factor),
        is_padding(is_padding) {}

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    int kIdx;
    float kVal;

    auto group = item.get_group();
    auto local_id_x = item.get_local_id(0);
    auto group_id_x = item.get_group(0);

    const int num_rows = item.get_group_range(0);
    const int block_row = group_id_x;

    const bool row_is_active = finished ? !finished[block_row] : true;
    const bool is_pad_row = is_padding != nullptr && is_padding[block_row];
    const int thread_read_offset = group_id_x * num_experts;
    float sum_val = 0.0f;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      kIdx = 0;
      kVal = -1.f;  // This is OK because inputs are probabilities

      int inpIdx;
      float inpVal;
      for (int expert = local_id_x; expert < num_experts; expert += TPB) {
        const int idx = thread_read_offset + expert;
        inpIdx = expert;
        inpVal =
            inputs_after_softmax[idx] + (bias != nullptr ? bias[expert] : 0.0f);

        for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
          const int prior_winning_expert = indices[k * block_row + prior_k];

          if (prior_winning_expert == expert) {
            inpIdx = kIdx;
            inpVal = kVal;
          }
        }

        if (inpVal > kVal) {
          kIdx = inpIdx;
          kVal = inpVal;
        }
      }

      const float resultVal =
          sycl::reduce_over_group(group, kVal, sycl::maximum<float>());
      const int resultIdx = sycl::reduce_over_group(
          group, resultVal == kVal ? kIdx : 0x7FFFFFFF, sycl::minimum<int>());
      sum_val += is_pad_row
                     ? 0.0f
                     : inputs_after_softmax[thread_read_offset + resultIdx];

      if (local_id_x == 0) {
        // Ignore experts the node isn't responsible for with expert parallelism
        const int expert = resultIdx;
        const bool node_uses_expert =
            expert >= start_expert && expert < end_expert;
        const bool should_process_row = row_is_active && node_uses_expert;

        const int idx = k * block_row + k_idx;
        output[idx] = is_pad_row
                          ? 0.0f
                          : inputs_after_softmax[thread_read_offset + expert];
        indices[idx] =
            is_pad_row
                ? static_cast<IndType>(-1)
                : (should_process_row ? (expert - start_expert) : num_experts);
        assert(is_pad_row || indices[idx] >= 0);
        source_rows[idx] = k_idx * num_rows + block_row;
      }
      sycl::group_barrier(item.get_group());
    }

    if (local_id_x == 0) {
      float scale = static_cast<float>(routed_scaling_factor);
      if (renormalize) {
        const float denom = sum_val > 0.0f ? sum_val : 1.0f;
        scale /= denom;
      }
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        const int idx = k * block_row + k_idx;
        output[idx] *= scale;
      }
    }
  }

 private:
  const float* inputs_after_softmax;
  const bool* finished;
  float* output;
  IndType* indices;
  int* source_rows;
  const int num_experts;
  const int k;
  const int start_expert;
  const int end_expert;
  const bool renormalize;
  const float* bias;
  const double routed_scaling_factor;
  const bool* is_padding;
};

// ====================== TopK softmax things ===============================

/*
  A Top-K gating softmax written to exploit when the number of experts in the
  MoE layers are a small power of 2. This allows us to cleanly share the rows
  among the threads in a single warp and eliminate communication between warps
  (so no need to use shared mem).

  It fuses the softmax, max and argmax into a single kernel.

  Limitations:
  1) This implementation is optimized for when the number of experts is a small
  power of 2. Additionally it also supports when number of experts is multiple
  of 64 which is still faster than the computing softmax and topK separately. 2)
  This implementation assumes k is small, but will work for any k.
*/

template <
    int VPT,
    int NUM_EXPERTS,
    int WARPS_PER_CTA,
    int BYTES_PER_LDG,
    int WARP_SIZE_PARAM,
    typename InputdType,
    typename IndType,
    ScoringFunc ScoringFuncParam>
class TopKGating {
 public:
  TopKGating(
      const InputdType* input,
      const bool* finished,
      float* output,
      const int num_rows,
      IndType* indices,
      int* source_rows,
      const int k,
      const int start_expert,
      const int end_expert,
      const bool renormalize,
      const float* bias,
      const double routed_scaling_factor,
      const bool* is_padding)
      : input(input),
        finished(finished),
        output(output),
        num_rows(num_rows),
        indices(indices),
        source_rows(source_rows),
        k(k),
        start_expert(start_expert),
        end_expert(end_expert),
        renormalize(renormalize),
        bias(bias),
        routed_scaling_factor(routed_scaling_factor),
        is_padding(is_padding) {}

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<2> item) const {
    auto sg = item.get_sub_group();
    auto local_id_x = item.get_local_id(1);
    auto local_id_y = item.get_local_id(0);
    auto group_id_x = item.get_group(1);
    // We begin by enforcing compile time assertions and setting up compile time
    // constants.
    static_assert(
        BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
        "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    // Number of bytes each thread pulls in per load
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputdType);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    // Restrictions based on previous section.
    static_assert(
        VPT % ELTS_PER_LDG == 0,
        "The elements per thread must be a multiple of the elements per ldg");
    static_assert(
        WARP_SIZE_PARAM % THREADS_PER_ROW == 0,
        "The threads per row must cleanly divide the threads per warp");
    static_assert(
        THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW),
        "THREADS_PER_ROW must be power of 2");
    static_assert(
        THREADS_PER_ROW <= WARP_SIZE_PARAM,
        "THREADS_PER_ROW can be at most warp size");

    // We have NUM_EXPERTS elements per row. We specialize for small #experts
    static constexpr int ELTS_PER_WARP = WARP_SIZE_PARAM * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    // Restrictions for previous section.
    static_assert(
        ELTS_PER_WARP % ELTS_PER_ROW == 0,
        "The elts per row must cleanly divide the total elt per warp");

    // ===================== From this point, we finally start computing
    // run-time variables. ========================

    // Compute CTA and warp rows. We pack multiple rows into a single warp, and
    // a block contains WARPS_PER_CTA warps. This, each block processes a chunk
    // of rows. We start by computing the start row for each block.
    const int cta_base_row = group_id_x * ROWS_PER_CTA;

    // Now, using the base row per thread block, we compute the base row per
    // warp.
    const int warp_base_row = cta_base_row + local_id_y * ROWS_PER_WARP;

    // The threads in a warp are split into sub-groups that will work on a row.
    // We compute row offset for each thread sub-group
    const int thread_row_in_warp = local_id_x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;

    // Threads with indices out of bounds should early exit here.
    if (thread_row >= num_rows) {
      return;
    }
    const bool row_is_active = finished ? !finished[thread_row] : true;
    const bool is_pad_row = is_padding != nullptr && is_padding[thread_row];

    // We finally start setting up the read pointers for each thread. First,
    // each thread jumps to the start of the row it will read.
    const InputdType* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

    // Now, we compute the group each thread belong to in order to determine the
    // first column to start loads.
    const int thread_group_idx = local_id_x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const InputdType* thread_read_ptr =
        thread_row_ptr + first_elt_read_by_thread;

    // Finally, we pull in the data from global mem
    InputdType row_chunk_load[VPT];
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
#pragma unroll
      for (int jj = 0; jj < ELTS_PER_LDG; ++jj) {
        row_chunk_load[ii * ELTS_PER_LDG + jj] =
            thread_read_ptr[ii * THREADS_PER_ROW * ELTS_PER_LDG + jj];
      }
    }

    float row_chunk[VPT];
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = static_cast<float>(row_chunk_load[ii]);
    }

    // First, we perform a max reduce within the thread. We can do the max in
    // fp16 safely (I think) and just convert to float afterwards for the exp +
    // sum reduction.
    if constexpr (ScoringFuncParam == ScoringFunc::SOFTMAX) {
      float thread_max = row_chunk[0];
#pragma unroll
      for (int ii = 1; ii < VPT; ++ii) {
        thread_max = MAX(thread_max, row_chunk[ii]);
      }

// Now, we find the max within the thread group and distribute among the
// threads. We use a butterfly reduce.
#pragma unroll
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        auto other_thread_max =
            sycl::permute_group_by_xor(sg, thread_max, mask);
        thread_max =
            thread_max > other_thread_max ? thread_max : other_thread_max;
      }

      // From this point, thread max in all the threads have the max within the
      // row. Now, we subtract the max from each element in the thread and take
      // the exp. We also compute the thread local sum.
      float row_sum = 0;
#pragma unroll
      for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = sycl::exp(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
      }

// Now, we perform the sum reduce within each thread group. Similar to the max
// reduce, we use a bufferfly pattern.
#pragma unroll
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        row_sum += sycl::permute_group_by_xor(sg, row_sum, mask);
      }

      // From this point, all threads have the max and the sum for their rows in
      // the thread_max and thread_sum variables respectively. Finally, we can
      // scale the rows for the softmax. Technically, for top-k gating we don't
      // need to compute the entire softmax row. We can likely look at the maxes
      // and only compute for the top-k values in the row. However, this kernel
      // will likely not be a bottle neck and it seems better to closer match
      // torch and find the argmax after computing the softmax.
      const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
      for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
      }
    } else {
#pragma unroll
      for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = sigmoid_typed(static_cast<InputdType>(row_chunk[ii]));
      }
    }

    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    // If bias is not null, use biased value for selection
    float row_chunk_with_bias[VPT];
    // Apply correction bias
    if (bias != nullptr) {
#pragma unroll
      for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
#pragma unroll
        for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
          const int expert =
              first_elt_read_by_thread + ldg * COLS_PER_GROUP_LDG + ii;
          float bias_val = expert < NUM_EXPERTS ? bias[expert] : 0.0f;
          row_chunk_with_bias[ldg * ELTS_PER_LDG + ii] =
              row_chunk[ldg * ELTS_PER_LDG + ii] + bias_val;
        }
      }
    } else {
#pragma unroll
      for (int ii = 0; ii < VPT; ++ii) {
        row_chunk_with_bias[ii] = row_chunk[ii];
      }
    }

    // Now, softmax_res contains the softmax of the row chunk. Now, I want to
    // find the topk elements in each row, along with the max index.
    int start_col = first_elt_read_by_thread;
    float sum_val = 0.0f;

    for (int k_idx = 0; k_idx < k; ++k_idx) {
      // First, each thread does the local argmax
      float max_val_with_bias = row_chunk_with_bias[0];
      float max_val = row_chunk[0];
      int expert_local = start_col;
      int max_val_idx = 0;
#pragma unroll
      for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD;
           ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
        for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
          float val_with_bias = row_chunk_with_bias[ldg * ELTS_PER_LDG + ii];
          float val = row_chunk[ldg * ELTS_PER_LDG + ii];

          // No check on the experts here since columns with the smallest index
          // are processed first and only updated if > (not >=)
          if (val_with_bias > max_val_with_bias) {
            max_val_with_bias = val_with_bias;
            max_val = val;
            expert_local = col + ii;
            max_val_idx = ldg * ELTS_PER_LDG + ii;
          }
        }
      }

      // Now, we perform the argmax reduce. We use the butterfly pattern so
      // threads reach consensus about the max. This will be useful for K > 1 so
      // that the threads can agree on "who" had the max value. That thread can
      // then blank out their max with -inf and the warp can run more
      // iterations...
      int expert = expert_local;
#pragma unroll
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        float other_max_with_bias =
            sycl::permute_group_by_xor(sg, max_val_with_bias, mask);
        float other_max = sycl::permute_group_by_xor(sg, max_val, mask);
        int other_expert = sycl::permute_group_by_xor(sg, expert, mask);

        // We want lower indices to "win" in every thread so we break ties this
        // way
        if (other_max_with_bias > max_val_with_bias ||
            (other_max_with_bias == max_val_with_bias &&
             other_expert < expert)) {
          max_val_with_bias = other_max_with_bias;
          max_val = other_max;
          expert = other_expert;
        }
      }

      sum_val += is_pad_row ? 0.0f : max_val;

      // Write the max for this k iteration to global memory.
      if (thread_group_idx == 0) {
        // Add a guard to ignore experts not included by this node
        const bool node_uses_expert =
            expert >= start_expert && expert < end_expert;
        const bool should_process_row = row_is_active && node_uses_expert;

        // The lead thread from each sub-group will write out the final results
        // to global memory. (This will be a single) thread per row of the
        // input/output matrices.
        const int idx = k * thread_row + k_idx;
        output[idx] = is_pad_row ? 0.0f : max_val;
        indices[idx] =
            is_pad_row
                ? static_cast<IndType>(-1)
                : (should_process_row ? (expert - start_expert) : NUM_EXPERTS);
        source_rows[idx] = k_idx * num_rows + thread_row;
      }

      // Finally, we clear the value in the thread with the current max if there
      // is another iteration to run.
      if (expert == expert_local) {
        row_chunk_with_bias[max_val_idx] = -10000.f;
      }
    }

    if (thread_group_idx == 0) {
      float scale = static_cast<float>(routed_scaling_factor);
      if (renormalize) {
        const float denom = sum_val > 0.0f ? sum_val : 1.0f;
        scale /= denom;
      }
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        const int idx = k * thread_row + k_idx;
        output[idx] *= scale;
      }
    }
  }

 private:
  const InputdType* input;
  const bool* finished;
  float* output;
  const int num_rows;
  IndType* indices;
  int* source_rows;
  const int k;
  const int start_expert;
  const int end_expert;
  const bool renormalize;
  const float* bias;
  const double routed_scaling_factor;
  const bool* is_padding;
};

namespace fast_topk_softmax {

constexpr int kSgSize = MST_SG;
constexpr int kSSgSize = MST_SSG;
constexpr int kWgSize = MST_WG;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

inline float ieee_div(float a, float b) {
  volatile float denom = b;
  return a / denom;
}

template <typename T, int W>
struct alignas(sizeof(T) * W) WideVec {
  T v[W];
};

using F4 = WideVec<float, 4>;

inline F4 ld4(const float* __restrict__ p) {
  return *reinterpret_cast<const F4*>(p);
}

constexpr int pow2_floor(int l) {
  int p = 1;
  while (p * 2 <= l)
    p *= 2;
  return p;
}

constexpr int lanes_for_vpl(int n, int vpl) {
  int l = n / vpl;
  if (l < MST_MIN_LANES) l = MST_MIN_LANES;
  if (l > kSgSize) l = kSgSize;
  if (l > n / 4) l = n / 4;
  if (l < 1) l = 1;
  return pow2_floor(l);
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

constexpr int store_width(int topk) {
  return topk % 4 == 0 ? 4 : (topk % 2 == 0 ? 2 : 1);
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

template <int TOPK, int W>
inline void store_row(
    float* __restrict__ weights,
    int* __restrict__ indices,
    int* __restrict__ source_rows,
    int64_t token,
    int64_t num_tokens,
    const float* __restrict__ num,
    const int* __restrict__ wid,
    float inv,
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
      tw.v[e] = is_pad ? 0.0f : num[k] * inv;
      ti.v[e] = is_pad ? -1 : wid[k];
      so[k] = k * num_tokens + token;
    }
    *reinterpret_cast<WideVec<float, W>*>(wo + b) = tw;
    *reinterpret_cast<WideVec<int, W>*>(io + b) = ti;
  }
}

template <int N, int TOPK, int LANES>
struct MoeSoftmaxTopk {
  static constexpr int kVpl = N / LANES;
  static constexpr int kG = kVpl / 4;
  static constexpr int kGpc = groups_per_chunk(kG);
  static constexpr int kChunks = kG / kGpc;
  static constexpr int kW = store_width(TOPK);

  static_assert(
      kVpl % 4 == 0, "per-lane slice must be a whole number of float4");
  static_assert(kChunks >= 1, "empty chunk cache");

  const float* __restrict__ gating;
  float* __restrict__ weights;
  int* __restrict__ indices;
  int* __restrict__ source_rows;
  const bool* __restrict__ is_padding;
  int64_t num_tokens;

  [[sycl::reqd_sub_group_size(kSgSize)]] void
  operator()(sycl::nd_item<1> it) const {
    const sycl::sub_group sg = it.get_sub_group();
    const int64_t gid = static_cast<int64_t>(it.get_global_linear_id());
    const int lane = LANES == 1 ? 0 : static_cast<int>(gid & (LANES - 1));
    const int64_t token = LANES == 1 ? gid : gid / LANES;
    const bool active = token < num_tokens;
    const float* __restrict__ row = gating + (active ? token : 0) * N;

    float cs[kChunks];
    int ci[kChunks];
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      float bs = kNegInf;
      int bi = N;
#pragma unroll
      for (int g = 0; g < kGpc; ++g) {
        const int i0 = ((c * kGpc + g) * LANES + lane) * 4;
        const F4 x = ld4(row + i0);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float val = x.v[e];
          const bool take = val > bs;
          bs = take ? val : bs;
          bi = take ? i0 + e : bi;
        }
      }
      cs[c] = bs;
      ci[c] = bi;
    }

    float num[TOPK];
    int wid[TOPK];
    float row_max = 0.0f;
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
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
      if (k == 0) row_max = ws;
      num[k] = sycl::native::exp(ws - row_max);
      sum += num[k];
      wid[k] = wi;

      const bool mine = LANES == 1 || (((wi >> 2) & (LANES - 1)) == lane);
      if (mine) {
        float ns = kNegInf;
        int ni = N;
#pragma unroll
        for (int g = 0; g < kGpc; ++g) {
          const int i0 = ((bc * kGpc + g) * LANES + lane) * 4;
          const F4 x = ld4(row + i0);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float val = x.v[e];
            const int idx = i0 + e;
            const bool below = (val < ws) | ((val == ws) & (idx > wi));
            const bool take = below & (val > ns);
            ns = take ? val : ns;
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
      store_row<TOPK, kW>(
          weights,
          indices,
          source_rows,
          token,
          num_tokens,
          num,
          wid,
          ieee_div(1.0f, sum),
          is_pad);
    }
  }
};

template <int N, int TOPK, int LANES>
struct MoeSoftmaxTopkSmall {
  static constexpr int kVpl = N / LANES;
  static constexpr int kG = kVpl / 4;
  static constexpr int kW = store_width(TOPK);

  static_assert(kG >= 1, "small kernel needs at least one float4 per lane");

  const float* __restrict__ gating;
  float* __restrict__ weights;
  int* __restrict__ indices;
  int* __restrict__ source_rows;
  const bool* __restrict__ is_padding;
  int64_t num_tokens;

  [[sycl::reqd_sub_group_size(kSSgSize)]] void
  operator()(sycl::nd_item<1> it) const {
    const sycl::sub_group sg = it.get_sub_group();
    const int64_t gid = static_cast<int64_t>(it.get_global_linear_id());
    const int lane = LANES == 1 ? 0 : static_cast<int>(gid & (LANES - 1));
    const int64_t token = LANES == 1 ? gid : gid / LANES;
    const bool active = token < num_tokens;
    const float* __restrict__ row = gating + (active ? token : 0) * N;

    F4 v[kG];
#pragma unroll
    for (int g = 0; g < kG; ++g)
      v[g] = ld4(row + (g * LANES + lane) * 4);

    float num[TOPK];
    int wid[TOPK];
    float row_max = 0.0f;
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      float acc[4];
#pragma unroll
      for (int e = 0; e < 4; ++e)
        acc[e] = v[0].v[e];
#pragma unroll
      for (int g = 1; g < kG; ++g) {
#pragma unroll
        for (int e = 0; e < 4; ++e)
          acc[e] = sycl::fmax(acc[e], v[g].v[e]);
      }
      acc[0] = sycl::fmax(acc[0], acc[2]);
      acc[1] = sycl::fmax(acc[1], acc[3]);
      float ws = sycl::fmax(acc[0], acc[1]);

      int iacc[4];
#pragma unroll
      for (int e = 0; e < 4; ++e)
        iacc[e] = v[0].v[e] == ws ? lane * 4 + e : N;
#pragma unroll
      for (int g = 1; g < kG; ++g) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int idx = (g * LANES + lane) * 4 + e;
          iacc[e] = v[g].v[e] == ws ? sycl::min(iacc[e], idx) : iacc[e];
        }
      }
      iacc[0] = sycl::min(iacc[0], iacc[2]);
      iacc[1] = sycl::min(iacc[1], iacc[3]);
      int wi = sycl::min(iacc[0], iacc[1]);

      if constexpr (LANES > 1) lane_group_argmax<LANES>(sg, ws, wi);
      if (k == 0) row_max = ws;
      num[k] = sycl::native::exp(ws - row_max);
      sum += num[k];
      wid[k] = wi;

#pragma unroll
      for (int g = 0; g < kG; ++g) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int idx = (g * LANES + lane) * 4 + e;
          v[g].v[e] = idx == wi ? kNegInf : v[g].v[e];
        }
      }
    }

    if (active & (lane == 0)) {
      const bool is_pad = is_padding != nullptr && is_padding[token];
      store_row<TOPK, kW>(
          weights,
          indices,
          source_rows,
          token,
          num_tokens,
          num,
          wid,
          ieee_div(1.0f, sum),
          is_pad);
    }
  }
};

template <int N, int TOPK, int LANES>
void launch_chunked(
    sycl::queue& q,
    const float* gating,
    float* weights,
    int* indices,
    int* source_rows,
    const bool* is_padding,
    int64_t num_tokens) {
  constexpr int kTokensPerWg = kWgSize / LANES;
  const size_t groups =
      static_cast<size_t>((num_tokens + kTokensPerWg - 1) / kTokensPerWg);
  q.parallel_for(
      sycl::nd_range<1>{
          sycl::range<1>{groups * kWgSize}, sycl::range<1>{kWgSize}},
      MoeSoftmaxTopk<N, TOPK, LANES>{
          gating, weights, indices, source_rows, is_padding, num_tokens});
}

template <int N, int TOPK>
void launch_fast(
    sycl::queue& q,
    const float* gating,
    float* weights,
    int* indices,
    int* source_rows,
    const bool* is_padding,
    int64_t num_tokens) {
  const int64_t elems = num_tokens * N;

  if (elems <= MST_SMALL_T) {
    constexpr int kSLanes = small_lanes_for(N);
    constexpr int kSTokensPerWg = kWgSize / kSLanes;
    const size_t groups =
        static_cast<size_t>((num_tokens + kSTokensPerWg - 1) / kSTokensPerWg);
    q.parallel_for(
        sycl::nd_range<1>{
            sycl::range<1>{groups * kWgSize}, sycl::range<1>{kWgSize}},
        MoeSoftmaxTopkSmall<N, TOPK, kSLanes>{
            gating, weights, indices, source_rows, is_padding, num_tokens});
    return;
  }

  if (elems <= MST_MID_T) {
    launch_chunked<N, TOPK, lanes_for_mid(N)>(
        q, gating, weights, indices, source_rows, is_padding, num_tokens);
    return;
  }

  launch_chunked<N, TOPK, lanes_for(N)>(
      q, gating, weights, indices, source_rows, is_padding, num_tokens);
}

template <int N>
bool dispatch_topk(
    sycl::queue& q,
    const float* gating,
    float* weights,
    int* indices,
    int* source_rows,
    const bool* is_padding,
    int64_t num_tokens,
    int topk) {
  switch (topk) {
    case 1:
      launch_fast<N, 1>(
          q, gating, weights, indices, source_rows, is_padding, num_tokens);
      return true;
    case 2:
      launch_fast<N, 2>(
          q, gating, weights, indices, source_rows, is_padding, num_tokens);
      return true;
    case 4:
      launch_fast<N, 4>(
          q, gating, weights, indices, source_rows, is_padding, num_tokens);
      return true;
    case 6:
      launch_fast<N, 6>(
          q, gating, weights, indices, source_rows, is_padding, num_tokens);
      return true;
    case 8:
      launch_fast<N, 8>(
          q, gating, weights, indices, source_rows, is_padding, num_tokens);
      return true;
    default:
      return false;
  }
}

bool dispatch_experts_topk(
    sycl::queue& q,
    const float* gating,
    float* weights,
    int* indices,
    int* source_rows,
    const bool* is_padding,
    int64_t num_tokens,
    int num_experts,
    int topk) {
  const bool aligned = (reinterpret_cast<uintptr_t>(gating) % 16 == 0) &&
                       (reinterpret_cast<uintptr_t>(weights) % 16 == 0) &&
                       (reinterpret_cast<uintptr_t>(indices) % 16 == 0);
  if (!aligned) return false;

  switch (num_experts) {
    case 32:
      return dispatch_topk<32>(
          q,
          gating,
          weights,
          indices,
          source_rows,
          is_padding,
          num_tokens,
          topk);
    case 64:
      return dispatch_topk<64>(
          q,
          gating,
          weights,
          indices,
          source_rows,
          is_padding,
          num_tokens,
          topk);
    case 128:
      return dispatch_topk<128>(
          q,
          gating,
          weights,
          indices,
          source_rows,
          is_padding,
          num_tokens,
          topk);
    case 256:
      return dispatch_topk<256>(
          q,
          gating,
          weights,
          indices,
          source_rows,
          is_padding,
          num_tokens,
          topk);
    default:
      return false;
  }
}

}  // namespace fast_topk_softmax

namespace detail {
// Constructs some constants needed to partition the work across threads at
// compile time.
template <
    int EXPERTS,
    int BYTES_PER_LDG,
    int WARP_SIZE_PARAM,
    typename InputdType>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputdType);
  static_assert(
      EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0 ||
          EXPERTS % (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0,
      "");
  static constexpr int VECs_PER_THREAD =
      MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM));
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static const int ROWS_PER_WARP = WARP_SIZE_PARAM / THREADS_PER_ROW;
};
}  // namespace detail

template <
    int EXPERTS,
    int WARPS_PER_TB,
    int WARP_SIZE_PARAM,
    int MAX_BYTES_PER_LDG,
    typename InputdType,
    typename IndType,
    ScoringFunc ScoringFuncParam>
void topk_gating_launcher_helper(
    const InputdType* input,
    const bool* finished,
    float* output,
    IndType* indices,
    int* source_row,
    const int num_rows,
    const int k,
    const int start_expert,
    const int end_expert,
    bool renormalize,
    float* bias,
    double routed_scaling_factor,
    const bool* is_padding,
    sycl::queue& queue) {
  static constexpr int BYTES_PER_LDG =
      MIN(MAX_BYTES_PER_LDG, sizeof(InputdType) * EXPERTS);
  using Constants = detail::
      TopkConstants<EXPERTS, BYTES_PER_LDG, WARP_SIZE_PARAM, InputdType>;
  static constexpr int VPT = Constants::VPT;
  static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
  const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

  sycl::range<2> grid(1, num_blocks);
  sycl::range<2> block(WARPS_PER_TB, WARP_SIZE_PARAM);
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>(grid * block, block),
        TopKGating<
            VPT,
            EXPERTS,
            WARPS_PER_TB,
            BYTES_PER_LDG,
            WARP_SIZE_PARAM,
            InputdType,
            IndType,
            ScoringFuncParam>(
            input,
            finished,
            output,
            num_rows,
            indices,
            source_row,
            k,
            start_expert,
            end_expert,
            renormalize,
            bias,
            routed_scaling_factor,
            is_padding));
  });
}

#define LAUNCH_TOPK(NUM_EXPERTS, WARPS_PER_TB, MAX_BYTES, SCORING_FUNC)        \
  static_assert(                                                               \
      WARP_SIZE == 32, "Unsupported warp size. Only 32 is supported for XPU"); \
  topk_gating_launcher_helper<                                                 \
      NUM_EXPERTS,                                                             \
      WARPS_PER_TB,                                                            \
      WARP_SIZE,                                                               \
      MAX_BYTES,                                                               \
      InputdType,                                                              \
      IndType,                                                                 \
      SCORING_FUNC>(                                                           \
      gating_output,                                                           \
      nullptr,                                                                 \
      topk_weights,                                                            \
      topk_indices,                                                            \
      token_expert_indices,                                                    \
      num_tokens,                                                              \
      topk,                                                                    \
      0,                                                                       \
      num_experts,                                                             \
      renormalize,                                                             \
      bias,                                                                    \
      routed_scaling_factor,                                                   \
      is_padding,                                                              \
      queue);

template <typename InputdType, typename IndType, ScoringFunc ScoringFuncParam>
void topk_gating_kernel_launcher(
    const InputdType* gating_output,
    float* topk_weights,
    IndType* topk_indices,
    int* token_expert_indices,
    float* scoring_workspace,
    const int num_tokens,
    const int num_experts,
    const int topk,
    const bool renormalize,
    float* bias,
    const double routed_scaling_factor,
    const bool* is_padding,
    sycl::queue& queue) {
  static constexpr int WARPS_PER_TB = 4;
  static constexpr int BYTES_PER_LDG_POWER_OF_2 = 16;
  static constexpr int BYTES_PER_LDG_MULTIPLE_64 = 2 * sizeof(InputdType);

  switch (num_experts) {
    case 1:
      LAUNCH_TOPK(1, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2, ScoringFuncParam);
      break;
    case 2:
      LAUNCH_TOPK(2, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2, ScoringFuncParam);
      break;
    case 4:
      LAUNCH_TOPK(4, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2, ScoringFuncParam);
      break;
    case 8:
      LAUNCH_TOPK(8, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2, ScoringFuncParam);
      break;
    case 16:
      LAUNCH_TOPK(16, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2, ScoringFuncParam);
      break;
    case 32:
      LAUNCH_TOPK(32, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2, ScoringFuncParam);
      break;
    case 64:
      LAUNCH_TOPK(64, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2, ScoringFuncParam);
      break;
    case 128:
      LAUNCH_TOPK(
          128, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2, ScoringFuncParam);
      break;
    case 256:
      LAUNCH_TOPK(
          256, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2, ScoringFuncParam);
      break;
    case 512:
      LAUNCH_TOPK(
          512, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2, ScoringFuncParam);
      break;
    case 192:
      LAUNCH_TOPK(
          192, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64, ScoringFuncParam);
      break;
    case 320:
      LAUNCH_TOPK(
          320, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64, ScoringFuncParam);
      break;
    case 384:
      LAUNCH_TOPK(
          384, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64, ScoringFuncParam);
      break;
    case 448:
      LAUNCH_TOPK(
          448, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64, ScoringFuncParam);
      break;
    case 576:
      LAUNCH_TOPK(
          576, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64, ScoringFuncParam);
      break;
    default: {
      TORCH_CHECK(
          scoring_workspace != nullptr,
          "scoring_workspace must be provided for num_experts that are "
          "not a power of 2 or multiple of 64.");
      static constexpr int TPB = 256;
      sycl::range<1> grid1(num_tokens);
      sycl::range<1> block1(TPB);
      if constexpr (ScoringFuncParam == ScoringFunc::SOFTMAX) {
        queue.submit([&](sycl::handler& cgh) {
          sycl::local_accessor<float, 1> slm(sycl::range<1>(2), cgh);
          cgh.parallel_for(
              sycl::nd_range<1>(grid1 * block1, block1),
              MoeSoftmax<TPB, InputdType>(
                  slm, gating_output, nullptr, scoring_workspace, num_experts));
        });
      } else {
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(grid1 * block1, block1),
              MoeSigmoid<TPB, InputdType>(
                  gating_output, nullptr, scoring_workspace, num_experts));
        });
      }

      sycl::range<1> grid2(num_tokens);
      sycl::range<1> block2(TPB);
      queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(grid2 * block2, block2),
            MoeTopK<TPB, IndType>(
                scoring_workspace,
                nullptr,
                topk_weights,
                topk_indices,
                token_expert_indices,
                num_experts,
                topk,
                0,
                num_experts,
                renormalize,
                bias,
                routed_scaling_factor,
                is_padding));
      });
    }
  }
}

#undef LAUNCH_TOPK

}  // namespace moe
}  // namespace vllm

#define LAUNCH_TOPK(INPUTDTYPE, INDTYPE, SCORING_FUNC)                       \
  vllm::moe::topk_gating_kernel_launcher<INPUTDTYPE, INDTYPE, SCORING_FUNC>( \
      reinterpret_cast<INPUTDTYPE*>(gating_output.mutable_data_ptr()),       \
      topk_weights.data_ptr<float>(),                                        \
      topk_indices.data_ptr<INDTYPE>(),                                      \
      token_expert_indices.data_ptr<int>(),                                  \
      scoring_workspace.data_ptr<float>(),                                   \
      num_tokens,                                                            \
      num_experts,                                                           \
      topk,                                                                  \
      renormalize,                                                           \
      bias.has_value() ? bias->data_ptr<float>() : nullptr,                  \
      routed_scaling_factor,                                                 \
      is_padding.has_value() ? is_padding->data_ptr<bool>() : nullptr,       \
      queue);

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

  const at::DeviceGuard device_guard(gating_output.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  const bool can_use_fast_softmax =
      renormalize && !bias.has_value() && topk < num_experts &&
      gating_output.scalar_type() == at::ScalarType::Float &&
      topk_indices.scalar_type() == at::ScalarType::Int;
  if (can_use_fast_softmax &&
      vllm::moe::fast_topk_softmax::dispatch_experts_topk(
          queue,
          gating_output.data_ptr<float>(),
          topk_weights.data_ptr<float>(),
          topk_indices.data_ptr<int>(),
          token_expert_indices.data_ptr<int>(),
          is_padding.has_value() ? is_padding->data_ptr<bool>() : nullptr,
          num_tokens,
          num_experts,
          topk)) {
    return;
  }

  const bool is_pow_2 =
      (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
  const bool needs_workspace = !is_pow_2 || num_experts > 256;
  const int64_t workspace_size = needs_workspace ? num_tokens * num_experts : 0;

  torch::Tensor scoring_workspace = torch::empty(
      {workspace_size}, gating_output.options().dtype(torch::kFloat));

  if (topk_indices.scalar_type() == at::ScalarType::Int) {
    if (gating_output.scalar_type() == at::ScalarType::Float)
      LAUNCH_TOPK(float, int, vllm::moe::ScoringFunc::SOFTMAX)
    else if (gating_output.scalar_type() == at::ScalarType::Half)
      LAUNCH_TOPK(sycl::half, int, vllm::moe::ScoringFunc::SOFTMAX)
    else
      LAUNCH_TOPK(
          sycl::ext::oneapi::bfloat16, int, vllm::moe::ScoringFunc::SOFTMAX)
  } else if (topk_indices.scalar_type() == at::ScalarType::UInt32) {
    if (gating_output.scalar_type() == at::ScalarType::Float)
      LAUNCH_TOPK(float, uint32_t, vllm::moe::ScoringFunc::SOFTMAX)
    else if (gating_output.scalar_type() == at::ScalarType::Half)
      LAUNCH_TOPK(sycl::half, uint32_t, vllm::moe::ScoringFunc::SOFTMAX)
    else
      LAUNCH_TOPK(
          sycl::ext::oneapi::bfloat16,
          uint32_t,
          vllm::moe::ScoringFunc::SOFTMAX)
  } else {
    TORCH_CHECK(topk_indices.scalar_type() == at::ScalarType::Long);
    if (gating_output.scalar_type() == at::ScalarType::Float)
      LAUNCH_TOPK(float, int64_t, vllm::moe::ScoringFunc::SOFTMAX)
    else if (gating_output.scalar_type() == at::ScalarType::Half)
      LAUNCH_TOPK(sycl::half, int64_t, vllm::moe::ScoringFunc::SOFTMAX)
    else
      LAUNCH_TOPK(
          sycl::ext::oneapi::bfloat16, int64_t, vllm::moe::ScoringFunc::SOFTMAX)
  }
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

  const bool is_pow_2 =
      (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
  const bool needs_workspace = !is_pow_2 || num_experts > 256;
  const int64_t workspace_size = needs_workspace ? num_tokens * num_experts : 0;

  const at::DeviceGuard device_guard(gating_output.device());
  auto& queue = vllm::xpu::vllmGetQueue();
  torch::Tensor scoring_workspace = torch::empty(
      {workspace_size}, gating_output.options().dtype(torch::kFloat));

  if (topk_indices.scalar_type() == at::ScalarType::Int) {
    if (gating_output.scalar_type() == at::ScalarType::Float)
      LAUNCH_TOPK(float, int, vllm::moe::ScoringFunc::SIGMOID)
    else if (gating_output.scalar_type() == at::ScalarType::Half)
      LAUNCH_TOPK(sycl::half, int, vllm::moe::ScoringFunc::SIGMOID)
    else
      LAUNCH_TOPK(
          sycl::ext::oneapi::bfloat16, int, vllm::moe::ScoringFunc::SIGMOID)
  } else if (topk_indices.scalar_type() == at::ScalarType::UInt32) {
    if (gating_output.scalar_type() == at::ScalarType::Float)
      LAUNCH_TOPK(float, uint32_t, vllm::moe::ScoringFunc::SIGMOID)
    else if (gating_output.scalar_type() == at::ScalarType::Half)
      LAUNCH_TOPK(sycl::half, uint32_t, vllm::moe::ScoringFunc::SIGMOID)
    else
      LAUNCH_TOPK(
          sycl::ext::oneapi::bfloat16,
          uint32_t,
          vllm::moe::ScoringFunc::SIGMOID)
  } else {
    TORCH_CHECK(topk_indices.scalar_type() == at::ScalarType::Long);
    if (gating_output.scalar_type() == at::ScalarType::Float)
      LAUNCH_TOPK(float, int64_t, vllm::moe::ScoringFunc::SIGMOID)
    else if (gating_output.scalar_type() == at::ScalarType::Half)
      LAUNCH_TOPK(sycl::half, int64_t, vllm::moe::ScoringFunc::SIGMOID)
    else
      LAUNCH_TOPK(
          sycl::ext::oneapi::bfloat16, int64_t, vllm::moe::ScoringFunc::SIGMOID)
  }

#undef LAUNCH_TOPK
}
