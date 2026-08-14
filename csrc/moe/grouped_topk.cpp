#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "../utils.h"
#include "../dispatch_utils.h"

namespace vllm {
namespace moe {

constexpr float kNegInfinity = INFINITY * -1;
constexpr int32_t WARP_SIZE = 32;

// Scoring function enums (mirror CUDA version)
enum ScoringFunc {
  SCORING_NONE = 0,    // no activation function
  SCORING_SIGMOID = 1  // apply sigmoid
};

namespace warp_topk {

template <int size, typename T>
constexpr T round_up_to_multiple_of(T len) {
  if (len == 0) {
    return 0;
  }
  return ((len - 1) / size + 1) * size;
}

template <typename T>
constexpr bool isPowerOf2(T v) {
  return (v && !(v & (v - 1)));
}

template <bool greater, typename T>
[[intel::device_indirectly_callable]] inline __attribute__((always_inline)) bool
is_better_than(T val, T baseline) {
  return (val > baseline && greater) || (val < baseline && !greater);
}

template <bool greater, typename T, typename idxT>
[[intel::device_indirectly_callable]] inline __attribute__((always_inline)) bool
is_better_than(T val, T baseline, idxT index, idxT baseline_index) {
  bool res = (val > baseline && greater) || (val < baseline && !greater);
  if (val == baseline) {
    res = (index < baseline_index && greater) ||
          (index < baseline_index && !greater);
  }
  return res;
}

template <
    int size,
    bool ascending,
    bool reverse,
    typename T,
    typename idxT,
    bool is_stable>
struct BitonicMerge {
  // input should be a bitonic sequence, and sort it to be a monotonic sequence
  [[intel::device_indirectly_callable]] static void merge(
      T* __restrict__ val_arr,
      idxT* __restrict__ idx_arr,
      sycl::sub_group const& sg,
      const int local_id) {
    static_assert(isPowerOf2(size));
    static_assert(size >= 2 * WARP_SIZE);
    constexpr int arr_len = size / WARP_SIZE;

    constexpr int stride = arr_len / 2;
    for (int i = 0; i < stride; ++i) {
      int const other_i = i + stride;
      T& val = val_arr[i];
      T& other_val = val_arr[other_i];
      bool is_better;
      if constexpr (is_stable) {
        is_better = is_better_than<ascending>(
            val, other_val, idx_arr[i], idx_arr[other_i]);
      } else {
        is_better = is_better_than<ascending>(val, other_val);
      }

      if (is_better) {
        T tmp = val;
        val = other_val;
        other_val = tmp;

        idxT tmp2 = idx_arr[i];
        idx_arr[i] = idx_arr[other_i];
        idx_arr[other_i] = tmp2;
      }
    }

    BitonicMerge<size / 2, ascending, reverse, T, idxT, is_stable>::merge(
        val_arr, idx_arr, sg, local_id);
    BitonicMerge<size / 2, ascending, reverse, T, idxT, is_stable>::merge(
        val_arr + arr_len / 2, idx_arr + arr_len / 2, sg, local_id);
  }
};

template <int size, bool ascending, typename T, typename idxT, bool is_stable>
struct BitonicSort {
  [[intel::device_indirectly_callable]] static void sort(
      T* __restrict__ val_arr,
      idxT* __restrict__ idx_arr,
      sycl::sub_group const& sg,
      const int local_id) {
    static_assert(isPowerOf2(size));
    static_assert(size >= 2 * WARP_SIZE);
    constexpr int arr_len = size / WARP_SIZE;

    BitonicSort<size / 2, true, T, idxT, is_stable>::sort(
        val_arr, idx_arr, sg, local_id);
    BitonicSort<size / 2, false, T, idxT, is_stable>::sort(
        val_arr + arr_len / 2, idx_arr + arr_len / 2, sg, local_id);
    BitonicMerge<size, ascending, ascending, T, idxT, is_stable>::merge(
        val_arr, idx_arr, sg, local_id);
  }
};

template <bool ascending, typename T, typename idxT, bool is_stable>
struct BitonicSort<32, ascending, T, idxT, is_stable> {
  [[intel::device_indirectly_callable]] static void sort(
      T* __restrict__ val_arr,
      idxT* __restrict__ idx_arr,
      sycl::sub_group const& sg,
      const int local_id) {
    int const lane = local_id % WARP_SIZE;

    // ascending doesn't matter before merging since all we need is a bitonic
    // sequence
    for (int stage = 0; stage < 4; ++stage) {
      for (int stride = (1 << stage); stride > 0; stride /= 2) {
        bool reverse = (lane >> stage) & 2;
        bool is_second = lane & stride;

        T other = sycl::permute_group_by_xor(sg, *val_arr, stride);
        idxT other_idx = sycl::permute_group_by_xor(sg, *idx_arr, stride);

        bool is_better;
        if constexpr (is_stable) {
          if constexpr (ascending) {
            is_better = ((*val_arr > other) ||
                         ((*val_arr == other) && (*idx_arr < other_idx))) !=
                        (reverse != is_second);
          } else {
            is_better = ((*val_arr > other) ||
                         ((*val_arr == other) && (*idx_arr > other_idx))) !=
                        (reverse != is_second);
          }
        } else {
          is_better =
              (*val_arr != other &&
               (*val_arr > other) != (reverse != is_second));
        }
        if (is_better) {
          *val_arr = other;
          *idx_arr = other_idx;
        }
      }
    }

    BitonicMerge<32, ascending, ascending, T, idxT, is_stable>::merge(
        val_arr, idx_arr, sg, local_id);
  }
};

template <
    bool ascending,
    bool reverse,
    typename T,
    typename idxT,
    bool is_stable>
struct BitonicMerge<32, ascending, reverse, T, idxT, is_stable> {
  [[intel::device_indirectly_callable]] static void merge(
      T* __restrict__ val_arr,
      idxT* __restrict__ idx_arr,
      sycl::sub_group const& sg,
      const int local_id) {
    int const lane = local_id % WARP_SIZE;
    for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
      bool is_second = lane & stride;
      T& val = *val_arr;
      T other = sycl::permute_group_by_xor(sg, val, stride);
      idxT& idx = *idx_arr;
      idxT other_idx = sycl::permute_group_by_xor(sg, idx, stride);

      bool is_better;
      if constexpr (is_stable) {
        if constexpr (ascending) {
          is_better = ((*val_arr > other) ||
                       ((*val_arr == other) && (*idx_arr < other_idx))) ==
                      (reverse != is_second);  // for min
        } else {
          is_better = ((*val_arr > other) ||
                       ((*val_arr == other) && (*idx_arr > other_idx))) ==
                      (reverse != is_second);  // for max
        }
      } else {
        is_better =
            (val != other && ((val > other) == (ascending != is_second)));
      }

      if (is_better) {
        val = other;
        idx = other_idx;
      }
    }
  }
};

template <int capacity, bool greater, typename T, typename idxT, bool is_stable>
class WarpSort {
 public:
  [[intel::device_indirectly_callable]] WarpSort(
      idxT k, T dummy, const int local_id)
      : lane_(local_id % WARP_SIZE), k_(k), dummy_(dummy) {
    static_assert(capacity >= WARP_SIZE && isPowerOf2(capacity));

    for (int i = 0; i < max_arr_len_; ++i) {
      val_arr_[i] = dummy_;
      idx_arr_[i] = 0;
    }
  }

  [[intel::device_indirectly_callable]] void
  dump(T* __restrict__ out, idxT* __restrict__ out_idx) const {
    for (int i = 0; i < max_arr_len_; ++i) {
      idxT out_i = i * WARP_SIZE + lane_;
      if (out_i < k_) {
        out[out_i] = val_arr_[i];
        out_idx[out_i] = idx_arr_[i];
      }
    }
  }

  [[intel::device_indirectly_callable]] void
  dumpIdx(idxT* __restrict__ out_idx) const {
    for (int i = 0; i < max_arr_len_; ++i) {
      idxT out_i = i * WARP_SIZE + lane_;
      if (out_i < k_) {
        out_idx[out_i] = idx_arr_[i];
      }
    }
  }

  // Accessors for per-lane selected value/index (mirrors CUDA WarpSort).
  [[intel::device_indirectly_callable]] inline idxT get_idx(int i = 0) const {
    return idx_arr_[i];
  }

  [[intel::device_indirectly_callable]] inline T get_val(int i = 0) const {
    return val_arr_[i];
  }

 protected:
  static constexpr int max_arr_len_ = capacity / WARP_SIZE;

  T val_arr_[max_arr_len_];
  idxT idx_arr_[max_arr_len_];

  int const lane_;
  idxT const k_;
  T const dummy_;

};  // end class WarpSort

template <int capacity, bool greater, typename T, typename idxT, bool is_stable>
class WarpSelect : public WarpSort<capacity, greater, T, idxT, is_stable> {
 public:
  [[intel::device_indirectly_callable]] WarpSelect(
      idxT k,
      T dummy,
      char* smem_buf,  // slm_buf
      const int local_id,
      const int local_range)
      : WarpSort<capacity, greater, T, idxT, is_stable>(k, dummy, local_id),
        k_th_(dummy),
        k_th_lane_((k - 1) % WARP_SIZE) {
    int const num_of_warp = local_range / WARP_SIZE;
    int const warp_id = local_id / WARP_SIZE;
    val_smem_ = reinterpret_cast<T*>(smem_buf);
    val_smem_ += warp_id * WARP_SIZE;
    idx_smem_ = reinterpret_cast<idxT*>(
        smem_buf +
        round_up_to_multiple_of<256>(num_of_warp * sizeof(T) * WARP_SIZE));
    idx_smem_ += warp_id * WARP_SIZE;
  }

  [[intel::device_indirectly_callable]] void
  add(T const* in,
      idxT start,
      idxT end,
      sycl::sub_group const& sg,
      const int local_id) {
    idxT const end_for_fullwarp =
        round_up_to_multiple_of<WARP_SIZE>(end - start) + start;
    for (idxT i = start + lane_; i < end_for_fullwarp; i += WARP_SIZE) {
      T val = (i < end) ? in[i] : dummy_;
      add(val, i, sg, local_id);
    }
  }

  [[intel::device_indirectly_callable]] void
  add(T val, idxT idx, sycl::sub_group const& sg, const int local_id) {
    bool do_add;
    if constexpr (is_stable) {
      do_add = is_better_than<greater>(val, k_th_, idx, k_th_idx_);
    } else {
      do_add = is_better_than<greater>(val, k_th_);
    }

    auto mask = sycl::ext::oneapi::group_ballot(sg, do_add);
    if (mask == 0) {
      return;
    }

    int pos = smem_buf_len_ + (mask & ((0x1u << lane_) - 1)).count();
    if (do_add && pos < WARP_SIZE) {
      val_smem_[pos] = val;
      idx_smem_[pos] = idx;
      do_add = false;
    }
    smem_buf_len_ += mask.count();
    if (smem_buf_len_ >= WARP_SIZE) {
      merge_buf_(val_smem_[lane_], idx_smem_[lane_], sg, local_id);
      smem_buf_len_ -= WARP_SIZE;
    }
    if (do_add) {
      pos -= WARP_SIZE;
      val_smem_[pos] = val;
      idx_smem_[pos] = idx;
    }
  }

  [[intel::device_indirectly_callable]] void
  done(sycl::sub_group const& sg, const int local_id) {
    if (smem_buf_len_) {
      T val = (lane_ < smem_buf_len_) ? val_smem_[lane_] : dummy_;
      idxT idx = (lane_ < smem_buf_len_) ? idx_smem_[lane_] : 0;
      merge_buf_(val, idx, sg, local_id);
    }
  }

 private:
  [[intel::device_indirectly_callable]] void
  set_k_th_(sycl::sub_group const& sg) {
    k_th_ = sycl::select_from_group(sg, val_arr_[max_arr_len_ - 1], k_th_lane_);
    if constexpr (is_stable) {
      k_th_idx_ =
          sycl::select_from_group(sg, idx_arr_[max_arr_len_ - 1], k_th_lane_);
    }
  }

  [[intel::device_indirectly_callable]] void
  merge_buf_(T val, idxT idx, sycl::sub_group const& sg, const int local_id) {
    BitonicSort<WARP_SIZE, greater, T, idxT, is_stable>::sort(
        &val, &idx, sg, local_id);

    T& old = val_arr_[max_arr_len_ - 1];

    bool is_better;
    if constexpr (is_stable) {
      is_better =
          is_better_than<greater>(val, old, idx, idx_arr_[max_arr_len_ - 1]);
    } else {
      is_better = is_better_than<greater>(val, old);
    }

    if (is_better) {
      old = val;
      idx_arr_[max_arr_len_ - 1] = idx;
    }

    BitonicMerge<capacity, greater, !greater, T, idxT, is_stable>::merge(
        val_arr_, idx_arr_, sg, local_id);

    set_k_th_(sg);
  }

  using WarpSort<capacity, greater, T, idxT, is_stable>::max_arr_len_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::val_arr_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::idx_arr_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::lane_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::k_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::dummy_;

  T* val_smem_;
  idxT* idx_smem_;
  int smem_buf_len_ = 0;

  T k_th_;
  idxT k_th_idx_;
  int const k_th_lane_;
};  // end class WarpSelect
}  // namespace warp_topk

template <typename T_OUT, typename T_IN>
[[intel::device_indirectly_callable]] inline T_OUT sycl_cast(T_IN val) {
  return val;
}

template <>
[[intel::device_indirectly_callable]] inline float
sycl_cast<float, sycl::ext::oneapi::bfloat16>(sycl::ext::oneapi::bfloat16 val) {
  return static_cast<float>(val);
}

[[intel::device_indirectly_callable]] inline float sigmoid_accurate(float x) {
  return 0.5f * sycl::tanh(0.5f * x) + 0.5f;
}

template <typename T>
[[intel::device_indirectly_callable]] inline T apply_sigmoid(T val) {
  float f = sycl_cast<float, T>(val);
  return sycl_cast<T, float>(sigmoid_accurate(f));
}

template <ScoringFunc SF, typename T>
[[intel::device_indirectly_callable]] inline T apply_scoring(T val) {
  if constexpr (SF == SCORING_NONE) {
    return val;
  } else {
    return apply_sigmoid(val);
  }
}

// -------------------------------------------------------------------------
// topk_with_k2: compute sum of top-2 biased scores for one expert group.
// Used as phase-1 in the fused kernel (one sub_group per expert group).
// -------------------------------------------------------------------------
template <typename T, typename BiasT, ScoringFunc SF>
[[intel::device_indirectly_callable]] inline void topk_with_k2_biased(
    T* output,
    T const* input,
    BiasT const* bias,
    sycl::sub_group const& sg,
    int32_t const lane_id,
    int const num_experts_per_group) {
  float largest = -INFINITY;
  float second_largest = -INFINITY;

  if (num_experts_per_group > WARP_SIZE) {
    for (int i = lane_id; i < num_experts_per_group; i += WARP_SIZE) {
      float value = sycl_cast<float, T>(apply_scoring<SF>(input[i]));
      value += sycl_cast<float, BiasT>(bias[i]);
      if (value > largest) {
        second_largest = largest;
        largest = value;
      } else if (value > second_largest) {
        second_largest = value;
      }
    }
  } else {
    // Each lane holds exactly one value (num_experts_per_group <= WARP_SIZE).
    for (int i = lane_id; i < num_experts_per_group; i += WARP_SIZE) {
      float value = sycl_cast<float, T>(apply_scoring<SF>(input[i]));
      value += sycl_cast<float, BiasT>(bias[i]);
      largest = value;
    }
  }

  float max1 = sycl::reduce_over_group(sg, largest, sycl::maximum<float>());

  float max2 = max1;
  bool equal_to_max1 = (max1 == largest);
  int count_max1 =
      sycl::reduce_over_group(sg, equal_to_max1 ? 1 : 0, sycl::plus<>());

  if (count_max1 == 1) {
    largest = (largest == max1) ? second_largest : largest;
    max2 = sycl::reduce_over_group(sg, largest, sycl::maximum<float>());
  }

  if (lane_id == 0) {
    *output = sycl_cast<T, float>(max1 + max2);
  }
}

// -------------------------------------------------------------------------
// Fused grouped_topk kernel (mirrors CUDA grouped_topk_fused_kernel).
//
// Launch parameters:
//   - one work-group per token
//   - work-group size = n_group * WARP_SIZE  (one sub_group per expert group)
//
// Shared-local memory layout (same as CUDA dynamic smem layout):
//   [0 .. val_bytes_aligned-1]          : WarpSelect val staging (T)
//   [val_bytes_aligned .. internal-1]   : WarpSelect idx staging (int32_t)
//   [internal .. internal+align-1]      : padding to 16-byte boundary
//   [internal+align .. end]             : s_group_scores[n_group] (T)
// -------------------------------------------------------------------------
template <typename T, typename BiasT, typename IdxT, ScoringFunc SF>
class grouped_topk_fused_kernel_impl {
 public:
  grouped_topk_fused_kernel_impl(
      sycl::local_accessor<char, 1>& slm,
      T* scores,
      float* topk_values,
      IdxT* topk_indices,
      BiasT const* bias,
      int64_t const num_tokens,
      int64_t const num_experts,
      int64_t const n_group,
      int64_t const topk_group,
      int64_t const topk,
      bool renormalize,
      double routed_scaling_factor)
      : slm(slm),
        scores(scores),
        topk_values(topk_values),
        topk_indices(topk_indices),
        bias(bias),
        num_tokens(num_tokens),
        num_experts(num_experts),
        n_group(n_group),
        topk_group(topk_group),
        topk(topk),
        renormalize(renormalize),
        routed_scaling_factor(routed_scaling_factor) {}

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    int32_t const token_id = static_cast<int32_t>(item.get_group(0));
    if (token_id >= static_cast<int32_t>(num_tokens)) {
      return;
    }

    int32_t const local_id = static_cast<int32_t>(item.get_local_id(0));
    int32_t const local_range = static_cast<int32_t>(item.get_local_range(0));
    int32_t const warp_id = local_id / WARP_SIZE;
    int32_t const lane_id = local_id % WARP_SIZE;
    int32_t const n_group_i32 = static_cast<int32_t>(n_group);
    int32_t const topk_group_i32 = static_cast<int32_t>(topk_group);
    int32_t const topk_i32 = static_cast<int32_t>(topk);
    int32_t const num_experts_i32 = static_cast<int32_t>(num_experts);
    int32_t const num_warps = local_range / WARP_SIZE;

    // Each work-group is launched with exactly n_group sub_groups.
    if (warp_id >= n_group_i32 || num_warps < n_group_i32) {
      return;
    }

    int32_t const num_experts_per_group = num_experts_i32 / n_group_i32;

    T* scores_token = scores + static_cast<int64_t>(token_id) * num_experts;

    auto sg = item.get_sub_group();

    // --- Shared memory layout ---
    char* smem_buf =
        slm.template get_multi_ptr<sycl::access::decorated::no>().get();

    // WarpSelect internal staging occupies the first `internal_bytes`.
    size_t const val_bytes =
        static_cast<size_t>(num_warps) * WARP_SIZE * sizeof(T);
    size_t const val_bytes_aligned =
        warp_topk::round_up_to_multiple_of<256>(val_bytes);
    size_t const idx_bytes =
        static_cast<size_t>(num_warps) * WARP_SIZE * sizeof(int32_t);
    size_t const internal_bytes = val_bytes_aligned + idx_bytes;

    // User-managed smem starts after internal staging, aligned to 16 bytes.
    uintptr_t ptr_u = reinterpret_cast<uintptr_t>(smem_buf + internal_bytes);
    ptr_u = (ptr_u + 15) & ~static_cast<uintptr_t>(15);
    T* s_group_scores = reinterpret_cast<T*>(ptr_u);

    // --- Phase 1: per-group scan (each sub_group handles one expert group) ---
    int32_t const group_offset = warp_id * num_experts_per_group;
    topk_with_k2_biased<T, BiasT, SF>(
        s_group_scores + warp_id,
        scores_token + group_offset,
        bias + group_offset,
        sg,
        lane_id,
        num_experts_per_group);

    item.barrier(sycl::access::fence_space::local_space);

    // --- Phase 2: sub_group 0 selects groups and experts ---
    if (warp_id != 0) {
      return;
    }

    float* topk_values_token =
        topk_values + static_cast<int64_t>(token_id) * topk;
    IdxT* topk_indices_token =
        topk_indices + static_cast<int64_t>(token_id) * topk;

    // Select topk_group groups by group score using WarpSelect.
    warp_topk::WarpSelect<
        /*capacity*/ WARP_SIZE,
        /*greater*/ true,
        T,
        int32_t,
        /*is_stable*/ true>
        group_sel(
            topk_group_i32,
            sycl_cast<T, float>(-INFINITY),
            smem_buf,
            local_id,
            local_range);

    // All lanes must participate; lanes beyond n_group feed -inf.
    T gscore = (lane_id < n_group_i32) ? s_group_scores[lane_id]
                                       : sycl_cast<T, float>(-INFINITY);
    group_sel.add(gscore, lane_id, sg, local_id);
    group_sel.done(sg, local_id);

    // Proceed only if k-th selected group score is not -inf.
    bool proceed = false;
    if (topk_group_i32 > 0) {
      int const kth_lane = topk_group_i32 - 1;
      T kth_val = sycl::select_from_group(sg, group_sel.get_val(0), kth_lane);
      proceed = (sycl_cast<float, T>(kth_val) != -INFINITY);
    }

    if (!proceed) {
      for (int i = lane_id; i < topk_i32; i += WARP_SIZE) {
        topk_indices_token[i] = static_cast<IdxT>(i);
        topk_values_token[i] = 1.0f / static_cast<float>(topk_i32);
      }
      return;
    }

    // Merge per-group topk candidates from selected groups, then select topk.
    warp_topk::WarpSelect<
        /*capacity*/ WARP_SIZE,
        /*greater*/ true,
        T,
        int32_t,
        /*is_stable*/ true>
        expert_sel(
            topk_i32,
            sycl_cast<T, float>(-INFINITY),
            smem_buf,
            local_id,
            local_range);

    // Selected group ids reside in lanes [0, topk_group).
    int32_t sel_gid_lane =
        (lane_id < topk_group_i32) ? group_sel.get_idx(0) : 0;

    int32_t const align_num_experts_per_group =
        warp_topk::round_up_to_multiple_of<WARP_SIZE>(num_experts_per_group);

    for (int32_t g = 0; g < topk_group_i32; ++g) {
      int32_t gid = sycl::select_from_group(sg, sel_gid_lane, g);
      int32_t const offset = gid * num_experts_per_group;
      for (int32_t i = lane_id; i < align_num_experts_per_group;
           i += WARP_SIZE) {
        // All lanes must call add() the same number of times.
        T cand = sycl_cast<T, float>(-INFINITY);
        int32_t idx = 0;
        if (i < num_experts_per_group) {
          idx = offset + i;
          T input_val = scores_token[idx];
          float input_f = sycl_cast<float, T>(input_val);
          if (sycl::isinf(input_f) == 0) {
            T score = apply_scoring<SF>(input_val);
            cand = score + static_cast<T>(bias[idx]);
          }
        }
        expert_sel.add(cand, idx, sg, local_id);
      }
    }
    expert_sel.done(sg, local_id);

    // Compute unbiased routing weights + optional renorm.
    float lane_unbiased = 0.0f;
    IdxT lane_idx = 0;
    if (lane_id < topk_i32) {
      lane_idx = static_cast<IdxT>(expert_sel.get_idx(0));
      T in = scores_token[static_cast<int32_t>(lane_idx)];
      lane_unbiased = sycl_cast<float, T>(apply_scoring<SF>(in));
    }

    float topk_sum = 1e-20f;
    if (renormalize) {
      topk_sum +=
          sycl::reduce_over_group(sg, lane_unbiased, sycl::plus<float>());
    }

    float scale = static_cast<float>(routed_scaling_factor);
    if (renormalize) {
      scale /= topk_sum;
    }

    if (lane_id < topk_i32) {
      topk_indices_token[lane_id] = lane_idx;
      topk_values_token[lane_id] = lane_unbiased * scale;
    }
  }

 private:
  sycl::local_accessor<char, 1> slm;
  T* scores;
  float* topk_values;
  IdxT* topk_indices;
  BiasT const* bias;
  int64_t const num_tokens;
  int64_t const num_experts;
  int64_t const n_group;
  int64_t const topk_group;
  int64_t const topk;
  bool renormalize;
  double routed_scaling_factor;
};

// -------------------------------------------------------------------------
// invokeNoAuxTcFused: fused path (mirrors CUDA grouped_topk_fused_kernel).
// Takes scores + separate bias tensor; no pre-computed scores_with_bias.
// -------------------------------------------------------------------------
template <typename T, typename BiasT, typename IdxT, ScoringFunc SF>
void invokeNoAuxTcFused(
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
    sycl::queue& queue) {
  // One work-group per token; one sub_group (warp) per expert group.
  int32_t const num_warps = static_cast<int32_t>(n_group);
  int32_t const block_size = num_warps * WARP_SIZE;

  // Compute dynamic shared memory size (mirrors CUDA).
  size_t const val_bytes =
      static_cast<size_t>(num_warps) * WARP_SIZE * sizeof(T);
  size_t const val_bytes_aligned =
      warp_topk::round_up_to_multiple_of<256>(val_bytes);
  size_t const idx_bytes =
      static_cast<size_t>(num_warps) * WARP_SIZE * sizeof(int32_t);
  size_t const internal_bytes = val_bytes_aligned + idx_bytes;
  // extra: 16-byte alignment padding + s_group_scores[n_group]
  size_t const extra_bytes = 16 + static_cast<size_t>(n_group) * sizeof(T);
  size_t const smem_bytes = internal_bytes + extra_bytes;

  sycl::range<1> grid(static_cast<size_t>(num_tokens));
  sycl::range<1> block(static_cast<size_t>(block_size));

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<char, 1> slm(sycl::range<1>(smem_bytes), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(grid * block, block),
        grouped_topk_fused_kernel_impl<T, BiasT, IdxT, SF>(
            slm,
            scores,
            topk_values,
            topk_indices,
            bias,
            num_tokens,
            num_experts,
            n_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor));
  });
}

#define INSTANTIATE_NOAUX_TC_FUSED(T, BiasT, IdxT, SF)  \
  template void invokeNoAuxTcFused<T, BiasT, IdxT, SF>( \
      T * scores,                                       \
      float* topk_values,                               \
      IdxT* topk_indices,                               \
      BiasT const* bias,                                \
      int64_t const num_tokens,                         \
      int64_t const num_experts,                        \
      int64_t const n_group,                            \
      int64_t const topk_group,                         \
      int64_t const topk,                               \
      bool const renormalize,                           \
      double const routed_scaling_factor,               \
      sycl::queue& queue);

INSTANTIATE_NOAUX_TC_FUSED(float, float, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC_FUSED(float, sycl::half, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC_FUSED(
    float, sycl::ext::oneapi::bfloat16, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC_FUSED(sycl::half, float, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC_FUSED(sycl::half, sycl::half, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC_FUSED(
    sycl::half, sycl::ext::oneapi::bfloat16, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC_FUSED(
    sycl::ext::oneapi::bfloat16, float, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC_FUSED(
    sycl::ext::oneapi::bfloat16, sycl::half, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC_FUSED(
    sycl::ext::oneapi::bfloat16,
    sycl::ext::oneapi::bfloat16,
    int32_t,
    SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC_FUSED(float, float, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC_FUSED(float, sycl::half, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC_FUSED(
    float, sycl::ext::oneapi::bfloat16, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC_FUSED(sycl::half, float, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC_FUSED(sycl::half, sycl::half, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC_FUSED(
    sycl::half, sycl::ext::oneapi::bfloat16, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC_FUSED(
    sycl::ext::oneapi::bfloat16, float, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC_FUSED(
    sycl::ext::oneapi::bfloat16, sycl::half, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC_FUSED(
    sycl::ext::oneapi::bfloat16,
    sycl::ext::oneapi::bfloat16,
    int32_t,
    SCORING_NONE);

}  // end namespace moe
}  // namespace vllm

// -------------------------------------------------------------------------
// Python binding: fused grouped topk with separate bias + scoring_func.
// Mirrors the CUDA grouped_topk signature in grouped_topk_kernels.cu.
// topk_values output is always float32.
// -------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> grouped_topk(
    torch::Tensor const& scores,
    int64_t n_group,
    int64_t topk_group,
    int64_t topk,
    bool renormalize,
    double routed_scaling_factor,
    torch::Tensor const& bias,
    int64_t scoring_func) {
  auto data_type = scores.scalar_type();
  auto bias_type = bias.scalar_type();
  auto input_size = scores.sizes();
  int64_t num_tokens = input_size[0];
  int64_t num_experts = input_size[1];
  TORCH_CHECK(input_size.size() == 2, "scores must be a 2D Tensor");
  TORCH_CHECK(
      num_experts % n_group == 0, "num_experts should be divisible by n_group");
  TORCH_CHECK(
      n_group <= 32, "n_group should be smaller than or equal to 32 for now");
  TORCH_CHECK(topk <= 32, "topk should be smaller than or equal to 32 for now");
  TORCH_CHECK(
      topk <= topk_group * (num_experts / n_group),
      "topk must be <= topk_group * (num_experts / n_group)");
  TORCH_CHECK(
      scoring_func == vllm::moe::SCORING_NONE ||
          scoring_func == vllm::moe::SCORING_SIGMOID,
      "scoring_func must be SCORING_NONE (0) or SCORING_SIGMOID (1)");

  // topk_values is always float32 (matches CUDA behavior).
  torch::Tensor topk_values = torch::empty(
      {num_tokens, topk}, torch::dtype(torch::kFloat32).device(torch::kXPU));
  torch::Tensor topk_indices = torch::empty(
      {num_tokens, topk}, torch::dtype(torch::kInt32).device(torch::kXPU));

  auto& queue = vllm::xpu::vllmGetQueue();

  auto sf = static_cast<vllm::moe::ScoringFunc>(scoring_func);

#define LAUNCH_FUSED_SF(T, BiasT, IdxT)                                     \
  do {                                                                      \
    switch (sf) {                                                           \
      case vllm::moe::SCORING_NONE:                                         \
        vllm::moe::                                                         \
            invokeNoAuxTcFused<T, BiasT, IdxT, vllm::moe::SCORING_NONE>(    \
                reinterpret_cast<T*>(scores.mutable_data_ptr()),            \
                reinterpret_cast<float*>(topk_values.mutable_data_ptr()),   \
                reinterpret_cast<IdxT*>(topk_indices.mutable_data_ptr()),   \
                reinterpret_cast<BiasT const*>(bias.data_ptr()),            \
                num_tokens,                                                 \
                num_experts,                                                \
                n_group,                                                    \
                topk_group,                                                 \
                topk,                                                       \
                renormalize,                                                \
                routed_scaling_factor,                                      \
                queue);                                                     \
        break;                                                              \
      case vllm::moe::SCORING_SIGMOID:                                      \
        vllm::moe::                                                         \
            invokeNoAuxTcFused<T, BiasT, IdxT, vllm::moe::SCORING_SIGMOID>( \
                reinterpret_cast<T*>(scores.mutable_data_ptr()),            \
                reinterpret_cast<float*>(topk_values.mutable_data_ptr()),   \
                reinterpret_cast<IdxT*>(topk_indices.mutable_data_ptr()),   \
                reinterpret_cast<BiasT const*>(bias.data_ptr()),            \
                num_tokens,                                                 \
                num_experts,                                                \
                n_group,                                                    \
                topk_group,                                                 \
                topk,                                                       \
                renormalize,                                                \
                routed_scaling_factor,                                      \
                queue);                                                     \
        break;                                                              \
      default:                                                              \
        throw std::invalid_argument("Unsupported scoring_func");            \
        break;                                                              \
    }                                                                       \
  } while (0)

#define LAUNCH_FUSED(T, IdxT)                                                \
  do {                                                                       \
    switch (bias_type) {                                                     \
      case torch::kFloat16:                                                  \
        LAUNCH_FUSED_SF(T, sycl::half, IdxT);                                \
        break;                                                               \
      case torch::kFloat32:                                                  \
        LAUNCH_FUSED_SF(T, float, IdxT);                                     \
        break;                                                               \
      case torch::kBFloat16:                                                 \
        LAUNCH_FUSED_SF(T, sycl::ext::oneapi::bfloat16, IdxT);               \
        break;                                                               \
      default:                                                               \
        throw std::invalid_argument(                                         \
            "Invalid bias dtype, only supports float16, float32, bfloat16"); \
        break;                                                               \
    }                                                                        \
  } while (0)

  switch (data_type) {
    case torch::kFloat16:
      LAUNCH_FUSED(sycl::half, int32_t);
      break;
    case torch::kFloat32:
      LAUNCH_FUSED(float, int32_t);
      break;
    case torch::kBFloat16:
      LAUNCH_FUSED(sycl::ext::oneapi::bfloat16, int32_t);
      break;
    default:
      throw std::invalid_argument(
          "Invalid dtype, only supports float16, float32, and bfloat16");
      break;
  }
#undef LAUNCH_FUSED
#undef LAUNCH_FUSED_SF

  return {topk_values, topk_indices};
}
