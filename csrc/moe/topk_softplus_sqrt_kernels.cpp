#include <type_traits>
#include <cstdint>
#include <torch/all.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <ATen/xpu/XPUContext.h>

#include "../utils.h"
#include "../dispatch_utils.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace vllm {
namespace moe {

namespace sycl_ext = sycl::ext::oneapi;
using bfloat16_t = sycl_ext::bfloat16;
using half_t = sycl::half;

template <typename T, int N, int Alignment = sizeof(T) * N>
struct alignas(Alignment) AlignedArray {
  T data[N];
};

template <typename T>
static inline float toFloat(T value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else if constexpr (std::is_same_v<T, bfloat16_t>) {
    return static_cast<float>(value);
  } else if constexpr (std::is_same_v<T, half_t>) {
    return static_cast<float>(value);
  }
}

template <typename T>
static inline T warpReduceSum(T val, sycl::sub_group sg) {
  const int sg_size = sg.get_local_range()[0];
#pragma unroll
  for (int mask = sg_size / 2; mask > 0; mask >>= 1) {
    val += sycl::permute_group_by_xor(sg, val, mask);
  }
  return val;
}

template <typename T>
static inline T
shflXorWidth(sycl::sub_group sg, T val, int mask, int /*width*/) {
  return sycl::permute_group_by_xor(sg, val, static_cast<unsigned>(mask));
}

template <typename T>
static inline T warpReduceSumWidth(T val, sycl::sub_group sg, int width) {
#pragma unroll
  for (int mask = width / 2; mask > 0; mask >>= 1) {
    val += shflXorWidth(sg, val, mask, width);
  }
  return val;
}

// Load a row chunk from global memory into float registers.
// Primary template — partial specializations below handle each InputType.
template <
    typename InputType,
    int ELTS_PER_LDG,
    int LDG_PER_THREAD,
    int THREADS_PER_ROW>
struct RowChunkLoader;

// float specialization: direct vector load, no conversion needed.
template <int ELTS_PER_LDG, int LDG_PER_THREAD, int THREADS_PER_ROW>
struct RowChunkLoader<float, ELTS_PER_LDG, LDG_PER_THREAD, THREADS_PER_ROW> {
  static inline void load(float* row_chunk, const float* read_ptr) {
    using VecType = AlignedArray<float, ELTS_PER_LDG>;
    VecType* row_chunk_vec_ptr = reinterpret_cast<VecType*>(row_chunk);
    const VecType* vec_read_ptr = reinterpret_cast<const VecType*>(read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
      row_chunk_vec_ptr[ii] = vec_read_ptr[ii * THREADS_PER_ROW];
    }
  }
};

// bfloat16 specialization: vector or scalar load + convert to float.
template <int ELTS_PER_LDG, int LDG_PER_THREAD, int THREADS_PER_ROW>
struct RowChunkLoader<
    bfloat16_t,
    ELTS_PER_LDG,
    LDG_PER_THREAD,
    THREADS_PER_ROW> {
  static inline void load(float* row_chunk, const bfloat16_t* read_ptr) {
    if constexpr (ELTS_PER_LDG >= 2) {
      using VecType = AlignedArray<bfloat16_t, ELTS_PER_LDG>;
      const VecType* vec_read_ptr = reinterpret_cast<const VecType*>(read_ptr);
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        VecType vec = vec_read_ptr[ii * THREADS_PER_ROW];
#pragma unroll
        for (int jj = 0; jj < ELTS_PER_LDG; ++jj) {
          row_chunk[ii * ELTS_PER_LDG + jj] = static_cast<float>(vec.data[jj]);
        }
      }
    } else {
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        row_chunk[ii] = static_cast<float>(*(read_ptr + ii * THREADS_PER_ROW));
      }
    }
  }
};

// half specialization: vector or scalar load + convert to float.
template <int ELTS_PER_LDG, int LDG_PER_THREAD, int THREADS_PER_ROW>
struct RowChunkLoader<half_t, ELTS_PER_LDG, LDG_PER_THREAD, THREADS_PER_ROW> {
  static inline void load(float* row_chunk, const half_t* read_ptr) {
    if constexpr (ELTS_PER_LDG >= 2) {
      using VecType = AlignedArray<half_t, ELTS_PER_LDG>;
      const VecType* vec_read_ptr = reinterpret_cast<const VecType*>(read_ptr);
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        VecType vec = vec_read_ptr[ii * THREADS_PER_ROW];
#pragma unroll
        for (int jj = 0; jj < ELTS_PER_LDG; ++jj) {
          row_chunk[ii * ELTS_PER_LDG + jj] = static_cast<float>(vec.data[jj]);
        }
      }
    } else {
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        row_chunk[ii] = static_cast<float>(*(read_ptr + ii * THREADS_PER_ROW));
      }
    }
  }
};

template <
    int VPT,
    int NUM_EXPERTS,
    int WARPS_PER_CTA,
    int BYTES_PER_LDG,
    int WARP_SIZE_PARAM,
    bool USE_HASH,
    typename IndType,
    typename InputType = float>
struct TopkGatingSoftplusSqrtKernel {
  const InputType* input;
  const bool* finished;
  float* output;
  int num_rows;
  IndType* indices;
  int* source_rows;
  int k;
  int start_expert;
  int end_expert;
  bool renormalize;
  double routed_scaling_factor;
  const float* correction_bias;
  const IndType* input_ids;
  const IndType* tid2eid;

  [[sycl::reqd_sub_group_size(WARP_SIZE_PARAM)]]
  void operator()(sycl::nd_item<3> item) const {
    static_assert(
        std::is_same_v<InputType, float> ||
            std::is_same_v<InputType, bfloat16_t> ||
            std::is_same_v<InputType, half_t>,
        "InputType must be float, bfloat16, or half");

    static_assert(
        BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
        "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    if constexpr (
        std::is_same_v<InputType, bfloat16_t> ||
        std::is_same_v<InputType, half_t>) {
      static_assert(
          ELTS_PER_LDG == 1 || ELTS_PER_LDG % 2 == 0,
          "ELTS_PER_LDG must be 1 or even for 16-bit conversion");
    }
    static_assert(
        VPT % ELTS_PER_LDG == 0, "VPT must be a multiple of ELTS_PER_LDG");
    static_assert(
        WARP_SIZE_PARAM % THREADS_PER_ROW == 0,
        "THREADS_PER_ROW must cleanly divide WARP_SIZE_PARAM");
    static_assert(
        THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW),
        "THREADS_PER_ROW must be power of 2");
    static_assert(
        THREADS_PER_ROW <= WARP_SIZE_PARAM,
        "THREADS_PER_ROW can be at most warp size");

    static constexpr int ELTS_PER_WARP = WARP_SIZE_PARAM * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    static_assert(
        ELTS_PER_WARP % ELTS_PER_ROW == 0,
        "ELTS_PER_WARP must be a multiple of ELTS_PER_ROW");

    const int block_x = static_cast<int>(item.get_group(0));
    const int tidy = static_cast<int>(item.get_local_id(1));
    const int tidx = static_cast<int>(item.get_local_id(2));

    const int cta_base_row = block_x * ROWS_PER_CTA;
    const int warp_base_row = cta_base_row + tidy * ROWS_PER_WARP;

    const int thread_row_in_warp = tidx / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;

    if (thread_row >= num_rows) {
      return;
    }
    const bool row_is_active = finished ? !finished[thread_row] : true;

    const InputType* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

    const int thread_group_idx = tidx % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const InputType* thread_read_ptr =
        thread_row_ptr + first_elt_read_by_thread;

    alignas(sizeof(float) * ELTS_PER_LDG) float row_chunk[VPT];
    RowChunkLoader<InputType, ELTS_PER_LDG, LDG_PER_THREAD, THREADS_PER_ROW>::
        load(row_chunk, thread_read_ptr);

    constexpr float threshold = 20.0f;
    constexpr float beta = 1.0f;

    sycl::sub_group sg = item.get_sub_group();

    if constexpr (USE_HASH) {
      const IndType token_id = input_ids[thread_row];
      const IndType* expert_indices_for_token = tid2eid + token_id * k;

      float selected_sum = 0.f;
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        const int expert = static_cast<int>(expert_indices_for_token[k_idx]);
        const int idx = k * thread_row + k_idx;

        float selected_weight_lane = 0.f;
#pragma unroll
        for (int ii = 0; ii < VPT; ++ii) {
          const int group_id = ii / ELTS_PER_LDG;
          const int local_id = ii % ELTS_PER_LDG;
          const int expert_idx = first_elt_read_by_thread +
                                 group_id * THREADS_PER_ROW * ELTS_PER_LDG +
                                 local_id;
          if (expert == expert_idx) {
            float val = row_chunk[ii];
            float val_b = val * beta;
            val = (val_b > threshold)
                      ? val
                      : (sycl::log(1.0f + sycl::exp(val_b))) / beta;
            val = sycl::sqrt(val);
            if (sycl::isnan(val) || sycl::isinf(val)) {
              val = 0.f;
            }
            selected_weight_lane = val;
            break;
          }
        }

        const float selected_weight =
            warpReduceSumWidth(selected_weight_lane, sg, THREADS_PER_ROW);

        if (thread_group_idx == 0) {
          const bool node_uses_expert =
              expert >= start_expert && expert < end_expert;
          const bool should_process_row = row_is_active && node_uses_expert;

          output[idx] = selected_weight;
          indices[idx] =
              should_process_row ? (expert - start_expert) : NUM_EXPERTS;
          source_rows[idx] = k_idx * num_rows + thread_row;
          if (renormalize) {
            selected_sum += selected_weight;
          }
        }
      }

      if (thread_group_idx == 0) {
        float scale = static_cast<float>(routed_scaling_factor);
        if (renormalize) {
          const float denom = selected_sum > 0.f ? selected_sum : 1.f;
          scale /= denom;
        }
#pragma unroll
        for (int k_idx = 0; k_idx < k; ++k_idx) {
          const int idx = k * thread_row + k_idx;
          output[idx] = output[idx] * scale;
        }
      }
      return;
    }

#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      float val = row_chunk[ii];
      float val_b = val * beta;
      val = (val_b > threshold) ? val
                                : (sycl::log(1.0f + sycl::exp(val_b))) / beta;
      val = sycl::sqrt(val);
      if (correction_bias) {
        const int group_id = ii / ELTS_PER_LDG;
        const int local_id = ii % ELTS_PER_LDG;
        const int expert_idx = first_elt_read_by_thread +
                               group_id * THREADS_PER_ROW * ELTS_PER_LDG +
                               local_id;
        val = val + correction_bias[expert_idx];
      }
      row_chunk[ii] = val;
    }

    int start_col = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    float selected_sum = 0.f;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      float max_val = row_chunk[0];
      int expert = start_col;
#pragma unroll
      for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD;
           ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
        for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
          float val = row_chunk[ldg * ELTS_PER_LDG + ii];
          if (val > max_val) {
            max_val = val;
            expert = col + ii;
          }
        }
      }

#pragma unroll
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        float other_max = shflXorWidth(sg, max_val, mask, THREADS_PER_ROW);
        int other_expert = shflXorWidth(sg, expert, mask, THREADS_PER_ROW);

        if (other_max > max_val ||
            (other_max == max_val && other_expert < expert)) {
          max_val = other_max;
          expert = other_expert;
        }
      }

      if (thread_group_idx == 0) {
        const bool node_uses_expert =
            expert >= start_expert && expert < end_expert;
        const bool should_process_row = row_is_active && node_uses_expert;

        const int idx = k * thread_row + k_idx;
        if (correction_bias != nullptr) {
          max_val -= correction_bias[expert];
        }
        output[idx] = max_val;
        indices[idx] =
            should_process_row ? (expert - start_expert) : NUM_EXPERTS;
        source_rows[idx] = k_idx * num_rows + thread_row;
        if (renormalize) {
          selected_sum += max_val;
        }
      }

      if (k_idx + 1 < k) {
        const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
        const int thread_to_clear_in_group =
            (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

        if (thread_group_idx == thread_to_clear_in_group) {
          const int offset_for_expert = expert % ELTS_PER_LDG;
          row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] =
              -10000.f;
        }
      }
    }

    if (thread_group_idx == 0) {
      float scale = static_cast<float>(routed_scaling_factor);
      if (renormalize) {
        const float denom = selected_sum > 0.f ? selected_sum : 1.f;
        scale /= denom;
      }
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        const int idx = k * thread_row + k_idx;
        output[idx] = output[idx] * scale;
      }
    }
  }
};

namespace detail {
template <
    int EXPERTS,
    int BYTES_PER_LDG,
    int WARP_SIZE_PARAM,
    typename InputType>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
  static_assert(
      EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0 ||
          EXPERTS % (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0,
      "");
  static constexpr int VECs_PER_THREAD =
      MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM));
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static constexpr int ROWS_PER_WARP = WARP_SIZE_PARAM / THREADS_PER_ROW;
};
}  // namespace detail

#define DISPATCH_HASH(use_hash, USE_HASH, ...) \
  if (use_hash) {                              \
    constexpr bool USE_HASH = true;            \
    __VA_ARGS__                                \
  } else {                                     \
    constexpr bool USE_HASH = false;           \
    __VA_ARGS__                                \
  }

template <
    int EXPERTS,
    int WARPS_PER_TB,
    int WARP_SIZE_PARAM,
    int MAX_BYTES_PER_LDG,
    typename IndType,
    typename InputType>
void topkGatingSoftplusSqrtLauncherHelper(
    const InputType* input,
    const bool* finished,
    float* output,
    IndType* indices,
    int* source_row,
    const int num_rows,
    const int k,
    const int start_expert,
    const int end_expert,
    const bool renormalize,
    double routed_scaling_factor,
    const float* correction_bias,
    const bool use_hash,
    const IndType* input_ids,
    const IndType* tid2eid,
    sycl::queue& q) {
  static constexpr int BYTES_PER_LDG =
      MIN(MAX_BYTES_PER_LDG, sizeof(InputType) * EXPERTS);
  using Constants =
      detail::TopkConstants<EXPERTS, BYTES_PER_LDG, WARP_SIZE_PARAM, InputType>;
  static constexpr int VPT = Constants::VPT;
  static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
  const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

  sycl::range<3> local_range(1, WARPS_PER_TB, WARP_SIZE_PARAM);
  sycl::range<3> global_range(
      static_cast<size_t>(num_blocks), WARPS_PER_TB, WARP_SIZE_PARAM);
  sycl::nd_range<3> ndr(global_range, local_range);

  DISPATCH_HASH(use_hash, USE_HASH, {
    using Kernel = TopkGatingSoftplusSqrtKernel<
        VPT,
        EXPERTS,
        WARPS_PER_TB,
        BYTES_PER_LDG,
        WARP_SIZE_PARAM,
        USE_HASH,
        IndType,
        InputType>;
    Kernel k_obj{
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
        routed_scaling_factor,
        correction_bias,
        input_ids,
        tid2eid};
    q.parallel_for(ndr, k_obj);
  })
}

#define LAUNCH_SOFTPLUS_SQRT(NUM_EXPERTS, WARPS_PER_TB, MAX_BYTES) \
  topkGatingSoftplusSqrtLauncherHelper<                            \
      NUM_EXPERTS,                                                 \
      WARPS_PER_TB,                                                \
      32,                                                          \
      MAX_BYTES>(                                                  \
      gating_output,                                               \
      nullptr,                                                     \
      topk_weights,                                                \
      topk_indices,                                                \
      token_expert_indices,                                        \
      num_tokens,                                                  \
      topk,                                                        \
      0,                                                           \
      num_experts,                                                 \
      renormalize,                                                 \
      routed_scaling_factor,                                       \
      correction_bias,                                             \
      use_hash,                                                    \
      input_ids,                                                   \
      tid2eid,                                                     \
      q);

template <typename IndType, typename InputType>
void topkGatingSoftplusSqrtKernelLauncher(
    const InputType* gating_output,
    float* topk_weights,
    IndType* topk_indices,
    int* token_expert_indices,
    const int num_tokens,
    const int num_experts,
    const int topk,
    const bool renormalize,
    double routed_scaling_factor,
    const float* correction_bias,
    const bool use_hash,
    const IndType* input_ids,
    const IndType* tid2eid,
    sycl::queue& q) {
  static constexpr int WARPS_PER_TB = 4;
  static constexpr int BYTES_PER_LDG_POWER_OF_2 = 16;
  static constexpr int BYTES_PER_LDG_MULTIPLE_64 =
      (std::is_same_v<InputType, bfloat16_t> ||
       std::is_same_v<InputType, half_t>)
          ? 4
          : 8;

  switch (num_experts) {
    case 1:
      LAUNCH_SOFTPLUS_SQRT(1, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 2:
      LAUNCH_SOFTPLUS_SQRT(2, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 4:
      LAUNCH_SOFTPLUS_SQRT(4, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 8:
      LAUNCH_SOFTPLUS_SQRT(8, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 16:
      LAUNCH_SOFTPLUS_SQRT(16, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 32:
      LAUNCH_SOFTPLUS_SQRT(32, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 64:
      LAUNCH_SOFTPLUS_SQRT(64, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 128:
      LAUNCH_SOFTPLUS_SQRT(128, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 256:
      LAUNCH_SOFTPLUS_SQRT(256, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 512:
      LAUNCH_SOFTPLUS_SQRT(512, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 192:
      LAUNCH_SOFTPLUS_SQRT(192, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    case 320:
      LAUNCH_SOFTPLUS_SQRT(320, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    case 384:
      LAUNCH_SOFTPLUS_SQRT(384, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    case 448:
      LAUNCH_SOFTPLUS_SQRT(448, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    case 576:
      LAUNCH_SOFTPLUS_SQRT(576, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    default:
      TORCH_CHECK(false, "Unsupported expert number: ", num_experts);
  }
}

}  // namespace moe
}  // namespace vllm

template <typename ComputeType>
void dispatch_topk_softplus_sqrt_launch(
    const ComputeType* gating_output,
    torch::Tensor& topk_weights,
    torch::Tensor& topk_indices,
    torch::Tensor& token_expert_indices,
    int num_tokens,
    int num_experts,
    int topk,
    bool renormalize,
    double routed_scaling_factor,
    const c10::optional<torch::Tensor>& correction_bias,
    const c10::optional<torch::Tensor>& input_ids,
    const c10::optional<torch::Tensor>& tid2eid,
    sycl::queue& q) {
  const float* bias_ptr = nullptr;
  if (correction_bias.has_value()) {
    TORCH_CHECK(
        correction_bias.value().scalar_type() == at::ScalarType::Float,
        "correction_bias must be float32");
    TORCH_CHECK(
        correction_bias.value().is_contiguous(),
        "correction_bias must be contiguous");
    bias_ptr = correction_bias.value().data_ptr<float>();
  }
  bool use_hash = false;
  if (tid2eid.has_value()) {
    TORCH_CHECK(input_ids.has_value(), "input_ids is required for hash MoE");
    TORCH_CHECK(
        input_ids.value().scalar_type() == topk_indices.scalar_type(),
        "input_ids dtype must match topk_indices dtype");
    TORCH_CHECK(
        tid2eid.value().scalar_type() == topk_indices.scalar_type(),
        "tid2eid dtype must match topk_indices dtype");
    use_hash = true;
  }
  if (topk_indices.scalar_type() == at::ScalarType::Int) {
    const int* input_ids_ptr = nullptr;
    const int* tid2eid_ptr = nullptr;
    if (tid2eid.has_value()) {
      input_ids_ptr = input_ids.value().data_ptr<int>();
      tid2eid_ptr = tid2eid.value().data_ptr<int>();
    }
    vllm::moe::topkGatingSoftplusSqrtKernelLauncher<int, ComputeType>(
        gating_output,
        topk_weights.data_ptr<float>(),
        topk_indices.data_ptr<int>(),
        token_expert_indices.data_ptr<int>(),
        num_tokens,
        num_experts,
        topk,
        renormalize,
        routed_scaling_factor,
        bias_ptr,
        use_hash,
        input_ids_ptr,
        tid2eid_ptr,
        q);
  } else if (topk_indices.scalar_type() == at::ScalarType::UInt32) {
    const uint32_t* input_ids_ptr = nullptr;
    const uint32_t* tid2eid_ptr = nullptr;
    if (tid2eid.has_value()) {
      input_ids_ptr = input_ids.value().data_ptr<uint32_t>();
      tid2eid_ptr = tid2eid.value().data_ptr<uint32_t>();
    }
    vllm::moe::topkGatingSoftplusSqrtKernelLauncher<uint32_t, ComputeType>(
        gating_output,
        topk_weights.data_ptr<float>(),
        topk_indices.data_ptr<uint32_t>(),
        token_expert_indices.data_ptr<int>(),
        num_tokens,
        num_experts,
        topk,
        renormalize,
        routed_scaling_factor,
        bias_ptr,
        use_hash,
        input_ids_ptr,
        tid2eid_ptr,
        q);
  } else {
    TORCH_CHECK(topk_indices.scalar_type() == at::ScalarType::Long);
    const int64_t* input_ids_ptr = nullptr;
    const int64_t* tid2eid_ptr = nullptr;
    if (tid2eid.has_value()) {
      input_ids_ptr = input_ids.value().data_ptr<int64_t>();
      tid2eid_ptr = tid2eid.value().data_ptr<int64_t>();
    }
    vllm::moe::topkGatingSoftplusSqrtKernelLauncher<int64_t, ComputeType>(
        gating_output,
        topk_weights.data_ptr<float>(),
        topk_indices.data_ptr<int64_t>(),
        token_expert_indices.data_ptr<int>(),
        num_tokens,
        num_experts,
        topk,
        renormalize,
        routed_scaling_factor,
        bias_ptr,
        use_hash,
        input_ids_ptr,
        tid2eid_ptr,
        q);
  }
}

void topk_softplus_sqrt(
    torch::Tensor& topk_weights,
    torch::Tensor& topk_indices,
    torch::Tensor& token_expert_indices,
    torch::Tensor& gating_output,
    bool renormalize,
    double routed_scaling_factor,
    const c10::optional<torch::Tensor>& correction_bias,
    const c10::optional<torch::Tensor>& input_ids,
    const c10::optional<torch::Tensor>& tid2eid) {
  const int num_experts = gating_output.size(-1);
  const auto num_tokens = gating_output.numel() / num_experts;
  const int topk = topk_weights.size(-1);

  const c10::OptionalDeviceGuard device_guard(device_of(gating_output));
  sycl::queue& q = at::xpu::getCurrentXPUStream().queue();

  using vllm::moe::bfloat16_t;
  using vllm::moe::half_t;

  if (gating_output.scalar_type() == at::ScalarType::Float) {
    dispatch_topk_softplus_sqrt_launch<float>(
        gating_output.data_ptr<float>(),
        topk_weights,
        topk_indices,
        token_expert_indices,
        num_tokens,
        num_experts,
        topk,
        renormalize,
        routed_scaling_factor,
        correction_bias,
        input_ids,
        tid2eid,
        q);
  } else if (gating_output.scalar_type() == at::ScalarType::Half) {
    dispatch_topk_softplus_sqrt_launch<half_t>(
        reinterpret_cast<const half_t*>(gating_output.data_ptr<at::Half>()),
        topk_weights,
        topk_indices,
        token_expert_indices,
        num_tokens,
        num_experts,
        topk,
        renormalize,
        routed_scaling_factor,
        correction_bias,
        input_ids,
        tid2eid,
        q);
  } else if (gating_output.scalar_type() == at::ScalarType::BFloat16) {
    dispatch_topk_softplus_sqrt_launch<bfloat16_t>(
        reinterpret_cast<const bfloat16_t*>(
            gating_output.data_ptr<at::BFloat16>()),
        topk_weights,
        topk_indices,
        token_expert_indices,
        num_tokens,
        num_experts,
        topk,
        renormalize,
        routed_scaling_factor,
        correction_bias,
        input_ids,
        tid2eid,
        q);
  } else {
    TORCH_CHECK(
        false,
        "Unsupported gating_output data type: ",
        gating_output.scalar_type());
  }
}