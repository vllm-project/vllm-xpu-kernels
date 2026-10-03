#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"

namespace vllm {
namespace moe {

template <typename T, int TOPK, int ElemsPerItem>
class MoeGather {
 public:
  using vec_t = vllm::xpu::aligned_vec<T, ElemsPerItem>;

  MoeGather(
      T* output,
      const T* moe_output,
      const float* topk_weights,
      const int* unpermuted_row_to_permuted_row,
      const int num_experts,
      const int num_tokens,
      const int hidden_size,
      const int lanes_per_token)
      : output(output),
        moe_output(moe_output),
        topk_weights(topk_weights),
        unpermuted_row_to_permuted_row(unpermuted_row_to_permuted_row),
        num_experts(num_experts),
        num_tokens(num_tokens),
        hidden_size(hidden_size),
        lanes_per_token(lanes_per_token) {}

  static constexpr int MaxGroupWorkItem = 1024;
  static constexpr int WARP_SIZE = 16;

  static inline int get_lanes_per_token(const int hidden_size) {
    const int vecs = hidden_size / ElemsPerItem;
    return vecs < MaxGroupWorkItem ? vecs : MaxGroupWorkItem;
  }

  static inline sycl::nd_range<1>
  get_nd_range(const int num_tokens, const int lanes_per_token) {
    const int tokens_per_wg = MaxGroupWorkItem / lanes_per_token;
    const int num_wgs = (num_tokens + tokens_per_wg - 1) / tokens_per_wg;
    sycl::range<1> local(lanes_per_token * tokens_per_wg);
    sycl::range<1> group(num_wgs);
    return sycl::nd_range<1>(local * group, local);
  }

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    const int local_id_x = item.get_local_id(0);
    const int tokens_per_wg = item.get_local_range(0) / lanes_per_token;
    const int token_idx =
        item.get_group(0) * tokens_per_wg + local_id_x / lanes_per_token;
    if (token_idx >= num_tokens) {
      return;
    }

    int moe_ids[TOPK];
    float scores[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      moe_ids[k] = unpermuted_row_to_permuted_row[token_idx * TOPK + k];
      scores[k] = topk_weights[token_idx * TOPK + k];
    }

    bool all_valid = true;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      all_valid &= moe_ids[k] != -1;
    }

    // Compiler batches top-k loads in branch-free path.
    if (all_valid) {
      gather_token<true>(token_idx, local_id_x, moe_ids, scores);
    } else {
      gather_token<false>(token_idx, local_id_x, moe_ids, scores);
    }
  }

 private:
  template <bool IsAllValid>
  inline void gather_token(
      const int token_idx,
      const int local_id_x,
      const int (&moe_ids)[TOPK],
      const float (&scores)[TOPK]) const {
    const int num_vecs = hidden_size / ElemsPerItem;
    for (int v = local_id_x % lanes_per_token; v < num_vecs;
         v += lanes_per_token) {
      const int64_t hidden_idx = static_cast<int64_t>(v) * ElemsPerItem;

      vec_t in[TOPK];
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        if (IsAllValid || moe_ids[k] != -1) {
          in[k] = *reinterpret_cast<const vec_t*>(
              moe_output + static_cast<int64_t>(moe_ids[k]) * hidden_size +
              hidden_idx);
        }
      }

      float accum[ElemsPerItem];
#pragma unroll
      for (int e = 0; e < ElemsPerItem; ++e) {
        accum[e] = 0.0f;
      }
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        if (!IsAllValid && moe_ids[k] == -1) {
          continue;
        }
#pragma unroll
        for (int e = 0; e < ElemsPerItem; ++e) {
          accum[e] += static_cast<float>(in[k][e]) * scores[k];
        }
      }

      vec_t out;
#pragma unroll
      for (int e = 0; e < ElemsPerItem; ++e) {
        out[e] = static_cast<T>(accum[e]);
      }
      *reinterpret_cast<vec_t*>(
          output + static_cast<int64_t>(token_idx) * hidden_size + hidden_idx) =
          out;
    }
  }

  T* output;
  const T* moe_output;
  const float* topk_weights;
  const int* unpermuted_row_to_permuted_row;
  const int num_experts;
  const int num_tokens;
  const int hidden_size;
  const int lanes_per_token;
};

template <typename T>
void MoeGatherLauncher(
    T* output,
    const T* moe_output,
    const float* topk_weights,
    const int* unpermuted_row_to_permuted_row,
    const int num_experts,
    const int num_tokens,
    const int topk,
    const int hidden_size,
    sycl::queue& queue) {
  if (num_tokens == 0 || hidden_size == 0) {
    return;
  }
  // Start at 16-byte vectors; halve the width until hidden_size is a multiple
  // of it and output / moe_output are aligned to it.
  int elems_per_item = sizeof(float) * 4 / sizeof(T);
  while (
      hidden_size % elems_per_item != 0 ||
      reinterpret_cast<uintptr_t>(output) % (elems_per_item * sizeof(T)) != 0 ||
      reinterpret_cast<uintptr_t>(moe_output) % (elems_per_item * sizeof(T)) !=
          0) {
    elems_per_item /= 2;
  }
#define CASE_TOPK(TOPK, ElemsPerItem)                                       \
  case TOPK: {                                                              \
    const int lanes_per_token =                                             \
        MoeGather<T, TOPK, ElemsPerItem>::get_lanes_per_token(hidden_size); \
    queue.submit([&](sycl::handler& cgh) {                                  \
      cgh.parallel_for(                                                     \
          MoeGather<T, TOPK, ElemsPerItem>::get_nd_range(                   \
              num_tokens, lanes_per_token),                                 \
          MoeGather<T, TOPK, ElemsPerItem>{                                 \
              output,                                                       \
              moe_output,                                                   \
              topk_weights,                                                 \
              unpermuted_row_to_permuted_row,                               \
              num_experts,                                                  \
              num_tokens,                                                   \
              hidden_size,                                                  \
              lanes_per_token});                                            \
    });                                                                     \
  } break;

#define CASE_ElemsPerItem(TOPK, ElemsPerItem)                                  \
  case ElemsPerItem:                                                           \
    switch (TOPK) {                                                            \
      CASE_TOPK(1, ElemsPerItem)                                               \
      CASE_TOPK(2, ElemsPerItem)                                               \
      CASE_TOPK(3, ElemsPerItem)                                               \
      CASE_TOPK(4, ElemsPerItem)                                               \
      CASE_TOPK(6, ElemsPerItem)                                               \
      CASE_TOPK(7, ElemsPerItem)                                               \
      CASE_TOPK(8, ElemsPerItem)                                               \
      CASE_TOPK(10, ElemsPerItem)                                              \
      CASE_TOPK(12, ElemsPerItem)                                              \
      CASE_TOPK(16, ElemsPerItem)                                              \
      default:                                                                 \
        TORCH_CHECK(false, "error: not support TOPK=" + std::to_string(TOPK)); \
    }                                                                          \
    break;

  switch (elems_per_item) {
    CASE_ElemsPerItem(topk, 1) CASE_ElemsPerItem(topk, 2)
        CASE_ElemsPerItem(topk, 4) CASE_ElemsPerItem(topk, 8) default
        : TORCH_CHECK(
              false,
              "error: not support elems_per_item=" +
                  std::to_string(elems_per_item));
  }
#undef CASE_ElemsPerItem
#undef CASE_TOPK
}

}  // namespace moe
}  // namespace vllm

void moe_gather(
    torch::Tensor& output,              // [num_tokens, hidden_size]
    const torch::Tensor& moe_output,    // [num_tokens * topk, hidden_size]
    const torch::Tensor& topk_weights,  // [num_tokens, topk]
    const torch::Tensor& unpermuted_row_to_permuted_row,  // [num_tokens * topk]
    const int64_t num_experts) {
  // Implementation of the gather operation
  const int num_tokens = topk_weights.size(0);
  const int topk = topk_weights.size(1);
  const int hidden_size = output.size(1);

  TORCH_CHECK(
      topk_weights.scalar_type() == torch::kFloat32,
      "topk_weights must be float32");

  const at::DeviceGuard device_guard(output.device());
  auto& queue = vllm::xpu::vllmGetQueue();

#define LAUNCH_MOE_GATHER(T)                                             \
  vllm::moe::MoeGatherLauncher<T>(                                       \
      reinterpret_cast<T*>(output.data_ptr()),                           \
      reinterpret_cast<T*>(moe_output.data_ptr()),                       \
      reinterpret_cast<float*>(topk_weights.data_ptr()),                 \
      reinterpret_cast<int*>(unpermuted_row_to_permuted_row.data_ptr()), \
      num_experts,                                                       \
      num_tokens,                                                        \
      topk,                                                              \
      hidden_size,                                                       \
      queue);

  if (output.scalar_type() == torch::kFloat16) {
    using scalar_t = sycl::half;
    LAUNCH_MOE_GATHER(scalar_t);
  } else if (output.scalar_type() == torch::kBFloat16) {
    using scalar_t = sycl::ext::oneapi::bfloat16;
    LAUNCH_MOE_GATHER(scalar_t);
  } else if (output.scalar_type() == torch::kFloat32) {
    using scalar_t = float;
    LAUNCH_MOE_GATHER(scalar_t);
  } else {
    throw std::runtime_error("Unsupported data type in moe_gather");
  }
#undef LAUNCH_MOE_GATHER
}