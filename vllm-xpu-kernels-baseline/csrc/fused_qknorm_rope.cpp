/*
 * Ported from the CUDA implementation in csrc/fused_qknorm_rope_kernel.cu.
 */

#include <sycl/sycl.hpp>

#include <ATen/DeviceGuard.h>
#include <cmath>

#include "dispatch_utils.h"
#include "utils.h"

namespace vllm {

template <
    typename scalar_t,
    typename scalar_t_cache,
    int head_dim,
    bool IS_NEOX>
class fused_qk_norm_rope_kernel {
 public:
  static constexpr int SUB_GROUP_SIZE = 32;
  static constexpr int NUM_ELEMS_PER_THREAD = head_dim / SUB_GROUP_SIZE;

  static_assert(
      head_dim % (SUB_GROUP_SIZE * 2) == 0, "head_dim must be divisible by 64");

  fused_qk_norm_rope_kernel(
      scalar_t* __restrict__ qkv_,
      const int num_heads_q_,
      const int num_heads_k_,
      const int num_heads_v_,
      const float eps_,
      const scalar_t* __restrict__ q_weight_,
      const scalar_t* __restrict__ k_weight_,
      const scalar_t_cache* __restrict__ cos_sin_cache_,
      const int64_t* __restrict__ position_ids_,
      const int num_tokens_,
      const int rotary_dim_)
      : qkv(qkv_),
        num_heads_q(num_heads_q_),
        num_heads_k(num_heads_k_),
        num_heads_v(num_heads_v_),
        eps(eps_),
        q_weight(q_weight_),
        k_weight(k_weight_),
        cos_sin_cache(cos_sin_cache_),
        position_ids(position_ids_),
        num_tokens(num_tokens_),
        rotary_dim(rotary_dim_) {}

  void operator() [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] (
      const sycl::nd_item<1>& item) const {
    auto sg = item.get_sub_group();
    const int lane_id = sg.get_local_linear_id();
    const int sg_id_in_wg = sg.get_group_linear_id();
    const int sgs_per_wg = sg.get_group_linear_range();
    const int global_sg_id = item.get_group(0) * sgs_per_wg + sg_id_in_wg;

    const int total_qk_heads = num_heads_q + num_heads_k;
    const int token_idx = global_sg_id / total_qk_heads;
    const int local_head_idx = global_sg_id % total_qk_heads;

    if (token_idx >= num_tokens) return;

    const bool is_q = local_head_idx < num_heads_q;
    const int head_idx = is_q ? local_head_idx : local_head_idx - num_heads_q;
    const int num_heads = num_heads_q + num_heads_k + num_heads_v;

    // Compute the offset into the QKV tensor for this sub-group's head.
    int offset_warp;
    if (is_q) {
      offset_warp = token_idx * num_heads * head_dim + head_idx * head_dim;
    } else {
      offset_warp = token_idx * num_heads * head_dim + num_heads_q * head_dim +
                    head_idx * head_dim;
    }
    int offset_thread = offset_warp + lane_id * NUM_ELEMS_PER_THREAD;

    // Load elements and compute sum of squares for RMSNorm.
    float elements[NUM_ELEMS_PER_THREAD];
    float sum_of_squares = 0.0f;

#pragma unroll
    for (int i = 0; i < NUM_ELEMS_PER_THREAD; i++) {
      float val = static_cast<float>(qkv[offset_thread + i]);
      elements[i] = val;
      sum_of_squares += val * val;
    }

    // Reduce sum across sub-group.
    sum_of_squares =
        sycl::reduce_over_group(sg, sum_of_squares, sycl::plus<float>());

    // Compute RMS normalization factor.
    float rms_rcp =
        sycl::rsqrt(sum_of_squares / static_cast<float>(head_dim) + eps);

    // Apply RMSNorm with learned weights.
    const scalar_t* weight = is_q ? q_weight : k_weight;
#pragma unroll
    for (int i = 0; i < NUM_ELEMS_PER_THREAD; i++) {
      int dim = lane_id * NUM_ELEMS_PER_THREAD + i;
      float w = static_cast<float>(weight[dim]);
      elements[i] *= rms_rcp * w;
    }

    // Apply RoPE.
    const int64_t pos_id = position_ids[token_idx];
    const int embed_dim = rotary_dim / 2;
    const scalar_t_cache* cache_ptr = cos_sin_cache + pos_id * rotary_dim;
    const scalar_t_cache* cos_ptr = cache_ptr;
    const scalar_t_cache* sin_ptr = cache_ptr + embed_dim;

    const int rotary_lanes = rotary_dim / NUM_ELEMS_PER_THREAD;

    if (lane_id < rotary_lanes) {
      if constexpr (!IS_NEOX) {
        // Interleaved-style RoPE (GPT-J style).
#pragma unroll
        for (int i = 0; i < NUM_ELEMS_PER_THREAD / 2; ++i) {
          const int idx0 = 2 * i;
          const int idx1 = 2 * i + 1;
          const int dim_idx = lane_id * NUM_ELEMS_PER_THREAD + idx0;

          const float val0 = elements[idx0];
          const float val1 = elements[idx1];

          const int half_dim = dim_idx / 2;
          const float cos_val = static_cast<float>(cos_ptr[half_dim]);
          const float sin_val = static_cast<float>(sin_ptr[half_dim]);

          elements[idx0] = val0 * cos_val - val1 * sin_val;
          elements[idx1] = val0 * sin_val + val1 * cos_val;
        }
      } else {
        // Neox-style RoPE: exchange data with partner lane.
        sycl::group_barrier(sg);
        const int pair_offset = (rotary_dim / 2) / NUM_ELEMS_PER_THREAD;

#pragma unroll
        for (int i = 0; i < NUM_ELEMS_PER_THREAD; i++) {
          float partner_val =
              sycl::permute_group_by_xor(sg, elements[i], pair_offset);

          if (lane_id < pair_offset) {
            partner_val = -partner_val;
          }

          int dim_idx = lane_id * NUM_ELEMS_PER_THREAD + i;
          dim_idx = (dim_idx * 2) % rotary_dim;
          int half_dim = dim_idx / 2;

          const float cos_val = static_cast<float>(cos_ptr[half_dim]);
          const float sin_val = static_cast<float>(sin_ptr[half_dim]);

          elements[i] = elements[i] * cos_val + partner_val * sin_val;
        }
        sycl::group_barrier(sg);
      }
    }

    // Store results back in-place.
#pragma unroll
    for (int i = 0; i < NUM_ELEMS_PER_THREAD; i++) {
      qkv[offset_thread + i] = static_cast<scalar_t>(elements[i]);
    }
  }

 private:
  scalar_t* __restrict__ qkv;
  const int num_heads_q;
  const int num_heads_k;
  const int num_heads_v;
  const float eps;
  const scalar_t* __restrict__ q_weight;
  const scalar_t* __restrict__ k_weight;
  const scalar_t_cache* __restrict__ cos_sin_cache;
  const int64_t* __restrict__ position_ids;
  const int num_tokens;
  const int rotary_dim;
};

template <typename scalar_t, typename scalar_t_cache>
void launch_fused_qk_norm_rope(
    torch::Tensor& qkv,
    int num_tokens,
    int num_heads_q,
    int num_heads_k,
    int num_heads_v,
    int head_dim,
    int rotary_dim,
    float eps,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight,
    torch::Tensor& cos_sin_cache,
    bool is_neox,
    torch::Tensor& position_ids) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  using sycl_cache_t = typename vllm::xpu::SyclTypeTrait<scalar_t_cache>::Type;

  constexpr int block_size = 256;
  constexpr int sgs_per_wg = block_size / 32;

  const int total_qk_heads = num_heads_q + num_heads_k;
  const int total_sgs = num_tokens * total_qk_heads;
  const int grid_size = (total_sgs + sgs_per_wg - 1) / sgs_per_wg;

  auto qkv_ptr = reinterpret_cast<sycl_t*>(qkv.data_ptr<scalar_t>());
  auto q_weight_ptr =
      reinterpret_cast<const sycl_t*>(q_weight.data_ptr<scalar_t>());
  auto k_weight_ptr =
      reinterpret_cast<const sycl_t*>(k_weight.data_ptr<scalar_t>());
  auto cos_sin_cache_ptr = reinterpret_cast<const sycl_cache_t*>(
      cos_sin_cache.data_ptr<scalar_t_cache>());
  auto position_ids_ptr = position_ids.data_ptr<int64_t>();

  auto& queue = vllm::xpu::vllmGetQueue();

#define DISPATCH_HEAD_DIM(HEAD_DIM)                                           \
  do {                                                                        \
    if (is_neox) {                                                            \
      queue.submit([&](sycl::handler& cgh) {                                  \
        cgh.parallel_for(                                                     \
            sycl::nd_range<1>(grid_size * block_size, block_size),            \
            fused_qk_norm_rope_kernel<sycl_t, sycl_cache_t, HEAD_DIM, true>(  \
                qkv_ptr,                                                      \
                num_heads_q,                                                  \
                num_heads_k,                                                  \
                num_heads_v,                                                  \
                eps,                                                          \
                q_weight_ptr,                                                 \
                k_weight_ptr,                                                 \
                cos_sin_cache_ptr,                                            \
                position_ids_ptr,                                             \
                num_tokens,                                                   \
                rotary_dim));                                                 \
      });                                                                     \
    } else {                                                                  \
      queue.submit([&](sycl::handler& cgh) {                                  \
        cgh.parallel_for(                                                     \
            sycl::nd_range<1>(grid_size * block_size, block_size),            \
            fused_qk_norm_rope_kernel<sycl_t, sycl_cache_t, HEAD_DIM, false>( \
                qkv_ptr,                                                      \
                num_heads_q,                                                  \
                num_heads_k,                                                  \
                num_heads_v,                                                  \
                eps,                                                          \
                q_weight_ptr,                                                 \
                k_weight_ptr,                                                 \
                cos_sin_cache_ptr,                                            \
                position_ids_ptr,                                             \
                num_tokens,                                                   \
                rotary_dim));                                                 \
      });                                                                     \
    }                                                                         \
  } while (0)

  switch (head_dim) {
    case 64:
      DISPATCH_HEAD_DIM(64);
      break;
    case 128:
      DISPATCH_HEAD_DIM(128);
      break;
    case 256:
      DISPATCH_HEAD_DIM(256);
      break;
    case 512:
      DISPATCH_HEAD_DIM(512);
      break;
    default:
      TORCH_CHECK(
          false,
          "Unsupported head dimension for fused_qk_norm_rope: ",
          head_dim);
  }

#undef DISPATCH_HEAD_DIM
}

}  // namespace vllm

void fused_qk_norm_rope(
    torch::Tensor& qkv,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight,
    torch::Tensor& cos_sin_cache,
    bool is_neox,
    torch::Tensor& position_ids) {
  const at::DeviceGuard device_guard(qkv.device());

  CHECK_DEVICE(qkv);
  CHECK_CONTIGUOUS(qkv);
  CHECK_DEVICE(position_ids);
  CHECK_CONTIGUOUS(position_ids);
  CHECK_DEVICE(q_weight);
  CHECK_CONTIGUOUS(q_weight);
  CHECK_DEVICE(k_weight);
  CHECK_CONTIGUOUS(k_weight);
  CHECK_DEVICE(cos_sin_cache);
  CHECK_CONTIGUOUS(cos_sin_cache);

  TORCH_CHECK(
      position_ids.scalar_type() == torch::kInt64,
      "position_ids must be int64");
  TORCH_CHECK(
      qkv.dim() == 2,
      "QKV tensor must be 2D: [num_tokens, "
      "(num_heads_q+num_heads_k+num_heads_v)*head_dim]");
  TORCH_CHECK(position_ids.dim() == 1, "Position IDs must be 1D: [num_tokens]");
  TORCH_CHECK(q_weight.dim() == 1, "Query weights must be 1D: [head_dim]");
  TORCH_CHECK(k_weight.dim() == 1, "Key weights must be 1D: [head_dim]");
  TORCH_CHECK(
      cos_sin_cache.dim() == 2,
      "Cos/sin cache must be 2D: [max_position, rotary_dim]");
  TORCH_CHECK(
      q_weight.size(0) == head_dim,
      "Query weights size must match head dimension");
  TORCH_CHECK(
      k_weight.size(0) == head_dim,
      "Key weights size must match head dimension");
  TORCH_CHECK(cos_sin_cache.size(1) % 2 == 0, "rotary_dim must be even");
  TORCH_CHECK(
      cos_sin_cache.size(1) <= head_dim,
      "rotary_dim must be less than or equal to head_dim");
  TORCH_CHECK(
      qkv.scalar_type() == q_weight.scalar_type() &&
          qkv.scalar_type() == k_weight.scalar_type(),
      "qkv, q_weight and k_weight must have the same dtype");

  int64_t num_tokens = qkv.size(0);
  TORCH_CHECK(
      position_ids.size(0) == num_tokens,
      "Number of tokens in position_ids must match QKV");

  int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
  TORCH_CHECK(
      qkv.size(1) == total_heads * head_dim,
      "QKV tensor size must match total number of heads and head dimension");

  VLLM_DISPATCH_HALF_TYPES(qkv.scalar_type(), "fused_qk_norm_rope", [&] {
    using qkv_scalar_t = scalar_t;
    VLLM_DISPATCH_FLOATING_TYPES(
        cos_sin_cache.scalar_type(), "fused_qk_norm_rope_cache", [&] {
          using cache_scalar_t = scalar_t;
          vllm::launch_fused_qk_norm_rope<qkv_scalar_t, cache_scalar_t>(
              qkv,
              static_cast<int>(num_tokens),
              static_cast<int>(num_heads_q),
              static_cast<int>(num_heads_k),
              static_cast<int>(num_heads_v),
              static_cast<int>(head_dim),
              static_cast<int>(cos_sin_cache.size(1)),
              static_cast<float>(eps),
              q_weight,
              k_weight,
              cos_sin_cache,
              is_neox,
              position_ids);
        });
  });
}
