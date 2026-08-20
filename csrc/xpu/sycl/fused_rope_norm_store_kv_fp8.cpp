/*
 * Fused QK-Norm + RoPE + KV-Cache Write + FP8 Q Quantization kernel.
 *
 * Single kernel dispatch fuses 5 operations:
 * 1) RMSNorm on Q and K heads (optional, policy-controlled)
 * 2) NeoX-style RoPE on Q and K
 * 3) Write K, V into paged KV cache (FP8 E4M3, per-tensor static scale)
 * 4) Quantize Q to FP8 E4M3 per-token-per-head (dynamic scale)
 *
 * Design: 1 sub-group = 1 (token, KV-group).
 * Each sub-group loops over Q_PER_KV Q heads + 1 K head + 1 V head.
 * cos/sin cache and norm weights are staged in SLM to avoid redundant
 * global memory reads across the Q_PER_KV+1 head iterations.
 *
 * KV cache layout:
 *   key_cache / value_cache: [num_blocks, block_size, num_kv_heads, head_dim]
 *   Separate K and V tensors with BSHD layout (stride-based access).
 */

#include <sycl/sycl.hpp>

#include <ATen/DeviceGuard.h>
#include <c10/util/Float8_e4m3fn.h>
#include <cmath>

#include "dispatch_utils.h"
#include "utils.h"

namespace vllm {

constexpr float kFp8E4m3Max = 448.0f;

enum class QkNormPolicy : int {
  NONE = 0,
  ROPE_THEN_NORM = 1,
  NORM_THEN_ROPE = 2,
};

template <typename scalar_t, int HEAD_DIM, bool USE_FP8, int Q_PER_KV>
class fused_rope_norm_store_kv_fp8_kernel {
 public:
  static constexpr int SG_SIZE = 32;
  static constexpr int ELEMS = HEAD_DIM / SG_SIZE;  // 4 for HEAD_DIM=128

  static_assert(
      HEAD_DIM % (SG_SIZE * 2) == 0, "HEAD_DIM must be divisible by 64");

  fused_rope_norm_store_kv_fp8_kernel(
      const scalar_t* __restrict__ qkv,
      uint8_t* __restrict__ out_q,
      float* __restrict__ q_scale_out,
      const float* __restrict__ cos_sin_cache,
      const int64_t* __restrict__ positions,
      uint8_t* __restrict__ k_cache,
      uint8_t* __restrict__ v_cache,
      const int64_t* __restrict__ slot_mapping,
      const int num_q_heads,
      const int num_kv_heads,
      const int num_tokens,
      const float* __restrict__ q_norm_weight,
      const float* __restrict__ k_norm_weight,
      const int qk_norm_policy,
      const float* __restrict__ k_scale,
      const float* __restrict__ v_scale,
      const int block_size,
      const int64_t kv_page_stride,
      const int64_t kv_seq_stride,
      const int64_t kv_head_stride,
      sycl::local_accessor<float, 1> slm)
      : qkv_(qkv),
        out_q_(out_q),
        q_scale_out_(q_scale_out),
        cos_sin_cache_(cos_sin_cache),
        positions_(positions),
        k_cache_(k_cache),
        v_cache_(v_cache),
        slot_mapping_(slot_mapping),
        num_q_heads_(num_q_heads),
        num_kv_heads_(num_kv_heads),
        num_tokens_(num_tokens),
        q_norm_weight_(q_norm_weight),
        k_norm_weight_(k_norm_weight),
        qk_norm_policy_(qk_norm_policy),
        k_scale_(k_scale),
        v_scale_(v_scale),
        block_size_(block_size),
        kv_page_stride_(kv_page_stride),
        kv_seq_stride_(kv_seq_stride),
        kv_head_stride_(kv_head_stride),
        slm_(slm) {}

  void operator() [[sycl::reqd_sub_group_size(SG_SIZE)]] (
      const sycl::nd_item<1>& item) const {
    auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int sg_in_wg = sg.get_group_linear_id();
    const int sgs_per_wg = sg.get_group_linear_range();
    const int global_sg = item.get_group(0) * sgs_per_wg + sg_in_wg;

    const int token_idx = global_sg / num_kv_heads_;
    const int kv_head = global_sg % num_kv_heads_;

    const bool valid = (token_idx < num_tokens_);

    float* my_cos_sin = slm_.get_pointer() + sg_in_wg * HEAD_DIM;
    float* slm_q_norm = slm_.get_pointer() + sgs_per_wg * HEAD_DIM;
    float* slm_k_norm = slm_q_norm + HEAD_DIM;

    const int lane_offset = lane * ELEMS;
    if (valid) {
      const int64_t pos_id = positions_[token_idx];
      const float* cs_global = cos_sin_cache_ + pos_id * HEAD_DIM;
#pragma unroll
      for (int i = 0; i < ELEMS; i++) {
        my_cos_sin[lane_offset + i] = cs_global[lane_offset + i];
      }
    }

    if (sg_in_wg == 0 &&
        qk_norm_policy_ != static_cast<int>(QkNormPolicy::NONE)) {
      if (q_norm_weight_ != nullptr) {
#pragma unroll
        for (int i = 0; i < ELEMS; i++) {
          slm_q_norm[lane_offset + i] = q_norm_weight_[lane_offset + i];
        }
      }
      if (k_norm_weight_ != nullptr) {
#pragma unroll
        for (int i = 0; i < ELEMS; i++) {
          slm_k_norm[lane_offset + i] = k_norm_weight_[lane_offset + i];
        }
      }
    }

    sycl::group_barrier(item.get_group());

    // Early exit for invalid sub-groups (AFTER barrier).
    if (!valid) return;

    // ---- Pre-compute KV cache destination (BSHD byte offset) ----
    const int64_t slot_idx = slot_mapping_[token_idx];
    const bool valid_slot = slot_idx >= 0;
    int64_t cache_offset = 0;
    if (valid_slot) {
      const int block_idx = slot_idx / block_size_;
      const int block_offset = slot_idx % block_size_;
      cache_offset = static_cast<int64_t>(block_idx) * kv_page_stride_ +
                     static_cast<int64_t>(block_offset) * kv_seq_stride_ +
                     static_cast<int64_t>(kv_head) * kv_head_stride_;
    }

    // Pre-compute QKV offsets.
    const int q_size = num_q_heads_ * HEAD_DIM;
    const int kv_size = num_kv_heads_ * HEAD_DIM;
    const int qkv_stride = q_size + 2 * kv_size;
    const int qkv_token_base = token_idx * qkv_stride;
    const int q_head_start = kv_head * Q_PER_KV;

    const int half_dim = HEAD_DIM / 2;
    const float* cos_ptr = my_cos_sin;
    const float* sin_ptr = my_cos_sin + half_dim;

    // ==========================================================
    // Process Q_PER_KV Q heads: Norm → RoPE → FP8 quant → store
    // ==========================================================
#pragma unroll
    for (int qh = 0; qh < Q_PER_KV; qh++) {
      const int head = q_head_start + qh;

      // Load Q head from global memory.
      float elems[ELEMS];
      const int q_offset = qkv_token_base + head * HEAD_DIM + lane_offset;
#pragma unroll
      for (int i = 0; i < ELEMS; i++) {
        elems[i] = static_cast<float>(qkv_[q_offset + i]);
      }

      // Norm before RoPE.
      if (qk_norm_policy_ == static_cast<int>(QkNormPolicy::NORM_THEN_ROPE)) {
        apply_rms_norm(sg, lane, elems, slm_q_norm);
      }

      // RoPE.
      apply_rope_neox(sg, lane, elems, cos_ptr, sin_ptr);

      // Norm after RoPE.
      if (qk_norm_policy_ == static_cast<int>(QkNormPolicy::ROPE_THEN_NORM)) {
        apply_rms_norm(sg, lane, elems, slm_q_norm);
      }

      // FP8 quantize Q and store.
      store_q(sg, lane, elems, token_idx, head);
    }

    // ==========================================================
    // Process K head: Norm → RoPE → FP8 quant → store to KV cache
    // ==========================================================
    {
      float elems[ELEMS];
      const int k_offset =
          qkv_token_base + q_size + kv_head * HEAD_DIM + lane_offset;
#pragma unroll
      for (int i = 0; i < ELEMS; i++) {
        elems[i] = static_cast<float>(qkv_[k_offset + i]);
      }

      if (qk_norm_policy_ == static_cast<int>(QkNormPolicy::NORM_THEN_ROPE)) {
        apply_rms_norm(sg, lane, elems, slm_k_norm);
      }

      apply_rope_neox(sg, lane, elems, cos_ptr, sin_ptr);

      if (qk_norm_policy_ == static_cast<int>(QkNormPolicy::ROPE_THEN_NORM)) {
        apply_rms_norm(sg, lane, elems, slm_k_norm);
      }

      // Write K to cache.
      if (valid_slot) {
        store_kv_to_cache(lane, elems, k_cache_, cache_offset, *k_scale_);
      }
    }

    // ==========================================================
    // Process V head: FP8 quant → store to KV cache (no norm/RoPE)
    // ==========================================================
    {
      float elems[ELEMS];
      const int v_offset =
          qkv_token_base + q_size + kv_size + kv_head * HEAD_DIM + lane_offset;
#pragma unroll
      for (int i = 0; i < ELEMS; i++) {
        elems[i] = static_cast<float>(qkv_[v_offset + i]);
      }

      if (valid_slot) {
        store_kv_to_cache(lane, elems, v_cache_, cache_offset, *v_scale_);
      }
    }
  }

 private:
  void apply_rms_norm(
      const sycl::sub_group& sg,
      int lane,
      float* elems,
      const float* norm_weight) const {
    float sum_sq = 0.0f;
#pragma unroll
    for (int i = 0; i < ELEMS; i++) {
      sum_sq += elems[i] * elems[i];
    }
    sum_sq = sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());
    float rms_rcp = sycl::rsqrt(sum_sq / static_cast<float>(HEAD_DIM) + 1e-6f);

#pragma unroll
    for (int i = 0; i < ELEMS; i++) {
      elems[i] *= rms_rcp * norm_weight[lane * ELEMS + i];
    }
  }

  void apply_rope_neox(
      const sycl::sub_group& sg,
      int lane,
      float* elems,
      const float* cos_ptr,
      const float* sin_ptr) const {
    constexpr int pair_offset = (HEAD_DIM / 2) / ELEMS;

    sycl::group_barrier(sg);
#pragma unroll
    for (int i = 0; i < ELEMS; i++) {
      float partner = sycl::permute_group_by_xor(sg, elems[i], pair_offset);
      if (lane < pair_offset) {
        partner = -partner;
      }
      int dim_idx = lane * ELEMS + i;
      dim_idx = (dim_idx * 2) % HEAD_DIM;
      int half_idx = dim_idx / 2;
      elems[i] = elems[i] * cos_ptr[half_idx] + partner * sin_ptr[half_idx];
    }
    sycl::group_barrier(sg);
  }

  /// Store Q: FP8 per-token-per-head dynamic quant, or bf16.
  void store_q(
      const sycl::sub_group& sg,
      int lane,
      float* elems,
      int token_idx,
      int head_idx) const {
    if constexpr (USE_FP8) {
      float local_max = 0.0f;
#pragma unroll
      for (int i = 0; i < ELEMS; i++) {
        local_max = sycl::fmax(local_max, sycl::fabs(elems[i]));
      }
      float absmax =
          sycl::reduce_over_group(sg, local_max, sycl::maximum<float>());
      float scale = sycl::fmax(absmax / kFp8E4m3Max, 1e-12f);
      float inv_scale = 1.0f / scale;

      if (lane == 0) {
        q_scale_out_[token_idx * num_q_heads_ + head_idx] = scale;
      }

      const int out_off =
          (token_idx * num_q_heads_ + head_idx) * HEAD_DIM + lane * ELEMS;
      uint32_t packed = 0;
#pragma unroll
      for (int i = 0; i < ELEMS; i++) {
        float v = sycl::fmin(
            sycl::fmax(elems[i] * inv_scale, -kFp8E4m3Max), kFp8E4m3Max);
        uint8_t byte =
            sycl::bit_cast<uint8_t>(static_cast<at::Float8_e4m3fn>(v));
        packed |= (static_cast<uint32_t>(byte) << (i * 8));
      }
      *reinterpret_cast<uint32_t*>(out_q_ + out_off) = packed;
    } else {
      const int out_off =
          (token_idx * num_q_heads_ + head_idx) * HEAD_DIM + lane * ELEMS;
      auto* dst = reinterpret_cast<scalar_t*>(out_q_) + out_off;
#pragma unroll
      for (int i = 0; i < ELEMS; i++) {
        dst[i] = static_cast<scalar_t>(elems[i]);
      }
    }
  }

  void store_kv_to_cache(
      int lane,
      float* elems,
      uint8_t* cache_ptr,
      int64_t base_offset,
      float scale) const {
    constexpr int KV_ELEM_SZ = USE_FP8 ? 1 : static_cast<int>(sizeof(scalar_t));
    uint8_t* dst = cache_ptr + base_offset + lane * ELEMS * KV_ELEM_SZ;

    if constexpr (USE_FP8) {
      float inv_scale = 1.0f / scale;
      uint32_t packed = 0;
#pragma unroll
      for (int i = 0; i < ELEMS; i++) {
        float v = sycl::fmin(
            sycl::fmax(elems[i] * inv_scale, -kFp8E4m3Max), kFp8E4m3Max);
        uint8_t byte =
            sycl::bit_cast<uint8_t>(static_cast<at::Float8_e4m3fn>(v));
        packed |= (static_cast<uint32_t>(byte) << (i * 8));
      }
      *reinterpret_cast<uint32_t*>(dst) = packed;
    } else {
      auto* bf_dst = reinterpret_cast<scalar_t*>(dst);
#pragma unroll
      for (int i = 0; i < ELEMS; i++) {
        bf_dst[i] = static_cast<scalar_t>(elems[i]);
      }
    }
  }

  const scalar_t* __restrict__ qkv_;
  uint8_t* __restrict__ out_q_;
  float* __restrict__ q_scale_out_;
  const float* __restrict__ cos_sin_cache_;
  const int64_t* __restrict__ positions_;
  uint8_t* __restrict__ k_cache_;
  uint8_t* __restrict__ v_cache_;
  const int64_t* __restrict__ slot_mapping_;
  const int num_q_heads_;
  const int num_kv_heads_;
  const int num_tokens_;
  const float* __restrict__ q_norm_weight_;
  const float* __restrict__ k_norm_weight_;
  const int qk_norm_policy_;
  const float* __restrict__ k_scale_;
  const float* __restrict__ v_scale_;
  const int block_size_;
  const int64_t kv_page_stride_;
  const int64_t kv_seq_stride_;
  const int64_t kv_head_stride_;
  sycl::local_accessor<float, 1> slm_;
};

template <typename scalar_t>
void launch_fused_rope_norm_store_kv_fp8(
    const torch::Tensor& qkv,
    torch::Tensor& out_q,
    torch::Tensor& q_scale_out,
    const torch::Tensor& cos_sin_cache,
    const torch::Tensor& positions,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    const torch::Tensor& slot_mapping,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    const float* k_scale,
    const float* v_scale,
    const float* q_norm_weight_ptr,
    const float* k_norm_weight_ptr,
    int qk_norm_policy,
    bool use_fp8) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  const int num_tokens = qkv.size(0);
  const int block_size = key_cache.size(1);
  // Byte strides for BSHD layout [num_blocks, block_size, num_kv_heads,
  // head_dim]
  const int64_t elem_size = key_cache.element_size();
  const int64_t kv_page_stride = key_cache.stride(0) * elem_size;
  const int64_t kv_seq_stride = key_cache.stride(1) * elem_size;
  const int64_t kv_head_stride = key_cache.stride(2) * elem_size;

  auto qkv_ptr = reinterpret_cast<const sycl_t*>(qkv.data_ptr<scalar_t>());
  auto out_q_ptr = reinterpret_cast<uint8_t*>(out_q.data_ptr());
  auto q_scale_ptr = q_scale_out.data_ptr<float>();
  auto cos_sin_ptr = cos_sin_cache.data_ptr<float>();
  auto positions_ptr = positions.data_ptr<int64_t>();
  auto k_cache_ptr = reinterpret_cast<uint8_t*>(key_cache.data_ptr());
  auto v_cache_ptr = reinterpret_cast<uint8_t*>(value_cache.data_ptr());
  auto slot_mapping_ptr = slot_mapping.data_ptr<int64_t>();

  auto& queue = vllm::xpu::vllmGetQueue();

  const int q_per_kv = num_q_heads / num_kv_heads;
  constexpr int WG_SIZE = 256;
  constexpr int SGS_PER_WG = WG_SIZE / 32;
  const int total_sgs = num_tokens * num_kv_heads;
  const int grid_sz = (total_sgs + SGS_PER_WG - 1) / SGS_PER_WG;
  const bool use_norm = qk_norm_policy != 0;
  const int slm_size = SGS_PER_WG * head_dim + (use_norm ? 2 * head_dim : 0);

#define LAUNCH(USE_FP8_V, Q_PER_KV_V)                                  \
  queue.submit([&](sycl::handler& cgh) {                               \
    sycl::local_accessor<float, 1> slm(sycl::range<1>(slm_size), cgh); \
    cgh.parallel_for(                                                  \
        sycl::nd_range<1>(grid_sz * WG_SIZE, WG_SIZE),                 \
        fused_rope_norm_store_kv_fp8_kernel<                           \
            sycl_t,                                                    \
            128,                                                       \
            USE_FP8_V,                                                 \
            Q_PER_KV_V>(                                               \
            qkv_ptr,                                                   \
            out_q_ptr,                                                 \
            q_scale_ptr,                                               \
            cos_sin_ptr,                                               \
            positions_ptr,                                             \
            k_cache_ptr,                                               \
            v_cache_ptr,                                               \
            slot_mapping_ptr,                                          \
            num_q_heads,                                               \
            num_kv_heads,                                              \
            num_tokens,                                                \
            q_norm_weight_ptr,                                         \
            k_norm_weight_ptr,                                         \
            qk_norm_policy,                                            \
            k_scale,                                                   \
            v_scale,                                                   \
            block_size,                                                \
            kv_page_stride,                                            \
            kv_seq_stride,                                             \
            kv_head_stride,                                            \
            slm));                                                     \
  })

  if (use_fp8) {
    if (q_per_kv == 8) {
      LAUNCH(true, 8);
    } else if (q_per_kv == 4) {
      LAUNCH(true, 4);
    } else if (q_per_kv == 16) {
      LAUNCH(true, 16);
    } else {
      TORCH_CHECK(false, "Unsupported GQA ratio: ", q_per_kv);
    }
  } else {
    if (q_per_kv == 8) {
      LAUNCH(false, 8);
    } else if (q_per_kv == 4) {
      LAUNCH(false, 4);
    } else {
      TORCH_CHECK(false, "Unsupported GQA ratio: ", q_per_kv);
    }
  }
#undef LAUNCH
}

}  // namespace vllm

torch::Tensor fused_rope_norm_store_kv_fp8(
    torch::Tensor& qkv,
    const torch::Tensor& cos_sin_cache,
    const torch::Tensor& positions,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    const torch::Tensor& slot_mapping,
    torch::Tensor& out_q,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    const torch::Tensor& k_scale,
    const torch::Tensor& v_scale,
    std::optional<torch::Tensor> q_norm_weight,
    std::optional<torch::Tensor> k_norm_weight,
    int64_t qk_norm_policy,
    bool use_fp8) {
  const at::DeviceGuard device_guard(qkv.device());

  CHECK_DEVICE(qkv);
  CHECK_DEVICE(cos_sin_cache);
  CHECK_DEVICE(positions);
  CHECK_DEVICE(key_cache);
  CHECK_DEVICE(value_cache);
  CHECK_DEVICE(slot_mapping);
  CHECK_DEVICE(out_q);
  CHECK_DEVICE(k_scale);
  CHECK_DEVICE(v_scale);

  const int num_tokens = qkv.size(0);
  TORCH_CHECK(qkv.dim() == 2, "qkv must be 2D");
  TORCH_CHECK(
      head_dim == 128, "Only head_dim=128 is supported, got ", head_dim);
  TORCH_CHECK(
      key_cache.dim() == 4,
      "key_cache must be 4D [num_blocks, block_size, num_kv_heads, head_dim]");
  TORCH_CHECK(
      value_cache.dim() == 4,
      "value_cache must be 4D [num_blocks, block_size, num_kv_heads, "
      "head_dim]");
  TORCH_CHECK(
      cos_sin_cache.scalar_type() == torch::kFloat32,
      "cos_sin_cache must be float32");
  TORCH_CHECK(
      positions.scalar_type() == torch::kInt64, "positions must be int64");

  auto q_scale_out = torch::empty(
      {num_tokens, num_q_heads},
      torch::TensorOptions().dtype(torch::kFloat32).device(qkv.device()));

  const float* k_scale_ptr = k_scale.data_ptr<float>();
  const float* v_scale_ptr = v_scale.data_ptr<float>();

  const float* q_norm_ptr = nullptr;
  const float* k_norm_ptr = nullptr;
  if (q_norm_weight.has_value() && q_norm_weight->defined()) {
    q_norm_ptr = q_norm_weight->data_ptr<float>();
  }
  if (k_norm_weight.has_value() && k_norm_weight->defined()) {
    k_norm_ptr = k_norm_weight->data_ptr<float>();
  }

  VLLM_DISPATCH_HALF_TYPES(
      qkv.scalar_type(), "fused_rope_norm_store_kv_fp8", [&] {
        vllm::launch_fused_rope_norm_store_kv_fp8<scalar_t>(
            qkv,
            out_q,
            q_scale_out,
            cos_sin_cache,
            positions,
            key_cache,
            value_cache,
            slot_mapping,
            static_cast<int>(num_q_heads),
            static_cast<int>(num_kv_heads),
            static_cast<int>(head_dim),
            k_scale_ptr,
            v_scale_ptr,
            q_norm_ptr,
            k_norm_ptr,
            static_cast<int>(qk_norm_policy),
            use_fp8);
      });

  return q_scale_out;
}
