// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Horizontally-fused MiniMax-M3 attention pre-processing kernel (SYCL port of
// csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu from
// vllm-project/vllm#45381, m3_release).
//
// Replaces the per-token Python sequence in the M3 attention forward:
//     q = q_norm(q); k = k_norm(k); q,k = rope(pos, q,k)
//     index_q = index_q_norm(index_q); index_k = index_k_norm(index_k)
//     index_q,index_k = rope(pos, index_q,index_k)
//     insert(k, v, index_k) into the paged caches
//
// All branches share head_dim=128 and the same partial-NeoX RoPE table
// (``rotary_dim`` leading dims rotated, the trailing dims pass through). The
// four norms are Gemma-style RMSNorm: ``x * rsqrt(mean(x^2)+eps) * (1+weight)``.
//
// The fused ``qkv`` packs, per token: [ q | k | v | index_q | index_k ] in the
// sparse layer, or just [ q | k | v ] in the dense layer (no index, no insert).
//
//   - q       : Gemma-norm + rope -> q_out (sparse) or in place in qkv (dense)
//   - k       : Gemma-norm + rope -> in place in qkv, scatter into kv_cache[:,0]
//   - v       : untouched,                          scatter into kv_cache[:,1]
//   - index_q : Gemma-norm + rope -> index_q_out (sparse)
//   - index_k : Gemma-norm + rope -> in place in qkv, scatter into index_cache
//
// One work-item handles one (token, head-slot); it loops over the 128-dim head.

#include <sycl/sycl.hpp>
#include "utils.h"
#include <torch/all.h>
#include <optional>

namespace vllm {

using bf16_t = sycl::ext::oneapi::bfloat16;

static constexpr int M3_HEAD_DIM = 128;

// Head-slot section ids (in processing order along the packed qkv tensor).
enum M3Section { SEC_Q = 0, SEC_K = 1, SEC_V = 2, SEC_IQ = 3, SEC_IK = 4 };

class fused_minimax_m3_qknorm_rope_kv_insert_kernel {
 public:
  fused_minimax_m3_qknorm_rope_kv_insert_kernel(
      bf16_t* __restrict__ qkv,
      const bf16_t* __restrict__ q_norm_w,
      const bf16_t* __restrict__ k_norm_w,
      const bf16_t* __restrict__ iq_norm_w,
      const bf16_t* __restrict__ ik_norm_w,
      const bf16_t* __restrict__ cos_sin_cache,
      const int64_t* __restrict__ positions,
      const int64_t* __restrict__ slot_mapping,
      const int64_t* __restrict__ index_slot_mapping,
      bf16_t* __restrict__ kv_cache,
      bf16_t* __restrict__ index_cache,
      bf16_t* __restrict__ q_out,
      bf16_t* __restrict__ index_q_out,
      int num_heads,
      int num_kv_heads,
      int num_index_heads,
      int rotary_dim,
      float eps,
      int block_size,
      int64_t qkv_row_stride,
      int64_t cos_sin_row_stride,
      bool do_insert)
      : qkv_(qkv),
        q_norm_w_(q_norm_w),
        k_norm_w_(k_norm_w),
        iq_norm_w_(iq_norm_w),
        ik_norm_w_(ik_norm_w),
        cos_sin_cache_(cos_sin_cache),
        positions_(positions),
        slot_mapping_(slot_mapping),
        index_slot_mapping_(index_slot_mapping),
        kv_cache_(kv_cache),
        index_cache_(index_cache),
        q_out_(q_out),
        index_q_out_(index_q_out),
        num_heads_(num_heads),
        num_kv_heads_(num_kv_heads),
        num_index_heads_(num_index_heads),
        rotary_dim_(rotary_dim),
        eps_(eps),
        block_size_(block_size),
        qkv_row_stride_(qkv_row_stride),
        cos_sin_row_stride_(cos_sin_row_stride),
        do_insert_(do_insert) {}

  void operator()(sycl::nd_item<2> item) const {
    const int token = item.get_global_id(0);
    const int slot = item.get_global_id(1);

    // Decode the head-slot -> (section, head index within section).
    const int q_end = num_heads_;
    const int k_end = q_end + num_kv_heads_;
    const int v_end = k_end + num_kv_heads_;
    const int iq_end = v_end + num_index_heads_;
    const int ik_end = iq_end + (num_index_heads_ > 0 ? 1 : 0);
    if (slot >= ik_end) return;

    int section, head;
    if (slot < q_end) {
      section = SEC_Q;
      head = slot;
    } else if (slot < k_end) {
      section = SEC_K;
      head = slot - q_end;
    } else if (slot < v_end) {
      section = SEC_V;
      head = slot - k_end;
    } else if (slot < iq_end) {
      section = SEC_IQ;
      head = slot - v_end;
    } else {
      section = SEC_IK;
      head = 0;
    }

    // Column offset of this head inside the packed qkv row.
    const int q_cols = num_heads_ * M3_HEAD_DIM;
    const int kv_cols = num_kv_heads_ * M3_HEAD_DIM;
    const int iq_cols = num_index_heads_ * M3_HEAD_DIM;
    int col;
    switch (section) {
      case SEC_Q: col = head * M3_HEAD_DIM; break;
      case SEC_K: col = q_cols + head * M3_HEAD_DIM; break;
      case SEC_V: col = q_cols + kv_cols + head * M3_HEAD_DIM; break;
      case SEC_IQ: col = q_cols + 2 * kv_cols + head * M3_HEAD_DIM; break;
      default: col = q_cols + 2 * kv_cols + iq_cols; break;  // SEC_IK
    }

    bf16_t* __restrict__ src = qkv_ + token * qkv_row_stride_ + col;

    // V is raw: no norm/rope. Only (optionally) scatter into kv_cache[:,1].
    if (section == SEC_V) {
      if (do_insert_ && kv_cache_ != nullptr) {
        insert_kv(token, src, head, /*kv_axis=*/1);
      }
      return;
    }

    const bf16_t* __restrict__ weight =
        section == SEC_Q ? q_norm_w_
        : section == SEC_K ? k_norm_w_
        : section == SEC_IQ ? iq_norm_w_ : ik_norm_w_;

    float buf[M3_HEAD_DIM];
    float sumsq = 0.0f;
#pragma unroll
    for (int d = 0; d < M3_HEAD_DIM; ++d) {
      float x = static_cast<float>(src[d]);
      buf[d] = x;
      sumsq += x * x;
    }
    // Gemma RMSNorm: x * rsqrt(mean(x^2)+eps) * (1 + weight).
    const float rms = sycl::rsqrt(sumsq / M3_HEAD_DIM + eps_);
#pragma unroll
    for (int d = 0; d < M3_HEAD_DIM; ++d) {
      buf[d] = buf[d] * rms * (1.0f + static_cast<float>(weight[d]));
    }

    // Partial NeoX RoPE on the leading rotary_dim dims; rest pass through.
    const int half = rotary_dim_ / 2;
    const bf16_t* cs = cos_sin_cache_ + positions_[token] * cos_sin_row_stride_;
    for (int i = 0; i < half; ++i) {
      const float cos = static_cast<float>(cs[i]);
      const float sin = static_cast<float>(cs[half + i]);
      const float x1 = buf[i];
      const float x2 = buf[half + i];
      buf[i] = x1 * cos - x2 * sin;
      buf[half + i] = x2 * cos + x1 * sin;
    }

    // Destination: q/index_q go to their gather buffers when provided
    // (sparse mode); k/index_k are rewritten in place; in dense mode q stays
    // in place too.
    bf16_t* __restrict__ dst;
    if (section == SEC_Q && q_out_ != nullptr) {
      dst = q_out_ + token * (int64_t)(num_heads_ * M3_HEAD_DIM) +
            head * M3_HEAD_DIM;
    } else if (section == SEC_IQ && index_q_out_ != nullptr) {
      dst = index_q_out_ + token * (int64_t)(num_index_heads_ * M3_HEAD_DIM) +
            head * M3_HEAD_DIM;
    } else {
      dst = src;
    }
#pragma unroll
    for (int d = 0; d < M3_HEAD_DIM; ++d) {
      dst[d] = static_cast<bf16_t>(buf[d]);
    }

    // Cache inserts (sparse mode only).
    if (do_insert_) {
      if (section == SEC_K && kv_cache_ != nullptr) {
        store_from_buf(buf, kv_slot_ptr(token, head, /*kv_axis=*/0));
      } else if (section == SEC_IK && index_cache_ != nullptr) {
        const int64_t s = index_slot_mapping_[token];
        if (s >= 0) {
          bf16_t* dstc = index_cache_ + s * M3_HEAD_DIM;
          store_from_buf(buf, dstc);
        }
      }
    }
  }

 private:
  // Pointer into kv_cache[block, kv_axis, pos, head, :] for this token.
  bf16_t* kv_slot_ptr(int token, int head, int kv_axis) const {
    const int64_t s = slot_mapping_[token];
    if (s < 0) return nullptr;
    const int64_t block = s / block_size_;
    const int64_t pos = s % block_size_;
    const int64_t kv_block_stride =
        (int64_t)2 * block_size_ * num_kv_heads_ * M3_HEAD_DIM;
    const int64_t kv_axis_stride = (int64_t)block_size_ * num_kv_heads_ * M3_HEAD_DIM;
    return kv_cache_ + block * kv_block_stride + kv_axis * kv_axis_stride +
           pos * (num_kv_heads_ * M3_HEAD_DIM) + head * M3_HEAD_DIM;
  }

  void insert_kv(int token, const bf16_t* __restrict__ src, int head,
                 int kv_axis) const {
    bf16_t* dst = kv_slot_ptr(token, head, kv_axis);
    if (dst == nullptr) return;
#pragma unroll
    for (int d = 0; d < M3_HEAD_DIM; ++d) dst[d] = src[d];
  }

  static void store_from_buf(const float* buf, bf16_t* dst) {
    if (dst == nullptr) return;
#pragma unroll
    for (int d = 0; d < M3_HEAD_DIM; ++d) dst[d] = static_cast<bf16_t>(buf[d]);
  }

  // token id is passed explicitly to the cache-insert helpers.
  bf16_t* __restrict__ qkv_;
  const bf16_t* __restrict__ q_norm_w_;
  const bf16_t* __restrict__ k_norm_w_;
  const bf16_t* __restrict__ iq_norm_w_;
  const bf16_t* __restrict__ ik_norm_w_;
  const bf16_t* __restrict__ cos_sin_cache_;
  const int64_t* __restrict__ positions_;
  const int64_t* __restrict__ slot_mapping_;
  const int64_t* __restrict__ index_slot_mapping_;
  bf16_t* __restrict__ kv_cache_;
  bf16_t* __restrict__ index_cache_;
  bf16_t* __restrict__ q_out_;
  bf16_t* __restrict__ index_q_out_;
  int num_heads_;
  int num_kv_heads_;
  int num_index_heads_;
  int rotary_dim_;
  float eps_;
  int block_size_;
  int64_t qkv_row_stride_;
  int64_t cos_sin_row_stride_;
  bool do_insert_;
};

}  // namespace vllm

void fused_minimax_m3_qknorm_rope_kv_insert(
    torch::Tensor& qkv,
    const torch::Tensor& q_norm_weight,
    const torch::Tensor& k_norm_weight,
    const torch::Tensor& cos_sin_cache,
    const torch::Tensor& positions,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t rotary_dim,
    double eps,
    std::optional<torch::Tensor> index_q_norm_weight,
    std::optional<torch::Tensor> index_k_norm_weight,
    int64_t num_index_heads,
    std::optional<torch::Tensor> slot_mapping,
    std::optional<torch::Tensor> index_slot_mapping,
    std::optional<torch::Tensor> kv_cache,
    std::optional<torch::Tensor> index_cache,
    int64_t block_size,
    std::optional<torch::Tensor> q_out,
    std::optional<torch::Tensor> index_q_out) {
  TORCH_CHECK(qkv.dtype() == torch::kBFloat16, "qkv must be bf16");
  TORCH_CHECK(qkv.dim() == 2, "qkv must be 2-D [num_tokens, packed]");
  TORCH_CHECK(positions.dtype() == torch::kInt64);
  TORCH_CHECK(rotary_dim % 2 == 0, "rotary_dim must be even");
  TORCH_CHECK(cos_sin_cache.size(-1) == rotary_dim,
              "cos_sin_cache last dim must equal rotary_dim");

  const int64_t num_tokens = qkv.size(0);
  const bool do_insert = kv_cache.has_value() || index_cache.has_value();

  auto bf16_ptr = [](const std::optional<torch::Tensor>& t) -> vllm::bf16_t* {
    if (!t.has_value()) return nullptr;
    return reinterpret_cast<vllm::bf16_t*>(t->data_ptr());
  };
  auto i64_ptr = [](const std::optional<torch::Tensor>& t) -> int64_t* {
    if (!t.has_value()) return nullptr;
    return t->data_ptr<int64_t>();
  };

  const int total_slots = num_heads + 2 * num_kv_heads +
                          (num_index_heads > 0 ? num_index_heads + 1 : 0);

  at::DeviceGuard dg(qkv.device());
  sycl::range<2> global(num_tokens, total_slots);

  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>(global, sycl::range<2>(1, total_slots)),
        vllm::fused_minimax_m3_qknorm_rope_kv_insert_kernel(
            reinterpret_cast<vllm::bf16_t*>(qkv.data_ptr()),
            reinterpret_cast<const vllm::bf16_t*>(q_norm_weight.data_ptr()),
            reinterpret_cast<const vllm::bf16_t*>(k_norm_weight.data_ptr()),
            index_q_norm_weight.has_value()
                ? reinterpret_cast<const vllm::bf16_t*>(
                      index_q_norm_weight->data_ptr())
                : nullptr,
            index_k_norm_weight.has_value()
                ? reinterpret_cast<const vllm::bf16_t*>(
                      index_k_norm_weight->data_ptr())
                : nullptr,
            reinterpret_cast<const vllm::bf16_t*>(cos_sin_cache.data_ptr()),
            positions.data_ptr<int64_t>(),
            i64_ptr(slot_mapping),
            i64_ptr(index_slot_mapping),
            bf16_ptr(kv_cache),
            bf16_ptr(index_cache),
            bf16_ptr(q_out),
            bf16_ptr(index_q_out),
            static_cast<int>(num_heads),
            static_cast<int>(num_kv_heads),
            static_cast<int>(num_index_heads),
            static_cast<int>(rotary_dim),
            static_cast<float>(eps),
            static_cast<int>(block_size),
            qkv.stride(0),
            cos_sin_cache.stride(0),
            do_insert));
  });
}
