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
// four norms are Gemma-style RMSNorm: ``x * rsqrt(mean(x^2)+eps) *
// (1+weight)``.
//
// The fused ``qkv`` packs, per token: [ q | k | v | index_q | index_k ] in the
// sparse layer, or just [ q | k | v ] in the dense layer (no index, no insert).
//
//   - q       : Gemma-norm + rope -> q_out (sparse) or in place in qkv (dense)
//   - k       : Gemma-norm + rope -> in place in qkv, scatter into
//   kv_cache[:,0]
//   - v       : untouched,                          scatter into kv_cache[:,1]
//   - index_q : Gemma-norm + rope -> index_q_out (sparse)
//   - index_k : Gemma-norm + rope -> in place in qkv, scatter into index_cache
//
// One work-item handles one (token, head-slot); it loops over the 128-dim head.

#include <sycl/sycl.hpp>
#include "utils.h"
#include <torch/all.h>
#include <ATen/DeviceGuard.h>
#include <optional>

namespace vllm {

using bf16_t = sycl::ext::oneapi::bfloat16;

static constexpr int M3_HEAD_DIM = 128;
// One sub-group cooperatively processes one (token, head-slot): the 128-dim
// head is split across the lanes. This launches num_tokens*total_slots
// sub-groups (thousands of hardware threads) instead of one partial sub-group
// per head, which is what restores occupancy on BMG/Xe2.
static constexpr int M3_SG_SIZE = 16;

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
      int num_tokens,
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
        num_tokens_(num_tokens),
        num_heads_(num_heads),
        num_kv_heads_(num_kv_heads),
        num_index_heads_(num_index_heads),
        rotary_dim_(rotary_dim),
        eps_(eps),
        block_size_(block_size),
        qkv_row_stride_(qkv_row_stride),
        cos_sin_row_stride_(cos_sin_row_stride),
        do_insert_(do_insert) {}

  [[sycl::reqd_sub_group_size(M3_SG_SIZE)]] void
  operator()(sycl::nd_item<1> item) const {
    auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();

    // Decode the head-slot -> (section, head index within section).
    const int q_end = num_heads_;
    const int k_end = q_end + num_kv_heads_;
    const int v_end = k_end + num_kv_heads_;
    const int iq_end = v_end + num_index_heads_;
    const int ik_end = iq_end + (num_index_heads_ > 0 ? 1 : 0);
    const int total_slots = ik_end;

    // Global sub-group index -> (token, slot). Padding sub-groups (from
    // rounding the global range up to the work-group size) fall out uniformly.
    const int64_t head_idx = item.get_global_id(0) / M3_SG_SIZE;
    if (head_idx >= (int64_t)num_tokens_ * total_slots) return;
    const int token = head_idx / total_slots;
    const int slot = head_idx % total_slots;

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
      case SEC_Q:
        col = head * M3_HEAD_DIM;
        break;
      case SEC_K:
        col = q_cols + head * M3_HEAD_DIM;
        break;
      case SEC_V:
        col = q_cols + kv_cols + head * M3_HEAD_DIM;
        break;
      case SEC_IQ:
        col = q_cols + 2 * kv_cols + head * M3_HEAD_DIM;
        break;
      default:
        col = q_cols + 2 * kv_cols + iq_cols;
        break;  // SEC_IK
    }

    bf16_t* __restrict__ src = qkv_ + token * qkv_row_stride_ + col;

    // V is raw: no norm/rope. Only (optionally) scatter into kv_cache[:,1].
    if (section == SEC_V) {
      if (do_insert_ && kv_cache_ != nullptr) {
        insert_kv(lane, token, src, head, /*kv_axis=*/1);
      }
      return;
    }

    const bf16_t* __restrict__ weight = section == SEC_Q    ? q_norm_w_
                                        : section == SEC_K  ? k_norm_w_
                                        : section == SEC_IQ ? iq_norm_w_
                                                            : ik_norm_w_;

    // Pass 1: sum of squares, split across the sub-group's lanes.
    float local_ss = 0.0f;
    for (int d = lane; d < M3_HEAD_DIM; d += M3_SG_SIZE) {
      const float x = static_cast<float>(src[d]);
      local_ss += x * x;
    }
    const float sumsq =
        sycl::reduce_over_group(sg, local_ss, sycl::plus<float>());
    // Gemma RMSNorm: x * rsqrt(mean(x^2)+eps) * (1 + weight).
    const float rms = sycl::rsqrt(sumsq / M3_HEAD_DIM + eps_);

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

    // Optional cache destination (sparse mode): K -> kv_cache[:,0],
    // index_k -> index_cache. Computed once so pass 2 can write both at once.
    bf16_t* __restrict__ cdst = nullptr;
    if (do_insert_) {
      if (section == SEC_K && kv_cache_ != nullptr) {
        cdst = kv_slot_ptr(token, head, /*kv_axis=*/0);
      } else if (section == SEC_IK && index_cache_ != nullptr) {
        const int64_t s = index_slot_mapping_[token];
        if (s >= 0) cdst = index_cache_ + s * M3_HEAD_DIM;
      }
    }

    // Pass 2: normalize + partial NeoX RoPE, streamed straight to dst (and the
    // cache). Each lane owns a strided slice of the head; the rotary pairs
    // (i, i+half) are both recomputed locally from src, so no cross-lane
    // communication is needed beyond the sum-of-squares reduction above. Reads
    // of src happen exactly as each index is written, so the in-place
    // (dst==src) case has no read-after-write hazard. (Register-caching src
    // between the two passes was tried and regressed ~19% — the second read is
    // L1-resident and free, while the cache added register pressure; see
    // opt_minimax-m3/OPTIMIZATION_LOG.md Phase 6.)
    const int half = rotary_dim_ / 2;
    const bf16_t* cs = cos_sin_cache_ + positions_[token] * cos_sin_row_stride_;
    auto norm = [&](int d) {
      return static_cast<float>(src[d]) * rms *
             (1.0f + static_cast<float>(weight[d]));
    };
    for (int i = lane; i < half; i += M3_SG_SIZE) {
      const float cos = static_cast<float>(cs[i]);
      const float sin = static_cast<float>(cs[half + i]);
      const float n1 = norm(i);
      const float n2 = norm(half + i);
      const bf16_t o1 = static_cast<bf16_t>(n1 * cos - n2 * sin);
      const bf16_t o2 = static_cast<bf16_t>(n2 * cos + n1 * sin);
      dst[i] = o1;
      dst[half + i] = o2;
      if (cdst != nullptr) {
        cdst[i] = o1;
        cdst[half + i] = o2;
      }
    }
    for (int d = rotary_dim_ + lane; d < M3_HEAD_DIM; d += M3_SG_SIZE) {
      const bf16_t v = static_cast<bf16_t>(norm(d));
      dst[d] = v;
      if (cdst != nullptr) cdst[d] = v;
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
    const int64_t kv_axis_stride =
        (int64_t)block_size_ * num_kv_heads_ * M3_HEAD_DIM;
    return kv_cache_ + block * kv_block_stride + kv_axis * kv_axis_stride +
           pos * (num_kv_heads_ * M3_HEAD_DIM) + head * M3_HEAD_DIM;
  }

  void insert_kv(
      int lane,
      int token,
      const bf16_t* __restrict__ src,
      int head,
      int kv_axis) const {
    bf16_t* dst = kv_slot_ptr(token, head, kv_axis);
    if (dst == nullptr) return;
    for (int d = lane; d < M3_HEAD_DIM; d += M3_SG_SIZE)
      dst[d] = src[d];
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
  int num_tokens_;
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
  CHECK_DEVICE(qkv);
  CHECK_CONTIGUOUS(qkv);
  CHECK_DEVICE(q_norm_weight);
  CHECK_CONTIGUOUS(q_norm_weight);
  CHECK_DEVICE(k_norm_weight);
  CHECK_CONTIGUOUS(k_norm_weight);
  CHECK_DEVICE(cos_sin_cache);
  CHECK_CONTIGUOUS(cos_sin_cache);
  CHECK_DEVICE(positions);
  CHECK_CONTIGUOUS(positions);

  TORCH_CHECK(qkv.dtype() == torch::kBFloat16, "qkv must be bf16");
  TORCH_CHECK(qkv.dim() == 2, "qkv must be 2-D [num_tokens, packed]");
  TORCH_CHECK(
      q_norm_weight.dtype() == torch::kBFloat16, "q_norm_weight must be bf16");
  TORCH_CHECK(
      k_norm_weight.dtype() == torch::kBFloat16, "k_norm_weight must be bf16");
  TORCH_CHECK(
      cos_sin_cache.dtype() == torch::kBFloat16, "cos_sin_cache must be bf16");
  TORCH_CHECK(
      q_norm_weight.numel() == vllm::M3_HEAD_DIM,
      "q_norm_weight must have head_dim (",
      vllm::M3_HEAD_DIM,
      ") elements");
  TORCH_CHECK(
      k_norm_weight.numel() == vllm::M3_HEAD_DIM,
      "k_norm_weight must have head_dim (",
      vllm::M3_HEAD_DIM,
      ") elements");
  TORCH_CHECK(positions.dtype() == torch::kInt64, "positions must be int64");
  TORCH_CHECK(positions.dim() == 1, "positions must be 1-D [num_tokens]");
  TORCH_CHECK(
      positions.size(0) == qkv.size(0),
      "positions length must match num_tokens");
  TORCH_CHECK(rotary_dim % 2 == 0, "rotary_dim must be even");
  TORCH_CHECK(
      rotary_dim <= vllm::M3_HEAD_DIM,
      "rotary_dim must be <= head_dim (",
      vllm::M3_HEAD_DIM,
      ")");
  TORCH_CHECK(
      cos_sin_cache.dim() == 2,
      "cos_sin_cache must be 2-D [max_position, rotary_dim]");
  TORCH_CHECK(
      cos_sin_cache.size(-1) == rotary_dim,
      "cos_sin_cache last dim must equal rotary_dim");
  TORCH_CHECK(
      num_heads > 0 && num_kv_heads > 0,
      "num_heads and num_kv_heads must be positive");
  TORCH_CHECK(num_index_heads >= 0, "num_index_heads must be non-negative");

  const int64_t expected_cols =
      ((int64_t)num_heads + 2 * num_kv_heads +
       (num_index_heads > 0 ? num_index_heads + 1 : 0)) *
      vllm::M3_HEAD_DIM;
  TORCH_CHECK(
      qkv.size(1) == expected_cols,
      "qkv packed dim (",
      qkv.size(1),
      ") must equal (num_heads + ",
      "2*num_kv_heads + (num_index_heads>0 ? num_index_heads+1 : 0)) * ",
      "head_dim (= ",
      expected_cols,
      ")");

  // Index branches require their Gemma-norm weights; index_q_out stays optional
  // (q/index_q fall back to in-place when no gather buffer is supplied).
  if (num_index_heads > 0) {
    TORCH_CHECK(
        index_q_norm_weight.has_value() && index_k_norm_weight.has_value(),
        "index_q_norm_weight and index_k_norm_weight are required when "
        "num_index_heads > 0");
    CHECK_DEVICE(index_q_norm_weight.value());
    CHECK_CONTIGUOUS(index_q_norm_weight.value());
    CHECK_DEVICE(index_k_norm_weight.value());
    CHECK_CONTIGUOUS(index_k_norm_weight.value());
    TORCH_CHECK(
        index_q_norm_weight->dtype() == torch::kBFloat16 &&
            index_k_norm_weight->dtype() == torch::kBFloat16,
        "index norm weights must be bf16");
    TORCH_CHECK(
        index_q_norm_weight->numel() == vllm::M3_HEAD_DIM &&
            index_k_norm_weight->numel() == vllm::M3_HEAD_DIM,
        "index norm weights must have head_dim (",
        vllm::M3_HEAD_DIM,
        ") elements");
  }

  if (q_out.has_value()) {
    CHECK_DEVICE(q_out.value());
    CHECK_CONTIGUOUS(q_out.value());
    TORCH_CHECK(q_out->dtype() == torch::kBFloat16, "q_out must be bf16");
  }
  if (index_q_out.has_value()) {
    CHECK_DEVICE(index_q_out.value());
    CHECK_CONTIGUOUS(index_q_out.value());
    TORCH_CHECK(
        index_q_out->dtype() == torch::kBFloat16, "index_q_out must be bf16");
  }

  // Cache inserts need their matching slot maps (and a valid block_size for the
  // paged kv_cache); without them the kernel would dereference a null map.
  if (kv_cache.has_value()) {
    TORCH_CHECK(
        slot_mapping.has_value(),
        "slot_mapping is required when kv_cache is provided");
    CHECK_DEVICE(kv_cache.value());
    CHECK_CONTIGUOUS(kv_cache.value());
    CHECK_DEVICE(slot_mapping.value());
    CHECK_CONTIGUOUS(slot_mapping.value());
    TORCH_CHECK(kv_cache->dtype() == torch::kBFloat16, "kv_cache must be bf16");
    TORCH_CHECK(
        slot_mapping->dtype() == torch::kInt64, "slot_mapping must be int64");
    TORCH_CHECK(
        slot_mapping->numel() == qkv.size(0),
        "slot_mapping length must match num_tokens");
    TORCH_CHECK(
        block_size > 0,
        "block_size must be positive when kv_cache is provided");
  }
  if (index_cache.has_value()) {
    TORCH_CHECK(
        index_slot_mapping.has_value(),
        "index_slot_mapping is required when index_cache is provided");
    CHECK_DEVICE(index_cache.value());
    CHECK_CONTIGUOUS(index_cache.value());
    CHECK_DEVICE(index_slot_mapping.value());
    CHECK_CONTIGUOUS(index_slot_mapping.value());
    TORCH_CHECK(
        index_cache->dtype() == torch::kBFloat16, "index_cache must be bf16");
    TORCH_CHECK(
        index_slot_mapping->dtype() == torch::kInt64,
        "index_slot_mapping must be int64");
    TORCH_CHECK(
        index_slot_mapping->numel() == qkv.size(0),
        "index_slot_mapping length must match num_tokens");
  }

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
  // One sub-group (M3_SG_SIZE lanes) per (token, head-slot). Flatten to 1-D and
  // round the global range up to the work-group size; padding sub-groups return
  // early. WG = 8 sub-groups for good occupancy.
  constexpr int kSgPerWg = 8;
  const int64_t wg = (int64_t)kSgPerWg * vllm::M3_SG_SIZE;
  const int64_t num_sg = (int64_t)num_tokens * total_slots;
  const int64_t global_items = ((num_sg * vllm::M3_SG_SIZE + wg - 1) / wg) * wg;

  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global_items), sycl::range<1>(wg)),
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
            static_cast<int>(num_tokens),
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
