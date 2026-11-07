// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// SYCL port of the DeepSeek-V4 sparse-attention KV insert kernel (fp8mix
// output).
//
// A single one-pass kernel handles every shape: one work-group per token
// gathers the compression window with an online softmax, then applies
// RMSNorm, RoPE and the fp8 quantized cache store.
//
// Data layout contract: fp8 NOPE + bf16 rope tail + UE8M0 scales.

#include "ops.h"
#include "utils.h"

#include <ATen/DeviceGuard.h>
#include <c10/util/Float8_e4m3fn.h>
#include <cstdint>
#include <type_traits>

// File-local constants, at global scope rather than inside `namespace vllm`:
// both the kernel class and the host-side validation need them, and a
// `vllm::` prefix would make TU-local values look like shared project symbols.
namespace {
constexpr float kE4m3FnMax = 448.0f;
constexpr float kUe8m0ExpBias = 127.0f;
constexpr float kQuantAbsmaxFloor = 1.0e-4f;
// Cache layout: 512-wide head, NOPE quantized in 64-element blocks.
constexpr int kSparseHeadSize = 512;
constexpr int kSparseQuantBlock = 64;
// Elements a work-item owns, and therefore the granularity at which the NOPE
// payload is written. Must match the kernel's PER_WI.
constexpr int kSparseStoreQuad = 4;
}  // namespace

namespace vllm {

// Runtime-parameterized fp8mix kernel: compress_ratio, overlap and
// rope_head_dim are plain arguments, so one instantiation covers every layer
// shape.
// WT is the RMSNorm weight scalar type (float or bfloat16); it is upcast to
// fp32 per-element inside the kernel.
template <typename WT>
class sparse_kv_insert_fp8mix_generic_kernel {
 public:
  static constexpr int HEAD_SIZE = kSparseHeadSize;
  static constexpr int WG_SIZE = 128;
  static constexpr int SG_SIZE = 32;
  static constexpr int PER_WI = HEAD_SIZE / WG_SIZE;  // 4
  static constexpr int QUANT_BLOCK = kSparseQuantBlock;
  using wi_vec_t = sycl::vec<float, PER_WI>;
  static constexpr float FP8_MAX = kE4m3FnMax;
  static constexpr float INV_FP8_MAX = 1.0f / FP8_MAX;
  static constexpr float NEG_LARGE = -1.0e30f;

  using bf16_t = sycl::ext::oneapi::bfloat16;
  using fp8_t = at::Float8_e4m3fn;

  // The PER_WI quantized bytes a work-item produces, stored as one unit.
  // A byte array with explicit alignment rather than a packed integer, so no
  // byte-order assumption is made.
  struct alignas(PER_WI) fp8_quad {
    uint8_t v[PER_WI];
  };

  // Block ids covering one gather range, staged in SLM (see the prefetch in
  // operator()). The range spans n_gather = (1 + overlap) * compress_ratio
  // consecutive positions, so it touches at most n_gather blocks (the bound is
  // reached only at state_cache block size 1). The host restricts the op to
  // cr=4/overlap=1 and cr=128/overlap=0, so n_gather is 8 or 128 and this
  // bound always holds — no fallback path is needed.
  static constexpr int MAX_BT_PREFETCH = 128;

  // Only the block ids need to cross work-items. The per-token metadata is
  // re-read by every work-item (same address, so the loads coalesce) and the
  // normalized head stays in registers (see the quantization phase).
  struct local_mem_t {
    int32_t blocks[MAX_BT_PREFETCH];
  };

  sparse_kv_insert_fp8mix_generic_kernel(
      const float* state_cache,
      int64_t state_cache_stride0,
      int64_t state_cache_stride1,
      int state_width,
      const int32_t* token_to_req,
      const int64_t* positions,
      const int64_t* slot_mapping,
      const int32_t* block_table,
      int64_t block_table_stride0,
      int state_cache_block_size,
      const WT* rms_w,
      float rms_eps,
      const float* cos_sin_cache,
      int64_t cos_sin_stride0,
      uint8_t* k_cache,
      const int64_t* kv_slot_mapping,
      int kv_cache_block_size,
      int64_t kv_block_stride,
      int token_stride,
      int scale_dim,
      int compress_ratio,
      int overlap,
      int rope_head_dim)
      : state_cache_(state_cache),
        state_cache_stride0_(state_cache_stride0),
        state_cache_stride1_(state_cache_stride1),
        state_width_(state_width),
        token_to_req_(token_to_req),
        positions_(positions),
        slot_mapping_(slot_mapping),
        block_table_(block_table),
        block_table_stride0_(block_table_stride0),
        state_cache_block_size_(state_cache_block_size),
        rms_w_(rms_w),
        rms_eps_(rms_eps),
        cos_sin_cache_(cos_sin_cache),
        cos_sin_stride0_(cos_sin_stride0),
        k_cache_(k_cache),
        kv_slot_mapping_(kv_slot_mapping),
        kv_cache_block_size_(kv_cache_block_size),
        kv_block_stride_(kv_block_stride),
        token_stride_(token_stride),
        scale_dim_(scale_dim),
        compress_ratio_(compress_ratio),
        overlap_(overlap),
        rope_head_dim_(rope_head_dim) {}

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void
  operator()(sycl::nd_item<1> it) const {
    const int token = it.get_group(0);
    const int lid = it.get_local_id(0);
    const int dim_base = lid * PER_WI;

    // Uniform addresses across the work-group, so these coalesce to one cache
    // line each; SLM staging would only add a barrier.
    const int64_t slot_id = slot_mapping_[token];
    const int64_t position = positions_[token];
    const int64_t kv_slot_idx = kv_slot_mapping_[token];
    const int32_t req_idx = token_to_req_[token];

    const bool should_store =
        (slot_id >= 0) && (position >= 0) &&
        (((position + 1) % static_cast<int64_t>(compress_ratio_)) == 0) &&
        (kv_slot_idx >= 0);

    // Uniform across the work-group, so the whole group returns together,
    // before touching SLM. At 1/cr density this skips most of the work.
    if (!should_store) return;

    auto& smem =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<local_mem_t>(
            it.get_group());

    const int64_t n_gather = static_cast<int64_t>(1 + overlap_) *
                             static_cast<int64_t>(compress_ratio_);
    const int64_t start = position - n_gather + 1;

    // ========================================================================
    // PHASE 0: Block-table prefetch
    // ========================================================================
    // Fetch the whole contiguous block-id window cooperatively. Otherwise every
    // gather iteration runs a serial block_table -> state_cache dependent load.
    const int64_t first_pos = start > 0 ? start : 0;
    const int64_t bt_row = static_cast<int64_t>(req_idx) * block_table_stride0_;
    const int64_t logical_lo = first_pos / state_cache_block_size_;
    const int64_t logical_hi = position / state_cache_block_size_;
    const int64_t bt_count = logical_hi - logical_lo + 1;
    for (int64_t i = lid; i < bt_count; i += WG_SIZE) {
      smem.blocks[i] = block_table_[bt_row + logical_lo + i];
    }
    sycl::group_barrier(it.get_group());

    // ========================================================================
    // PHASE 1: Gather + online softmax compression
    // ========================================================================

    float m_run[PER_WI], s_run[PER_WI], acc[PER_WI];
#pragma unroll
    for (int j = 0; j < PER_WI; ++j) {
      m_run[j] = NEG_LARGE;
      s_run[j] = 0.f;
      acc[j] = 0.f;
    }

    // Positions before 0 are inert: they leave m_run at NEG_LARGE, so the first
    // in-range score underflows their alpha to 0. Skipping them is bit-exact.
    const int64_t first_idx = start < 0 ? -start : 0;

    // gather_pos advances by 1, so carry the block index instead of a 64-bit
    // div/mod per iteration. logical_lo == first_pos / bs, hence blk starts 0.
    int64_t blk = 0;
    int64_t pos_in_block = first_pos % state_cache_block_size_;

    for (int64_t gather_idx = first_idx; gather_idx < n_gather; ++gather_idx) {
      const int head_offset = (gather_idx >= compress_ratio_) ? HEAD_SIZE : 0;

      const float* row =
          state_cache_ +
          static_cast<int64_t>(smem.blocks[blk]) * state_cache_stride0_ +
          pos_in_block * state_cache_stride1_;

      // One 16-byte vector load per work-item makes the sub-group read one
      // contiguous block. The host validates the alignment this needs.
      const int h = dim_base + head_offset;
      const wi_vec_t score_v =
          *reinterpret_cast<const wi_vec_t*>(row + state_width_ + h);
      const wi_vec_t kv_v = *reinterpret_cast<const wi_vec_t*>(row + h);

#pragma unroll
      for (int j = 0; j < PER_WI; ++j) {
        const float score = score_v[j];
        const float kv = kv_v[j];
        const float new_m = sycl::max(m_run[j], score);
        const float alpha = sycl::exp(m_run[j] - new_m);
        const float p = sycl::exp(score - new_m);
        s_run[j] = s_run[j] * alpha + p;
        acc[j] = acc[j] * alpha + p * kv;
        m_run[j] = new_m;
      }

      if (++pos_in_block == state_cache_block_size_) {
        pos_in_block = 0;
        ++blk;
      }
    }

    float ckv[PER_WI];
#pragma unroll
    for (int j = 0; j < PER_WI; ++j) {
      ckv[j] = (s_run[j] > 0.f) ? (acc[j] / s_run[j]) : 0.f;
    }

    // ========================================================================
    // PHASE 2: RMSNorm (sum of squares reduced across the whole work-group)
    // ========================================================================

    // Issued before the reduction: it does not depend on the reduced value, and
    // reduce_over_group's barriers would otherwise fence this load after it.
    float w[PER_WI];
#pragma unroll
    for (int j = 0; j < PER_WI; ++j) {
      w[j] = static_cast<float>(rms_w_[dim_base + j]);
    }

    float ps = 0.f;
#pragma unroll
    for (int j = 0; j < PER_WI; ++j) {
      ps += ckv[j] * ckv[j];
    }
    const float ss =
        sycl::reduce_over_group(it.get_group(), ps, sycl::plus<float>());
    const float inv =
        sycl::rsqrt(ss / static_cast<float>(HEAD_SIZE) + rms_eps_);

    float normed[PER_WI];
#pragma unroll
    for (int j = 0; j < PER_WI; ++j) {
      normed[j] = ckv[j] * inv * w[j];
    }

    // ========================================================================
    // PHASE 3: GPT-J RoPE on the tail pairs (each pair is work-item local)
    // ========================================================================
    const int nope_head_dim = HEAD_SIZE - rope_head_dim_;
    const int nope_pairs = nope_head_dim / 2;
    const int half_rope = rope_head_dim_ / 2;

    // should_store gives position == m * cr - 1, so (position / cr) * cr
    // reduces to position + 1 - cr; no 64-bit div/mul needed.
    const int64_t compressed_pos =
        position + 1 - static_cast<int64_t>(compress_ratio_);
    const float* rope_base = cos_sin_cache_ + compressed_pos * cos_sin_stride0_;
#pragma unroll
    for (int p = 0; p < PER_WI / 2; ++p) {
      const int pair_idx = (dim_base / 2) + p;
      // pair_idx < HEAD_SIZE / 2 == nope_pairs + half_rope, so a non-negative
      // rope_pair_local is already below half_rope.
      const int rope_pair_local = pair_idx - nope_pairs;
      if (rope_pair_local >= 0) {
        const float cv = rope_base[rope_pair_local];
        const float sv = rope_base[half_rope + rope_pair_local];
        const float even = normed[p * 2];
        const float odd = normed[p * 2 + 1];
        normed[p * 2] = even * cv - odd * sv;
        normed[p * 2 + 1] = odd * cv + even * sv;
      }
    }

    // ========================================================================
    // PHASE 4: FP8 quantization and KV cache write
    // ========================================================================
    // Nothing crosses work-items here, so the result stays in registers. The
    // quant path is defined on bf16-rounded inputs, hence the round-trip.
    bf16_t nb[PER_WI];
#pragma unroll
    for (int j = 0; j < PER_WI; ++j) {
      nb[j] = static_cast<bf16_t>(normed[j]);
    }

    const int64_t kv_block = kv_slot_idx / kv_cache_block_size_;
    const int64_t kv_pos = kv_slot_idx % kv_cache_block_size_;
    uint8_t* block_base = k_cache_ + kv_block * kv_block_stride_;
    uint8_t* fp8_ptr = block_base + kv_pos * token_stride_;
    uint8_t* scale_ptr =
        block_base + kv_cache_block_size_ * token_stride_ + kv_pos * scale_dim_;

    // 16 WIs per 64-element quant block; the last block may be short.
    constexpr int WIS_PER_BLOCK = QUANT_BLOCK / PER_WI;  // 16
    const int scale_blocks = (nope_head_dim + QUANT_BLOCK - 1) / QUANT_BLOCK;
    const int total_quant_wis = scale_blocks * WIS_PER_BLOCK;

    // Sub-group collectives below run unpredicated (they require converged
    // control flow); only the memory accesses are guarded. Lanes outside the
    // NOPE range feed the identity 0 into the max and drop the broadcast.
    const bool active = lid < total_quant_wis;
    // Equals PER_WI * lid == dim_base for active lanes: each lane quantizes
    // exactly what it computed, so nb[] can be read directly.
    const int b = active ? (lid / WIS_PER_BLOCK) : 0;
    // rope_head_dim % PER_WI == 0 (host-checked), so a work-item's elements are
    // either all NOPE or all rope tail — no per-element masking on either side.
    const bool in_nope = dim_base < nope_head_dim;

    auto sg = it.get_sub_group();
    const int sg_lid = static_cast<int>(sg.get_local_linear_id());
    const int half_leader =
        (sg_lid / WIS_PER_BLOCK) * WIS_PER_BLOCK;  // 0 or 16 inside subgroup

    float block_absmax = 0.f;
    if (in_nope) {
#pragma unroll
      for (int j = 0; j < PER_WI; ++j) {
        block_absmax =
            sycl::max(block_absmax, sycl::fabs(static_cast<float>(nb[j])));
      }
    }

    // 16-lane reduction inside the half-subgroup (all lanes participate).
#pragma unroll
    for (int mask = WIS_PER_BLOCK / 2; mask > 0; mask >>= 1) {
      const float other = sycl::permute_group_by_xor(sg, block_absmax, mask);
      block_absmax = sycl::max(block_absmax, other);
    }

    float inv_scale_local = 0.f;
    if (active && sg_lid == half_leader) {
      block_absmax = sycl::max(block_absmax, kQuantAbsmaxFloor);
      const float raw_scale = block_absmax * INV_FP8_MAX;
      const float exponent = sycl::ceil(sycl::log2(raw_scale));
      inv_scale_local = sycl::exp2(-exponent);

      float encoded = exponent + kUe8m0ExpBias;
      encoded = sycl::clamp(encoded, 0.f, 255.f);
      scale_ptr[b] = static_cast<uint8_t>(encoded);
    }

    const float inv_scale =
        sycl::select_from_group(sg, inv_scale_local, half_leader);

    if (in_nope) {
      // One 4-byte store per work-item, so the sub-group emits a single block
      // write instead of PER_WI scattered byte writes.
      fp8_quad quad;
#pragma unroll
      for (int j = 0; j < PER_WI; ++j) {
        float x = static_cast<float>(nb[j]) * inv_scale;
        x = sycl::clamp(x, -FP8_MAX, FP8_MAX);
        quad.v[j] = sycl::bit_cast<uint8_t>(static_cast<fp8_t>(x));
      }
      *reinterpret_cast<fp8_quad*>(fp8_ptr + dim_base) = quad;
    }

    // Pad the scale bytes past the real quant blocks with zero.
    for (int i = scale_blocks + lid; i < scale_dim_; i += WG_SIZE) {
      scale_ptr[i] = static_cast<uint8_t>(0);
    }

    // Rope tail, stored by exactly the work-items in_nope excludes, so the two
    // phases cover the head once.
    if (!in_nope) {
      auto* rope_dst = reinterpret_cast<bf16_t*>(fp8_ptr + nope_head_dim) +
                       (dim_base - nope_head_dim);
#pragma unroll
      for (int j = 0; j < PER_WI; ++j) {
        rope_dst[j] = nb[j];
      }
    }
  }

 private:
  const float* state_cache_;
  int64_t state_cache_stride0_;
  int64_t state_cache_stride1_;
  int state_width_;
  const int32_t* token_to_req_;
  const int64_t* positions_;
  const int64_t* slot_mapping_;
  const int32_t* block_table_;
  int64_t block_table_stride0_;
  int state_cache_block_size_;
  const WT* rms_w_;
  float rms_eps_;
  const float* cos_sin_cache_;
  int64_t cos_sin_stride0_;
  uint8_t* k_cache_;
  const int64_t* kv_slot_mapping_;
  int kv_cache_block_size_;
  int64_t kv_block_stride_;
  int token_stride_;
  int scale_dim_;
  int compress_ratio_;
  int overlap_;
  int rope_head_dim_;
};

}  // namespace vllm

namespace {

constexpr const char* kOpName =
    "fused_kv_compress_norm_rope_insert_sparse_attn";

}  // namespace
void fused_kv_compress_norm_rope_insert_sparse_attn(
    const torch::Tensor& state_cache,
    const torch::Tensor& token_to_req_indices,
    const torch::Tensor& positions,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& block_table,
    const torch::Tensor& rms_norm_weight,
    double rms_norm_eps,
    const torch::Tensor& cos_sin_cache,
    torch::Tensor& k_cache,
    const torch::Tensor& kv_slot_mapping,
    int64_t kv_cache_block_size,
    int64_t compress_ratio,
    int64_t overlap,
    int64_t rope_head_dim,
    int64_t token_stride,
    int64_t scale_dim,
    int64_t kv_block_stride) {
  // Bind all queue lookups and device allocations to the input tensor's device,
  // regardless of the ambient current device, before vllmGetQueue() is called.
  const at::DeviceGuard device_guard(state_cache.device());
  TORCH_CHECK(
      state_cache.dtype() == torch::kFloat32,
      kOpName,
      " SYCL: state_cache must be fp32");
  TORCH_CHECK(k_cache.dtype() == torch::kUInt8, "k_cache must be uint8");
  TORCH_CHECK(
      rms_norm_weight.scalar_type() == torch::kFloat32 ||
          rms_norm_weight.scalar_type() == torch::kBFloat16,
      kOpName,
      ": rms_norm_weight must be float32 or bfloat16");
  TORCH_CHECK(cos_sin_cache.dtype() == torch::kFloat32);
  TORCH_CHECK(positions.dtype() == torch::kInt64);
  TORCH_CHECK(slot_mapping.dtype() == torch::kInt64);
  TORCH_CHECK(kv_slot_mapping.dtype() == torch::kInt64);
  TORCH_CHECK(token_to_req_indices.dtype() == torch::kInt32);
  TORCH_CHECK(block_table.dtype() == torch::kInt32);
  TORCH_CHECK(state_cache.dim() == 3);
  TORCH_CHECK(
      k_cache.dim() == 3,
      kOpName,
      ": k_cache must be rank-3 [num_blocks, block_size, payload], got dim=",
      k_cache.dim());
  TORCH_CHECK(
      k_cache.size(-1) >= token_stride + scale_dim,
      kOpName,
      ": k_cache last dim too small, got ",
      k_cache.size(-1),
      ", required >= token_stride + scale_dim = ",
      token_stride,
      " + ",
      scale_dim,
      " = ",
      (token_stride + scale_dim));
  TORCH_CHECK(
      k_cache.stride(2) == 1,
      kOpName,
      ": k_cache must be byte-contiguous on last dim, got stride(2)=",
      k_cache.stride(2));
  TORCH_CHECK(
      k_cache.stride(1) >= token_stride,
      kOpName,
      ": unsupported k_cache layout. Need stride(1) >= token_stride so each "
      "token payload has enough contiguous bytes, got stride(1)=",
      k_cache.stride(1),
      ", token_stride=",
      token_stride);
  // Only the two layer shapes vLLM actually builds are supported. The kernel
  // relies on this: its SLM block-table prefetch is sized for
  // n_gather = (1 + overlap) * compress_ratio <= 128, and it has no fallback
  // for a wider gather window.
  TORCH_CHECK(
      (compress_ratio == 4 && overlap == 1) ||
          (compress_ratio == 128 && overlap == 0),
      kOpName,
      ": unsupported (compress_ratio, overlap). Only (4, 1) and (128, 0) are "
      "supported, got (",
      compress_ratio,
      ", ",
      overlap,
      ")");
  TORCH_CHECK(
      rope_head_dim >= 0 && rope_head_dim <= kSparseHeadSize &&
          (rope_head_dim % kSparseStoreQuad == 0),
      kOpName,
      ": rope_head_dim must be a multiple of ",
      kSparseStoreQuad,
      " and in [0, ",
      kSparseHeadSize,
      "]");
  TORCH_CHECK(token_stride > 0, kOpName, ": token_stride must be > 0");
  TORCH_CHECK(scale_dim > 0, kOpName, ": scale_dim must be > 0");
  const int64_t nope_head_dim = kSparseHeadSize - rope_head_dim;
  const int64_t required_scale_blocks =
      (nope_head_dim + kSparseQuantBlock - 1) / kSparseQuantBlock;
  TORCH_CHECK(
      scale_dim >= required_scale_blocks,
      kOpName,
      ": scale_dim too small for NOPE quant blocks");
  TORCH_CHECK(
      token_stride >= nope_head_dim + rope_head_dim * 2,
      kOpName,
      ": token_stride too small for fp8+rope payload");
  TORCH_CHECK(
      kv_block_stride >= kv_cache_block_size * (token_stride + scale_dim),
      kOpName,
      ": kv_block_stride too small for mixed layout");
  TORCH_CHECK(
      kv_block_stride == k_cache.stride(0),
      kOpName,
      ": kv_block_stride must match k_cache.stride(0), got kv_block_stride=",
      kv_block_stride,
      ", k_cache.stride(0)=",
      k_cache.stride(0));

  const int64_t num_tokens = positions.numel();
  if (num_tokens == 0) return;

  const int state_width = static_cast<int>(state_cache.size(-1) / 2);
  const int state_block_size = static_cast<int>(state_cache.size(1));

  // The gather reads row[state_width + head_offset + h] and row[head_offset +
  // h] with h in [0, HEAD_SIZE) and head_offset in {0, HEAD_SIZE}, where the
  // second (head_offset == HEAD_SIZE) segment is only accessed when overlap
  // == 1. Each half of the row (kv / score) must therefore hold (1 + overlap)
  // HEAD_SIZE segments, i.e. state_width >= HEAD_SIZE * (1 + overlap). Validate
  // here so a mismatched caller layout fails cleanly instead of reading out of
  // bounds on the device.
  const int64_t required_state_width =
      static_cast<int64_t>(kSparseHeadSize) * (1 + overlap);
  TORCH_CHECK(
      static_cast<int64_t>(state_width) >= required_state_width,
      kOpName,
      ": state_cache last dim too small. Need size(-1) >= 2 * HEAD_SIZE * (1 + "
      "overlap) = ",
      2 * required_state_width,
      " (HEAD_SIZE=",
      kSparseHeadSize,
      ", overlap=",
      overlap,
      "), got size(-1)=",
      state_cache.size(-1));

  auto* sc = state_cache.data_ptr<float>();
  auto* tr = token_to_req_indices.data_ptr<int32_t>();
  auto* pos = positions.data_ptr<int64_t>();
  auto* sm = slot_mapping.data_ptr<int64_t>();
  auto* bt = block_table.data_ptr<int32_t>();
  auto* cs = cos_sin_cache.data_ptr<float>();
  auto* kc = k_cache.data_ptr<uint8_t>();
  auto* kvs = kv_slot_mapping.data_ptr<int64_t>();

  auto& q = vllm::xpu::vllmGetQueue();
  const float eps = static_cast<float>(rms_norm_eps);

  constexpr int64_t kVecFloats = 4;
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(sc) % (kVecFloats * sizeof(float)) == 0 &&
          state_cache.stride(0) % kVecFloats == 0 &&
          state_cache.stride(1) % kVecFloats == 0 &&
          static_cast<int64_t>(state_width) % kVecFloats == 0,
      kOpName,
      ": state_cache must be 16-byte aligned for the vectorized gather. Need "
      "the data pointer 16B-aligned and stride(0), stride(1), size(-1)/2 all "
      "multiples of ",
      kVecFloats,
      ", got stride(0)=",
      state_cache.stride(0),
      ", stride(1)=",
      state_cache.stride(1),
      ", size(-1)=",
      state_cache.size(-1));

  // Each work-item writes its NOPE bytes as one kSparseStoreQuad-wide store,
  // so every token payload must start on that boundary.
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(kc) % kSparseStoreQuad == 0 &&
          kv_block_stride % kSparseStoreQuad == 0 &&
          token_stride % kSparseStoreQuad == 0,
      kOpName,
      ": k_cache data pointer, kv_block_stride and token_stride must all be ",
      kSparseStoreQuad,
      "-byte aligned for the vectorized NOPE store, got kv_block_stride=",
      kv_block_stride,
      ", token_stride=",
      token_stride);

  // The RMSNorm weight dtype is the only thing that differs between the two
  // branches, so the launch lives in a generic lambda and everything else is
  // captured. Threading 20+ positional arguments through a helper instead
  // would make a misordered argument invisible at the call site.
  const auto launch = [&](const auto* rw) {
    using WT = std::remove_const_t<std::remove_pointer_t<decltype(rw)>>;
    using K = vllm::sparse_kv_insert_fp8mix_generic_kernel<WT>;
    constexpr int WG = K::WG_SIZE;
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(num_tokens * WG, WG),
          K{sc,
            state_cache.stride(0),
            state_cache.stride(1),
            state_width,
            tr,
            pos,
            sm,
            bt,
            block_table.stride(0),
            state_block_size,
            rw,
            eps,
            cs,
            cos_sin_cache.stride(0),
            kc,
            kvs,
            static_cast<int>(kv_cache_block_size),
            kv_block_stride,
            static_cast<int>(token_stride),
            static_cast<int>(scale_dim),
            static_cast<int>(compress_ratio),
            static_cast<int>(overlap),
            static_cast<int>(rope_head_dim)});
    });
  };

  if (rms_norm_weight.scalar_type() == torch::kBFloat16) {
    using bf16_t = sycl::ext::oneapi::bfloat16;
    launch(
        reinterpret_cast<const bf16_t*>(
            rms_norm_weight.data_ptr<at::BFloat16>()));
  } else {
    launch(rms_norm_weight.data_ptr<float>());
  }
}
