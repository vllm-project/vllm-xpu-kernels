// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// SYCL port of the DeepSeek-V4 C4 KV insert kernel (fp8mix output).
//
// Runtime dispatch keeps two paths:
// 1) split-gather 2-pass for small num_tokens under fixed fp8mix layout
// 2) generic one-pass for all other cases
//
// Data layout contract: fp8 NOPE + bf16 rope tail + UE8M0 scales.

#include <sycl/sycl.hpp>

#include "ops.h"
#include "utils.h"

#include <ATen/DeviceGuard.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/xpu/XPUFunctions.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace vllm {

namespace {
constexpr float kE4m3FnMax = 448.0f;
constexpr float kUe8m0ExpBias = 127.0f;
constexpr float kQuantAbsmaxFloor = 1.0e-4f;
}  // namespace

// Runtime-parameterized generic fp8mix path.
// This path keeps the same cache layout contract but does not rely on
// compile-time (CR, OL, rope) specializations.
// WT is the RMSNorm weight scalar type (float or bfloat16); it is upcast to
// fp32 per-element inside the kernel, mirroring the Triton load semantics.
template <typename WT>
class c4_kv_insert_fp8mix_generic_kernel {
 public:
  static constexpr int HEAD_SIZE = 512;
  static constexpr int WG_SIZE = 128;
  static constexpr int SG_SIZE = 32;
  static constexpr int PER_WI = HEAD_SIZE / WG_SIZE;  // 4
  static constexpr int QUANT_BLOCK = 64;
  static constexpr float FP8_MAX = kE4m3FnMax;
  static constexpr float INV_FP8_MAX = 1.0f / FP8_MAX;
  static constexpr float NEG_LARGE = -1.0e30f;

  using bf16_t = sycl::ext::oneapi::bfloat16;
  using fp8_t = at::Float8_e4m3fn;

  struct local_mem_t {
    int64_t meta[4];
    bf16_t normed[HEAD_SIZE];
  };

  c4_kv_insert_fp8mix_generic_kernel(
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

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(
      sycl::nd_item<1> it) const {
    const int token = it.get_group(0);
    const int lid = it.get_local_id(0);
    const int dim_base = lid * PER_WI;

    auto& smem =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<local_mem_t>(
            it.get_group());

    if (lid == 0) {
      smem.meta[0] = slot_mapping_[token];
      smem.meta[1] = positions_[token];
      smem.meta[2] = kv_slot_mapping_[token];
      smem.meta[3] = static_cast<int64_t>(token_to_req_[token]);
    }
    sycl::group_barrier(it.get_group());

    const int64_t slot_id = smem.meta[0];
    const int64_t position = smem.meta[1];
    const int64_t kv_slot_idx = smem.meta[2];
    const int32_t req_idx = static_cast<int32_t>(smem.meta[3]);

    const bool should_store =
        (slot_id >= 0) &&
        (((position + 1) % static_cast<int64_t>(compress_ratio_)) == 0) &&
        (kv_slot_idx >= 0);

    const int64_t n_gather =
        static_cast<int64_t>(1 + overlap_) * static_cast<int64_t>(compress_ratio_);
    const int64_t start = position - n_gather + 1;

    float m_run[PER_WI], s_run[PER_WI], acc[PER_WI];
#pragma unroll
    for (int j = 0; j < PER_WI; ++j) {
      m_run[j] = NEG_LARGE;
      s_run[j] = 0.f;
      acc[j] = 0.f;
    }

    for (int64_t gather_idx = 0; gather_idx < n_gather; ++gather_idx) {
      const int64_t gather_pos = start + gather_idx;
      const bool valid = gather_pos >= 0;
      const int64_t safe_pos = valid ? gather_pos : 0;
      const int head_offset = (gather_idx >= compress_ratio_) ? HEAD_SIZE : 0;

      const int64_t logical = safe_pos / state_cache_block_size_;
      const int32_t state_block = block_table_
          [static_cast<int64_t>(req_idx) * block_table_stride0_ + logical];
      const int64_t pos_in_block = safe_pos % state_cache_block_size_;
      const float* row = state_cache_ +
          static_cast<int64_t>(state_block) * state_cache_stride0_ +
          pos_in_block * state_cache_stride1_;

#pragma unroll
      for (int j = 0; j < PER_WI; ++j) {
        const int h = dim_base + j;
        const float score =
            valid ? row[state_width_ + head_offset + h] : NEG_LARGE;
        const float kv = valid ? row[head_offset + h] : 0.f;
        const float new_m = sycl::max(m_run[j], score);
        const float alpha = sycl::exp(m_run[j] - new_m);
        const float p = sycl::exp(score - new_m);
        s_run[j] = s_run[j] * alpha + p;
        acc[j] = acc[j] * alpha + p * kv;
        m_run[j] = new_m;
      }
    }

    float ckv[PER_WI];
#pragma unroll
    for (int j = 0; j < PER_WI; ++j) {
      ckv[j] = (s_run[j] > 0.f) ? (acc[j] / s_run[j]) : 0.f;
    }

    float ps = 0.f;
#pragma unroll
    for (int j = 0; j < PER_WI; ++j) {
      ps += ckv[j] * ckv[j];
    }
    const float ss =
        sycl::reduce_over_group(it.get_group(), ps, sycl::plus<float>());
    const float inv = sycl::rsqrt(ss / static_cast<float>(HEAD_SIZE) + rms_eps_);

    const int nope_head_dim = HEAD_SIZE - rope_head_dim_;
    const int nope_pairs = nope_head_dim / 2;
    const int half_rope = rope_head_dim_ / 2;

    float normed[PER_WI];
#pragma unroll
    for (int j = 0; j < PER_WI; ++j) {
      const int h = dim_base + j;
      normed[j] = ckv[j] * inv * static_cast<float>(rms_w_[h]);
    }

    {
      const int64_t compressed_pos =
          (position / static_cast<int64_t>(compress_ratio_)) *
          static_cast<int64_t>(compress_ratio_);
      const float* rope_base =
          cos_sin_cache_ + compressed_pos * cos_sin_stride0_;
#pragma unroll
      for (int p = 0; p < PER_WI / 2; ++p) {
        const int pair_idx = (dim_base / 2) + p;
        const int rope_pair_local = pair_idx - nope_pairs;
        if (rope_pair_local >= 0 && rope_pair_local < half_rope) {
          const int cs_idx = rope_pair_local;
          const float cv = rope_base[cs_idx];
          const float sv = rope_base[half_rope + cs_idx];
          const float even = normed[p * 2];
          const float odd = normed[p * 2 + 1];
          normed[p * 2] = even * cv - odd * sv;
          normed[p * 2 + 1] = odd * cv + even * sv;
        }
      }
    }

#pragma unroll
    for (int j = 0; j < PER_WI; ++j) {
      const int h = dim_base + j;
      smem.normed[h] = static_cast<bf16_t>(normed[j]);
    }

    sycl::group_barrier(it.get_group());

    if (!should_store) return;

    const int64_t kv_block = kv_slot_idx / kv_cache_block_size_;
    const int64_t kv_pos = kv_slot_idx % kv_cache_block_size_;
    uint8_t* block_base = k_cache_ + kv_block * kv_block_stride_;
    uint8_t* fp8_ptr = block_base + kv_pos * token_stride_;
    uint8_t* scale_ptr =
        block_base + kv_cache_block_size_ * token_stride_ + kv_pos * scale_dim_;

    // Runtime-safe parallel NOPE quantization:
    // - 16 WIs per 64-element quant block (PER_WI=4)
    // - tail elements in the last block are masked
    // - scale bytes beyond real blocks are padded with zero
    constexpr int WIS_PER_BLOCK = QUANT_BLOCK / PER_WI;  // 16
    const int scale_blocks = (nope_head_dim + QUANT_BLOCK - 1) / QUANT_BLOCK;
    const int total_quant_wis = scale_blocks * WIS_PER_BLOCK;

    if (lid < total_quant_wis) {
      const int b = lid / WIS_PER_BLOCK;
      const int wi_in_block = lid % WIS_PER_BLOCK;
      const int base = b * QUANT_BLOCK + wi_in_block * PER_WI;

      auto sg = it.get_sub_group();
      const int sg_lid = static_cast<int>(sg.get_local_linear_id());
      const int half_leader =
          (sg_lid / WIS_PER_BLOCK) * WIS_PER_BLOCK;  // 0 or 16 inside subgroup

      float block_absmax = 0.f;
#pragma unroll
      for (int j = 0; j < PER_WI; ++j) {
        const int i = base + j;
        if (i < nope_head_dim) {
          block_absmax = sycl::max(
              block_absmax,
              sycl::fabs(static_cast<float>(smem.normed[i])));
        }
      }

      // 16-lane reduction inside the half-subgroup.
#pragma unroll
      for (int mask = WIS_PER_BLOCK / 2; mask > 0; mask >>= 1) {
        const float other = sycl::permute_group_by_xor(sg, block_absmax, mask);
        block_absmax = sycl::max(block_absmax, other);
      }

      float inv_scale_local = 0.f;
      if (sg_lid == half_leader) {
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

#pragma unroll
      for (int j = 0; j < PER_WI; ++j) {
        const int i = base + j;
        if (i < nope_head_dim) {
          float x = static_cast<float>(smem.normed[i]) * inv_scale;
          x = sycl::clamp(x, -FP8_MAX, FP8_MAX);
          const fp8_t q = static_cast<fp8_t>(x);
          fp8_ptr[i] = sycl::bit_cast<uint8_t>(q);
        }
      }
    }

    for (int b = lid; b < scale_dim_; b += WG_SIZE) {
      if (b >= scale_blocks) {
        scale_ptr[b] = static_cast<uint8_t>(0);
      }
    }

    auto* rope_dst = reinterpret_cast<bf16_t*>(fp8_ptr + nope_head_dim);
    for (int i = lid; i < rope_head_dim_; i += WG_SIZE) {
      rope_dst[i] = smem.normed[nope_head_dim + i];
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

// ---------------------------------------------------------------------------
// Split-gather 2-pass for small num_tokens (Layer B decode path).
//
// The single-WG-per-token kernel above is launch-bound at nt=1
// (1/20 Xe-cores busy, ~125 us). We split the N_GATHER axis across
// G_SHARDS extra WGs per token: pass A writes per-shard partial
// (m, s, acc) triplets to a scratch tensor; pass B reads the G_SHARDS
// triplets, online-merges them, then runs RMSNorm + RoPE + bf16 store.
//
// Scratch layout (fp32, contiguous): [num_tokens, G_SHARDS, 3, HEAD_SIZE]
//   - [t, g, 0, h] = m partial
//   - [t, g, 1, h] = s partial
//   - [t, g, 2, h] = acc partial
// ---------------------------------------------------------------------------

template <int COMPRESS_RATIO, int OVERLAP, int G_SHARDS>
class c4_kv_split_gather_kernel {
 public:
  static constexpr int HEAD_SIZE = 512;
  static constexpr int N_GATHER = (1 + OVERLAP) * COMPRESS_RATIO;
  static constexpr int WG_SIZE = 128;
  static constexpr int SG_SIZE = 32;
  static constexpr int PER_WI = HEAD_SIZE / WG_SIZE;  // 4
  static constexpr int ROWS_PER_SHARD = N_GATHER / G_SHARDS;
  static_assert(N_GATHER % G_SHARDS == 0, "G_SHARDS must divide N_GATHER");
  static_assert(ROWS_PER_SHARD >= 1, "");
  static constexpr float NEG_LARGE = -1.0e30f;

  c4_kv_split_gather_kernel(
      const float* state_cache,
      int64_t state_cache_stride0,
      int64_t state_cache_stride1,
      int state_width,
      const int32_t* token_to_req,
      const int64_t* positions,
      const int32_t* block_table,
      int64_t block_table_stride0,
      int state_cache_block_size,
      float* scratch,          // [nt, G, 3, HEAD]
      int64_t scratch_stride0,  // = G * 3 * HEAD
      int64_t scratch_stride1)  // = 3 * HEAD
      : state_cache_(state_cache),
        state_cache_stride0_(state_cache_stride0),
        state_cache_stride1_(state_cache_stride1),
        state_width_(state_width),
        token_to_req_(token_to_req),
        positions_(positions),
        block_table_(block_table),
        block_table_stride0_(block_table_stride0),
        state_cache_block_size_(state_cache_block_size),
        scratch_(scratch),
        scratch_stride0_(scratch_stride0),
        scratch_stride1_(scratch_stride1) {}

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(
      sycl::nd_item<2> it) const {
    const int token = it.get_group(0);
    const int shard = it.get_group(1);
    const int lid = it.get_local_id(1);
    const int dim_base = lid * PER_WI;

    // Per-token metadata broadcast via SLM (only need position+req here).
    auto& meta =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int64_t[2]>(
            it.get_group());
    if (lid == 0) {
      meta[0] = positions_[token];
      meta[1] = static_cast<int64_t>(token_to_req_[token]);
    }
    sycl::group_barrier(it.get_group());
    const int64_t position = meta[0];
    const int32_t req_idx = static_cast<int32_t>(meta[1]);

    const int64_t start = position - N_GATHER + 1;
    const int row_lo = shard * ROWS_PER_SHARD;

    float m_run[PER_WI], s_run[PER_WI], acc[PER_WI];
#pragma unroll
    for (int j = 0; j < PER_WI; ++j) {
      m_run[j] = NEG_LARGE;
      s_run[j] = 0.f;
      acc[j] = 0.f;
    }

    for (int r = 0; r < ROWS_PER_SHARD; ++r) {
      const int gather_idx = row_lo + r;
      const int64_t gather_pos = start + gather_idx;
      const bool valid = gather_pos >= 0;
      const int64_t safe_pos = valid ? gather_pos : 0;
      const int head_offset =
          (gather_idx >= COMPRESS_RATIO) ? HEAD_SIZE : 0;

      const int64_t logical = safe_pos / state_cache_block_size_;
      const int32_t state_block = block_table_
          [static_cast<int64_t>(req_idx) * block_table_stride0_ + logical];
      const int64_t pos_in_block = safe_pos % state_cache_block_size_;
      const float* row = state_cache_ +
          static_cast<int64_t>(state_block) * state_cache_stride0_ +
          pos_in_block * state_cache_stride1_;

#pragma unroll
      for (int j = 0; j < PER_WI; ++j) {
        const int h = dim_base + j;
        const float score =
            valid ? row[state_width_ + head_offset + h] : NEG_LARGE;
        const float kv = valid ? row[head_offset + h] : 0.f;
        const float new_m = sycl::max(m_run[j], score);
        const float alpha = sycl::exp(m_run[j] - new_m);
        const float p = sycl::exp(score - new_m);
        s_run[j] = s_run[j] * alpha + p;
        acc[j] = acc[j] * alpha + p * kv;
        m_run[j] = new_m;
      }
    }

    // Write partial (m, s, acc) for this shard to scratch.
    float* base = scratch_ +
        static_cast<int64_t>(token) * scratch_stride0_ +
        static_cast<int64_t>(shard) * scratch_stride1_;
#pragma unroll
    for (int j = 0; j < PER_WI; ++j) {
      const int h = dim_base + j;
      base[0 * HEAD_SIZE + h] = m_run[j];
      base[1 * HEAD_SIZE + h] = s_run[j];
      base[2 * HEAD_SIZE + h] = acc[j];
    }
  }

 private:
  const float* state_cache_;
  int64_t state_cache_stride0_;
  int64_t state_cache_stride1_;
  int state_width_;
  const int32_t* token_to_req_;
  const int64_t* positions_;
  const int32_t* block_table_;
  int64_t block_table_stride0_;
  int state_cache_block_size_;
  float* scratch_;
  int64_t scratch_stride0_;
  int64_t scratch_stride1_;
};

template <int COMPRESS_RATIO, int ROPE_HEAD_DIM_C, int G_SHARDS, typename WT>
class c4_kv_finalize_fp8mix_kernel {
 public:
  static constexpr int HEAD_SIZE = 512;
  static constexpr int WG_SIZE = 128;
  static constexpr int SG_SIZE = 32;
  static constexpr int PER_WI = HEAD_SIZE / WG_SIZE;  // 4
  static constexpr int NOPE_HEAD_DIM = HEAD_SIZE - ROPE_HEAD_DIM_C;
  static constexpr int HALF_ROPE = ROPE_HEAD_DIM_C / 2;
  static constexpr int NOPE_PAIRS = NOPE_HEAD_DIM / 2;
  static constexpr int QUANT_BLOCK = 64;
  static constexpr int NOPE_BLOCKS = NOPE_HEAD_DIM / QUANT_BLOCK;  // 7
  static constexpr float FP8_MAX = kE4m3FnMax;
  static constexpr float INV_FP8_MAX = 1.0f / FP8_MAX;
  static constexpr float NEG_LARGE = -1.0e30f;

  using bf16_t = sycl::ext::oneapi::bfloat16;
  using fp8_t = at::Float8_e4m3fn;

  struct local_mem_t {
    int64_t meta[3];
    bf16_t pre_nope[NOPE_HEAD_DIM];
    bf16_t post_rope[ROPE_HEAD_DIM_C];
  };

  c4_kv_finalize_fp8mix_kernel(
      const float* scratch,
      int64_t scratch_stride0,
      int64_t scratch_stride1,
      const int64_t* positions,
      const int64_t* slot_mapping,
      const int64_t* kv_slot_mapping,
      const WT* rms_w,
      float rms_eps,
      const float* cos_sin_cache,
      int64_t cos_sin_stride0,
      uint8_t* k_cache,
      int kv_cache_block_size,
      int64_t kv_block_stride,
      int token_stride,
      int scale_dim)
      : scratch_(scratch),
        scratch_stride0_(scratch_stride0),
        scratch_stride1_(scratch_stride1),
        positions_(positions),
        slot_mapping_(slot_mapping),
        kv_slot_mapping_(kv_slot_mapping),
        rms_w_(rms_w),
        rms_eps_(rms_eps),
        cos_sin_cache_(cos_sin_cache),
        cos_sin_stride0_(cos_sin_stride0),
        k_cache_(k_cache),
        kv_cache_block_size_(kv_cache_block_size),
        kv_block_stride_(kv_block_stride),
        token_stride_(token_stride),
        scale_dim_(scale_dim) {}

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(
      sycl::nd_item<1> it) const {
    const int token = it.get_group(0);
    const int lid = it.get_local_id(0);
    const int dim_base = lid * PER_WI;

    auto& smem =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<local_mem_t>(
            it.get_group());
    if (lid == 0) {
      smem.meta[0] = slot_mapping_[token];
      smem.meta[1] = positions_[token];
      smem.meta[2] = kv_slot_mapping_[token];
    }
    sycl::group_barrier(it.get_group());
    const int64_t slot_id = smem.meta[0];
    const int64_t position = smem.meta[1];
    const int64_t kv_slot_idx = smem.meta[2];
    const bool should_store = (slot_id >= 0) &&
        (((position + 1) % COMPRESS_RATIO) == 0) && (kv_slot_idx >= 0);

    // Online-merge G_SHARDS partials in registers.
    float m_run[PER_WI], s_run[PER_WI], acc[PER_WI];
#pragma unroll
    for (int j = 0; j < PER_WI; ++j) {
      m_run[j] = NEG_LARGE;
      s_run[j] = 0.f;
      acc[j] = 0.f;
    }

    const float* tok_base =
        scratch_ + static_cast<int64_t>(token) * scratch_stride0_;
    for (int g = 0; g < G_SHARDS; ++g) {
      const float* shard_base = tok_base + g * scratch_stride1_;
#pragma unroll
      for (int j = 0; j < PER_WI; ++j) {
        const int h = dim_base + j;
        const float m_g = shard_base[0 * HEAD_SIZE + h];
        const float s_g = shard_base[1 * HEAD_SIZE + h];
        const float a_g = shard_base[2 * HEAD_SIZE + h];
        const float new_m = sycl::max(m_run[j], m_g);
        const float a_r = sycl::exp(m_run[j] - new_m);
        const float a_c = sycl::exp(m_g - new_m);
        acc[j] = acc[j] * a_r + a_g * a_c;
        s_run[j] = s_run[j] * a_r + s_g * a_c;
        m_run[j] = new_m;
      }
    }

    float ckv[PER_WI];
#pragma unroll
    for (int j = 0; j < PER_WI; ++j) {
      ckv[j] = (s_run[j] > 0.f) ? (acc[j] / s_run[j]) : 0.f;
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
      const int h = dim_base + j;
      normed[j] = ckv[j] * inv * static_cast<float>(rms_w_[h]);
      // Triton parity: FP8 quant path uses bf16 roundtrip before quant.
      if (h < NOPE_HEAD_DIM) {
        smem.pre_nope[h] = static_cast<bf16_t>(normed[j]);
      }
    }

    {
      const int64_t compressed_pos =
          (position / COMPRESS_RATIO) * COMPRESS_RATIO;
      const float* rope_base =
          cos_sin_cache_ + compressed_pos * cos_sin_stride0_;
#pragma unroll
      for (int p = 0; p < PER_WI / 2; ++p) {
        const int pair_idx = (dim_base / 2) + p;
        const int rope_pair_local = pair_idx - NOPE_PAIRS;
        if (rope_pair_local >= 0) {
          const int cs_idx = rope_pair_local;
          const float cv = rope_base[cs_idx];
          const float sv = rope_base[HALF_ROPE + cs_idx];
          const float even = normed[p * 2];
          const float odd = normed[p * 2 + 1];
          normed[p * 2] = even * cv - odd * sv;
          normed[p * 2 + 1] = odd * cv + even * sv;
        }
      }
    }

#pragma unroll
    for (int j = 0; j < PER_WI; ++j) {
      const int h = dim_base + j;
      if (h >= NOPE_HEAD_DIM) {
        smem.post_rope[h - NOPE_HEAD_DIM] = static_cast<bf16_t>(normed[j]);
      }
    }

    sycl::group_barrier(it.get_group());

    if (!should_store) {
      return;
    }

    const int64_t kv_block = kv_slot_idx / kv_cache_block_size_;
    const int64_t kv_pos = kv_slot_idx % kv_cache_block_size_;
    uint8_t* block_base = k_cache_ + kv_block * kv_block_stride_;
    uint8_t* fp8_ptr = block_base + kv_pos * token_stride_;
    uint8_t* scale_ptr =
        block_base + kv_cache_block_size_ * token_stride_ + kv_pos * scale_dim_;

    // Parallelize NOPE quantization over 7 blocks x 16 WIs (112 WIs total).
    // Each WI handles 4 contiguous elements for both absmax partial and write.
    constexpr int WIS_PER_BLOCK = QUANT_BLOCK / PER_WI;  // 16
    constexpr int TOTAL_QUANT_WIS = NOPE_BLOCKS * WIS_PER_BLOCK;  // 112

    if (lid < TOTAL_QUANT_WIS) {
      const int b = lid / WIS_PER_BLOCK;
      const int wi_in_block = lid % WIS_PER_BLOCK;
      const int base = b * QUANT_BLOCK + wi_in_block * PER_WI;
      auto sg = it.get_sub_group();
      const int sg_lid = static_cast<int>(sg.get_local_linear_id());
      const int half_leader =
          (sg_lid / WIS_PER_BLOCK) * WIS_PER_BLOCK;  // 0 or 16

      float block_absmax = 0.f;
#pragma unroll
      for (int j = 0; j < PER_WI; ++j) {
        block_absmax =
            sycl::max(block_absmax,
                      sycl::fabs(static_cast<float>(smem.pre_nope[base + j])));
      }

      // 16-lane reduction inside half-subgroup (lane bits [0..3]).
#pragma unroll
      for (int mask = WIS_PER_BLOCK / 2; mask > 0; mask >>= 1) {
        const float other = sycl::permute_group_by_xor(sg, block_absmax, mask);
        block_absmax = sycl::max(block_absmax, other);
      }

      float inv_scale_local = 0.f;
      if (sg_lid == half_leader) {
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

      float x0 = static_cast<float>(smem.pre_nope[base + 0]) * inv_scale;
      float x1 = static_cast<float>(smem.pre_nope[base + 1]) * inv_scale;
      float x2 = static_cast<float>(smem.pre_nope[base + 2]) * inv_scale;
      float x3 = static_cast<float>(smem.pre_nope[base + 3]) * inv_scale;
      x0 = sycl::clamp(x0, -FP8_MAX, FP8_MAX);
      x1 = sycl::clamp(x1, -FP8_MAX, FP8_MAX);
      x2 = sycl::clamp(x2, -FP8_MAX, FP8_MAX);
      x3 = sycl::clamp(x3, -FP8_MAX, FP8_MAX);

      const fp8_t q0 = static_cast<fp8_t>(x0);
      const fp8_t q1 = static_cast<fp8_t>(x1);
      const fp8_t q2 = static_cast<fp8_t>(x2);
      const fp8_t q3 = static_cast<fp8_t>(x3);
      fp8_ptr[base + 0] = sycl::bit_cast<uint8_t>(q0);
      fp8_ptr[base + 1] = sycl::bit_cast<uint8_t>(q1);
      fp8_ptr[base + 2] = sycl::bit_cast<uint8_t>(q2);
      fp8_ptr[base + 3] = sycl::bit_cast<uint8_t>(q3);
    }

    // scale pad byte (7 real + 1 pad for head=512/nope=448, block=64)
    if (lid == 0) {
      scale_ptr[NOPE_BLOCKS] = static_cast<uint8_t>(0);
    }

    // Store rotated rope tail as bf16 bytes immediately after fp8 NOPE payload.
    auto* rope_dst = reinterpret_cast<bf16_t*>(fp8_ptr + NOPE_HEAD_DIM);
    for (int i = lid; i < ROPE_HEAD_DIM_C; i += WG_SIZE) {
      rope_dst[i] = smem.post_rope[i];
    }
  }

 private:
  const float* scratch_;
  int64_t scratch_stride0_;
  int64_t scratch_stride1_;
  const int64_t* positions_;
  const int64_t* slot_mapping_;
  const int64_t* kv_slot_mapping_;
  const WT* rms_w_;
  float rms_eps_;
  const float* cos_sin_cache_;
  int64_t cos_sin_stride0_;
  uint8_t* k_cache_;
  int kv_cache_block_size_;
  int64_t kv_block_stride_;
  int token_stride_;
  int scale_dim_;
};
}  // namespace vllm

namespace {

template <typename WT>
void launch_one_fp8mix_generic(
    sycl::queue& q,
    int64_t num_tokens,
    const float* state_cache,
    int64_t s0,
    int64_t s1,
    int state_width,
    const int32_t* tokenreq,
    const int64_t* positions,
    const int64_t* slot_mapping,
    const int32_t* block_table,
    int64_t bt0,
    int state_blk,
    const WT* rms_w,
    float rms_eps,
    const float* cos_sin,
    int64_t cs0,
    uint8_t* k_cache,
    const int64_t* kv_slot,
    int kv_blk,
    int64_t kv_block_stride,
    int token_stride,
    int scale_dim,
    int compress_ratio,
    int overlap,
    int rope_head_dim) {

  using K = vllm::c4_kv_insert_fp8mix_generic_kernel<WT>;
  constexpr int WG = K::WG_SIZE;
  at::DeviceGuard dg(at::Device(at::kXPU, at::xpu::current_device()));
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(num_tokens * WG, WG),
        K{state_cache,
          s0,
          s1,
          state_width,
          tokenreq,
          positions,
          slot_mapping,
          block_table,
          bt0,
          state_blk,
          rms_w,
          rms_eps,
          cos_sin,
          cs0,
          k_cache,
          kv_slot,
          kv_blk,
          kv_block_stride,
          token_stride,
          scale_dim,
          compress_ratio,
          overlap,
            rope_head_dim});
  });
}

// Keep per-invocation scratch tensors alive until the finalize kernel
// completes. This avoids reusing the same scratch region across overlapping
// calls, which can race under concurrent submissions.
struct split_gather_inflight_scratch_t {
  sycl::event done;
  at::Tensor scratch;
};

inline void reap_split_gather_inflight_locked(
    std::vector<split_gather_inflight_scratch_t>& inflight) {
  inflight.erase(
      std::remove_if(
          inflight.begin(),
          inflight.end(),
          [](const split_gather_inflight_scratch_t& e) {
            return e.done.get_info<sycl::info::event::command_execution_status>() ==
                sycl::info::event_command_status::complete;
          }),
      inflight.end());
}

inline void hold_split_gather_scratch_until_done(
    int dev,
    sycl::event done,
    at::Tensor scratch) {
  static std::mutex mtx;
  static std::unordered_map<int, std::vector<split_gather_inflight_scratch_t>>
      inflight_by_dev;

  std::lock_guard<std::mutex> lk(mtx);
  auto& inflight = inflight_by_dev[dev];
  reap_split_gather_inflight_locked(inflight);
  inflight.push_back({std::move(done), std::move(scratch)});
}

template <int CR, int OL, int ROPE_DIM, int G_SHARDS, typename WT>
void launch_split_gather_fp8mix(
    sycl::queue& q,
    int64_t num_tokens,
    const float* state_cache,
    int64_t s0,
    int64_t s1,
    int state_width,
    const int32_t* tokenreq,
    const int64_t* positions,
    const int64_t* slot_mapping,
    const int32_t* block_table,
    int64_t bt0,
    int state_blk,
    const WT* rms_w,
    float rms_eps,
    const float* cos_sin,
    int64_t cs0,
    uint8_t* k_cache,
    const int64_t* kv_slot,
    int kv_blk,
    int64_t kv_block_stride,
    int token_stride,
    int scale_dim) {
  using KA = vllm::c4_kv_split_gather_kernel<CR, OL, G_SHARDS>;
  using KB = vllm::c4_kv_finalize_fp8mix_kernel<CR, ROPE_DIM, G_SHARDS, WT>;
  constexpr int HEAD = KA::HEAD_SIZE;
  constexpr int WG = KA::WG_SIZE;

  at::DeviceGuard dg(at::Device(at::kXPU, at::xpu::current_device()));
    const int dev = at::xpu::current_device();

    // Scratch [num_tokens, G_SHARDS, 3, HEAD] fp32 — per invocation.
  const int64_t scratch_numel =
      num_tokens * static_cast<int64_t>(G_SHARDS) * 3 *
      static_cast<int64_t>(HEAD);
    at::Tensor scratch = at::empty(
      {scratch_numel},
      at::TensorOptions().dtype(at::kFloat).device(at::kXPU, dev));
    float* scr = scratch.data_ptr<float>();
  const int64_t scr0 = G_SHARDS * 3 * HEAD;
  const int64_t scr1 = 3 * HEAD;

  // Pass A: grid = (num_tokens, G_SHARDS * WG), local = (1, WG).
    sycl::event pass_a = q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(num_tokens, G_SHARDS * WG),
            sycl::range<2>(1, WG)),
        KA{state_cache,
           s0,
           s1,
           state_width,
           tokenreq,
           positions,
           block_table,
           bt0,
           state_blk,
           scr,
           scr0,
           scr1});
  });

  // Pass B: grid = (num_tokens * WG,), local = (WG,).
  sycl::event pass_b = q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(pass_a);
    cgh.parallel_for(
        sycl::nd_range<1>(num_tokens * WG, WG),
        KB{scr,
           scr0,
           scr1,
           positions,
           slot_mapping,
           kv_slot,
           rms_w,
           rms_eps,
           cos_sin,
           cs0,
           k_cache,
           kv_blk,
           kv_block_stride,
           token_stride,
           scale_dim});
  });

  hold_split_gather_scratch_until_done(dev, std::move(pass_b), std::move(scratch));
}

// Path selection shared by both weight dtypes. WT is the RMSNorm weight scalar
// type (float or bfloat16); the conversion to fp32 happens inside the kernel.
template <typename WT>
void dispatch_fp8mix(
    sycl::queue& q,
    int64_t num_tokens,
    const float* sc,
    int64_t scs0,
    int64_t scs1,
    int state_width,
    const int32_t* tr,
    const int64_t* pos,
    const int64_t* sm,
    const int32_t* bt,
    int64_t bt0,
    int state_block_size,
    const WT* rw,
    float rms_eps,
    const float* cs,
    int64_t cs0,
    uint8_t* kc,
    const int64_t* kvs,
    int kv_cache_block_size,
    int64_t kv_block_stride,
    int token_stride,
    int scale_dim,
    int compress_ratio,
    int overlap,
    int rope_head_dim) {
  static const bool kForceGeneric = []() {
    const char* e = std::getenv("VLLM_C4_KV_FORCE_GENERIC");
    return e && e[0] == '1';
  }();

  const bool can_use_split_gather_layout =
      (rope_head_dim == 64) && (token_stride == 576) && (scale_dim == 8);

  // Only two runtime paths:
  // 1) split-gather for B small-token under fixed layout
  // 2) generic one-pass for all other cases
  if (!kForceGeneric && can_use_split_gather_layout && compress_ratio == 128 &&
      overlap == 0) {
    constexpr int64_t kSplitGatherThreshold = 96;
    static const bool kDisableSplit = []() {
      const char* e = std::getenv("VLLM_C4_KV_DISABLE_SPLIT_GATHER");
      return e && e[0] == '1';
    }();

    if (!kDisableSplit && num_tokens < kSplitGatherThreshold) {
      launch_split_gather_fp8mix<128, 0, 64, 8, WT>(
          q, num_tokens, sc, scs0, scs1, state_width, tr, pos, sm, bt, bt0,
          state_block_size, rw, rms_eps, cs, cs0, kc, kvs,
          kv_cache_block_size, kv_block_stride, token_stride, scale_dim);
      return;
    }
  }

  launch_one_fp8mix_generic<WT>(
      q, num_tokens, sc, scs0, scs1, state_width, tr, pos, sm, bt, bt0,
      state_block_size, rw, rms_eps, cs, cs0, kc, kvs, kv_cache_block_size,
      kv_block_stride, token_stride, scale_dim, compress_ratio, overlap,
      rope_head_dim);
}

}  // namespace
void xpu_compress_insert_fp8mix(
    torch::Tensor const& state_cache,
    torch::Tensor const& token_to_req_indices,
    torch::Tensor const& positions,
    torch::Tensor const& slot_mapping,
    torch::Tensor const& block_table,
    torch::Tensor const& rms_norm_weight,
    double rms_norm_eps,
    torch::Tensor const& cos_sin_cache,
    torch::Tensor& k_cache,
    torch::Tensor const& kv_slot_mapping,
    int64_t kv_cache_block_size,
    int64_t compress_ratio,
    int64_t overlap,
    int64_t rope_head_dim,
    int64_t token_stride,
    int64_t scale_dim,
    int64_t kv_block_stride) {
  TORCH_CHECK(
      state_cache.dtype() == torch::kFloat32,
      "xpu_compress_insert_fp8mix SYCL: state_cache must be fp32");
  TORCH_CHECK(k_cache.dtype() == torch::kUInt8, "k_cache must be uint8");
  TORCH_CHECK(
      rms_norm_weight.scalar_type() == torch::kFloat32 ||
          rms_norm_weight.scalar_type() == torch::kBFloat16,
      "xpu_compress_insert_fp8mix: rms_norm_weight must be float32 or bfloat16");
  TORCH_CHECK(cos_sin_cache.dtype() == torch::kFloat32);
  TORCH_CHECK(positions.dtype() == torch::kInt64);
  TORCH_CHECK(slot_mapping.dtype() == torch::kInt64);
  TORCH_CHECK(kv_slot_mapping.dtype() == torch::kInt64);
  TORCH_CHECK(token_to_req_indices.dtype() == torch::kInt32);
  TORCH_CHECK(block_table.dtype() == torch::kInt32);
  TORCH_CHECK(state_cache.dim() == 3);
  TORCH_CHECK(
      k_cache.dim() == 3,
      "xpu_compress_insert_fp8mix: k_cache must be rank-3 [num_blocks, block_size, payload], got dim=",
      k_cache.dim());
  TORCH_CHECK(
      k_cache.size(-1) >= token_stride + scale_dim,
      "xpu_compress_insert_fp8mix: k_cache last dim too small, got ",
      k_cache.size(-1),
      ", required >= token_stride + scale_dim = ",
      token_stride,
      " + ",
      scale_dim,
      " = ",
      (token_stride + scale_dim));
      TORCH_CHECK(
        k_cache.stride(2) == 1,
        "xpu_compress_insert_fp8mix: k_cache must be byte-contiguous on last dim, got stride(2)=",
        k_cache.stride(2));
      TORCH_CHECK(
          k_cache.stride(1) >= token_stride,
          "xpu_compress_insert_fp8mix: unsupported k_cache layout. Need stride(1) >= token_stride so each token payload has enough contiguous bytes, got stride(1)=",
        k_cache.stride(1),
        ", token_stride=",
          token_stride);
      TORCH_CHECK(
        kv_block_stride == k_cache.stride(0),
        "xpu_compress_insert_fp8mix: kv_block_stride must match k_cache.stride(0), got kv_block_stride=",
        kv_block_stride,
        ", k_cache.stride(0)=",
        k_cache.stride(0));
    TORCH_CHECK(compress_ratio > 0, "xpu_compress_insert_fp8mix: compress_ratio must be > 0");
    TORCH_CHECK(overlap >= 0 && overlap <= 1, "xpu_compress_insert_fp8mix: overlap must be 0 or 1");
    TORCH_CHECK(
      rope_head_dim >= 0 && rope_head_dim <= 512 && (rope_head_dim % 2 == 0),
      "xpu_compress_insert_fp8mix: rope_head_dim must be even and in [0, 512]");
    TORCH_CHECK(token_stride > 0, "xpu_compress_insert_fp8mix: token_stride must be > 0");
    TORCH_CHECK(scale_dim > 0, "xpu_compress_insert_fp8mix: scale_dim must be > 0");
    const int64_t nope_head_dim = 512 - rope_head_dim;
    const int64_t required_scale_blocks = (nope_head_dim + 63) / 64;
    TORCH_CHECK(
      scale_dim >= required_scale_blocks,
      "xpu_compress_insert_fp8mix: scale_dim too small for NOPE quant blocks");
    TORCH_CHECK(
      token_stride >= nope_head_dim + rope_head_dim * 2,
      "xpu_compress_insert_fp8mix: token_stride too small for fp8+rope payload");
  TORCH_CHECK(
      kv_block_stride >= kv_cache_block_size * (token_stride + scale_dim),
      "xpu_compress_insert_fp8mix: kv_block_stride too small for mixed layout");

  const int64_t num_tokens = positions.numel();
  if (num_tokens == 0) return;

  const int state_width = static_cast<int>(state_cache.size(-1) / 2);
  const int state_block_size = static_cast<int>(state_cache.size(1));

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

  // The RMSNorm weight is consumed (and upcast to fp32) inside the kernel,
  // so we dispatch on its native dtype instead of forcing a host-side
  // .float() conversion on the hot path (Triton parity).
  if (rms_norm_weight.scalar_type() == torch::kBFloat16) {
    using bf16_t = sycl::ext::oneapi::bfloat16;
    const auto* rw = reinterpret_cast<const bf16_t*>(
        rms_norm_weight.data_ptr<at::BFloat16>());
    dispatch_fp8mix<bf16_t>(
        q, num_tokens, sc, state_cache.stride(0), state_cache.stride(1),
        state_width, tr, pos, sm, bt, block_table.stride(0), state_block_size,
        rw, eps, cs, cos_sin_cache.stride(0), kc, kvs,
        static_cast<int>(kv_cache_block_size), kv_block_stride,
        static_cast<int>(token_stride), static_cast<int>(scale_dim),
        static_cast<int>(compress_ratio), static_cast<int>(overlap),
        static_cast<int>(rope_head_dim));
  } else {
    const float* rw = rms_norm_weight.data_ptr<float>();
    dispatch_fp8mix<float>(
        q, num_tokens, sc, state_cache.stride(0), state_cache.stride(1),
        state_width, tr, pos, sm, bt, block_table.stride(0), state_block_size,
        rw, eps, cs, cos_sin_cache.stride(0), kc, kvs,
        static_cast<int>(kv_cache_block_size), kv_block_stride,
        static_cast<int>(token_stride), static_cast<int>(scale_dim),
        static_cast<int>(compress_ratio), static_cast<int>(overlap),
        static_cast<int>(rope_head_dim));
  }
}
