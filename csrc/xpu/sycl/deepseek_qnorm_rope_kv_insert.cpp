// SPDX-License-Identifier: Apache-2.0
// Fused DeepSeek-V4 Q-RMSNorm + GPT-J RoPE + KV cache insert.
// One sub-group per (token, head_or_kv). Each kernel reads every lane's data
// from GDDR with a 4-byte stride between lanes for XPU-friendly coalescing,
// then performs normalization or quantization, RoPE, and cache stores.

#include <sycl/sycl.hpp>
#include "utils.h"
#include <c10/util/Float8_e4m3fn.h>
#include <cstdint>
#include <torch/all.h>

namespace vllm {

using bf16_t = sycl::ext::oneapi::bfloat16;

static constexpr float FP8_MAX = 448.0f;

// Register-only bf16x2 (un)packing; no address-taking, so the packed
// register arrays never spill to private memory.
inline float bf16x2_lo(uint32_t w) {
  return static_cast<float>(
      sycl::bit_cast<bf16_t>(static_cast<uint16_t>(w)));
}
inline float bf16x2_hi(uint32_t w) {
  return static_cast<float>(
      sycl::bit_cast<bf16_t>(static_cast<uint16_t>(w >> 16)));
}
inline uint32_t pack_bf16x2(float lo, float hi) {
  const uint16_t l = sycl::bit_cast<uint16_t>(static_cast<bf16_t>(lo));
  const uint16_t h = sycl::bit_cast<uint16_t>(static_cast<bf16_t>(hi));
  return static_cast<uint32_t>(l) | (static_cast<uint32_t>(h) << 16);
}

// Ceil to power-of-2 via IEEE754 bit hack; also returns biased exponent.
inline float fast_ceil_pow2(float x, uint32_t& out_scale_exp) {
  uint32_t bits = sycl::bit_cast<uint32_t>(x);
  if (bits & 0x7FFFFF) {
    bits = (bits + 0x800000) & 0xFF800000u;
  }
  out_scale_exp = bits >> 23;
  return sycl::bit_cast<float>(bits);
}

// fp8 e4m3 quantization via the plain c10 static_cast (no vISA asm).
inline uint32_t fp8_quant_cast(float v, float inv_scale) {
  float x = v * inv_scale;
  x = sycl::fmax(-FP8_MAX, sycl::fmin(x, FP8_MAX));
  return static_cast<at::Float8_e4m3fn>(x).x;
}

template <int HEAD_DIM, int ROPE_DIM>
void qnorm_rope_kv_insert_fp8_kernel_decode(
    sycl::queue& queue,
    bf16_t* __restrict__ q_,
    const bf16_t* __restrict__ kv_,
    uint8_t* __restrict__ cache_,
    const int64_t* __restrict__ slot_mapping_,
    const int64_t* __restrict__ position_ids_,
    const float* __restrict__ cos_sin_cache_,
    float eps_,
    int64_t num_tokens_,
    int64_t num_heads_q_,
    int64_t block_size_,
    int64_t q_stride_token_,
    int64_t q_stride_head_,
    int64_t kv_stride_token_,
    int64_t cache_block_stride_,
    int64_t cache_stride_pos_) {
  static constexpr int NOPE_DIM = HEAD_DIM - ROPE_DIM;
  static constexpr int HALF_ROPE = ROPE_DIM / 2;
  static constexpr int SG_SIZE = 32;
  static constexpr int WG_SIZE = 32;
  static constexpr int SUBGROUPS_PER_WG = WG_SIZE / SG_SIZE;
  // Fixed by the fp8_ds_mla cache format, not by the sub-group size.
  static constexpr int QUANT_BLOCK = 64;
  static constexpr int NUM_QUANT_BLOCKS = NOPE_DIM / QUANT_BLOCK;
  static constexpr int SCALE_BYTES_PER_TOKEN = NUM_QUANT_BLOCKS + 1;
  static constexpr int TOKEN_DATA_BYTES = NOPE_DIM + ROPE_DIM * 2;
  static constexpr int NOPE_ITERS = NOPE_DIM / (2 * SG_SIZE);
  static constexpr int ROPE_ITERS = HALF_ROPE / SG_SIZE;
  static_assert(
      NOPE_DIM % (2 * SG_SIZE) == 0,
      "NOPE_DIM must be divisible by 2*SG_SIZE");
  static_assert(
      HALF_ROPE % SG_SIZE == 0, "ROPE_DIM/2 must be divisible by SG_SIZE");
  static_assert(
      QUANT_BLOCK == 2 * SG_SIZE,
      "one register per lane must span exactly one quant block");
  static_assert(WG_SIZE % SG_SIZE == 0);
  const size_t total_subgroups =
      static_cast<size_t>(num_tokens_) * (num_heads_q_ + 1);
  const size_t num_workgroups =
      (total_subgroups + SUBGROUPS_PER_WG - 1) / SUBGROUPS_PER_WG;
  sycl::range<2> global(num_workgroups, WG_SIZE);
  sycl::range<2> local(1, WG_SIZE);

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>(global, local),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
          // 32-bit index math: 64-bit div/mod is emulated and slow on XPU.
          const uint32_t workgroup_id =
              static_cast<uint32_t>(item.get_group(0));
          const uint32_t subgroup_id =
              static_cast<uint32_t>(item.get_local_id(1)) / SG_SIZE;
          const sycl::sub_group sg = item.get_sub_group();
          const uint32_t lane =
              static_cast<uint32_t>(sg.get_local_linear_id());
          const uint32_t global_subgroup_id =
              workgroup_id * SUBGROUPS_PER_WG + subgroup_id;
          const uint32_t slots_per_token =
              static_cast<uint32_t>(num_heads_q_) + 1;
          const uint32_t token_idx = global_subgroup_id / slots_per_token;
          const uint32_t slot_idx = global_subgroup_id % slots_per_token;

          if (token_idx >= num_tokens_) return;
          const bool is_kv = (slot_idx == num_heads_q_);

          // Issue the longest dependent-load chain first: pos → cos/sin.
          const int64_t pos = position_ids_[token_idx];
          const int64_t slot_id =
              is_kv ? slot_mapping_[token_idx] : int64_t{0};

          // Read this lane's data from GDDR. Consecutive lanes touch
          // consecutive uint32 words → 4-byte stride between lanes, the
          // coalesced XPU access pattern.
          const bf16_t* src;
          uint8_t* token_data_ptr = nullptr;
          uint8_t* token_scale_ptr = nullptr;
          if (is_kv) {
            if (slot_id < 0) return;
            src = kv_ + token_idx * kv_stride_token_;
            const uint32_t uslot = static_cast<uint32_t>(slot_id);
            const uint32_t ublock = static_cast<uint32_t>(block_size_);
            const uint32_t block_idx = uslot / ublock;
            const uint32_t pos_in_block = uslot % ublock;
            uint8_t* block_base =
                cache_ + static_cast<int64_t>(block_idx) * cache_block_stride_;
            token_data_ptr = block_base + pos_in_block * TOKEN_DATA_BYTES;
            token_scale_ptr =
                block_base + block_size_ * TOKEN_DATA_BYTES +
                pos_in_block * SCALE_BYTES_PER_TOKEN;
          } else {
            src = q_ + token_idx * q_stride_token_ + slot_idx * q_stride_head_;
          }

          uint32_t nope_regs[NOPE_ITERS];
          const uint32_t* s4 = reinterpret_cast<const uint32_t*>(src);
          uint32_t rope_regs[ROPE_ITERS];
          const uint32_t* rope_s4 =
              reinterpret_cast<const uint32_t*>(src + NOPE_DIM);
          // Fuse per-register ALU math into the load loops so the work of
          // iteration i overlaps the in-flight loads of later iterations:
          // sum_sq FMAs on the Q path, per-lane |max| on the KV path.
          // Sub-group collectives stay out of the loop — their shuffle
          // chains serialize against the outstanding loads. With SG32 one
          // register spans one 64-elem quant block; this lane holds
          // elements {2*lane, 2*lane+1}.
          float sum_sq = 0.0f;
          float local_max[NUM_QUANT_BLOCKS];
          static_assert(SCALE_BYTES_PER_TOKEN == 8);
          static_assert(NOPE_ITERS == NUM_QUANT_BLOCKS);
#pragma unroll
          for (int i = 0; i < NOPE_ITERS; ++i) {
            nope_regs[i] = s4[lane + i * SG_SIZE];
            const float v0 = bf16x2_lo(nope_regs[i]);
            const float v1 = bf16x2_hi(nope_regs[i]);
            if (!is_kv) {
              sum_sq += v0 * v0 + v1 * v1;
            } else {
              local_max[i] = sycl::fmax(sycl::fabs(v0), sycl::fabs(v1));
            }
          }
#pragma unroll
          for (int pp = 0; pp < ROPE_ITERS; ++pp) {
            rope_regs[pp] = rope_s4[lane + pp * SG_SIZE];
            if (!is_kv) {
              const float x_even = bf16x2_lo(rope_regs[pp]);
              const float x_odd = bf16x2_hi(rope_regs[pp]);
              sum_sq += x_even * x_even + x_odd * x_odd;
            }
          }
          const float* cos_ptr = cos_sin_cache_ + pos * cache_stride_pos_;
          const float* sin_ptr = cos_ptr + HALF_ROPE;
          float cos_regs[ROPE_ITERS];
          float sin_regs[ROPE_ITERS];
#pragma unroll
          for (int pp = 0; pp < ROPE_ITERS; ++pp) {
            const int p = lane + pp * SG_SIZE;
            cos_regs[pp] = cos_ptr[p];
            sin_regs[pp] = sin_ptr[p];
          }

          // Overlap window: everything below until the RoPE rotation only
          // consumes nope/rope regs, so it runs (and its stores retire)
          // while the cos/sin loads are still in flight.
          float rrms = 1.0f;
          uint32_t* rope_dst;
          if (!is_kv) {
            sum_sq =
                sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());
            rrms =
                sycl::rsqrt(sum_sq / static_cast<float>(HEAD_DIM) + eps_);

            // Store normalized NoPE now; it does not depend on cos/sin.
            uint32_t* dst = reinterpret_cast<uint32_t*>(
                q_ + token_idx * q_stride_token_ +
                slot_idx * q_stride_head_);
#pragma unroll
            for (int i = 0; i < NOPE_ITERS; ++i) {
              dst[lane + i * SG_SIZE] = pack_bf16x2(
                  bf16x2_lo(nope_regs[i]) * rrms,
                  bf16x2_hi(nope_regs[i]) * rrms);
            }
            rope_dst = dst + NOPE_DIM / 2;
          } else {
            uint64_t packed_scales = 0;
#pragma unroll
            for (int i = 0; i < NUM_QUANT_BLOCKS; ++i) {
              float block_absmax = sycl::reduce_over_group(
                  sg, local_max[i], sycl::maximum<float>());
              block_absmax = sycl::fmax(block_absmax, 1e-4f);
              uint32_t scale_exp;
              fast_ceil_pow2(block_absmax * (1.0f / FP8_MAX), scale_exp);
              packed_scales |= static_cast<uint64_t>(scale_exp & 0xFFu)
                               << (8 * i);

              // Pow2 scale → exact inverse: bits = (254 - scale_exp) << 23.
              const float inv_scale =
                  sycl::bit_cast<float>((254u - scale_exp) << 23);

              const uint32_t b0 =
                  fp8_quant_cast(bf16x2_lo(nope_regs[i]), inv_scale);
              const uint32_t b1 =
                  fp8_quant_cast(bf16x2_hi(nope_regs[i]), inv_scale);
              reinterpret_cast<uint16_t*>(
                  token_data_ptr + i * QUANT_BLOCK)[lane] =
                  static_cast<uint16_t>(b0 | (b1 << 8));
            }
            if (lane == 0) {
              // token_scale_ptr is 8-byte aligned by the cache layout;
              // byte NUM_QUANT_BLOCKS stays 0 as the padding byte.
              *reinterpret_cast<uint64_t*>(token_scale_ptr) = packed_scales;
            }
            rope_dst =
                reinterpret_cast<uint32_t*>(token_data_ptr + NOPE_DIM);
          }

          // RoPE rotation last: the only consumer of the cos/sin loads,
          // whose latency is hidden by the norm/quant work above. Fusing
          // the rotation into the cos/sin load loop measures ~1.2 µs slower:
          // it stalls on those loads before the norm/quant work can start.
#pragma unroll
          for (int pp = 0; pp < ROPE_ITERS; ++pp) {
            const float x_even = bf16x2_lo(rope_regs[pp]) * rrms;
            const float x_odd = bf16x2_hi(rope_regs[pp]) * rrms;
            const float new_even =
                x_even * cos_regs[pp] - x_odd * sin_regs[pp];
            const float new_odd =
                x_even * sin_regs[pp] + x_odd * cos_regs[pp];
            rope_dst[lane + pp * SG_SIZE] = pack_bf16x2(new_even, new_odd);
          }
        });
  });
}


template <int HEAD_DIM, int ROPE_DIM>
void qnorm_rope_kv_insert_fp8_kernel_reduce_grid(
    sycl::queue& queue,
    bf16_t* __restrict__ q_,
    const bf16_t* __restrict__ kv_,
    uint8_t* __restrict__ cache_,
    const int64_t* __restrict__ slot_mapping_,
    const int64_t* __restrict__ position_ids_,
    const float* __restrict__ cos_sin_cache_,
    float eps_,
    int64_t num_tokens_,
    int64_t num_heads_q_,
    int64_t block_size_,
    int64_t q_stride_token_,
    int64_t q_stride_head_,
    int64_t kv_stride_token_,
    int64_t cache_block_stride_,
    int64_t cache_stride_pos_) {
  static constexpr int NOPE_DIM = HEAD_DIM - ROPE_DIM;
  static constexpr int HALF_ROPE = ROPE_DIM / 2;
  static constexpr int SG_SIZE = 16;
  static constexpr int WG_SIZE = 512;
  static constexpr int SUBGROUPS_PER_WG = WG_SIZE / SG_SIZE;
  // Fixed by the fp8_ds_mla cache format, not by the sub-group size.
  static constexpr int QUANT_BLOCK = 64;
  static constexpr int NUM_QUANT_BLOCKS = NOPE_DIM / QUANT_BLOCK;
  static constexpr int SCALE_BYTES_PER_TOKEN = NUM_QUANT_BLOCKS + 1;
  static constexpr int TOKEN_DATA_BYTES = NOPE_DIM + ROPE_DIM * 2;
  static constexpr int NOPE_ITERS = NOPE_DIM / (2 * SG_SIZE);
  static constexpr int ROPE_ITERS = HALF_ROPE / SG_SIZE;
  static_assert(
      NOPE_DIM % (2 * SG_SIZE) == 0,
      "NOPE_DIM must be divisible by 2*SG_SIZE");
  static_assert(
      HALF_ROPE % SG_SIZE == 0, "ROPE_DIM/2 must be divisible by SG_SIZE");
  static_assert(
      QUANT_BLOCK == 4 * SG_SIZE,
      "two registers per lane must span exactly one quant block");
  static_assert(WG_SIZE % SG_SIZE == 0);
  const size_t total_subgroups =
      static_cast<size_t>(num_tokens_) * (num_heads_q_ + 1);
  const size_t num_workgroups =
      (total_subgroups + SUBGROUPS_PER_WG - 1) / SUBGROUPS_PER_WG;
  sycl::range<2> global(num_workgroups, WG_SIZE);
  sycl::range<2> local(1, WG_SIZE);

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>(global, local),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
          // 32-bit index math: 64-bit div/mod is emulated and slow on XPU.
          const uint32_t workgroup_id =
              static_cast<uint32_t>(item.get_group(0));
          const uint32_t subgroup_id =
              static_cast<uint32_t>(item.get_local_id(1)) / SG_SIZE;
          const sycl::sub_group sg = item.get_sub_group();
          const uint32_t lane =
              static_cast<uint32_t>(sg.get_local_linear_id());
          const uint32_t global_subgroup_id =
              workgroup_id * SUBGROUPS_PER_WG + subgroup_id;
          const uint32_t slots_per_token =
              static_cast<uint32_t>(num_heads_q_) + 1;
          const uint32_t token_idx = global_subgroup_id / slots_per_token;
          const uint32_t slot_idx = global_subgroup_id % slots_per_token;

          if (token_idx >= num_tokens_) return;
          const bool is_kv = (slot_idx == num_heads_q_);

          // Issue the longest dependent-load chain first: pos → cos/sin.
          const int64_t pos = position_ids_[token_idx];
          const int64_t slot_id =
              is_kv ? slot_mapping_[token_idx] : int64_t{0};

          // Read this lane's data from GDDR. Consecutive lanes touch
          // consecutive uint32 words → 4-byte stride between lanes, the
          // coalesced XPU access pattern.
          const bf16_t* src;
          uint8_t* token_data_ptr = nullptr;
          uint8_t* token_scale_ptr = nullptr;
          if (is_kv) {
            if (slot_id < 0) return;
            src = kv_ + token_idx * kv_stride_token_;
            const uint32_t uslot = static_cast<uint32_t>(slot_id);
            const uint32_t ublock = static_cast<uint32_t>(block_size_);
            const uint32_t block_idx = uslot / ublock;
            const uint32_t pos_in_block = uslot % ublock;
            uint8_t* block_base =
                cache_ + static_cast<int64_t>(block_idx) * cache_block_stride_;
            token_data_ptr = block_base + pos_in_block * TOKEN_DATA_BYTES;
            token_scale_ptr =
                block_base + block_size_ * TOKEN_DATA_BYTES +
                pos_in_block * SCALE_BYTES_PER_TOKEN;
          } else {
            src = q_ + token_idx * q_stride_token_ + slot_idx * q_stride_head_;
          }

          uint32_t nope_regs[NOPE_ITERS];
          const uint32_t* s4 = reinterpret_cast<const uint32_t*>(src);
          uint32_t rope_regs[ROPE_ITERS];
          const uint32_t* rope_s4 =
              reinterpret_cast<const uint32_t*>(src + NOPE_DIM);
          // Fuse per-register ALU math into the load loops so the work of
          // iteration i overlaps the in-flight loads of later iterations:
          // sum_sq FMAs on the Q path, per-lane |max| on the KV path.
          // Sub-group collectives stay out of the loop — their shuffle
          // chains serialize against the outstanding loads. With SG16 two
          // consecutive registers span one 64-elem quant block; this lane
          // holds elements {2*lane, 2*lane+1} of each 32-elem half.
          float sum_sq = 0.0f;
          float local_max[NUM_QUANT_BLOCKS];
          static_assert(SCALE_BYTES_PER_TOKEN == 8);
          static_assert(NOPE_ITERS == 2 * NUM_QUANT_BLOCKS);
#pragma unroll
          for (int i = 0; i < NOPE_ITERS; ++i) {
            nope_regs[i] = s4[lane + i * SG_SIZE];
            const float v0 = bf16x2_lo(nope_regs[i]);
            const float v1 = bf16x2_hi(nope_regs[i]);
            if (!is_kv) {
              sum_sq += v0 * v0 + v1 * v1;
            } else {
              const float m = sycl::fmax(sycl::fabs(v0), sycl::fabs(v1));
              local_max[i / 2] =
                  (i % 2 == 0) ? m : sycl::fmax(local_max[i / 2], m);
            }
          }
#pragma unroll
          for (int pp = 0; pp < ROPE_ITERS; ++pp) {
            rope_regs[pp] = rope_s4[lane + pp * SG_SIZE];
            if (!is_kv) {
              const float x_even = bf16x2_lo(rope_regs[pp]);
              const float x_odd = bf16x2_hi(rope_regs[pp]);
              sum_sq += x_even * x_even + x_odd * x_odd;
            }
          }
          const float* cos_ptr = cos_sin_cache_ + pos * cache_stride_pos_;
          const float* sin_ptr = cos_ptr + HALF_ROPE;
          float cos_regs[ROPE_ITERS];
          float sin_regs[ROPE_ITERS];
#pragma unroll
          for (int pp = 0; pp < ROPE_ITERS; ++pp) {
            const int p = lane + pp * SG_SIZE;
            cos_regs[pp] = cos_ptr[p];
            sin_regs[pp] = sin_ptr[p];
          }

          // Overlap window: everything below until the RoPE rotation only
          // consumes nope/rope regs, so it runs (and its stores retire)
          // while the cos/sin loads are still in flight.
          float rrms = 1.0f;
          uint32_t* rope_dst;
          if (!is_kv) {
            sum_sq =
                sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());
            rrms =
                sycl::rsqrt(sum_sq / static_cast<float>(HEAD_DIM) + eps_);

            // Store normalized NoPE now; it does not depend on cos/sin.
            uint32_t* dst = reinterpret_cast<uint32_t*>(
                q_ + token_idx * q_stride_token_ +
                slot_idx * q_stride_head_);
#pragma unroll
            for (int i = 0; i < NOPE_ITERS; ++i) {
              dst[lane + i * SG_SIZE] = pack_bf16x2(
                  bf16x2_lo(nope_regs[i]) * rrms,
                  bf16x2_hi(nope_regs[i]) * rrms);
            }
            rope_dst = dst + NOPE_DIM / 2;
          } else {
            uint64_t packed_scales = 0;
#pragma unroll
            for (int b = 0; b < NUM_QUANT_BLOCKS; ++b) {
              float block_absmax = sycl::reduce_over_group(
                  sg, local_max[b], sycl::maximum<float>());
              block_absmax = sycl::fmax(block_absmax, 1e-4f);
              uint32_t scale_exp;
              fast_ceil_pow2(block_absmax * (1.0f / FP8_MAX), scale_exp);
              packed_scales |= static_cast<uint64_t>(scale_exp & 0xFFu)
                               << (8 * b);

              // Pow2 scale → exact inverse: bits = (254 - scale_exp) << 23.
              const float inv_scale =
                  sycl::bit_cast<float>((254u - scale_exp) << 23);

              const uint32_t w_lo = nope_regs[b * 2];
              const uint32_t w_hi = nope_regs[b * 2 + 1];
              // v: low uint16 = this lane's fp8 pair of the block's first
              // 32-elem half, high uint16 = pair of the second half.
              const uint32_t v =
                  fp8_quant_cast(bf16x2_lo(w_lo), inv_scale) |
                  (fp8_quant_cast(bf16x2_hi(w_lo), inv_scale) << 8) |
                  (fp8_quant_cast(bf16x2_lo(w_hi), inv_scale) << 16) |
                  (fp8_quant_cast(bf16x2_hi(w_hi), inv_scale) << 24);
              // Shuffle pairs from lanes (2*lane, 2*lane+1) mod 16 so every
              // lane stores one aligned 4B word: lanes 0-7 cover half 0,
              // lanes 8-15 cover half 1 → one coalesced 64B block store.
              const uint32_t p = (2 * lane) & 15u;
              const uint32_t w0 = sycl::select_from_group(sg, v, p);
              const uint32_t w1 = sycl::select_from_group(sg, v, p + 1);
              const uint32_t out = (lane < 8)
                  ? ((w0 & 0xFFFFu) | (w1 << 16))
                  : ((w0 >> 16) | (w1 & 0xFFFF0000u));
              reinterpret_cast<uint32_t*>(
                  token_data_ptr + b * QUANT_BLOCK)[lane] = out;
            }
            if (lane < 2) {
              // Scale bytes as two 4B lane writes (8-byte aligned layout);
              // byte NUM_QUANT_BLOCKS stays 0 as the padding byte.
              reinterpret_cast<uint32_t*>(token_scale_ptr)[lane] =
                  static_cast<uint32_t>(packed_scales >> (32 * lane));
            }
            rope_dst =
                reinterpret_cast<uint32_t*>(token_data_ptr + NOPE_DIM);
          }

          // RoPE rotation last: the only consumer of the cos/sin loads,
          // whose latency is hidden by the norm/quant work above. Fusing
          // the rotation into the cos/sin load loop measures ~1.2 µs slower:
          // it stalls on those loads before the norm/quant work can start.
#pragma unroll
          for (int pp = 0; pp < ROPE_ITERS; ++pp) {
            const float x_even = bf16x2_lo(rope_regs[pp]) * rrms;
            const float x_odd = bf16x2_hi(rope_regs[pp]) * rrms;
            const float new_even =
                x_even * cos_regs[pp] - x_odd * sin_regs[pp];
            const float new_odd =
                x_even * sin_regs[pp] + x_odd * cos_regs[pp];
            rope_dst[lane + pp * SG_SIZE] = pack_bf16x2(new_even, new_odd);
          }
        });
  });
}

// bf16 paged-cache variant: computation mirrors the backup base kernel; the
// memory access and WG mapping reuse the optimized fp8 layout (flat
// sub-group→(token,slot) mapping, 32-bit index math, 4B-per-lane accesses,
// load/compute overlap, RoPE rotation deferred past the cos/sin loads).
template <int HEAD_DIM, int ROPE_DIM>
void qnorm_rope_kv_insert_bf16_kernel(
    sycl::queue& queue,
    bf16_t* __restrict__ q_,
    const bf16_t* __restrict__ kv_,
    bf16_t* __restrict__ cache_,
    const int64_t* __restrict__ slot_mapping_,
    const int64_t* __restrict__ position_ids_,
    const float* __restrict__ cos_sin_cache_,
    float eps_,
    int64_t num_tokens_,
    int64_t num_heads_q_,
    int64_t block_size_,
    int64_t q_stride_token_,
    int64_t q_stride_head_,
    int64_t kv_stride_token_,
    int64_t cache_stride_block_,
    int64_t cache_stride_pos_) {
  static constexpr int NOPE_DIM = HEAD_DIM - ROPE_DIM;
  static constexpr int HALF_ROPE = ROPE_DIM / 2;
  static constexpr int SG_SIZE = 32;
  static constexpr int WG_SIZE = 32;
  static constexpr int SUBGROUPS_PER_WG = WG_SIZE / SG_SIZE;
  static constexpr int NOPE_ITERS = NOPE_DIM / (2 * SG_SIZE);
  static constexpr int ROPE_ITERS = HALF_ROPE / SG_SIZE;
  static_assert(
      NOPE_DIM % (2 * SG_SIZE) == 0,
      "NOPE_DIM must be divisible by 2*SG_SIZE");
  static_assert(
      HALF_ROPE % SG_SIZE == 0, "ROPE_DIM/2 must be divisible by SG_SIZE");
  static_assert(WG_SIZE % SG_SIZE == 0);
  const size_t total_subgroups =
      static_cast<size_t>(num_tokens_) * (num_heads_q_ + 1);
  const size_t num_workgroups =
      (total_subgroups + SUBGROUPS_PER_WG - 1) / SUBGROUPS_PER_WG;
  sycl::range<2> global(num_workgroups, WG_SIZE);
  sycl::range<2> local(1, WG_SIZE);

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>(global, local),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
          // 32-bit index math: 64-bit div/mod is emulated and slow on XPU.
          const uint32_t workgroup_id =
              static_cast<uint32_t>(item.get_group(0));
          const uint32_t subgroup_id =
              static_cast<uint32_t>(item.get_local_id(1)) / SG_SIZE;
          const sycl::sub_group sg = item.get_sub_group();
          const uint32_t lane =
              static_cast<uint32_t>(sg.get_local_linear_id());
          const uint32_t global_subgroup_id =
              workgroup_id * SUBGROUPS_PER_WG + subgroup_id;
          const uint32_t slots_per_token =
              static_cast<uint32_t>(num_heads_q_) + 1;
          const uint32_t token_idx = global_subgroup_id / slots_per_token;
          const uint32_t slot_idx = global_subgroup_id % slots_per_token;

          if (token_idx >= num_tokens_) return;
          const bool is_kv = (slot_idx == num_heads_q_);

          // Issue the longest dependent-load chain first: pos → cos/sin.
          const int64_t pos = position_ids_[token_idx];
          const int64_t slot_id =
              is_kv ? slot_mapping_[token_idx] : int64_t{0};

          // Consecutive lanes touch consecutive uint32 words → 4-byte
          // stride between lanes, the coalesced XPU access pattern.
          const bf16_t* src;
          bf16_t* kv_dst = nullptr;
          if (is_kv) {
            if (slot_id < 0) return;
            src = kv_ + token_idx * kv_stride_token_;
            const uint32_t uslot = static_cast<uint32_t>(slot_id);
            const uint32_t ublock = static_cast<uint32_t>(block_size_);
            const uint32_t block_idx = uslot / ublock;
            const uint32_t pos_in_block = uslot % ublock;
            kv_dst = cache_ +
                static_cast<int64_t>(block_idx) * cache_stride_block_ +
                pos_in_block * HEAD_DIM;
          } else {
            src = q_ + token_idx * q_stride_token_ + slot_idx * q_stride_head_;
          }

          uint32_t nope_regs[NOPE_ITERS];
          const uint32_t* s4 = reinterpret_cast<const uint32_t*>(src);
          uint32_t rope_regs[ROPE_ITERS];
          const uint32_t* rope_s4 =
              reinterpret_cast<const uint32_t*>(src + NOPE_DIM);
          // Fuse the sum_sq FMAs into the load loops so the math of
          // iteration i overlaps the in-flight loads of later iterations.
          float sum_sq = 0.0f;
#pragma unroll
          for (int i = 0; i < NOPE_ITERS; ++i) {
            nope_regs[i] = s4[lane + i * SG_SIZE];
            if (!is_kv) {
              const float v0 = bf16x2_lo(nope_regs[i]);
              const float v1 = bf16x2_hi(nope_regs[i]);
              sum_sq += v0 * v0 + v1 * v1;
            }
          }
#pragma unroll
          for (int pp = 0; pp < ROPE_ITERS; ++pp) {
            rope_regs[pp] = rope_s4[lane + pp * SG_SIZE];
            if (!is_kv) {
              const float x_even = bf16x2_lo(rope_regs[pp]);
              const float x_odd = bf16x2_hi(rope_regs[pp]);
              sum_sq += x_even * x_even + x_odd * x_odd;
            }
          }
          const float* cos_ptr = cos_sin_cache_ + pos * cache_stride_pos_;
          const float* sin_ptr = cos_ptr + HALF_ROPE;
          float cos_regs[ROPE_ITERS];
          float sin_regs[ROPE_ITERS];
#pragma unroll
          for (int pp = 0; pp < ROPE_ITERS; ++pp) {
            const int p = lane + pp * SG_SIZE;
            cos_regs[pp] = cos_ptr[p];
            sin_regs[pp] = sin_ptr[p];
          }

          // NoPE stores don't need cos/sin: they retire while those loads
          // are still in flight.
          float rrms = 1.0f;
          uint32_t* rope_dst;
          if (!is_kv) {
            sum_sq =
                sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());
            rrms =
                sycl::rsqrt(sum_sq / static_cast<float>(HEAD_DIM) + eps_);

            uint32_t* dst = reinterpret_cast<uint32_t*>(
                q_ + token_idx * q_stride_token_ +
                slot_idx * q_stride_head_);
#pragma unroll
            for (int i = 0; i < NOPE_ITERS; ++i) {
              dst[lane + i * SG_SIZE] = pack_bf16x2(
                  bf16x2_lo(nope_regs[i]) * rrms,
                  bf16x2_hi(nope_regs[i]) * rrms);
            }
            rope_dst = dst + NOPE_DIM / 2;
          } else {
            // KV path: raw bf16 NoPE copy into the paged cache.
            uint32_t* dst = reinterpret_cast<uint32_t*>(kv_dst);
#pragma unroll
            for (int i = 0; i < NOPE_ITERS; ++i) {
              dst[lane + i * SG_SIZE] = nope_regs[i];
            }
            rope_dst = dst + NOPE_DIM / 2;
          }

          // RoPE rotation last: the only consumer of the cos/sin loads.
#pragma unroll
          for (int pp = 0; pp < ROPE_ITERS; ++pp) {
            const float x_even = bf16x2_lo(rope_regs[pp]) * rrms;
            const float x_odd = bf16x2_hi(rope_regs[pp]) * rrms;
            const float new_even =
                x_even * cos_regs[pp] - x_odd * sin_regs[pp];
            const float new_odd =
                x_even * sin_regs[pp] + x_odd * cos_regs[pp];
            rope_dst[lane + pp * SG_SIZE] = pack_bf16x2(new_even, new_odd);
          }
        });
  });
}

}  // namespace vllm

using vllm::bf16_t;

void deepseek_qnorm_rope_kv_insert(
    torch::Tensor& q,
    const torch::Tensor& kv,
    torch::Tensor& cache,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& position_ids,
    const torch::Tensor& cos_sin_cache,
    double eps,
    int64_t block_size,
    const std::string& kv_cache_dtype) {
  const int64_t num_tokens = q.size(0);
  const int64_t num_heads_q = q.size(1);
  const int64_t head_dim = q.size(2);
  const int64_t rope_dim = cos_sin_cache.size(1);

  TORCH_CHECK(
      kv.size(1) == head_dim,
      "kv head_dim must match q head_dim (",
      head_dim,
      "), got ",
      kv.size(1));
  TORCH_CHECK(
      head_dim == 512 && rope_dim == 64,
      "deepseek_qnorm_rope_kv_insert: unsupported head_dim=",
      head_dim,
      " rope_dim=",
      rope_dim,
      ". Supported: head_dim=512, rope_dim=64");

  const int64_t q_stride_token = q.stride(0);
  const int64_t q_stride_head = q.stride(1);
  const int64_t kv_stride_token = kv.stride(0);
  const int64_t cache_stride_pos = cos_sin_cache.stride(0);

  at::DeviceGuard device_guard(q.device());
  auto& q_xpu = vllm::xpu::vllmGetQueue();

  TORCH_CHECK(
      kv_cache_dtype == "bf16" || kv_cache_dtype == "fp8_ds_mla",
      "deepseek_qnorm_rope_kv_insert: unsupported kv_cache_dtype '",
      kv_cache_dtype,
      "', expected 'bf16' or 'fp8_ds_mla'");

  auto* q_ptr = reinterpret_cast<bf16_t*>(q.data_ptr());
  auto* kv_ptr = reinterpret_cast<const bf16_t*>(kv.data_ptr());
  auto* slot_ptr = slot_mapping.data_ptr<int64_t>();
  auto* pos_ptr = position_ids.data_ptr<int64_t>();
  auto* cs_ptr = cos_sin_cache.data_ptr<float>();
  const int64_t cache_block_stride = cache.stride(0);

  if (kv_cache_dtype == "bf16") {
    auto* cache_bf16 = reinterpret_cast<bf16_t*>(cache.data_ptr());
    vllm::qnorm_rope_kv_insert_bf16_kernel<512, 64>(
        q_xpu,
        q_ptr,
        kv_ptr,
        cache_bf16,
        slot_ptr,
        pos_ptr,
        cs_ptr,
        static_cast<float>(eps),
        num_tokens,
        num_heads_q,
        block_size,
        q_stride_token,
        q_stride_head,
        kv_stride_token,
        cache_block_stride,
        cache_stride_pos);
    return;
  }

  auto* cache_ptr = reinterpret_cast<uint8_t*>(cache.data_ptr());
  if (num_tokens < 256) {
  vllm::qnorm_rope_kv_insert_fp8_kernel_decode<512, 64>(
      q_xpu,
      q_ptr,
      kv_ptr,
      cache_ptr,
      slot_ptr,
      pos_ptr,
      cs_ptr,
      static_cast<float>(eps),
      num_tokens,
      num_heads_q,
      block_size,
      q_stride_token,
      q_stride_head,
      kv_stride_token,
      cache_block_stride,
      cache_stride_pos);
  } else {
    vllm::qnorm_rope_kv_insert_fp8_kernel_reduce_grid<512, 64>(
        q_xpu,
        q_ptr,
        kv_ptr,
        cache_ptr,
        slot_ptr,
        pos_ptr,
        cs_ptr,
        static_cast<float>(eps),
        num_tokens,
        num_heads_q,
        block_size,
        q_stride_token,
        q_stride_head,
        kv_stride_token,
        cache_block_stride,
        cache_stride_pos);
  }
}
