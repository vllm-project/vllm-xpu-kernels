// SPDX-License-Identifier: Apache-2.0
// Fused DeepSeek-V4 Q-RMSNorm + GPT-J RoPE + KV cache insert.
// One sub-group (16 lanes) per (token, head_or_kv). Dispatches bf16 or fp8
// cache format via kv_cache_dtype parameter.

#include <sycl/sycl.hpp>
#include "utils.h"
#include <cstdint>
#include <torch/all.h>

namespace vllm {

using bf16_t = sycl::ext::oneapi::bfloat16;

static constexpr float FP8_MAX = 448.0f;

// Ceil to power-of-2 via IEEE754 bit hack; also returns biased exponent.
inline float fast_ceil_pow2(float x, uint32_t& out_scale_exp) {
  uint32_t bits;
  __builtin_memcpy(&bits, &x, 4);
  if (bits & 0x7FFFFF) {
    bits = (bits + 0x800000) & 0xFF800000u;
  }
  out_scale_exp = bits >> 23;
  float result;
  __builtin_memcpy(&result, &bits, 4);
  return result;
}

// Divide by pow2 scale and convert to fp8_e4m3 via integer arithmetic.
inline uint8_t fused_scale_to_fp8(float val, uint32_t scale_exp) {
  uint32_t bits;
  __builtin_memcpy(&bits, &val, 4);
  uint32_t sign = bits >> 31;
  uint32_t v_exp = (bits >> 23) & 0xFFu;
  uint32_t v_mant = bits & 0x7FFFFFu;

  uint32_t mant3 = v_mant >> 20;
  uint32_t remainder = v_mant & 0xFFFFFu;
  // Round-to-nearest-even
  if (remainder > 0x80000u || (remainder == 0x80000u && (mant3 & 1u))) {
    mant3++;
  }
  uint32_t e4m3_exp = v_exp - scale_exp + 7u + (mant3 >> 3);
  mant3 &= 7u;
  uint32_t result = (e4m3_exp << 3) | mant3;
  result = (v_exp + 7u >= scale_exp) ? result : 0u;

  return static_cast<uint8_t>((sign << 7) | (result & 0x7Fu));
}

template <int HEAD_DIM_, int ROPE_DIM_>
class qnorm_rope_kv_insert_bf16_kernel {
 public:
  static constexpr int HEAD_DIM = HEAD_DIM_;
  static constexpr int ROPE_DIM = ROPE_DIM_;
  static constexpr int NOPE_DIM = HEAD_DIM - ROPE_DIM;
  static constexpr int HALF_ROPE = ROPE_DIM / 2;
  static constexpr int SG_SIZE = 16;
  static constexpr int NOPE_ITERS = NOPE_DIM / (4 * SG_SIZE);
  static constexpr int ROPE_ITERS = HALF_ROPE / SG_SIZE;
  static_assert(
      NOPE_DIM % (4 * SG_SIZE) == 0, "NOPE_DIM must be divisible by 4*SG_SIZE");
  static_assert(
      HALF_ROPE % SG_SIZE == 0, "ROPE_DIM/2 must be divisible by SG_SIZE");

  qnorm_rope_kv_insert_bf16_kernel(
      bf16_t* __restrict__ q,
      const bf16_t* __restrict__ kv,
      bf16_t* __restrict__ cache,
      const int64_t* __restrict__ slot_mapping,
      const int64_t* __restrict__ position_ids,
      const float* __restrict__ cos_sin_cache,
      float eps,
      int64_t num_tokens,
      int64_t num_heads_q,
      int64_t block_size,
      int64_t q_stride_token,
      int64_t q_stride_head,
      int64_t kv_stride_token,
      int64_t cache_stride_block,
      int64_t cache_stride_pos)
      : q_(q),
        kv_(kv),
        cache_(cache),
        slot_mapping_(slot_mapping),
        position_ids_(position_ids),
        cos_sin_cache_(cos_sin_cache),
        eps_(eps),
        num_tokens_(num_tokens),
        num_heads_q_(num_heads_q),
        block_size_(block_size),
        q_stride_token_(q_stride_token),
        q_stride_head_(q_stride_head),
        kv_stride_token_(kv_stride_token),
        cache_stride_block_(cache_stride_block),
        cache_stride_pos_(cache_stride_pos) {}

  [[sycl::reqd_sub_group_size(SG_SIZE)]]
  void operator()(sycl::nd_item<3> item) const {
    const int64_t token_idx = item.get_global_id(0);
    const int64_t slot_idx = item.get_global_id(1);
    const int64_t lane = item.get_local_id(2);

    if (token_idx >= num_tokens_) return;
    const bool is_kv = (slot_idx == num_heads_q_);

    const bf16_t* src;
    if (is_kv) {
      src = kv_ + token_idx * kv_stride_token_;
    } else {
      src = q_ + token_idx * q_stride_token_ + slot_idx * q_stride_head_;
    }

    // Load NoPE dims [0, NOPE_DIM) as packed uint64 (4×bf16 per lane).
    float sum_sq = 0.0f;
    uint64_t nope_regs[NOPE_ITERS];
    {
      const uint64_t* s8 = reinterpret_cast<const uint64_t*>(src);
#pragma unroll
      for (int i = 0; i < NOPE_ITERS; ++i) {
        uint64_t val = s8[lane + i * SG_SIZE];
        nope_regs[i] = val;
        if (!is_kv) {
          const bf16_t* vals = reinterpret_cast<const bf16_t*>(&val);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            float fv = static_cast<float>(vals[j]);
            sum_sq += fv * fv;
          }
        }
      }
    }

    // Load interleaved RoPE pairs [NOPE_DIM, HEAD_DIM).
    float rope_even[ROPE_ITERS], rope_odd[ROPE_ITERS];
#pragma unroll
    for (int pp = 0; pp < ROPE_ITERS; ++pp) {
      const int p = lane + pp * SG_SIZE;
      float x_even = static_cast<float>(src[NOPE_DIM + 2 * p]);
      float x_odd = static_cast<float>(src[NOPE_DIM + 2 * p + 1]);
      rope_even[pp] = x_even;
      rope_odd[pp] = x_odd;
      if (!is_kv) {
        sum_sq += x_even * x_even + x_odd * x_odd;
      }
    }

    // Q: sub-group reduction for RMSNorm.
    float rrms = 1.0f;
    if (!is_kv) {
      auto sg = item.get_sub_group();
      sum_sq = sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());
      rrms = sycl::rsqrt(sum_sq / static_cast<float>(HEAD_DIM) + eps_);
    }

    const int64_t pos = position_ids_[token_idx];
    const float* cos_ptr = cos_sin_cache_ + pos * cache_stride_pos_;
    const float* sin_ptr = cos_ptr + HALF_ROPE;

    if (is_kv) {
      // KV path: copy NoPE + apply RoPE into paged bf16 cache.
      int64_t slot_id = slot_mapping_[token_idx];
      if (slot_id < 0) return;
      int64_t blk = slot_id / block_size_;
      int64_t pos_in_blk = slot_id % block_size_;
      bf16_t* dst = cache_ + blk * cache_stride_block_ + pos_in_blk * HEAD_DIM;

      uint64_t* d8 = reinterpret_cast<uint64_t*>(dst);
#pragma unroll
      for (int i = 0; i < NOPE_ITERS; ++i)
        d8[lane + i * SG_SIZE] = nope_regs[i];

#pragma unroll
      for (int pp = 0; pp < ROPE_ITERS; ++pp) {
        const int p = lane + pp * SG_SIZE;
        float c = cos_ptr[p], s = sin_ptr[p];
        float new_even = rope_even[pp] * c - rope_odd[pp] * s;
        float new_odd = rope_even[pp] * s + rope_odd[pp] * c;
        dst[NOPE_DIM + 2 * p] = static_cast<bf16_t>(new_even);
        dst[NOPE_DIM + 2 * p + 1] = static_cast<bf16_t>(new_odd);
      }
    } else {
      // Q path: apply RMSNorm + RoPE in-place.
      bf16_t* dst =
          q_ + token_idx * q_stride_token_ + slot_idx * q_stride_head_;
      {
        uint64_t* d8 = reinterpret_cast<uint64_t*>(dst);
#pragma unroll
        for (int i = 0; i < NOPE_ITERS; ++i) {
          uint64_t val = nope_regs[i];
          const bf16_t* in_vals = reinterpret_cast<const bf16_t*>(&val);
          bf16_t out_vals[4];
#pragma unroll
          for (int j = 0; j < 4; ++j)
            out_vals[j] =
                static_cast<bf16_t>(static_cast<float>(in_vals[j]) * rrms);
          uint64_t out_packed;
          __builtin_memcpy(&out_packed, out_vals, sizeof(uint64_t));
          d8[lane + i * SG_SIZE] = out_packed;
        }
      }
#pragma unroll
      for (int pp = 0; pp < ROPE_ITERS; ++pp) {
        const int p = lane + pp * SG_SIZE;
        float c = cos_ptr[p], s = sin_ptr[p];
        float x_e = rope_even[pp] * rrms;
        float x_o = rope_odd[pp] * rrms;
        float new_even = x_e * c - x_o * s;
        float new_odd = x_e * s + x_o * c;
        dst[NOPE_DIM + 2 * p] = static_cast<bf16_t>(new_even);
        dst[NOPE_DIM + 2 * p + 1] = static_cast<bf16_t>(new_odd);
      }
    }
  }

 private:
  bf16_t* __restrict__ q_;
  const bf16_t* __restrict__ kv_;
  bf16_t* __restrict__ cache_;
  const int64_t* __restrict__ slot_mapping_;
  const int64_t* __restrict__ position_ids_;
  const float* __restrict__ cos_sin_cache_;
  float eps_;
  int64_t num_tokens_, num_heads_q_, block_size_;
  int64_t q_stride_token_, q_stride_head_;
  int64_t kv_stride_token_;
  int64_t cache_stride_block_;
  int64_t cache_stride_pos_;
};

template <int HEAD_DIM_, int ROPE_DIM_>
class qnorm_rope_kv_insert_fp8_kernel {
 public:
  static constexpr int HEAD_DIM = HEAD_DIM_;
  static constexpr int ROPE_DIM = ROPE_DIM_;
  static constexpr int NOPE_DIM = HEAD_DIM - ROPE_DIM;
  static constexpr int HALF_ROPE = ROPE_DIM / 2;
  static constexpr int SG_SIZE = 16;
  static constexpr int QUANT_BLOCK = SG_SIZE * 4;
  static constexpr int NUM_QUANT_BLOCKS = NOPE_DIM / QUANT_BLOCK;
  static constexpr int SCALE_BYTES_PER_TOKEN = NUM_QUANT_BLOCKS + 1;
  static constexpr int TOKEN_DATA_BYTES = NOPE_DIM + ROPE_DIM * 2;
  static constexpr int NOPE_ITERS = NOPE_DIM / (4 * SG_SIZE);
  static constexpr int ROPE_ITERS = HALF_ROPE / SG_SIZE;
  static_assert(
      NOPE_DIM % (4 * SG_SIZE) == 0, "NOPE_DIM must be divisible by 4*SG_SIZE");
  static_assert(
      HALF_ROPE % SG_SIZE == 0, "ROPE_DIM/2 must be divisible by SG_SIZE");
  static_assert(
      NOPE_DIM % QUANT_BLOCK == 0, "NOPE_DIM must be divisible by QUANT_BLOCK");

  qnorm_rope_kv_insert_fp8_kernel(
      bf16_t* __restrict__ q,
      const bf16_t* __restrict__ kv,
      uint8_t* __restrict__ cache,
      const int64_t* __restrict__ slot_mapping,
      const int64_t* __restrict__ position_ids,
      const float* __restrict__ cos_sin_cache,
      float eps,
      int64_t num_tokens,
      int64_t num_heads_q,
      int64_t block_size,
      int64_t q_stride_token,
      int64_t q_stride_head,
      int64_t kv_stride_token,
      int64_t cache_block_stride,
      int64_t cache_stride_pos)
      : q_(q),
        kv_(kv),
        cache_(cache),
        slot_mapping_(slot_mapping),
        position_ids_(position_ids),
        cos_sin_cache_(cos_sin_cache),
        eps_(eps),
        num_tokens_(num_tokens),
        num_heads_q_(num_heads_q),
        block_size_(block_size),
        q_stride_token_(q_stride_token),
        q_stride_head_(q_stride_head),
        kv_stride_token_(kv_stride_token),
        cache_block_stride_(cache_block_stride),
        cache_stride_pos_(cache_stride_pos) {}

  [[sycl::reqd_sub_group_size(SG_SIZE)]]
  void operator()(sycl::nd_item<3> item) const {
    const int64_t token_idx = item.get_global_id(0);
    const int64_t slot_idx = item.get_global_id(1);
    const int64_t lane = item.get_local_id(2);

    if (token_idx >= num_tokens_) return;
    const bool is_kv = (slot_idx == num_heads_q_);

    const bf16_t* src;
    if (is_kv) {
      src = kv_ + token_idx * kv_stride_token_;
    } else {
      src = q_ + token_idx * q_stride_token_ + slot_idx * q_stride_head_;
    }

    // Load NoPE dims [0, NOPE_DIM), keep float copy for fp8 quantization.
    float sum_sq = 0.0f;
    float nope_floats[NOPE_ITERS * 4];
    uint64_t nope_regs[NOPE_ITERS];
    {
      const uint64_t* s8 = reinterpret_cast<const uint64_t*>(src);
#pragma unroll
      for (int i = 0; i < NOPE_ITERS; ++i) {
        uint64_t val = s8[lane + i * SG_SIZE];
        nope_regs[i] = val;
        const bf16_t* vals = reinterpret_cast<const bf16_t*>(&val);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float fv = static_cast<float>(vals[j]);
          nope_floats[i * 4 + j] = fv;
          if (!is_kv) {
            sum_sq += fv * fv;
          }
        }
      }
    }

    // Load interleaved RoPE pairs [NOPE_DIM, HEAD_DIM).
    float rope_even[ROPE_ITERS], rope_odd[ROPE_ITERS];
#pragma unroll
    for (int pp = 0; pp < ROPE_ITERS; ++pp) {
      const int p = lane + pp * SG_SIZE;
      float x_even = static_cast<float>(src[NOPE_DIM + 2 * p]);
      float x_odd = static_cast<float>(src[NOPE_DIM + 2 * p + 1]);
      rope_even[pp] = x_even;
      rope_odd[pp] = x_odd;
      if (!is_kv) {
        sum_sq += x_even * x_even + x_odd * x_odd;
      }
    }

    auto sg = item.get_sub_group();

    if (is_kv) {
      // KV path: fp8 block-quantize NoPE + RoPE into paged cache.
      int64_t slot_id = slot_mapping_[token_idx];
      if (slot_id < 0) return;
      int64_t blk = slot_id / block_size_;
      int64_t pos_in_blk = slot_id % block_size_;

      uint8_t* block_base = cache_ + blk * cache_block_stride_;
      uint8_t* token_data_ptr = block_base + pos_in_blk * TOKEN_DATA_BYTES;
      uint8_t* token_scale_ptr =
          block_base + static_cast<int64_t>(block_size_) * TOKEN_DATA_BYTES +
          pos_in_blk * SCALE_BYTES_PER_TOKEN;

// Round to bf16 precision before quantization (matches CUDA path).
#pragma unroll
      for (int i = 0; i < NOPE_ITERS * 4; ++i) {
        nope_floats[i] =
            static_cast<float>(static_cast<bf16_t>(nope_floats[i]));
      }

// Per-block absmax quantization: scale = ceil_pow2(absmax / FP8_MAX).
#pragma unroll
      for (int i = 0; i < NOPE_ITERS; ++i) {
        float local_max = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          local_max = sycl::fmax(local_max, sycl::fabs(nope_floats[i * 4 + j]));
        }

        float block_absmax =
            sycl::reduce_over_group(sg, local_max, sycl::maximum<float>());
        block_absmax = sycl::fmax(block_absmax, 1e-4f);

        float scale_raw = block_absmax * (1.0f / FP8_MAX);
        uint32_t scale_exp;
        fast_ceil_pow2(scale_raw, scale_exp);

        uint32_t packed = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          packed |=
              (static_cast<uint32_t>(
                   fused_scale_to_fp8(nope_floats[i * 4 + j], scale_exp))
               << (j * 8));
        }

        int byte_offset = i * QUANT_BLOCK + lane * 4;
        *reinterpret_cast<uint32_t*>(token_data_ptr + byte_offset) = packed;

        if (lane == 0) {
          token_scale_ptr[i] = static_cast<uint8_t>(scale_exp);
        }
      }

      if (lane == 0) {
        token_scale_ptr[NUM_QUANT_BLOCKS] = 0;  // padding byte
      }

      // Apply forward RoPE and store tail as bf16.
      const int64_t pos = position_ids_[token_idx];
      const float* cos_ptr = cos_sin_cache_ + pos * cache_stride_pos_;
      const float* sin_ptr = cos_ptr + HALF_ROPE;
      bf16_t* rope_dst = reinterpret_cast<bf16_t*>(token_data_ptr + NOPE_DIM);

#pragma unroll
      for (int pp = 0; pp < ROPE_ITERS; ++pp) {
        const int p = lane + pp * SG_SIZE;
        float c = cos_ptr[p], s = sin_ptr[p];
        float new_even = rope_even[pp] * c - rope_odd[pp] * s;
        float new_odd = rope_even[pp] * s + rope_odd[pp] * c;
        rope_dst[2 * p] = static_cast<bf16_t>(new_even);
        rope_dst[2 * p + 1] = static_cast<bf16_t>(new_odd);
      }
    } else {
      // Q path: sub-group reduction for RMSNorm, then apply in-place.
      sum_sq = sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());
      float rrms = sycl::rsqrt(sum_sq / static_cast<float>(HEAD_DIM) + eps_);

      bf16_t* dst =
          q_ + token_idx * q_stride_token_ + slot_idx * q_stride_head_;
      {
        uint64_t* d8 = reinterpret_cast<uint64_t*>(dst);
#pragma unroll
        for (int i = 0; i < NOPE_ITERS; ++i) {
          uint64_t val = nope_regs[i];
          const bf16_t* in_vals = reinterpret_cast<const bf16_t*>(&val);
          bf16_t out_vals[4];
#pragma unroll
          for (int j = 0; j < 4; ++j)
            out_vals[j] =
                static_cast<bf16_t>(static_cast<float>(in_vals[j]) * rrms);
          uint64_t out_packed;
          __builtin_memcpy(&out_packed, out_vals, sizeof(uint64_t));
          d8[lane + i * SG_SIZE] = out_packed;
        }
      }
      const int64_t pos = position_ids_[token_idx];
      const float* cos_ptr = cos_sin_cache_ + pos * cache_stride_pos_;
      const float* sin_ptr = cos_ptr + HALF_ROPE;
#pragma unroll
      for (int pp = 0; pp < ROPE_ITERS; ++pp) {
        const int p = lane + pp * SG_SIZE;
        float c = cos_ptr[p], s = sin_ptr[p];
        float x_e = rope_even[pp] * rrms;
        float x_o = rope_odd[pp] * rrms;
        float new_even = x_e * c - x_o * s;
        float new_odd = x_e * s + x_o * c;
        dst[NOPE_DIM + 2 * p] = static_cast<bf16_t>(new_even);
        dst[NOPE_DIM + 2 * p + 1] = static_cast<bf16_t>(new_odd);
      }
    }
  }

 private:
  bf16_t* __restrict__ q_;
  const bf16_t* __restrict__ kv_;
  uint8_t* __restrict__ cache_;
  const int64_t* __restrict__ slot_mapping_;
  const int64_t* __restrict__ position_ids_;
  const float* __restrict__ cos_sin_cache_;
  float eps_;
  int64_t num_tokens_, num_heads_q_, block_size_;
  int64_t q_stride_token_, q_stride_head_;
  int64_t kv_stride_token_;
  int64_t cache_block_stride_;
  int64_t cache_stride_pos_;
};

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

  constexpr int SG_SIZE = 16;
  sycl::range<3> global(num_tokens, num_heads_q + 1, SG_SIZE);
  sycl::range<3> local(1, 1, SG_SIZE);

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
    auto* cache_ptr = reinterpret_cast<bf16_t*>(cache.data_ptr());
    q_xpu.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(global, local),
          vllm::qnorm_rope_kv_insert_bf16_kernel<512, 64>(
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
              cache_stride_pos));
    });
  } else {
    auto* cache_ptr = reinterpret_cast<uint8_t*>(cache.data_ptr());
    q_xpu.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(global, local),
          vllm::qnorm_rope_kv_insert_fp8_kernel<512, 64>(
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
              cache_stride_pos));
    });
  }
}
