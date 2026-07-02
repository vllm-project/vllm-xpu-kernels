// SPDX-License-Identifier: Apache-2.0
// Fused DeepSeek-V4 indexer Q: GPT-J RoPE + per-head FP8 quantize + weight
// fold. One sub-group (16 lanes) handles 2 heads (COARSEN=2) per token. Matches
// the Triton _fused_indexer_q_rope_quant_kernel semantics.

#include <sycl/sycl.hpp>
#include "utils.h"
#include <cstdint>
#include <torch/all.h>

namespace vllm {

using bf16_t = sycl::ext::oneapi::bfloat16;

static constexpr int SG_SIZE = 16;
static constexpr float INV_FP8_MAX = 1.0f / 448.0f;

// Division-free power-of-2 scale + inv_scale computation.
inline void fp8_scale_pair(float amax, float& scale, float& inv_scale) {
  float ratio = sycl::fmax(amax, 1e-4f) * INV_FP8_MAX;
  uint32_t bits;
  __builtin_memcpy(&bits, &ratio, 4);
  if (bits & 0x7FFFFFu) bits = (bits + 0x800000u) & 0xFF800000u;
  if ((bits & 0x7F800000u) == 0) bits = 0x00800000u;
  __builtin_memcpy(&scale, &bits, 4);
  uint32_t inv_bits = (254u - ((bits >> 23) & 0xFFu)) << 23;
  __builtin_memcpy(&inv_scale, &inv_bits, 4);
}

// Branchless FP8 E4M3FN conversion.
inline uint8_t fp8_cvt(float x) {
  uint32_t b;
  __builtin_memcpy(&b, &x, 4);
  uint32_t sign = (b >> 24) & 0x80u;
  b &= 0x7FFFFFFFu;
  uint32_t rd = b + (1u << 19);
  rd = (rd < 0x43E00000u) ? rd : 0x43E00000u;
  uint32_t e32 = (rd >> 23) & 0xFFu, m3 = (rd >> 20) & 0x7u;
  int32_t e8 = (int32_t)e32 - 120;
  int32_t sh = 1 - e8;
  sh = (sh > 7) ? 7 : sh;
  uint32_t sub = ((8u + m3) >> sh) & 0x7u;
  uint32_t nor = ((uint32_t)e8 << 3) | m3;
  uint32_t mk = (uint32_t)-(e8 > 0);
  uint32_t res = (nor & mk) | (sub & ~mk);
  uint32_t nan_m = (uint32_t)-(res == 0x7Fu);
  res = (res & ~nan_m) | (0x7Eu & nan_m);
  return (uint8_t)(sign | res);
}

template <int HEAD_DIM_, int ROPE_DIM_, int HEADS_COARSEN_ = 2>
class deepseek_fused_indexer_q_rope_fp8_kernel {
 public:
  static constexpr int HEAD_DIM = HEAD_DIM_;
  static constexpr int ROPE_DIM = ROPE_DIM_;
  static constexpr int NOPE_DIM = HEAD_DIM - ROPE_DIM;
  static constexpr int HALF_ROPE = ROPE_DIM / 2;
  static constexpr int HALF_BLOCK = SG_SIZE;
  static constexpr int HEADS_COARSEN = HEADS_COARSEN_;
  static_assert(
      ROPE_DIM % 2 == 0 && HALF_ROPE >= SG_SIZE,
      "ROPE_DIM/2 must be >= SG_SIZE");
  static_assert(NOPE_DIM > 0, "HEAD_DIM must be > ROPE_DIM");

  deepseek_fused_indexer_q_rope_fp8_kernel(
      const bf16_t* __restrict__ q,
      const int64_t* __restrict__ positions,
      const float* __restrict__ cos_sin_cache,
      const bf16_t* __restrict__ weights,
      uint8_t* __restrict__ q_fp8,
      float* __restrict__ weights_out,
      float weight_combined,
      int64_t num_tokens,
      int64_t num_heads,
      int64_t cos_sin_stride)
      : q_(q),
        positions_(positions),
        cos_sin_cache_(cos_sin_cache),
        weights_(weights),
        q_fp8_(q_fp8),
        weights_out_(weights_out),
        weight_combined_(weight_combined),
        num_tokens_(num_tokens),
        num_heads_(num_heads),
        cos_sin_stride_(cos_sin_stride) {}

  [[sycl::reqd_sub_group_size(SG_SIZE)]]
  void operator()(sycl::nd_item<2> item) const {
    const int64_t tok = item.get_global_id(0);
    const int64_t hg = item.get_global_id(1) / SG_SIZE;
    const int lane = item.get_local_id(1) % SG_SIZE;
    const int64_t bh = hg * HEADS_COARSEN;
    if (tok >= num_tokens_ || bh >= num_heads_) return;
    auto sg = item.get_sub_group();

    // Load cos/sin for this token's position.
    const float* csr = cos_sin_cache_ + positions_[tok] * cos_sin_stride_;
    const float c0 = csr[lane], s0 = csr[HALF_ROPE + lane];
    const float c1 = csr[HALF_BLOCK + lane],
                s1 = csr[HALF_ROPE + HALF_BLOCK + lane];
    const int64_t q_token_stride = num_heads_ * HEAD_DIM;

    // Prefetch 2 heads of Q data as packed uint32 (2×bf16 per lane).
    uint32_t dd[HEADS_COARSEN][4];
#pragma unroll
    for (int i = 0; i < HEADS_COARSEN; ++i) {
      const uint32_t* q32 = reinterpret_cast<const uint32_t*>(
          q_ + tok * q_token_stride + (bh + i) * HEAD_DIM);
      dd[i][0] = q32[lane];
      dd[i][1] = q32[HALF_BLOCK + lane];
      dd[i][2] = q32[NOPE_DIM / 2 + lane];
      dd[i][3] = q32[NOPE_DIM / 2 + HALF_BLOCK + lane];
    }

#pragma unroll
    for (int hc = 0; hc < HEADS_COARSEN; ++hc) {
      const int64_t h = bh + hc;

      // Decode bf16 pairs from packed uint32.
      float n0l = (float)sycl::bit_cast<bf16_t>((uint16_t)dd[hc][0]);
      float n0h = (float)sycl::bit_cast<bf16_t>((uint16_t)(dd[hc][0] >> 16));
      float n1l = (float)sycl::bit_cast<bf16_t>((uint16_t)dd[hc][1]);
      float n1h = (float)sycl::bit_cast<bf16_t>((uint16_t)(dd[hc][1] >> 16));
      float xe0 = (float)sycl::bit_cast<bf16_t>((uint16_t)dd[hc][2]);
      float xo0 = (float)sycl::bit_cast<bf16_t>((uint16_t)(dd[hc][2] >> 16));
      float xe1 = (float)sycl::bit_cast<bf16_t>((uint16_t)dd[hc][3]);
      float xo1 = (float)sycl::bit_cast<bf16_t>((uint16_t)(dd[hc][3] >> 16));

      // GPT-J RoPE + bf16 roundtrip (matches Triton reference numerics).
      float re0 = xe0 * c0 - xo0 * s0, ro0 = xo0 * c0 + xe0 * s0;
      re0 = (float)(bf16_t)re0;
      ro0 = (float)(bf16_t)ro0;
      float re1 = xe1 * c1 - xo1 * s1, ro1 = xo1 * c1 + xe1 * s1;
      re1 = (float)(bf16_t)re1;
      ro1 = (float)(bf16_t)ro1;

      // Per-head absmax reduction.
      float lm = sycl::fmax(
          sycl::fmax(sycl::fabs(n0l), sycl::fabs(n0h)),
          sycl::fmax(sycl::fabs(n1l), sycl::fabs(n1h)));
      lm = sycl::fmax(
          lm,
          sycl::fmax(
              sycl::fmax(sycl::fabs(re0), sycl::fabs(ro0)),
              sycl::fmax(sycl::fabs(re1), sycl::fabs(ro1))));
      float amax = sycl::reduce_over_group(sg, lm, sycl::maximum<float>());

      // Scale (division-free power-of-2).
      float scale, inv_s;
      fp8_scale_pair(amax, scale, inv_s);

      // Quantize to FP8 E4M3FN.
      uint8_t f0 = fp8_cvt(n0l * inv_s), f1 = fp8_cvt(n0h * inv_s);
      uint8_t f2 = fp8_cvt(n1l * inv_s), f3 = fp8_cvt(n1h * inv_s);
      uint8_t f4 = fp8_cvt(re0 * inv_s), f5 = fp8_cvt(ro0 * inv_s);
      uint8_t f6 = fp8_cvt(re1 * inv_s), f7 = fp8_cvt(ro1 * inv_s);

      // Store FP8 as uint16 pairs (coalesced).
      uint8_t* out = q_fp8_ + (tok * num_heads_ + h) * HEAD_DIM;
      reinterpret_cast<uint16_t*>(out)[lane] =
          (uint16_t)f0 | ((uint16_t)f1 << 8);
      reinterpret_cast<uint16_t*>(out + 32)[lane] =
          (uint16_t)f2 | ((uint16_t)f3 << 8);
      reinterpret_cast<uint16_t*>(out + NOPE_DIM)[lane] =
          (uint16_t)f4 | ((uint16_t)f5 << 8);
      reinterpret_cast<uint16_t*>(out + NOPE_DIM + 32)[lane] =
          (uint16_t)f6 | ((uint16_t)f7 << 8);

      // Weight fold: weights_out = weights * scale * softmax_scale * head_scale
      if (lane == 0) {
        weights_out_[tok * num_heads_ + h] =
            (float)weights_[tok * num_heads_ + h] * scale * weight_combined_;
      }
    }
  }

 private:
  const bf16_t* __restrict__ q_;
  const int64_t* __restrict__ positions_;
  const float* __restrict__ cos_sin_cache_;
  const bf16_t* __restrict__ weights_;
  uint8_t* __restrict__ q_fp8_;
  float* __restrict__ weights_out_;
  float weight_combined_;
  int64_t num_tokens_, num_heads_;
  int64_t cos_sin_stride_;
};

}  // namespace vllm

using vllm::bf16_t;

void deepseek_fused_indexer_q_rope_fp8(
    const torch::Tensor& q,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin_cache,
    const torch::Tensor& index_weights,
    double softmax_scale,
    double head_scale,
    torch::Tensor& q_fp8,
    torch::Tensor& weights_out) {
  TORCH_CHECK(q.dim() == 3, "q must be [T, H, HEAD_DIM]");

  const int64_t num_tokens = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t head_dim = q.size(2);
  const int64_t rope_dim = cos_sin_cache.size(1);
  const int64_t cos_sin_stride = cos_sin_cache.stride(0);
  const float weight_combined =
      static_cast<float>(softmax_scale) * static_cast<float>(head_scale);

  // Dispatch based on (head_dim, rope_dim). Add new cases for future models.
  TORCH_CHECK(
      head_dim == 128 && rope_dim == 64,
      "deepseek_fused_indexer_q_rope_fp8: unsupported head_dim=",
      head_dim,
      " rope_dim=",
      rope_dim,
      ". Supported: head_dim=128, rope_dim=64");

  TORCH_CHECK(num_heads % 2 == 0, "num_heads must be divisible by 2");

  const int64_t nhg = num_heads / 2;

  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    auto* q_ptr = reinterpret_cast<const bf16_t*>(q.data_ptr());
    auto* pos_ptr = positions.data_ptr<int64_t>();
    auto* cs_ptr = cos_sin_cache.data_ptr<float>();
    auto* w_ptr = reinterpret_cast<const bf16_t*>(index_weights.data_ptr());
    auto* fp8_ptr = reinterpret_cast<uint8_t*>(q_fp8.data_ptr());
    auto* wo_ptr = weights_out.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<2>(
            {(size_t)num_tokens, (size_t)(nhg * vllm::SG_SIZE)},
            {1, (size_t)vllm::SG_SIZE}),
        vllm::deepseek_fused_indexer_q_rope_fp8_kernel<128, 64, 2>(
            q_ptr,
            pos_ptr,
            cs_ptr,
            w_ptr,
            fp8_ptr,
            wo_ptr,
            weight_combined,
            num_tokens,
            num_heads,
            cos_sin_stride));
  });
}
