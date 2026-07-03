// SPDX-License-Identifier: Apache-2.0
// Fused DeepSeek-V4 indexer Q: GPT-J RoPE + per-block MXFP4 quantize + weight
// fold. One sub-group (16 lanes) handles 4 heads (COARSEN=4) per token. Each
// head has 4 x 32-element blocks quantized to MXFP4 with ue8m0 scales.

#include <sycl/sycl.hpp>
#include "utils.h"
#include <cstdint>
#include <torch/all.h>

namespace vllm {

using bf16_t = sycl::ext::oneapi::bfloat16;

static constexpr int SG_SIZE = 16;
static constexpr float FP4_MAX = 6.0f;
static constexpr float INV_FP4_MAX = 1.0f / 6.0f;

// Staircase FP4 E2M1 encoder: maps float -> 4-bit code (sign|mantissa).
inline uint8_t encode_fp4(float x) {
  uint8_t sign = (x < 0.0f) ? 0x8u : 0x0u;
  float a = sycl::fabs(x);
  uint8_t code = (a > 0.25f) + (a > 0.75f) + (a > 1.25f) + (a > 1.75f) +
                 (a > 2.5f) + (a > 3.5f) + (a > 5.0f);
  return code | sign;
}

// Compute 1/scale and ue8m0 scale byte directly from amax.
// scale = 2^ceil(log2(amax/FP4_MAX)), always power-of-2.
// inv = 2^(127-k), computed via bit manipulation (no division).
inline float fast_inv_scale(float amax, uint8_t& ue8m0) {
  float ratio = amax * INV_FP4_MAX;
  uint32_t bits;
  __builtin_memcpy(&bits, &ratio, 4);
  if ((bits & 0x7F800000u) == 0) bits = 0x00800000u;
  if (bits & 0x7FFFFFu) bits = (bits + 0x800000u) & 0xFF800000u;
  ue8m0 = static_cast<uint8_t>((bits >> 23) & 0xFFu);
  const uint32_t inv_bits = 0x7F000000u - bits;
  float inv;
  __builtin_memcpy(&inv, &inv_bits, 4);
  return inv;
}

// Quantize a pair of floats to packed MXFP4 byte (lo=bits[3:0], hi=bits[7:4]).
inline uint8_t quant_pair(float x0, float x1, float inv_s) {
  float q0 = sycl::fmax(-FP4_MAX, sycl::fmin(x0 * inv_s, FP4_MAX));
  float q1 = sycl::fmax(-FP4_MAX, sycl::fmin(x1 * inv_s, FP4_MAX));
  return ((encode_fp4(q1) & 0xFu) << 4) | (encode_fp4(q0) & 0xFu);
}

template <int HEAD_DIM_, int ROPE_DIM_, int HEADS_COARSEN_ = 4>
class deepseek_fused_indexer_q_rope_mxfp4_kernel {
 public:
  static constexpr int HEAD_DIM = HEAD_DIM_;
  static constexpr int ROPE_DIM = ROPE_DIM_;
  static constexpr int NOPE_DIM = HEAD_DIM - ROPE_DIM;
  static constexpr int HALF_ROPE = ROPE_DIM / 2;
  static constexpr int HALF_BLOCK = SG_SIZE;
  static constexpr int HEADS_COARSEN = HEADS_COARSEN_;
  static constexpr int MXFP4_BLOCK_SIZE = 32;
  static constexpr int NUM_BLOCKS = HEAD_DIM / MXFP4_BLOCK_SIZE;
  static_assert(
      ROPE_DIM % 2 == 0 && HALF_ROPE >= SG_SIZE,
      "ROPE_DIM/2 must be >= SG_SIZE");
  static_assert(NOPE_DIM > 0, "HEAD_DIM must be > ROPE_DIM");

  deepseek_fused_indexer_q_rope_mxfp4_kernel(
      const bf16_t* __restrict__ q,
      const int64_t* __restrict__ pos,
      const float* __restrict__ cs,
      const bf16_t* __restrict__ w,
      uint8_t* __restrict__ packed,
      uint8_t* __restrict__ scales,
      float* __restrict__ wo,
      int64_t num_tokens,
      int64_t num_heads,
      float combined_scale)
      : q_(q),
        pos_(pos),
        cs_(cs),
        w_(w),
        packed_(packed),
        scales_(scales),
        wo_(wo),
        T_(num_tokens),
        H_(num_heads),
        combined_scale_(combined_scale) {}

  [[sycl::reqd_sub_group_size(SG_SIZE)]]
  void operator()(sycl::nd_item<2> item) const {
    const int tok = item.get_global_id(0);
    const int hg = item.get_global_id(1) / SG_SIZE;
    const int lane = item.get_local_id(1) % SG_SIZE;
    const int bh = hg * HEADS_COARSEN;

    if (tok >= T_ || bh >= H_) return;

    auto sg = item.get_sub_group();

    // Load cos/sin for this token's position
    const float* csr = cs_ + pos_[tok] * ROPE_DIM;
    const float c0 = csr[lane];
    const float s0 = csr[HALF_ROPE + lane];
    const float c1 = csr[HALF_BLOCK + lane];
    const float s1 = csr[HALF_ROPE + HALF_BLOCK + lane];

    const int64_t qs0 = (int64_t)H_ * HEAD_DIM;

    // Batch-load all 4 heads upfront (16 uint32 loads)
    uint32_t data[HEADS_COARSEN][4];
#pragma unroll
    for (int hc = 0; hc < HEADS_COARSEN; ++hc) {
      const uint32_t* q32 = reinterpret_cast<const uint32_t*>(
          q_ + tok * qs0 + (bh + hc) * HEAD_DIM);
      data[hc][0] = q32[lane];
      data[hc][1] = q32[HALF_BLOCK + lane];
      data[hc][2] = q32[NOPE_DIM / 2 + lane];
      data[hc][3] = q32[NOPE_DIM / 2 + HALF_BLOCK + lane];
    }

#pragma unroll
    for (int hc = 0; hc < HEADS_COARSEN; ++hc) {
      const int h = bh + hc;
      uint8_t* pd = packed_ + ((int64_t)tok * H_ + h) * (HEAD_DIM / 2);
      uint8_t* sd = scales_ + ((int64_t)tok * H_ + h) * NUM_BLOCKS;

      // NoPE block 0 & 1
      float n0_lo = static_cast<float>(
          sycl::bit_cast<bf16_t>(static_cast<uint16_t>(data[hc][0])));
      float n0_hi = static_cast<float>(
          sycl::bit_cast<bf16_t>(static_cast<uint16_t>(data[hc][0] >> 16)));
      float n1_lo = static_cast<float>(
          sycl::bit_cast<bf16_t>(static_cast<uint16_t>(data[hc][1])));
      float n1_hi = static_cast<float>(
          sycl::bit_cast<bf16_t>(static_cast<uint16_t>(data[hc][1] >> 16)));

      // RoPE block 0: GPT-J style
      float r0_e = static_cast<float>(
          sycl::bit_cast<bf16_t>(static_cast<uint16_t>(data[hc][2])));
      float r0_o = static_cast<float>(
          sycl::bit_cast<bf16_t>(static_cast<uint16_t>(data[hc][2] >> 16)));
      float r0_er = r0_e * c0 - r0_o * s0;
      float r0_or = r0_o * c0 + r0_e * s0;
      r0_er = static_cast<float>(static_cast<bf16_t>(r0_er));
      r0_or = static_cast<float>(static_cast<bf16_t>(r0_or));

      // RoPE block 1: GPT-J style
      float r1_e = static_cast<float>(
          sycl::bit_cast<bf16_t>(static_cast<uint16_t>(data[hc][3])));
      float r1_o = static_cast<float>(
          sycl::bit_cast<bf16_t>(static_cast<uint16_t>(data[hc][3] >> 16)));
      float r1_er = r1_e * c1 - r1_o * s1;
      float r1_or = r1_o * c1 + r1_e * s1;
      r1_er = static_cast<float>(static_cast<bf16_t>(r1_er));
      r1_or = static_cast<float>(static_cast<bf16_t>(r1_or));

      // Per-block absmax reduction
      float am0 = sycl::fmax(sycl::fabs(n0_lo), sycl::fabs(n0_hi));
      float am1 = sycl::fmax(sycl::fabs(n1_lo), sycl::fabs(n1_hi));
      float am2 = sycl::fmax(sycl::fabs(r0_er), sycl::fabs(r0_or));
      float am3 = sycl::fmax(sycl::fabs(r1_er), sycl::fabs(r1_or));

      am0 = sycl::reduce_over_group(sg, am0, sycl::maximum<float>());
      am1 = sycl::reduce_over_group(sg, am1, sycl::maximum<float>());
      am2 = sycl::reduce_over_group(sg, am2, sycl::maximum<float>());
      am3 = sycl::reduce_over_group(sg, am3, sycl::maximum<float>());

      // Scale + quantize all 4 blocks
      uint8_t ue0, ue1, ue2, ue3;
      float inv0 = fast_inv_scale(am0, ue0);
      float inv1 = fast_inv_scale(am1, ue1);
      float inv2 = fast_inv_scale(am2, ue2);
      float inv3 = fast_inv_scale(am3, ue3);

      uint8_t p0 = quant_pair(n0_lo, n0_hi, inv0);
      uint8_t p1 = quant_pair(n1_lo, n1_hi, inv1);
      uint8_t p2 = quant_pair(r0_er, r0_or, inv2);
      uint8_t p3 = quant_pair(r1_er, r1_or, inv3);

      // Batched writes
      pd[lane] = p0;
      pd[HALF_BLOCK + lane] = p1;
      pd[NOPE_DIM / 2 + lane] = p2;
      pd[NOPE_DIM / 2 + HALF_BLOCK + lane] = p3;

      if (lane == 0) {
        uint32_t sp = (uint32_t)ue0 | ((uint32_t)ue1 << 8) |
                      ((uint32_t)ue2 << 16) | ((uint32_t)ue3 << 24);
        *reinterpret_cast<uint32_t*>(sd) = sp;
        wo_[tok * H_ + h] =
            static_cast<float>(w_[tok * H_ + h]) * combined_scale_;
      }
    }
  }

 private:
  const bf16_t* __restrict__ q_;
  const int64_t* __restrict__ pos_;
  const float* __restrict__ cs_;
  const bf16_t* __restrict__ w_;
  uint8_t* __restrict__ packed_;
  uint8_t* __restrict__ scales_;
  float* __restrict__ wo_;
  int64_t T_;
  int64_t H_;
  float combined_scale_;
};

}  // namespace vllm

using vllm::bf16_t;

void deepseek_fused_indexer_q_rope_mxfp4(
    const at::Tensor& q,
    const at::Tensor& positions,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& index_weights,
    double softmax_scale,
    double head_scale,
    at::Tensor& packed_out,
    at::Tensor& scales_out,
    at::Tensor& weights_out) {
  using namespace vllm;

  const int64_t num_tokens = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t head_dim = q.size(2);
  const int64_t rope_dim = cos_sin_cache.size(1);
  const float combined_scale =
      static_cast<float>(softmax_scale) * static_cast<float>(head_scale);

  // Dispatch based on (head_dim, rope_dim). Add new cases for future models.
  TORCH_CHECK(
      head_dim == 128 && rope_dim == 64,
      "deepseek_fused_indexer_q_rope_mxfp4: unsupported head_dim=",
      head_dim,
      " rope_dim=",
      rope_dim,
      ". Supported: head_dim=128, rope_dim=64");

  TORCH_CHECK(num_heads % 4 == 0, "num_heads must be divisible by 4");

  const int nhg = num_heads / 4;

  const bf16_t* q_ptr = reinterpret_cast<const bf16_t*>(q.data_ptr());
  const int64_t* pos_ptr = positions.data_ptr<int64_t>();
  const float* cs_ptr = cos_sin_cache.data_ptr<float>();
  const bf16_t* w_ptr =
      reinterpret_cast<const bf16_t*>(index_weights.data_ptr());
  uint8_t* packed_ptr = packed_out.data_ptr<uint8_t>();
  uint8_t* scales_ptr = scales_out.data_ptr<uint8_t>();
  float* wo_ptr = weights_out.data_ptr<float>();

  auto& queue = xpu::vllmGetQueue();

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>(
            {static_cast<size_t>(num_tokens),
             static_cast<size_t>(nhg * SG_SIZE)},
            {1, static_cast<size_t>(SG_SIZE)}),
        deepseek_fused_indexer_q_rope_mxfp4_kernel<128, 64, 4>(
            q_ptr,
            pos_ptr,
            cs_ptr,
            w_ptr,
            packed_ptr,
            scales_ptr,
            wo_ptr,
            num_tokens,
            num_heads,
            combined_scale));
  });
}
