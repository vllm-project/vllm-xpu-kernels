#include <sycl/sycl.hpp>
#include "utils.h"
#include <cstdint>
#include <torch/all.h>

namespace vllm {

using bf16_t = sycl::ext::oneapi::bfloat16;
using fp8_t = uint8_t;

static constexpr int QUANT_GROUP_SIZE = 128;
static constexpr float FP8_E4M3_MAX = 448.0f;

// Ceil to power-of-2; also returns biased exponent in *out_scale_exp.
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

// Divide by pow2 scale + convert to fp8_e4m3 in pure integer arithmetic.
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

// Fused inv RoPE + FP8 quant kernel. One sub-group (16 lanes) per (token,
// head). Loads all 512 elements upfront, applies inv RoPE, then per-128-block
// quantizes.
class deepseek_inv_rope_fp8_quant_kernel {
 public:
  static constexpr int SG_SIZE = 16;
  static constexpr int CHUNKS_PER_HEAD = 4;
  static constexpr int ELEMS_PER_LANE = QUANT_GROUP_SIZE / SG_SIZE;

  deepseek_inv_rope_fp8_quant_kernel(
      const bf16_t* __restrict__ o,
      const int64_t* __restrict__ positions,
      const float* __restrict__ cos_sin_cache,
      fp8_t* __restrict__ fp8_out,
      float* __restrict__ scale_out,
      int64_t num_tokens,
      int64_t num_heads,
      int64_t heads_per_group,
      int64_t head_dim,
      int64_t nope_dim,
      int64_t half_rope,
      int64_t o_stride_token,
      int64_t o_stride_head,
      int64_t cache_stride_pos,
      int64_t fp8_stride_group,
      int64_t fp8_stride_token,
      int64_t scale_stride_group,
      int64_t scale_stride_token)
      : o_(o),
        positions_(positions),
        cos_sin_cache_(cos_sin_cache),
        fp8_out_(fp8_out),
        scale_out_(scale_out),
        num_tokens_(num_tokens),
        num_heads_(num_heads),
        heads_per_group_(heads_per_group),
        head_dim_(head_dim),
        nope_dim_(nope_dim),
        half_rope_(half_rope),
        o_stride_token_(o_stride_token),
        o_stride_head_(o_stride_head),
        cache_stride_pos_(cache_stride_pos),
        fp8_stride_group_(fp8_stride_group),
        fp8_stride_token_(fp8_stride_token),
        scale_stride_group_(scale_stride_group),
        scale_stride_token_(scale_stride_token) {}

  [[sycl::reqd_sub_group_size(SG_SIZE)]]
  void operator()(sycl::nd_item<3> item) const {
    const int64_t token_idx = item.get_global_id(0);
    const int64_t head_idx = item.get_global_id(1);
    const int64_t lane = item.get_local_id(2);

    if (token_idx >= num_tokens_ || head_idx >= num_heads_) return;

    const int64_t group = head_idx / heads_per_group_;
    const int64_t head_in_group = head_idx % heads_per_group_;

    const bf16_t* src =
        o_ + token_idx * o_stride_token_ + head_idx * o_stride_head_;

    fp8_t* fp8_dst = fp8_out_ + group * fp8_stride_group_ +
                     token_idx * fp8_stride_token_ + head_in_group * head_dim_;

    float* scale_dst = scale_out_ + group * scale_stride_group_ +
                       token_idx * scale_stride_token_ +
                       head_in_group * CHUNKS_PER_HEAD;

    const int lane32 = static_cast<int>(lane);
    auto sg = item.get_sub_group();

    // Load all 4 chunks into registers upfront.
    float vals[CHUNKS_PER_HEAD][ELEMS_PER_LANE];

#pragma unroll
    for (int chunk = 0; chunk < CHUNKS_PER_HEAD; ++chunk) {
      const uint32_t* s = reinterpret_cast<const uint32_t*>(
          src + chunk * QUANT_GROUP_SIZE + lane32 * ELEMS_PER_LANE);
#pragma unroll
      for (int e = 0; e < ELEMS_PER_LANE / 2; ++e) {
        uint32_t pair = s[e];
        vals[chunk][e * 2] = static_cast<float>(
            sycl::bit_cast<bf16_t>(static_cast<uint16_t>(pair)));
        vals[chunk][e * 2 + 1] = static_cast<float>(
            sycl::bit_cast<bf16_t>(static_cast<uint16_t>(pair >> 16)));
      }
    }

    const int64_t pos = positions_[token_idx];
    const float* cos_ptr = cos_sin_cache_ + pos * cache_stride_pos_;
    const float* sin_ptr = cos_ptr + half_rope_;

    // Inverse RoPE on chunk 3 (lanes 8-15 = rope region).
    if (lane32 >= 8) {
      const int pair_base = (lane32 - 8) * 4;
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        float c = cos_ptr[pair_base + p];
        float s = sin_ptr[pair_base + p];
        float x_even = vals[3][2 * p];
        float x_odd = vals[3][2 * p + 1];
        vals[3][2 * p] = x_even * c + x_odd * s;
        vals[3][2 * p + 1] = x_odd * c - x_even * s;
      }
    }

// Per-128-block: absmax reduce → pow2 scale → fused fp8 quantize.
#pragma unroll
    for (int chunk = 0; chunk < CHUNKS_PER_HEAD; ++chunk) {
      float local_absmax = 0.0f;
#pragma unroll
      for (int e = 0; e < ELEMS_PER_LANE; ++e)
        local_absmax = sycl::fmax(local_absmax, sycl::fabs(vals[chunk][e]));

      float block_absmax =
          sycl::reduce_over_group(sg, local_absmax, sycl::maximum<float>());
      float scale_raw =
          sycl::fmax(block_absmax, 1e-10f) * (1.0f / FP8_E4M3_MAX);
      uint32_t scale_exp;
      float scale = fast_ceil_pow2(scale_raw, scale_exp);
      if (lane == 0) scale_dst[chunk] = scale;

      uint64_t packed = 0;
#pragma unroll
      for (int e = 0; e < ELEMS_PER_LANE; ++e)
        packed |=
            (static_cast<uint64_t>(
                 fused_scale_to_fp8(vals[chunk][e], scale_exp))
             << (e * 8));
      *reinterpret_cast<uint64_t*>(
          fp8_dst + chunk * QUANT_GROUP_SIZE + lane32 * ELEMS_PER_LANE) =
          packed;
    }
  }

 private:
  const bf16_t* __restrict__ o_;
  const int64_t* __restrict__ positions_;
  const float* __restrict__ cos_sin_cache_;
  fp8_t* __restrict__ fp8_out_;
  float* __restrict__ scale_out_;
  int64_t num_tokens_, num_heads_;
  int64_t heads_per_group_, head_dim_, nope_dim_, half_rope_;
  int64_t o_stride_token_, o_stride_head_;
  int64_t cache_stride_pos_;
  int64_t fp8_stride_group_, fp8_stride_token_;
  int64_t scale_stride_group_, scale_stride_token_;
};

}  // namespace vllm

// PyTorch wrapper: inv RoPE + FP8 quant.
// Returns (fp8_out [G,T,hpg*D], scale_out [G,T,hpg*chunks]) .
std::tuple<torch::Tensor, torch::Tensor> deepseek_inv_rope_fp8_quant(
    const torch::Tensor& attn_output,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin_cache,
    int64_t n_groups,
    int64_t heads_per_group,
    int64_t nope_dim,
    int64_t rope_dim) {
  using bf16_t = sycl::ext::oneapi::bfloat16;
  using fp8_t = uint8_t;
  constexpr int SG_SIZE = vllm::deepseek_inv_rope_fp8_quant_kernel::SG_SIZE;
  constexpr int QUANT_GROUP_SIZE = vllm::QUANT_GROUP_SIZE;

  const auto& o = attn_output;
  TORCH_CHECK(o.dim() == 3, "o must be 3D [num_tokens, num_heads, head_dim]");
  TORCH_CHECK(o.scalar_type() == torch::kBFloat16, "o must be bfloat16");
  TORCH_CHECK(o.is_contiguous(), "o must be contiguous");
  TORCH_CHECK(
      positions.scalar_type() == torch::kLong, "positions must be int64");
  TORCH_CHECK(
      cos_sin_cache.scalar_type() == torch::kFloat32,
      "cos_sin_cache must be float32");
  TORCH_CHECK(
      cos_sin_cache.stride(-1) == 1,
      "cos_sin_cache must have contiguous last dim");

  const int64_t num_tokens = o.size(0);
  const int64_t num_heads = o.size(1);
  const int64_t head_dim = o.size(2);

  TORCH_CHECK(
      positions.size(0) == num_tokens,
      "positions length must equal num_tokens");

  TORCH_CHECK(
      num_heads == n_groups * heads_per_group,
      "num_heads must equal n_groups * heads_per_group");
  TORCH_CHECK(
      head_dim == nope_dim + rope_dim,
      "head_dim must equal nope_dim + rope_dim");
  TORCH_CHECK(rope_dim % 2 == 0, "rope_dim must be even");
  TORCH_CHECK(
      head_dim % QUANT_GROUP_SIZE == 0,
      "head_dim must be divisible by QUANT_GROUP_SIZE (128)");
  TORCH_CHECK(
      cos_sin_cache.size(-1) == rope_dim,
      "cos_sin_cache last dim must equal rope_dim");
  TORCH_CHECK(
      head_dim == 512 && nope_dim == 448 && rope_dim == 64,
      "Currently only supports head_dim=512, nope_dim=448, rope_dim=64");

  const int64_t half_rope = rope_dim / 2;
  const int64_t d = heads_per_group * head_dim;
  const int64_t chunks_per_head = head_dim / QUANT_GROUP_SIZE;
  const int64_t num_scale_blocks = heads_per_group * chunks_per_head;

  auto fp8_out = torch::empty(
      {n_groups, num_tokens, d}, o.options().dtype(torch::kFloat8_e4m3fn));

  auto scale_out = torch::empty(
      {n_groups, num_tokens, num_scale_blocks},
      o.options().dtype(torch::kFloat32));

  if (num_tokens == 0) {
    return {fp8_out, scale_out};
  }

  const int64_t o_stride_token = o.stride(0);
  const int64_t o_stride_head = o.stride(1);
  const int64_t cache_stride_pos = cos_sin_cache.stride(0);
  const int64_t fp8_stride_group = fp8_out.stride(0);
  const int64_t fp8_stride_token = fp8_out.stride(1);
  const int64_t scale_stride_group = scale_out.stride(0);
  const int64_t scale_stride_token = scale_out.stride(1);

  sycl::range<3> local(1, 1, SG_SIZE);
  sycl::range<3> global(num_tokens, num_heads, SG_SIZE);

  auto& q = vllm::xpu::vllmGetQueue();
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(global, local),
        vllm::deepseek_inv_rope_fp8_quant_kernel(
            reinterpret_cast<const bf16_t*>(o.data_ptr()),
            reinterpret_cast<const int64_t*>(positions.data_ptr()),
            reinterpret_cast<const float*>(cos_sin_cache.data_ptr()),
            reinterpret_cast<fp8_t*>(fp8_out.data_ptr()),
            reinterpret_cast<float*>(scale_out.data_ptr()),
            num_tokens,
            num_heads,
            heads_per_group,
            head_dim,
            nope_dim,
            half_rope,
            o_stride_token,
            o_stride_head,
            cache_stride_pos,
            fp8_stride_group,
            fp8_stride_token,
            scale_stride_group,
            scale_stride_token));
  });

  return {fp8_out, scale_out};
}
