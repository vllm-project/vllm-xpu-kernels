#include <sycl/sycl.hpp>
#include "utils.h"
#include <cstdint>
#include <torch/all.h>

namespace vllm {

using bf16_t = sycl::ext::oneapi::bfloat16;

/**
 * Fused inverse GPT-J RoPE kernel for DeepSeek-V4.
 *
 * Each work-item (one sub-group of SG_SIZE lanes) processes one (token, head)
 * pair.  The kernel:
 *   1. Copies nope_dim bf16 elements unchanged (bulk uint64 copies).
 *   2. Applies inverse GPT-J rotation on rope_dim elements using fp32
 *      cos/sin cache.
 *   3. Writes output in group-major layout: [T, G, hpg*D].
 *
 * Input:  o             [num_tokens, num_heads, head_dim] bf16
 *         positions     [num_tokens] int64
 *         cos_sin_cache [max_pos, rope_dim] float32  (cos||sin concatenated)
 * Output: out           [num_tokens, n_groups, heads_per_group * head_dim] bf16
 */
class deepseek_inv_rope_bf16_kernel {
 public:
  static constexpr int SG_SIZE = 16;

  deepseek_inv_rope_bf16_kernel(
      const bf16_t* __restrict__ o,
      const int64_t* __restrict__ positions,
      const float* __restrict__ cos_sin_cache,
      bf16_t* __restrict__ out,
      int64_t num_tokens,
      int64_t num_heads,
      int64_t heads_per_group,
      int64_t head_dim,
      int64_t nope_dim,
      int64_t half_rope,
      int64_t nope_iters,
      int64_t rope_iters,
      int64_t o_stride_token,
      int64_t o_stride_head,
      int64_t cache_stride_pos,
      int64_t out_stride_token,
      int64_t out_stride_group)
      : o_(o),
        positions_(positions),
        cos_sin_cache_(cos_sin_cache),
        out_(out),
        num_tokens_(num_tokens),
        num_heads_(num_heads),
        heads_per_group_(heads_per_group),
        head_dim_(head_dim),
        nope_dim_(nope_dim),
        half_rope_(half_rope),
        nope_iters_(nope_iters),
        rope_iters_(rope_iters),
        o_stride_token_(o_stride_token),
        o_stride_head_(o_stride_head),
        cache_stride_pos_(cache_stride_pos),
        out_stride_token_(out_stride_token),
        out_stride_group_(out_stride_group) {}

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
    bf16_t* dst = out_ + token_idx * out_stride_token_ +
                  group * out_stride_group_ + head_in_group * head_dim_;

    // Bulk copy NoPE dimensions (uint64 = 4 bf16 elements per lane per iter)
    {
      const uint64_t* s8 = reinterpret_cast<const uint64_t*>(src);
      uint64_t* d8 = reinterpret_cast<uint64_t*>(dst);
      for (int64_t i = 0; i < nope_iters_; ++i) {
        d8[lane + i * SG_SIZE] = s8[lane + i * SG_SIZE];
      }
    }

    // Inverse GPT-J RoPE rotation in fp32
    const int64_t pos = positions_[token_idx];
    const float* cos_ptr = cos_sin_cache_ + pos * cache_stride_pos_;
    const float* sin_ptr = cos_ptr + half_rope_;

    for (int64_t pp = 0; pp < rope_iters_; ++pp) {
      const int64_t p = lane + pp * SG_SIZE;
      const float x_even = static_cast<float>(src[nope_dim_ + 2 * p]);
      const float x_odd = static_cast<float>(src[nope_dim_ + 2 * p + 1]);
      const float c = cos_ptr[p];
      const float s = sin_ptr[p];
      // Inverse rotation: x_new = x*cos + y*sin, y_new = y*cos - x*sin
      dst[nope_dim_ + 2 * p] = static_cast<bf16_t>(x_even * c + x_odd * s);
      dst[nope_dim_ + 2 * p + 1] = static_cast<bf16_t>(x_odd * c - x_even * s);
    }
  }

 private:
  const bf16_t* __restrict__ o_;
  const int64_t* __restrict__ positions_;
  const float* __restrict__ cos_sin_cache_;
  bf16_t* __restrict__ out_;
  int64_t num_tokens_, num_heads_;
  int64_t heads_per_group_, head_dim_, nope_dim_, half_rope_;
  int64_t nope_iters_, rope_iters_;
  int64_t o_stride_token_, o_stride_head_;
  int64_t cache_stride_pos_;
  int64_t out_stride_token_, out_stride_group_;
};

}  // namespace vllm

/**
 * @brief Fused inverse GPT-J RoPE for DeepSeek-V4 attention output.
 *
 * Applies inverse RoPE rotation and regroups heads into group-major layout.
 *
 * @param attn_output   Attention output [num_tokens, num_heads, head_dim] bf16
 * @param positions     Token positions [num_tokens] int64
 * @param cos_sin_cache Precomputed [max_pos, rope_dim] float32 (cos||sin)
 * @param n_groups      Number of head groups
 * @param heads_per_group Heads per group
 * @param nope_dim      Non-RoPE dimensions per head
 * @param rope_dim      RoPE dimensions per head
 * @return Output tensor [num_tokens, n_groups, heads_per_group * head_dim] bf16
 */
torch::Tensor deepseek_inv_rope_bf16(
    const torch::Tensor& attn_output,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin_cache,
    int64_t n_groups,
    int64_t heads_per_group,
    int64_t nope_dim,
    int64_t rope_dim) {
  using bf16_t = sycl::ext::oneapi::bfloat16;
  constexpr int SG_SIZE = vllm::deepseek_inv_rope_bf16_kernel::SG_SIZE;

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
      cos_sin_cache.size(-1) == rope_dim,
      "cos_sin_cache last dim must equal rope_dim");
  TORCH_CHECK(
      nope_dim % (4 * SG_SIZE) == 0,
      "nope_dim must be divisible by 4*SG_SIZE (64)");
  TORCH_CHECK(
      (rope_dim / 2) % SG_SIZE == 0,
      "half_rope must be divisible by SG_SIZE (16)");

  const int64_t half_rope = rope_dim / 2;
  const int64_t nope_iters = nope_dim / (4 * SG_SIZE);
  const int64_t rope_iters = half_rope / SG_SIZE;

  // Output: [num_tokens, n_groups, heads_per_group * head_dim]
  const int64_t d = heads_per_group * head_dim;
  auto out = torch::empty(
      {num_tokens, n_groups, d}, o.options().dtype(torch::kBFloat16));

  if (num_tokens == 0) {
    return out;
  }

  const int64_t o_stride_token = o.stride(0);
  const int64_t o_stride_head = o.stride(1);
  const int64_t cache_stride_pos = cos_sin_cache.stride(0);
  const int64_t out_stride_token = out.stride(0);
  const int64_t out_stride_group = out.stride(1);

  sycl::range<3> local(1, 1, SG_SIZE);
  sycl::range<3> global(num_tokens, num_heads, SG_SIZE);

  auto& q = vllm::xpu::vllmGetQueue();
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(global, local),
        vllm::deepseek_inv_rope_bf16_kernel(
            reinterpret_cast<const bf16_t*>(o.data_ptr()),
            reinterpret_cast<const int64_t*>(positions.data_ptr()),
            reinterpret_cast<const float*>(cos_sin_cache.data_ptr()),
            reinterpret_cast<bf16_t*>(out.data_ptr()),
            num_tokens,
            num_heads,
            heads_per_group,
            head_dim,
            nope_dim,
            half_rope,
            nope_iters,
            rope_iters,
            o_stride_token,
            o_stride_head,
            cache_stride_pos,
            out_stride_token,
            out_stride_group));
  });

  return out;
}
