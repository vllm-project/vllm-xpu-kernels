// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// apply_rotary_emb: takes cos/sin directly (flash_attn style) for diffusion
// models.

#include <sycl/sycl.hpp>
#include "utils.h"
#include "dispatch_utils.h"

namespace vllm {

template <typename scalar_t, bool IS_NEOX>
class apply_rotary_emb_kernel {
 public:
  apply_rotary_emb_kernel(
      scalar_t* __restrict__ output_,
      const scalar_t* __restrict__ input_,
      const scalar_t* __restrict__ cos_,
      const scalar_t* __restrict__ sin_,
      const int num_tokens_,
      const int num_heads_,
      const int head_size_,
      const int rot_dim_,
      const int64_t input_stride_,
      const int64_t head_stride_,
      const int64_t cos_stride_)
      : output(output_),
        input(input_),
        cos_ptr(cos_),
        sin_ptr(sin_),
        num_tokens(num_tokens_),
        num_heads(num_heads_),
        head_size(head_size_),
        rot_dim(rot_dim_),
        input_stride(input_stride_),
        head_stride(head_stride_),
        cos_stride(cos_stride_) {}

  void operator()
      [[sycl::reqd_sub_group_size(32)]] (const sycl::nd_item<3>& item) const {
    const int token_idx = item.get_group(2);
    const int embed_dim = rot_dim / 2;

    const scalar_t* token_cos = cos_ptr + token_idx * cos_stride;
    const scalar_t* token_sin = sin_ptr + token_idx * cos_stride;

    const int nq = num_heads * embed_dim;
    for (int i = item.get_local_id(2); i < nq; i += item.get_local_range(2)) {
      const int head_idx = i / embed_dim;
      const int rot_offset = i % embed_dim;

      const int64_t in_base = token_idx * input_stride + head_idx * head_stride;

      int x_index, y_index;
      scalar_t cos_val, sin_val;
      if constexpr (IS_NEOX) {
        x_index = rot_offset;
        y_index = embed_dim + rot_offset;
        cos_val = token_cos[rot_offset];
        sin_val = token_sin[rot_offset];
      } else {
        x_index = 2 * rot_offset;
        y_index = 2 * rot_offset + 1;
        cos_val = token_cos[rot_offset];
        sin_val = token_sin[rot_offset];
      }

      const scalar_t x = input[in_base + x_index];
      const scalar_t y = input[in_base + y_index];
      output[in_base + x_index] = x * cos_val - y * sin_val;
      output[in_base + y_index] = y * cos_val + x * sin_val;
    }

    if (rot_dim < head_size) {
      const int remaining = head_size - rot_dim;
      for (int i = item.get_local_id(2); i < num_heads * remaining;
           i += item.get_local_range(2)) {
        const int head_idx = i / remaining;
        const int dim_offset = rot_dim + (i % remaining);
        const int64_t idx =
            token_idx * input_stride + head_idx * head_stride + dim_offset;
        output[idx] = input[idx];
      }
    }
  }

 private:
  scalar_t* __restrict__ output;
  const scalar_t* __restrict__ input;
  const scalar_t* __restrict__ cos_ptr;
  const scalar_t* __restrict__ sin_ptr;
  const int num_tokens;
  const int num_heads;
  const int head_size;
  const int rot_dim;
  const int64_t input_stride;
  const int64_t head_stride;
  const int64_t cos_stride;
};

}  // namespace vllm

void apply_rotary_emb(
    torch::Tensor& output,  // [num_tokens, num_heads, head_size]
    torch::Tensor& input,   // [num_tokens, num_heads, head_size]
    torch::Tensor& cos,     // [num_tokens, rot_dim/2]
    torch::Tensor& sin,     // [num_tokens, rot_dim/2]
    bool is_neox) {
  TORCH_CHECK(
      input.dim() == 3, "input must be 3D [num_tokens, num_heads, head_size]");
  TORCH_CHECK(cos.dim() == 2, "cos must be 2D [num_tokens, rot_dim/2]");
  TORCH_CHECK(sin.dim() == 2, "sin must be 2D [num_tokens, rot_dim/2]");
  TORCH_CHECK(input.stride(-1) == 1, "input must be contiguous in last dim");
  TORCH_CHECK(cos.stride(-1) == 1, "cos must be contiguous in last dim");

  int num_tokens = input.size(0);
  int num_heads = input.size(1);
  int head_size = input.size(2);
  int rot_dim = cos.size(1) * 2;

  TORCH_CHECK(
      cos.size(0) == num_tokens,
      "cos num_tokens mismatch: ",
      cos.size(0),
      " vs ",
      num_tokens);
  TORCH_CHECK(
      rot_dim <= head_size,
      "rot_dim (",
      rot_dim,
      ") must be <= head_size (",
      head_size,
      ")");

  int64_t input_stride = input.stride(0);
  int64_t head_stride = input.stride(1);
  int64_t cos_stride = cos.stride(0);

  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min<int64_t>(num_heads * rot_dim / 2, 512));

  at::DeviceGuard device_guard(input.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "apply_rotary_emb", [&] {
    using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
    auto input_ptr = (sycl_t*)input.data_ptr<scalar_t>();
    auto output_ptr = (sycl_t*)output.data_ptr<scalar_t>();
    auto cos_ptr = (sycl_t*)cos.data_ptr<scalar_t>();
    auto sin_ptr = (sycl_t*)sin.data_ptr<scalar_t>();

    if (is_neox) {
      queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            vllm::apply_rotary_emb_kernel<sycl_t, true>(
                output_ptr,
                input_ptr,
                cos_ptr,
                sin_ptr,
                num_tokens,
                num_heads,
                head_size,
                rot_dim,
                input_stride,
                head_stride,
                cos_stride));
      });
    } else {
      queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            vllm::apply_rotary_emb_kernel<sycl_t, false>(
                output_ptr,
                input_ptr,
                cos_ptr,
                sin_ptr,
                num_tokens,
                num_heads,
                head_size,
                rot_dim,
                input_stride,
                head_stride,
                cos_stride));
      });
    }
  });
}
