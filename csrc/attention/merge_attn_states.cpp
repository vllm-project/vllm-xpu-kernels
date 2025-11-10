#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>

#include <sycl/sycl.hpp>
#include <optional>
#include <torch/all.h>
#include <algorithm>
#include "utils.h"

namespace vllm {

// Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
// can be used to combine partial attention results (in the split-KV case)
template <typename scalar_t, const uint NUM_THREADS>
void merge_attn_states_kernel(scalar_t* output, float* output_lse,
                              const scalar_t* prefix_output,
                              const float* prefix_lse,
                              const scalar_t* suffix_output,
                              const float* suffix_lse, const uint num_tokens,
                              const uint num_heads, const uint head_size,
                              const sycl::nd_item<3>& item_ct1) {
  using pack_128b_t = sycl::uint4;
  const uint pack_size = 16 / sizeof(scalar_t);
  const uint threads_per_head = head_size / pack_size;

  const uint global_idx =
      item_ct1.get_group(2) * NUM_THREADS + item_ct1.get_local_id(2);
  const uint token_head_threads = num_tokens * num_heads * threads_per_head;

  if (global_idx >= token_head_threads) return;

  // global_idx -> token_idx + head_idx + pack_idx
  const uint token_head_idx = global_idx / threads_per_head;
  const uint pack_idx = global_idx % threads_per_head;

  const uint token_idx = token_head_idx / num_heads;
  const uint head_idx = token_head_idx % num_heads;

  const uint pack_offset = pack_idx * pack_size;  // (0~15)*8, etc.
  const uint head_offset =
      token_idx * num_heads * head_size + head_idx * head_size;
  const scalar_t* prefix_head_ptr = prefix_output + head_offset;
  const scalar_t* suffix_head_ptr = suffix_output + head_offset;
  scalar_t* output_head_ptr = output + head_offset;

  float p_lse = prefix_lse[head_idx * num_tokens + token_idx];
  float s_lse = suffix_lse[head_idx * num_tokens + token_idx];
  p_lse = sycl::isinf(p_lse) ? -std::numeric_limits<float>::infinity() : p_lse;
  s_lse = sycl::isinf(s_lse) ? -std::numeric_limits<float>::infinity() : s_lse;

  const float max_lse = sycl::fmax(p_lse, s_lse);
  p_lse = p_lse - max_lse;
  s_lse = s_lse - max_lse;
  const float p_se = sycl::native::exp(p_lse);
  const float s_se = sycl::native::exp(s_lse);
  const float out_se = p_se + s_se;
  const float p_scale = p_se / out_se;
  const float s_scale = s_se / out_se;

  if (pack_offset < head_size) {
    // Pack 128b load
    pack_128b_t p_out_pack = reinterpret_cast<const pack_128b_t*>(
        prefix_head_ptr)[pack_offset / pack_size];
    pack_128b_t s_out_pack = reinterpret_cast<const pack_128b_t*>(
        suffix_head_ptr)[pack_offset / pack_size];
    pack_128b_t o_out_pack;

#pragma unroll
  for (uint i = 0; i < pack_size; ++i) {
      // Always use float for FMA to keep high precision.
      // half(uint16_t), bfloat16, float -> float.
      const float p_out_f =
          vllm::xpu::to_float(reinterpret_cast<const scalar_t*>(&p_out_pack)[i]);
      const float s_out_f =
          vllm::xpu::to_float(reinterpret_cast<const scalar_t*>(&s_out_pack)[i]);
      // fma: a * b + c = p_out_f * p_scale + (s_out_f * s_scale)
      const float o_out_f = p_out_f * p_scale + (s_out_f * s_scale);
      // float -> half(uint16_t), bfloat16, float.
      vllm::xpu::from_float(reinterpret_cast<scalar_t*>(&o_out_pack)[i], o_out_f);
    }

    // Pack 128b storage
    reinterpret_cast<pack_128b_t*>(output_head_ptr)[pack_offset / pack_size] =
        o_out_pack;
  }
  // We only need to write to output_lse once per head.
  if (output_lse != nullptr && pack_idx == 0) {
    float out_lse = sycl::log((float)out_se) + max_lse;
    output_lse[head_idx * num_tokens + token_idx] = out_lse;
  }
}

}  // namespace vllm

// The following macro is used to dispatch the conversion function based on
// the output data type. The FN is a macro that calls a function with
// template<typename scalar_t>.
#define DISPATCH_BY_SCALAR_DTYPE(scalar_dtype, fn)                      \
  {                                                                     \
    if (scalar_dtype == at::ScalarType::Float) {                        \
      fn(float);                                                        \
    } else if (scalar_dtype == at::ScalarType::Half) {                  \
      fn(sycl::half);                                                     \
    } else if (scalar_dtype == at::ScalarType::BFloat16) {              \
      fn(sycl::ext::oneapi::bfloat16);                                  \
    } else {                                                            \
      TORCH_CHECK(false, "Unsupported data type of O: ", scalar_dtype); \
    }                                                                   \
  }

#define LAUNCH_MERGE_ATTN_STATES(scalar_t, NUM_THREADS)                  \
  {                                                                      \
    ((sycl::queue)(queue)).submit([&](sycl::handler& cgh) {           \
      auto output_data_ptr_ct0 =                                         \
          reinterpret_cast<scalar_t*>(output.data_ptr());                \
      auto output_lse_ptr_ct1 = output_lse_ptr;                          \
      auto prefix_output_data_ptr_ct2 =                                  \
          reinterpret_cast<scalar_t*>(prefix_output.data_ptr());         \
      auto prefix_lse_data_ptr_ct3 =                                     \
          reinterpret_cast<float*>(prefix_lse.data_ptr());               \
      auto suffix_output_data_ptr_ct4 =                                  \
          reinterpret_cast<scalar_t*>(suffix_output.data_ptr());         \
      auto suffix_lse_data_ptr_ct5 =                                     \
          reinterpret_cast<float*>(suffix_lse.data_ptr());               \
      auto num_tokens_ct6 = num_tokens;                                  \
      auto num_heads_ct7 = num_heads;                                    \
      auto head_size_ct8 = head_size;                                    \
                                                                         \
      cgh.parallel_for(                                                  \
          sycl::nd_range<3>(grid * block, block),                        \
          [=](sycl::nd_item<3> item_ct1) {                               \
            vllm::merge_attn_states_kernel<scalar_t, NUM_THREADS>(       \
                output_data_ptr_ct0, output_lse_ptr_ct1,                 \
                prefix_output_data_ptr_ct2, prefix_lse_data_ptr_ct3,     \
                suffix_output_data_ptr_ct4, suffix_lse_data_ptr_ct5,     \
                num_tokens_ct6, num_heads_ct7, head_size_ct8, item_ct1); \
          });                                                            \
    });                                                                  \
  }

/*@brief Merges the attention states from prefix and suffix
 * into the output tensor. NUM_TOKENS: n, NUM_HEADS: h, HEAD_SIZE: d
 *
 * @param output [n,h,d] The output tensor to store the merged attention states.
 * @param output_lse [h,d] Optional tensor to store the log-sum-exp values.
 * @param prefix_output [n,h,d] The prefix attention states.
 * @param prefix_lse [h,n] The log-sum-exp values for the prefix attention
 * states.
 * @param suffix_output [n,h,d] The suffix attention states.
 * @param suffix_lse [h,n] The log-sum-exp values for the suffix attention
 * states.
 */
template <typename scalar_t>
void merge_attn_states_launcher(torch::Tensor& output,
                                std::optional<torch::Tensor> output_lse,
                                const torch::Tensor& prefix_output,
                                const torch::Tensor& prefix_lse,
                                const torch::Tensor& suffix_output,
                                const torch::Tensor& suffix_lse) {
  constexpr uint NUM_THREADS = 128;
  const uint num_tokens = output.size(0);
  const uint num_heads = output.size(1);
  const uint head_size = output.size(2);
  const uint pack_size = 16 / sizeof(scalar_t);
  TORCH_CHECK(head_size % pack_size == 0,
              "headsize must be multiple of pack_size:", pack_size);
  TORCH_CHECK(output.stride(-2) == head_size && output.stride(-1) == 1,
              "output heads must be contiguous in memory");
  TORCH_CHECK(
      prefix_output.stride(-2) == head_size && prefix_output.stride(-1) == 1,
      "prefix_output heads must be contiguous in memory");
  TORCH_CHECK(
      suffix_output.stride(-2) == head_size && suffix_output.stride(-1) == 1,
      "suffix_output heads must be contiguous in memory");
  float* output_lse_ptr = nullptr;
  if (output_lse.has_value()) {
    output_lse_ptr = output_lse.value().data_ptr<float>();
  }
  // Process one pack elements per thread. for float, the
  // pack_size is 4 for half/bf16, the pack_size is 8.
  const uint threads_per_head = head_size / pack_size;
  const uint total_threads = num_tokens * num_heads * threads_per_head;

  sycl::range<3> block(1,1,NUM_THREADS);
  sycl::range<3> grid(1,1,(total_threads + NUM_THREADS - 1) / NUM_THREADS);

  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);
  auto& queue = vllm::xpu::vllmGetQueue();

  LAUNCH_MERGE_ATTN_STATES(scalar_t, NUM_THREADS);
}

#define CALL_MERGE_ATTN_STATES_LAUNCHER(scalar_t)                           \
  {                                                                         \
    merge_attn_states_launcher<scalar_t>(output, output_lse, prefix_output, \
                                         prefix_lse, suffix_output,         \
                                         suffix_lse);                       \
  }

void merge_attn_states(torch::Tensor& output,
                       std::optional<torch::Tensor> output_lse,
                       const torch::Tensor& prefix_output,
                       const torch::Tensor& prefix_lse,
                       const torch::Tensor& suffix_output,
                       const torch::Tensor& suffix_lse) {
  DISPATCH_BY_SCALAR_DTYPE(output.dtype(), CALL_MERGE_ATTN_STATES_LAUNCHER);
}