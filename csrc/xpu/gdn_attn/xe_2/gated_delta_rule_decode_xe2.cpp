#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "gated_delta_rule_decode_xe2.h"
#include "gated_delta_rule_decode_xe2.hpp"

void gated_delta_rule_decode_xe2(
    sycl::queue& queue,
    torch::Tensor& core_attn_out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& b,
    const torch::Tensor& a,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    torch::Tensor& ssm_state,
    const torch::Tensor& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const int batch_size,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim,
    const int* token_indx) {
  TORCH_CHECK(cache_indices.is_contiguous(), "cache_indices must be contiguous");
  TORCH_CHECK(cache_indices.dtype() == torch::kInt32, "cache_indices must be int32");
  TORCH_CHECK(cache_indices.dim() == 1, "cache_indices must be 1D");
  TORCH_CHECK(cache_indices.size(0) == batch_size, "cache_indices size must match batch_size");
  TORCH_CHECK(num_v_heads % num_k_heads == 0, "num_v_heads must be divisible by num_k_heads");
  TORCH_CHECK(
      A_log.scalar_type() == at::kFloat,
      "A_log dtype must be float32, but got ",
      A_log.scalar_type());
  TORCH_CHECK(
      dt_bias.scalar_type() == core_attn_out.scalar_type(),
      "dt_bias dtype must match core_attn_out dtype, but got dt_bias=",
      dt_bias.scalar_type(),
      ", core_attn_out=",
      core_attn_out.scalar_type());
  TORCH_CHECK(head_k_dim % gdn::xe2::decode_sub_group_size == 0);

  const int ssm_state_stride_0 = ssm_state.stride(0);
  const bool* has_initial_state_ptr =
      has_initial_state.has_value()
          ? reinterpret_cast<bool*>(has_initial_state->data_ptr())
          : nullptr;
    const int* cache_indices_ptr =
      reinterpret_cast<const int*>(cache_indices.data_ptr());
  const int k_bucket_size = head_k_dim / gdn::xe2::decode_sub_group_size;

  if (core_attn_out.scalar_type() == at::kBFloat16) {
    using scalar_t = sycl::ext::oneapi::bfloat16;
    gdn::xe2::dispatch_state_dtype_decode_xe2<scalar_t>(
        queue,
        core_attn_out,
        q,
        k,
        v,
        b,
        a,
        A_log,
        dt_bias,
        ssm_state,
        ssm_state_stride_0,
        token_indx,
        cache_indices_ptr,
        has_initial_state_ptr,
        batch_size,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
        k_bucket_size);
  } else if (core_attn_out.scalar_type() == at::kHalf) {
    using scalar_t = sycl::half;
    gdn::xe2::dispatch_state_dtype_decode_xe2<scalar_t>(
        queue,
        core_attn_out,
        q,
        k,
        v,
        b,
        a,
        A_log,
        dt_bias,
        ssm_state,
        ssm_state_stride_0,
        token_indx,
        cache_indices_ptr,
        has_initial_state_ptr,
        batch_size,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
        k_bucket_size);
  } else {
    using scalar_t = float;
    gdn::xe2::dispatch_state_dtype_decode_xe2<scalar_t>(
        queue,
        core_attn_out,
        q,
        k,
        v,
        b,
        a,
        A_log,
        dt_bias,
        ssm_state,
        ssm_state_stride_0,
        token_indx,
        cache_indices_ptr,
        has_initial_state_ptr,
        batch_size,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
        k_bucket_size);
  }
}