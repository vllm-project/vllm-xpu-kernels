#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "chunk_gated_delta_rule_kernels_xe2.hpp"
#include "chunk_gated_delta_rule_xe2.h"

void chunk_gated_delta_rule_xe2(
    sycl::queue& queue,
    torch::Tensor& core_attn_out,  // [total_seqlen, num_v_heads, head_v_dim]
    const torch::Tensor& q,  // [total_virtual_seqlen, num_k_heads, head_k_dim]
    const torch::Tensor& k,  // [total_virtual_seqlen, num_k_heads, head_k_dim]
    const torch::Tensor& v,  // [total_virtual_seqlen, num_v_heads, head_v_dim]
    const torch::Tensor& b,  // [num_v_heads, total_virtual_seqlen]
    const torch::Tensor& a,  // [num_v_heads，total_virtual_seqlen]
    const torch::Tensor& A_log,    // [num_v_heads]
    const torch::Tensor& dt_bias,  // [num_v_heads]
    torch::Tensor&
        ssm_state,  // [cache_batch_size, num_v_heads, head_v_dim, head_k_dim]
    const torch::Tensor& query_start_loc,  // [batch_size + 1]
    const torch::Tensor& cache_indices,    // [batch_size]
    const std::optional<torch::Tensor>&
        has_initial_state,  // [batch_size] or None
    const int num_prefills,
    const int num_decodes) {
  gdn::chunk_gated_delta_rule_impl_xe2(
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
      query_start_loc,
      cache_indices,
      has_initial_state,
      num_prefills,
      num_decodes);
}