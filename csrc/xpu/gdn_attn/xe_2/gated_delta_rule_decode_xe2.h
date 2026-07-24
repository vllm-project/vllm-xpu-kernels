#include <sycl/sycl.hpp>
#include <torch/all.h>

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
    const int batch_size,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim,
    const int* token_indx = nullptr);

void gated_delta_rule_decode_xe2_spec(
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
    const torch::Tensor& num_accepted_tokens,
    const int num_spec_decodes,
    const int num_spec_tokens,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim,
    const int* token_indx = nullptr);