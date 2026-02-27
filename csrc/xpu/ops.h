#pragma once

#include <torch/all.h>

/**
 * Make sure the shape of A and B is correctly setting before calling below gemm
 * method implemented with OneDNN.
 *  A should be one of [b, m, k] and [m, k]
 *  B should be [k, n] or [k//8, n] in int4 precision, where [k//8, n] indicates
 * a packed representation with 8 int4 values packed into one byte along the k
 * dimension.
 */
torch::Tensor fp8_gemm(
    const torch::Tensor& A,
    const torch::Tensor& B,
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& A_scale_,
    const std::optional<torch::Tensor>& B_scale_,
    const std::optional<torch::Tensor>& bias_);

torch::Tensor fp8_gemm_w8a16(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const std::optional<torch::Tensor>& B_scale_,
    const std::optional<torch::Tensor>& bias_);

torch::Tensor int4_gemm_w4a16(
    const torch::Tensor& A_,
    const torch::Tensor& B,
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& B_scale,
    const torch::Tensor& B_zp,
    int64_t group_size,
    const std::optional<torch::Tensor>& g_idx);

torch::Tensor int4_gemm_w4a8(
    const torch::Tensor& A_,
    const torch::Tensor& A_scale,
    const torch::Tensor& A_zp,
    const torch::Tensor& B,
    const torch::Tensor& B_scale,
    const torch::Tensor& B_zp,
    int64_t group_size,
    const std::optional<torch::Tensor>& g_idx,
    const std::optional<torch::Tensor>& bias);

torch::Tensor cutlass_grouped_gemm_interface(
    torch::Tensor ptr_A,
    torch::Tensor ptr_B,
    const c10::optional<at::Tensor>& ptr_scales,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor expert_first_token_offset,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    bool is_B_int4,
    bool is_B_mxfp4);

std::tuple<at::Tensor, at::Tensor> deepseek_scaling_rope(
    const at::Tensor& positions,
    const at::Tensor& query,
    const at::Tensor& key,
    const c10::optional<at::Tensor>& offsets_opt,
    const at::Tensor& cos_sin_cache,
    int64_t rotary_dim,
    bool is_neox);

void gdn_attention(
    torch::Tensor& core_attn_out,
    torch::Tensor& z,
    const torch::Tensor& projected_states_qkvz,
    const torch::Tensor& projected_states_ba,
    const int64_t num_k_heads,
    const int64_t num_v_heads,
    const int64_t head_k_dim,
    const int64_t head_v_dim,
    torch::Tensor& conv_state,
    torch::Tensor& ssm_state,
    const torch::Tensor& conv_weights,
    const std::optional<torch::Tensor>& conv_bias,
    const std::string& activation,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    const int64_t num_prefills,
    const int64_t num_decodes,
    const std::optional<torch::Tensor>& has_initial_state,
    const torch::Tensor& non_spec_query_start_loc,
    const torch::Tensor& non_spec_state_indices_tensor,
    const int64_t num_actual_tokens,
    const int64_t tp_size);

bool is_bmg(int64_t device_index);

bool is_pvc(int64_t device_index);
