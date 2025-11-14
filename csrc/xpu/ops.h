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

torch::Tensor cutlass_grouped_gemm(
    torch::Tensor ptr_A,
    torch::Tensor ptr_B,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor expert_token_count,
    torch::Tensor expert_first_token_offset,
    int64_t N,
    int64_t K,
    int64_t groups);

std::tuple<at::Tensor, at::Tensor> deepseek_scaling_rope(
    const at::Tensor& positions,
    const at::Tensor& query,
    const at::Tensor& key,
    const c10::optional<at::Tensor>& offsets_opt,
    const at::Tensor& cos_sin_cache,
    int64_t rotary_dim,
    bool is_neox);

void fused_moe(
    torch::Tensor output,
    torch::Tensor input,
    torch::Tensor token_selected_experts,
    torch::Tensor token_final_scales,
    torch::Tensor fc1_expert_weights,
    torch::Tensor fc2_expert_weights,
    torch::Tensor workspace);
