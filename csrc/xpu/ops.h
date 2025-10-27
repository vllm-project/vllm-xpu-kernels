#pragma once

#include <torch/all.h>

torch::Tensor fp8_gemm(const torch::Tensor& A, const torch::Tensor& B,
                       std::optional<c10::ScalarType> out_dtype, bool trans_B,
                       const std::optional<torch::Tensor>& A_scale_,
                       const std::optional<torch::Tensor>& B_scale_,
                       const std::optional<torch::Tensor>& bias_);

torch::Tensor fp8_gemm_w8a16(const torch::Tensor& A, const torch::Tensor& B,
                             bool trans_B,
                             const std::optional<torch::Tensor>& B_scale_,
                             const std::optional<torch::Tensor>& bias_);

torch::Tensor int4_gemm_w4a16(const torch::Tensor& A_, const torch::Tensor& B,
                              const std::optional<torch::Tensor>& bias,
                              const torch::Tensor& B_scale,
                              const torch::Tensor& B_zp, int64_t group_size,
                              bool trans_B,
                              const std::optional<torch::Tensor>& g_idx);

torch::Tensor cutlass_grouped_gemm(torch::Tensor ptr_A, torch::Tensor ptr_B,
                                   torch::Tensor ptr_D, torch::Tensor ptr_alpha,
                                   torch::Tensor ptr_beta, torch::Tensor offset,
                                   int64_t N, int64_t K, int64_t groups);

std::tuple<at::Tensor, at::Tensor> deepseek_scaling_rope(
    const at::Tensor& positions, const at::Tensor& query, const at::Tensor& key,
    const c10::optional<at::Tensor>& offsets_opt,
    const at::Tensor& cos_sin_cache, int64_t rotary_dim, bool is_neox);
