#pragma once

#include <torch/all.h>

torch::Tensor fp8_gemm_w8a16(const torch::Tensor& A, const torch::Tensor& B,
                             bool trans_B,
                             const std::optional<torch::Tensor>& B_scale_,
                             const std::optional<torch::Tensor>& bias_);

torch::Tensor cutlass_grouped_gemm(torch::Tensor ptr_A, torch::Tensor ptr_B,
                                   torch::Tensor ptr_D, torch::Tensor tokens_per_expert,
                                   int64_t N, int64_t K, int64_t groups);

std::tuple<at::Tensor, at::Tensor> deepseek_scaling_rope(
    const at::Tensor& positions, const at::Tensor& query, const at::Tensor& key,
    const c10::optional<at::Tensor>& offsets_opt,
    const at::Tensor& cos_sin_cache, int64_t rotary_dim, bool is_neox);
