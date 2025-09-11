#pragma once

#include <torch/all.h>

torch::Tensor fp8_gemm_w8a16(const torch::Tensor& A, const torch::Tensor& B,
                             bool trans_B,
                             const std::optional<torch::Tensor>& B_scale_,
                             const std::optional<torch::Tensor>& bias_);

torch::Tensor cutlass_grouped_gemm(torch::Tensor ptr_A,
                                   torch::Tensor ptr_B,
                                   torch::Tensor ptr_D,
                                   torch::Tensor ptr_alpha,
                                   torch::Tensor ptr_beta,
                                   torch::Tensor offset, 
                                   int64_t N, 
                                   int64_t K, 
                                   int64_t groups);
