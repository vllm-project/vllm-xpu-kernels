#pragma once

#include <torch/all.h>

torch::Tensor fp8_gemm_w8a16(const torch::Tensor& A, const torch::Tensor& B,
                             bool trans_B,
                             const std::optional<torch::Tensor>& B_scale_,
                             const std::optional<torch::Tensor>& bias_);

torch::Tensor cutlass_grouped_gemm(torch::Tensor input, torch::Tensor weight, torch::Tensor res, torch::Tensor offset, int64_t hidden_size, int64_t intermediate_size, int64_t num_of_expert);
