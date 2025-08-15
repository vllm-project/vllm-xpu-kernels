#pragma once

#include <torch/all.h>

void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
              double epsilon);

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        torch::Tensor& weight, double epsilon);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      std::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox);

void reshape_and_cache(torch::Tensor& key, torch::Tensor& value,
                       torch::Tensor& key_cache, torch::Tensor& value_cache,
                       torch::Tensor& slot_mapping,
                       const std::string& kv_cache_dtype,
                       torch::Tensor& k_scale, torch::Tensor& v_scale);

void reshape_and_cache_flash(torch::Tensor& key, torch::Tensor& value,
                             torch::Tensor& key_cache,
                             torch::Tensor& value_cache,
                             torch::Tensor& slot_mapping,
                             const std::string& kv_cache_dtype,
                             torch::Tensor& k_scale, torch::Tensor& v_scale);

void static_scaled_fp8_quant(torch::Tensor& out, torch::Tensor const& input,
                             torch::Tensor const& scale);

void dynamic_scaled_fp8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor& scale);

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scales,
    std::optional<at::Tensor> const& scale_ub);

torch::Tensor fp8_gemm_w8a16(const torch::Tensor& A, const torch::Tensor& B,
                             bool trans_B,
                             const std::optional<torch::Tensor>& B_scale_,
                             const std::optional<torch::Tensor>& bias_);
