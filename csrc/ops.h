#pragma once

#include <torch/all.h>

torch::Tensor weak_ref_tensor(torch::Tensor& tensor);

void rms_norm(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    double epsilon);

void fused_add_rms_norm(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    double epsilon);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void mul_and_silu(torch::Tensor& out, torch::Tensor& input);

void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_fast(torch::Tensor& out, torch::Tensor& input);

void gelu_new(torch::Tensor& out, torch::Tensor& input);

void gelu_quick(torch::Tensor& out, torch::Tensor& input);

void rotary_embedding(
    torch::Tensor& positions,
    torch::Tensor& query,
    std::optional<torch::Tensor> key,
    int64_t head_size,
    torch::Tensor& cos_sin_cache,
    bool is_neox);

void reshape_and_cache(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale);

void reshape_and_cache_flash(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale);

void concat_and_cache_mla(
    torch::Tensor& kv_c,
    torch::Tensor& k_pe,
    torch::Tensor& kv_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor& scale);

void gather_cache(
    torch::Tensor const& src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::Tensor const& dst,          // [TOT_TOKENS, ENTRIES...]
    torch::Tensor const& block_table,  // [BATCH, BLOCK_INDICES]
    torch::Tensor const& cu_seq_lens,  // [BATCH+1]
    int64_t batch_size,
    std::optional<torch::Tensor> seq_starts = std::nullopt);

void static_scaled_fp8_quant(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor const& scale);

void dynamic_scaled_fp8_quant(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scale);

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor& scales,
    std::optional<at::Tensor> const& scale_ub);

void per_token_group_quant_fp8(
    const torch::Tensor& input,
    torch::Tensor& output_q,
    torch::Tensor& output_s,
    int64_t group_size,
    double eps,
    double fp8_min,
    double fp8_max,
    bool scale_ue8m0);

void swigluoai_and_mul(
    torch::Tensor& out,
    torch::Tensor& input,
    double alpha = 1.702,
    double limit = 7.0);

torch::Tensor get_xpu_view_from_cpu_tensor(torch::Tensor& cpu_tensor);
