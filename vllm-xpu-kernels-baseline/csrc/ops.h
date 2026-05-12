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

// Fused RMSNorm + dynamic per-token quantization (FP8 or INT8 output).
void rms_norm_dynamic_per_token_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor& scales,
    double const epsilon,
    std::optional<torch::Tensor> scale_ub,
    std::optional<torch::Tensor> residual);

// Fused RMSNorm + per-column-block quantization (FP8 or INT8 output).
void rms_norm_per_block_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor& scales,
    double const epsilon,
    std::optional<torch::Tensor> scale_ub,
    std::optional<torch::Tensor> residual,
    int64_t group_size,
    bool is_scale_transposed);

void rms_norm_static_fp8_quant(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& scale,
    double epsilon);

void fused_add_rms_norm_static_fp8_quant(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    torch::Tensor& scale,
    double epsilon);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void silu_and_mul_quant(
    torch::Tensor& out, torch::Tensor& input, torch::Tensor& scale);

void mul_and_silu(torch::Tensor& out, torch::Tensor& input);

void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);

void fatrelu_and_mul(
    torch::Tensor& out, torch::Tensor& input, double threshold);

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

void fused_qk_norm_rope(
    torch::Tensor& qkv,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight,
    torch::Tensor& cos_sin_cache,
    bool is_neox,
    torch::Tensor& position_ids);

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

void indexer_k_quant_and_cache(
    torch::Tensor& k,             // [num_tokens, head_dim]
    torch::Tensor& kv_cache,      // [num_blocks, block_size, cache_stride]
    torch::Tensor& slot_mapping,  // [num_tokens]
    int64_t quant_block_size,     // quantization block size
    const std::string& scale_fmt);

void cp_gather_indexer_k_quant_cache(
    const torch::Tensor& kv_cache,  // [num_blocks, block_size, cache_stride]
    torch::Tensor& dst_k,           // [num_tokens, head_dim]
    torch::Tensor& dst_scale,  // [num_tokens, head_dim / quant_block_size * 4]
    const torch::Tensor& block_table,  // [batch_size, num_blocks]
    const torch::Tensor& cu_seq_lens   // [batch_size + 1]
);

void gather_and_maybe_dequant_cache(
    torch::Tensor const& src_cache,     // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::Tensor const& dst,           // [TOT_TOKENS, ENTRIES...]
    torch::Tensor const& block_table,   // [BATCH, BLOCK_INDICES]
    torch::Tensor const& cu_seq_lens,   // [BATCH+1]
    torch::Tensor const& token_to_seq,  // [MAX_TOKEN_ACROSS_CHUNKS]
    int64_t num_tokens,
    const std::string& kv_cache_dtype,
    torch::Tensor const& scale,
    std::optional<torch::Tensor> seq_starts = std::nullopt);

void static_scaled_fp8_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor const& scale,
    std::optional<std::tuple<int64_t, int64_t>> group_shape = std::nullopt);

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

void per_token_group_quant_mxfp4(
    const torch::Tensor& input,
    torch::Tensor& output_q,
    torch::Tensor& output_s,
    int64_t group_size,
    double eps);

void swigluoai_and_mul(
    torch::Tensor& out,
    torch::Tensor& input,
    double alpha = 1.702,
    double limit = 7.0);

void relu2_no_mul(torch::Tensor& out, torch::Tensor& input);

void swiglustep_and_mul(
    torch::Tensor& out, torch::Tensor& input, double limit = 7.0);

torch::Tensor get_xpu_view_from_cpu_tensor(torch::Tensor& cpu_tensor);

// Just for unittest
void convert_fp8(
    torch::Tensor& dst,
    const torch::Tensor& src,
    const double scale,
    const std::string& kv_cache_dtype);

void swap_blocks(
    torch::Tensor& src,
    torch::Tensor& dst,
    int64_t block_size_in_bytes,
    const torch::Tensor& block_mapping);

void swap_blocks_batch(
    const torch::Tensor& src_ptrs,
    const torch::Tensor& dst_ptrs,
    const torch::Tensor& sizes);

void top_k_per_row_decode(
    const torch::Tensor& logits,
    int64_t next_n,
    const torch::Tensor& seqLens,
    torch::Tensor& indices,
    int64_t numRows,
    int64_t stride0,
    int64_t stride1,
    int64_t topK);

void top_k_per_row_prefill(
    const torch::Tensor& logits,
    const torch::Tensor& rowStarts,
    const torch::Tensor& rowEnds,
    torch::Tensor& indices,
    int64_t numRows,
    int64_t stride0,
    int64_t stride1,
    int64_t topK);

void xpu_memcpy_sync(
    int64_t dst_ptr,
    int64_t src_ptr,
    int64_t n_bytes,
    int64_t kind,
    int64_t device = -1);

void merge_attn_states(
    torch::Tensor& output,
    std::optional<torch::Tensor> output_lse,
    const torch::Tensor& prefix_output,
    const torch::Tensor& prefix_lse,
    const torch::Tensor& suffix_output,
    const torch::Tensor& suffix_lse);
