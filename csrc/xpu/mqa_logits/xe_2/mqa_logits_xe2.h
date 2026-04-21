#pragma once

#include <torch/all.h>

torch::Tensor fp8_mqa_logits_xe2(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& kv_scales,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seqlen_ks,
    const torch::Tensor& cu_seqlen_ke,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t seq_len_kv);

torch::Tensor fp8_paged_mqa_logits_xe2(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& kv_scales,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_tables,
    int64_t batch_size,
    int64_t next_n,
    int64_t heads,
    int64_t index_dim,
    int64_t num_blocks,
    int64_t block_size,
    int64_t max_blocks,
    int64_t max_model_len);
