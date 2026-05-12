#pragma once

#include <torch/all.h>

torch::Tensor fp8_mqa_logits(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& kv_scales,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seqlen_ks,
    const torch::Tensor& cu_seqlen_ke);

torch::Tensor fp8_paged_mqa_logits(
    const torch::Tensor& q_fp8,
    const torch::Tensor& kv_cache_fp8,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_tables,
    const c10::optional<at::Tensor>& schedule_metadata,
    int64_t max_model_len);
