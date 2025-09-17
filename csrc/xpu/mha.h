#pragma once

void cutlass_chunk_prefill_impl(
    at::Tensor& query,      // [seq_q, heads, head_size]
    at::Tensor& key_cache,  // [num_block, block_size, heads, head_size]
    at::Tensor& value_cache, at::Tensor& out, at::Tensor& block_table,
    at::Tensor& cu_seqlens_q, at::Tensor& cu_seqlens_k, int max_seqlen_q,
    int max_seqlen_k, double sm_scale, bool is_causal);