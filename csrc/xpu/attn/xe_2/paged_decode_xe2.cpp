#include "chunk_prefill_xe2.h"
#include "chunk_prefill.hpp"

void cutlass_paged_decode_xe2(
    sycl::queue& queue,
    const at::Tensor& query,      // [seq_q, heads, head_size]
    const at::Tensor& key_cache,  // [num_block, block_size, heads, head_size]
    const at::Tensor& value_cache,
    at::Tensor& out,
    at::Tensor& temp_out,     // [batch, num_head_q, seq_q, head_size, num_kv_splits]
    at::Tensor& exp_sums,     // [batch, num_head_q, seq_q, num_kv_splits]
    at::Tensor& max_logits,   // [batch, num_head_q, seq_q, num_kv_splits]
    const at::Tensor& block_table,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    double sm_scale,
    std::optional<const at::Tensor>& sm_sink_,
    int window_size_left,
    int window_size_right,
    bool is_varlen,
    bool is_paged,
    bool is_causal,
    bool is_local,
    bool is_sink,
    int num_kv_splits) {
  cutlass_chunk_prefill_impl(
      queue,
      query,
      key_cache,
      value_cache,
      out,
      temp_out,
      exp_sums,
      max_logits,
      block_table,
      cu_seqlens_q,
      cu_seqlens_k,
      max_seqlen_q,
      max_seqlen_k,
      sm_scale,
      sm_sink_,
      window_size_left,
      window_size_right,
      is_varlen,
      is_paged,
      is_causal,
      is_local,
      is_sink);
}
