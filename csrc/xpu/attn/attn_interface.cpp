#include "utils.h"
#include "attn_interface.h"

#ifdef VLLM_FA2_ATTN_ENABLED
#ifdef VLLM_XPU_ENABLE_XE2
  #include "csrc/xpu/attn/xe_2/fmha_xe2.h"
  #include "csrc/xpu/attn/xe_2/paged_decode_xe2.h"
#endif
#endif

#ifdef VLLM_SPARSE_MLA_ENABLED

namespace vllm::xpu::attn::mla {

std::vector<at::Tensor> flash_mla_sparse_attn_prefill_impl(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    float sm_scale,
    int d_v,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& output,
    bool return_softmax_lse);

std::tuple<at::Tensor, std::optional<at::Tensor>>
flash_mla_sparse_attn_decode_impl(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& out,
    std::optional<at::Tensor>& tile_scheduler_metadata,
    std::optional<at::Tensor>& num_splits,
    const std::optional<at::Tensor>& extra_kv,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_topk_length,
    int d_v,
    float sm_scale,
    bool return_softmax_lse);

}  // namespace vllm::xpu::attn::mla

#endif

#ifdef VLLM_FA2_ATTN_ENABLED

void cutlass_chunk_prefill_interface(
    sycl::queue& queue,
    const at::Tensor& query,      // [seq_q, heads, head_size]
    const at::Tensor& key_cache,  // [num_block, block_size, heads, head_size]
    const at::Tensor& value_cache,
    at::Tensor& out,
    const at::Tensor& block_table,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    std::optional<const at::Tensor>& k_scale,
    std::optional<const at::Tensor>& v_scale,
    double sm_scale,
    std::optional<const at::Tensor>& sm_sink_,
    int window_size_left,
    int window_size_right,
    bool is_varlen,
    bool is_paged,
    bool is_causal,
    bool is_local,
    bool is_sink,
    std::optional<at::Tensor>& softmax_lse,
    std::optional<const at::Tensor>& is_prefill) {
  if (vllm::xpu::is_xe2_arch() || vllm::xpu::is_xe3_arch()) {
#ifdef VLLM_XPU_ENABLE_XE2
    // Use XE2 cutlass kernel (also used as WA for XE3/XE3P)
    cutlass_chunk_prefill_xe2(
        queue,
        query,
        key_cache,
        value_cache,
        out,
        block_table,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        k_scale,
        v_scale,
        sm_scale,
        sm_sink_,
        window_size_left,
        window_size_right,
        is_varlen,
        is_paged,
        is_causal,
        is_local,
        is_sink,
        softmax_lse,
        is_prefill);
#else
    TORCH_CHECK(false, "XE2 cutlass kernel is not enabled in this build.");
#endif
  } else {
    TORCH_CHECK(false, "Only XE2/XE3 cutlass kernel is supported currently.");
  }
}

void cutlass_paged_decode_interface(
    sycl::queue& queue,
    const at::Tensor& query,      // [seq_q, heads, head_size]
    const at::Tensor& key_cache,  // [num_block, block_size, heads, head_size]
    const at::Tensor& value_cache,
    at::Tensor& out,
    at::Tensor&
        temp_out,  // [batch, num_head_q, seq_q, head_size, num_kv_splits]
    at::Tensor& exp_sums,    // [batch, num_head_q, seq_q, num_kv_splits]
    at::Tensor& max_logits,  // [batch, num_head_q, seq_q, num_kv_splits]
    const at::Tensor& block_table,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    std::optional<const at::Tensor>& k_scale,
    std::optional<const at::Tensor>& v_scale,
    double sm_scale,
    std::optional<const at::Tensor>& sm_sink_,
    int window_size_left,
    int window_size_right,
    bool is_varlen,
    bool is_paged,
    bool is_causal,
    bool is_local,
    bool is_sink,
    int num_kv_splits,
    std::optional<const at::Tensor>& is_prefill,
    std::optional<at::Tensor>& splits_per_seq,
    std::optional<at::Tensor>& work_list) {
  if (vllm::xpu::is_xe2_arch() || vllm::xpu::is_xe3_arch()) {
#ifdef VLLM_XPU_ENABLE_XE2
    // Use XE2 cutlass kernel (also used as WA for XE3/XE3P)
    cutlass_paged_decode_xe2(
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
        k_scale,
        v_scale,
        sm_scale,
        sm_sink_,
        window_size_left,
        window_size_right,
        is_varlen,
        is_paged,
        is_causal,
        is_local,
        is_sink,
        num_kv_splits,
        is_prefill,
        splits_per_seq,
        work_list);
#else
    TORCH_CHECK(false, "XE2 cutlass kernel is not enabled in this build.");
#endif
  } else {
    TORCH_CHECK(false, "Only XE2/XE3 cutlass kernel is supported currently.");
  }
}

#endif

#ifdef VLLM_SPARSE_MLA_ENABLED
std::vector<at::Tensor> flash_mla_sparse_attn_prefill_interface(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    float sm_scale,
    int d_v,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& output,
    bool return_softmax_lse) {
  return vllm::xpu::attn::mla::flash_mla_sparse_attn_prefill_impl(
      q,
      kv,
      indices,
      sm_scale,
      d_v,
      attn_sink,
      topk_length,
      output,
      return_softmax_lse);
}

std::tuple<at::Tensor, std::optional<at::Tensor>>
flash_mla_sparse_attn_decode_interface(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& out,
    std::optional<at::Tensor>& tile_scheduler_metadata,
    std::optional<at::Tensor>& num_splits,
    const std::optional<at::Tensor>& extra_kv,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_topk_length,
    int d_v,
    float sm_scale,
    bool return_softmax_lse) {
  return vllm::xpu::attn::mla::flash_mla_sparse_attn_decode_impl(
      q,
      kv,
      indices,
      topk_length,
      attn_sink,
      out,
      tile_scheduler_metadata,
      num_splits,
      extra_kv,
      extra_indices,
      extra_topk_length,
      d_v,
      sm_scale,
      return_softmax_lse);
}

#endif
