#include "chunk_prefill.hpp"

using namespace cute;

// Only Paged is dispatched at compile time; Causal, Local, Sink are disabled.
template <typename chunk_policy>
void policy_dispatch_func(
    sycl::queue& queue,
    CutlassQKType& cuQKType,
    const chunk_prefill_args_t& args,
    bool is_paged) {
  if (is_paged) {
    policy_dispatch_impl<chunk_policy, true>(queue, cuQKType, args);
  } else {
    policy_dispatch_impl<chunk_policy, false>(queue, cuQKType, args);
  }
}

void cutlass_chunk_prefill_impl(
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
    bool is_sink);
