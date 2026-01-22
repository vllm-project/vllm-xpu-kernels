#include "chunk_prefill.hpp"

using namespace cute;

template <typename chunk_policy, bool... Bs>
void policy_dispatch_func(
    sycl::queue& queue, CutlassType cuType, const chunk_prefill_args_t& args) {
  policy_dispatch_impl<chunk_policy, Bs...>(queue, cuType, args);
}

template <typename chunk_policy, bool... Bs, typename... Ts>
void policy_dispatch_func(
    sycl::queue& queue,
    CutlassType cuType,
    const chunk_prefill_args_t& args,
    bool b,
    Ts... ts) {
  if (b) {
    policy_dispatch_func<chunk_policy, Bs..., true>(queue, cuType, args, ts...);
  } else {
    policy_dispatch_func<chunk_policy, Bs..., false>(
        queue, cuType, args, ts...);
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
    double sm_scale,
    std::optional<const at::Tensor>& sm_sink_,
    int window_size_left,
    int window_size_right,
    bool is_varlen,
    bool is_paged,
    bool is_causal,
    bool is_local,
    bool is_sink);
