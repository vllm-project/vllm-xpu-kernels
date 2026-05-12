#include "chunk_prefill.hpp"

using namespace cute;

template <typename chunk_policy, bool... Bs>
void policy_dispatch_func(
    sycl::queue& queue,
    CutlassQKType& cuQKType,
    const chunk_prefill_args_t& args) {
  // Pack is expected in order: (Paged, Causal, Local, Sink, SoftmaxLSE).
  // SoftmaxLSE=true is only supported when Paged=false, Local=false,
  // and Sink=false; other combos are not instantiated (no TUs generated),
  // so statically skip the call for those to avoid implicit instantiation
  // and guarantee the runtime TORCH_CHECK in fmha_xe2.cpp is the only
  // path that can surface a bad request.
  constexpr bool flags[] = {Bs...};
  static_assert(
      sizeof...(Bs) == 5, "policy_dispatch_func expects 5 bool parameters");
  constexpr bool Paged = flags[0];
  constexpr bool Local = flags[2];
  constexpr bool Sink = flags[3];
  constexpr bool SoftmaxLSE = flags[4];
  if constexpr (SoftmaxLSE && (Paged || Local || Sink)) {
    TORCH_CHECK(
        false,
        "Unreachable: softmax_lse is only supported when is_paged=false, "
        "is_local=false, is_sink=false");
  } else {
    policy_dispatch_impl<chunk_policy, Bs...>(queue, cuQKType, args);
  }
}

template <typename chunk_policy, bool... Bs, typename... Ts>
void policy_dispatch_func(
    sycl::queue& queue,
    CutlassQKType& cuQKType,
    const chunk_prefill_args_t& args,
    bool b,
    Ts... ts) {
  if (b) {
    policy_dispatch_func<chunk_policy, Bs..., true>(
        queue, cuQKType, args, ts...);
  } else {
    policy_dispatch_func<chunk_policy, Bs..., false>(
        queue, cuQKType, args, ts...);
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
    bool is_sink,
    std::optional<at::Tensor>& softmax_lse);
