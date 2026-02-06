#include "paged_decode.hpp"

using namespace cute;

// Runtime dispatcher helper
template <typename decode_policy, bool... Bs>
void decode_policy_dispatch_func(
    sycl::queue& queue, CutlassDType cuType, const paged_decode_args_t& args) {
  decode_policy_dispatch_impl<decode_policy, Bs...>(queue, cuType, args);
}

template <typename decode_policy, bool... Bs, typename... Ts>
void decode_policy_dispatch_func(
    sycl::queue& queue,
    CutlassDType cuType,
    const paged_decode_args_t& args,
    bool b,
    Ts... ts) {
  if (b) {
    decode_policy_dispatch_func<decode_policy, Bs..., true>(
        queue, cuType, args, ts...);
  } else {
    decode_policy_dispatch_func<decode_policy, Bs..., false>(
        queue, cuType, args, ts...);
  }
}

template <class QGroup>
inline void dispatch_by_head_size(
    const int head_case,
    sycl::queue& queue,
    CutlassDType cuType,
    const paged_decode_args_t& args) {
  switch (head_case) {
    case 0:
      decode_policy_dispatch_func<decode_policy_qpacked_head<QGroup, _64>>(
          queue, cuType, args, args.is_causal, args.is_local, args.is_sink);
      break;
    case 1:
      decode_policy_dispatch_func<decode_policy_qpacked_head<QGroup, _96>>(
          queue, cuType, args, args.is_causal, args.is_local, args.is_sink);
      break;
    case 2:
      decode_policy_dispatch_func<decode_policy_qpacked_head<QGroup, _128>>(
          queue, cuType, args, args.is_causal, args.is_local, args.is_sink);
      break;
    case 3:
      decode_policy_dispatch_func<decode_policy_qpacked_head<QGroup, _192>>(
          queue, cuType, args, args.is_causal, args.is_local, args.is_sink);
      break;
    case 4:
      decode_policy_dispatch_func<decode_policy_qpacked_head<QGroup, _256>>(
          queue, cuType, args, args.is_causal, args.is_local, args.is_sink);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size for fmha");
  }
}

void cutlass_paged_decode_impl(
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
    double sm_scale,
    std::optional<const at::Tensor>& sm_sink_,
    int window_size_left,
    int window_size_right,
    bool is_varlen,
    bool is_paged,
    bool is_causal,
    bool is_local,
    bool is_sink,
    int num_kv_splits);