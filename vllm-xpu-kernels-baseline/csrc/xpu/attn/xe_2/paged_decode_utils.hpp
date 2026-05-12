#include "paged_decode.hpp"

using namespace cute;

// Runtime dispatcher helper
template <typename decode_policy, bool... Bs>
void decode_policy_dispatch_func(
    sycl::queue& queue,
    CutlassQKType& cuQKType,
    const paged_decode_args_t& args) {
  decode_policy_dispatch_impl<decode_policy, Bs...>(queue, cuQKType, args);
}

template <typename decode_policy, bool... Bs, typename... Ts>
void decode_policy_dispatch_func(
    sycl::queue& queue,
    CutlassQKType& cuQKType,
    const paged_decode_args_t& args,
    bool b,
    Ts... ts) {
  if (b) {
    decode_policy_dispatch_func<decode_policy, Bs..., true>(
        queue, cuQKType, args, ts...);
  } else {
    decode_policy_dispatch_func<decode_policy, Bs..., false>(
        queue, cuQKType, args, ts...);
  }
}

template <typename QGroup, typename PageSize>
inline void dispatch_by_head_size(
    const int head_case,
    sycl::queue& queue,
    CutlassQKType& cuQKType,
    const paged_decode_args_t& args) {
  switch (head_case) {
    case 0:
      decode_policy_dispatch_func<
          decode_policy_qpacked_head<QGroup, _64, PageSize>>(
          queue, cuQKType, args, args.is_causal, args.is_local, args.is_sink);
      break;
    case 1:
      decode_policy_dispatch_func<
          decode_policy_qpacked_head<QGroup, _96, PageSize>>(
          queue, cuQKType, args, args.is_causal, args.is_local, args.is_sink);
      break;
    case 2:
      decode_policy_dispatch_func<
          decode_policy_qpacked_head<QGroup, _128, PageSize>>(
          queue, cuQKType, args, args.is_causal, args.is_local, args.is_sink);
      break;
    case 3:
      decode_policy_dispatch_func<
          decode_policy_qpacked_head<QGroup, _192, PageSize>>(
          queue, cuQKType, args, args.is_causal, args.is_local, args.is_sink);
      break;
    case 4:
      decode_policy_dispatch_func<
          decode_policy_qpacked_head<QGroup, _256, PageSize>>(
          queue, cuQKType, args, args.is_causal, args.is_local, args.is_sink);
      break;
    case 5:
      decode_policy_dispatch_func<
          decode_policy_qpacked_head<QGroup, _512, PageSize>>(
          queue, cuQKType, args, args.is_causal, args.is_local, args.is_sink);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size for fmha");
  }
}

template <typename QGroup>
inline void dispatch_by_page_size(
    const int page_size,
    const int head_case,
    sycl::queue& queue,
    CutlassQKType& cuQKType,
    const paged_decode_args_t& args) {
  // The mainloop iterates page_size / TileShapeQK[1] sub-tiles per page using
  // the page-table indirection, so any page_size that is a positive multiple
  // of the policy's kv_tile is supported.
  //
  // NOTE: We intentionally route page_size that is a multiple of 128 through
  // the kv_tile=_64 policy (instead of _128) because the _128 decode policy
  // uses SubgroupLayoutQK<_1,_8,_1> (8-SG K split, ReduceK=8) which exercises
  // a buggy cross-SG SLM reduction path in
  // chunk_prefill_epilogue.hpp::reduce_A. The bug only manifests in real
  // autoregressive serving (gpt-oss-20b gsm8k strict-match drops ~12pp) and
  // stems from the degenerate SGTileShapeO=(1,32) produced when ReduceK=8. The
  // _64 policy uses ReduceK=4 with the well-tested SGTileShapeO=(2,32) layout.
  // The mainloop iterates page_size/64 sub-tiles per page (2 for page=128, 4
  // for page=256, ...), so correctness is preserved at a small parallelism
  // cost. Restore the _128 routing once the underlying ReduceK=8 bug is fixed
  // upstream.
  if (page_size == 16) {
    dispatch_by_head_size<QGroup, _16>(head_case, queue, cuQKType, args);
  } else if (page_size == 32) {
    dispatch_by_head_size<QGroup, _32>(head_case, queue, cuQKType, args);
  } else if (page_size > 0 && (page_size % 64) == 0) {
    dispatch_by_head_size<QGroup, _64>(head_case, queue, cuQKType, args);
  } else {
    TORCH_CHECK(
        false,
        "Unsupported page size for fmha: ",
        page_size,
        " (supported: 16, 32, or any positive multiple of 64)");
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
    int num_kv_splits);
