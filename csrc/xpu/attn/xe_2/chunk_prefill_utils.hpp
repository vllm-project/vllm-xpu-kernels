#include "chunk_prefill.hpp"

using namespace cute;

template <typename chunk_policy, bool... Bs>
void policy_dispatch_func(
    sycl::queue& queue,
    CutlassQKType& cuQKType,
    const chunk_prefill_args_t& args) {
  policy_dispatch_impl<chunk_policy, Bs...>(queue, cuQKType, args);
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

// Dispatch by head size for non-paged or page_size >= 64 paths.
// Paged=false is passed as a bool arg so both paged and non-paged can use it.
template <bool IsPaged>
void dispatch_by_head_size_default(
    sycl::queue& queue,
    CutlassQKType& cuQKType,
    const chunk_prefill_args_t& args,
    bool is_causal,
    bool is_local,
    bool is_sink) {
  if (args.head_size <= HEAD_SIZE_LIMIT_0) {
    policy_dispatch_func<chunk_policy_head64, IsPaged>(
        queue, cuQKType, args, is_causal, is_local, is_sink);
  } else if (args.head_size <= HEAD_SIZE_LIMIT_1) {
    policy_dispatch_func<chunk_policy_head96, IsPaged>(
        queue, cuQKType, args, is_causal, is_local, is_sink);
  } else if (args.head_size <= HEAD_SIZE_LIMIT_2) {
    policy_dispatch_func<chunk_policy_head128, IsPaged>(
        queue, cuQKType, args, is_causal, is_local, is_sink);
  } else if (args.head_size <= HEAD_SIZE_LIMIT_3) {
    policy_dispatch_func<chunk_policy_head192, IsPaged>(
        queue, cuQKType, args, is_causal, is_local, is_sink);
  } else if (args.head_size <= HEAD_SIZE_LIMIT_4) {
    policy_dispatch_func<chunk_policy_head256, IsPaged>(
        queue, cuQKType, args, is_causal, is_local, is_sink);
  } else {
    TORCH_CHECK(false, "Unsupported head size for fmha");
  }
}

// Dispatch by head size for paged KV with page_size=32.
// head96/128 need a p32 policy (K-tile=32); others fall back to default.
inline void dispatch_by_head_size_p32(
    sycl::queue& queue,
    CutlassQKType& cuQKType,
    const chunk_prefill_args_t& args,
    bool is_causal,
    bool is_local,
    bool is_sink) {
  if (args.head_size <= HEAD_SIZE_LIMIT_0) {
    policy_dispatch_func<chunk_policy_head64, true>(
        queue, cuQKType, args, is_causal, is_local, is_sink);
  } else if (args.head_size <= HEAD_SIZE_LIMIT_1) {
    policy_dispatch_func<chunk_policy_head96_p32, true>(
        queue, cuQKType, args, is_causal, is_local, is_sink);
  } else if (args.head_size <= HEAD_SIZE_LIMIT_2) {
    policy_dispatch_func<chunk_policy_head128_p32, true>(
        queue, cuQKType, args, is_causal, is_local, is_sink);
  } else if (args.head_size <= HEAD_SIZE_LIMIT_3) {
    policy_dispatch_func<chunk_policy_head192, true>(
        queue, cuQKType, args, is_causal, is_local, is_sink);
  } else if (args.head_size <= HEAD_SIZE_LIMIT_4) {
    policy_dispatch_func<chunk_policy_head256, true>(
        queue, cuQKType, args, is_causal, is_local, is_sink);
  } else {
    TORCH_CHECK(false, "Unsupported head size for fmha");
  }
}

// Dispatch by head size for paged KV with page_size=16 (K-tile=16 for all).
inline void dispatch_by_head_size_p16(
    sycl::queue& queue,
    CutlassQKType& cuQKType,
    const chunk_prefill_args_t& args,
    bool is_causal,
    bool is_local,
    bool is_sink) {
  if (args.head_size <= HEAD_SIZE_LIMIT_0) {
    policy_dispatch_func<chunk_policy_head64_p16, true>(
        queue, cuQKType, args, is_causal, is_local, is_sink);
  } else if (args.head_size <= HEAD_SIZE_LIMIT_1) {
    policy_dispatch_func<chunk_policy_head96_p16, true>(
        queue, cuQKType, args, is_causal, is_local, is_sink);
  } else if (args.head_size <= HEAD_SIZE_LIMIT_2) {
    policy_dispatch_func<chunk_policy_head128_p16, true>(
        queue, cuQKType, args, is_causal, is_local, is_sink);
  } else if (args.head_size <= HEAD_SIZE_LIMIT_3) {
    policy_dispatch_func<chunk_policy_head192_p16, true>(
        queue, cuQKType, args, is_causal, is_local, is_sink);
  } else if (args.head_size <= HEAD_SIZE_LIMIT_4) {
    policy_dispatch_func<chunk_policy_head256_p16, true>(
        queue, cuQKType, args, is_causal, is_local, is_sink);
  } else {
    TORCH_CHECK(false, "Unsupported head size for fmha");
  }
}

// Top-level dispatch: select head-size dispatch table by page size.
inline void dispatch_by_page_size(
    sycl::queue& queue,
    CutlassQKType& cuQKType,
    const chunk_prefill_args_t& args,
    bool is_paged,
    bool is_causal,
    bool is_local,
    bool is_sink) {
  if (!is_paged) {
    dispatch_by_head_size_default<false>(
        queue, cuQKType, args, is_causal, is_local, is_sink);
  } else if (args.block_size < 32) {
    dispatch_by_head_size_p16(
        queue, cuQKType, args, is_causal, is_local, is_sink);
  } else if (args.block_size < 64) {
    dispatch_by_head_size_p32(
        queue, cuQKType, args, is_causal, is_local, is_sink);
  } else {
    dispatch_by_head_size_default<true>(
        queue, cuQKType, args, is_causal, is_local, is_sink);
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
