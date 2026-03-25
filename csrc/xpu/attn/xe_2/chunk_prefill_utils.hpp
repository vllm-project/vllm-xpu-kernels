#include "chunk_prefill.hpp"

using namespace cute;

// ============================================================================
// Optimized dispatch to reduce compile-time recursion depth
// Instead of recursive template instantiation, we use a flattened approach
// ============================================================================

// Base case - forward to the actual implementation
template <typename chunk_policy, bool Paged, bool Causal, bool Local, bool Sink>
inline void policy_dispatch_exec(
    sycl::queue& queue,
    CutlassQKType& cuQKType,
    const chunk_prefill_args_t& args) {
  policy_dispatch_impl<chunk_policy, Paged, Causal, Local, Sink>(
      queue, cuQKType, args);
}

// Flattened dispatch: directly compute all 4 booleans without recursion
// This reduces template instantiation overhead significantly
template <typename chunk_policy>
inline void policy_dispatch_func(
    sycl::queue& queue,
    CutlassQKType& cuQKType,
    const chunk_prefill_args_t& args,
    bool is_paged,
    bool is_causal,
    bool is_local,
    bool is_sink) {
  // Use a switch-like approach to minimize branching and template
  // instantiations 16 combinations total (2^4), organized by binary encoding
  if (!is_paged) {
    if (!is_causal) {
      if (!is_local) {
        if (!is_sink) {
          policy_dispatch_exec<chunk_policy, false, false, false, false>(
              queue, cuQKType, args);
        } else {
          policy_dispatch_exec<chunk_policy, false, false, false, true>(
              queue, cuQKType, args);
        }
      } else {
        if (!is_sink) {
          policy_dispatch_exec<chunk_policy, false, false, true, false>(
              queue, cuQKType, args);
        } else {
          policy_dispatch_exec<chunk_policy, false, false, true, true>(
              queue, cuQKType, args);
        }
      }
    } else {
      if (!is_local) {
        if (!is_sink) {
          policy_dispatch_exec<chunk_policy, false, true, false, false>(
              queue, cuQKType, args);
        } else {
          policy_dispatch_exec<chunk_policy, false, true, false, true>(
              queue, cuQKType, args);
        }
      } else {
        if (!is_sink) {
          policy_dispatch_exec<chunk_policy, false, true, true, false>(
              queue, cuQKType, args);
        } else {
          policy_dispatch_exec<chunk_policy, false, true, true, true>(
              queue, cuQKType, args);
        }
      }
    }
  } else {
    if (!is_causal) {
      if (!is_local) {
        if (!is_sink) {
          policy_dispatch_exec<chunk_policy, true, false, false, false>(
              queue, cuQKType, args);
        } else {
          policy_dispatch_exec<chunk_policy, true, false, false, true>(
              queue, cuQKType, args);
        }
      } else {
        if (!is_sink) {
          policy_dispatch_exec<chunk_policy, true, false, true, false>(
              queue, cuQKType, args);
        } else {
          policy_dispatch_exec<chunk_policy, true, false, true, true>(
              queue, cuQKType, args);
        }
      }
    } else {
      if (!is_local) {
        if (!is_sink) {
          policy_dispatch_exec<chunk_policy, true, true, false, false>(
              queue, cuQKType, args);
        } else {
          policy_dispatch_exec<chunk_policy, true, true, false, true>(
              queue, cuQKType, args);
        }
      } else {
        if (!is_sink) {
          policy_dispatch_exec<chunk_policy, true, true, true, false>(
              queue, cuQKType, args);
        } else {
          policy_dispatch_exec<chunk_policy, true, true, true, true>(
              queue, cuQKType, args);
        }
      }
    }
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
