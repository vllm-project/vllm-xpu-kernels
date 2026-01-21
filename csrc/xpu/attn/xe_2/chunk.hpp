
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "collective/chunk_prefill_scheduler.hpp"
#include "collective/chunk_prefill_epilogue.hpp"
#include "kernel/chunk_prefill_kernel.hpp"
#include "chunk_prefill.hpp"
#include "fmha_utils.hpp"

using namespace cute;

template <typename chunk_policy, bool... Bs>
void policy_dispatch_impl(
    sycl::queue& queue, CutlassType cuType, const chunk_prefill_args_t& args) {
  policy_dispatch<chunk_policy, Bs...>(queue, cuType, args);
}

template <typename chunk_policy, bool... Bs, typename... Ts>
void policy_dispatch_impl(
    sycl::queue& queue,
    CutlassType cuType,
    const chunk_prefill_args_t& args,
    bool b,
    Ts... ts) {
  if (b) {
    policy_dispatch_impl<chunk_policy, Bs..., true>(queue, cuType, args, ts...);
  } else {
    policy_dispatch_impl<chunk_policy, Bs..., false>(
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
    bool is_sink) {
  // general params
  int batch_size, num_heads_q, num_heads_kv, head_size;
  // additional params
  int total_seqlen_q, total_seqlen_k;
  int num_blocks, block_size, max_blocks_per_seq;
  if (is_varlen) {
    // query: [total_seq, num_heads, head_size]
    batch_size = cu_seqlens_q.numel() - 1;
    num_heads_q = query.size(1);
    num_heads_kv = key_cache.size(1);
    head_size = query.size(2);
    total_seqlen_q = query.size(0);
    total_seqlen_k = key_cache.size(0);
  } else {
    // query: [batch, num_heads, seq, head_size]
    batch_size = query.size(0);
    num_heads_q = query.size(1);
    num_heads_kv = key_cache.size(1);
    head_size = query.size(3);
    max_seqlen_q = query.size(2);
    max_seqlen_k = key_cache.size(2);
  }
  if (is_paged) {
    num_blocks = key_cache.size(0);
    block_size = key_cache.size(1);
    num_heads_kv = key_cache.size(2);
    max_blocks_per_seq = block_table.size(1);
    total_seqlen_k = num_blocks * block_size;
  }

  if (is_local) {
    window_size_left = window_size_left == -1 ? max_seqlen_k : window_size_left;
    window_size_right =
        window_size_right == -1 ? max_seqlen_k : window_size_right;
    if (is_causal) {
      window_size_right = 0;
      is_causal = false;
    }
  }

  chunk_prefill_args_t args = {
      query.data_ptr(),
      key_cache.data_ptr(),
      value_cache.data_ptr(),
      out.data_ptr(),
      is_paged ? block_table.data_ptr() : nullptr,
      cu_seqlens_q.data_ptr(),
      cu_seqlens_k.data_ptr(),
      max_seqlen_q,
      max_seqlen_k,
      total_seqlen_q,
      total_seqlen_k,
      static_cast<float>(sm_scale),
      is_sink ? sm_sink_.value().data_ptr() : nullptr,
      batch_size,
      num_heads_q,
      num_heads_kv,
      head_size,
      max_blocks_per_seq,
      block_size,
      window_size_left,
      window_size_right,
      is_varlen,  // varlen
      is_paged,   // paged
      is_causal,
      is_local,
      is_sink};

  CutlassType cuType = aten_to_Cutlass_dtype(query);

  static constexpr int max_head_size = 256;
  TORCH_CHECK(
      head_size <= max_head_size,
      "FMHA forward only supports head dimension at most " +
          std::to_string(max_head_size));

  if (args.head_size <= HEAD_SIZE_LIMIT_0) {
    // policy_dispatch<chunk_policy_head64>(queue, cuType, args);
    policy_dispatch_impl<chunk_policy_head64>(
        queue, cuType, args, is_paged, is_causal, is_local, is_sink);
  } else if (args.head_size <= HEAD_SIZE_LIMIT_1) {
    policy_dispatch_impl<chunk_policy_head96>(
        queue, cuType, args, is_paged, is_causal, is_local, is_sink);
  } else if (args.head_size <= HEAD_SIZE_LIMIT_2) {
    policy_dispatch_impl<chunk_policy_head128>(
        queue, cuType, args, is_paged, is_causal, is_local, is_sink);
  } else if (args.head_size <= HEAD_SIZE_LIMIT_3) {
    policy_dispatch_impl<chunk_policy_head192>(
        queue, cuType, args, is_paged, is_causal, is_local, is_sink);
  } else if (args.head_size <= HEAD_SIZE_LIMIT_4) {
    policy_dispatch_impl<chunk_policy_head256>(
        queue, cuType, args, is_paged, is_causal, is_local, is_sink);
  } else {
    TORCH_CHECK(false, "Unsupported head size for fmha");
  }
}
