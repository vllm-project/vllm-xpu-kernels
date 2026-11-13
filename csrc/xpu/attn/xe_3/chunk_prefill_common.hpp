#pragma once

// Chunk-prefill declarations kept free of any kernel headers: the argument
// struct, the causal tile-shape selection and the dispatch entry point.

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>

#include "csrc/xpu/attn/fmha_utils.hpp"

// See chunk_prefill.hpp for why everything stays in vllm::xpu::xe3.
namespace vllm::xpu::xe3 {
using namespace cute;

struct chunk_prefill_args_t {
  void* query;
  void* key;
  void* value;
  void* out;
  void* block_table;
  void* cu_seqlens_q;
  void* cu_seqlens_k;
  int max_queries;
  int max_keys;
  int total_seqlen_q;
  int total_seqlen_k;
  void* q_scale;
  void* k_scale;
  void* v_scale;
  float sm_scale;
  void* sm_sink;
  int batch_size;
  int num_heads_q;
  int num_heads_k;
  int head_size;
  int max_blocks_per_seq;
  int block_size;
  int window_size_left = -1;
  int window_size_right = -1;
  bool is_varlen = false;
  bool is_paged = false;
  bool is_causal = false;
  bool is_local = false;
  bool is_sink = false;
  // softmax_lse output (nullptr when not requested)
  float* softmax_lse = nullptr;
  int lse_stride = 0;  // stride along head dim (= total_seqlen_q)
  // Q/O strides in CUTLASS order: (seq, head_size=1, heads, batch)
  int q_stride_seq = 0;
  int q_stride_heads = 0;
  int q_stride_batch = 0;
  // K strides in CUTLASS order: (seq, head_size=1, heads, batch)
  int k_stride_seq = 0;
  int k_stride_heads = 0;
  int k_stride_batch = 0;
  // V strides in CUTLASS order: (head_size=1, seq, heads, batch)
  int v_stride_seq = 0;
  int v_stride_heads = 0;
  int v_stride_batch = 0;
  // O strides in CUTLASS order: (seq, head_size=1, heads, batch)
  int o_stride_seq = 0;
  int o_stride_heads = 0;
  int o_stride_batch = 0;
  // per-batch mask: true = prefill, false = decode; nullptr = process all
  void* is_prefill = nullptr;
  int page_stride_elements = 0;
};

template <typename Policy, bool Causal, typename = void>
struct chunk_policy_dispatch_shapes {
  using ShapeQK = typename Policy::ShapeQK;
  using ShapePV = typename Policy::ShapePV;
  using ShapeOut = typename Policy::ShapeOut;
  using SubgroupLayoutQK = typename Policy::SubgroupLayoutQK;
};

template <typename Policy>
struct chunk_policy_dispatch_shapes<
    Policy,
    true,
    cute::void_t<typename Policy::ShapeQK_Causal>> {
  using ShapeQK = typename Policy::ShapeQK_Causal;
  using ShapePV = typename Policy::ShapePV_Causal;
  using ShapeOut = typename Policy::ShapeOut_Causal;
  using SubgroupLayoutQK = typename Policy::SubgroupLayoutQK_Causal;
};

// bf16/fp16 Q (including fp8 KV-cache); defined in chunk_prefill.hpp.
template <
    typename chunk_policy,
    bool Paged,
    bool Causal,
    bool Local,
    bool Sink,
    bool SoftmaxLSE>
__attribute__((visibility("hidden"))) void policy_dispatch_impl(
    sycl::queue& queue,
    CutlassQKType& cuQKType,
    const chunk_prefill_args_t& args);

}  // namespace vllm::xpu::xe3
