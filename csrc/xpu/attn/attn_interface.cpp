#include "csrc/utils.h"
#include "attn_interface.h"

#ifdef VLLM_XPU_ENABLE_XE2
  #include "csrc/xpu/attn/xe_2/fmha_xe2.h"
  #include "csrc/xpu/attn/xe_2/paged_decode_xe2.h"
#endif
#ifdef VLLM_XPU_ENABLE_XE3P
  #include "csrc/xpu/attn/xe_3/fmha_xe3.h"
  #include "csrc/xpu/attn/xe_3/paged_decode_xe3.h"
#endif

#ifdef VLLM_XPU_ENABLE_XE2
namespace {

// Short queries underfill chunk-prefill tiles, while query heads in the
// same GQA group issue separate loads for their shared KV head.
//
// Pack G = num_heads_q / num_heads_kv query heads into the row dimension:
// packed_row = q_pos * G + head_in_group
// The kernel then sees num_heads_q == num_heads_kv, allowing heads within
// each packed tile to share KV loads. Causal/local masks recover the query
// position as packed_row / G.
//
// Packing reduces the number of head tiles but may add Q tiles, each of
// which still loads KV separately. So it is only enabled when the estimated
// total tile count decreases.

// Mirror TileM from the Xe2 chunk-prefill policies (chunk_policy_head* in
// xe_2/fmha_utils.hpp). This value affects only the packing heuristic, not
// kernel correctness.
// TODO: Read TileM from the selected policy's ShapeQK.
constexpr int kPackGqaSmallTileHeadSize = 96;
constexpr int kPackGqaSmallTileM = 128;
constexpr int kPackGqaLargeTileM = 256;

// Default: enabled. Read on every call, so it can be flipped between
// launches. Set VLLM_XPU_CHUNK_PREFILL_PACK_GQA=0 (or false/FALSE) to always
// take the unpacked path.
bool pack_gqa_enabled() {
  auto env_val = vllm::xpu::getEnv("VLLM_XPU_CHUNK_PREFILL_PACK_GQA");
  if (env_val.has_value()) {
    return env_val.value() != "0" && env_val.value() != "false" &&
           env_val.value() != "FALSE";
  }
  return true;
}

// Returns the packing factor, or 1 when packing does not apply.
int chunk_prefill_pack_gqa_factor(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    at::Tensor& out,
    int max_seqlen_q,
    bool is_varlen,
    bool is_paged,
    bool is_sink,
    bool has_lse,
    bool has_decode_mask) {
  // is_sink indexes a per-query-head tensor and softmax_lse is laid out over
  // query heads; both would need their own unpacking, so leave them alone.
  if (!pack_gqa_enabled() || !is_varlen || !is_paged || is_sink || has_lse) {
    return 1;
  }
  // With a decode mask the kernel skips those batches, so their rows in the
  // packed output stay uninitialized and the unpack still copies them into
  // `out`. That is harmless for a distinct buffer, since paged decode
  // overwrites them on the same in-order queue, but when `out` aliases
  // `query` it would clobber the rows that same decode launch reads.
  if (has_decode_mask && out.is_alias_of(query)) return 1;
  if (max_seqlen_q <= 1) return 1;
  if (query.dim() != 3 || out.dim() != 3) return 1;
  // The unpack writes `out` through a permuted view rather than the raw
  // strides the unpacked path uses, so it needs the two to share a shape.
  if (!out.sizes().equals(query.sizes())) return 1;
  if (query.stride(2) != 1 || out.stride(2) != 1) return 1;
  const auto num_heads_q = query.size(1);
  const auto num_heads_kv = key_cache.size(2);
  if (num_heads_kv <= 0 || num_heads_q <= num_heads_kv) return 1;
  if (num_heads_q % num_heads_kv != 0) return 1;
  const int64_t pack = num_heads_q / num_heads_kv;

  // Pack only when it strictly reduces the number of Q tiles. Equal tile
  // counts leave the same KV work plus the Q/O permutation overhead.
  const int64_t tile_m = query.size(2) <= kPackGqaSmallTileHeadSize
                             ? kPackGqaSmallTileM
                             : kPackGqaLargeTileM;
  if (max_seqlen_q >= tile_m) return 1;
  const int64_t packed_tiles = (max_seqlen_q * pack + tile_m - 1) / tile_m;
  const int64_t plain_tiles = ((max_seqlen_q + tile_m - 1) / tile_m) * pack;
  if (packed_tiles >= plain_tiles) return 1;

  return static_cast<int>(pack);
}

}  // namespace
#endif

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
    std::optional<const at::Tensor>& q_scale,
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
    const int pack_gqa = chunk_prefill_pack_gqa_factor(
        query,
        key_cache,
        out,
        max_seqlen_q,
        is_varlen,
        is_paged,
        is_sink,
        softmax_lse.has_value(),
        is_prefill.has_value());

    at::Tensor query_in = query;
    at::Tensor out_in = out;
    at::Tensor cu_seqlens_q_in = cu_seqlens_q;
    int max_seqlen_q_in = max_seqlen_q;
    const auto total_q = query.size(0);
    const auto head_size = query.size(2);
    const auto num_heads_kv = query.size(1) / pack_gqa;
    if (pack_gqa > 1) {
      // [T, h_kv * G, d] -> [T * G, h_kv, d], row = q_pos * G + head_in_group
      query_in = query.contiguous()
                     .view({total_q, num_heads_kv, pack_gqa, head_size})
                     .permute({0, 2, 1, 3})
                     .contiguous()
                     .view({total_q * pack_gqa, num_heads_kv, head_size});
      out_in = at::empty_like(query_in);
      cu_seqlens_q_in = cu_seqlens_q * pack_gqa;
      max_seqlen_q_in = max_seqlen_q * pack_gqa;
    }

    // Use XE2 cutlass kernel (also used as WA for XE3/XE3P)
    vllm::xpu::xe2::cutlass_chunk_prefill_xe2(
        queue,
        query_in,
        key_cache,
        value_cache,
        out_in,
        block_table,
        cu_seqlens_q_in,
        cu_seqlens_k,
        max_seqlen_q_in,
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
        is_prefill,
        pack_gqa);

    if (pack_gqa > 1) {
      // Rows belonging to decode batches are skipped by the kernel and carry
      // garbage here; the caller launches paged decode afterwards on the same
      // in-order queue, which overwrites them. The gate rejects the case where
      // that garbage would reach decode through an `out` aliasing `query`.
      out.view({total_q, num_heads_kv, pack_gqa, head_size})
          .copy_(out_in.view({total_q, pack_gqa, num_heads_kv, head_size})
                     .permute({0, 2, 1, 3}));
    }
#else
    TORCH_CHECK(false, "XE2 cutlass kernel is not enabled in this build.");
#endif
  }
#ifdef VLLM_XPU_ENABLE_XE3P
  else if (vllm::xpu::is_xe3p_arch()) {
    // Use XE3 cutlass kernel
    vllm::xpu::xe3::cutlass_chunk_prefill_xe3(
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
        q_scale,
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
  }
#endif
  else {
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
    at::Tensor& softmax_lse_accum,  // [batch, num_head_q, seq_q, num_kv_splits]
    const at::Tensor& block_table,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    std::optional<const at::Tensor>& q_scale,
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
    std::optional<at::Tensor>& work_list,
    std::optional<at::Tensor>& softmax_lse) {
  if (vllm::xpu::is_xe2_arch() || vllm::xpu::is_xe3_arch()) {
#ifdef VLLM_XPU_ENABLE_XE2
    // Use XE2 cutlass kernel (also used as WA for XE3/XE3P)
    vllm::xpu::xe2::cutlass_paged_decode_xe2(
        queue,
        query,
        key_cache,
        value_cache,
        out,
        temp_out,
        softmax_lse_accum,
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
        work_list,
        softmax_lse);
#else
    TORCH_CHECK(false, "XE2 cutlass kernel is not enabled in this build.");
#endif
  }
#ifdef VLLM_XPU_ENABLE_XE3P
  else if (vllm::xpu::is_xe3p_arch()) {
    // Use XE3 cutlass kernel for XE3P (CRI simulator)
    vllm::xpu::xe3::cutlass_paged_decode_xe3(
        queue,
        query,
        key_cache,
        value_cache,
        out,
        temp_out,
        softmax_lse_accum,
        block_table,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q_scale,
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
        work_list,
        softmax_lse);
  }
#endif
  else {
    TORCH_CHECK(false, "Only XE2/XE3 cutlass kernel is supported currently.");
  }
}
