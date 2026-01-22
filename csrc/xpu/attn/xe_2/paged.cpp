#include "paged.hpp"
#include "paged_decode_extern.hpp"

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
    int num_kv_splits) {
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
    // num_blocks is used to build total_seqlen_k for shape_K in kernels
    // it is not just the meaning of used blocks for kv.
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
  }

  paged_decode_args_t args = {
      query.data_ptr(),
      key_cache.data_ptr(),
      value_cache.data_ptr(),
      out.data_ptr(),
      temp_out.data_ptr(),
      exp_sums.data_ptr(),
      max_logits.data_ptr(),
      block_table.data_ptr(),
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
      is_sink,
      num_kv_splits};

  CutlassType cuType = aten_to_Cutlass_dtype(query);

  static constexpr int max_head_size = 256;
  TORCH_CHECK(
      head_size <= max_head_size,
      "FMHA forward only supports head dimension at most " +
          std::to_string(max_head_size));

  auto get_head_size_case = [](int head_size) -> int {
    if (head_size <= HEAD_SIZE_LIMIT_0) return 0;
    if (head_size <= HEAD_SIZE_LIMIT_1) return 1;
    if (head_size <= HEAD_SIZE_LIMIT_2) return 2;
    if (head_size <= HEAD_SIZE_LIMIT_3) return 3;
    if (head_size <= HEAD_SIZE_LIMIT_4) return 4;
    return -1;
  };

  int head_case = get_head_size_case(args.head_size);
  int num_q_group_size = num_heads_q / num_heads_kv;

  if (num_q_group_size <= 8) {
    dispatch_by_head_size<_8>(head_case, queue, cuType, args);
  } else if (num_q_group_size <= 16) {
    dispatch_by_head_size<_16>(head_case, queue, cuType, args);
  } else {
    TORCH_CHECK(false, "Unsupported num_heads_q / num_heads_kv for fmha");
  }
}
