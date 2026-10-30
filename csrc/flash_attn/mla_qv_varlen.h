#pragma once

#include <torch/all.h>

namespace vllm::mla_qv {

// Absorbed-MLA varlen attention over a paged KV cache, the FA3 ``q_v``
// variant used by vLLM sparse MLA backends:
//
//   score[t, h, n] = (q[t, h] . k[n] + q_v[t, h] . v[n]) * softmax_scale
//   out[t, h]      = softmax_n(score[t, h, :]) @ v
//
// q:            [total_q, num_heads, d_qk]
// q_v:          [total_q, num_heads, d_v]
// k:            [num_blocks, page_size, num_heads_kv, d_qk]   (strided ok)
// v:            [num_blocks, page_size, num_heads_kv, d_v]    (strided ok)
// out:          [total_q, num_heads, d_v], written in place
// cu_seqlens_q: [batch + 1] int32
// seqused_k:    [batch] int32, number of keys visible to each sequence
// block_table:  [batch, max_blocks] int32, negative entries are skipped;
//               keys past max_blocks * page_size are skipped as well
//
// Returns softmax_lse: [num_heads, total_q] float32 when return_softmax_lse
// is set and an empty tensor otherwise.
at::Tensor mla_qv_varlen_fwd(
    const at::Tensor& q,
    const at::Tensor& q_v,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& out,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqused_k,
    const at::Tensor& block_table,
    double softmax_scale,
    bool is_causal,
    bool return_softmax_lse);

}  // namespace vllm::mla_qv
