#pragma once

#include <optional>
#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <sycl/sycl.hpp>

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
    std::optional<const at::Tensor>& is_prefill);

void cutlass_paged_decode_interface(
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
    int num_kv_splits,
    std::optional<const at::Tensor>& is_prefill,
    std::optional<at::Tensor>& splits_per_seq,
    std::optional<at::Tensor>& work_list);

std::vector<at::Tensor> flash_mla_sparse_attn_prefill_interface(
	const at::Tensor& q,
	const at::Tensor& kv,
	const at::Tensor& indices,
	float sm_scale,
	int d_v,
	const std::optional<at::Tensor>& attn_sink,
	const std::optional<at::Tensor>& topk_length,
	const std::optional<at::Tensor>& output,
	bool return_softmax_lse);

std::tuple<at::Tensor, std::optional<at::Tensor>>
flash_mla_sparse_attn_decode_interface(
	const at::Tensor& q,
	const at::Tensor& kv,
	const at::Tensor& indices,
	const std::optional<at::Tensor>& topk_length,
	const std::optional<at::Tensor>& attn_sink,
	const std::optional<at::Tensor>& out,
	std::optional<at::Tensor>& tile_scheduler_metadata,
	std::optional<at::Tensor>& num_splits,
	const std::optional<at::Tensor>& extra_kv,
	const std::optional<at::Tensor>& extra_indices,
	const std::optional<at::Tensor>& extra_topk_length,
	int d_v,
	float sm_scale,
	bool return_softmax_lse);
