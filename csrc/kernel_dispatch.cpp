// SPDX-License-Identifier: Apache-2.0
// Runtime dispatch wrappers for cache operations.
// This file is compiled ONCE in the main _C extension and dispatches to the
// arch-specific implementations in per-arch shared libraries (e.g. _C_xe2.so).

#include <torch/torch.h>
#include <optional>
#include <string>

#include "utils.h"

using vllm::xpu::is_xe2_arch;

// Declarations for XE2 variants
extern void reshape_and_cache_xe2(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale);
extern void reshape_and_cache_flash_xe2(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale);
extern void concat_and_cache_mla_xe2(
    torch::Tensor& kv_c,
    torch::Tensor& k_pe,
    torch::Tensor& kv_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor& scale);
extern void gather_cache_xe2(
    torch::Tensor const& src_cache,
    torch::Tensor const& dst,
    torch::Tensor const& block_table,
    torch::Tensor const& cu_seq_lens,
    int64_t batch_size,
    std::optional<torch::Tensor> seq_starts);
extern void gather_and_maybe_dequant_cache_xe2(
    torch::Tensor const& src_cache,
    torch::Tensor const& dst,
    torch::Tensor const& block_table,
    torch::Tensor const& cu_seq_lens,
    torch::Tensor const& token_to_seq,
    int64_t num_tokens,
    const std::string& kv_cache_dtype,
    torch::Tensor const& scale,
    std::optional<torch::Tensor> seq_starts);
extern void swap_blocks_xe2(
    at::Tensor& src,
    at::Tensor& dst,
    int64_t block_size_in_bytes,
    const torch::Tensor& block_map);
extern void swap_blocks_batch_xe2(
    const torch::Tensor& src_ptrs,
    const torch::Tensor& dst_ptrs,
    const torch::Tensor& sizes);
extern void convert_fp8_xe2(
    torch::Tensor& dst,
    const torch::Tensor& src,
    const double scale,
    const std::string& kv_cache_dtype);
extern void indexer_k_quant_and_cache_xe2(
    torch::Tensor& k,
    torch::Tensor& kv_cache,
    torch::Tensor& slot_mapping,
    int64_t quant_block_size,
    const std::string& scale_fmt);
extern void cp_gather_indexer_k_quant_cache_xe2(
    const torch::Tensor& kv_cache,
    torch::Tensor& dst_k,
    torch::Tensor& dst_scale,
    const torch::Tensor& block_table,
    const torch::Tensor& cu_seq_lens);

// Dispatch wrappers — these are the symbols that torch_bindings.cpp references.

void reshape_and_cache(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  if (is_xe2_arch())
    reshape_and_cache_xe2(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale);
  else
    throw std::runtime_error("Unsupported architecture: only XE2 is supported");
}

void reshape_and_cache_flash(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  if (is_xe2_arch())
    reshape_and_cache_flash_xe2(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale);
  else
    throw std::runtime_error("Unsupported architecture: only XE2 is supported");
}

void concat_and_cache_mla(
    torch::Tensor& kv_c,
    torch::Tensor& k_pe,
    torch::Tensor& kv_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor& scale) {
  if (is_xe2_arch())
    concat_and_cache_mla_xe2(
        kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale);
  else
    throw std::runtime_error("Unsupported architecture: only XE2 is supported");
}

void gather_cache(
    torch::Tensor const& src_cache,
    torch::Tensor const& dst,
    torch::Tensor const& block_table,
    torch::Tensor const& cu_seq_lens,
    int64_t batch_size,
    std::optional<torch::Tensor> seq_starts) {
  if (is_xe2_arch())
    gather_cache_xe2(
        src_cache, dst, block_table, cu_seq_lens, batch_size, seq_starts);
  else
    throw std::runtime_error("Unsupported architecture: only XE2 is supported");
}

void gather_and_maybe_dequant_cache(
    torch::Tensor const& src_cache,
    torch::Tensor const& dst,
    torch::Tensor const& block_table,
    torch::Tensor const& cu_seq_lens,
    torch::Tensor const& token_to_seq,
    int64_t num_tokens,
    const std::string& kv_cache_dtype,
    torch::Tensor const& scale,
    std::optional<torch::Tensor> seq_starts) {
  if (is_xe2_arch())
    gather_and_maybe_dequant_cache_xe2(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        token_to_seq,
        num_tokens,
        kv_cache_dtype,
        scale,
        seq_starts);
  else
    throw std::runtime_error("Unsupported architecture: only XE2 is supported");
}

void swap_blocks(
    at::Tensor& src,
    at::Tensor& dst,
    int64_t block_size_in_bytes,
    const torch::Tensor& block_map) {
  if (is_xe2_arch())
    swap_blocks_xe2(src, dst, block_size_in_bytes, block_map);
  else
    throw std::runtime_error("Unsupported architecture: only XE2 is supported");
}

void swap_blocks_batch(
    const torch::Tensor& src_ptrs,
    const torch::Tensor& dst_ptrs,
    const torch::Tensor& sizes) {
  if (is_xe2_arch())
    swap_blocks_batch_xe2(src_ptrs, dst_ptrs, sizes);
  else
    throw std::runtime_error("Unsupported architecture: only XE2 is supported");
}

void convert_fp8(
    torch::Tensor& dst,
    const torch::Tensor& src,
    const double scale,
    const std::string& kv_cache_dtype) {
  if (is_xe2_arch())
    convert_fp8_xe2(dst, src, scale, kv_cache_dtype);
  else
    throw std::runtime_error("Unsupported architecture: only XE2 is supported");
}

void indexer_k_quant_and_cache(
    torch::Tensor& k,
    torch::Tensor& kv_cache,
    torch::Tensor& slot_mapping,
    int64_t quant_block_size,
    const std::string& scale_fmt) {
  if (is_xe2_arch())
    indexer_k_quant_and_cache_xe2(
        k, kv_cache, slot_mapping, quant_block_size, scale_fmt);
  else
    throw std::runtime_error("Unsupported architecture: only XE2 is supported");
}

void cp_gather_indexer_k_quant_cache(
    const torch::Tensor& kv_cache,
    torch::Tensor& dst_k,
    torch::Tensor& dst_scale,
    const torch::Tensor& block_table,
    const torch::Tensor& cu_seq_lens) {
  if (is_xe2_arch())
    cp_gather_indexer_k_quant_cache_xe2(
        kv_cache, dst_k, dst_scale, block_table, cu_seq_lens);
  else
    throw std::runtime_error("Unsupported architecture: only XE2 is supported");
}
