#pragma once

#include <limits>
#include <torch/all.h>

// Detect whether a paged KV cache tensor uses HND layout.
//
// paged KV supports two layouts:
//   NHD: [num_blocks, block_size, num_heads, head_size]  (contiguous)
//   HND: [num_blocks, num_heads, block_size, head_size]  (permuted from NHD)
//
// HND is created by permuting an NHD tensor via .permute(0, 2, 1, 3).
// After permutation the strides encode the layout:
//   NHD contiguous:  stride(1) = num_heads*head_size  = size(2)*stride(2)
//   HND (permuted):  stride(1) = head_size            < size(2)*stride(2)
//
// This check is robust for cross-layer (non-contiguous page stride) tensors
// because stride(1) and stride(2) within each block are unchanged by the
// outer non-contiguous stride(0).
inline bool is_paged_kv_hnd_layout(const at::Tensor& key_cache) {
  return key_cache.stride(1) != key_cache.size(2) * key_cache.stride(2);
}

// Normalize the physical page stride to sequence-position units.
// paged KV: NHD [num_blocks, block_size, num_heads, head_size]
//        or HND [num_blocks, num_heads, block_size, head_size] (permuted).
// For a regular contiguous NHD layout this equals block_size.
// For interleaved or cross-layer layouts it includes the physical gaps between
// logical KV blocks.
inline int64_t
get_paged_kv_cache_page_stride_elements(const at::Tensor& key_cache) {
  // seq stride: NHD→stride(1), HND (permuted)→stride(2)
  int64_t seq_stride = is_paged_kv_hnd_layout(key_cache) ? key_cache.stride(2)
                                                         : key_cache.stride(1);
  return key_cache.stride(0) / seq_stride;
}

// Return the sequence length needed by the 2D load surface to cover the full
// physical KV extent, not just the logical num_blocks * block_size range.
inline int64_t
get_paged_kv_cache_effective_total_seqlen(const at::Tensor& key_cache) {
  int64_t effective_total =
      key_cache.size(0) * get_paged_kv_cache_page_stride_elements(key_cache);
  return effective_total;
}

// Validate that the physical page stride can be converted to sequence-position
// units and safely passed through int32 kernel parameters.
// Supports both NHD [num_blocks, block_size, num_heads, head_size] and
// HND [num_blocks, num_heads, block_size, head_size] (permuted) layouts.
inline void check_paged_kv_cache_strides(
    const at::Tensor& key_cache, const at::Tensor& value_cache) {
  // seq stride: NHD→stride(1), HND (permuted)→stride(2)
  int64_t k_stride_seq = is_paged_kv_hnd_layout(key_cache)
                             ? key_cache.stride(2)
                             : key_cache.stride(1);
  int64_t k_physical_page_stride = key_cache.stride(0);
  int64_t v_stride_seq = is_paged_kv_hnd_layout(value_cache)
                             ? value_cache.stride(2)
                             : value_cache.stride(1);
  int64_t v_physical_page_stride = value_cache.stride(0);
  TORCH_CHECK(
      k_stride_seq > 0,
      "Paged K sequence stride must be positive: k_stride_seq=",
      k_stride_seq);
  TORCH_CHECK(
      k_physical_page_stride > 0,
      "Paged K page stride must be positive: k_physical_page_stride=",
      k_physical_page_stride);
  TORCH_CHECK(
      k_physical_page_stride % k_stride_seq == 0,
      "Paged K page stride must be divisible by K sequence stride: ",
      "k_physical_page_stride=",
      k_physical_page_stride,
      " k_stride_seq=",
      k_stride_seq);
  int64_t page_stride_elements = k_physical_page_stride / k_stride_seq;
  TORCH_CHECK(
      v_stride_seq > 0,
      "Paged V sequence stride must be positive: v_stride_seq=",
      v_stride_seq);
  TORCH_CHECK(
      v_physical_page_stride % v_stride_seq == 0,
      "Paged V page stride must be divisible by V sequence stride: ",
      "v_physical_page_stride=",
      v_physical_page_stride,
      " v_stride_seq=",
      v_stride_seq);
  TORCH_CHECK(
      v_physical_page_stride / v_stride_seq == page_stride_elements,
      "Paged K/V page strides must match in sequence-position units: ",
      "k_page_stride_elements=",
      page_stride_elements,
      " v_page_stride_elements=",
      v_physical_page_stride / v_stride_seq);
  TORCH_CHECK(
      page_stride_elements <= std::numeric_limits<int>::max(),
      "Paged K page stride in sequence elements exceeds int32 range: ",
      page_stride_elements);
  TORCH_CHECK(
      get_paged_kv_cache_effective_total_seqlen(key_cache) <=
          std::numeric_limits<int>::max(),
      "effective_total exceeds int32 range");
}
