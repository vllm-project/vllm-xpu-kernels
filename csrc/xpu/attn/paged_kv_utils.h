#pragma once

#include <limits>
#include <torch/all.h>

// Normalize the physical page stride to sequence-position units. For a regular
// contiguous layout [num_blocks, block_size, num_heads, head_size], this equals
// block_size. For interleaved or cross-layer layouts, it includes the physical
// gaps between logical KV blocks.
inline int64_t
get_paged_kv_cache_page_stride_elements(const at::Tensor& key_cache) {
  return key_cache.stride(0) / key_cache.stride(1);
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
inline void check_paged_kv_cache_strides(
    const at::Tensor& key_cache, const at::Tensor& value_cache) {
  int64_t k_stride_seq = key_cache.stride(1);
  int64_t k_physical_page_stride = key_cache.stride(0);
  int64_t v_stride_seq = value_cache.stride(1);
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
