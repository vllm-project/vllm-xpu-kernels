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
//
// The surface must reach the last row of the last page, i.e.
//   (num_blocks - 1) * page_stride_elements + block_size
// and must NOT be rounded up to num_blocks * page_stride_elements: for
// non-contiguous layouts (page_stride_elements > block_size) the trailing
// page_stride_elements - block_size rows lie in the physical gap after the
// last page. Including them makes the 2D block surface descriptor extend past
// the end of the allocation, which faults for V (whose base pointer is already
// offset into the same allocation).
inline int64_t
get_paged_kv_cache_effective_total_seqlen(const at::Tensor& key_cache) {
  int64_t num_blocks = key_cache.size(0);
  int64_t block_size = key_cache.size(1);
  if (num_blocks <= 0) {
    return 0;
  }
  return (num_blocks - 1) * get_paged_kv_cache_page_stride_elements(key_cache) +
         block_size;
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

// INT8 per-token-head KV cache: int8 K/V views over the block-local split
// layout
// ([int8 K|V data for all (slot, head) | K scales fp32 | V scales fp32] per
// block) plus fp32 scale caches [num_blocks, block_size, num_heads_kv] passed
// as k_scale / v_scale. Validates everything the kernels rely on.
inline void check_int8_kv_cache(
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const std::optional<const at::Tensor>& k_scale,
    const std::optional<const at::Tensor>& v_scale) {
  TORCH_CHECK(
      key_cache.scalar_type() == at::ScalarType::Char &&
          value_cache.scalar_type() == at::ScalarType::Char,
      "INT8 KV cache: key and value caches must both be int8");
  TORCH_CHECK(
      key_cache.dim() == 4 && value_cache.dim() == 4,
      "INT8 KV cache must be paged: [num_blocks, block_size, num_heads_kv, "
      "head_size]");
  TORCH_CHECK(
      k_scale.has_value() && v_scale.has_value(),
      "INT8 KV cache requires per-token-head k_scale and v_scale caches");
  for (const auto* s : {&*k_scale, &*v_scale}) {
    TORCH_CHECK(
        s->scalar_type() == at::ScalarType::Float && s->dim() == 3 &&
            s->size(0) == key_cache.size(0) &&
            s->size(1) == key_cache.size(1) && s->size(2) == key_cache.size(2),
        "INT8 KV scale caches must be float32 [num_blocks, block_size, "
        "num_heads_kv] matching the KV cache");
  }
  TORCH_CHECK(
      k_scale->strides() == v_scale->strides(),
      "INT8 KV k_scale and v_scale caches must have identical strides");
  TORCH_CHECK(
      key_cache.strides() == value_cache.strides(),
      "INT8 KV key and value caches must have identical strides");
  // Xe 2D block loads need 64-byte aligned surface bases; the kernels build one
  // surface per (block, head) for K and for V.
  auto addr = [](const at::Tensor& t) {
    return reinterpret_cast<uintptr_t>(t.data_ptr());
  };
  TORCH_CHECK(
      addr(key_cache) % 64 == 0 && addr(value_cache) % 64 == 0 &&
          key_cache.stride(0) % 64 == 0 && key_cache.stride(2) % 64 == 0,
      "INT8 KV cache: block, head and K/V base addresses must be 64-byte "
      "aligned (head_size must be a multiple of 64 for the split layout); got "
      "page stride ",
      key_cache.stride(0),
      ", head stride ",
      key_cache.stride(2),
      ", V offset ",
      static_cast<int64_t>(addr(value_cache) - addr(key_cache)));
}
