#pragma once

#include <limits>
#include <torch/all.h>

inline bool is_interleaved_kv_cache(
    const at::Tensor& key_cache, const at::Tensor& value_cache) {
  auto value_stride = value_cache.strides();

  if (key_cache.dim() == value_cache.dim() &&
      key_cache.sizes() == value_cache.sizes() &&
      key_cache.strides() == value_stride &&
      key_cache.element_size() == value_cache.element_size() &&
      key_cache.storage().data_ptr() == value_cache.storage().data_ptr() &&
      key_cache.storage_offset() == 0 &&
      value_cache.storage_offset() == value_stride[0] / 2) {
    return true;
  }

  return false;
}

inline int64_t get_paged_kv_cache_k_stride_page(
    const at::Tensor& key_cache, bool is_interleaved_kv) {
  return is_interleaved_kv ? key_cache.stride(0) / 2 : key_cache.stride(0);
}

inline int64_t get_paged_kv_cache_page_stride_elements(
    const at::Tensor& key_cache, bool is_interleaved_kv) {
  return get_paged_kv_cache_k_stride_page(key_cache, is_interleaved_kv) /
         key_cache.stride(1);
}

inline int64_t get_paged_kv_cache_effective_total_seqlen(
    const at::Tensor& key_cache, bool is_interleaved_kv) {
  int64_t effective_total =
      key_cache.size(0) *
      get_paged_kv_cache_page_stride_elements(key_cache, is_interleaved_kv);
  if (is_interleaved_kv) effective_total *= 2;
  return effective_total;
}

inline void check_paged_kv_cache_strides(
    const at::Tensor& key_cache, const at::Tensor& value_cache) {
  bool is_interleaved_kv = is_interleaved_kv_cache(key_cache, value_cache);
  int64_t k_stride_seq = key_cache.stride(1);
  int64_t raw_k_stride_page = key_cache.stride(0);
  TORCH_CHECK(
      k_stride_seq > 0,
      "Paged K sequence stride must be positive: k_stride_seq=",
      k_stride_seq);
  TORCH_CHECK(
      raw_k_stride_page > 0,
      "Paged K page stride must be positive: k_stride_page=",
      raw_k_stride_page);
  if (is_interleaved_kv) {
    TORCH_CHECK(
        raw_k_stride_page % 2 == 0,
        "Interleaved paged K page stride must be even: k_stride_page=",
        raw_k_stride_page);
  }

  int64_t k_stride_page =
      get_paged_kv_cache_k_stride_page(key_cache, is_interleaved_kv);
  TORCH_CHECK(
      k_stride_page % k_stride_seq == 0,
      "Paged K page stride must be divisible by K sequence stride: ",
      "k_stride_page=",
      k_stride_page,
      " k_stride_seq=",
      k_stride_seq);
  int64_t page_stride_elements = k_stride_page / k_stride_seq;
  TORCH_CHECK(
      page_stride_elements <= std::numeric_limits<int>::max(),
      "Paged K page stride in sequence elements exceeds int32 range: ",
      page_stride_elements);
  TORCH_CHECK(
      get_paged_kv_cache_effective_total_seqlen(key_cache, is_interleaved_kv) <=
          std::numeric_limits<int>::max(),
      "effective_total exceeds int32 range");
}
