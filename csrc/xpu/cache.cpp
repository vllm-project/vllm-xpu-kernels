#include <sycl/sycl.hpp>

#include <algorithm>
#include <iostream>
#include <string>

#include "dispatch_utils.h"
#include "quantization/fp8/quant_utils.h"
#include "utils.h"

namespace vllm {

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // [num_blocks, num_heads, head_size/x,
                                         // block_size, x]
    cache_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size,
                                         // block_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size, const int x,
    const float* k_scale, const float* v_scale, const sycl::nd_item<1>& item) {
  int group_idx = item.get_group(0);
  int local_idx = item.get_local_id(0);
  int local_range = item.get_local_range(0);
  int slot_idx = slot_mapping[group_idx];
  if (slot_idx < 0) return;

  const int block_idx = slot_idx / block_size;
  const int block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = local_idx; i < n; i += local_range) {
    const int src_key_idx = group_idx * key_stride + i;
    const int src_value_idx = group_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int tgt_key_idx =
        block_idx * num_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
        block_offset * x + x_offset;
    const int tgt_value_idx = block_idx * num_heads * head_size * block_size +
                              head_idx * head_size * block_size +
                              head_offset * block_size + block_offset;
    scalar_t tgt_key = key[src_key_idx];
    scalar_t tgt_value = value[src_value_idx];
    if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
      key_cache[tgt_key_idx] =
          static_cast<at::Float8_e5m2>(tgt_key * (*k_scale));
      value_cache[tgt_value_idx] =
          static_cast<at::Float8_e5m2>(tgt_value * (*v_scale));
    } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
      key_cache[tgt_key_idx] =
          static_cast<at::Float8_e4m3fn>(tgt_key * (*k_scale));
      value_cache[tgt_value_idx] =
          static_cast<at::Float8_e4m3fn>(tgt_value * (*v_scale));
    } else {
      key_cache[tgt_key_idx] = tgt_key;
      value_cache[tgt_value_idx] = tgt_value;
    }
  }
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
void call_reshape_and_cache(
    const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
    cache_t* __restrict__ key_cache, cache_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping, int num_tokens,
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size, const int x,
    const float* k_scale, const float* v_scale) {
  auto& queue = vllm::xpu::vllmGetQueue();
  int wg = std::min(1024, static_cast<int>(num_heads * head_size));

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_tokens * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> item) {
          reshape_and_cache_kernel<scalar_t, cache_t, kv_dt>(
              key, value, key_cache, value_cache, slot_mapping, key_stride,
              value_stride, num_heads, head_size, block_size, x, k_scale,
              v_scale, item);
        });
  });
}

// Used by vectorization_utils to copy/convert one element
template <typename OutT, typename InT, Fp8KVCacheDataType kv_dt>
struct CopyWithScaleOp {
  float scale;

  inline void operator()(OutT& dst, const InT src) const {
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      dst = static_cast<OutT>(src);
    } else {
      float x = (float)src / scale;
      if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
        dst = static_cast<at::Float8_e4m3fn>(x);
      } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
        dst = static_cast<at::Float8_e5m2>(x);
      }
    }
  }
};

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // [num_blocks, block_size, num_heads,
                                         // head_size]
    cache_t* __restrict__ value_cache,   // [num_blocks, block_size, num_heads,
                                         // head_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t block_stride, const int64_t page_stride,
    const int64_t head_stride, const int64_t key_stride,
    const int64_t value_stride, const int num_heads, const int head_size,
    const int block_size, const float* k_scale, const float* v_scale,
    const sycl::nd_item<1>& item) {
  int64_t group_idx = item.get_group(0);
  int64_t local_idx = item.get_local_id(0);
  int local_range = item.get_local_range(0);
  int64_t slot_idx = slot_mapping[group_idx];
  if (slot_idx < 0) return;

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size;

  // pointers to the beginning of the source row for this token.
  const scalar_t* __restrict__ key_src = key + group_idx * key_stride;
  const scalar_t* __restrict__ value_src = value + group_idx * value_stride;

  // find the start position inside the kv-cache for this token.
  cache_t* __restrict__ key_dst =
      key_cache + block_idx * block_stride + block_offset * page_stride;
  cache_t* __restrict__ value_dst =
      value_cache + block_idx * block_stride + block_offset * page_stride;

  float k_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *k_scale;
  float v_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *v_scale;

  CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
  CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};
  fp8::scaled_convert_vec(key_src, key_dst, n, local_idx, local_range, k_op);
  fp8::scaled_convert_vec(value_src, value_dst, n, local_idx, local_range,
                          v_op);
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
void call_reshape_and_cache_flash(
    const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
    cache_t* __restrict__ key_cache, cache_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping, const int64_t block_stride,
    const int64_t page_stride, const int64_t head_stride,
    const int64_t key_stride, const int64_t value_stride, const int num_tokens,
    const int num_heads, const int head_size, const int block_size,
    const float* k_scale, const float* v_scale) {
  auto& queue = vllm::xpu::vllmGetQueue();
  int wg = std::min(1024, static_cast<int>(num_heads * head_size));

  TORCH_CHECK(head_stride == head_size,
              "Only support contiguous heads for vectorization.");
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_tokens * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> item) {
          reshape_and_cache_flash_kernel<scalar_t, cache_t, kv_dt>(
              key, value, key_cache, value_cache, slot_mapping, block_stride,
              page_stride, head_stride, key_stride, value_stride, num_heads,
              head_size, block_size, k_scale, v_scale, item);
        });
  });
}

}  // namespace vllm

// KV_T is the stored data type of kv-cache.
// CACHE_T is the data type of key and value tensors.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)                      \
  VLLM_DISPATCH_FLOATING_TYPES(key.scalar_type(), "reshape_and_cache", [&] { \
    vllm::call_reshape_and_cache<KV_T, CACHE_T, KV_DTYPE>(                   \
        reinterpret_cast<KV_T*>(key.data_ptr()),                             \
        reinterpret_cast<KV_T*>(value.data_ptr()),                           \
        reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                    \
        reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),                  \
        slot_mapping.data_ptr<int64_t>(), num_tokens, key_stride,            \
        value_stride, num_heads, head_size, block_size, x,                   \
        reinterpret_cast<const float*>(k_scale.data_ptr()),                  \
        reinterpret_cast<const float*>(v_scale.data_ptr()));                 \
  });

void reshape_and_cache(torch::Tensor& key, torch::Tensor& value,
                       torch::Tensor& key_cache, torch::Tensor& value_cache,
                       torch::Tensor& slot_mapping,
                       const std::string& kv_cache_dtype,
                       torch::Tensor& k_scale, torch::Tensor& v_scale) {
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  DISPATCH_BY_KV_CACHE_DTYPE(key.scalar_type(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE);
}

// KV_T is the stored data type of kv-cache.
// CACHE_T is the data type of key and value tensors.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE_FLASH(KV_T, CACHE_T, KV_DTYPE)             \
  VLLM_DISPATCH_FLOATING_TYPES(                                           \
      key.scalar_type(), "reshape_and_cache_flash", [&] {                 \
        vllm::call_reshape_and_cache_flash<KV_T, CACHE_T, KV_DTYPE>(      \
            reinterpret_cast<KV_T*>(key.data_ptr()),                      \
            reinterpret_cast<KV_T*>(value.data_ptr()),                    \
            reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),             \
            reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),           \
            slot_mapping.data_ptr<int64_t>(), block_stride, page_stride,  \
            head_stride, key_stride, value_stride, num_tokens, num_heads, \
            head_size, block_size,                                        \
            reinterpret_cast<const float*>(k_scale.data_ptr()),           \
            reinterpret_cast<const float*>(v_scale.data_ptr()));          \
      });

void reshape_and_cache_flash(torch::Tensor& key, torch::Tensor& value,
                             torch::Tensor& key_cache,
                             torch::Tensor& value_cache,
                             torch::Tensor& slot_mapping,
                             const std::string& kv_cache_dtype,
                             torch::Tensor& k_scale, torch::Tensor& v_scale) {
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(1);

  int64_t key_stride = key.stride(0);
  int64_t value_stride = value.stride(0);
  int64_t block_stride = key_cache.stride(0);
  int64_t page_stride = key_cache.stride(1);
  int64_t head_stride = key_cache.stride(2);
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  DISPATCH_BY_KV_CACHE_DTYPE(key.scalar_type(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE_FLASH);
}