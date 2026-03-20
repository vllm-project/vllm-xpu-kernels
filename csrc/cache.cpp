#include <sycl/sycl.hpp>

#include <algorithm>
#include <iostream>
#include <string>

#include "dispatch_utils.h"
#include "quantization/fp8/quant_utils.h"
#include "utils.h"
#include "utils/mem_cpy.h"

// FP8 E4M3 scale divisor for Intel GPU
constexpr float kFp8E4M3ScaleDivisor = 448.f;

namespace vllm {

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
class reshape_and_cache_kernel {
 public:
  reshape_and_cache_kernel(
      const scalar_t* __restrict__ key,
      const scalar_t* __restrict__ value,
      cache_t* __restrict__ key_cache,
      cache_t* __restrict__ value_cache,
      const int64_t* __restrict__ slot_mapping,
      int key_stride,
      int value_stride,
      int num_heads,
      int head_size,
      int block_size,
      int x,
      const float* k_scale,
      const float* v_scale)
      : key_(key),
        value_(value),
        key_cache_(key_cache),
        value_cache_(value_cache),
        slot_mapping_(slot_mapping),
        key_stride_(key_stride),
        value_stride_(value_stride),
        num_heads_(num_heads),
        head_size_(head_size),
        block_size_(block_size),
        x_(x),
        k_scale_(k_scale),
        v_scale_(v_scale) {}

  void operator()(const sycl::nd_item<1>& item) const {
    int group_idx = item.get_group(0);
    int local_idx = item.get_local_id(0);
    int local_range = item.get_local_range(0);
    int slot_idx = slot_mapping_[group_idx];
    if (slot_idx < 0) return;

    const int block_idx = slot_idx / block_size_;
    const int block_offset = slot_idx % block_size_;
    const int n = num_heads_ * head_size_;
    for (int i = local_idx; i < n; i += local_range) {
      const int src_key_idx = group_idx * key_stride_ + i;
      const int src_value_idx = group_idx * value_stride_ + i;
      const int head_idx = i / head_size_;
      const int head_offset = i % head_size_;
      const int x_idx = head_offset / x_;
      const int x_offset = head_offset % x_;
      const int dst_key_idx =
          block_idx * num_heads_ * (head_size_ / x_) * block_size_ * x_ +
          head_idx * (head_size_ / x_) * block_size_ * x_ +
          x_idx * block_size_ * x_ + block_offset * x_ + x_offset;
      const int dst_value_idx = block_idx * n * block_size_ +
                                head_idx * head_size_ * block_size_ +
                                head_offset * block_size_ + block_offset;
      scalar_t tgt_key = key_[src_key_idx];
      scalar_t tgt_value = value_[src_value_idx];
      if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
        key_cache_[dst_key_idx] =
            static_cast<at::Float8_e5m2>(tgt_key * (*k_scale_));
        value_cache_[dst_value_idx] =
            static_cast<at::Float8_e5m2>(tgt_value * (*v_scale_));
      } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
        key_cache_[dst_key_idx] =
            static_cast<at::Float8_e4m3fn>(tgt_key * (*k_scale_));
        value_cache_[dst_value_idx] =
            static_cast<at::Float8_e4m3fn>(tgt_value * (*v_scale_));
      } else {  // kv_dt == Fp8KVCacheDataType::kAuto
        key_cache_[dst_key_idx] = tgt_key;
        value_cache_[dst_value_idx] = tgt_value;
      }
    }
  }

 private:
  const scalar_t* __restrict__ key_;    // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ value_;  // [num_tokens, num_heads, head_size]
  cache_t* __restrict__ key_cache_;     // [num_blocks, num_heads, head_size/x,
                                        // block_size, x]
  cache_t* __restrict__ value_cache_;   // [num_blocks, num_heads, head_size,
                                        // block_size]
  const int64_t* __restrict__ slot_mapping_;  // [num_tokens]
  const int key_stride_;
  const int value_stride_;
  const int num_heads_;
  const int head_size_;
  const int block_size_;
  const int x_;
  const float* k_scale_;
  const float* v_scale_;
};

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
class reshape_and_cache_flash_kernel {
 public:
  reshape_and_cache_flash_kernel(
      const scalar_t* __restrict__ key,
      const scalar_t* __restrict__ value,
      cache_t* __restrict__ key_cache,
      cache_t* __restrict__ value_cache,
      const int64_t* __restrict__ slot_mapping,
      const int64_t block_stride,
      const int64_t page_stride,
      const int64_t head_stride,
      const int64_t key_stride,
      const int64_t value_stride,
      const int num_heads,
      const int head_size,
      const int block_size,
      const float* k_scale,
      const float* v_scale)
      : key_(key),
        value_(value),
        key_cache_(key_cache),
        value_cache_(value_cache),
        slot_mapping_(slot_mapping),
        block_stride_(block_stride),
        page_stride_(page_stride),
        head_stride_(head_stride),
        key_stride_(key_stride),
        value_stride_(value_stride),
        num_heads_(num_heads),
        head_size_(head_size),
        block_size_(block_size),
        k_scale_(k_scale),
        v_scale_(v_scale) {}

  void operator()(const sycl::nd_item<1>& item) const {
    int64_t group_idx = item.get_group(0);
    int64_t local_idx = item.get_local_id(0);
    int local_range = item.get_local_range(0);
    int64_t slot_idx = slot_mapping_[group_idx];
    if (slot_idx < 0) return;

    const int64_t block_idx = slot_idx / block_size_;
    const int64_t block_offset = slot_idx % block_size_;
    const int n = num_heads_ * head_size_;

    // pointers to the beginning of the source row for this token.
    const scalar_t* __restrict__ key_src = key_ + group_idx * key_stride_;
    const scalar_t* __restrict__ value_src = value_ + group_idx * value_stride_;

    // find the start position inside the kv-cache for this token.
    cache_t* __restrict__ key_dst =
        key_cache_ + block_idx * block_stride_ + block_offset * page_stride_;
    cache_t* __restrict__ value_dst =
        value_cache_ + block_idx * block_stride_ + block_offset * page_stride_;

    float k_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *k_scale_;
    float v_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *v_scale_;

    fp8::CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
    fp8::CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};
    fp8::scaled_convert_vec(key_src, key_dst, n, local_idx, local_range, k_op);
    fp8::scaled_convert_vec(
        value_src, value_dst, n, local_idx, local_range, v_op);
  }

 private:
  const scalar_t* __restrict__ key_;    // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ value_;  // [num_tokens, num_heads, head_size]
  cache_t* __restrict__ key_cache_;     // [num_blocks, block_size, num_heads,
                                        // head_size]
  cache_t* __restrict__ value_cache_;   // [num_blocks, block_size, num_heads,
                                        // head_size]
  const int64_t* __restrict__ slot_mapping_;  // [num_tokens]
  const int64_t block_stride_;
  const int64_t page_stride_;
  const int64_t head_stride_;
  const int64_t key_stride_;
  const int64_t value_stride_;
  const int num_heads_;
  const int head_size_;
  const int block_size_;
  const float* k_scale_;
  const float* v_scale_;
};

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
class concat_and_cache_mla_kernel {
 public:
  concat_and_cache_mla_kernel(
      const scalar_t* __restrict__ kv_c,
      const scalar_t* __restrict__ k_pe,
      cache_t* __restrict__ kv_cache,
      const int64_t* __restrict__ slot_mapping,
      const int block_stride,
      const int entry_stride,
      const int kv_c_stride,
      const int k_pe_stride,
      const int kv_lora_rank,
      const int pe_dim,
      const int block_size,
      const float* scale)
      : kv_c(kv_c),
        k_pe(k_pe),
        kv_cache(kv_cache),
        slot_mapping(slot_mapping),
        block_stride(block_stride),
        entry_stride(entry_stride),
        kv_c_stride(kv_c_stride),
        k_pe_stride(k_pe_stride),
        kv_lora_rank(kv_lora_rank),
        pe_dim(pe_dim),
        block_size(block_size),
        scale(scale) {}

  void operator()(const sycl::nd_item<1> item_id) const {
    const int64_t token_idx = item_id.get_group(0);
    const int64_t slot_idx = slot_mapping[token_idx];
    // NOTE: slot_idx can be -1 if the token is padded
    if (slot_idx < 0) {
      return;
    }
    const int block_idx = slot_idx / block_size;
    const int block_offset = slot_idx % block_size;

    auto copy = [&](const scalar_t* __restrict__ src,
                    cache_t* dst,
                    int src_stride,
                    int dst_stride,
                    int size,
                    int offset) {
      for (int i = item_id.get_local_id(0); i < size;
           i += item_id.get_local_range(0)) {
        const int src_idx = token_idx * src_stride + i;
        const int dst_idx =
            block_idx * block_stride + block_offset * entry_stride + i + offset;
        if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
          dst[dst_idx] = static_cast<at::Float8_e4m3fn>(src[src_idx] * *scale);
        } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
          dst[dst_idx] = static_cast<at::Float8_e5m2>(src[src_idx] * *scale);
        } else {
          dst[dst_idx] = src[src_idx];
        }
      }
    };

    copy(kv_c, kv_cache, kv_c_stride, block_stride, kv_lora_rank, 0);
    copy(k_pe, kv_cache, k_pe_stride, block_stride, pe_dim, kv_lora_rank);
  }

 private:
  const scalar_t* __restrict__ kv_c;  // [num_tokens, kv_lora_rank]
  const scalar_t* __restrict__ k_pe;  // [num_tokens, pe_dim]
  cache_t* __restrict__ kv_cache;  // [num_blocks, block_size, (kv_lora_rank +
                                   // pe_dim)]
  const int64_t* __restrict__ slot_mapping;  // [num_tokens]
  const int block_stride;                    //
  const int entry_stride;                    //
  const int kv_c_stride;                     //
  const int k_pe_stride;                     //
  const int kv_lora_rank;                    //
  const int pe_dim;                          //
  const int block_size;                      //
  const float* scale;                        //
};

// grid is launched with dimensions (batch, num_splits)
template <typename scalar_t>
class gather_cache_kernel {
 public:
  gather_cache_kernel(
      const scalar_t* __restrict__ src_cache,
      scalar_t* __restrict__ dst,
      const int32_t* __restrict__ block_table,
      const int32_t* __restrict__ cu_seq_lens,
      const int32_t block_size,
      const int32_t entry_size,
      const int64_t block_table_stride,
      const int64_t cache_block_stride,
      const int64_t cache_entry_stride,
      const int64_t dst_entry_stride,
      const int32_t* __restrict__ seq_starts)
      : src_cache(src_cache),
        dst(dst),
        block_table(block_table),
        cu_seq_lens(cu_seq_lens),
        block_size(block_size),
        entry_size(entry_size),
        block_table_stride(block_table_stride),
        cache_block_stride(cache_block_stride),
        cache_entry_stride(cache_entry_stride),
        dst_entry_stride(dst_entry_stride),
        seq_starts(seq_starts) {}

  void operator()(const sycl::nd_item<2> item_id) const {
    const int64_t bid = item_id.get_group(1);  // Batch ID
    const int32_t num_splits = item_id.get_global_range(0);
    const int32_t split = item_id.get_group(0);
    const int32_t seq_start = cu_seq_lens[bid];
    const int32_t seq_end = cu_seq_lens[bid + 1];
    const int32_t seq_len = seq_end - seq_start;
    const int32_t tot_blocks = (seq_len + block_size - 1) / block_size;
    const int32_t split_blocks = (tot_blocks + num_splits - 1) / num_splits;

    const int32_t split_start = split * split_blocks;
    const int32_t split_end = std::min((split + 1) * split_blocks, tot_blocks);

    const bool is_active_split = (split_start < tot_blocks);
    const bool is_last_split = (split_end == tot_blocks);

    if (!is_active_split) return;

    int32_t full_blocks_end = split_end;
    int32_t partial_block_size = 0;

    // Adjust the pointer for the block_table for this batch.
    // If seq_starts is provided, compute an offset based on (seq_starts[bid] /
    // page_size)
    const int32_t batch_offset = bid * block_table_stride;
    int32_t offset = 0;
    if (seq_starts != nullptr) {
      offset = seq_starts[bid] / block_size;
    }
    const int32_t* batch_block_table = block_table + batch_offset + offset;

    // Adjust dst pointer based on the cumulative sequence lengths.
    scalar_t* dst_ptr = dst + seq_start * dst_entry_stride;

    if (is_last_split) {
      partial_block_size = seq_len % block_size;
      if (partial_block_size) full_blocks_end -= 1;
    }

    auto copy_entry = [&](const scalar_t* __restrict__ _src,
                          scalar_t* __restrict__ _dst) {
      for (int i = item_id.get_local_id(1); i < entry_size;
           i += item_id.get_local_range(1))
        _dst[i] = _src[i];
    };

    for (int pid = split_start; pid < full_blocks_end; ++pid) {
      auto block_id = batch_block_table[pid];
      auto block_start_ptr = src_cache + block_id * cache_block_stride;
      auto block_dst_ptr = dst_ptr + pid * block_size * dst_entry_stride;
      for (int eid = 0; eid < block_size; ++eid) {
        copy_entry(
            block_start_ptr + eid * cache_entry_stride,
            block_dst_ptr + eid * dst_entry_stride);
      }
    }

    if (partial_block_size) {
      auto block_id = batch_block_table[full_blocks_end];
      auto block_start_ptr = src_cache + block_id * cache_block_stride;
      auto block_dst_ptr =
          dst_ptr + full_blocks_end * block_size * dst_entry_stride;
      for (int eid = 0; eid < partial_block_size; ++eid) {
        copy_entry(
            block_start_ptr + eid * cache_entry_stride,
            block_dst_ptr + eid * dst_entry_stride);
      }
    }
  }

 private:
  const scalar_t* __restrict__ src_cache;   // [NUM_BLOCKS, BLOCK_SIZE,
                                            // ENTRIES...]
  scalar_t* __restrict__ dst;               // [TOT_TOKENS, ENTRIES...]
  const int32_t* __restrict__ block_table;  // [BATCH, BLOCK_INDICES]
  const int32_t* __restrict__ cu_seq_lens;  // [BATCH+1]
  const int32_t block_size;
  const int32_t entry_size;
  const int64_t block_table_stride;
  const int64_t cache_block_stride;
  const int64_t cache_entry_stride;
  const int64_t dst_entry_stride;
  const int32_t* __restrict__ seq_starts;  // Optional: starting offsets per
};

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
class indexer_k_quant_and_cache_kernel {
 public:
  indexer_k_quant_and_cache_kernel(
      const scalar_t* __restrict__ k,
      cache_t* __restrict__ kv_cache,
      const int64_t* __restrict__ slot_mapping,
      const int head_dim,
      const int quant_block_size,
      const int cache_block_size,
      const int cache_stride,
      bool use_ue8m0)
      : k_(k),
        kv_cache_(kv_cache),
        slot_mapping_(slot_mapping),
        head_dim_(head_dim),
        quant_block_size_(quant_block_size),
        cache_block_size_(cache_block_size),
        cache_stride_(cache_stride),
        use_ue8m0_(use_ue8m0) {}

  void operator()(const sycl::nd_item<2>& item_id) const {
    constexpr int VEC_SIZE = 4;
    int64_t local_x = item_id.get_local_id(0);
    int64_t local_y = item_id.get_local_id(1);
    int64_t group_x = item_id.get_group(0);
    int64_t token_idx = item_id.get_group(1);
    int64_t head_dim_idx =
        (group_x * item_id.get_local_range(0) * item_id.get_local_range(1) +
         local_x * item_id.get_local_range(1) + local_y) *
        VEC_SIZE;

    int64_t slot_idx = slot_mapping_[token_idx];
    const int64_t block_idx = slot_idx / cache_block_size_;
    const int64_t block_offset = slot_idx % cache_block_size_;

    if (slot_idx < 0 || head_dim_idx >= head_dim_) return;

    // Compute local amax
    float amax = 0.f;
    scalar_t k_vals[VEC_SIZE];
    for (int i = 0; i < VEC_SIZE; i++) {
      k_vals[i] = k_[token_idx * head_dim_ + head_dim_idx + i];
      amax = sycl::fmax(amax, sycl::fabs(static_cast<float>(k_vals[i])));
    }

    // group-level reduction (sub-group reduce max)
    auto sg = item_id.get_sub_group();
    amax = sycl::reduce_over_group(sg, amax, sycl::maximum<float>{});

    float scale = sycl::fmax(amax, 1e-4f) / kFp8E4M3ScaleDivisor;

    if (use_ue8m0_) {
      scale = sycl::exp2(sycl::ceil(sycl::log2(scale)));
    }
    // Put scale in the back of quanted values for the sake of data contiuity
    const int64_t dst_offset = block_idx * cache_block_size_ * cache_stride_ +
                               block_offset * head_dim_ + head_dim_idx;

    fp8::CopyWithScaleOp<cache_t, scalar_t, kv_dt> op{scale};
    for (int i = 0; i < VEC_SIZE; i++) {
      op(kv_cache_[dst_offset + i], k_vals[i]);
    }

    if (local_y == 0) {
      const int64_t dst_scale_idx =
          block_idx * cache_block_size_ * cache_stride_ +
          cache_block_size_ * head_dim_ +
          (block_offset * head_dim_ + head_dim_idx) * 4 / quant_block_size_;
      reinterpret_cast<float*>(kv_cache_)[dst_scale_idx / 4] = scale;
    }
  }

 private:
  const scalar_t* __restrict__ k_;  // [num_tokens, head_dim]
  cache_t* __restrict__ kv_cache_;  // [num_blocks, block_size, cache_stride]
  const int64_t* __restrict__ slot_mapping_;  // [num_tokens]
  const int64_t head_dim_;
  const int64_t quant_block_size_;
  const int64_t cache_block_size_;
  const int64_t cache_stride_;
  const bool use_ue8m0_;
};

// Kernel to gather K data and its FP8 block-scales back from the paged KV-cache
// into flat contiguous destination tensors, used during attention prefill.
template <int BLOCK_Y_SIZE>
class cp_gather_indexer_k_quant_cache_kernel {
 public:
  cp_gather_indexer_k_quant_cache_kernel(
      const char* __restrict__ kv_cache,
      char* __restrict__ dst_k,
      char* __restrict__ dst_scale,
      const int32_t* __restrict__ block_table,
      const int32_t* __restrict__ cu_seq_lens,
      sycl::local_accessor<int, 1> batch_idx_acc,
      const int batch_size,
      const int64_t token_stride,
      const int64_t head_dim,
      const int64_t block_stride,
      const int64_t cache_block_size,
      const int num_blocks,
      const int num_tokens,
      const int quant_block_size)
      : kv_cache_(kv_cache),
        dst_k_(dst_k),
        dst_scale_(dst_scale),
        block_table_(block_table),
        cu_seq_lens_(cu_seq_lens),
        batch_idx_acc_(batch_idx_acc),
        batch_size_(batch_size),
        token_stride_(token_stride),
        head_dim_(head_dim),
        block_stride_(block_stride),
        cache_block_size_(cache_block_size),
        num_blocks_(num_blocks),
        num_tokens_(num_tokens),
        quant_block_size_(quant_block_size) {}

  void operator()(sycl::nd_item<2> item) const {
    // VEC_SIZE = sizeof(sycl::float4) / sizeof(char) = 16
    // This matches CUDA's float4 vectorised load/store of fp8 bytes.
    constexpr int VEC_SIZE = 16;

    const int local_x = static_cast<int>(item.get_local_id(0));
    const int local_y = static_cast<int>(item.get_local_id(1));
    const int local_range_x =
        static_cast<int>(item.get_local_range(0));  // BLOCK_Y_SIZE
    const int local_range_y = static_cast<int>(item.get_local_range(1));  // 8

    const int token_idx =
        static_cast<int>(item.get_group(1) * local_range_x + local_x);
    const int head_idx =
        static_cast<int>(item.get_group(0) * local_range_y + local_y) *
        VEC_SIZE;

    for (int iter = 0; iter < (batch_size_ + local_range_y - 1) / local_range_y;
         ++iter) {
      int tid = iter * local_range_y + local_y;
      if (tid < batch_size_) {
        const int seq_start = cu_seq_lens_[tid];
        const int seq_end = cu_seq_lens_[tid + 1];
        if (token_idx >= seq_start && token_idx < seq_end) {
          batch_idx_acc_[local_x] = tid;
        }
      }
    }

    // Synchronise SLM writes before any thread reads batch_idx_acc_
    item.barrier(sycl::access::fence_space::local_space);

    if (head_idx >= static_cast<int>(head_dim_) || token_idx >= num_tokens_) {
      return;
    }

    const int my_batch = batch_idx_acc_[local_x];
    const int inbatch_seq_idx = token_idx - cu_seq_lens_[my_batch];
    const int block_idx = block_table_
        [my_batch * num_blocks_ + inbatch_seq_idx / cache_block_size_];

    const int64_t src_block_offset =
        static_cast<int64_t>(block_idx) * block_stride_;
    const int64_t cache_inblock_offset =
        static_cast<int64_t>(inbatch_seq_idx % cache_block_size_) * head_dim_ +
        head_idx;
    const int64_t src_inblock_offset = src_block_offset + cache_inblock_offset;
    const int64_t dst_inblock_offset =
        static_cast<int64_t>(token_idx) * token_stride_ + head_idx;

    // Vectorised 16-byte copy: read 16 fp8 bytes from cache, write to dst_k
    reinterpret_cast<sycl::float4*>(dst_k_)[dst_inblock_offset / VEC_SIZE] =
        reinterpret_cast<const sycl::float4*>(
            kv_cache_)[src_inblock_offset / VEC_SIZE];

    // Scale copy: only local_y==0 writes (one scale per quant_block)
    if (local_y == 0) {
      const int64_t src_scale_offset =
          src_block_offset +
          static_cast<int64_t>(cache_block_size_) * head_dim_ +
          cache_inblock_offset * 4 / quant_block_size_;
      reinterpret_cast<float*>(
          dst_scale_)[dst_inblock_offset / quant_block_size_] =
          reinterpret_cast<const float*>(kv_cache_)[src_scale_offset / 4];
    }
  }

 private:
  const char* kv_cache_;
  char* dst_k_;
  char* dst_scale_;
  const int32_t* block_table_;
  const int32_t* cu_seq_lens_;
  sycl::local_accessor<int, 1> batch_idx_acc_;
  int batch_size_;
  int64_t token_stride_;
  int64_t head_dim_;
  int64_t block_stride_;
  int64_t cache_block_size_;
  int num_blocks_;
  int num_tokens_;
  int quant_block_size_;
};

}  // namespace vllm

// KV_T is the stored data type of kv-cache.
// CACHE_T is the data type of key and value tensors.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)           \
  queue.submit([&](sycl::handler& cgh) {                          \
    cgh.parallel_for(                                             \
        sycl::nd_range<1>(grid * block, block),                   \
        vllm::reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>(  \
            reinterpret_cast<KV_T*>(key.data_ptr()),              \
            reinterpret_cast<KV_T*>(value.data_ptr()),            \
            reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),     \
            reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),   \
            slot_mapping.data_ptr<int64_t>(),                     \
            key_stride,                                           \
            value_stride,                                         \
            num_heads,                                            \
            head_size,                                            \
            block_size,                                           \
            x,                                                    \
            reinterpret_cast<const float*>(k_scale.data_ptr()),   \
            reinterpret_cast<const float*>(v_scale.data_ptr()))); \
  });

void reshape_and_cache(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(std::min(num_heads * head_size, 1024));
  const at::DeviceGuard device_guard(key.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  DISPATCH_BY_KV_CACHE_DTYPE(
      key.scalar_type(), kv_cache_dtype, CALL_RESHAPE_AND_CACHE);
}

// KV_T is the stored data type of kv-cache.
// CACHE_T is the data type of key and value tensors.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE_FLASH(KV_T, CACHE_T, KV_DTYPE)          \
  queue.submit([&](sycl::handler& cgh) {                               \
    cgh.parallel_for(                                                  \
        sycl::nd_range<1>(grid * block, block),                        \
        vllm::reshape_and_cache_flash_kernel<KV_T, CACHE_T, KV_DTYPE>( \
            reinterpret_cast<KV_T*>(key.data_ptr()),                   \
            reinterpret_cast<KV_T*>(value.data_ptr()),                 \
            reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),          \
            reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),        \
            slot_mapping.data_ptr<int64_t>(),                          \
            block_stride,                                              \
            page_stride,                                               \
            head_stride,                                               \
            key_stride,                                                \
            value_stride,                                              \
            num_heads,                                                 \
            head_size,                                                 \
            block_size,                                                \
            reinterpret_cast<const float*>(k_scale.data_ptr()),        \
            reinterpret_cast<const float*>(v_scale.data_ptr())));      \
  });

void reshape_and_cache_flash(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
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

  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(std::min(num_heads * head_size, 1024));
  const at::DeviceGuard device_guard(key.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  DISPATCH_BY_KV_CACHE_DTYPE(
      key.scalar_type(), kv_cache_dtype, CALL_RESHAPE_AND_CACHE_FLASH);
}

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_CONCAT_AND_CACHE_MLA(KV_T, CACHE_T, KV_DTYPE)          \
  queue.submit([&](sycl::handler& cgh) {                            \
    cgh.parallel_for(                                               \
        sycl::nd_range<1>(grid * block, block),                     \
        vllm::concat_and_cache_mla_kernel<KV_T, CACHE_T, KV_DTYPE>( \
            reinterpret_cast<KV_T*>(kv_c.data_ptr()),               \
            reinterpret_cast<KV_T*>(k_pe.data_ptr()),               \
            reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),        \
            slot_mapping.data_ptr<int64_t>(),                       \
            block_stride,                                           \
            entry_stride,                                           \
            kv_c_stride,                                            \
            k_pe_stride,                                            \
            kv_lora_rank,                                           \
            pe_dim,                                                 \
            block_size,                                             \
            reinterpret_cast<const float*>(scale.data_ptr())));     \
  });

void concat_and_cache_mla(
    torch::Tensor& kv_c,          // [num_tokens, kv_lora_rank]
    torch::Tensor& k_pe,          // [num_tokens, pe_dim]
    torch::Tensor& kv_cache,      // [num_blocks, block_size, (kv_lora_rank +
                                  // pe_dim)]
    torch::Tensor& slot_mapping,  // [num_tokens] or [num_actual_tokens]
    const std::string& kv_cache_dtype,
    torch::Tensor& scale) {
  // NOTE(woosuk): In vLLM V1, key.size(0) can be different from
  // slot_mapping.size(0) because of padding for CUDA graphs.
  // In vLLM V0, key.size(0) is always equal to slot_mapping.size(0) because
  // both include padding.
  // In vLLM V1, however, key.size(0) can be larger than slot_mapping.size(0)
  // since key includes padding for CUDA graphs, while slot_mapping does not.
  // In this case, slot_mapping.size(0) represents the actual number of tokens
  // before padding.
  // For compatibility with both cases, we use slot_mapping.size(0) as the
  // number of tokens.
  int num_tokens = slot_mapping.size(0);
  int kv_lora_rank = kv_c.size(1);
  int pe_dim = k_pe.size(1);
  int block_size = kv_cache.size(1);

  TORCH_CHECK(kv_cache.size(2) == kv_lora_rank + pe_dim);

  int kv_c_stride = kv_c.stride(0);
  int k_pe_stride = k_pe.stride(0);
  int block_stride = kv_cache.stride(0);
  int entry_stride = kv_cache.stride(1);

  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(std::min(kv_lora_rank, 1024));
  const at::DeviceGuard device_guard(kv_c.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  DISPATCH_BY_KV_CACHE_DTYPE(
      kv_c.scalar_type(), kv_cache_dtype, CALL_CONCAT_AND_CACHE_MLA);
}

// Macro to dispatch the kernel based on the data type.
#define CALL_GATHER_CACHE(CPY_DTYPE)                            \
  queue.submit([&](sycl::handler& cgh) {                        \
    cgh.parallel_for(                                           \
        sycl::nd_range<2>(grid * block, block),                 \
        vllm::gather_cache_kernel<CPY_DTYPE>(                   \
            reinterpret_cast<CPY_DTYPE*>(src_cache.data_ptr()), \
            reinterpret_cast<CPY_DTYPE*>(dst.data_ptr()),       \
            block_table.data_ptr<int32_t>(),                    \
            cu_seq_lens.data_ptr<int32_t>(),                    \
            block_size,                                         \
            entry_size,                                         \
            block_table_stride,                                 \
            cache_block_stride,                                 \
            cache_entry_stride,                                 \
            dst_entry_stride,                                   \
            seq_starts_ptr));                                   \
  });

// Gather sequences from the cache into the destination tensor.
//  - cu_seq_lens contains the cumulative sequence lengths for each batch
//  - block_table contains the cache block indices for each sequence
//  - Optionally, seq_starts (if provided) offsets the starting block index by
//  (seq_starts[bid] / page_size)
void gather_cache(
    torch::Tensor const& src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::Tensor const& dst,          // [TOT_TOKENS, ENTRIES...]
    torch::Tensor const& block_table,  // [BATCH, BLOCK_INDICES]
    torch::Tensor const& cu_seq_lens,  // [BATCH+1]
    int64_t batch_size,
    std::optional<torch::Tensor> seq_starts = std::nullopt) {
  const at::DeviceGuard device_guard(src_cache.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  int32_t block_size = src_cache.size(1);
  int32_t entry_size = src_cache.flatten(2, -1).size(2);

  TORCH_CHECK(block_table.dtype() == at::kInt, "block_table must be int32");
  TORCH_CHECK(cu_seq_lens.dtype() == at::kInt, "cu_seq_lens must be int32");
  if (seq_starts.has_value()) {
    TORCH_CHECK(
        seq_starts.value().dtype() == at::kInt, "seq_starts must be int32");
  }

  TORCH_CHECK(
      src_cache.device() == dst.device(),
      "src_cache and dst must be on the same device");
  TORCH_CHECK(
      src_cache.device() == block_table.device(),
      "src_cache and block_table must be on the same device");
  TORCH_CHECK(
      src_cache.device() == cu_seq_lens.device(),
      "src_cache and cu_seq_lens must be on the same device");
  if (seq_starts.has_value()) {
    TORCH_CHECK(
        src_cache.device() == seq_starts.value().device(),
        "src_cache and seq_starts must be on the same device");
  }

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);
  int64_t cache_entry_stride = src_cache.stride(1);
  int64_t dst_entry_stride = dst.stride(0);

  // Decide on the number of splits based on the batch size.
  int num_splits = batch_size > 128 ? 2 : batch_size > 64 ? 4 : 16;
  sycl::range<2> grid(num_splits, batch_size);
  sycl::range<2> block(1, 1024);

  TORCH_CHECK(
      src_cache.dtype() == dst.dtype(),
      "src_cache and dst must have the same dtype");

  const int dtype_bits = src_cache.element_size() * 8;
  const int32_t* seq_starts_ptr =
      seq_starts.has_value() ? seq_starts.value().data_ptr<int32_t>() : nullptr;

  if (dtype_bits == 32) {
    CALL_GATHER_CACHE(uint32_t);
  } else if (dtype_bits == 16) {
    CALL_GATHER_CACHE(uint16_t);
  } else if (dtype_bits == 8) {
    CALL_GATHER_CACHE(uint8_t);
  } else {
    TORCH_CHECK(false, "Unsupported data type width: ", dtype_bits);
  }
}

/**
 * @brief Swaps data blocks between source and destination tensors for KV cache
 * offloading.
 *
 * Typically used to move blocks between GPU HBM and host memory (CPU DRAM) or
 * between different memory tiers without full tensor copies. Supports
 * XPU-to-XPU, XPU-to-CPU, and CPU-to-XPU transfers using asynchronous memory
 * operations.
 *
 * @param src                  Source tensor containing KV cache blocks to be
 * moved
 * @param dst                  Destination tensor to receive the swapped blocks
 * @param block_size_in_bytes  Size of each KV cache block in bytes
 * @param block_map            Mapping tensor of shape [num_pairs, 2] where each
 * row contains [src_block_idx, dst_block_idx] pairs defining which blocks to
 * swap and their destination locations. Must be a contiguous CPU tensor with
 * int64 dtype.
 *
 * @throws std::runtime_error  If device combination is invalid, tensors are not
 * on expected devices, or block_map has wrong shape/dtype
 *
 * @note The block_map tensor must reside on CPU and be contiguous. For pinned
 * (page-locked) host memory, the host context (hctx) is extracted to enable
 * faster DMA transfers. This function initiates async copies; synchronization
 * must be handled externally.
 */
void swap_blocks(
    at::Tensor& src,
    at::Tensor& dst,
    int64_t block_size_in_bytes,
    const torch::Tensor& block_map  // [num_pairs, 2]
) {
  at::Device src_device = src.device();
  at::Device dst_device = dst.device();

  const at::OptionalDeviceGuard device_guard(
      src_device.is_xpu()
          ? src_device
          : (dst_device.is_xpu() ? dst_device : at::Device(at::kCPU)));

  vllm::xpu::xpuMemcpyKind cpy_kind;
  if (src_device.is_xpu() && dst_device.is_xpu()) {
    TORCH_CHECK(
        src_device.index() == dst_device.index(),
        "src and dst must be on the same XPU");
    cpy_kind = vllm::xpu::xpuMemcpyKind::DeviceToDevice;
  } else if (src_device.is_xpu() && dst_device.is_cpu()) {
    cpy_kind = vllm::xpu::xpuMemcpyKind::DeviceToHost;
  } else if (src_device.is_cpu() && dst_device.is_xpu()) {
    cpy_kind = vllm::xpu::xpuMemcpyKind::HostToDevice;
  } else {
    TORCH_CHECK(false, "Invalid device combination");
  }

  TORCH_CHECK(block_map.device().is_cpu(), "block_map must be on CPU");
  TORCH_CHECK(
      block_map.scalar_type() == at::kLong,
      "block_map must have dtype int64 (Long)");
  TORCH_CHECK(
      block_map.dim() == 2,
      "block_map must be a 2D tensor of shape (N, 2); got dim() = ",
      block_map.dim());
  TORCH_CHECK(
      block_map.size(1) == 2,
      "block_map must have shape (N, 2); got size(1) = ",
      block_map.size(1));
  TORCH_CHECK(
      block_map.is_contiguous(),
      "block_map must be contiguous to be indexed as a flat (N, 2) array");
  TORCH_CHECK(
      block_size_in_bytes > 0,
      "block_size_in_bytes must be positive; got ",
      block_size_in_bytes);

  char* src_ptr = static_cast<char*>(src.data_ptr());
  char* dst_ptr = static_cast<char*>(dst.data_ptr());

  const int64_t* block_map_data = block_map.data_ptr<int64_t>();
  const int64_t num_blocks = block_map.size(0);

  // Identify the host tensor based on copy direction and extract hctx
  const at::Tensor* host_tensor = nullptr;
  if (cpy_kind == vllm::xpu::xpuMemcpyKind::HostToDevice) {
    host_tensor = &src;  // Host is source
  } else if (cpy_kind == vllm::xpu::xpuMemcpyKind::DeviceToHost) {
    host_tensor = &dst;  // Host is destination
  }

  bool is_pinned = false;
  const void* hctx = nullptr;
  if (host_tensor != nullptr) {
    is_pinned = host_tensor->is_pinned();
    if (is_pinned) {
      // Extract hctx from the tensor's storage DataPtr context
      hctx = host_tensor->storage().data_ptr().get_context();
    }
  }

  for (int64_t i = 0; i < num_blocks; i++) {
    int64_t src_block_number = block_map_data[i * 2];
    int64_t dst_block_number = block_map_data[i * 2 + 1];

    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;

    vllm::xpu::xpuAsyncMemcpy(
        dst_ptr + dst_offset,
        src_ptr + src_offset,
        block_size_in_bytes,
        cpy_kind,
        hctx,
        is_pinned);
  }

  return;
}

namespace vllm {

// Kernel for FP8 conversion
// Converts between FP8 and FP16/BF16/FP32 formats with scaling
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
class convert_fp8_kernel {
 public:
  convert_fp8_kernel(
      cache_t* __restrict__ dst,
      const scalar_t* __restrict__ src,
      const float scale,
      const int64_t numel)
      : dst_(dst), src_(src), scale_(scale), numel_(numel) {}

  void operator()(const sycl::nd_item<1>& item) const {
    const int64_t idx = item.get_global_id(0);
    if (idx >= numel_) return;

    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      // Dequantize: FP8 -> FP16/BF16/FP32
      // In this case, cache_t is the output type (FP16/BF16/FP32)
      // and scalar_t is the input type (FP8)
      dst_[idx] = static_cast<cache_t>(static_cast<float>(src_[idx]) * scale_);
    } else {
      // Quantize: FP16/BF16/FP32 -> FP8 E5M2/FP8 E4M3
      using out_dtype = std::conditional_t<
          kv_dt == Fp8KVCacheDataType::kFp8E5M2,
          at::Float8_e5m2,
          at::Float8_e4m3fn>;
      float fp8_max = vllm::fp8::quant_type_max_v<out_dtype>;
      float x = static_cast<float>(src_[idx]) / scale_;
      x = sycl::fmax(-fp8_max, sycl::fmin(x, fp8_max));
      auto fp8_val = static_cast<out_dtype>(x);
      dst_[idx] = sycl::bit_cast<cache_t>(fp8_val);
    }
  }

 private:
  cache_t* __restrict__ dst_;
  const scalar_t* __restrict__ src_;
  const float scale_;
  const int64_t numel_;
};

}  // namespace vllm

#define CALL_CONVERT_FP8_KERNEL(SCALAR_T, CACHE_T, KV_DTYPE)   \
  queue.submit([&](sycl::handler& cgh) {                       \
    cgh.parallel_for(                                          \
        sycl::nd_range<1>(grid * block, block),                \
        vllm::convert_fp8_kernel<SCALAR_T, CACHE_T, KV_DTYPE>( \
            reinterpret_cast<CACHE_T*>(dst.data_ptr()),        \
            reinterpret_cast<const SCALAR_T*>(src.data_ptr()), \
            scale,                                             \
            numel));                                           \
  });

// Only for testing.
/**
 * @brief Converts between FP8 and FP16/BF16/FP32 formats with scaling.
 *
 * Supports both quantization (FP16/BF16/FP32 -> FP8) and dequantization
 * (FP8 -> FP16/BF16/FP32) operations. The conversion direction is determined
 * by the kv_cache_dtype parameter: "auto" indicates dequantization, while
 * "fp8_e4m3" or "fp8_e5m2" indicates quantization.
 *
 * @param dst              Destination tensor on XPU device
 * @param src              Source tensor on XPU device (same device as dst)
 * @param scale            Scaling factor for quantization/dequantization.
 *                         For quantize: dst = src / scale
 *                         For dequantize: dst = src * scale
 * @param kv_cache_dtype   Target FP8 format: "fp8_e4m3", "fp8_e5m2", or "auto".
 *                         "auto" indicates dequantization (src is FP8, dst is
 * FP16/BF16/FP32). Other values indicate quantization (src is FP16/BF16/FP32,
 * dst is FP8).
 *
 * @throws std::runtime_error  If src/dst are not on XPU, not on the same XPU
 * device, or if dtype combination is unsupported
 *
 * @note Both tensors must reside on the same XPU device. The kernel is launched
 *       asynchronously on the default SYCL queue; synchronization is caller's
 * responsibility.
 */
void convert_fp8(
    torch::Tensor& dst,
    const torch::Tensor& src,
    const double scale,
    const std::string& kv_cache_dtype) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  TORCH_CHECK(src_device.is_xpu(), "src must be on a XPU");
  TORCH_CHECK(dst_device.is_xpu(), "dst must be on a XPU");
  TORCH_CHECK(
      src_device.index() == dst_device.index(),
      "src and dst must be on the same XPU");

  const int64_t numel = src.numel();
  const int threads = 256;
  const int64_t num_blocks = (numel + threads - 1) / threads;

  const at::DeviceGuard device_guard(src.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  sycl::range<1> grid(num_blocks);
  sycl::range<1> block(threads);

  // Dispatch based on conversion direction
  // If kv_cache_dtype is "auto", we're dequantizing (FP8 -> FP16/BF16/FP32)
  // Otherwise, we're quantizing (FP16/BF16/FP32 -> FP8)
  if (kv_cache_dtype == "auto") {
    // Dequantization: src is FP8, dst is FP16/BF16/FP32
    if (dst.scalar_type() == at::ScalarType::Float) {
      if (src.scalar_type() == at::ScalarType::Float8_e4m3fn) {
        CALL_CONVERT_FP8_KERNEL(
            at::Float8_e4m3fn, float, vllm::Fp8KVCacheDataType::kAuto);
      } else if (src.scalar_type() == at::ScalarType::Float8_e5m2) {
        CALL_CONVERT_FP8_KERNEL(
            at::Float8_e5m2, float, vllm::Fp8KVCacheDataType::kAuto);
      } else {
        TORCH_CHECK(
            false,
            "Unsupported src type for dequantization: ",
            src.scalar_type());
      }
    } else if (dst.scalar_type() == at::ScalarType::Half) {
      if (src.scalar_type() == at::ScalarType::Float8_e4m3fn) {
        CALL_CONVERT_FP8_KERNEL(
            at::Float8_e4m3fn, at::Half, vllm::Fp8KVCacheDataType::kAuto);
      } else if (src.scalar_type() == at::ScalarType::Float8_e5m2) {
        CALL_CONVERT_FP8_KERNEL(
            at::Float8_e5m2, at::Half, vllm::Fp8KVCacheDataType::kAuto);
      } else {
        TORCH_CHECK(
            false,
            "Unsupported src type for dequantization: ",
            src.scalar_type());
      }
    } else if (dst.scalar_type() == at::ScalarType::BFloat16) {
      if (src.scalar_type() == at::ScalarType::Float8_e4m3fn) {
        CALL_CONVERT_FP8_KERNEL(
            at::Float8_e4m3fn, at::BFloat16, vllm::Fp8KVCacheDataType::kAuto);
      } else if (src.scalar_type() == at::ScalarType::Float8_e5m2) {
        CALL_CONVERT_FP8_KERNEL(
            at::Float8_e5m2, at::BFloat16, vllm::Fp8KVCacheDataType::kAuto);
      } else {
        TORCH_CHECK(
            false,
            "Unsupported src type for dequantization: ",
            src.scalar_type());
      }
    } else {
      TORCH_CHECK(
          false,
          "Unsupported dst type for dequantization: ",
          dst.scalar_type());
    }
  } else {
    // Quantization: src is FP16/BF16/FP32, dst is FP8
    DISPATCH_BY_KV_CACHE_DTYPE(
        src.scalar_type(), kv_cache_dtype, CALL_CONVERT_FP8_KERNEL);
  }
}
#define CALL_INDEXER_K_QUANT_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)          \
  queue.submit([&](sycl::handler& cgh) {                                 \
    cgh.parallel_for(                                                    \
        sycl::nd_range<2>(grid * block, block),                          \
        vllm::indexer_k_quant_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>( \
            reinterpret_cast<KV_T*>(k.data_ptr()),                       \
            reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),             \
            slot_mapping.data_ptr<int64_t>(),                            \
            head_dim,                                                    \
            quant_block_size,                                            \
            cache_block_size,                                            \
            cache_stride,                                                \
            use_ue8m0));                                                 \
  });

void indexer_k_quant_and_cache(
    torch::Tensor& k,             // [num_tokens, head_dim]
    torch::Tensor& kv_cache,      // [num_blocks, block_size, cache_stride]
    torch::Tensor& slot_mapping,  // [num_tokens]
    int64_t quant_block_size,     // quantization block size
    const std::string& scale_fmt) {
  int num_tokens = k.size(0);
  int head_dim = k.size(1);
  int cache_block_size = kv_cache.size(1);
  int cache_stride = kv_cache.size(2);
  bool use_ue8m0 = scale_fmt == "ue8m0";

  TORCH_CHECK(
      k.device() == kv_cache.device(),
      "k and kv_cache must be on the same device");
  TORCH_CHECK(
      k.device() == slot_mapping.device(),
      "k and slot_mapping must be on the same device");
  TORCH_CHECK(
      head_dim % quant_block_size == 0,
      "head_dim must be divisible by quant_block_size");

  constexpr int vec_size = 4;
  sycl::range<2> grid(
      (head_dim + quant_block_size * vec_size - 1) /
          (quant_block_size * vec_size),
      num_tokens);
  sycl::range<2> block(vec_size, 32);
  const at::DeviceGuard device_guard(k.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  static const std::string kv_cache_dtype = "fp8_e4m3";
  DISPATCH_BY_KV_CACHE_DTYPE(
      k.scalar_type(), kv_cache_dtype, CALL_INDEXER_K_QUANT_AND_CACHE);
}

// Dispatch macro: submit a single SYCL kernel with the given BLOCK_Y_SIZE.
// The local work-group is (BLOCK_Y_SIZE, 8)
#define CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(BLOCK_Y_SIZE)              \
  do {                                                                  \
    sycl::range<2> global(                                              \
        ((head_dim + 8 * vec_size - 1) / (8 * vec_size)),               \
        ((num_tokens + (BLOCK_Y_SIZE) - 1) / (BLOCK_Y_SIZE)));          \
    sycl::range<2> local((BLOCK_Y_SIZE), 8);                            \
    queue.submit([&](sycl::handler& cgh) {                              \
      sycl::local_accessor<int, 1> batch_idx(                           \
          sycl::range<1>((BLOCK_Y_SIZE)), cgh);                         \
      cgh.parallel_for(                                                 \
          sycl::nd_range<2>(global * local, local),                     \
          vllm::cp_gather_indexer_k_quant_cache_kernel<(BLOCK_Y_SIZE)>( \
              reinterpret_cast<const char*>(kv_cache.data_ptr()),       \
              reinterpret_cast<char*>(dst_k.data_ptr()),                \
              reinterpret_cast<char*>(dst_scale.data_ptr()),            \
              block_table.data_ptr<int32_t>(),                          \
              cu_seq_lens.data_ptr<int32_t>(),                          \
              batch_idx,                                                \
              batch_size,                                               \
              dst_k.stride(0),                                          \
              head_dim,                                                 \
              kv_cache.stride(0),                                       \
              kv_cache.size(1),                                         \
              static_cast<int>(block_table.size(1)),                    \
              num_tokens,                                               \
              quant_block_size));                                       \
    });                                                                 \
  } while (0)

void cp_gather_indexer_k_quant_cache(
    const torch::Tensor& kv_cache,  // [num_blocks, block_size, cache_stride]
    torch::Tensor& dst_k,           // [num_tokens, head_dim]
    torch::Tensor& dst_scale,  // [num_tokens, head_dim / quant_block_size * 4]
    const torch::Tensor& block_table,  // [batch_size, num_blocks]
    const torch::Tensor& cu_seq_lens   // [batch_size + 1]
) {
  int batch_size = static_cast<int>(block_table.size(0));
  int num_tokens = static_cast<int>(dst_k.size(0));
  int64_t head_dim = dst_k.size(1);
  int quant_block_size = static_cast<int>(head_dim * 4 / dst_scale.size(1));

  TORCH_CHECK(
      kv_cache.device() == dst_k.device(),
      "kv_cache and dst_k must be on the same device");
  TORCH_CHECK(
      kv_cache.device() == dst_scale.device(),
      "kv_cache and dst_scale must be on the same device");
  TORCH_CHECK(
      kv_cache.device() == block_table.device(),
      "kv_cache and block_table must be on the same device");
  TORCH_CHECK(
      kv_cache.device() == cu_seq_lens.device(),
      "kv_cache and cu_seq_lens must be on the same device");
  TORCH_CHECK(
      head_dim % quant_block_size == 0,
      "head_dim must be divisible by quant_block_size");

  constexpr int vec_size = 16;
  const at::DeviceGuard device_guard(kv_cache.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  // Select BLOCK_Y_SIZE (token parallelism per work-group) based on batch
  // size, mirroring the CUDA heuristic.
  if (num_tokens < 32) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(1);
  } else if (num_tokens < 64) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(2);
  } else if (num_tokens < 128) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(4);
  } else if (num_tokens < 256) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(8);
  } else if (num_tokens < 512) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(16);
  } else {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(32);
  }
}
