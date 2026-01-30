
#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>

#include <sycl/sycl.hpp>

#include "utils.h"
#include "dispatch_utils.h"

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

// Round a up to the next multiple of b. The caller is responsible for making
// sure that b is non-zero
template <typename T>
inline constexpr T round_to_next_multiple_of(T a, T b) {
  return a % b == 0 ? a : ((a / b) + 1) * b;
}

namespace vllm {
namespace moe {

constexpr int32_t WARP_SIZE = 32;
constexpr int32_t EXCLUSIVE_SIZE = 1024;

namespace batched_moe_align_block_size {

// Note num_threads needs to be 1024 for BlockScan Reduction in the kernel.
static constexpr int32_t num_threads = 1024;
static constexpr int32_t num_blocks = 1;

class batched_moe_align_block_size_kernel {
 private:
  const int32_t num_batches;
  const int32_t max_tokens_per_batch;
  const int32_t block_size;
  const int32_t* __restrict__ batch_num_tokens;
  int32_t* __restrict__ sorted_ids;
  int32_t* __restrict__ block_ids;
  int32_t* __restrict__ num_tokens_post_pad;
  sycl::local_accessor<int32_t, 1> slm;

 public:
  batched_moe_align_block_size_kernel(
      int32_t num_batches,
      int32_t max_tokens_per_batch,
      int32_t block_size,
      const int32_t* batch_num_tokens,
      int32_t* sorted_ids,
      int32_t* block_ids,
      int32_t* num_tokens_post_pad,
      sycl::local_accessor<int32_t, 1> slm)
      : num_batches(num_batches),
        max_tokens_per_batch(max_tokens_per_batch),
        block_size(block_size),
        batch_num_tokens(batch_num_tokens),
        sorted_ids(sorted_ids),
        block_ids(block_ids),
        num_tokens_post_pad(num_tokens_post_pad),
        slm(slm) {}

  void operator()(sycl::nd_item<1> item) const {
    auto local_id_x = item.get_local_id(0);
    auto local_range_x = item.get_local_range(0);
    auto group_range_x = item.get_group_range(0);

    int32_t* temp_storage = static_cast<int32_t*>(
        slm.template get_multi_ptr<sycl::access::decorated::no>().get());

    // TODO: This is a naive implementation. Could be optimized.

    size_t const batch_id = local_id_x;
    size_t const stride = local_range_x * group_range_x;
    int32_t const num_blocks_per_batch =
        CEILDIV(max_tokens_per_batch, block_size);
    int32_t const sorted_ids_size =
        num_blocks_per_batch * num_batches * block_size;
    int32_t const block_ids_size = sorted_ids_size / block_size;
    int32_t const SENTINEL =
        num_batches * max_tokens_per_batch;  // To denote invalid entries.
    // Initialize sorted_ids
    for (size_t i = local_id_x; i < sorted_ids_size; i += stride) {
      sorted_ids[i] = SENTINEL;
    }
    // Initialize expert_ids with -1
    for (size_t i = local_id_x; i < block_ids_size; i += stride) {
      block_ids[i] = -1;
    }

    int32_t b_num_tokens = 0;
    if (batch_id < num_batches) {
      b_num_tokens = batch_num_tokens[batch_id];
    }
    int32_t const ceil_b_num_tokens =
        CEILDIV(b_num_tokens, block_size) * block_size;

    // Compute prefix sum over token counts per expert
    temp_storage[local_id_x] = ceil_b_num_tokens;
    item.barrier(sycl::access::fence_space::local_space);

    int cumsum_val;
    sycl::joint_exclusive_scan(
        item.get_group(),
        temp_storage,
        temp_storage + local_range_x,
        temp_storage,
        0,
        sycl::plus<int>{});
    cumsum_val = temp_storage[local_id_x];

    bool const is_last_batch = batch_id == (num_batches - 1);
    if (is_last_batch) {
      *num_tokens_post_pad = cumsum_val + ceil_b_num_tokens;
    }

    if (batch_id < num_batches) {
      int32_t const batch_offset = batch_id * max_tokens_per_batch;
      for (size_t i = 0; i < b_num_tokens; ++i) {
        sorted_ids[cumsum_val + i] = batch_offset + i;
      }

      int32_t const block_start = cumsum_val / block_size;
      int32_t const num_blocks = ceil_b_num_tokens / block_size;
      for (size_t i = 0; i < num_blocks; ++i) {
        block_ids[block_start + i] = batch_id;
      }
    }
  }
};

}  // namespace batched_moe_align_block_size

template <typename scalar_t>
void _moe_align_block_size(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ expert_map,
    int32_t num_experts,
    int32_t padded_num_experts,
    int32_t experts_per_warp,
    int32_t block_size,
    size_t numel,
    int32_t* __restrict__ cumsum,
    int32_t max_num_tokens_padded,
    int32_t max_num_m_blocks,
    int32_t model_offset,
    int32_t inactive_expert_id,
    int32_t topk_num,
    int32_t* token_mask,
    bool has_expert_map,
    sycl::local_accessor<int32_t, 1> slm,
    sycl::nd_item<1> item) {
  auto local_id_x = item.get_local_id(0);
  auto group_x = item.get_group(0);
  auto local_range_x = item.get_local_range(0);
  int32_t* temp_storage = static_cast<int32_t*>(
      slm.template get_multi_ptr<sycl::access::decorated::no>().get());

  int32_t* shared_counts = temp_storage + local_range_x;

  // Compute input buffer offsets. Typically these will all be 0, except when
  // using Multi LoRA.
  int sorted_token_ids_offset = max_num_tokens_padded * model_offset;
  int expert_ids_offset = max_num_m_blocks * model_offset;
  int cumsum_offset = (num_experts + 1) * model_offset;

  // Use separate threadblocks to fill sorted_token_ids.
  // This is safe since the current kernel does not use sorted_token_ids.
  if (group_x % 2) {
    // Initialize sorted_token_ids with numel
    for (size_t it = local_id_x; it < max_num_tokens_padded;
         it += local_range_x) {
      sorted_token_ids[sorted_token_ids_offset + it] = numel;
    }
    return;
  }

  const int warp_id = local_id_x / WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;

  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < padded_num_experts) {
      shared_counts[warp_id * experts_per_warp + i] = 0;
    }
  }

  item.barrier(sycl::access::fence_space::local_space);

  const size_t tid = local_id_x;
  const size_t stride = local_range_x;

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = static_cast<int>(topk_ids[i]);
    if (expert_id >= num_experts) {
      continue;
    }
    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      // filter invalid experts
      if (expert_id == -1) continue;
    }
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    int mask = token_mask == nullptr ? 1 : token_mask[i / topk_num];

    sycl::atomic_ref<
        int,
        sycl::memory_order::relaxed,
        sycl::memory_scope::work_group,
        sycl::access::address_space::local_space>
        atomic_count(
            shared_counts[warp_idx * experts_per_warp + expert_offset]);
    atomic_count.fetch_add(mask);
  }

  item.barrier(sycl::access::fence_space::local_space);

  // Compute prefix sum over token counts per expert
  int expert_count = 0;
  int expert_id = local_id_x;
  if (expert_id < num_experts) {
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
    expert_count = CEILDIV(expert_count, block_size) * block_size;
  }

  temp_storage[local_id_x] = expert_count;
  item.barrier(sycl::access::fence_space::local_space);

  int cumsum_val;
  sycl::joint_exclusive_scan(
      item.get_group(),
      temp_storage,
      temp_storage + local_range_x,
      temp_storage,
      0,
      sycl::plus<int>{});
  cumsum_val = temp_storage[local_id_x];

  if (expert_id <= num_experts) {
    cumsum[cumsum_offset + expert_id] = cumsum_val;
  }

  if (expert_id == num_experts) {
    total_tokens_post_pad[model_offset] = cumsum_val;
  }

  item.barrier(sycl::access::fence_space::local_space);

  if (local_id_x < num_experts) {
    for (int i = cumsum[cumsum_offset + local_id_x];
         i < cumsum[cumsum_offset + local_id_x + 1];
         i += block_size) {
      expert_ids[expert_ids_offset + i / block_size] = local_id_x;
    }
  }

  // Fill remaining expert_ids with inactive_expert_id
  const size_t fill_start_idx =
      cumsum[cumsum_offset + num_experts] / block_size + local_id_x;
  const size_t expert_ids_size = CEILDIV(max_num_tokens_padded, block_size);
  for (size_t i = fill_start_idx; i < max_num_m_blocks; i += local_range_x) {
    expert_ids[expert_ids_offset + i] = inactive_expert_id;
  }
}

template <typename scalar_t, int32_t fill_threads>
void _moe_align_block_size_small_batch_expert(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    const int32_t* __restrict__ expert_map,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    int32_t max_num_tokens_padded,
    int32_t max_num_m_blocks,
    int32_t inactive_expert_id,
    int32_t model_offset,
    int32_t topk_num,
    int32_t* __restrict__ token_mask,
    bool has_expert_map,
    sycl::local_accessor<int32_t, 1> slm,
    sycl::nd_item<1> item) {
  auto local_id_x = item.get_local_id(0);
  auto local_range_x = item.get_local_range(0);
  // Compute input buffer offsets. Typically these will all be 0, except when
  // using Multi LoRA.
  int sorted_token_ids_offset = max_num_tokens_padded * model_offset;
  int expert_ids_offset = max_num_m_blocks * model_offset;

  // Use an additional group of threads to fill sorted_token_ids.
  // Since the current kernel will use sorted_token_ids afterward,
  // we fill sorted_token_ids within the same threadblock to make
  // synchronization easier.
  if (local_id_x < fill_threads) {
    // Initialize sorted_token_ids with numel
    for (size_t it = local_id_x; it < max_num_tokens_padded;
         it += fill_threads) {
      sorted_token_ids[sorted_token_ids_offset + it] =
          static_cast<int32_t>(numel);
    }

    item.barrier(sycl::access::fence_space::local_space);
    return;
  }

  const size_t tid = local_id_x - fill_threads;
  const size_t stride = local_range_x - fill_threads;

  void* slm_ptr = static_cast<void*>(
      slm.template get_multi_ptr<sycl::access::decorated::no>().get());
  int32_t* cumsum = reinterpret_cast<int32_t*>(slm_ptr);
  int32_t* tokens_cnts = cumsum + num_experts + 1;

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(tid + 1) * num_experts + i] = 0;
  }

  item.barrier(sycl::access::fence_space::local_space);

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = static_cast<int32_t>(topk_ids[i]);
    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      // filter invalid expert
      if (expert_id == -1) continue;
    }
    int mask = token_mask == nullptr ? 1 : token_mask[i / topk_num];
    tokens_cnts[(tid + 1) * num_experts + expert_id] += mask;
  }

  item.barrier(sycl::access::fence_space::local_space);

  if (tid < num_experts) {
    tokens_cnts[tid] = 0;
    for (int i = 1; i <= stride; ++i) {
      tokens_cnts[i * num_experts + tid] +=
          tokens_cnts[(i - 1) * num_experts + tid];
    }
  }

  item.barrier(sycl::access::fence_space::local_space);

  if (tid == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] =
          cumsum[i - 1] +
          CEILDIV(tokens_cnts[stride * num_experts + i - 1], block_size) *
              block_size;
    }
    total_tokens_post_pad[model_offset] =
        static_cast<int32_t>(cumsum[num_experts]);
  }

  item.barrier(sycl::access::fence_space::local_space);

  if (tid < num_experts) {
    for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
      expert_ids[expert_ids_offset + i / block_size] =
          static_cast<int32_t>(tid);
    }
  }

  // Fill remaining expert_ids with inactive_expert_id
  const size_t fill_start_idx = cumsum[num_experts] / block_size + tid;
  for (size_t i = fill_start_idx; i < max_num_m_blocks; i += stride) {
    expert_ids[expert_ids_offset + i] = inactive_expert_id;
  }

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = static_cast<int32_t>(topk_ids[i]);
    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      // filter invalid expert
      if (expert_id == -1) continue;
    }
    int32_t rank_post_pad =
        tokens_cnts[tid * num_experts + expert_id] + cumsum[expert_id];

    if (token_mask == nullptr || token_mask[i / topk_num]) {
      sorted_token_ids[sorted_token_ids_offset + rank_post_pad] =
          static_cast<int32_t>(i);
      ++tokens_cnts[tid * num_experts + expert_id];
    }
  }
}

template <typename scalar_t>
void _count_and_sort_expert_tokens(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    int32_t* __restrict__ expert_map,
    size_t numel,
    int32_t num_experts,
    int32_t max_num_tokens_padded,
    int32_t* __restrict__ token_mask,
    int32_t model_offset,
    int32_t topk_num,
    bool has_expert_map,
    sycl::nd_item<2> item) {
  auto group_y = item.get_group(1);
  auto local_id_x = item.get_local_id(0);
  auto local_range_x = item.get_local_range(0);
  auto group_range_y = item.get_group_range(1);

  const size_t tid = group_y * local_range_x + local_id_x;
  const size_t stride = local_range_x * group_range_y;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = static_cast<int32_t>(topk_ids[i]);
    if (expert_id >= num_experts) {
      continue;
    }

    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      // filter invalid experts
      if (expert_id == -1) continue;
    }

    if (token_mask == nullptr || token_mask[i / topk_num]) {
      int32_t* cumsum_ptr =
          &cumsum_buffer[(model_offset * (num_experts + 1)) + expert_id];

      sycl::atomic_ref<
          int32_t,
          sycl::memory_order::relaxed,
          sycl::memory_scope::device,
          sycl::access::address_space::global_space>
          atomic_count(*cumsum_ptr);

      int32_t rank_post_pad = atomic_count.fetch_add(1);

      sorted_token_ids[max_num_tokens_padded * model_offset + rank_post_pad] =
          static_cast<int32_t>(i);
    }
  }
}

template <typename scalar_t>
class moe_align_block_size_kernel {
 private:
  const scalar_t* __restrict__ topk_ids;
  int32_t* __restrict__ sorted_token_ids;
  int32_t* __restrict__ expert_ids;
  int32_t* __restrict__ total_tokens_post_pad;
  int32_t* __restrict__ expert_map;
  int32_t num_experts;
  int32_t padded_num_experts;
  int32_t experts_per_warp;
  int32_t block_size;
  size_t numel;
  int32_t* __restrict__ cumsum;
  int32_t max_num_tokens_padded;
  int32_t topk_num;
  bool has_expert_map;
  sycl::local_accessor<int32_t, 1> slm;

 public:
  moe_align_block_size_kernel(
      const scalar_t* topk_ids,
      int32_t* sorted_token_ids,
      int32_t* expert_ids,
      int32_t* total_tokens_post_pad,
      int32_t* expert_map,
      int32_t num_experts,
      int32_t padded_num_experts,
      int32_t experts_per_warp,
      int32_t block_size,
      size_t numel,
      int32_t* cumsum,
      int32_t max_num_tokens_padded,
      int32_t topk_num,
      bool has_expert_map,
      sycl::local_accessor<int32_t, 1> slm)
      : topk_ids(topk_ids),
        sorted_token_ids(sorted_token_ids),
        expert_ids(expert_ids),
        total_tokens_post_pad(total_tokens_post_pad),
        expert_map(expert_map),
        num_experts(num_experts),
        padded_num_experts(padded_num_experts),
        experts_per_warp(experts_per_warp),
        block_size(block_size),
        numel(numel),
        cumsum(cumsum),
        max_num_tokens_padded(max_num_tokens_padded),
        topk_num(topk_num),
        has_expert_map(has_expert_map),
        slm(slm) {}

  void operator()(sycl::nd_item<1> item) const {
    _moe_align_block_size<scalar_t>(
        topk_ids,
        sorted_token_ids,
        expert_ids,
        total_tokens_post_pad,
        expert_map,
        num_experts,
        padded_num_experts,
        experts_per_warp,
        block_size,
        numel,
        cumsum,
        max_num_tokens_padded,
        CEILDIV(max_num_tokens_padded, block_size),
        0,
        0,
        topk_num,
        nullptr,
        has_expert_map,
        slm,
        item);
  }
};

template <typename scalar_t>
class count_and_sort_expert_tokens_kernel {
 private:
  const scalar_t* __restrict__ topk_ids;
  int32_t* __restrict__ sorted_token_ids;
  int32_t* __restrict__ cumsum_buffer;
  int32_t* __restrict__ expert_map;
  size_t numel;
  int32_t num_experts;
  int32_t max_num_tokens_padded;
  int32_t topk_num;
  bool has_expert_map;

 public:
  count_and_sort_expert_tokens_kernel(
      const scalar_t* topk_ids,
      int32_t* sorted_token_ids,
      int32_t* cumsum_buffer,
      int32_t* expert_map,
      size_t numel,
      int32_t num_experts,
      int32_t max_num_tokens_padded,
      int32_t topk_num,
      bool has_expert_map)
      : topk_ids(topk_ids),
        sorted_token_ids(sorted_token_ids),
        cumsum_buffer(cumsum_buffer),
        expert_map(expert_map),
        numel(numel),
        num_experts(num_experts),
        max_num_tokens_padded(max_num_tokens_padded),
        topk_num(topk_num),
        has_expert_map(has_expert_map) {}

  void operator()(sycl::nd_item<2> item) const {
    _count_and_sort_expert_tokens(
        topk_ids,
        sorted_token_ids,
        cumsum_buffer,
        expert_map,
        numel,
        num_experts,
        max_num_tokens_padded,
        nullptr,
        0,
        topk_num,
        has_expert_map,
        item);
  }
};

template <typename scalar_t, int TOPK>
class moe_sum_kernel {
 private:
  scalar_t* output;       // [..., d]
  const scalar_t* input;  // [..., topk, d]
  int d;

 public:
  moe_sum_kernel(scalar_t* output, const scalar_t* input, int d)
      : output(output), input(input), d(d) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t token_idx = item.get_group(0);
    for (int64_t idx = item.get_local_id(0); idx < d;
         idx += item.get_local_range(0)) {
      scalar_t x = 0.0;
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        x += input[token_idx * TOPK * d + k * d + idx];
      }
      output[token_idx * d + idx] = x;
    }
  }
};

template <typename scalar_t, int32_t fill_threads>
class moe_align_block_size_small_batch_expert_kernel {
 private:
  const scalar_t* __restrict__ topk_ids;
  int32_t* __restrict__ sorted_token_ids;
  int32_t* __restrict__ expert_ids;
  int32_t* __restrict__ total_tokens_post_pad;
  int32_t* __restrict__ expert_map;
  int32_t num_experts;
  int32_t block_size;
  size_t numel;
  int32_t max_num_tokens_padded;
  int32_t topk_num;
  bool has_expert_map;
  sycl::local_accessor<int32_t, 1> slm;

 public:
  moe_align_block_size_small_batch_expert_kernel(
      const scalar_t* topk_ids,
      int32_t* sorted_token_ids,
      int32_t* expert_ids,
      int32_t* total_tokens_post_pad,
      int32_t* expert_map,
      int32_t num_experts,
      int32_t block_size,
      size_t numel,
      int32_t max_num_tokens_padded,
      int32_t topk_num,
      bool has_expert_map,
      sycl::local_accessor<int32_t, 1> slm)
      : topk_ids(topk_ids),
        sorted_token_ids(sorted_token_ids),
        expert_ids(expert_ids),
        total_tokens_post_pad(total_tokens_post_pad),
        expert_map(expert_map),
        num_experts(num_experts),
        block_size(block_size),
        numel(numel),
        max_num_tokens_padded(max_num_tokens_padded),
        topk_num(topk_num),
        has_expert_map(has_expert_map),
        slm(slm) {}

  void operator()(sycl::nd_item<1> item) const {
    _moe_align_block_size_small_batch_expert<scalar_t, fill_threads>(
        topk_ids,
        sorted_token_ids,
        expert_ids,
        total_tokens_post_pad,
        expert_map,
        num_experts,
        block_size,
        numel,
        max_num_tokens_padded,
        CEILDIV(max_num_tokens_padded, block_size),
        0,
        0,
        topk_num,
        nullptr,
        has_expert_map,
        slm,
        item);
  }
};

template <typename scalar_t>
class moe_lora_align_block_size_kernel {
 private:
  const scalar_t* __restrict__ topk_ids;
  int32_t* __restrict__ token_lora_mapping;
  int64_t block_size;
  int32_t* __restrict__ expert_map;
  int num_experts;
  int max_loras;
  size_t numel;
  int max_num_tokens_padded;
  int max_num_m_blocks;
  int32_t* __restrict__ sorted_token_ids;
  int32_t* __restrict__ expert_ids;
  int32_t topk_num;
  int32_t* total_tokens_post_pad;
  int32_t* adapter_enabled;
  int32_t* __restrict__ cumsum;
  int32_t experts_per_warp;
  int32_t padded_num_experts;
  int32_t* lora_ids;
  int32_t* __restrict__ token_mask;
  bool has_expert_map;
  sycl::local_accessor<int32_t, 1> slm;

 public:
  moe_lora_align_block_size_kernel(
      const scalar_t* topk_ids,
      int32_t* token_lora_mapping,
      int64_t block_size,
      int32_t* expert_map,
      int num_experts,
      int max_loras,
      size_t numel,
      int max_num_tokens_padded,
      int max_num_m_blocks,
      int32_t* sorted_token_ids,
      int32_t* expert_ids,
      int32_t topk_num,
      int32_t* total_tokens_post_pad,
      int32_t* adapter_enabled,
      int32_t* cumsum,
      int32_t experts_per_warp,
      int32_t padded_num_experts,
      int32_t* lora_ids,
      int32_t* token_mask,
      bool has_expert_map,
      sycl::local_accessor<int32_t, 1> slm)
      : topk_ids(topk_ids),
        token_lora_mapping(token_lora_mapping),
        block_size(block_size),
        expert_map(expert_map),
        num_experts(num_experts),
        max_loras(max_loras),
        numel(numel),
        max_num_tokens_padded(max_num_tokens_padded),
        max_num_m_blocks(max_num_m_blocks),
        sorted_token_ids(sorted_token_ids),
        expert_ids(expert_ids),
        topk_num(topk_num),
        total_tokens_post_pad(total_tokens_post_pad),
        adapter_enabled(adapter_enabled),
        cumsum(cumsum),
        experts_per_warp(experts_per_warp),
        padded_num_experts(padded_num_experts),
        lora_ids(lora_ids),
        token_mask(token_mask),
        has_expert_map(has_expert_map),
        slm(slm) {}

  void operator()(sycl::nd_item<1> item) const {
    auto group_x = item.get_group(0);
    int lora_idx = group_x / 2;
    int lora_id = lora_ids[lora_idx];
    if (lora_id == -1 || adapter_enabled[lora_id] == 0) {
      return;
    }

    // Populate the token_mask based on the token-LoRA mapping
    int num_tokens = numel / topk_num;
    if (item.get_local_id(0) == 0) {
      total_tokens_post_pad[lora_id] = 0;

      for (int i = 0; i < num_tokens; i++) {
        token_mask[(lora_id * num_tokens) + i] =
            static_cast<int>(token_lora_mapping[i]) == lora_id;
      }
    }

    item.barrier(sycl::access::fence_space::local_space);

    _moe_align_block_size(
        topk_ids,
        sorted_token_ids,
        expert_ids,
        total_tokens_post_pad,
        expert_map,
        num_experts,
        padded_num_experts,
        experts_per_warp,
        block_size,
        numel,
        cumsum,
        max_num_tokens_padded,
        max_num_m_blocks,
        lora_id,
        -1,
        topk_num,
        &token_mask[(lora_id * num_tokens)],
        has_expert_map,
        slm,
        item);
  }
};

template <typename scalar_t>
class lora_count_and_sort_expert_tokens_kernel {
 private:
  const scalar_t* __restrict__ topk_ids;
  int32_t* __restrict__ sorted_token_ids;
  int32_t* __restrict__ cumsum_buffer;
  int32_t* __restrict__ expert_map;
  size_t numel;
  int32_t num_experts;
  int32_t max_num_tokens_padded;
  int32_t topk_num;
  int32_t* token_mask;
  int32_t* lora_ids;
  bool has_expert_map;

 public:
  lora_count_and_sort_expert_tokens_kernel(
      const scalar_t* topk_ids,
      int32_t* sorted_token_ids,
      int32_t* cumsum_buffer,
      int32_t* expert_map,
      size_t numel,
      int32_t num_experts,
      int32_t max_num_tokens_padded,
      int32_t topk_num,
      int32_t* token_mask,
      int32_t* lora_ids,
      bool has_expert_map)
      : topk_ids(topk_ids),
        sorted_token_ids(sorted_token_ids),
        cumsum_buffer(cumsum_buffer),
        expert_map(expert_map),
        numel(numel),
        num_experts(num_experts),
        max_num_tokens_padded(max_num_tokens_padded),
        topk_num(topk_num),
        token_mask(token_mask),
        lora_ids(lora_ids),
        has_expert_map(has_expert_map) {}

  void operator()(sycl::nd_item<2> item) const {
    int lora_idx = item.get_group(0);
    int lora_id = lora_ids[lora_idx];
    if (lora_id == -1) {
      return;
    }

    int num_tokens = numel / topk_num;

    _count_and_sort_expert_tokens(
        topk_ids,
        sorted_token_ids,
        cumsum_buffer,
        expert_map,
        numel,
        num_experts,
        max_num_tokens_padded,
        &token_mask[(lora_id * num_tokens)],
        lora_id,
        topk_num,
        has_expert_map,
        item);
  }
};

template <typename scalar_t, int32_t fill_threads>
class moe_lora_align_block_size_small_batch_expert_kernel {
 private:
  const scalar_t* __restrict__ topk_ids;
  int32_t* __restrict__ token_lora_mapping;
  int64_t block_size;
  int32_t* __restrict__ expert_map;
  int num_experts;
  int max_loras;
  size_t numel;
  int max_num_tokens_padded;
  int max_num_m_blocks;
  int32_t* __restrict__ sorted_token_ids;
  int32_t* __restrict__ expert_ids;
  int topk_num;
  int32_t* total_tokens_post_pad;
  const int32_t* adapter_enabled;
  const int32_t* lora_ids;
  int32_t* token_mask;
  bool has_expert_map;
  sycl::local_accessor<int32_t, 1> slm;

 public:
  moe_lora_align_block_size_small_batch_expert_kernel(
      const scalar_t* topk_ids,
      int32_t* token_lora_mapping,
      int64_t block_size,
      int32_t* expert_map,
      int num_experts,
      int max_loras,
      size_t numel,
      int max_num_tokens_padded,
      int max_num_m_blocks,
      int32_t* sorted_token_ids,
      int32_t* expert_ids,
      int topk_num,
      int32_t* total_tokens_post_pad,
      const int32_t* adapter_enabled,
      const int32_t* lora_ids,
      int32_t* token_mask,
      bool has_expert_map,
      sycl::local_accessor<int32_t, 1> slm)
      : topk_ids(topk_ids),
        token_lora_mapping(token_lora_mapping),
        block_size(block_size),
        expert_map(expert_map),
        num_experts(num_experts),
        max_loras(max_loras),
        numel(numel),
        max_num_tokens_padded(max_num_tokens_padded),
        max_num_m_blocks(max_num_m_blocks),
        sorted_token_ids(sorted_token_ids),
        expert_ids(expert_ids),
        topk_num(topk_num),
        total_tokens_post_pad(total_tokens_post_pad),
        adapter_enabled(adapter_enabled),
        lora_ids(lora_ids),
        token_mask(token_mask),
        has_expert_map(has_expert_map),
        slm(slm) {}

  void operator()(sycl::nd_item<1> item) const {
    int lora_idx = item.get_group(0);
    int lora_id = lora_ids[lora_idx];
    if (lora_id == -1 || adapter_enabled[lora_id] == 0) {
      return;
    }

    int num_tokens = numel / topk_num;
    if (item.get_local_id(0) == 0) {
      total_tokens_post_pad[lora_id] = 0;

      for (int i = 0; i < num_tokens; i++) {
        token_mask[(lora_id * num_tokens) + i] =
            static_cast<int>(token_lora_mapping[i]) == lora_id;
      }
    }

    item.barrier(sycl::access::fence_space::local_space);

    _moe_align_block_size_small_batch_expert<scalar_t, fill_threads>(
        topk_ids,
        sorted_token_ids,
        expert_ids,
        total_tokens_post_pad,
        expert_map,
        num_experts,
        block_size,
        numel,
        max_num_tokens_padded,
        max_num_m_blocks,
        -1,
        lora_id,
        topk_num,
        &token_mask[(lora_id * num_tokens)],
        has_expert_map,
        slm,
        item);
  }
};

}  // namespace moe
}  // namespace vllm

// taken from
// https://github.com/sgl-project/sglang/blob/8b5f83ed3b7d2a49ad5c5cd5aa61c5d502f47dbc
void moe_align_block_size(
    torch::Tensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    torch::Tensor sorted_token_ids,
    torch::Tensor experts_ids,
    torch::Tensor num_tokens_post_pad,
    std::optional<torch::Tensor> maybe_expert_map) {
  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);
  auto& queue = vllm::xpu::vllmGetQueue();
  size_t device_max_shared_mem =
      queue.get_device().get_info<sycl::info::device::local_mem_size>();

  int64_t padded_num_experts =
      ((num_experts + vllm::moe::WARP_SIZE - 1) / vllm::moe::WARP_SIZE) *
      vllm::moe::WARP_SIZE;
  int experts_per_warp = vllm::moe::WARP_SIZE;
  int threads = 1024;
  threads = ((threads + vllm::moe::WARP_SIZE - 1) / vllm::moe::WARP_SIZE) *
            vllm::moe::WARP_SIZE;

  // BlockScan uses 1024 threads and assigns one thread per expert.
  TORCH_CHECK(
      padded_num_experts < 1024, "padded_num_experts must be less than 1024");

  auto options_int =
      torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
  bool has_expert_map = maybe_expert_map.has_value();
  torch::Tensor expert_map;
  if (has_expert_map) {
    expert_map = maybe_expert_map.value();
  } else {
    expert_map = torch::empty({0}, options_int);
  }

  VLLM_DISPATCH_INTEGRAL_AND_UNSIGNED_TYPES(
      topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
        // calc needed amount of shared mem for `cumsum` tensors
        bool small_batch_expert_mode =
            (topk_ids.numel() < 1024) && (num_experts <= 64);

        if (small_batch_expert_mode) {
          const int32_t threads =
              std::max(static_cast<int32_t>(num_experts), vllm::moe::WARP_SIZE);
          const int32_t shared_mem_size =
              ((threads + 1) * num_experts + (num_experts + 1)) *
              sizeof(int32_t);

          if (shared_mem_size > device_max_shared_mem) {
            TORCH_CHECK(false, "Shared memory usage exceeds device limit.");
          }

          // threadIdx.x >= fill_threads: counting experts and aligning
          // threadIdx.x < fill_threads: filling sorted_token_ids
          constexpr int32_t fill_threads = 256;
          const int32_t total_threads = fill_threads + threads;
          sycl::range<1> local_range(total_threads);
          sycl::range<1> global_range(total_threads);

          queue.submit([&](sycl::handler& h) {
            sycl::local_accessor<int32_t, 1> shared_mem(
                sycl::range<1>(shared_mem_size / sizeof(int32_t)), h);
            vllm::moe::moe_align_block_size_small_batch_expert_kernel<
                scalar_t,
                fill_threads>
                kfn(topk_ids.data_ptr<scalar_t>(),
                    sorted_token_ids.data_ptr<int32_t>(),
                    experts_ids.data_ptr<int32_t>(),
                    num_tokens_post_pad.data_ptr<int32_t>(),
                    expert_map.data_ptr<int32_t>(),
                    num_experts,
                    block_size,
                    topk_ids.numel(),
                    sorted_token_ids.size(0),
                    topk_ids.size(1),
                    has_expert_map,
                    shared_mem);
            h.parallel_for(sycl::nd_range<1>(global_range, local_range), kfn);
          });
        } else {
          const int num_work_groups = 2;
          const int global_size = num_work_groups * threads;

          sycl::range<1> global_range1(global_size);
          sycl::range<1> local_range1(threads);
          torch::Tensor cumsum_buffer =
              torch::empty({num_experts + 1}, options_int);

          size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
          size_t shared_mem_size =
              (threads + num_warps * experts_per_warp) * sizeof(int32_t);

          // launch two threadblocks
          // blockIdx.x == 0: counting experts and aligning
          // blockIdx.x == 1: filling sorted_token_ids
          queue.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<int32_t, 1> slm(
                sycl::range<1>(shared_mem_size / sizeof(int32_t)), cgh);
            vllm::moe::moe_align_block_size_kernel<scalar_t> kfn(
                topk_ids.data_ptr<scalar_t>(),
                sorted_token_ids.data_ptr<int32_t>(),
                experts_ids.data_ptr<int32_t>(),
                num_tokens_post_pad.data_ptr<int32_t>(),
                expert_map.data_ptr<int32_t>(),
                num_experts,
                padded_num_experts,
                experts_per_warp,
                block_size,
                topk_ids.numel(),
                cumsum_buffer.data_ptr<int32_t>(),
                sorted_token_ids.size(0),
                topk_ids.size(1),
                has_expert_map,
                slm);
            cgh.parallel_for(
                sycl::nd_range<1>(global_range1, local_range1), kfn);
          });

          const int block_threads = std::min(256, (int)threads);
          const int num_blocks =
              (topk_ids.numel() + block_threads - 1) / block_threads;
          const int max_blocks = 65535;

          const int actual_blocks = std::min(num_blocks, max_blocks);
          const int num_groups_y = actual_blocks;
          const int global_size_x = block_threads;
          const int global_size_y = num_groups_y;
          sycl::range<2> global_range2(global_size_x, global_size_y);
          sycl::range<2> local_range2(block_threads, 1);

          queue.submit([&](sycl::handler& cgh) {
            vllm::moe::count_and_sort_expert_tokens_kernel<scalar_t> kfn(
                topk_ids.data_ptr<scalar_t>(),
                sorted_token_ids.data_ptr<int32_t>(),
                cumsum_buffer.data_ptr<int32_t>(),
                expert_map.data_ptr<int32_t>(),
                topk_ids.numel(),
                num_experts,
                sorted_token_ids.size(0),
                topk_ids.size(1),
                has_expert_map);
            cgh.parallel_for(
                sycl::nd_range<2>(global_range2, local_range2), kfn);
          });
        }
      });
}

void batched_moe_align_block_size(
    int64_t max_tokens_per_batch,
    int64_t block_size,
    torch::Tensor const& batch_num_tokens,
    torch::Tensor sorted_ids,
    torch::Tensor batch_ids,
    torch::Tensor num_tokens_post_pad) {
  namespace batched_kernel = vllm::moe::batched_moe_align_block_size;

  auto& queue = vllm::xpu::vllmGetQueue();
  int32_t const B = batch_num_tokens.size(0);
  int32_t const num_blocks_per_batch =
      round_to_next_multiple_of(max_tokens_per_batch, block_size) / block_size;
  int32_t const num_blocks = num_blocks_per_batch * B;
  int64_t const sorted_ids_size = num_blocks * block_size;

  TORCH_CHECK(sorted_ids.size(0) == sorted_ids_size);
  TORCH_CHECK(batch_ids.size(0) == sorted_ids_size / block_size);
  TORCH_CHECK(num_tokens_post_pad.size(0) == 1);
  TORCH_CHECK(B <= batched_kernel::num_threads);

  sycl::range<1> global_range(
      batched_kernel::num_blocks * batched_kernel::num_threads);
  sycl::range<1> local_range(batched_kernel::num_threads);

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> slm(
        sycl::range<1>(vllm::moe::EXCLUSIVE_SIZE), cgh);
    batched_kernel::batched_moe_align_block_size_kernel kfn(
        B,
        max_tokens_per_batch,
        block_size,
        batch_num_tokens.data_ptr<int32_t>(),
        sorted_ids.data_ptr<int32_t>(),
        batch_ids.data_ptr<int32_t>(),
        num_tokens_post_pad.data_ptr<int32_t>(),
        slm);
    cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), kfn);
  });
}

void moe_sum(
    torch::Tensor& input,   // [num_tokens, topk, hidden_size]
    torch::Tensor& output)  // [num_tokens, hidden_size]
{
  const int hidden_size = input.size(-1);
  const auto num_tokens = output.numel() / hidden_size;
  const int topk = input.size(1);

  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);
  auto& queue = vllm::xpu::vllmGetQueue();

  const int local_size = std::min(hidden_size, 1024);
  const int global_size = num_tokens * local_size;
  sycl::range<1> global_range(global_size);
  sycl::range<1> local_range(local_size);

  switch (topk) {
    case 2:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(global_range, local_range),
              vllm::moe::moe_sum_kernel<scalar_t, 2>(
                  output.data_ptr<scalar_t>(),
                  input.data_ptr<scalar_t>(),
                  hidden_size));
        });
      });
      break;

    case 3:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(global_range, local_range),
              vllm::moe::moe_sum_kernel<scalar_t, 3>(
                  output.data_ptr<scalar_t>(),
                  input.data_ptr<scalar_t>(),
                  hidden_size));
        });
      });
      break;

    case 4:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(global_range, local_range),
              vllm::moe::moe_sum_kernel<scalar_t, 4>(
                  output.data_ptr<scalar_t>(),
                  input.data_ptr<scalar_t>(),
                  hidden_size));
        });
      });
      break;

    default:
      at::sum_out(output, input, 1);
      break;
  }
}

void moe_lora_align_block_size(
    torch::Tensor topk_ids,
    torch::Tensor token_lora_mapping,
    int64_t num_experts,
    int64_t block_size,
    int64_t max_loras,
    int64_t max_num_tokens_padded,
    int64_t max_num_m_blocks,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad,
    torch::Tensor adapter_enabled,
    torch::Tensor lora_ids,
    std::optional<torch::Tensor> maybe_expert_map) {
  const int topk_num = static_cast<int>(topk_ids.size(1));

  TORCH_CHECK(block_size > 0, "block_size should be greater than 0. ");

  auto& queue = vllm::xpu::vllmGetQueue();

  size_t device_max_shared_mem =
      queue.get_device().get_info<sycl::info::device::local_mem_size>();

  int64_t padded_num_experts =
      ((num_experts + vllm::moe::WARP_SIZE - 1) / vllm::moe::WARP_SIZE) *
      vllm::moe::WARP_SIZE;

  // BlockScan uses 1024 threads and assigns one thread per expert.
  TORCH_CHECK(
      padded_num_experts < 1024, "padded_num_experts must be less than 1024");

  auto options_int =
      torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
  torch::Tensor token_mask =
      torch::empty({max_loras * topk_ids.size(0)}, options_int);
  bool has_expert_map = maybe_expert_map.has_value();
  torch::Tensor expert_map;
  if (has_expert_map) {
    expert_map = maybe_expert_map.value();
  } else {
    expert_map = torch::empty({0}, options_int);
  }

  VLLM_DISPATCH_INTEGRAL_TYPES(
      topk_ids.scalar_type(), "moe_lora_align_sum_kernel", [&] {
        bool small_batch_expert_mode =
            (topk_ids.numel() < 1024) && (num_experts <= 64);

        if (small_batch_expert_mode) {
          const int32_t num_thread =
              std::max(static_cast<int32_t>(num_experts), 128);
          const int32_t shared_mem =
              ((num_thread + 1) * num_experts + (num_experts + 1)) *
              sizeof(int32_t);

          if (shared_mem > device_max_shared_mem) {
            TORCH_CHECK(false, "Shared memory usage exceeds device limit.");
          }

          // threadIdx.x >= fill_threads: counting experts and aligning
          // threadIdx.x < fill_threads: filling sorted_token_ids
          constexpr int32_t fill_threads = 256;
          const int32_t total_threads = num_thread + fill_threads;
          const int32_t global_size = max_loras * total_threads;

          queue.submit([&](sycl::handler& h) {
            auto slm = sycl::local_accessor<int32_t, 1>(
                shared_mem / sizeof(int32_t), h);
            vllm::moe::moe_lora_align_block_size_small_batch_expert_kernel<
                scalar_t,
                fill_threads>
                kfn(topk_ids.data_ptr<scalar_t>(),
                    token_lora_mapping.data_ptr<int32_t>(),
                    block_size,
                    expert_map.data_ptr<int32_t>(),
                    num_experts,
                    max_loras,
                    topk_ids.numel(),
                    max_num_tokens_padded,
                    max_num_m_blocks,
                    sorted_token_ids.data_ptr<int32_t>(),
                    expert_ids.data_ptr<int32_t>(),
                    topk_num,
                    num_tokens_post_pad.data_ptr<int32_t>(),
                    adapter_enabled.data_ptr<int32_t>(),
                    lora_ids.data_ptr<int32_t>(),
                    token_mask.data_ptr<int32_t>(),
                    has_expert_map,
                    slm);
            h.parallel_for(
                sycl::nd_range<1>(
                    sycl::range<1>(global_size), sycl::range<1>(total_threads)),
                kfn);
          });

        } else {
          int num_thread = 1024;
          size_t num_warps = CEILDIV(padded_num_experts, vllm::moe::WARP_SIZE);
          size_t shared_mem_size =
              (num_thread + num_warps * vllm::moe::WARP_SIZE) * sizeof(int32_t);

          // cumsum buffer
          torch::Tensor cumsum =
              torch::zeros({max_loras * (num_experts + 1)}, options_int);
          const int work_groups_per_lora = 2;
          const int total_work_groups = max_loras * work_groups_per_lora;
          const int global_size = total_work_groups * num_thread;
          queue.submit([&](sycl::handler& h) {
            auto slm = sycl::local_accessor<int32_t, 1>(
                shared_mem_size / sizeof(int32_t), h);
            vllm::moe::moe_lora_align_block_size_kernel<scalar_t> kfn(
                topk_ids.data_ptr<scalar_t>(),
                token_lora_mapping.data_ptr<int32_t>(),
                block_size,
                has_expert_map ? expert_map.data_ptr<int32_t>() : nullptr,
                num_experts,
                max_loras,
                topk_ids.numel(),
                max_num_tokens_padded,
                max_num_m_blocks,
                sorted_token_ids.data_ptr<int32_t>(),
                expert_ids.data_ptr<int32_t>(),
                topk_num,
                num_tokens_post_pad.data_ptr<int32_t>(),
                adapter_enabled.data_ptr<int32_t>(),
                cumsum.data_ptr<int32_t>(),
                vllm::moe::WARP_SIZE,
                padded_num_experts,
                lora_ids.data_ptr<int32_t>(),
                token_mask.data_ptr<int32_t>(),
                has_expert_map,
                slm);
            h.parallel_for(
                sycl::nd_range<1>(
                    sycl::range<1>(global_size), sycl::range<1>(num_thread)),
                kfn);
          });

          const int block_threads = std::min(256, num_thread);
          const int num_blocks = static_cast<int>(
              (topk_ids.numel() + block_threads - 1) / block_threads);
          const int max_blocks = 65535;
          const int actual_blocks = std::min(num_blocks, max_blocks);

          const int num_groups_x = max_loras;
          const int num_groups_y = actual_blocks;
          const int global_size_x = block_threads * num_groups_x;
          const int global_size_y = num_groups_y;
          sycl::range<2> global_range(global_size_x, global_size_y);
          sycl::range<2> local_range(block_threads, 1);

          queue.submit([&](sycl::handler& h) {
            vllm::moe::lora_count_and_sort_expert_tokens_kernel<scalar_t> kfn(
                topk_ids.data_ptr<scalar_t>(),
                sorted_token_ids.data_ptr<int32_t>(),
                cumsum.data_ptr<int32_t>(),
                expert_map.data_ptr<int32_t>(),
                topk_ids.numel(),
                num_experts,
                max_num_tokens_padded,
                topk_num,
                token_mask.data_ptr<int32_t>(),
                lora_ids.data_ptr<int32_t>(),
                has_expert_map);
            h.parallel_for(sycl::nd_range<2>(global_range, local_range), kfn);
          });
        }
      });
}
