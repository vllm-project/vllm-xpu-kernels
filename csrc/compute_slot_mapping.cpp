#include <sycl/sycl.hpp>
#include <cstdint>

#include "dispatch_utils.h"
#include "utils.h"

namespace vllm {

template <
    int TOTAL_CP_WORLD_SIZE,
    int TOTAL_CP_RANK,
    int CP_KV_CACHE_INTERLEAVE_SIZE,
    int64_t PAD_ID>
class ComputeSlotMappingKernel {
 public:
  ComputeSlotMappingKernel(
      int64_t num_tokens,
      int64_t max_num_tokens,
      const int32_t* query_start_loc,
      const int64_t* positions,
      const int32_t* block_table,
      int64_t block_table_stride,
      int32_t block_size,
      int64_t* slot_mapping)
      : num_tokens_(num_tokens),
        max_num_tokens_(max_num_tokens),
        query_start_loc_(query_start_loc),
        positions_(positions),
        block_table_(block_table),
        block_table_stride_(block_table_stride),
        block_size_(block_size),
        slot_mapping_(slot_mapping) {}

  constexpr static int32_t BLOCK_SIZE = 256;

  void operator()(sycl::nd_item<1> item) const {
    const int req_idx = static_cast<int>(item.get_group(0));
    const int num_groups = static_cast<int>(item.get_group_range(0));
    const int lid = static_cast<int>(item.get_local_id(0));

    if (req_idx == num_groups - 1) {
      for (int64_t base = num_tokens_; base < max_num_tokens_;
           base += BLOCK_SIZE) {
        const int64_t off = base + lid;
        if (off < max_num_tokens_) slot_mapping_[off] = PAD_ID;
      }
      return;
    }

    const int64_t start_idx = static_cast<int64_t>(query_start_loc_[req_idx]);
    const int64_t end_idx = static_cast<int64_t>(query_start_loc_[req_idx + 1]);

    const int32_t virtual_block_size = block_size_ * TOTAL_CP_WORLD_SIZE;
    const int64_t row_offset =
        static_cast<int64_t>(req_idx) * block_table_stride_;

    for (int64_t base = start_idx; base < end_idx; base += BLOCK_SIZE) {
      const int64_t off = base + lid;
      if (off >= end_idx) continue;

      const int64_t pos = positions_[off];

      const int64_t block_index = pos / virtual_block_size;
      const int64_t block_number =
          static_cast<int64_t>(block_table_[row_offset + block_index]);
      const int64_t virtual_block_offset =
          pos - block_index * virtual_block_size;

      const bool is_local =
          ((virtual_block_offset / CP_KV_CACHE_INTERLEAVE_SIZE) %
           TOTAL_CP_WORLD_SIZE) == TOTAL_CP_RANK;

      const int64_t local_block_offset =
          (virtual_block_offset /
           (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)) *
              CP_KV_CACHE_INTERLEAVE_SIZE +
          (virtual_block_offset % CP_KV_CACHE_INTERLEAVE_SIZE);

      const int64_t slot_id = block_number * block_size_ + local_block_offset;

      slot_mapping_[off] = is_local ? slot_id : PAD_ID;
    }
  }

 private:
  int64_t num_tokens_;
  int64_t max_num_tokens_;
  const int32_t* query_start_loc_;
  const int64_t* positions_;
  const int32_t* block_table_;
  int64_t block_table_stride_;
  int32_t block_size_;
  int64_t* slot_mapping_;
};

}  // namespace vllm

#define VLLM_DISPATCH_CP_WORLD_SIZE(WS, CONST_WS, ...)       \
  switch (WS) {                                              \
    case 1: {                                                \
      constexpr int CONST_WS = 1;                            \
      __VA_ARGS__();                                         \
      break;                                                 \
    }                                                        \
    case 2: {                                                \
      constexpr int CONST_WS = 2;                            \
      __VA_ARGS__();                                         \
      break;                                                 \
    }                                                        \
    case 4: {                                                \
      constexpr int CONST_WS = 4;                            \
      __VA_ARGS__();                                         \
      break;                                                 \
    }                                                        \
    case 8: {                                                \
      constexpr int CONST_WS = 8;                            \
      __VA_ARGS__();                                         \
      break;                                                 \
    }                                                        \
    default:                                                 \
      TORCH_CHECK(false, "Unsupported cp_world_size: ", WS); \
  }

#define VLLM_DISPATCH_CP_RANK(RANK, CONST_WS, CONST_RANK, ...)                 \
  switch (RANK) {                                                              \
    case 0: {                                                                  \
      constexpr int CONST_RANK = 0;                                            \
      __VA_ARGS__();                                                           \
      break;                                                                   \
    }                                                                          \
    case 1: {                                                                  \
      if constexpr (CONST_WS > 1) {                                            \
        constexpr int CONST_RANK = 1;                                          \
        __VA_ARGS__();                                                         \
      }                                                                        \
      break;                                                                   \
    }                                                                          \
    case 2: {                                                                  \
      if constexpr (CONST_WS > 2) {                                            \
        constexpr int CONST_RANK = 2;                                          \
        __VA_ARGS__();                                                         \
      }                                                                        \
      break;                                                                   \
    }                                                                          \
    case 3: {                                                                  \
      if constexpr (CONST_WS > 3) {                                            \
        constexpr int CONST_RANK = 3;                                          \
        __VA_ARGS__();                                                         \
      }                                                                        \
      break;                                                                   \
    }                                                                          \
    case 4: {                                                                  \
      if constexpr (CONST_WS > 4) {                                            \
        constexpr int CONST_RANK = 4;                                          \
        __VA_ARGS__();                                                         \
      }                                                                        \
      break;                                                                   \
    }                                                                          \
    case 5: {                                                                  \
      if constexpr (CONST_WS > 5) {                                            \
        constexpr int CONST_RANK = 5;                                          \
        __VA_ARGS__();                                                         \
      }                                                                        \
      break;                                                                   \
    }                                                                          \
    case 6: {                                                                  \
      if constexpr (CONST_WS > 6) {                                            \
        constexpr int CONST_RANK = 6;                                          \
        __VA_ARGS__();                                                         \
      }                                                                        \
      break;                                                                   \
    }                                                                          \
    case 7: {                                                                  \
      if constexpr (CONST_WS > 7) {                                            \
        constexpr int CONST_RANK = 7;                                          \
        __VA_ARGS__();                                                         \
      }                                                                        \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      TORCH_CHECK(                                                             \
          false, "Unsupported cp_rank: ", RANK, " for world_size=", CONST_WS); \
  }

#define VLLM_DISPATCH_CP_INTERLEAVE(IL, CONST_IL, ...)                     \
  switch (IL) {                                                            \
    case 1: {                                                              \
      constexpr int CONST_IL = 1;                                          \
      __VA_ARGS__();                                                       \
      break;                                                               \
    }                                                                      \
    case 16: {                                                             \
      constexpr int CONST_IL = 16;                                         \
      __VA_ARGS__();                                                       \
      break;                                                               \
    }                                                                      \
    case 64: {                                                             \
      constexpr int CONST_IL = 64;                                         \
      __VA_ARGS__();                                                       \
      break;                                                               \
    }                                                                      \
    default:                                                               \
      TORCH_CHECK(false, "Unsupported cp_kv_cache_interleave_size: ", IL); \
  }

void compute_slot_mapping(
    int64_t num_reqs,
    int64_t num_tokens,
    int64_t max_num_tokens,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& positions,
    const torch::Tensor& block_table,
    int64_t block_table_stride,
    int64_t block_size,
    torch::Tensor& slot_mapping,
    int64_t total_cp_world_size,
    int64_t total_cp_rank,
    int64_t cp_kv_cache_interleave_size,
    int64_t pad_id) {
  TORCH_CHECK(query_start_loc.dim() == 1);
  TORCH_CHECK(positions.dim() == 1);
  TORCH_CHECK(block_table.dim() == 2);
  TORCH_CHECK(slot_mapping.dim() == 1);

  TORCH_CHECK(query_start_loc.size(0) == num_reqs + 1);
  TORCH_CHECK(positions.size(0) == num_tokens);
  TORCH_CHECK(
      block_table.size(0) * block_table.size(1) >=
      (num_tokens + block_table_stride - 1) / block_table_stride);
  TORCH_CHECK(slot_mapping.size(0) >= max_num_tokens);

  TORCH_CHECK(query_start_loc.dtype() == torch::kInt32);
  TORCH_CHECK(positions.dtype() == torch::kInt64);
  TORCH_CHECK(block_table.dtype() == torch::kInt32);
  TORCH_CHECK(slot_mapping.dtype() == torch::kInt64);

  TORCH_CHECK(
      pad_id == -1,
      "Only PAD_SLOT_ID=-1 is templated; "
      "got pad_id=",
      pad_id);

  auto& queue = vllm::xpu::vllmGetQueue();

  const int32_t bs = static_cast<int32_t>(block_size);
  const int64_t bt_stride = block_table_stride;

  auto* qsl_ptr = query_start_loc.data_ptr<int32_t>();
  auto* pos_ptr = positions.data_ptr<int64_t>();
  auto* bt_ptr = block_table.data_ptr<int32_t>();
  auto* sm_ptr = slot_mapping.data_ptr<int64_t>();

  VLLM_DISPATCH_CP_WORLD_SIZE(total_cp_world_size, CP_WS, [&] {
    VLLM_DISPATCH_CP_RANK(total_cp_rank, CP_WS, CP_RANK, [&] {
      VLLM_DISPATCH_CP_INTERLEAVE(cp_kv_cache_interleave_size, CP_IL, [&] {
        using Kernel = vllm::
            ComputeSlotMappingKernel<CP_WS, CP_RANK, CP_IL, /*PAD_ID=*/-1>;

        const sycl::range<1> global(
            (num_reqs + 1) * static_cast<size_t>(Kernel::BLOCK_SIZE));
        const sycl::range<1> local(Kernel::BLOCK_SIZE);

        Kernel k(
            num_tokens,
            max_num_tokens,
            qsl_ptr,
            pos_ptr,
            bt_ptr,
            bt_stride,
            bs,
            sm_ptr);

        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(sycl::nd_range<1>(global, local), k);
        });
      });
    });
  });
}
