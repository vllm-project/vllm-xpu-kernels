#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"

namespace vllm {
namespace moe {

class RowsPerExpertCount {
 public:
  RowsPerExpertCount(
      int* expert_map,
      int* rows_per_expert,
      void* topk_ids,
      bool is_topk_ids_int32,
      int* unpermuted_row_to_permuted_row,
      const int num_rows,
      const int TopK,
      const int local_experts_num,
      sycl::local_accessor<int32_t, 1> local_counts)
      : expert_map(expert_map),
        rows_per_expert(rows_per_expert),
        topk_ids(topk_ids),
        is_topk_ids_int32(is_topk_ids_int32),
        unpermuted_row_to_permuted_row(unpermuted_row_to_permuted_row),
        num_rows(num_rows),
        TopK(TopK),
        local_experts_num(local_experts_num),
        local_counts(local_counts) {}

  static constexpr int GroupWorkItem = 256;
  static constexpr int WARP_SIZE = 32;

  static inline sycl::nd_range<1>
  get_nd_range(const int num_rows, const int TopK) {
    int group_nums = (num_rows * TopK + GroupWorkItem - 1) / GroupWorkItem;
    sycl::range<1> local(GroupWorkItem);
    sycl::range<1> group(group_nums);
    return sycl::nd_range<1>(local * group, local);
  }

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    auto global_id = item.get_global_linear_id();
    auto local_id = item.get_local_id(0);
    auto local_range = item.get_local_range(0);

    // ===== Phase 1: init SLM =====
    for (int i = local_id; i < local_experts_num; i += local_range) {
      local_counts[i] = 0;
    }
    item.barrier(sycl::access::fence_space::local_space);

    // ===== Phase 2: local atomic =====
    if (global_id < num_rows * TopK) {
      int global_expert_id =
          is_topk_ids_int32 ? reinterpret_cast<int32_t*>(topk_ids)[global_id]
                            : reinterpret_cast<int64_t*>(topk_ids)[global_id];
      int local_expert_id = global_expert_id;
      if (expert_map != nullptr) {
        local_expert_id = expert_map[global_expert_id];
      }

      if (local_expert_id == -1) {
        unpermuted_row_to_permuted_row[global_id] = -1;
      } else {
        auto local_atomic = sycl::atomic_ref<
            int,
            sycl::memory_order_relaxed,
            sycl::memory_scope_work_group,
            sycl::access::address_space::local_space>(
            local_counts[local_expert_id]);
        int local_old = local_atomic.fetch_add(1);

        unpermuted_row_to_permuted_row[global_id] = local_old;
      }
    }

    item.barrier(sycl::access::fence_space::local_space);

    // ===== Phase 3: global atomic =====
    for (int i = local_id; i < local_experts_num; i += local_range) {
      int count = local_counts[i];
      if (count > 0) {
        auto global_atomic = sycl::atomic_ref<
            int,
            sycl::memory_order_relaxed,
            sycl::memory_scope_device,
            sycl::access::address_space::global_space>(rows_per_expert[i]);
        int base = global_atomic.fetch_add(count);
        local_counts[i] = base;
      }
    }

    item.barrier(sycl::access::fence_space::local_space);

    // ===== Phase 4: fix unpermuted_row_to_permuted_row =====
    if (global_id < num_rows * TopK) {
      int global_expert_id =
          is_topk_ids_int32 ? reinterpret_cast<int32_t*>(topk_ids)[global_id]
                            : reinterpret_cast<int64_t*>(topk_ids)[global_id];
      int local_expert_id = global_expert_id;
      if (expert_map != nullptr) {
        local_expert_id = expert_map[global_expert_id];
      }

      if (local_expert_id != -1) {
        // local_old + base_offset = global_offset
        unpermuted_row_to_permuted_row[global_id] +=
            local_counts[local_expert_id];
      }
    }
  }

 private:
  int* expert_map;
  int* rows_per_expert;
  void* topk_ids;
  bool is_topk_ids_int32;
  int* unpermuted_row_to_permuted_row;
  const int num_rows;
  const int TopK;
  const int local_experts_num;
  sycl::local_accessor<int32_t, 1> local_counts;
};

template <typename TA, typename TS, int TopK>
class RemapHiddenStates {
 public:
  RemapHiddenStates(
      sycl::local_accessor<int32_t, 1>& slm,
      TA* hidden_states,
      TS* hidden_states_scales,
      TA* remapped_hidden_states,
      TS* remapped_hidden_states_scales,
      int* expert_map,
      int* unpermuted_row_to_permuted_row,
      int* rows_per_expert,
      void* topk_ids,
      bool is_topk_ids_int32,
      const int num_rows,
      const int hidden_size,
      const int block_k,
      const int total_experts_num,
      const int local_experts_num)
      : slm(slm),
        hidden_states(hidden_states),
        hidden_states_scales(hidden_states_scales),
        remapped_hidden_states(remapped_hidden_states),
        remapped_hidden_states_scales(remapped_hidden_states_scales),
        expert_map(expert_map),
        unpermuted_row_to_permuted_row(unpermuted_row_to_permuted_row),
        rows_per_expert(rows_per_expert),
        topk_ids(topk_ids),
        is_topk_ids_int32(is_topk_ids_int32),
        num_rows(num_rows),
        hidden_size(hidden_size),
        block_k(block_k),
        total_experts_num(total_experts_num),
        local_experts_num(local_experts_num) {}

  static constexpr int GroupWorkItem = 256;
  static constexpr int WARP_SIZE = 16;
  static constexpr int ElemsPerItem = sizeof(float) * 4 / sizeof(TA);
  static constexpr int EXCLUSIVE_SIZE = 1024;

  static inline sycl::nd_range<1>
  get_nd_range(const int num_rows, const int hidden_size) {
    int local_num = GroupWorkItem;
    if (local_num * ElemsPerItem > hidden_size) {
      local_num = (hidden_size + ElemsPerItem - 1) / ElemsPerItem;
    }
    sycl::range<1> local(local_num);
    sycl::range<1> group(num_rows);
    return sycl::nd_range<1>(local * group, local);
  }

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    auto local_id = item.get_local_id(0);
    auto local_range = item.get_local_range(0);
    auto group_id = item.get_group(0);

    int32_t* expert_cumsum_ptr = static_cast<int32_t*>(
        slm.template get_multi_ptr<sycl::access::decorated::no>().get());

    if (local_id == 0) {
      expert_cumsum_ptr[0] = 0;
    }
    for (int i = local_id; i < local_experts_num - 1; i += local_range) {
      expert_cumsum_ptr[i + 1] = rows_per_expert[i];
    }

    item.barrier(sycl::access::fence_space::local_space);

    sycl::joint_inclusive_scan(
        item.get_group(),
        expert_cumsum_ptr,
        expert_cumsum_ptr + local_experts_num,
        expert_cumsum_ptr,
        sycl::plus<int>{});

    int row = group_id;
    int global_expert_id[TopK];
    int local_expert_id[TopK];

    if (is_topk_ids_int32) {
      auto topk_ids_32 = reinterpret_cast<int32_t*>(topk_ids);
#pragma unroll
      for (int i = 0; i < TopK; ++i) {
        global_expert_id[i] = topk_ids_32[row * TopK + i];
      }
    } else {
      auto topk_ids_64 = reinterpret_cast<int64_t*>(topk_ids);
#pragma unroll
      for (int i = 0; i < TopK; ++i) {
        global_expert_id[i] = topk_ids_64[row * TopK + i];
      }
    }

    if (expert_map != nullptr) {
#pragma unroll
      for (int i = 0; i < TopK; ++i) {
        local_expert_id[i] = expert_map[global_expert_id[i]];
      }
    } else {
#pragma unroll
      for (int i = 0; i < TopK; ++i) {
        local_expert_id[i] = global_expert_id[i];
      }
    }

    int rows_offset[TopK];
#pragma unroll
    for (int i = 0; i < TopK; ++i) {
      if (local_expert_id[i] != -1) {
        int cumsum_offset =
            local_expert_id[i] == 0 ? 0 : expert_cumsum_ptr[local_expert_id[i]];
        rows_offset[i] =
            unpermuted_row_to_permuted_row[row * TopK + i] + cumsum_offset;
      } else {
        rows_offset[i] = -1;
      }
    }

    item.barrier(sycl::access::fence_space::local_space);

    auto hidden_states_base = hidden_states +
                              row * static_cast<int64_t>(hidden_size) +
                              local_id * ElemsPerItem;
    TA* remapped_hidden_states_base[TopK];
#pragma unroll
    for (int i = 0; i < TopK; ++i) {
      remapped_hidden_states_base[i] =
          remapped_hidden_states +
          rows_offset[i] * static_cast<int64_t>(hidden_size) +
          local_id * ElemsPerItem;
    }

    int stride = local_range * ElemsPerItem;
    int loop_count = (hidden_size + stride - 1) / stride;

    for (int l = 0; l < loop_count; ++l) {
      using load_type = sycl::vec<TA, ElemsPerItem>;
      load_type data;
      if (l * stride + local_id * ElemsPerItem < hidden_size) {
        data = *(reinterpret_cast<load_type*>(hidden_states_base + l * stride));
#pragma unroll
        for (int i = 0; i < TopK; ++i) {
          if (rows_offset[i] == -1) continue;
          *(reinterpret_cast<load_type*>(
              remapped_hidden_states_base[i] + l * stride)) = data;
        }
      }
    }

    if (hidden_states_scales != nullptr &&
        remapped_hidden_states_scales != nullptr) {
      int64_t scaled_hidden_size = hidden_size / block_k;
      loop_count = (scaled_hidden_size + stride - 1) / stride;

      for (int l = 0; l < loop_count; ++l) {
        int start_id = l * stride + local_id * ElemsPerItem;
        int remained_elems = scaled_hidden_size - start_id;

        if (remained_elems >= ElemsPerItem) {
          using load_type = sycl::vec<TS, ElemsPerItem>;
          load_type data;
          data = *(reinterpret_cast<load_type*>(
              hidden_states_scales + row * scaled_hidden_size + start_id));
#pragma unroll
          for (int i = 0; i < TopK; ++i) {
            int offset = rows_offset[i];
            if (offset == -1) continue;
            *(reinterpret_cast<load_type*>(
                remapped_hidden_states_scales + offset * scaled_hidden_size +
                start_id)) = data;
          }
        } else if (remained_elems > 0) {
          TS data[ElemsPerItem];
#pragma unroll
          for (int e = 0; e < remained_elems; ++e) {
            data[e] =
                hidden_states_scales[row * scaled_hidden_size + start_id + e];
          }

#pragma unroll
          for (int i = 0; i < TopK; ++i) {
            int offset = rows_offset[i];
            if (offset == -1) continue;
#pragma unroll
            for (int e = 0; e < remained_elems; ++e) {
              remapped_hidden_states_scales
                  [offset * scaled_hidden_size + start_id + e] = data[e];
            }
          }
        }
      }
    }

    if (local_id == 0) {
#pragma unroll
      for (int i = 0; i < TopK; ++i) {
        unpermuted_row_to_permuted_row[row * TopK + i] = rows_offset[i];
      }
    }
  }

 private:
  sycl::local_accessor<int32_t, 1> slm;
  TA* hidden_states;
  TS* hidden_states_scales;
  TA* remapped_hidden_states;
  TS* remapped_hidden_states_scales;
  int* expert_map;
  int* unpermuted_row_to_permuted_row;
  int* rows_per_expert;
  void* topk_ids;
  bool is_topk_ids_int32;
  const int num_rows;
  const int hidden_size;
  const int block_k;
  const int total_experts_num;
  const int local_experts_num;
};

template <typename TA, typename TS, int TopK>
void RemapHiddenStatesLauncher(
    TA* hidden_states,
    TS* hidden_states_scales,
    TA* remapped_hidden_states,
    TS* remapped_hidden_states_scales,
    int* expert_map,
    int* rows_per_expert,
    int* unpermuted_row_to_permuted_row,
    void* topk_ids,
    bool is_topk_ids_int32,
    const int num_rows,
    const int hidden_size,
    const int block_k,
    const int total_experts_num,
    const int local_experts_num,
    sycl::queue& queue) {
  TORCH_CHECK(
      (local_experts_num <= (RemapHiddenStates<TA, TS, TopK>::EXCLUSIVE_SIZE)),
      "local_experts_num exceeds the maximum supported number");
  TORCH_CHECK(
      (hidden_size % RemapHiddenStates<TA, TS, TopK>::ElemsPerItem == 0),
      "hidden_size must be divisible by ElemsPerItem");

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_counts(
        sycl::range<1>(local_experts_num), cgh);
    cgh.parallel_for(
        RowsPerExpertCount::get_nd_range(num_rows, TopK),
        RowsPerExpertCount{
            expert_map,
            rows_per_expert,
            topk_ids,
            is_topk_ids_int32,
            unpermuted_row_to_permuted_row,
            num_rows,
            TopK,
            local_experts_num,
            local_counts});
  });

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> slm(
        sycl::range<1>(local_experts_num), cgh);
    cgh.parallel_for(
        RemapHiddenStates<TA, TS, TopK>::get_nd_range(num_rows, hidden_size),
        RemapHiddenStates<TA, TS, TopK>{
            slm,
            hidden_states,
            hidden_states_scales,
            remapped_hidden_states,
            remapped_hidden_states_scales,
            expert_map,
            unpermuted_row_to_permuted_row,
            rows_per_expert,
            topk_ids,
            is_topk_ids_int32,
            num_rows,
            hidden_size,
            block_k,
            total_experts_num,
            local_experts_num});
  });
}

}  // namespace moe
}  // namespace vllm

void remap_hidden_states(
    torch::Tensor& hidden_states,  // [num_rows, hidden_size]
    const c10::optional<torch::Tensor>&
        hidden_states_scales,  // [num_rows, hidden_size // block_k] or empty
    torch::Tensor& remapped_hidden_states,  // [num_rows * TopK,
                                            // hidden_size]
    const c10::optional<torch::Tensor>&
        remapped_hidden_states_scales,  // [num_rows, hidden_size // block_k] or
                                        // empty
    const c10::optional<torch::Tensor>& expert_map,  // [total_experts_num]
    torch::Tensor& rows_per_expert,                  // [local_experts_num]
    torch::Tensor& unpermuted_row_to_permuted_row,   // [num_rows, TopK]
    torch::Tensor& topk_ids,                         // [num_rows, TopK]
    int64_t total_experts_num,
    int64_t local_experts_num) {
  // dtype check
  TORCH_CHECK(
      hidden_states.scalar_type() == remapped_hidden_states.scalar_type(),
      "hidden_states and remapped_hidden_states must have save dtype");
  if (hidden_states_scales.has_value()) {
    TORCH_CHECK(
        remapped_hidden_states_scales.has_value(),
        "if hidden_states_scales is provided, remapped_hidden_states_scales "
        "must also be provided");
    TORCH_CHECK(
        hidden_states_scales->scalar_type() ==
            remapped_hidden_states_scales->scalar_type(),
        "hidden_states_scales and remapped_hidden_states_scales must have save "
        "dtype");
  }

  if (expert_map.has_value()) {
    TORCH_CHECK(
        expert_map->scalar_type() == torch::kInt32, "expert_map must be int32");
  }

  TORCH_CHECK(
      rows_per_expert.scalar_type() == torch::kInt32,
      "rows_per_expert must be int32");

  TORCH_CHECK(
      topk_ids.scalar_type() == torch::kInt64 ||
          topk_ids.scalar_type() == torch::kInt32,
      "topk_ids must be int64 or int32");

  int num_rows = hidden_states.size(0);
  int hidden_size = hidden_states.size(1);
  int TopK = topk_ids.size(1);
  int block_k = 1;
  // shape check
  TORCH_CHECK(
      remapped_hidden_states.size(0) == num_rows * TopK &&
          hidden_states.size(1) == remapped_hidden_states.size(1),
      "remapped_hidden_states must be [num_rows * TopK, "
      "hidden_size]");
  if (hidden_states_scales.has_value()) {
    block_k = hidden_size / hidden_states_scales->size(1);
    TORCH_CHECK(
        remapped_hidden_states_scales->size(0) == num_rows * TopK &&
            hidden_states_scales->size(1) ==
                remapped_hidden_states_scales->size(1),
        "remapped_hidden_states_scales must be [num_rows * "
        "TopK, hidden_size // block_k]");
  }

  if (expert_map.has_value()) {
    TORCH_CHECK(
        expert_map->size(0) == total_experts_num,
        "expert_map must be [total_experts_num]");
  }
  TORCH_CHECK(
      rows_per_expert.size(0) == local_experts_num,
      "rows_per_expert must be [local_experts_num]");
  TORCH_CHECK(
      topk_ids.size(0) == num_rows && topk_ids.size(1) == TopK,
      "topk_ids must be [num_rows, TopK]");

  const at::DeviceGuard device_guard(hidden_states.device());
  auto& queue = vllm::xpu::vllmGetQueue();

#define LAUNCH_REMAP_HIDDEN_STATES(TA, TS, TopK)                              \
  vllm::moe::RemapHiddenStatesLauncher<TA, TS, TopK>(                         \
      reinterpret_cast<TA*>(hidden_states.data_ptr()),                        \
      hidden_states_scales.has_value()                                        \
          ? reinterpret_cast<TS*>(hidden_states_scales->data_ptr())           \
          : nullptr,                                                          \
      reinterpret_cast<TA*>(remapped_hidden_states.data_ptr()),               \
      remapped_hidden_states_scales.has_value()                               \
          ? reinterpret_cast<TS*>(remapped_hidden_states_scales->data_ptr())  \
          : nullptr,                                                          \
      expert_map.has_value() ? reinterpret_cast<int*>(expert_map->data_ptr()) \
                             : nullptr,                                       \
      reinterpret_cast<int*>(rows_per_expert.data_ptr()),                     \
      reinterpret_cast<int*>(unpermuted_row_to_permuted_row.data_ptr()),      \
      reinterpret_cast<void*>(topk_ids.data_ptr()),                           \
      topk_ids.scalar_type() == torch::kInt32,                                \
      num_rows,                                                               \
      hidden_size,                                                            \
      block_k,                                                                \
      total_experts_num,                                                      \
      local_experts_num,                                                      \
      queue);

#define DISPATCH_TOPK_LAUNCH(TA, TS, TopK)              \
  if (TopK == 1) {                                      \
    LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 1);              \
  } else if (TopK == 2) {                               \
    LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 2);              \
  } else if (TopK == 4) {                               \
    LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 4);              \
  } else if (TopK == 6) {                               \
    LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 6);              \
  } else if (TopK == 8) {                               \
    LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 8);              \
  } else if (TopK == 10) {                              \
    LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 10);             \
  } else {                                              \
    throw std::runtime_error("Unsupported TopK value"); \
  }

  if (hidden_states.scalar_type() == torch::kFloat16) {
    using scalar_t = sycl::half;
    DISPATCH_TOPK_LAUNCH(scalar_t, float, TopK);
  } else if (hidden_states.scalar_type() == torch::kBFloat16) {
    using scalar_t = sycl::ext::oneapi::bfloat16;
    DISPATCH_TOPK_LAUNCH(scalar_t, float, TopK);
  } else if (hidden_states.scalar_type() == torch::kFloat32) {
    using scalar_t = float;
    DISPATCH_TOPK_LAUNCH(scalar_t, float, TopK);
  } else if (hidden_states.scalar_type() == torch::kFloat8_e4m3fn) {
    using scalar_t = uint8_t;
    if (hidden_states_scales->scalar_type() == torch::kFloat32) {
      DISPATCH_TOPK_LAUNCH(scalar_t, float, TopK);
    } else if (hidden_states_scales->scalar_type() == torch::kFloat8_e8m0fnu) {
      DISPATCH_TOPK_LAUNCH(scalar_t, uint8_t, TopK);
    } else {
      throw std::runtime_error("Unsupported data type in hidden_states_scales");
    }
  } else if (hidden_states.scalar_type() == torch::kFloat4_e2m1fn_x2) {
    using scalar_t = uint8_t;
    if (hidden_states_scales->scalar_type() == torch::kFloat8_e8m0fnu) {
      DISPATCH_TOPK_LAUNCH(scalar_t, uint8_t, TopK);
    } else {
      throw std::runtime_error("Unsupported data type in hidden_states_scales");
    }
  } else {
    throw std::runtime_error("Unsupported data type in hidden_states");
  }
}