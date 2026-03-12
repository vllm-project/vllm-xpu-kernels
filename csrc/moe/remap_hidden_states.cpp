#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"

namespace vllm {
namespace moe {

class RowsPerExpertCount {
 public:
  RowsPerExpertCount(
      int* expert_map,
      int64_t* expert_first_token_offset,
      int64_t* topk_ids,
      int* unpermuted_row_to_permuted_row,
      const int num_rows,
      const int n_experts_per_token)
      : expert_map(expert_map),
        expert_first_token_offset(expert_first_token_offset),
        topk_ids(topk_ids),
        unpermuted_row_to_permuted_row(unpermuted_row_to_permuted_row),
        num_rows(num_rows),
        n_experts_per_token(n_experts_per_token) {}

  static constexpr int GroupWorkItem = 256;
  static constexpr int WARP_SIZE = 32;

  static inline sycl::nd_range<1>
  get_nd_range(const int num_rows, const int n_experts_per_token) {
    int group_nums =
        (num_rows * n_experts_per_token + GroupWorkItem - 1) / GroupWorkItem;
    sycl::range<1> local(GroupWorkItem);
    sycl::range<1> group(group_nums);
    return sycl::nd_range<1>(local * group, local);
  }

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    auto global_id = item.get_global_linear_id();

    if (global_id >= num_rows * n_experts_per_token) {
      // early return
      return;
    }

    int global_expert_id = topk_ids[global_id];
    int local_expert_id = expert_map[global_expert_id];

    if (local_expert_id == -1) {
      // selected expert is not at current rank
      unpermuted_row_to_permuted_row[global_id] = -1;
      return;
    }

    auto atomic_count = sycl::atomic_ref<
        int64_t,
        sycl::memory_order_relaxed,
        sycl::memory_scope_device,
        sycl::access::address_space::global_space>(
        expert_first_token_offset[local_expert_id + 1]);
    int64_t old = atomic_count.fetch_add(1);
    unpermuted_row_to_permuted_row[global_id] = old;
  }

 private:
  int* expert_map;
  int64_t* expert_first_token_offset;
  int64_t* topk_ids;
  int* unpermuted_row_to_permuted_row;
  const int num_rows;
  const int n_experts_per_token;
};

class CalculateFristTokenOffset {
 public:
  CalculateFristTokenOffset(
      int64_t* expert_first_token_offset, const int local_experts_num)
      : expert_first_token_offset(expert_first_token_offset),
        local_experts_num(local_experts_num) {}

  static constexpr int WARP_SIZE = 32;
  static constexpr int MAX_LOCAL_STORAGE = 32;

  static inline sycl::nd_range<1> get_nd_range() {
    sycl::range<1> local(WARP_SIZE);
    sycl::range<1> group(1);
    return sycl::nd_range<1>(local * group, local);
  }

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    auto sg = item.get_sub_group();
    int sg_local_id = sg.get_local_linear_id();

    int elems_per_item = (local_experts_num + WARP_SIZE - 1) / WARP_SIZE;

    int local_sum = 0;
    int local_start = elems_per_item * sg_local_id;
    int remained_elems = local_experts_num - local_start;
    int local_elems =
        remained_elems > elems_per_item ? elems_per_item : remained_elems;

    int local_storage[MAX_LOCAL_STORAGE];

    if (remained_elems > 0) {
#pragma unroll
      for (int i = 0; i < local_elems; ++i) {
        int idx = local_start + i;
        local_storage[i] = expert_first_token_offset[idx + 1];
        local_sum += local_storage[i];
      }
    }

    int global_sum =
        sycl::exclusive_scan_over_group(sg, local_sum, sycl::plus<int>());

    if (remained_elems > 0) {
#pragma unroll
      for (int i = 0; i < local_elems; ++i) {
        int idx = local_start + i;
        global_sum += local_storage[i];
        expert_first_token_offset[idx + 1] = global_sum;
      }
    }
  }

 private:
  int64_t* expert_first_token_offset;
  const int local_experts_num;
};

template <typename T>
class RemapHiddenStates {
 public:
  RemapHiddenStates(
      T* hidden_states,
      T* remapped_hidden_states,
      int* expert_map,
      int* unpermuted_row_to_permuted_row,
      int64_t* expert_first_token_offset,
      int64_t* topk_ids,
      const int num_rows,
      const int hidden_size,
      const int n_experts_per_token,
      const int total_experts_num)
      : hidden_states(hidden_states),
        remapped_hidden_states(remapped_hidden_states),
        expert_map(expert_map),
        unpermuted_row_to_permuted_row(unpermuted_row_to_permuted_row),
        expert_first_token_offset(expert_first_token_offset),
        topk_ids(topk_ids),
        num_rows(num_rows),
        hidden_size(hidden_size),
        n_experts_per_token(n_experts_per_token),
        total_experts_num(total_experts_num) {}

  static constexpr int GroupWorkItem = 256;
  static constexpr int WARP_SIZE = 32;
  static constexpr int ElemsPerItem = 4;

  static inline sycl::nd_range<1> get_nd_range(
      const int num_rows,
      const int n_experts_per_token,
      const int hidden_size) {
    int group_nums = num_rows * n_experts_per_token;
    int local_nums = GroupWorkItem;
    if (GroupWorkItem * ElemsPerItem > hidden_size) {
      local_nums = (hidden_size + ElemsPerItem - 1) / ElemsPerItem;
    }
    sycl::range<1> local(local_nums);
    sycl::range<1> group(group_nums);
    return sycl::nd_range<1>(local * group, local);
  }

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    auto local_id = item.get_local_id(0);
    auto local_range = item.get_local_range(0);
    auto group_id = item.get_group(0);

    int row = group_id / n_experts_per_token;
    int topi = group_id % n_experts_per_token;

    int global_expert_id = topk_ids[row * n_experts_per_token + topi];

    int local_expert_id = expert_map[global_expert_id];

    if (local_expert_id == -1) {
      // selected expert is not at current rank
      if (local_id == 0) {
        unpermuted_row_to_permuted_row[row * n_experts_per_token + topi] = -1;
      }
      return;
    }

    int rows_offset =
        unpermuted_row_to_permuted_row[row * n_experts_per_token + topi];
    int first_token_offset = expert_first_token_offset[local_expert_id];
    rows_offset += first_token_offset;

    item.barrier(sycl::access::fence_space::local_space);

    if (local_id == 0) {
      unpermuted_row_to_permuted_row[row * n_experts_per_token + topi] =
          rows_offset;
    }

    int loop_count = (hidden_size + local_range * ElemsPerItem - 1) /
                     (local_range * ElemsPerItem);

    for (int l = 0; l < loop_count; ++l) {
      int start_id = l * local_range * ElemsPerItem + local_id * ElemsPerItem;
      int remained_elems = hidden_size - start_id;
      if (remained_elems >= ElemsPerItem) {
#pragma unroll
        for (int i = 0; i < ElemsPerItem; ++i) {
          int idx = start_id + i;
          remapped_hidden_states[rows_offset * hidden_size + idx] =
              hidden_states[row * hidden_size + idx];
        }
      } else if (remained_elems > 0) {
#pragma unroll
        for (int i = 0; i < remained_elems; ++i) {
          int idx = start_id + i;
          remapped_hidden_states[rows_offset * hidden_size + idx] =
              hidden_states[row * hidden_size + idx];
        }
      }
    }
  }

 private:
  T* hidden_states;
  T* remapped_hidden_states;
  int* expert_map;
  int* unpermuted_row_to_permuted_row;
  int64_t* expert_first_token_offset;
  int64_t* topk_ids;
  const int num_rows;
  const int hidden_size;
  const int n_experts_per_token;
  const int total_experts_num;
};

template <typename T>
void RemapHiddenStatesLauncher(
    T* hidden_states,
    T* remapped_hidden_states,
    int* expert_map,
    int64_t* expert_first_token_offset,
    int* unpermuted_row_to_permuted_row,
    int64_t* topk_ids,
    const int num_rows,
    const int hidden_size,
    const int n_experts_per_token,
    const int total_experts_num,
    const int local_experts_num,
    sycl::queue& queue) {
  assert(
      local_experts_num <= CalculateFristTokenOffset::MAX_LOCAL_STORAGE *
                               CalculateFristTokenOffset::WARP_SIZE);

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        RowsPerExpertCount::get_nd_range(num_rows, n_experts_per_token),
        RowsPerExpertCount{
            expert_map,
            expert_first_token_offset,
            topk_ids,
            unpermuted_row_to_permuted_row,
            num_rows,
            n_experts_per_token});
  });

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        CalculateFristTokenOffset::get_nd_range(),
        CalculateFristTokenOffset{
            expert_first_token_offset, local_experts_num});
  });

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        RemapHiddenStates<T>::get_nd_range(
            num_rows, n_experts_per_token, hidden_size),
        RemapHiddenStates<T>{
            hidden_states,
            remapped_hidden_states,
            expert_map,
            unpermuted_row_to_permuted_row,
            expert_first_token_offset,
            topk_ids,
            num_rows,
            hidden_size,
            n_experts_per_token,
            total_experts_num});
  });
}

}  // namespace moe
}  // namespace vllm

void remap_hidden_states(
    torch::Tensor& hidden_states,           // [num_rows, hidden_size]
    torch::Tensor& remapped_hidden_states,  // [num_rows * n_experts_per_token,
                                            // hidden_size]
    torch::Tensor& expert_map,              // [total_experts_num]
    torch::Tensor& expert_first_token_offset,  // [local_experts_num  + 1]
    torch::Tensor&
        unpermuted_row_to_permuted_row,  // [num_rows, n_experts_per_token]
    torch::Tensor& topk_ids,             // [num_rows, n_experts_per_token]
    int64_t num_rows,
    int64_t hidden_size,
    int64_t n_experts_per_token,
    int64_t total_experts_num,
    int64_t local_experts_num) {
  // dtype check
  TORCH_CHECK(
      hidden_states.scalar_type() == remapped_hidden_states.scalar_type(),
      "hidden_states and remapped_hidden_states must have save dtype");

  TORCH_CHECK(
      expert_map.scalar_type() == torch::kInt32, "expert_map must be int32");

  TORCH_CHECK(
      expert_first_token_offset.scalar_type() == torch::kInt64,
      "expert_first_token_offset must be int64");

  TORCH_CHECK(
      topk_ids.scalar_type() == torch::kInt64, "topk_ids must be int64");

  // shape check
  TORCH_CHECK(
      hidden_states.size(0) == num_rows && hidden_states.size(1) == hidden_size,
      "hidden_states must be [num_rows, hidden_size]");
  TORCH_CHECK(
      remapped_hidden_states.size(0) == num_rows * n_experts_per_token &&
          remapped_hidden_states.size(1) == hidden_size,
      "remapped_hidden_states must be [num_rows * n_experts_per_token, "
      "hidden_size]");
  TORCH_CHECK(
      expert_map.size(0) == total_experts_num,
      "expert_map must be [total_experts_num]");
  TORCH_CHECK(
      expert_first_token_offset.size(0) == local_experts_num + 1,
      "expert_map must be [local_experts_num + 1]");
  TORCH_CHECK(
      topk_ids.size(0) == num_rows && topk_ids.size(1) == n_experts_per_token,
      "topk_ids must be [num_rows, n_experts_per_token]");

  const at::DeviceGuard device_guard(hidden_states.device());
  auto& queue = vllm::xpu::vllmGetQueue();

#define LAUNCH_REMAP_HIDDEN_STATES(T)                                    \
  vllm::moe::RemapHiddenStatesLauncher<T>(                               \
      reinterpret_cast<T*>(hidden_states.data_ptr()),                    \
      reinterpret_cast<T*>(remapped_hidden_states.data_ptr()),           \
      reinterpret_cast<int*>(expert_map.data_ptr()),                     \
      reinterpret_cast<int64_t*>(expert_first_token_offset.data_ptr()),  \
      reinterpret_cast<int*>(unpermuted_row_to_permuted_row.data_ptr()), \
      reinterpret_cast<int64_t*>(topk_ids.data_ptr()),                   \
      num_rows,                                                          \
      hidden_size,                                                       \
      n_experts_per_token,                                               \
      total_experts_num,                                                 \
      local_experts_num,                                                 \
      queue);

  if (hidden_states.scalar_type() == torch::kFloat16) {
    using scalar_t = sycl::half;
    LAUNCH_REMAP_HIDDEN_STATES(scalar_t);
  } else if (hidden_states.scalar_type() == torch::kBFloat16) {
    using scalar_t = sycl::ext::oneapi::bfloat16;
    LAUNCH_REMAP_HIDDEN_STATES(scalar_t);
  } else if (hidden_states.scalar_type() == torch::kFloat32) {
    using scalar_t = float;
    LAUNCH_REMAP_HIDDEN_STATES(scalar_t);
  } else {
    throw std::runtime_error("Unsupported data type in remap_hidden_states");
  }
}