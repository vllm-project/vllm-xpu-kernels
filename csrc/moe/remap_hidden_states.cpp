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

template <typename TA, typename TS>
class RemapHiddenStates {
 public:
  RemapHiddenStates(
      TA* hidden_states,
      TS* hidden_states_scales,
      TA* remapped_hidden_states,
      TS* remapped_hidden_states_scales,
      int* expert_map,
      int* unpermuted_row_to_permuted_row,
      int64_t* expert_first_token_offset,
      int64_t* topk_ids,
      const int num_rows,
      const int hidden_size,
      const int block_k,
      const int n_experts_per_token,
      const int total_experts_num)
      : hidden_states(hidden_states),
        hidden_states_scales(hidden_states_scales),
        remapped_hidden_states(remapped_hidden_states),
        remapped_hidden_states_scales(remapped_hidden_states_scales),
        expert_map(expert_map),
        unpermuted_row_to_permuted_row(unpermuted_row_to_permuted_row),
        expert_first_token_offset(expert_first_token_offset),
        topk_ids(topk_ids),
        num_rows(num_rows),
        hidden_size(hidden_size),
        block_k(block_k),
        n_experts_per_token(n_experts_per_token),
        total_experts_num(total_experts_num) {}

  static constexpr int GroupWorkItem = 256;
  static constexpr int WARP_SIZE = 32;
  static constexpr int ElemsPerItem = 4;
  static constexpr int TopK = 8;

  static inline sycl::nd_range<1> get_nd_range(
      const int num_rows,
      const int n_experts_per_token,
      const int hidden_size) {
    sycl::range<1> local(GroupWorkItem);
    sycl::range<1> group(num_rows);
    return sycl::nd_range<1>(local * group, local);
  }

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    auto local_id = item.get_local_id(0);
    auto group_id = item.get_group(0);

    int row = group_id;
    int global_expert_id[TopK];
    int local_expert_id[TopK];

#pragma unroll
    for(int i = 0; i < TopK; ++i){
      global_expert_id[i] = topk_ids[row * TopK + i];
    }
#pragma unroll
    for(int i = 0; i < TopK; ++i){
      local_expert_id[i] = expert_map[global_expert_id[i]];
    }

    int rows_offset[TopK];
#pragma unroll
    for(int i = 0; i < TopK; ++i){
      if(local_expert_id[i] != -1){
        rows_offset[i] = unpermuted_row_to_permuted_row[row * TopK + i] + expert_first_token_offset[local_expert_id[i]];
      } else {
        rows_offset[i] = -1;
      }
    }

    int stride = GroupWorkItem * ElemsPerItem;
    int loop_count = (hidden_size + stride - 1) / stride;

    for (int l = 0; l < loop_count; ++l) {
      int start_id = l * stride + local_id * ElemsPerItem;

      using load_type = sycl::vec<TA, ElemsPerItem>;
      load_type data;
      if(start_id < hidden_size){
        data = *(reinterpret_cast<load_type*>(hidden_states + row * hidden_size + start_id));
        #pragma unroll
        for (int i = 0; i < TopK; ++i) {
          int offset = rows_offset[i];
          if (offset == -1)
            continue;
          *(reinterpret_cast<load_type*>(remapped_hidden_states + offset * hidden_size + start_id)) = data;
        }
      }
    }

//     if (hidden_states_scales != nullptr &&
//         remapped_hidden_states_scales != nullptr) {
//       int scaled_hidden_size = hidden_size / block_k;
//       loop_count = (scaled_hidden_size + local_range * ElemsPerItem - 1) /
//                    (local_range * ElemsPerItem);
//       for (int l = 0; l < loop_count; ++l) {
//         int start_id = l * local_range * ElemsPerItem + local_id * ElemsPerItem;
//         int remained_elems = scaled_hidden_size - start_id;
//         if (remained_elems >= ElemsPerItem) {
// #pragma unroll
//           for (int i = 0; i < ElemsPerItem; ++i) {
//             int idx = start_id + i;
//             remapped_hidden_states_scales
//                 [rows_offset * scaled_hidden_size + idx] =
//                     hidden_states_scales[row * scaled_hidden_size + idx];
//           }
//         } else if (remained_elems > 0) {
// #pragma unroll
//           for (int i = 0; i < remained_elems; ++i) {
//             int idx = start_id + i;
//             remapped_hidden_states_scales
//                 [rows_offset * scaled_hidden_size + idx] =
//                     hidden_states_scales[row * scaled_hidden_size + idx];
//           }
//         }
//       }
//     }
  item.barrier(sycl::access::fence_space::local_space);

  if (local_id == 0) {
#pragma unroll
      for(int i = 0; i < TopK; ++i){
        unpermuted_row_to_permuted_row[row * TopK + i] = rows_offset[i];
      }
    }
  }

 private:
  TA* hidden_states;
  TS* hidden_states_scales;
  TA* remapped_hidden_states;
  TS* remapped_hidden_states_scales;
  int* expert_map;
  int* unpermuted_row_to_permuted_row;
  int64_t* expert_first_token_offset;
  int64_t* topk_ids;
  const int num_rows;
  const int hidden_size;
  const int block_k;
  const int n_experts_per_token;
  const int total_experts_num;
};

template <typename TA, typename TS>
void RemapHiddenStatesLauncher(
    TA* hidden_states,
    TS* hidden_states_scales,
    TA* remapped_hidden_states,
    TS* remapped_hidden_states_scales,
    int* expert_map,
    int64_t* expert_first_token_offset,
    int* unpermuted_row_to_permuted_row,
    int64_t* topk_ids,
    const int num_rows,
    const int hidden_size,
    const int block_k,
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
        RemapHiddenStates<TA, TS>::get_nd_range(
            num_rows, n_experts_per_token, hidden_size),
        RemapHiddenStates<TA, TS>{
            hidden_states,
            hidden_states_scales,
            remapped_hidden_states,
            remapped_hidden_states_scales,
            expert_map,
            unpermuted_row_to_permuted_row,
            expert_first_token_offset,
            topk_ids,
            num_rows,
            hidden_size,
            block_k,
            n_experts_per_token,
            total_experts_num});
  });
}

}  // namespace moe
}  // namespace vllm

void remap_hidden_states(
    torch::Tensor& hidden_states,  // [num_rows, hidden_size]
    const c10::optional<torch::Tensor>&
        hidden_states_scales,  // [num_rows, hidden_size // block_k] or empty
    torch::Tensor& remapped_hidden_states,  // [num_rows * n_experts_per_token,
                                            // hidden_size]
    const c10::optional<torch::Tensor>&
        remapped_hidden_states_scales,  // [num_rows, hidden_size // block_k] or
                                        // empty
    torch::Tensor& expert_map,          // [total_experts_num]
    torch::Tensor& expert_first_token_offset,  // [local_experts_num  + 1]
    torch::Tensor&
        unpermuted_row_to_permuted_row,  // [num_rows, n_experts_per_token]
    torch::Tensor& topk_ids,             // [num_rows, n_experts_per_token]
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

  TORCH_CHECK(
      expert_map.scalar_type() == torch::kInt32, "expert_map must be int32");

  TORCH_CHECK(
      expert_first_token_offset.scalar_type() == torch::kInt64,
      "expert_first_token_offset must be int64");

  TORCH_CHECK(
      topk_ids.scalar_type() == torch::kInt64, "topk_ids must be int64");

  int num_rows = hidden_states.size(0);
  int hidden_size = hidden_states.size(1);
  int n_experts_per_token = topk_ids.size(1);
  int block_k = 1;
  // shape check
  TORCH_CHECK(
      remapped_hidden_states.size(0) == num_rows * n_experts_per_token &&
          hidden_states.size(1) == remapped_hidden_states.size(1),
      "remapped_hidden_states must be [num_rows * n_experts_per_token, "
      "hidden_size]");
  if (hidden_states_scales.has_value()) {
    block_k = hidden_size / hidden_states_scales->size(1);
    TORCH_CHECK(
        remapped_hidden_states_scales->size(0) ==
                num_rows * n_experts_per_token &&
            hidden_states_scales->size(1) ==
                remapped_hidden_states_scales->size(1),
        "remapped_hidden_states_scales must be [num_rows * "
        "n_experts_per_token, hidden_size // block_k]");
  }
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

#define LAUNCH_REMAP_HIDDEN_STATES(TA, TS)                                   \
  vllm::moe::RemapHiddenStatesLauncher<TA, TS>(                              \
      reinterpret_cast<TA*>(hidden_states.data_ptr()),                       \
      hidden_states_scales.has_value()                                       \
          ? reinterpret_cast<TS*>(hidden_states_scales->data_ptr())          \
          : nullptr,                                                         \
      reinterpret_cast<TA*>(remapped_hidden_states.data_ptr()),              \
      remapped_hidden_states_scales.has_value()                              \
          ? reinterpret_cast<TS*>(remapped_hidden_states_scales->data_ptr()) \
          : nullptr,                                                         \
      reinterpret_cast<int*>(expert_map.data_ptr()),                         \
      reinterpret_cast<int64_t*>(expert_first_token_offset.data_ptr()),      \
      reinterpret_cast<int*>(unpermuted_row_to_permuted_row.data_ptr()),     \
      reinterpret_cast<int64_t*>(topk_ids.data_ptr()),                       \
      num_rows,                                                              \
      hidden_size,                                                           \
      block_k,                                                               \
      n_experts_per_token,                                                   \
      total_experts_num,                                                     \
      local_experts_num,                                                     \
      queue);

  if (hidden_states.scalar_type() == torch::kFloat16) {
    using scalar_t = sycl::half;
    LAUNCH_REMAP_HIDDEN_STATES(scalar_t, float);
  } else if (hidden_states.scalar_type() == torch::kBFloat16) {
    using scalar_t = sycl::ext::oneapi::bfloat16;
    LAUNCH_REMAP_HIDDEN_STATES(scalar_t, float);
  } else if (hidden_states.scalar_type() == torch::kFloat32) {
    using scalar_t = float;
    LAUNCH_REMAP_HIDDEN_STATES(scalar_t, float);
  } else if (hidden_states.scalar_type() == torch::kFloat8_e4m3fn) {
    using scalar_t = uint8_t;
    if (hidden_states_scales->scalar_type() == torch::kFloat32) {
      LAUNCH_REMAP_HIDDEN_STATES(scalar_t, float);
    } else if (hidden_states_scales->scalar_type() == torch::kFloat8_e8m0fnu) {
      LAUNCH_REMAP_HIDDEN_STATES(scalar_t, uint8_t);
    } else {
      throw std::runtime_error("Unsupported data type in hidden_states_scales");
    }
  } else if (hidden_states.scalar_type() == torch::kFloat4_e2m1fn_x2) {
    using scalar_t = uint8_t;
    if (hidden_states_scales->scalar_type() == torch::kFloat8_e8m0fnu) {
      LAUNCH_REMAP_HIDDEN_STATES(scalar_t, uint8_t);
    } else {
      throw std::runtime_error("Unsupported data type in hidden_states_scales");
    }
  } else {
    throw std::runtime_error("Unsupported data type in hidden_states");
  }
}