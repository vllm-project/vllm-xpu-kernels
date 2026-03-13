#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"

namespace vllm {
namespace moe {

class InitExpertMap {
 public:
  InitExpertMap(
      int* expert_map,
      const int num_experts,
      const int ep_rank,
      const int ep_size)
      : expert_map(expert_map),
        num_experts(num_experts),
        ep_rank(ep_rank),
        ep_size(ep_size) {}

  static constexpr int GroupWorkItem = 256;
  static constexpr int WARP_SIZE = 32;

  static inline sycl::nd_range<1>
  get_nd_range(const int num_experts, const int ep_size) {
    int group_nums =
        (ep_size * num_experts + GroupWorkItem - 1) / GroupWorkItem;
    sycl::range<1> local(GroupWorkItem);
    sycl::range<1> group(group_nums);
    return sycl::nd_range<1>(local * group, local);
  }

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    auto local_id = item.get_local_id(0);
    auto group_id = item.get_group(0);

    int experts_id = group_id * GroupWorkItem + local_id;
    int total_experts = num_experts * ep_size;

    if (experts_id >= total_experts) {
      return;
    }

    int left_offset = ep_rank * num_experts;
    int right_offset = left_offset + num_experts;

    if (experts_id >= left_offset && experts_id < right_offset) {
      expert_map[experts_id] = experts_id - left_offset;
    } else {
      expert_map[experts_id] = -1;
    }
  }

 private:
  int* expert_map;
  const int num_experts;
  const int ep_rank;
  const int ep_size;
};

void InitExpertMapLauncher(
    int* expert_map,
    const int num_experts,
    const int ep_rank,
    const int ep_size,
    sycl::queue& queue) {
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        InitExpertMap::get_nd_range(num_experts, ep_size),
        InitExpertMap{expert_map, num_experts, ep_rank, ep_size});
  });
}

}  // namespace moe
}  // namespace vllm

void init_expert_map(
    torch::Tensor& expert_map,  // [num_experts * ep_size]
    const int64_t num_experts,
    const int64_t ep_rank,
    const int64_t ep_size) {
  TORCH_CHECK(
      expert_map.scalar_type() == torch::kInt32, "expert_map must be int32");

  const at::DeviceGuard device_guard(expert_map.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  vllm::moe::InitExpertMapLauncher(
      reinterpret_cast<int*>(expert_map.data_ptr()),
      num_experts,
      ep_rank,
      ep_size,
      queue);
}