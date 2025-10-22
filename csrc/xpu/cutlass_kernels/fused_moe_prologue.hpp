#include <torch/all.h>
#include "utils.h"

#define MAX_SUBGROUP_SIZE 32

template <typename T>
inline T ceilDiv(T a, T b) {
    return (a + b - 1) / b;
}

int64_t computeNumTokensPerBlock(int64_t const num_tokens, int64_t const num_experts_per_node) {
  for (int64_t num_tokens_per_block = 32; num_tokens_per_block <= 1024; num_tokens_per_block *= 2) {
    int64_t const num_blocks_per_seq = ceilDiv(num_tokens, num_tokens_per_block);
    if (num_blocks_per_seq * num_experts_per_node <= num_tokens_per_block) {
      return num_tokens_per_block;
    }
  }
  return 1024;
}

template <typename T>
inline T shuffle_up(sycl::sub_group sg, T value, int delta) {
  int lane_id = sg.get_local_linear_id();
  if (lane_id < delta) {
    delta = 0;
  }
  return sycl::select_from_group(sg, value, lane_id - delta);
}

template <int RANGE_DIM, typename T>
inline T get_subgroup_prefix(
    sycl::group<RANGE_DIM> group,
    sycl::sub_group sg,
    T subgroup_aggregate,
    T& group_aggregate) {
  auto lane_id = sg.get_local_linear_id();
  auto subgroup_id = sg.get_group_linear_id();
  auto subgroup_range = sg.get_group_linear_range();
  auto subgroup_local_range = sg.get_local_linear_range();

  // Use shared memory to store the subgroup aggregate
  auto& subgroup_aggregates =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<
          T[MAX_SUBGROUP_SIZE]>(group);

  if (lane_id == subgroup_local_range - 1) {
    subgroup_aggregates[subgroup_id] = subgroup_aggregate;
  }

  sycl::group_barrier(group);

  group_aggregate = subgroup_aggregates[0];

  T subgroup_prefix;
#pragma unroll
  for (int subgroup_offset = 1; subgroup_offset < subgroup_range;
       ++subgroup_offset) {
    if (subgroup_id == subgroup_offset) {
      subgroup_prefix = group_aggregate;
    }
    group_aggregate =
        group_aggregate + subgroup_aggregates[subgroup_offset];
  }

  return subgroup_prefix;
}

// implementation of inclusive_scan_stem by shuffle
template <typename T>
inline T inclusive_scan_over_subgroup_step(
    sycl::sub_group sg,
    T input,
    int offset) {
  int lane_id = sg.get_local_linear_id();
  T temp = shuffle_up(sg, input, offset);

  T output = temp + input;

  if (lane_id < offset) {
    output = input;
  }

  return output;
}

// sub_group scan
template <typename T>
inline void inclusive_scan_over_subgroup(
    sycl::sub_group sg,
    T input,
    T& inclusive_output) {
  auto sub_group_range = sg.get_local_linear_range();

  int STEPS = sycl::log2(static_cast<float>(sub_group_range));

  inclusive_output = input;
#pragma unroll
  for (int STEP = 0; STEP < STEPS; ++STEP) {
    inclusive_output = inclusive_scan_over_subgroup_step(
        sg, inclusive_output, (1 << STEP));
  }
}

template <int RANGE_DIM, typename T>
inline T exclusive_scan_over_group(
    sycl::nd_item<RANGE_DIM> item,
    T input,
    T& group_aggregate) {
  auto sg = item.get_sub_group();

  T inclusive_output;
  inclusive_scan_over_subgroup(sg, input, inclusive_output);

  int lane_id = sg.get_local_linear_id();
  T exclusive_output = shuffle_up(sg, inclusive_output, 1);

  auto group = item.get_group();

  T subgroup_prefix = get_subgroup_prefix(
      group, sg, inclusive_output, group_aggregate);
  auto subgroup_id = sg.get_group_linear_id();

  if (subgroup_id != 0) {
    exclusive_output = subgroup_prefix + exclusive_output;

    if (lane_id == 0) {
      exclusive_output = subgroup_prefix;
    }
  }

  return exclusive_output;
}

template <int kNumTokensPerBlock>
class blockExpertPrefixSumKernel {
  public:
    blockExpertPrefixSumKernel(int const* token_selected_experts,
                               int* blocked_expert_counts,
                               int* blocked_row_to_unpermuted_row,
                               int64_t const num_tokens,
                               int64_t const num_experts_per_token,
                               int const start_expert_id)
      : token_selected_experts(token_selected_experts),
        blocked_expert_counts(blocked_expert_counts),
        blocked_row_to_unpermuted_row(blocked_row_to_unpermuted_row),
        num_tokens(num_tokens),
        num_experts_per_token(num_experts_per_token),
        start_expert_id(start_expert_id) {}
  
  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {

  // target_expert_id and expert_id are offset by start_expert_id
  int const target_expert_id = item_ct1.get_group(2);
  int const block_id = item_ct1.get_group(1);
  int const num_blocks_per_seq = item_ct1.get_group_range(1);
  int const token_id = block_id * kNumTokensPerBlock + item_ct1.get_local_id(2);

  int expanded_token_id = -1;
  if (token_id < num_tokens) {
    for (int i = 0; i < num_experts_per_token; i++) {
      // TODO(enweiz): Fix uncoalesced access with shared memory.
      int const expert_id =
          token_selected_experts[token_id * num_experts_per_token + i] - start_expert_id;
      if (expert_id == target_expert_id) {
        expanded_token_id = i * num_tokens + token_id;
        break;
      }
    }
  }

  int const has_matched = expanded_token_id >= 0 ? 1 : 0;
  int index = 0;
  // BlockScan(temp_storage).ExclusiveSum(has_matched, index);
  exclusive_scan_over_group(item_ct1, has_matched, index);
  if (has_matched) {
    blocked_row_to_unpermuted_row[target_expert_id * num_tokens + block_id * kNumTokensPerBlock +
                                  index] = expanded_token_id;
  }
  if (item_ct1.get_local_id(2) == kNumTokensPerBlock - 1) {
    blocked_expert_counts[target_expert_id * num_blocks_per_seq + block_id] = index + has_matched;
  }
}

  
  private:
      int const* token_selected_experts;
      int* blocked_expert_counts;
      int* blocked_row_to_unpermuted_row;
      int64_t const num_tokens;
      int64_t const num_experts_per_token;
      int const start_expert_id;

};

#define LAUNCH_PREFIXSUM_KERNEL(kNumTokensPerBlock) \
 stream.submit([&](sycl::handler& cgh) {            \                     
    cgh.parallel_for(sycl::nd_range<3>(grid * block, block), \ 
        blockExpertPrefixSumKernel<kNumTokensPerBlock>(token_selected_experts, blocked_expert_counts, \
             blocked_row_to_unpermuted_row, num_tokens, num_experts_per_token, \
             start_expert_id));      \
  });


void blockExpertPrefixSum(int const* token_selected_experts, int* blocked_expert_counts,
                          int* blocked_row_to_unpermuted_row, int64_t const num_tokens,
                          int64_t const num_experts_per_node, int64_t const num_experts_per_token,
                          int64_t const num_tokens_per_block, int64_t const num_blocks_per_seq,
                          int const start_expert_id, sycl::queue& stream) {

  sycl::range<3> grid(1, num_blocks_per_seq, num_experts_per_node); 
  sycl::range<3> block(1, 1, num_tokens_per_block);
  
  if (num_tokens_per_block <= 32) {
    LAUNCH_PREFIXSUM_KERNEL(32);
  } else if (num_tokens_per_block <= 64) {
    LAUNCH_PREFIXSUM_KERNEL(64);
  } else if (num_tokens_per_block <= 128) {
    LAUNCH_PREFIXSUM_KERNEL(128);
  } else if (num_tokens_per_block <= 256) {
    LAUNCH_PREFIXSUM_KERNEL(256);
  } else if (num_tokens_per_block <= 512) {
    LAUNCH_PREFIXSUM_KERNEL(512);
  } else {
    LAUNCH_PREFIXSUM_KERNEL(1024);
  }

 
}


void threeStepBuildExpertMapsSortFirstToken(
    int const* token_selected_experts, int* permuted_token_selected_experts,
    int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset, int* blocked_expert_counts,
    int* blocked_expert_counts_cumsum, int* blocked_row_to_unpermuted_row, int64_t const num_tokens,
    int64_t const num_experts_per_node, int64_t const num_experts_per_token,
    int const start_expert_id, sycl::queue& stream){
  std::cout << "three steps" << std::endl;
  int64_t const num_tokens_per_block = computeNumTokensPerBlock(num_tokens, num_experts_per_node);
  int64_t const num_blocks_per_seq = ceilDiv(num_tokens, num_tokens_per_block);

  blockExpertPrefixSum(token_selected_experts, blocked_expert_counts, blocked_row_to_unpermuted_row,
                       num_tokens, num_experts_per_node, num_experts_per_token,
                       num_tokens_per_block, num_blocks_per_seq, start_expert_id, stream);
  stream.wait();
  std::cout << "finish 3 steps" << std::endl;
}
