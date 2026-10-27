#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"
#include "sycl_launch.h"

namespace vllm {
namespace moe {

static inline int
to_local_expert_id(int global_expert_id, const int* expert_map) {
  if (global_expert_id < 0) {
    return -1;
  }
  return expert_map == nullptr ? global_expert_id
                               : expert_map[global_expert_id];
}

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
    sycl::group_barrier(item.get_group());

    // ===== Phase 2: local atomic =====
    if (global_id < num_rows * TopK) {
      int global_expert_id =
          is_topk_ids_int32 ? reinterpret_cast<int32_t*>(topk_ids)[global_id]
                            : reinterpret_cast<int64_t*>(topk_ids)[global_id];
      int local_expert_id = to_local_expert_id(global_expert_id, expert_map);

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

    sycl::group_barrier(item.get_group());

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

    sycl::group_barrier(item.get_group());

    // ===== Phase 4: fix unpermuted_row_to_permuted_row =====
    if (global_id < num_rows * TopK) {
      int global_expert_id =
          is_topk_ids_int32 ? reinterpret_cast<int32_t*>(topk_ids)[global_id]
                            : reinterpret_cast<int64_t*>(topk_ids)[global_id];
      int local_expert_id = to_local_expert_id(global_expert_id, expert_map);

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

class ExpertCumsum {
 public:
  ExpertCumsum(
      sycl::local_accessor<int32_t, 1>& slm,
      int* rows_per_expert,
      int* expert_cumsum,
      const int local_experts_num,
      int* expert_scale_desc)
      : slm(slm),
        rows_per_expert(rows_per_expert),
        expert_cumsum(expert_cumsum),
        local_experts_num(local_experts_num),
        expert_scale_desc(expert_scale_desc) {}

  static constexpr int GroupWorkItem = 256;
  static constexpr int WARP_SIZE = 16;
  static constexpr int EXCLUSIVE_SIZE = 1024;

  static inline sycl::nd_range<1> get_nd_range(const int local_experts_num) {
    int local_num = WARP_SIZE;
    while (local_num < local_experts_num && local_num < GroupWorkItem) {
      local_num <<= 1;
    }
    sycl::range<1> local(local_num);
    sycl::range<1> group(1);
    return sycl::nd_range<1>(local * group, local);
  }

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    auto local_id = item.get_local_id(0);
    auto local_range = item.get_local_range(0);

    int32_t* expert_cumsum_ptr = static_cast<int32_t*>(
        slm.template get_multi_ptr<sycl::access::decorated::no>().get());

    // Second SLM region: exclusive prefix sum of round_up_4(rows_per_expert),
    // i.e. the per-expert base row of the padded MN-major scale surface. Only
    // populated when that layout is requested.
    const bool padded_scales = expert_scale_desc != nullptr;
    const int desc_len = local_experts_num + 1;
    int32_t* expert_pad_cumsum_ptr = expert_cumsum_ptr + desc_len;

    if (local_id == 0) {
      expert_cumsum_ptr[0] = 0;
      if (padded_scales) {
        expert_pad_cumsum_ptr[0] = 0;
      }
    }
    for (int i = local_id; i < local_experts_num; i += local_range) {
      expert_cumsum_ptr[i + 1] = rows_per_expert[i];
      if (padded_scales) {
        expert_pad_cumsum_ptr[i + 1] = (rows_per_expert[i] + 3) & ~3;
      }
    }

    sycl::group_barrier(item.get_group());

    // Scanning desc_len entries makes index e the exclusive prefix for every
    // expert and index local_experts_num the total, which is exactly the
    // descriptor consumers need.
    sycl::joint_inclusive_scan(
        item.get_group(),
        expert_cumsum_ptr,
        expert_cumsum_ptr + desc_len,
        expert_cumsum_ptr,
        sycl::plus<int>{});

    for (int i = local_id; i < local_experts_num; i += local_range) {
      expert_cumsum[i] = expert_cumsum_ptr[i + 1];
    }

    if (padded_scales) {
      sycl::joint_inclusive_scan(
          item.get_group(),
          expert_pad_cumsum_ptr,
          expert_pad_cumsum_ptr + desc_len,
          expert_pad_cumsum_ptr,
          sycl::plus<int>{});

      for (int i = local_id; i < desc_len; i += local_range) {
        expert_scale_desc[i] = expert_cumsum_ptr[i];
        expert_scale_desc[desc_len + i] = expert_pad_cumsum_ptr[i];
      }
    }
  }

 private:
  sycl::local_accessor<int32_t, 1> slm;
  int* rows_per_expert;
  int* expert_cumsum;
  const int local_experts_num;
  int* expert_scale_desc;
};

// Publish immutable copy destinations before the bandwidth-bound phase. This
// avoids a workgroup barrier in every route-copy group protecting the in-place
// transition from local expert offsets to global row offsets.
class FinalizeRowMapping {
 public:
  int* expert_map;
  int* row_mapping;
  int* inverse_mapping;
  const int* expert_cumsum;
  const void* topk_ids;
  bool ids_int32;
  int topk;

  void operator()(sycl::id<1> id) const {
    const int route = id[0];
    const int global_expert =
        ids_int32 ? static_cast<const int32_t*>(topk_ids)[route]
                  : static_cast<const int64_t*>(topk_ids)[route];
    const int expert = to_local_expert_id(global_expert, expert_map);
    const int offset = expert < 0
                           ? -1
                           : row_mapping[route] +
                                 (expert == 0 ? 0 : expert_cumsum[expert - 1]);
    row_mapping[route] = offset;
    if (inverse_mapping && offset >= 0) inverse_mapping[offset] = route / topk;
  }
};

template <typename TA, typename TS, int TopK>
class RemapHiddenStates {
 public:
  RemapHiddenStates(
      TA* hidden_states,
      TS* hidden_states_scales,
      TA* remapped_hidden_states,
      TS* remapped_hidden_states_scales,
      int* unpermuted_row_to_permuted_row,
      const int hidden_size,
      const int block_k,
      int* expert_scale_desc)
      : hidden_states(hidden_states),
        hidden_states_scales(hidden_states_scales),
        remapped_hidden_states(remapped_hidden_states),
        remapped_hidden_states_scales(remapped_hidden_states_scales),
        unpermuted_row_to_permuted_row(unpermuted_row_to_permuted_row),
        hidden_size(hidden_size),
        block_k(block_k),
        expert_scale_desc(expert_scale_desc) {}

  static constexpr int GroupWorkItem = 256;
  static constexpr int WARP_SIZE = 32;
  static constexpr int ElemsPerItem = sizeof(float) * 4 / sizeof(TA);
  static constexpr int CopyWords = 4;
  static constexpr int CopyElemsPerItem =
      CopyWords * sizeof(uint32_t) / sizeof(TA);

  static inline sycl::nd_range<3>
  get_nd_range(const int num_rows, const int hidden_size) {
    int local_num = GroupWorkItem;
    if (local_num * CopyElemsPerItem > hidden_size) {
      local_num = sycl::max(
          WARP_SIZE,
          ((hidden_size + CopyElemsPerItem * WARP_SIZE - 1) /
           (CopyElemsPerItem * WARP_SIZE)) *
              WARP_SIZE);
    }
    sycl::range<3> local(1, 1, local_num);
    // One group per source token (not per route): the group reads the token
    // once and scatters it to all of its TopK destination rows, so each source
    // row is read from DRAM exactly once instead of relying on TopK sibling
    // groups happening to overlap in cache.
    sycl::range<3> group(1, 1, num_rows);
    return sycl::nd_range<3>(local * group, local);
  }

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<3> item) const {
    auto local_id = item.get_local_id(2);
    auto local_range = item.get_local_range(2);
    auto group_id = item.get_group(2);
    auto subgroup = item.get_sub_group();

    // Each group owns one source token `row` and writes it to all TopK
    // destination rows. Reading the source once and fanning out the TopK
    // stores keeps the source read at num_rows * hidden_size regardless of
    // TopK; the earlier per-route grid re-read the source TopK times and only
    // avoided the DRAM cost when siblings overlapped in cache.
    const int row = group_id;
    int output_row[TopK];
    bool any_live = false;
#pragma unroll
    for (int i = 0; i < TopK; ++i) {
      output_row[i] = unpermuted_row_to_permuted_row[row * TopK + i];
      any_live |= (output_row[i] >= 0);
    }
    if (!any_live) return;

    int stride = local_range * ElemsPerItem;
    int copy_stride = local_range * CopyElemsPerItem;
    int loop_count = (hidden_size + copy_stride - 1) / copy_stride;

    for (int l = 0; l < loop_count; ++l) {
      const int subgroup_start = l * copy_stride + (local_id / WARP_SIZE) *
                                                       WARP_SIZE *
                                                       CopyElemsPerItem;
      if (subgroup_start + WARP_SIZE * CopyElemsPerItem <= hidden_size) {
        // A whole subgroup loads a contiguous block once, then stores it to
        // every live destination row.
        using CopyVector = sycl::vec<uint32_t, CopyWords>;
        namespace syclex = sycl::ext::oneapi::experimental;
        constexpr auto properties = syclex::properties{
            syclex::contiguous_memory, syclex::alignment<sizeof(CopyVector)>};
        CopyVector values;
        const auto* source = reinterpret_cast<const uint32_t*>(
            hidden_states + row * static_cast<int64_t>(hidden_size) +
            subgroup_start);
        syclex::group_load(subgroup, source, values, properties);
#pragma unroll
        for (int i = 0; i < TopK; ++i) {
          if (output_row[i] < 0) continue;
          auto* destination = reinterpret_cast<uint32_t*>(
              remapped_hidden_states +
              output_row[i] * static_cast<int64_t>(hidden_size) +
              subgroup_start);
          syclex::group_store(subgroup, values, destination, properties);
        }
        continue;
      }
      using load_type = sycl::vec<TA, ElemsPerItem>;
#pragma unroll
      for (int v = 0; v < CopyElemsPerItem / ElemsPerItem; ++v) {
        const int offset =
            subgroup_start +
            (local_id % WARP_SIZE + v * WARP_SIZE) * ElemsPerItem;
        if (offset >= hidden_size) continue;
        load_type data = *reinterpret_cast<load_type*>(
            hidden_states + row * static_cast<int64_t>(hidden_size) + offset);
#pragma unroll
        for (int i = 0; i < TopK; ++i) {
          if (output_row[i] < 0) continue;
          *reinterpret_cast<load_type*>(
              remapped_hidden_states +
              output_row[i] * static_cast<int64_t>(hidden_size) + offset) =
              data;
        }
      }
    }

    if (hidden_states_scales != nullptr &&
        remapped_hidden_states_scales != nullptr) {
      int64_t scaled_hidden_size = hidden_size / block_k;

      if (expert_scale_desc == nullptr) {
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
              if (output_row[i] < 0) continue;
              *(reinterpret_cast<load_type*>(
                  remapped_hidden_states_scales +
                  output_row[i] * scaled_hidden_size + start_id)) = data;
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
              if (output_row[i] < 0) continue;
#pragma unroll
              for (int e = 0; e < remained_elems; ++e) {
                remapped_hidden_states_scales
                    [output_row[i] * scaled_hidden_size + start_id + e] =
                        data[e];
              }
            }
          }
        }
      }
    }
  }

 private:
  TA* hidden_states;
  TS* hidden_states_scales;
  TA* remapped_hidden_states;
  TS* remapped_hidden_states_scales;
  int* unpermuted_row_to_permuted_row;
  const int hidden_size;
  const int block_k;
  int* expert_scale_desc;
};

// Fused 2-kernel path (RowsPerExpertCount + this). Computes the expert cumsum
// in SLM and scatters the padded MN-major scales directly, avoiding the
// ExpertCumsum and ScaleRemap launches of the split path. Preferred for small
// row counts where those extra launches dominate.
template <typename TA, typename TS, int TopK>
class RemapHiddenStatesFused {
 public:
  RemapHiddenStatesFused(
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
      const int local_experts_num,
      int* expert_scale_desc)
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
        local_experts_num(local_experts_num),
        expert_scale_desc(expert_scale_desc) {}

  static constexpr int GroupWorkItem = 256;
  static constexpr int WARP_SIZE = 16;
  static constexpr int ElemsPerItem = sizeof(float) * 4 / sizeof(TA);

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

    // Second SLM region: exclusive prefix sum of round_up_4(rows_per_expert),
    // i.e. the per-expert base row of the padded MN-major scale surface. Only
    // populated when that layout is requested.
    const bool padded_scales = expert_scale_desc != nullptr;
    const int desc_len = local_experts_num + 1;
    int32_t* expert_pad_cumsum_ptr = expert_cumsum_ptr + desc_len;

    if (local_id == 0) {
      expert_cumsum_ptr[0] = 0;
      if (padded_scales) {
        expert_pad_cumsum_ptr[0] = 0;
      }
    }
    for (int i = local_id; i < local_experts_num; i += local_range) {
      expert_cumsum_ptr[i + 1] = rows_per_expert[i];
      if (padded_scales) {
        expert_pad_cumsum_ptr[i + 1] = (rows_per_expert[i] + 3) & ~3;
      }
    }

    sycl::group_barrier(item.get_group());

    // Scanning desc_len entries makes index e the exclusive prefix for every
    // expert and index local_experts_num the total, which is exactly the
    // descriptor consumers need.
    sycl::joint_inclusive_scan(
        item.get_group(),
        expert_cumsum_ptr,
        expert_cumsum_ptr + desc_len,
        expert_cumsum_ptr,
        sycl::plus<int>{});

    if (padded_scales) {
      sycl::joint_inclusive_scan(
          item.get_group(),
          expert_pad_cumsum_ptr,
          expert_pad_cumsum_ptr + desc_len,
          expert_pad_cumsum_ptr,
          sycl::plus<int>{});

      // Every work-group computes the same prefixes; group 0 publishes them so
      // that downstream consumers need neither a scan kernel nor a reorder
      // pass. Row 0 is the source row prefix, row 1 the padded scale row
      // prefix; per-expert row counts and leading dimensions are the
      // successive differences.
      if (group_id == 0) {
        for (int i = local_id; i < desc_len; i += local_range) {
          expert_scale_desc[i] = expert_cumsum_ptr[i];
          expert_scale_desc[desc_len + i] = expert_pad_cumsum_ptr[i];
        }
      }
    }

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

#pragma unroll
    for (int i = 0; i < TopK; ++i) {
      local_expert_id[i] = to_local_expert_id(global_expert_id[i], expert_map);
    }

    int rows_offset[TopK];
    int scale_row_local[TopK];
    int scale_block_off[TopK];
    int scale_block_ld[TopK];
#pragma unroll
    for (int i = 0; i < TopK; ++i) {
      if (local_expert_id[i] != -1) {
        int cumsum_offset =
            local_expert_id[i] == 0 ? 0 : expert_cumsum_ptr[local_expert_id[i]];
        int local_row = unpermuted_row_to_permuted_row[row * TopK + i];
        rows_offset[i] = local_row + cumsum_offset;
        scale_row_local[i] = local_row;
        if (padded_scales) {
          scale_block_off[i] = expert_pad_cumsum_ptr[local_expert_id[i]];
          scale_block_ld[i] = (rows_per_expert[local_expert_id[i]] + 3) & ~3;
        } else {
          scale_block_off[i] = 0;
          scale_block_ld[i] = 0;
        }
      } else {
        rows_offset[i] = -1;
        scale_row_local[i] = -1;
        scale_block_off[i] = 0;
        scale_block_ld[i] = 0;
      }
    }

    sycl::group_barrier(item.get_group());

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

      if (expert_scale_desc != nullptr) {
        // Scatter directly into the per-expert MN-major, M-padded-to-4 surface
        // consumed by the mxfp grouped-GEMM mainloop, so no separate reorder
        // pass is needed. Element (m_local, k) of expert e lives at
        //   pad_prefix[e] * scale_k + k * round_up_4(rows[e]) + m_local
        for (int k = local_id; k < scaled_hidden_size; k += local_range) {
          TS data = hidden_states_scales[row * scaled_hidden_size + k];
#pragma unroll
          for (int i = 0; i < TopK; ++i) {
            if (rows_offset[i] == -1) continue;
            int64_t dst =
                static_cast<int64_t>(scale_block_off[i]) * scaled_hidden_size +
                static_cast<int64_t>(k) * scale_block_ld[i] +
                scale_row_local[i];
            remapped_hidden_states_scales[dst] = data;
          }
        }
      } else {
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
  int* expert_scale_desc;
};

template <typename TS>
class ScaleRemap {
 public:
  ScaleRemap(
      TS* hidden_states_scales,
      TS* remapped_hidden_states_scales,
      int* permuted_to_unpermuted_row,
      int* rows_per_expert,
      int* expert_scale_desc,
      const int64_t scaled_hidden_size,
      const int local_experts_num)
      : hidden_states_scales(hidden_states_scales),
        remapped_hidden_states_scales(remapped_hidden_states_scales),
        permuted_to_unpermuted_row(permuted_to_unpermuted_row),
        rows_per_expert(rows_per_expert),
        expert_scale_desc(expert_scale_desc),
        scaled_hidden_size(scaled_hidden_size),
        local_experts_num(local_experts_num) {}

  static constexpr int GroupWorkItem = 64;
  static constexpr int WARP_SIZE = 16;
  // Rows (m) each work-item transposes per column, matches a 16-byte vector.
  static constexpr int ScaleElemsPerItem = sizeof(float) * 4 / sizeof(TS);
  // Columns (k) handled by one work-group. Reading KBlock contiguous scales of
  // a gathered row at once amortizes the per-row gather across KBlock columns
  // (the whole row shares a cache line) instead of touching it one byte per
  // pass. Kept small so the (expert, k-block) grid stays highly parallel.
  static constexpr int KBlock = 4;

  static inline sycl::nd_range<1>
  get_nd_range(const int64_t scaled_hidden_size, const int local_experts_num) {
    int64_t k_blocks = (scaled_hidden_size + KBlock - 1) / KBlock;
    sycl::range<1> local(GroupWorkItem);
    sycl::range<1> group(static_cast<size_t>(k_blocks) * local_experts_num);
    return sycl::nd_range<1>(local * group, local);
  }

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    auto local_id = item.get_local_id(0);
    auto local_range = item.get_local_range(0);
    int64_t group_id = static_cast<int64_t>(item.get_group(0));

    int64_t k_blocks = (scaled_hidden_size + KBlock - 1) / KBlock;
    int kb = static_cast<int>(group_id % k_blocks);
    int e = static_cast<int>(group_id / k_blocks);
    int k0 = kb * KBlock;
    int k_cnt = KBlock;
    if (scaled_hidden_size - k0 < KBlock) {
      k_cnt = static_cast<int>(scaled_hidden_size - k0);
    }

    const int desc_len = local_experts_num + 1;
    int raw_base = expert_scale_desc[e];
    int pad_base = expert_scale_desc[desc_len + e];
    int rows_e = rows_per_expert[e];
    int ld = (rows_e + 3) & ~3;

    if (rows_e == 0) return;

    int64_t pad_col_base = static_cast<int64_t>(pad_base) * scaled_hidden_size;
    const bool row_stride_aligned = (scaled_hidden_size % KBlock == 0);

    int m_stride = local_range * ScaleElemsPerItem;
    for (int m_start = local_id * ScaleElemsPerItem; m_start < rows_e;
         m_start += m_stride) {
      int remained = rows_e - m_start;
      int m_cnt = remained >= ScaleElemsPerItem ? ScaleElemsPerItem : remained;

      // Gather the source row indices for this m-tile once; reused by every
      // column in the k-block.
      int orig_row[ScaleElemsPerItem];
#pragma unroll
      for (int mi = 0; mi < ScaleElemsPerItem; ++mi) {
        if (mi < m_cnt) {
          orig_row[mi] = permuted_to_unpermuted_row[raw_base + m_start + mi];
        }
      }

      if (m_cnt == ScaleElemsPerItem && k_cnt == KBlock) {
        // Fast path: one wide load pulls KBlock contiguous scales per gathered
        // row (amortizing the gather), transpose in registers, then write each
        // column back coalesced along m.
        using read_type = sycl::vec<TS, KBlock>;
        using write_type = sycl::vec<TS, ScaleElemsPerItem>;
        read_type rows[ScaleElemsPerItem];
        if (row_stride_aligned) {
#pragma unroll
          for (int mi = 0; mi < ScaleElemsPerItem; ++mi) {
            rows[mi] = *(reinterpret_cast<read_type*>(
                hidden_states_scales +
                static_cast<int64_t>(orig_row[mi]) * scaled_hidden_size + k0));
          }
        } else {
#pragma unroll
          for (int mi = 0; mi < ScaleElemsPerItem; ++mi) {
            int64_t src =
                static_cast<int64_t>(orig_row[mi]) * scaled_hidden_size + k0;
#pragma unroll
            for (int kk = 0; kk < KBlock; ++kk) {
              rows[mi][kk] = hidden_states_scales[src + kk];
            }
          }
        }
#pragma unroll
        for (int kk = 0; kk < KBlock; ++kk) {
          write_type w;
#pragma unroll
          for (int mi = 0; mi < ScaleElemsPerItem; ++mi) {
            w[mi] = rows[mi][kk];
          }
          *(reinterpret_cast<write_type*>(
              remapped_hidden_states_scales + pad_col_base +
              static_cast<int64_t>(k0 + kk) * ld + m_start)) = w;
        }
      } else {
        // Tail: partial m-tile or partial k-block, scalar fallback.
        for (int kk = 0; kk < k_cnt; ++kk) {
          int64_t dst_col_base =
              pad_col_base + static_cast<int64_t>(k0 + kk) * ld;
          for (int mi = 0; mi < m_cnt; ++mi) {
            remapped_hidden_states_scales[dst_col_base + m_start + mi] =
                hidden_states_scales
                    [static_cast<int64_t>(orig_row[mi]) * scaled_hidden_size +
                     k0 + kk];
          }
        }
      }
    }
  }

 private:
  TS* hidden_states_scales;
  TS* remapped_hidden_states_scales;
  int* permuted_to_unpermuted_row;
  int* rows_per_expert;
  int* expert_scale_desc;
  const int64_t scaled_hidden_size;
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
    int* expert_cumsum,
    int* unpermuted_row_to_permuted_row,
    int* permuted_to_unpermuted_row,
    void* topk_ids,
    bool is_topk_ids_int32,
    const int num_rows,
    const int hidden_size,
    const int block_k,
    const int total_experts_num,
    const int local_experts_num,
    int* expert_scale_desc,
    bool use_split,
    sycl::queue& queue) {
  // The cumsum kernels keep two (local_experts_num + 1)-entry prefix tables
  // in SLM, which is more than RowsPerExpertCount needs.
  TORCH_CHECK(
      2 * (static_cast<size_t>(local_experts_num) + 1) * sizeof(int32_t) <=
          queue.get_device().get_info<sycl::info::device::local_mem_size>(),
      "local_experts_num exceeds the device local memory capacity");
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

  if (!use_split) {
    // Fused 2-kernel path: compute the cumsum in SLM and scatter scales inline,
    // avoiding the split path's prefix, mapping, and scale-remap launches.
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<int32_t, 1> slm(
          sycl::range<1>(2 * (local_experts_num + 1)), cgh);
      cgh.parallel_for(
          RemapHiddenStatesFused<TA, TS, TopK>::get_nd_range(
              num_rows, hidden_size),
          RemapHiddenStatesFused<TA, TS, TopK>{
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
              local_experts_num,
              expert_scale_desc});
    });
    return;
  }

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> slm(
        sycl::range<1>(2 * (local_experts_num + 1)), cgh);
    cgh.parallel_for(
        ExpertCumsum::get_nd_range(local_experts_num),
        ExpertCumsum{
            slm,
            rows_per_expert,
            expert_cumsum,
            local_experts_num,
            expert_scale_desc});
  });

  vllm::xpu::sycl_kernel_submit(
      queue,
      sycl::range<1>(num_rows * TopK),
      FinalizeRowMapping{
          expert_map,
          unpermuted_row_to_permuted_row,
          permuted_to_unpermuted_row,
          expert_cumsum,
          topk_ids,
          is_topk_ids_int32,
          TopK});

  vllm::xpu::sycl_kernel_submit(
      queue,
      RemapHiddenStates<TA, TS, TopK>::get_nd_range(num_rows, hidden_size),
      RemapHiddenStates<TA, TS, TopK>{
          hidden_states,
          hidden_states_scales,
          remapped_hidden_states,
          remapped_hidden_states_scales,
          unpermuted_row_to_permuted_row,
          hidden_size,
          block_k,
          expert_scale_desc});

  if (expert_scale_desc != nullptr) {
    int64_t scaled_hidden_size = hidden_size / block_k;
    vllm::xpu::sycl_kernel_submit(
        queue,
        ScaleRemap<TS>::get_nd_range(scaled_hidden_size, local_experts_num),
        ScaleRemap<TS>{
            hidden_states_scales,
            remapped_hidden_states_scales,
            permuted_to_unpermuted_row,
            rows_per_expert,
            expert_scale_desc,
            scaled_hidden_size,
            local_experts_num});
  }
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
    int64_t local_experts_num,
    const c10::optional<torch::Tensor>& expert_scale_desc) {
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
    if (expert_scale_desc.has_value()) {
      // Padded MN-major surface: [total_padded_rows, scale_k] with
      // total_padded_rows >= num_rows * TopK + 3 * local_experts_num, since
      // each expert's M is rounded up to a multiple of 4.
      TORCH_CHECK(
          remapped_hidden_states_scales->scalar_type() == at::kFloat8_e8m0fnu,
          "the padded MN-major scale layout requires float8_e8m0fnu scales");
      TORCH_CHECK(
          remapped_hidden_states_scales->size(0) >=
                  num_rows * TopK + 3 * local_experts_num &&
              hidden_states_scales->size(1) ==
                  remapped_hidden_states_scales->size(1),
          "remapped_hidden_states_scales must be [num_rows * TopK + 3 * "
          "local_experts_num, hidden_size // block_k] when "
          "expert_scale_desc is given");
    } else {
      TORCH_CHECK(
          remapped_hidden_states_scales->size(0) == num_rows * TopK &&
              hidden_states_scales->size(1) ==
                  remapped_hidden_states_scales->size(1),
          "remapped_hidden_states_scales must be [num_rows * "
          "TopK, hidden_size // block_k]");
    }
  } else {
    TORCH_CHECK(
        !expert_scale_desc.has_value(),
        "expert_scale_desc requires hidden_states_scales");
  }

  if (expert_scale_desc.has_value()) {
    TORCH_CHECK(
        expert_scale_desc->scalar_type() == torch::kInt32 &&
            expert_scale_desc->is_contiguous() &&
            expert_scale_desc->dim() == 2 && expert_scale_desc->size(0) == 2 &&
            expert_scale_desc->size(1) == local_experts_num + 1,
        "expert_scale_desc must be a contiguous int32 [2, local_experts_num + "
        "1] tensor");
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

  // Small row counts don't have enough rows per expert to amortize the split
  // path's prefix, mapping, and scale-remap launches (plus the inverse-map
  // allocation), so use the fused 2-kernel implementation there. Large row
  // counts benefit from independent route copies and ScaleRemap's coalesced
  // MN-major transpose.
  constexpr int kSplitRowThreshold = 512;
  const bool use_split = num_rows > kSplitRowThreshold;

  auto expert_cumsum = torch::empty_like(rows_per_expert);
  auto permuted_to_unpermuted_row =
      (use_split && expert_scale_desc.has_value())
          ? torch::empty({num_rows * TopK}, rows_per_expert.options())
          : torch::Tensor();

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
      reinterpret_cast<int*>(expert_cumsum.data_ptr()),                       \
      reinterpret_cast<int*>(unpermuted_row_to_permuted_row.data_ptr()),      \
      (use_split && expert_scale_desc.has_value())                            \
          ? reinterpret_cast<int*>(permuted_to_unpermuted_row.data_ptr())     \
          : nullptr,                                                          \
      reinterpret_cast<void*>(topk_ids.data_ptr()),                           \
      topk_ids.scalar_type() == torch::kInt32,                                \
      num_rows,                                                               \
      hidden_size,                                                            \
      block_k,                                                                \
      total_experts_num,                                                      \
      local_experts_num,                                                      \
      expert_scale_desc.has_value()                                           \
          ? reinterpret_cast<int*>(expert_scale_desc->data_ptr())             \
          : nullptr,                                                          \
      use_split,                                                              \
      queue);

#define DISPATCH_TOPK_LAUNCH(TA, TS, TopK)              \
  if (TopK == 1) {                                      \
    LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 1);              \
  } else if (TopK == 2) {                               \
    LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 2);              \
  } else if (TopK == 3) {                               \
    LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 3);              \
  } else if (TopK == 4) {                               \
    LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 4);              \
  } else if (TopK == 6) {                               \
    LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 6);              \
  } else if (TopK == 7) {                               \
    LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 7);              \
  } else if (TopK == 8) {                               \
    LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 8);              \
  } else if (TopK == 10) {                              \
    LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 10);             \
  } else if (TopK == 12) {                              \
    LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 12);             \
  } else if (TopK == 16) {                              \
    LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 16);             \
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
    if (!hidden_states_scales.has_value() ||
        hidden_states_scales->scalar_type() == torch::kFloat32) {
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
