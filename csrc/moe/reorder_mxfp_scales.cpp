#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"

namespace vllm {
namespace moe {

// Reorders and pads the per-expert mxfp (e8m0) activation scales into the
// MN-major, M-padded-to-4 layout expected by the optimized mxfp grouped-GEMM
// mainloop (see grouped_gemm/xe_3/collective/moe_gemm_array_cooperative.hpp,
// which advances the per-expert scale offset by (rows + 3) & ~3).
//
// Source `A_scales` is [total_rows, scale_k] row-major, concatenated per
// expert. For expert e with M = rows_per_expert[e] rows, its scale block is
// written into `A_scale_k` starting at cumulative padded offset
//   dst_off = sum_{i<e} round_up_4(rows_per_expert[i])
// as a column-major-along-M surface with leading dim padded_M = round_up_4(M):
//   A_scale_k[dst_off + m + k * padded_M] = A_scales[src_off + m, k]
// for m in [0, M), k in [0, scale_k). The padding rows [M, padded_M) are left
// as zeros (guaranteed by the caller's torch::zeros allocation).
//
// `rows_per_expert` stays on device, so no device->host sync is required.

// Phase 1: single work-group serial scan producing per-expert source-row and
// padded-destination-row prefix offsets. num_experts is small (tens to low
// hundreds); this runs once and feeds the data-parallel phase.
class ReorderMxfpScalesPrefix {
 public:
  ReorderMxfpScalesPrefix(
      const int* rows_per_expert,
      int* src_prefix,
      int* dst_prefix,
      const int num_experts)
      : rows_per_expert(rows_per_expert),
        src_prefix(src_prefix),
        dst_prefix(dst_prefix),
        num_experts(num_experts) {}

  static constexpr int GroupWorkItem = 1;

  static inline sycl::nd_range<1> get_nd_range() {
    return sycl::nd_range<1>(GroupWorkItem, GroupWorkItem);
  }

  void operator()(sycl::nd_item<1>) const {
    int src_off = 0;
    int dst_off = 0;
    for (int e = 0; e < num_experts; ++e) {
      src_prefix[e] = src_off;
      dst_prefix[e] = dst_off;
      int r = rows_per_expert[e];
      src_off += r;
      dst_off += (r + 3) & ~3;
    }
  }

 private:
  const int* rows_per_expert;
  int* src_prefix;
  int* dst_prefix;
  const int num_experts;
};

// Phase 2: data-parallel transpose. The grid is sized to the total number of
// (padded) destination rows, so parallelism tracks the data volume rather than
// the number of experts. Each work-group owns a contiguous tile of RowsPerGroup
// destination rows and binary-searches the padded prefix to find its expert.
class ReorderMxfpScalesCopy {
 public:
  ReorderMxfpScalesCopy(
      const uint8_t* src,
      uint8_t* dst,
      const int* rows_per_expert,
      const int* src_prefix,
      const int* dst_prefix,
      const int scale_k,
      const int num_experts,
      const int num_row_groups)
      : src(src),
        dst(dst),
        rows_per_expert(rows_per_expert),
        src_prefix(src_prefix),
        dst_prefix(dst_prefix),
        scale_k(scale_k),
        num_experts(num_experts),
        num_row_groups(num_row_groups) {}

  static constexpr int GroupWorkItem = 256;
  // Each work-group transposes a tile of this many destination rows.
  static constexpr int RowsPerGroup = 64;

  static inline sycl::nd_range<1> get_nd_range(const int num_row_groups) {
    return sycl::nd_range<1>(
        static_cast<size_t>(num_row_groups) * GroupWorkItem, GroupWorkItem);
  }

  void operator()(sycl::nd_item<1> item) const {
    const int group_id = item.get_group(0);
    const int local_id = item.get_local_id(0);
    const int local_range = item.get_local_range(0);

    // The destination-row range this work-group is responsible for.
    const int row_begin = group_id * RowsPerGroup;

    // Binary-search the padded destination prefix to find the first expert
    // whose block starts at or before row_begin.
    int lo = 0;
    int hi = num_experts;
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      if (dst_prefix[mid] <= row_begin) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    int expert = lo - 1;

    // Walk experts starting at `expert`, handling each expert's overlap with
    // this work-group's [row_begin, row_begin + RowsPerGroup) tile. A tile may
    // span more than one (small) expert.
    int tile_end = row_begin + RowsPerGroup;
    while (expert < num_experts && dst_prefix[expert] < tile_end) {
      const int rows = rows_per_expert[expert];
      if (rows == 0) {
        ++expert;
        continue;
      }
      const int padded_rows = (rows + 3) & ~3;
      const int e_dst_off = dst_prefix[expert];
      const int e_src_off = src_prefix[expert];

      // Local (within-expert) source-row range this work-group covers. Padding
      // rows [rows, padded_rows) are skipped (left zero by the caller).
      int m_lo = row_begin - e_dst_off;
      if (m_lo < 0) m_lo = 0;
      int m_hi = tile_end - e_dst_off;
      if (m_hi > rows) m_hi = rows;

      if (m_lo < m_hi) {
        const uint8_t* src_base =
            src + static_cast<int64_t>(e_src_off) * scale_k;
        uint8_t* dst_base = dst + static_cast<int64_t>(e_dst_off) * scale_k;
        const int count = (m_hi - m_lo) * scale_k;
        // src is (rows, scale_k) row-major; dst block is
        // (scale_k, padded_rows) row-major.
        for (int idx = local_id; idx < count; idx += local_range) {
          int mm = idx / scale_k;
          int k = idx - mm * scale_k;
          int m = m_lo + mm;
          dst_base[static_cast<int64_t>(k) * padded_rows + m] =
              src_base[static_cast<int64_t>(m) * scale_k + k];
        }
      }
      ++expert;
    }
  }

 private:
  const uint8_t* src;
  uint8_t* dst;
  const int* rows_per_expert;
  const int* src_prefix;
  const int* dst_prefix;
  const int scale_k;
  const int num_experts;
  const int num_row_groups;
};

}  // namespace moe
}  // namespace vllm

torch::Tensor reorder_mxfp_scales(
    const torch::Tensor& A_scales,
    const torch::Tensor& rows_per_expert,
    const int64_t total_padded_rows) {
  TORCH_CHECK(A_scales.dim() == 2, "A_scales must be 2D [total_rows, scale_k]");
  TORCH_CHECK(
      A_scales.scalar_type() == at::kFloat8_e8m0fnu,
      "A_scales must be float8_e8m0fnu");
  TORCH_CHECK(A_scales.is_contiguous(), "A_scales must be contiguous");
  TORCH_CHECK(
      rows_per_expert.scalar_type() == at::kInt,
      "rows_per_expert must be int32");
  TORCH_CHECK(rows_per_expert.is_contiguous(), "rows_per_expert must be int32");

  const int num_experts = rows_per_expert.size(0);
  const int scale_k = A_scales.size(1);

  auto A_scale_k =
      torch::zeros({total_padded_rows, scale_k}, A_scales.options());

  if (num_experts == 0 || scale_k == 0 || total_padded_rows == 0) {
    return A_scale_k;
  }

  auto int_opts = rows_per_expert.options();
  auto src_prefix = torch::empty({num_experts}, int_opts);
  auto dst_prefix = torch::empty({num_experts}, int_opts);

  const int num_row_groups =
      (total_padded_rows + vllm::moe::ReorderMxfpScalesCopy::RowsPerGroup - 1) /
      vllm::moe::ReorderMxfpScalesCopy::RowsPerGroup;

  auto& queue = vllm::xpu::vllmGetQueue();

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        vllm::moe::ReorderMxfpScalesPrefix::get_nd_range(),
        vllm::moe::ReorderMxfpScalesPrefix{
            reinterpret_cast<const int*>(rows_per_expert.data_ptr()),
            reinterpret_cast<int*>(src_prefix.data_ptr()),
            reinterpret_cast<int*>(dst_prefix.data_ptr()),
            num_experts});
  });

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        vllm::moe::ReorderMxfpScalesCopy::get_nd_range(num_row_groups),
        vllm::moe::ReorderMxfpScalesCopy{
            reinterpret_cast<const uint8_t*>(A_scales.data_ptr()),
            reinterpret_cast<uint8_t*>(A_scale_k.data_ptr()),
            reinterpret_cast<const int*>(rows_per_expert.data_ptr()),
            reinterpret_cast<const int*>(src_prefix.data_ptr()),
            reinterpret_cast<const int*>(dst_prefix.data_ptr()),
            scale_k,
            num_experts,
            num_row_groups});
  });

  return A_scale_k;
}
