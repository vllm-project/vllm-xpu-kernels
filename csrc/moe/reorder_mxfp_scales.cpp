#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"
#include "sycl_launch.h"

namespace vllm {
namespace moe {

// Reorders and pads the per-expert mxfp (e8m0) activation scales into the
// MN-major, M-padded-to-4 layout expected by the optimized mxfp grouped-GEMM
// mainloop (see grouped_gemm/xe_3/kernel/xe_moe_gemm_greedy.hpp, which
// advances the per-expert scale offset by (rows + 3) & ~3).
//
// Source `A_scales` is [total_rows, scale_k] row-major, concatenated per
// expert. For expert e with M = rows_per_expert[e] rows, its scale block is
// written into `A_scale_k` starting at cumulative padded offset
//   dst_off = sum_{i<e} round_up_4(rows_per_expert[i])
// as a column-major-along-M surface with leading dim padded_M = round_up_4(M):
//   A_scale_k[dst_off + m + k * padded_M] = A_scales[src_off + m, k]
// for m in [0, M), k in [0, scale_k). The padding rows [M, padded_M) are
// written as zeros here; everything past the last expert is left to the
// caller's torch::zeros allocation.
//
// `rows_per_expert` stays on device, so no device->host sync is required.
//
// The whole op is one kernel. Each work-group
//   1. scans rows_per_expert in SLM to get the per-expert source and padded
//      destination prefixes (num_experts is tens to a few hundred, so this is
//      a handful of group scans and far cheaper than a dependent launch),
//   2. owns a tile of RowsPerGroup destination rows and binary-searches the
//      SLM prefix for the first expert overlapping it,
//   3. transposes each overlapping expert segment through SLM in KChunk-wide
//      column slabs, so global loads walk source rows along k and global
//      stores walk destination rows along m. With 16-byte aligned rows the
//      loads are 16-byte vectors and the stores are 4-byte (4 rows) words.
//
// Work-group 0 also publishes the int32 [2, num_experts + 1]
// expert_scale_desc that remap_hidden_states produces when asked (row 0:
// exclusive prefix of rows, row 1: exclusive prefix of round_up_4(rows),
// totals appended). Callers whose permutation happens elsewhere (e.g. in a
// dispatch collective) can hand it to the fused act+quant kernel so gemm2's
// scales are produced in GEMM layout without a second reorder.
template <bool VecLoad>
class ReorderMxfpScalesKernel : public vllm::xpu::SyclKernelConfigConvention {
 public:
  static constexpr int GroupWorkItem = 256;
  // Destination rows owned by one work-group.
  static constexpr int RowsPerGroup = 64;
  // Columns (scale_k entries) staged through SLM per pass.
  static constexpr int KChunk = 64;
  static constexpr int KChunkShift = 6;
  // SLM row pitch. Rows stay 16-byte aligned for the vector SLM stores, and
  // the extra 16 bytes stagger rows across banks for the column-wise reads.
  static constexpr int SlmPitch = KChunk + 16;
  static constexpr int TileBytes = RowsPerGroup * SlmPitch;
  static constexpr int VecBytes = 16;
  static constexpr int VecPerRow = KChunk / VecBytes;
  static constexpr int VecPerRowShift = 2;
  static constexpr int Quads = RowsPerGroup / 4;
  static constexpr int QuadShift = 4;

  static_assert((1 << KChunkShift) == KChunk);
  static_assert((1 << VecPerRowShift) == VecPerRow);
  static_assert((1 << QuadShift) == Quads);
  static_assert(GroupWorkItem % Quads == 0);
  static_assert(GroupWorkItem % RowsPerGroup == 0);

  ReorderMxfpScalesKernel(
      const uint8_t* src,
      uint8_t* dst,
      const int* rows_per_expert,
      int* desc,
      const int scale_k,
      const int num_experts)
      : src(src),
        dst(dst),
        rows_per_expert(rows_per_expert),
        desc(desc),
        scale_k(scale_k),
        num_experts(num_experts) {}

  // Shared local memory has to be bound while the command group is being
  // built, so sycl_kernel_submit() calls this back on the handler.
  void sycl_ker_config_convention(sycl::handler& cgh) {
    tile_slm = sycl::local_accessor<uint8_t, 1>(sycl::range<1>(TileBytes), cgh);
    prefix_slm = sycl::local_accessor<int, 1>(
        sycl::range<1>(2 * (num_experts + 1)), cgh);
  }

  static inline sycl::nd_range<1> get_nd_range(const int num_row_groups) {
    return sycl::nd_range<1>(
        static_cast<size_t>(num_row_groups) * GroupWorkItem, GroupWorkItem);
  }

  void operator()(sycl::nd_item<1> item) const {
    auto group = item.get_group();
    const int group_id = item.get_group(0);
    const int local_id = item.get_local_id(0);
    uint8_t* tile =
        tile_slm.template get_multi_ptr<sycl::access::decorated::no>().get();
    // [0, E]: source prefix, [E + 1, 2E + 1]: padded destination prefix,
    // each with the total appended (the same layout as expert_scale_desc).
    int* src_prefix =
        prefix_slm.template get_multi_ptr<sycl::access::decorated::no>().get();
    int* dst_prefix = src_prefix + num_experts + 1;

    // ---- Phase 1: per-expert prefixes, computed by every work-group. ----
    int src_carry = 0;
    int dst_carry = 0;
    for (int base = 0; base < num_experts; base += GroupWorkItem) {
      const int e = base + local_id;
      const int r = e < num_experts ? rows_per_expert[e] : 0;
      const int pr = (r + 3) & ~3;
      const int s_excl =
          sycl::exclusive_scan_over_group(group, r, sycl::plus<int>());
      const int d_excl =
          sycl::exclusive_scan_over_group(group, pr, sycl::plus<int>());
      if (e < num_experts) {
        src_prefix[e] = src_carry + s_excl;
        dst_prefix[e] = dst_carry + d_excl;
      }
      src_carry += sycl::reduce_over_group(group, r, sycl::plus<int>());
      dst_carry += sycl::reduce_over_group(group, pr, sycl::plus<int>());
    }
    if (local_id == 0) {
      src_prefix[num_experts] = src_carry;
      dst_prefix[num_experts] = dst_carry;
    }
    sycl::group_barrier(group);

    if (desc != nullptr && group_id == 0) {
      for (int i = local_id; i < 2 * (num_experts + 1); i += GroupWorkItem) {
        desc[i] = src_prefix[i];
      }
    }

    // ---- Phase 2: SLM transpose of this work-group's destination tile. ----
    const int row_begin = group_id * RowsPerGroup;
    const int tile_end = row_begin + RowsPerGroup;
    if (row_begin >= dst_carry) return;  // uniform across the work-group

    // Last expert whose padded block starts at or before row_begin. Experts
    // with zero rows share a prefix with their successor and are skipped.
    int lo = 0;
    int hi = num_experts;
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if (dst_prefix[mid] <= row_begin) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    int expert = lo - 1;

    // All values steering the loops below are uniform across the
    // work-group, so every work-item reaches every barrier.
    for (; expert < num_experts && dst_prefix[expert] < tile_end; ++expert) {
      const int e_src_off = src_prefix[expert];
      const int rows = src_prefix[expert + 1] - e_src_off;
      if (rows == 0) continue;
      const int e_dst_off = dst_prefix[expert];
      const int padded_rows = dst_prefix[expert + 1] - e_dst_off;

      // Within-expert row range [m_lo, m_hi) covered by this tile, padding
      // rows included so they are written as zeros. Both ends are multiples
      // of 4 because row_begin, the tile size and every padded block are.
      const int m_lo = sycl::max(row_begin - e_dst_off, 0);
      const int m_hi = sycl::min(tile_end - e_dst_off, padded_rows);
      const int nm = m_hi - m_lo;
      // Rows of the segment that come from the source; the rest is padding.
      const int nm_valid = sycl::max(sycl::min(rows - m_lo, nm), 0);

      const uint8_t* src_base =
          src + (static_cast<int64_t>(e_src_off) + m_lo) * scale_k;
      // dst block is (scale_k, padded_rows) row-major.
      uint8_t* dst_base =
          dst + static_cast<int64_t>(e_dst_off) * scale_k + m_lo;

      for (int k0 = 0; k0 < scale_k; k0 += KChunk) {
        const int kc = sycl::min(KChunk, scale_k - k0);

        // Load the [nm_valid, kc] source slab, zero-filling padding rows.
        if constexpr (VecLoad) {
          // scale_k % 16 == 0, so every chunk is a whole number of vectors
          // and every source row starts 16-byte aligned.
          const int vec_per_row = kc / VecBytes;
          const int count = nm << VecPerRowShift;
          for (int idx = local_id; idx < count; idx += GroupWorkItem) {
            const int mm = idx >> VecPerRowShift;
            const int vv = idx & (VecPerRow - 1);
            if (vv < vec_per_row) {
              sycl::uint4 v{0, 0, 0, 0};
              if (mm < nm_valid) {
                v = *reinterpret_cast<const sycl::uint4*>(
                    src_base + static_cast<int64_t>(mm) * scale_k + k0 +
                    vv * VecBytes);
              }
              *reinterpret_cast<sycl::uint4*>(
                  tile + mm * SlmPitch + vv * VecBytes) = v;
            }
          }
        } else {
          const int count = nm << KChunkShift;
          for (int idx = local_id; idx < count; idx += GroupWorkItem) {
            const int mm = idx >> KChunkShift;
            const int kk = idx & (KChunk - 1);
            if (kk < kc) {
              tile[mm * SlmPitch + kk] =
                  mm < nm_valid
                      ? src_base[static_cast<int64_t>(mm) * scale_k + k0 + kk]
                      : uint8_t{0};
            }
          }
        }
        sycl::group_barrier(group);

        // Store: each work-item packs 4 consecutive rows of one column into a
        // 32-bit word. dst_base, padded_rows and m are all multiples of 4, so
        // the word stores are aligned.
        const int nq = nm >> 2;
        for (int idx = local_id; idx < Quads * KChunk; idx += GroupWorkItem) {
          const int q = idx & (Quads - 1);
          const int kk = idx >> QuadShift;
          if (q < nq && kk < kc) {
            const uint8_t* t = tile + (q * 4) * SlmPitch + kk;
            const uint32_t w = static_cast<uint32_t>(t[0]) |
                               (static_cast<uint32_t>(t[SlmPitch]) << 8) |
                               (static_cast<uint32_t>(t[2 * SlmPitch]) << 16) |
                               (static_cast<uint32_t>(t[3 * SlmPitch]) << 24);
            *reinterpret_cast<uint32_t*>(
                dst_base + static_cast<int64_t>(k0 + kk) * padded_rows +
                q * 4) = w;
          }
        }
        sycl::group_barrier(group);
      }
    }
  }

 private:
  const uint8_t* src;
  uint8_t* dst;
  const int* rows_per_expert;
  int* desc;
  const int scale_k;
  const int num_experts;
  sycl::local_accessor<uint8_t, 1> tile_slm;
  sycl::local_accessor<int, 1> prefix_slm;
};

template <bool VecLoad>
void launch_reorder_mxfp_scales(
    sycl::queue& queue,
    const uint8_t* src,
    uint8_t* dst,
    const int* rows_per_expert,
    int* desc,
    const int scale_k,
    const int num_experts,
    const int num_row_groups) {
  using Kernel = ReorderMxfpScalesKernel<VecLoad>;
  vllm::xpu::sycl_kernel_submit(
      queue,
      Kernel::get_nd_range(num_row_groups),
      Kernel{src, dst, rows_per_expert, desc, scale_k, num_experts});
}

}  // namespace moe
}  // namespace vllm

torch::Tensor reorder_mxfp_scales(
    const torch::Tensor& A_scales,
    const torch::Tensor& rows_per_expert,
    const int64_t total_padded_rows,
    const c10::optional<torch::Tensor>& expert_scale_desc) {
  TORCH_CHECK(A_scales.dim() == 2, "A_scales must be 2D [total_rows, scale_k]");
  TORCH_CHECK(
      A_scales.scalar_type() == at::kFloat8_e8m0fnu,
      "A_scales must be float8_e8m0fnu");
  TORCH_CHECK(A_scales.is_contiguous(), "A_scales must be contiguous");
  TORCH_CHECK(
      rows_per_expert.scalar_type() == at::kInt,
      "rows_per_expert must be int32");
  TORCH_CHECK(
      rows_per_expert.is_contiguous(), "rows_per_expert must be contiguous");

  const int num_experts = rows_per_expert.size(0);
  const int scale_k = A_scales.size(1);

  int* desc_ptr = nullptr;
  if (expert_scale_desc.has_value()) {
    const auto& desc = expert_scale_desc.value();
    TORCH_CHECK(
        desc.scalar_type() == at::kInt && desc.is_contiguous() &&
            desc.dim() == 2 && desc.size(0) == 2 &&
            desc.size(1) == num_experts + 1,
        "expert_scale_desc must be a contiguous int32 [2, num_experts + 1] "
        "tensor");
    TORCH_CHECK(
        desc.device() == rows_per_expert.device(),
        "expert_scale_desc must be on the same device as rows_per_expert");
    desc_ptr = desc.data_ptr<int32_t>();
  }

  auto A_scale_k =
      torch::zeros({total_padded_rows, scale_k}, A_scales.options());

  if (num_experts == 0) {
    if (desc_ptr != nullptr) expert_scale_desc->zero_();
    return A_scale_k;
  }

  // Always launch at least one work-group so the descriptor gets published
  // even when there is nothing to copy.
  constexpr int RowsPerGroup =
      vllm::moe::ReorderMxfpScalesKernel<true>::RowsPerGroup;
  const int num_row_groups = std::max<int64_t>(
      (total_padded_rows + RowsPerGroup - 1) / RowsPerGroup, 1);
  const int launch_scale_k = total_padded_rows == 0 ? 0 : scale_k;

  auto& queue = vllm::xpu::vllmGetQueue();
  const auto* src = reinterpret_cast<const uint8_t*>(A_scales.data_ptr());
  auto* dst = reinterpret_cast<uint8_t*>(A_scale_k.data_ptr());
  const auto* rows = reinterpret_cast<const int*>(rows_per_expert.data_ptr());

  // 16-byte vector loads need every source row 16-byte aligned.
  const bool vec_load =
      scale_k % 16 == 0 && reinterpret_cast<uintptr_t>(src) % 16 == 0;
  if (vec_load) {
    vllm::moe::launch_reorder_mxfp_scales<true>(
        queue,
        src,
        dst,
        rows,
        desc_ptr,
        launch_scale_k,
        num_experts,
        num_row_groups);
  } else {
    vllm::moe::launch_reorder_mxfp_scales<false>(
        queue,
        src,
        dst,
        rows,
        desc_ptr,
        launch_scale_k,
        num_experts,
        num_row_groups);
  }

  return A_scale_k;
}
