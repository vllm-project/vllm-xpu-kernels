// SPDX-License-Identifier: Apache-2.0
//
// CuTe mainloops for the Xe3 MoE grouped GEMM.
//
// These are plain device functions over CuTe tensors -- there is no
// CollectiveMma, no CollectiveEpilogue and no GemmUniversal wrapper. A tile of
// work is fully described by the (already expert-offset) A/B/C/D tensors, the
// scale operands and an N tile index, exactly like the SYCL-TLA MoE grouped
// GEMM benchmark.
//
// All four scale flavors share one body:
//   * None      -- 16-bit DPAS
//   * MXBlock   -- e8m0 block scales zipped into the block-scaled DPAS
//   * Fp8Block  -- fp32 128x128 scales folded once per scale group
//   * Fp8Tensor -- one fp32 scalar per operand, folded after the K loop
#pragma once

#include <cute/tensor.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/fp8_to_fp16.h"
#include "cutlass/gemm/collective/xe_common_blockscaled_mxfp.hpp"

#include "../runner/moe_types.hpp"
#include "xe_moe_scale_copy.hpp"

namespace gpu::cutlass_kernel::grouped_gemm {

using namespace cute;

namespace detail {

// Subgroup geometry of a TiledMMA built by TiledMMAHelper.
template <class TiledMMA, class WGTile>
struct MoESubgroupShape {
  static constexpr int BLK_M = decltype(get<0>(WGTile{}))::value;
  static constexpr int BLK_N = decltype(get<1>(WGTile{}))::value;
  static constexpr int BLK_K = decltype(get<2>(WGTile{}))::value;
  static constexpr int SG_NUMS_M =
      get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr int SG_NUMS_N =
      get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr int SG_NUMS_K =
      get<3>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr int SG_M = ceil_div(BLK_M, SG_NUMS_M);
  static constexpr int SG_N = ceil_div(BLK_N, SG_NUMS_N);
  static constexpr int SG_K = ceil_div(BLK_K, SG_NUMS_K);
  static constexpr int MMA_K = get<2>(typename TiledMMA::Shape_MNK{});
};

}  // namespace detail

// -----------------------------------------------------------------------------
// Epilogue: optional fp32 bias, convert, store.
// -----------------------------------------------------------------------------
//
// D = convert(accum + C). The bias flavor walks the tile one epilogue tile at a
// time so only a single MMA-atom-sized C fragment is live next to the
// accumulator: materializing the whole bias tile plus its reordered copy costs
// three accumulator-sized fp32 fragments, which overflows the 512-register
// budget on the large (M320/M256) buckets and spills.
template <
    bool AddBias,
    class AccTensor,
    class CTensor,
    class DTensor,
    class GDTensor,
    class TiledMMA>
CUTE_DEVICE void moe_greedy_epilogue_bias(
    AccTensor& accum,
    CTensor const& C,
    DTensor& D,
    GDTensor const& gD,
    TiledMMA const& mma,
    int local_id) {
  using ElementC = cute::remove_const_t<typename CTensor::element_type>;
  using ElementD = cute::remove_const_t<typename DTensor::element_type>;
  using MMATile = decltype(take<0, 2>(typename TiledMMA::AtomShape_MNK{}));

  static constexpr int CopyBitsC = sizeof_bits_v<ElementC>;
  static constexpr int CopyBitsD = sizeof_bits_v<ElementD>;
  // One cache line wide, clipped to the MMA atom so epilogue tiles align with
  // whole accumulator atoms.
  static constexpr int EpiR = cute::gcd(8, get<0>(MMATile{}));
  static constexpr int EpiC =
      cute::gcd(512 / cute::min(CopyBitsC, CopyBitsD), get<1>(MMATile{}));
  using EpilogueTile = Shape<Int<EpiR>, Int<EpiC>>;

  using CopyOpC = XE_LOAD_2D<
      CopyBitsC,
      cute::gcd(8, EpiR),
      cute::gcd(512 / CopyBitsC, EpiC)>;
  using CopyOpD = XE_STORE_2D<
      CopyBitsD,
      cute::gcd(8, EpiR),
      cute::gcd(512 / CopyBitsD, EpiC)>;

  auto thr_mma = mma.get_slice(local_id);
  auto tCDgCD = thr_mma.partition_C(gD);  // (mma_v,mma_m,mma_n) -> coord

  // Retile the accumulator and the subgroup's coordinates into epilogue tiles.
  auto mma_per_epi = shape_div(EpilogueTile{}, MMATile{});
  auto tiled_acc_layout = group<0, 3>(prepend(
      flat_divide(remove<0>(accum.layout()), mma_per_epi),
      get<0>(accum.layout())));
  auto tiled_acc =
      make_tensor(accum.data(), tiled_acc_layout);  // ((v,m,n),epi_m,epi_n)
  auto sg_v_coord = prepend(
      flat_divide(remove<0>(tCDgCD.layout()), mma_per_epi),
      get<0>(tCDgCD.layout()));

  auto copy_c = make_block_2d_copy(CopyOpC{}, C);
  auto copy_d = make_block_2d_copy(CopyOpD{}, D);
  const int wi_idx = local_id % intel::sg_size;
  auto thr_copy_c = copy_c.get_slice(wi_idx);
  auto thr_copy_d = copy_d.get_slice(wi_idx);

  auto gCD_epi_layout = append(
      append(make_identity_layout(EpilogueTile{}), get<3>(sg_v_coord)),
      get<4>(sg_v_coord));
  auto gCD_epi =
      make_tensor(tCDgCD.data(), gCD_epi_layout);  // (m,n,epi_m,epi_n)

  auto tCgC = thr_copy_c.partition_S(gCD_epi);
  auto tDgD = thr_copy_d.partition_D(gCD_epi);
  auto tCrC = thr_copy_c.partition_sg_fragment_D(gCD_epi(_, _, 0, 0));
  auto tDrD = thr_copy_d.partition_sg_fragment_S(gCD_epi(_, _, 0, 0));

  using AccTVLayout = decltype(thr_mma.partition_sg_fragment_C(gD).tv_layout());
  auto cd_compute_tv =
      make_layout(get<0>(AccTVLayout{}), sg_v_coord(_, _, _, _0{}, _0{}));

  auto tCrC_compute_wi = make_fragment_like<ElementC>(tiled_acc(_, _0{}, _0{}));
  auto tCrC_compute = make_subgroup_tensor(tCrC_compute_wi, cd_compute_tv);

  constexpr auto EpiTilesM = size<2>(gCD_epi);
  constexpr auto EpiTilesN = size<3>(gCD_epi);

  CUTE_UNROLL
  for (int epi_m = 0; epi_m < EpiTilesM; ++epi_m) {
    CUTE_UNROLL
    for (int epi_n = 0; epi_n < EpiTilesN; ++epi_n) {
      auto acc_epi_wi = make_tensor(
          tiled_acc(_, epi_m, epi_n).data(), tiled_acc(_, _0{}, _0{}).layout());
      if constexpr (AddBias) {
        copy(copy_c, tCgC(_, _, _, epi_m, epi_n), tCrC);
        reorder(tCrC, tCrC_compute);
        CUTE_UNROLL
        for (int i = 0; i < cute::size(acc_epi_wi.layout()); ++i) {
          acc_epi_wi(i) += tCrC_compute_wi(i);
        }
      }

      auto acc_epi = make_subgroup_tensor(acc_epi_wi, cd_compute_tv);
      reorder(acc_epi, tDrD);
      copy(copy_d, tDrD, tDgD(_, _, _, epi_m, epi_n));
    }
  }
}

template <
    class StoreCache,
    bool MaybeBias,
    bool Chunked,
    class AccTensor,
    class CTensor,
    class DTensor,
    class GDTensor,
    class TiledMMA>
CUTE_DEVICE void moe_greedy_epilogue(
    AccTensor& accum,
    CTensor const& C,
    DTensor& D,
    GDTensor const& gD,
    TiledMMA const& mma,
    int local_id,
    bool has_bias) {
  if constexpr (MaybeBias) {
    if (has_bias) {
      moe_greedy_epilogue_bias<true>(accum, C, D, gD, mma, local_id);
      return;
    }
  }
  if constexpr (Chunked) {
    moe_greedy_epilogue_bias<false>(accum, C, D, gD, mma, local_id);
    return;
  }

  auto tiled_copy_d = get_moe_block_2d_copy_D<StoreCache>(mma, D);
  auto thr_copy_d = tiled_copy_d.get_slice(local_id);
  auto tCrD_final = thr_copy_d.partition_sg_fragment_S(gD);
  auto tCgD = thr_copy_d.partition_D(gD);
  reorder(accum, tCrD_final);
  copy(tiled_copy_d, tCrD_final, tCgD);
}

// -----------------------------------------------------------------------------
// Greedy tile GEMM.
// -----------------------------------------------------------------------------
//
// A/B/C/D pointers are already advanced to the tile's expert and row origin, so
// the M-row origin inside this function is always 0 and only the N tile index
// varies. `Direct` selects the benchmark-style scale pipeline (aligned 2D scale
// loads, workgroup barriers and the config's store cache policy); otherwise the
// alignment-safe scale copies and the deeper, barrier-free block-scaled
// pipeline are used.
//
//   SA/SB are: MX -> (M, scale_k) / (N, scale_k) tensors
//              FP8 -> raw fp32 pointers
//              plain -> unused
template <
    class Cfg,
    bool Direct,
    bool MaybeBias,
    bool Barrier,
    class ScaleCopyA,
    class ScaleCopyB,
    class ATensor,
    class BTensor,
    class SATensor,
    class SBTensor,
    class CTensor,
    class DTensor,
    class TiledMMA>
CUTE_DEVICE void moe_greedy_gemm(
    ATensor const& A,
    BTensor const& B,
    SATensor const& SA,
    SBTensor const& SB,
    CTensor const& C,
    DTensor& D,
    int wg_n,
    TiledMMA const& mma,
    bool has_bias) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int local_id = int(item.get_local_linear_id());

  auto wg_tile = mma.tile_mnk();

  Tensor cA = make_identity_tensor(A.shape());  // (M,K)
  Tensor cB = make_identity_tensor(B.shape());  // (N,K)
  Tensor cD = make_identity_tensor(D.shape());  // (M,N)

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(_0{}, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
  Tensor gD =
      local_tile(cD, wg_tile, make_coord(_0{}, wg_n, 0), Step<_1, _1, X>{});

  auto thr_mma = mma.get_slice(local_id);

  using StoreCache =
      cute::conditional_t<Direct, typename Cfg::StoreCache, void>;

  auto tiled_copy_a = get_block_2d_copy_A<void>(mma, A);
  auto tiled_copy_b = get_block_2d_copy_B<void>(mma, B);

  auto thr_copy_a = tiled_copy_a.get_slice(local_id);
  auto thr_copy_b = tiled_copy_b.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
  auto tCrD = thr_mma.partition_sg_fragment_C(gD);

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  auto prefetch_a = make_block_2d_prefetch(tiled_copy_a);
  auto prefetch_b = make_block_2d_prefetch(tiled_copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  clear(tCrD);

  constexpr SPIRVScope barrier_scope = ScopeWorkgroup;
  // The benchmark scale pipeline keeps two K-tiles in flight regardless of
  // dtype; deeper prefetch only adds cache pressure on its large uniform
  // tiles. The gather-capable pipeline stays at the per-dtype stage count.
  constexpr int prefetch_dist = Direct ? 2 : Cfg::Stages;
  const int k_tile_count = ceil_div(int(shape<1>(A)), int(get<2>(wg_tile)));
  int prefetch_k = 0;

  using Geometry = detail::MoESubgroupShape<TiledMMA, decltype(wg_tile)>;

  if constexpr (Cfg::ScaleKind == MoEScaleKind::MXBlock) {
    namespace coll = cutlass::gemm::collective;
    using ElementScaleA = typename Cfg::ElementScaleA;
    using ElementScaleB = typename Cfg::ElementScaleB;
    constexpr int GroupK = Cfg::ScaleBlock;
    constexpr int MMA_K = Geometry::MMA_K;
    constexpr int SG_M = Geometry::SG_M;
    constexpr int SG_N = Geometry::SG_N;
    constexpr int SG_K = Geometry::SG_K;
    constexpr int k_reload_factor = cute::max(GroupK / Geometry::BLK_K, 1);

    using GemmIterM = Int<decltype(size<1>(tCrA.shape()))::value>;
    using GemmIterN = Int<decltype(size<1>(tCrB.shape()))::value>;
    using GemmIterK = Int<decltype(size<2>(tCrB.shape()))::value>;

    // A-scale row/B-scale column of this subgroup. The row origin is 0 because
    // the caller already advanced the scale pointer to the tile's first row.
    const int sg_id = int(cutlass::get_sub_group_id());
    const int m_coord = (sg_id / Geometry::SG_NUMS_N) * SG_M;
    const int n_coord =
        wg_n * Geometry::BLK_N + (sg_id % Geometry::SG_NUMS_N) * SG_N;
    const int l_coord = 0;

    auto [tiled_copy_scaleA, copy_iter_scaleA, fragment_scaleA] =
        coll::make_scaled_copy<ScaleCopyA, ElementScaleA, SG_M, SG_K, GroupK>(
            SA, m_coord, l_coord, k_tile_count);
    auto [tiled_copy_scaleB, copy_iter_scaleB, fragment_scaleB] =
        coll::make_scaled_copy<ScaleCopyB, ElementScaleB, SG_N, SG_K, GroupK>(
            SB, n_coord, l_coord, k_tile_count);
    auto
        [scale_m_offsets, scale_n_offsets, scale_ak_offsets, scale_bk_offsets] =
            coll::make_scaled_offsets<
                GemmIterM::value,
                GemmIterN::value,
                GemmIterK::value,
                MMA_K,
                GroupK,
                typename decltype(tiled_copy_scaleA)::BlockShape,
                typename decltype(tiled_copy_scaleB)::BlockShape>();
    auto [tiled_prefetch_scaleA, prefetch_iter_scaleA] = coll::
        make_scaled_prefetch<decltype(tiled_copy_scaleA), SG_M, SG_K, GroupK>(
            tiled_copy_scaleA, m_coord, l_coord, k_tile_count);
    auto [tiled_prefetch_scaleB, prefetch_iter_scaleB] = coll::
        make_scaled_prefetch<decltype(tiled_copy_scaleB), SG_N, SG_K, GroupK>(
            tiled_copy_scaleB, n_coord, l_coord, k_tile_count);

    using scaleA_vec_t = cute::intel::
        vector_t<ElementScaleA, decltype(size(fragment_scaleA))::value>;
    using scaleB_vec_t = cute::intel::
        vector_t<ElementScaleB, decltype(size(fragment_scaleB))::value>;

    CUTE_UNROLL
    for (; prefetch_k < prefetch_dist; prefetch_k++) {
      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
      prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
      prefetch(
          tiled_prefetch_scaleA,
          prefetch_iter_scaleA(_, _, _, prefetch_k / k_reload_factor));
      prefetch(
          tiled_prefetch_scaleB,
          prefetch_iter_scaleB(_, _, _, prefetch_k / k_reload_factor));
    }

    for (int k_tile = 0; k_tile < k_tile_count; k_tile++, prefetch_k++) {
      if constexpr (Barrier) barrier_arrive(barrier_scope);

      copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA);
      copy(tiled_copy_b, tBgB(_, _, _, k_tile), tBrB);
      copy(
          tiled_copy_scaleA,
          copy_iter_scaleA(_, _, _, k_tile / k_reload_factor),
          fragment_scaleA);
      copy(
          tiled_copy_scaleB,
          copy_iter_scaleB(_, _, _, k_tile / k_reload_factor),
          fragment_scaleB);

      if (prefetch_k < k_tile_count) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
        prefetch(
            tiled_prefetch_scaleA,
            prefetch_iter_scaleA(_, _, _, prefetch_k / k_reload_factor));
        prefetch(
            tiled_prefetch_scaleB,
            prefetch_iter_scaleB(_, _, _, prefetch_k / k_reload_factor));
      }

      reorder(tArA, tCrA);
      reorder(tBrB, tCrB);

      Tensor scaleA = make_tensor(
          recast<scaleA_vec_t>(fragment_scaleA).data(),
          make_layout(Shape<_1, GemmIterM, _1>{}, Stride<_1, _0, _0>{}));
      Tensor scaleB = make_tensor(
          recast<scaleB_vec_t>(fragment_scaleB).data(),
          make_layout(Shape<_1, GemmIterN, _1>{}, Stride<_1, _0, _0>{}));

      cute::gemm(
          mma,
          make_zip_tensor(tCrA, scaleA, scale_m_offsets, scale_ak_offsets),
          make_zip_tensor(tCrB, scaleB, scale_n_offsets, scale_bk_offsets),
          tCrD);

      if constexpr (Barrier) barrier_wait(barrier_scope);
    }
  } else if constexpr (Cfg::ScaleKind == MoEScaleKind::Fp8Block) {
    // Fold the fp32 block scales once per GROUP_K instead of once per K tile:
    // both scales are constant across k_tiles_per_group consecutive tiles, so
    // those tiles accumulate into `scaler` and are promoted into `tCrD` only
    // when the next tile crosses a scale-group boundary.
    constexpr int GROUP_K = Cfg::ScaleBlock;
    constexpr int SG_K = Geometry::SG_K;
    static_assert(
        GROUP_K % SG_K == 0,
        "block-FP8 scale folding needs GROUP_K to be a whole number of K "
        "tiles");
    constexpr int k_tiles_per_group = cute::max(GROUP_K / SG_K, 1);
    constexpr int SubgroupSize = 16;
    constexpr int acc_m0 = decltype(cute::size<0>(tCrD.layout()))::value;
    constexpr int acc_m1 = decltype(cute::size<1>(tCrD.layout()))::value;
    constexpr int sg_m_rows = acc_m0 * acc_m1;

    const int sg_id = int(cutlass::get_sub_group_id());
    const int m_coord = (sg_id / Geometry::SG_NUMS_N) * Geometry::SG_M;
    const int n_coord =
        wg_n * Geometry::BLK_N + (sg_id % Geometry::SG_NUMS_N) * Geometry::SG_N;

    const int scale_M = int(shape<0>(A));
    const int scale_N = int(shape<0>(B));
    const int scale_K_groups = ceil_div(int(shape<1>(A)), GROUP_K);
    // Clamp the scale coordinates once instead of predicating every fetch: a
    // tile that runs past M or N has its accumulator dropped by the store, and
    // its A rows load as zero, so a clamped finite scale is as good as zero --
    // without the branch.
    const int scale_rows =
        cute::min(sg_m_rows, cute::max(1, scale_M - m_coord));
    const int scale_m_base =
        cute::max(0, cute::min(m_coord, scale_M - 1)) * scale_K_groups;
    const int scale_n_base =
        cute::max(
            0, cute::min(n_coord / GROUP_K, ceil_div(scale_N, GROUP_K) - 1)) *
        scale_K_groups;

    // lane l of packed_scale[v] holds the combined scale of accumulator row
    // v * SubgroupSize + l. ptr_SA is row-major [M][K/GROUP_K], so giving each
    // lane its own row turns 16 scalar sends into one gather and shrinks the
    // live scale state from sg_m_rows GRF to ceil(sg_m_rows / 16).
    constexpr int kScaleVecs = ceil_div(sg_m_rows, SubgroupSize);
    float packed_scale[kScaleVecs];
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    const int scale_lane = int(cutlass::get_sub_group_local_id());
    auto fetch_scales = [&](int k_group) {
      const float scaleB = float(SB[scale_n_base + k_group]);
      const int base = scale_m_base + k_group;
      CUTE_UNROLL
      for (int v = 0; v < kScaleVecs; ++v) {
        const int row = v * SubgroupSize + scale_lane;
        // Rows past the tile's valid extent re-read the last valid row, which
        // is what the scalar walk did by not advancing its offset.
        const int r = row < scale_rows ? row : scale_rows - 1;
        packed_scale[v] = float(SA[base + r * scale_K_groups]) * scaleB;
      }
    };

    int k_group_idx = 0;
    int k_group_fill = 0;
    Tensor scaler = make_tensor_like(tCrD);
    clear(scaler);
    fetch_scales(k_group_idx);

    CUTE_UNROLL
    for (; prefetch_k < prefetch_dist; prefetch_k++) {
      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
      prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
    }

    for (int k_tile = 0; k_tile < k_tile_count; k_tile++, prefetch_k++) {
      barrier_arrive(barrier_scope);

      copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA);
      copy(tiled_copy_b, tBgB(_, _, _, k_tile), tBrB);

      if (prefetch_k < k_tile_count) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
      }

      reorder(tArA, tCrA);
      reorder(tBrB, tCrB);

      cute::gemm(mma, tCrA, tCrB, scaler);

      // One increment and one compare per K tile; no division, no scale
      // addressing, nothing else on the DPAS critical path.
      if (++k_group_fill == k_tiles_per_group || k_tile + 1 == k_tile_count) {
        CUTE_UNROLL
        for (int v = 0; v < kScaleVecs; ++v) {
          CUTE_UNROLL
          for (int lane = 0; lane < SubgroupSize; ++lane) {
            const int row = v * SubgroupSize + lane;
            if (row >= sg_m_rows) {
              continue;
            }
            // Broadcast this row's scale out of the packed vector: a
            // register-region read, not a memory message.
            const float row_scale =
                sycl::select_from_group(sg, packed_scale[v], lane);
            const int i0 = row % acc_m0;
            const int i1 = row / acc_m0;
            CUTE_UNROLL
            for (int i2 = 0; i2 < cute::size<2>(tCrD.layout()); ++i2) {
              tCrD[make_coord(i0, i1, i2)] +=
                  scaler[make_coord(i0, i1, i2)] * row_scale;
            }
          }
        }

        clear(scaler);
        k_group_fill = 0;
        // Issue the next group's scale loads now: they are not read until the
        // next promotion, k_tiles_per_group DPAS tiles away, which keeps their
        // latency off the critical path.
        if (++k_group_idx < scale_K_groups) {
          fetch_scales(k_group_idx);
        }
      }

      barrier_wait(barrier_scope);
    }
  } else {
    // Plain 16-bit GEMM, and FP8 per-tensor which only differs by a single
    // scalar multiply after the K loop.
    CUTE_UNROLL
    for (; prefetch_k < prefetch_dist; prefetch_k++) {
      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
      prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
    }

    for (int k_tile = 0; k_tile < k_tile_count; k_tile++, prefetch_k++) {
      barrier_arrive(barrier_scope);

      copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA);
      copy(tiled_copy_b, tBgB(_, _, _, k_tile), tBrB);

      if (prefetch_k < k_tile_count) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
      }

      reorder(tArA, tCrA);
      reorder(tBrB, tCrB);

      cute::gemm(mma, tCrA, tCrB, tCrD);

      barrier_wait(barrier_scope);
    }

    if constexpr (Cfg::ScaleKind == MoEScaleKind::Fp8Tensor) {
      const float scaleAB = float(SA[0]) * float(SB[0]);
      CUTE_UNROLL
      for (int i = 0; i < cute::size(tCrD.layout()); ++i) {
        tCrD(i) *= scaleAB;
      }
    }
  }

  // Walking the tile in epilogue-sized chunks is what the collective epilogue
  // did, and it stays ahead on the plain and fp8-block stores. The MX kernels
  // are the exception: their skewed tiles measure faster with the full-tile
  // store, so they only chunk when a bias actually has to be added.
  constexpr bool ChunkedStore =
      !Direct && Cfg::ScaleKind != MoEScaleKind::MXBlock;
  moe_greedy_epilogue<StoreCache, MaybeBias, ChunkedStore>(
      tCrD, C, D, gD, mma, local_id, has_bias);
}

}  // namespace gpu::cutlass_kernel::grouped_gemm
