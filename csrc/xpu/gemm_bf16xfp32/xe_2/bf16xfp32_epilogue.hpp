/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * BF16xFP32 GEMM Epilogue: combines two accumulator results into one output.
 *   D = acc_high + acc_low * scale
 *
 * Based on xe_dual_gemm_epilogue_elementwise_activation.hpp
 **************************************************************************************************/
#pragma once

#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/detail/layout.hpp"

#include "cute/tensor.hpp"

namespace cutlass {
namespace epilogue {
namespace collective {

template <
    class DispatchPolicy,
    class CtaTileMNK_,
    class ElementC_,
    class StrideC_,
    class ElementD_,
    class StrideD_,
    class CopyOpG2R_,
    class CopyOpR2G_>
class BF16xFP32Epilogue;

template <
    class CtaTileMNK_,
    class ElementC_,
    class StrideC_,
    class ElementD_,
    class StrideD_,
    class CopyOpG2R_,
    class CopyOpR2G_>
class BF16xFP32Epilogue<
    IntelXeXMX16,
    CtaTileMNK_,
    ElementC_,
    StrideC_,
    ElementD_,
    StrideD_,
    CopyOpG2R_,
    CopyOpR2G_> {
 public:
  using DispatchPolicy = IntelXeXMX16;
  using CtaTileMNK = CtaTileMNK_;
  using ElementC = ElementC_;
  using ElementAccumulator = ElementD_;
  using StrideC = StrideC_;
  using ElementD = ElementD_;
  using StrideD = StrideD_;
  using CopyOpG2R = CopyOpG2R_;
  using CopyOpR2G = CopyOpR2G_;

  using GmemTiledCopyC = CopyOpG2R;
  using GmemTiledCopyD = cute::conditional_t<
      not cute::is_void_v<ElementD> && not cute::is_void_v<CopyOpR2G>,
      CopyOpR2G,
      XE_2D_U32x8x16_ST_N>;
  using ElementOutput = ElementAccumulator;
  using ElementCompute = ElementAccumulator;

  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

  static_assert(
      cute::rank(CtaTileMNK{}) == 3,
      "CtaTileMNK must be rank-3: [CTA_M, CTA_N, CTA_K]");
  static_assert(
      cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]");

  static_assert(std::is_same_v<ElementC, void>, "No Source memory needed");
  static_assert(
      std::is_same_v<CopyOpG2R, void>, "No Source Copy operation required");

  using CopyThreadShape = cute::Shape<cute::_1, cute::Int<SubgroupSize>>;
  using Trait_D = cute::Copy_Traits<GmemTiledCopyD, StrideD>;
  using XE_Copy_D = decltype(cute::make_tiled_copy(
      cute::Copy_Atom<Trait_D, ElementD>{},
      cute::Layout<CopyThreadShape>{},
      cute::make_layout(
          cute::shape_div(typename Trait_D::BlockShape{}, CopyThreadShape{}))));

 private:
  constexpr static bool is_destination_supported =
      not cute::is_void_v<ElementD> && not cute::is_void_v<CopyOpR2G>;

 public:
  // Host side epilogue arguments
  struct Arguments {
    ElementD* ptr_D;
    StrideD dD;
    float scale;  // The scale factor for BF16xFP32 reconstruction
  };

  // Device side epilogue params
  struct Params {
    XE_Copy_D xe_store_d;
    float scale;
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    auto problem_shape_MNKL = cute::append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    XE_Copy_D xe_store_d = {};
    if constexpr (is_destination_supported) {
      auto mD = cute::make_tensor(
          cute::make_gmem_ptr(static_cast<ElementD*>(args.ptr_D)),
          cute::make_layout(cute::make_shape(M, N, L), args.dD));
      xe_store_d = cute::make_tiled_copy(
          cute::Copy_Atom<Trait_D, ElementD>{}.with(mD),
          cute::Layout<CopyThreadShape>{},
          cute::make_layout(
              cute::shape_div(
                  typename Trait_D::BlockShape{}, CopyThreadShape{})));
    }

    return {xe_store_d, args.scale};
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace,
      cudaStream_t stream,
      CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  BF16xFP32Epilogue(Params const& params_) : params(params_) {}

  CUTLASS_DEVICE
  bool is_producer_load_needed() const { return false; }

  template <
      class ProblemShapeMNKL,
      class TileShapeMNK,
      class TileCoordMNKL,
      class Accumulator0,
      class Accumulator1,
      class TiledMma,
      class ResidueMNK>
  CUTLASS_DEVICE void operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK tile_shape_MNK,
      TileCoordMNKL tile_coord_mnkl,
      Accumulator0 accumulators0,  // acc_high = X * W_high
      Accumulator1 accumulators1,  // acc_low  = X * W_low
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      char* smem) {
    (void)tiled_mma;
    (void)residue_mnk;
    (void)smem;
    using namespace cute;

    using MmaAtomShape = typename TiledMma::AtomShape_MNK;
    static constexpr auto BLK_M = get<0>(CtaTileMNK{});
    static constexpr auto BLK_N = get<1>(CtaTileMNK{});
    static constexpr auto BLK_K = get<2>(CtaTileMNK{});

    static constexpr auto ATOM_M =
        get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_N =
        get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_K =
        get<3>(typename TiledMma::ThrLayoutVMNK{}.shape());

    static constexpr auto SG_M = BLK_M / ATOM_M;
    static constexpr auto SG_N = BLK_N / ATOM_N;
    static constexpr auto SG_K = BLK_K / ATOM_K;
    using SubgroupTileShape =
        Shape<decltype(SG_M), decltype(SG_N), decltype(SG_K)>;

    static constexpr int FragsM =
        get<0>(SubgroupTileShape{}) / get<0>(MmaAtomShape());
    static constexpr int FragsN =
        get<1>(SubgroupTileShape{}) / get<1>(MmaAtomShape());
    static constexpr int FragmentSize =
        (get<0>(MmaAtomShape()) * get<1>(MmaAtomShape())) / SubgroupSize;

    auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;
    auto m_sg = get_sub_group_id() / ATOM_N;
    auto n_sg = get_sub_group_id() % ATOM_N;

    Tensor mD_mnl = cute::get_xe_tensor(make_shape(M, N, L));
    Tensor g_wg_D = local_tile(
        mD_mnl,
        take<0, 2>(CtaTileMNK{}),
        make_coord(m_coord, n_coord, l_coord));
    Tensor gD = local_tile(
        g_wg_D, take<0, 2>(SubgroupTileShape{}), make_coord(m_sg, n_sg));

    auto thread_xe_store_d = params.xe_store_d.get_thread_slice(thread_idx);
    Tensor tCgD = thread_xe_store_d.partition_D(gD);

    float scale = params.scale;

    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < FragsN; epi_n++) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_m = 0; epi_m < FragsM; epi_m++) {
        Tensor trD =
            make_tensor<ElementAccumulator>(Shape<Int<FragmentSize>>{});
        CUTLASS_PRAGMA_UNROLL
        for (int epi_v = 0; epi_v < FragmentSize; epi_v++) {
          // D = acc_high + acc_low * scale
          trD(epi_v) = accumulators0(epi_v, epi_m, epi_n) +
                       accumulators1(epi_v, epi_m, epi_n) * scale;
        }

        if constexpr (is_destination_supported) {
          copy(params.xe_store_d, trD, tCgD(_, epi_m, epi_n));
        }
      }
    }
  }

 private:
  Params const& params;
};

}  // namespace collective
}  // namespace epilogue
}  // namespace cutlass
