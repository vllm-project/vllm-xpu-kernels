#pragma once

#include <torch/all.h>
#include "csrc/utils.h"

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/packed_stride.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

using namespace cute;

using ElementInputA = cutlass::bfloat16_t;
using ElementInputB = cutlass::bfloat16_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementComputeEpilogue = float;

using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using GmemTiledCopyA = void;
using GmemTiledCopyB = void;

constexpr int PipelineStages = 2;
using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<PipelineStages>;
using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;
using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementOutput,
    ElementComputeEpilogue,
    ElementAccumulator,
    ElementAccumulator,
    cutlass::FloatRoundStyle::round_to_nearest>;

template <class TileShape_, class SGLayout_, class LayoutA_, class LayoutB_>
struct GemmConfig {
  using TileShape = TileShape_;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, cutlass::bfloat16_t>>,
      Layout<TileShape>,
      SGLayout_>::TiledMMA;

  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy,
      EpilogueOp,
      TileShape,
      decltype(tile_shape(TiledMma()))>;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      void,
      ElementAccumulator,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      FusionCallbacks,
      void,
      void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      ElementInputA,
      cutlass::gemm::TagToStrideA_t<LayoutA_>,
      ElementInputB,
      cutlass::gemm::TagToStrideB_t<LayoutB_>,
      TiledMma,
      GmemTiledCopyA,
      void,
      void,
      cute::identity,
      GmemTiledCopyB,
      void,
      void,
      cute::identity>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <class TS, class SG>
using ConfigRR =
    GemmConfig<TS, SG, cutlass::layout::RowMajor, cutlass::layout::RowMajor>;

template <class TS, class SG>
using ConfigRC =
    GemmConfig<TS, SG, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;

using SG_1x4 = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
using SG_2x4 = Layout<Shape<_2, _4, _1>, Stride<_4, _1, _0>>;
using SG_4x4 = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
using SG_8x4 = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

// tile_n=256 configs
using Config_8_RR = ConfigRR<Shape<_8, _256, _32>, SG_1x4>;
using Config_16_RR = ConfigRR<Shape<_16, _256, _32>, SG_2x4>;
using Config_32_RR = ConfigRR<Shape<_32, _256, _32>, SG_4x4>;
using Config_64_RR = ConfigRR<Shape<_64, _256, _32>, SG_8x4>;
using Config_128_RR = ConfigRR<Shape<_128, _256, _32>, SG_8x4>;
using Config_256_RR = ConfigRR<Shape<_256, _256, _32>, SG_8x4>;

// tile_n=128 configs (more N-tiles for better occupancy at mid-range M)
using SG_8x2 = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;
using Config_128n_RR = ConfigRR<Shape<_128, _128, _32>, SG_8x2>;
using Config_256n_RR = ConfigRR<Shape<_256, _128, _32>, SG_8x4>;
using Config_384_RR = ConfigRR<Shape<_384, _128, _32>, SG_8x4>;

using Config_8_RC = ConfigRC<Shape<_8, _256, _32>, SG_1x4>;
using Config_16_RC = ConfigRC<Shape<_16, _256, _32>, SG_2x4>;
using Config_32_RC = ConfigRC<Shape<_32, _256, _32>, SG_4x4>;
using Config_64_RC = ConfigRC<Shape<_64, _256, _32>, SG_8x4>;
using Config_128_RC = ConfigRC<Shape<_128, _256, _32>, SG_8x4>;
using Config_256_RC = ConfigRC<Shape<_256, _256, _32>, SG_8x4>;

using Config_128n_RC = ConfigRC<Shape<_128, _128, _32>, SG_8x2>;
using Config_256n_RC = ConfigRC<Shape<_256, _128, _32>, SG_8x4>;
using Config_384_RC = ConfigRC<Shape<_384, _128, _32>, SG_8x4>;

template <typename Config>
torch::Tensor run_gemm(
    sycl::queue& queue,
    const torch::Tensor& A,
    const torch::Tensor& B,
    int M,
    int N,
    int K);

// Extern template declarations to prevent implicit instantiation
#define DECLARE_EXTERN(CONFIG)                    \
  extern template torch::Tensor run_gemm<CONFIG>( \
      sycl::queue & queue,                        \
      const torch::Tensor& A,                     \
      const torch::Tensor& B,                     \
      int M,                                      \
      int N,                                      \
      int K);

DECLARE_EXTERN(Config_8_RR)
DECLARE_EXTERN(Config_16_RR)
DECLARE_EXTERN(Config_32_RR)
DECLARE_EXTERN(Config_64_RR)
DECLARE_EXTERN(Config_128_RR)
DECLARE_EXTERN(Config_256_RR)
DECLARE_EXTERN(Config_128n_RR)
DECLARE_EXTERN(Config_256n_RR)
DECLARE_EXTERN(Config_384_RR)
DECLARE_EXTERN(Config_8_RC)
DECLARE_EXTERN(Config_16_RC)
DECLARE_EXTERN(Config_32_RC)
DECLARE_EXTERN(Config_64_RC)
DECLARE_EXTERN(Config_128_RC)
DECLARE_EXTERN(Config_256_RC)
DECLARE_EXTERN(Config_128n_RC)
DECLARE_EXTERN(Config_256n_RC)
DECLARE_EXTERN(Config_384_RC)

#undef DECLARE_EXTERN
