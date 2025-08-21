#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "helper.h"
#include "sycl_common.hpp"

#include "scaled_mm/collective/xe_scaled_mm_mma_fp8.hpp"
#include "scaled_mm/kernel/xe_scaled_mm_fp8.hpp"

namespace gpu::cutlass_kernel {
namespace scaled_mm {

using namespace cute;

struct Arguments {
  int m, n, k, l;
  float alpha, beta;

  Arguments() = default;
  Arguments(int m, int n, int k, int l, float alpha, float beta)
      : m(m), n(n), k(k), l(l), alpha(alpha), beta(beta) {}
};

template <class Gemm>
struct GemmRunner {
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using StrideScaleA = typename Gemm::GemmKernel::StrideA;
  using StrideScaleB = typename Gemm::GemmKernel::StrideB;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  using ElementAcc = typename Gemm::ElementAccumulator;
  using ElementScaleA = cutlass::half_t;
  using ElementScaleB = cutlass::half_t;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using ElementAccumulator = typename CollectiveEpilogue::ElementAccumulator;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  StrideScaleA stride_scaleA;
  StrideScaleB stride_scaleB;

  void initialize(const ProblemShapeType& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_A =
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B =
        cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C =
        cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D =
        cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));
    stride_scaleA = cutlass::make_cute_packed_stride(
        StrideScaleA{}, cute::make_shape(M, K, L));
    stride_scaleB = cutlass::make_cute_packed_stride(
        StrideScaleB{}, cute::make_shape(N, K, L));
  }

  cutlass::Status run(
      sycl::queue* stream,
      const Arguments& args,
      const cutlass::KernelHardwareInfo& hw_info,
      ElementA* inputA,
      ElementB* inputB,
      ElementScaleA* scaleA,
      ElementScaleB* scaleB,
      float* res) {
    ProblemShapeType problem_size =
        ProblemShapeType{args.m, args.n, args.k, args.l};

    initialize(problem_size);

    typename Gemm::GemmKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {inputA,
         stride_A,
         inputB,
         stride_B,
         scaleA,
         stride_scaleA,
         scaleB,
         stride_scaleB},
        {{args.alpha, args.beta}, nullptr, stride_C, res, stride_D},
        hw_info};

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (args.n < 16) {
      std::cout
          << "Invalid Problem Size: N must be >= 16 for FP8 input with F16 MMA (XE_8x16x16_F32F16F16F32_TT). Got N="
          << args.n << std::endl;
      std::exit(1);
    }

    if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess) {
      std::cout << "Invalid Problem Size: " << args.m << 'x' << args.n << 'x'
                << args.k << 'x' << args.l << std::endl;
      std::exit(1);
    }

    gemm_op.initialize(arguments, workspace.get());

    // Run the GEMM
    gemm_op.run(stream);

    stream->throw_asynchronous();

    return cutlass::Status::kSuccess;
  }
};

/*
Several points:
1. scaled_A/B must be half, need to check if could be bf16, convert before
passing
2. scaled_A/B must be same shape as A, B, need to broadcast before kernel
3. inputC is not needed, need to check if empty pointer accpetable
4. output need to be converted from fp32 to bf16
 */
void kernel_functor(
    sycl::queue* stream,
    void* inputA,
    void* inputB,
    void* scaleA,
    void* scaleB,
    void* res,
    const int m,
    const int n,
    const int k,
    float alpha,
    float beta) {
  Arguments args(m, n, k, 1, alpha, beta);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = cutlass::float_e4m3_t;
  using ElementInputB = cutlass::float_e4m3_t;
  using ElementOutput = float;
  using ElementScale = cutlass::half_t;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using TileShape = Shape<_256, _256, _32>;
  using GmemTiledCopyA =
      XE_2D_U8x32x32_LD_N; // Note: This shape has to match the shape used for
                           // the scaling factors
  using GmemTiledCopyB =
      XE_2D_U8x32x32_LD_V; // Note: This shape has to match the shape used for
                           // the scaling factors

  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_8x16x16_F32F16F16F32_TT>,
      Layout<TileShape>,
      Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy =
      cutlass::gemm::MainloopIntelScaledMMW8A8<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput,
      ElementComputeEpilogue,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy,
      EpilogueOp,
      TileShape,
      decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      ElementAccumulator,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      FusionCallBacks,
      XE_2D_U32x8x16_LD_N,
      void,
      void,
      XE_2D_U32x8x16_ST_N,
      void,
      void>;

  // Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      ElementInputA,
      cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementInputB,
      cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma,
      GmemTiledCopyA,
      void,
      void,
      cute::identity,
      GmemTiledCopyB,
      void,
      void,
      cute::identity>;

  using GemmKernel = cutlass::gemm::kernel::GemmScaledMM<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  GemmRunner<Gemm> runner;
  runner.run(
      stream,
      args,
      hw_info,
      reinterpret_cast<ElementInputA*>(inputA),
      reinterpret_cast<ElementInputB*>(inputB),
      reinterpret_cast<ElementScale*>(scaleA),
      reinterpret_cast<ElementScale*>(scaleB),
      reinterpret_cast<ElementOutput*>(res));
}

} // namespace scaled_mm
} // namespace gpu::cutlass_kernel
