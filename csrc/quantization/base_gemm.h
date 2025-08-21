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

namespace gpu::cutlass_kernel {
namespace basic_gemm {

using namespace cute;
using bf16 = sycl::ext::oneapi::bfloat16;
using fp16 = sycl::half;

struct Args {
  int m, n, k, l;
  float alpha, beta;

  Args() = default;
  Args(int m, int n, int k, int l, float alpha, float beta)
      : m(m), n(n), k(k), l(l), alpha(alpha), beta(beta) {}
};

template <class Gemm, typename T>
struct GemmRunner {
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  using ElementAcc = typename Gemm::ElementAccumulator;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using ElementAccumulator = typename CollectiveEpilogue::ElementAccumulator;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementOutput> block_D;

  void initialize(const ProblemShapeType& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    std::cout << M << " " << N << " " << K << " " << L << " " << std::endl;

    // Complete the stride by combining static layout info (StrideA) with
    // runtime size info (M,K,L)
    stride_A =
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B =
        cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C =
        cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D =
        cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    // block_A.reset(static_cast<std::size_t>(M) * K * L);
    // block_B.reset(static_cast<std::size_t>(K) * N * L);
    // block_C.reset(static_cast<std::size_t>(M) * N * L);
    // block_D.reset(static_cast<std::size_t>(M) * N * L);
    // initialize_block(block_A, seed + 2023);
    // initialize_block(block_B, seed + 2022);
    // initialize_block(block_C, seed + 2021);
  }

  cutlass::Status run(
      sycl::queue* stream,
      const Args& args,
      const cutlass::KernelHardwareInfo& hw_info,
      T* inputA,
      T* inputB,
      float* inputC,
      float* res) {
    std::cout << "into Runner" << std::endl;
    std::cout << args.alpha << " " << args.beta << std::endl;

    ProblemShapeType problem_size =
        ProblemShapeType{args.m, args.n, args.k, args.l};

    initialize(problem_size);

    typename Gemm::GemmKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {inputA, stride_A, inputB, stride_B},
        {{args.alpha, args.beta}, inputC, stride_C, res, stride_D},
        hw_info};

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess) {
      std::cout << "Invalid Problem Size: " << args.m << 'x' << args.n << 'x'
                << args.k << 'x' << args.l << std::endl;
      std::exit(1);
    }

    std::cout << "into initialize" << std::endl;

    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get(), stream));

    std::cout << "into gemm run" << std::endl;

    // Run the GEMM
    CUTLASS_CHECK(gemm_op.run());

    syclcompat::wait();

    std::cout << "finish gemm run" << std::endl;

    return cutlass::Status::kSuccess;
  }
};

template <typename T>
void gemm_functor(
    sycl::queue* stream,
    void* inputA,
    void* inputB,
    void* inputC,
    void* res,
    const int m,
    const int n,
    const int k,
    float alpha,
    float beta) {
  Args args(m, n, k, 1, alpha, beta);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = T;
  using ElementInputB = T;
  using ElementOutput = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
  using GmemTiledCopyB = XE_2D_U16x32x32_LD_N;

  using TileShape = Shape<_256, _256, _32>;

  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
      cute::Layout<TileShape>,
      cute::Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy =
      cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
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

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  GemmRunner<Gemm, T> runner;
  CUTLASS_CHECK(runner.run(
      stream,
      args,
      hw_info,
      reinterpret_cast<T*>(inputA),
      reinterpret_cast<T*>(inputB),
      reinterpret_cast<float*>(inputC),
      reinterpret_cast<float*>(res)));
}

} // namespace basic_gemm
} // namespace gpu::cutlass_kernel
