/***************************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief CUTLASS Intel BMG Group Gemm

    This file is almost a complete copy of 04_bmg_grouped_gemm,
    except that it's used for FP8 (E5M2 & E4M3) datatype inputs.

    This example demonstrates fusing multiple GEMM operations into one kernel.

    Note that the scalar arguments to e.g. the standard 00_bmg_gemm example, have been
    replaced with vector equivalents, as each individual GEMM has its own inputs and outputs, which
    needn't be contiguous in memory. For example, where 00_bmg_gemm receives an `ElementA *`
    defining Matrix A, grouped gemm receives a `ElementA **`, i.e. a pointer to pointers, each
    pointing to a distinct Matrix A. Likewise, each individual GEMM operation may have its own alpha
    and beta factors for linear combination. This example demonstrates two approaches: the user can
    provide `options.alpha` and `options.beta`, in which case they will apply to all GEMMs;
    otherwise, random values are generated per GEMM.

    Group GEMM scheduling (cutlass::gemm::GroupScheduler) is more complex than standard GEMM,
    because each GEMM may have a unique size, only known at runtime. Thus, the scheduler will
    distribute an a priori unknown number of tiles to each work-group. See
    include/cutlass/gemm/kernel/xe_gemm_array_cooperative.hpp for implementation.

    Note that for simplicity, this example sets every GEMM in the group to the same shape.

    Verification for this example is a conventional GEMM kernel, executed iteratively per group.

    To build & run this example (from your build dir):

      $ ninja 09_bmg_grouped_gemm_fp8
      $ ./examples/sycl/09_bmg_grouped_gemm_fp8/09_bmg_grouped_gemm_fp8

    Call with `--help` for information about available options.

    Note: the code may spill registers once compiled which will result in sub-optimal performance. This is because
    of an issue inside Intel Graphics Compiler (IGC) related to VectorAliasBBThreshold being debugged internally.
    To avoid register spills, build the example by setting the environment variable:
      $ export IGC_VectorAliasBBThreshold=10000
*/
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_array_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "sycl_common.hpp"
#include "helper.h"

#include <cfloat>

using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group

using ElementAccumulator = float;     // <- data type of accumulator
using ElementComputeEpilogue = float; // <- data type of epilogue operations
using ElementA = bfloat16_t;          // <- data type of elements in input matrix A
using ElementB = bfloat16_t;          // <- data type of elements in input matrix B
using ElementOutput = float;          // <- data type of elements in output matrix D

///////////////////////////////////////////////////////////////////////////////////////////////////


namespace gpu::cutlass_kernel {
namespace grouped_gemm {

struct Options {

  bool error = false;
  bool help = false;

  float alpha, beta;
  int iterations;
  int m, n, k, groups;
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;
  
  int hidden_size;
  int intermediate_size;
  int* offset;
  int num_of_expert;

  Options() : error(false), help(false), alpha(FLT_MAX), beta(FLT_MAX), iterations(100),
              m(5120), n(4096), k(4096), groups(2) {
    problem_sizes_host.reserve(groups);
    for(int i = 0; i < groups; i++) {
      problem_sizes_host.push_back({m, n, k});
    }
  }

};


template <class Gemm>
struct GroupedGemmRunner {
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
  using ElementOffset = int64_t;

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
      const cutlass::KernelHardwareInfo& hw_info,
      ElementA* inputA,
      ElementB* inputB,
      ElementOffset* offset,
      ElementOutput* res,
      int64_t hidden_size,
      int64_t intermediate_size,
      int64_t num_of_expert) {
  
    Options options(offset, hidden_size, intermediate_size, num_of_expert);

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

void kernel_functor(
    sycl::queue* stream,
    void* input,
    void* weight,
    void* res,
    void* offset,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t num_of_expert){
 //
  // Run examples
  //

  // The KernelHardwareInfo struct holds the number of EUs on the GPU with a given device ID. This
  // information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;

  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementOffset = int64_t;
  using ElementOutput = float;
  using ElementScale = cutlass::bfloat16_t;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using TileShape = Shape<_256, _256, _32>;
  using GmemTiledCopyA =
      XE_2D_U16x32x32_LD_N; // Note: This shape has to match the shape used for
                           // the scaling factors
  using GmemTiledCopyB =
      XE_2D_U16x32x32_LD_V; // Note: This shape has to match the shape used for
                           // the scaling factors
  
  using TiledMma =
      TiledMMA<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
               Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>,
               Tile<Layout<Shape<_8, _8, _4>, Stride<_1, _32, _8>>,
                    Layout<Shape<_16, _4, _4>, Stride<_1, _64, _16>>, _32>>;
  
  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16Group<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16Group;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementComputeEpilogue,
          ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          ElementAccumulator,
          cutlass::gemm::TagToStrideC_t<LayoutC*>,
          ElementOutput,
          cutlass::gemm::TagToStrideC_t<LayoutD*>,
          FusionCallBacks,
          XE_2D_U32x8x16_LD_N,
          void, void,
          XE_2D_U32x8x16_ST_N,
          void, void>;

// Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
          GEMMDispatchPolicy,
          TileShape,
          ElementA,
          cutlass::gemm::TagToStrideA_t<LayoutA*>,
          ElementB,
          cutlass::gemm::TagToStrideB_t<LayoutB*>,
          TiledMma,
          GmemTiledCopyA, void, void, cute::identity,  // A
          GmemTiledCopyB, void, void, cute::identity   // B
  >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
  ProblemShape,
  CollectiveMainloop,
  CollectiveEpilogue,
  cutlass::gemm::GroupScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  GroupedGemmRunner<Gemm> runner;
  runner.run(
      stream,
      hw_info,
      reinterpret_cast<ElementA*>(input),
      reinterpret_cast<ElementB*>(weight),
      reinterpret_cast<ElementOffset*>(offset),
      reinterpret_cast<ElementOutput*>(res),
      hidden_size,
      intermediate_size,
      num_of_expert);
 
}

} // namespace grouped_gemm
} // namespace gpu::cutlass_kernel
