/***************************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief CUTLASS Intel BMG Group Gemm

    This file is almost a complete copy of 04_bmg_grouped_gemm,
    except that it's used for FP8 (E5M2 & E4M3) datatype inputs.

    This example demonstrates fusing multiple GEMM operations into one kernel.

    Note that the scalar arguments to e.g. the standard 00_bmg_gemm example,
   have been replaced with vector equivalents, as each individual GEMM has its
   own inputs and outputs, which needn't be contiguous in memory. For example,
   where 00_bmg_gemm receives an `ElementA *` defining Matrix A, grouped gemm
   receives a `ElementA **`, i.e. a pointer to pointers, each pointing to a
   distinct Matrix A. Likewise, each individual GEMM operation may have its own
   alpha and beta factors for linear combination. This example demonstrates two
   approaches: the user can provide `options.alpha` and `options.beta`, in which
   case they will apply to all GEMMs; otherwise, random values are generated per
   GEMM.

    Group GEMM scheduling (cutlass::gemm::GroupScheduler) is more complex than
   standard GEMM, because each GEMM may have a unique size, only known at
   runtime. Thus, the scheduler will distribute an a priori unknown number of
   tiles to each work-group. See
    include/cutlass/gemm/kernel/xe_gemm_array_cooperative.hpp for
   implementation.

    Note that for simplicity, this example sets every GEMM in the group to the
   same shape.

    Verification for this example is a conventional GEMM kernel, executed
   iteratively per group.

    To build & run this example (from your build dir):

      $ ninja 09_bmg_grouped_gemm_fp8
      $ ./examples/sycl/09_bmg_grouped_gemm_fp8/09_bmg_grouped_gemm_fp8

    Call with `--help` for information about available options.

    Note: the code may spill registers once compiled which will result in
   sub-optimal performance. This is because of an issue inside Intel Graphics
   Compiler (IGC) related to VectorAliasBBThreshold being debugged internally.
    To avoid register spills, build the example by setting the environment
   variable: $ export IGC_VectorAliasBBThreshold=10000
*/

#pragma once

#include "cutlass/gemm/group_array_problem_shape.hpp"
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
#include <cfloat>

#include "cutlass/gemm/collective/collective_mma_decl.hpp"
#include "collective/gemm/moe_array_mma.hpp"
#include "collective/gemm/moe_array_epilogue.hpp"
#include "collective/gemm/moe_callbacks.hpp"
#include "collective/gemm/moe_dtype_policy.hpp"
#include "collective/gemm/moe_gemm_array_cooperative.hpp"
#include "collective/gemm/moe_tile_scheduler.hpp"

using namespace cute;
using ProblemShape =
    cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per
                                                             // group

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace gpu::cutlass_kernel {
namespace grouped_gemm {

template <class Gemm>
struct GroupedGemmRunner {
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementOutput = typename Gemm::ElementA;
  using ElementAccumulator = float_t;

  /// Populates a Gemm::Arguments structure from the given commandline options
  typename Gemm::Arguments args_from_options(
      const cutlass::KernelHardwareInfo& hw_info,
      int64_t const* expert_first_token_offset,
      const ElementA* ptr_A,
      const ElementB* ptr_B,
      const ElementC* ptr_C,
      ElementOutput* ptr_D,
      int64_t N,
      int64_t K,
      int64_t groups) {
    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args;

    // If pointers to alpha/beta are provided, i.e., alpha/beta can differ
    // between batches/groups.
    fusion_args.alpha = 1;
    fusion_args.beta = ptr_C ? 1 : 0;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    // One alpha and beta per each group
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 1};
    using RasterOrderOptions = typename cutlass::gemm::kernel::detail::
        PersistentTileSchedulerMoE::RasterOrderOptions;

    bool has_bias = ptr_C ? true : false;
    // Per-GEMM problem shape info may only exist on the device.
    arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {ptr_A,
         ptr_B,
         cutlass::make_cute_packed_stride(
             StrideB{}, {static_cast<int>(N), static_cast<int>(K), 1})},
        {fusion_args, ptr_C, ptr_D, has_bias},
        expert_first_token_offset,
        N,
        K,
        groups,
        hw_info,
        {1, RasterOrderOptions::AlongN}};

    return arguments;
  }

  cutlass::Status
  run(sycl::queue& stream,
      const cutlass::KernelHardwareInfo& hw_info,
      int64_t const* expert_first_token_offset,
      const ElementA* ptr_A,
      const ElementB* ptr_B,
      const ElementC* ptr_C,
      ElementOutput* ptr_D,
      int64_t N,
      int64_t K,
      int64_t groups) {
    Gemm gemm_op;

    auto arguments = args_from_options(
        hw_info,
        expert_first_token_offset,
        ptr_A,
        ptr_B,
        ptr_C,
        ptr_D,
        N,
        K,
        groups);

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm_op.can_implement(arguments));

    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

    // Run the GEMM
    CUTLASS_CHECK(gemm_op.run());
    stream.throw_asynchronous();
    return cutlass::Status::kSuccess;
  }
};

template <class moe_policy>
void kernel_functor(
    sycl::queue& stream,
    void* ptr_A,
    void* ptr_B,
    void* ptr_bias,
    void* ptr_D,
    void* expert_first_token_offset,
    int64_t N,
    int64_t K,
    int64_t groups) {
  //
  // Run examples
  //
  compat::set_default_queue(stream);

  // The KernelHardwareInfo struct holds the number of EUs on the GPU with a
  // given device ID. This information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;

  // Change device_id to another value if you are running on a machine with
  // multiple GPUs and wish to use a GPU other than that with device ID 0.
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  using ElementAccumulator = typename moe_policy::ElementAccumulator;
  using ElementComputeEpilogue = typename moe_policy::ElementComputeEpilogue;
  using ElementA = typename moe_policy::ElementA;
  using ElementB = typename moe_policy::ElementB;
  using ElementOutput = typename moe_policy::ElementOutput;
  using ElementScale = typename moe_policy::ElementScale;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using TileShape = Shape<_256, _256, _32>;
  using GmemTiledCopyA =
      void;  // XE_LOAD_2D<16, 32, 32>; //XE_2D_U16x32x32_LD_N;  // Note: This
             // shape has to match the shape used for
             //  the scaling factors
  using GmemTiledCopyB = void;  // XE_LOAD_2D_VNNI<16, 32, 32>;
                                // XE_2D_U16x32x32_LD_V;  // Note: This shape
                                // has to match the shape used for
                                //  the scaling factors
  using MMAOperation = moe_policy::MMAOperation;

  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementA>>,
      Layout<TileShape>,
      Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopMoE16Group<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::MoE16Group;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      float_t,
      float_t,
      float_t,
      float_t,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy,
      EpilogueOp,
      TileShape,
      decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      ElementAccumulator,
      cutlass::detail::TagToStrideC_t<LayoutC*>,
      ElementOutput,
      cutlass::detail::TagToStrideC_t<LayoutD*>,
      FusionCallbacks,
      XE_2D_U32x8x16_LD_N,
      void,
      void,
      XE_2D_U16x8x16_ST_N,
      void,
      void>;

  // Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      ElementA,
      cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementB,
      cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma,
      GmemTiledCopyA,
      void,
      void,
      cute::identity,  // A
      GmemTiledCopyB,
      void,
      void,
      cute::identity  // B
      >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::GroupScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  GroupedGemmRunner<Gemm> runner;
  runner.run(
      stream,
      hw_info,
      reinterpret_cast<const int64_t*>(expert_first_token_offset),
      reinterpret_cast<const ElementA*>(ptr_A),
      reinterpret_cast<const ElementB*>(ptr_B),
      reinterpret_cast<const ElementAccumulator*>(ptr_bias),
      reinterpret_cast<ElementOutput*>(ptr_D),
      N,
      K,
      groups);
}

template void kernel_functor<moe_bf16_policy>(
    sycl::queue& stream,
    void* ptr_A,
    void* ptr_B,
    void* ptr_bias,
    void* ptr_D,
    void* expert_first_token_offset,
    int64_t N,
    int64_t K,
    int64_t groups);
template void kernel_functor<moe_fp16_policy>(
    sycl::queue& stream,
    void* ptr_A,
    void* ptr_B,
    void* ptr_bias,
    void* ptr_D,
    void* expert_first_token_offset,
    int64_t N,
    int64_t K,
    int64_t groups);

}  // namespace grouped_gemm
}  // namespace gpu::cutlass_kernel
