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

    This file is based off on the default cutlass group gemm example in the
   cutlass-sycl repo, but it uses custom kernels.

    Note: the code may spill registers once compiled which will result in
   sub-optimal performance. This is because of an issue inside Intel Graphics
   Compiler (IGC) related to VectorAliasBBThreshold being debugged internally.
    To avoid register spills, build the example by setting the environment
   variable: $ export IGC_VectorAliasBBThreshold=10000
*/

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_array_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
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
#include "helper.h"
#include <cfloat>

#include "cutlass/gemm/collective/collective_mma_decl.hpp"

using namespace cute;
using ProblemShape =
    cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per
                                                             // group

using ElementAccumulator = float;      // <- data type of accumulator
using ElementComputeEpilogue = float;  // <- data type of epilogue operations
using ElementA = bfloat16_t;  // <- data type of elements in input tensor A
using ElementB = bfloat16_t;  // <- data type of elements in input tensor B
using ElementOutput =
    bfloat16_t;  // <- data type of elements in output tensor D
bool debug = false;
bool collect_gflops = false;
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace gpu::cutlass_kernel {
namespace grouped_gemm {

template <class Gemm>
struct GroupedGemmRunner {
  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementOutput = bfloat16_t;
  using ElementAccumulator = float_t;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  // Compiler fails if std::vector<const ElementA *> ptr_A_host(1)
  // is used, so using resize() as a workaround
  std::vector<const ElementA*> ptr_A_host;
  std::vector<const ElementB*> ptr_B_host;
  std::vector<const ElementC*> ptr_C_host;
  std::vector<ElementOutput*> ptr_D_host;

  // We need to pass pointers of pointers, so this allocation is unavoidable
  // because the pointer to a GPU pointer obtained via & is obviously a CPU
  // pointer.
  cutlass::DeviceAllocation<const ElementA*> ptr_A_device;
  cutlass::DeviceAllocation<const ElementB*> ptr_B_device;
  cutlass::DeviceAllocation<const ElementC*> ptr_C_device;
  cutlass::DeviceAllocation<ElementOutput*> ptr_D_device;

  /// Populates a Gemm::Arguments structure
  typename Gemm::Arguments generate_gemm_args(
      const cutlass::KernelHardwareInfo& hw_info, const ElementA* ptr_A,
      const ElementB* ptr_B, ElementOutput* ptr_D,
      const int32_t* num_rows_per_expert, const int32_t& N, const int32_t& K,
      const int32_t& num_groups) {
    ptr_A_host.resize(1);
    ptr_B_host.resize(1);
    ptr_C_host.resize(1);
    ptr_D_host.resize(1);
    ptr_A_device.reset(1);
    ptr_B_device.reset(1);
    ptr_C_device.reset(1);
    ptr_D_device.reset(1);
    ptr_A_host.at(0) = ptr_A;
    ptr_B_host.at(0) = ptr_B;
    ptr_C_host.at(0) = nullptr;
    ptr_D_host.at(0) = ptr_D;
    ptr_A_device.copy_from_host(ptr_A_host.data());
    ptr_B_device.copy_from_host(ptr_B_host.data());
    ptr_C_device.copy_from_host(ptr_C_host.data());
    ptr_D_device.copy_from_host(ptr_D_host.data());
    typename Gemm::Arguments arguments;
    decltype(arguments.fusion_args) fusion_args;
    fusion_args.alpha = 1;
    fusion_args.beta = 0;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    // One alpha and beta per each group
    // Doesn't matter, because we won't use it.
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 1};
    using RasterOrderOptions =
        typename cutlass::gemm::kernel::detail::PersistentTileSchedulerXeGroup<
            ProblemShape>::RasterOrderOptions;

    arguments =
        typename Gemm::Arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
                                 ptr_A_device.get(),
                                 ptr_B_device.get(),
                                 ptr_C_device.get(),
                                 ptr_D_device.get(),
                                 fusion_args,
                                 hw_info,
                                 {1, RasterOrderOptions::AlongN},
                                 num_rows_per_expert,
                                 num_groups,
                                 N,
                                 K};

    return arguments;
  }

  cutlass::Status run(sycl::queue& stream,
                      const cutlass::KernelHardwareInfo& hw_info,
                      const ElementA* ptr_A, const ElementB* ptr_B,
                      ElementOutput* ptr_D, const int32_t* num_rows_per_expert,
                      const int32_t& N, const int32_t& K,
                      const int32_t& num_groups) {
    if (debug) {
      std::cout << "enter run" << std::endl;
    }

    Gemm gemm_op;

    auto arguments = generate_gemm_args(hw_info, ptr_A, ptr_B, ptr_D,
                                        num_rows_per_expert, N, K, num_groups);

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm_op.can_implement(arguments));

    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

    if (debug) {
      std::cout << "before run kernel" << std::endl;
    }
    // Run the GEMM
    CUTLASS_CHECK(gemm_op.run());
    return cutlass::Status::kSuccess;
  }
};

void kernel_functor(sycl::queue& stream, void* ptr_A, void* ptr_B, void* ptr_D,
                    void* tokens_per_expert, int32_t N, int32_t K,
                    int32_t groups) {
  auto num_rows_per_expert = reinterpret_cast<int32_t*>(tokens_per_expert);
  // The KernelHardwareInfo struct holds the number of EUs on the GPU with a
  // given device ID. This information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;

  // Change device_id to another value if you are running on a machine with
  // multiple GPUs and wish to use a GPU other than that with device ID 0.
  // TODO: we should get device id with a vllm or torch API.
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementOutput = bfloat16_t;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using TileShape = Shape<_256, _256, _32>;
  using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
  using GmemTiledCopyB = XE_2D_U16x16x16_LD_T;
  // This TiledMMA is the default one in intel/cutlass-sycl examples
  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<TileShape>,
                                    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy =
      cutlass::gemm::MainloopIntelXeXMX16Group<PipelineStages,
                                               cutlass::gemm::KernelXeMoEGEMM>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16Group;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      float_t, float_t, float_t, float_t,
      cutlass::FloatRoundStyle::round_to_nearest, false>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp, TileShape,
          Shape<_1, _1, _1>, cutlass::epilogue::collective::EpilogueTileAuto,
          float, float, float, LayoutC, 1, ElementOutput, LayoutC, 1,
          EpilogueDispatchPolicy, EpilogueOp>::CollectiveOp;

  // Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy, TileShape, ElementA,
      cutlass::gemm::TagToStrideA_t<LayoutA*>, ElementB,
      cutlass::gemm::TagToStrideB_t<LayoutB*>, TiledMma, GmemTiledCopyA, void,
      void, cute::identity,                       // A
      GmemTiledCopyB, void, void, cute::identity  // B
      >;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                           CollectiveEpilogue,
                                           cutlass::gemm::GroupScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  GroupedGemmRunner<Gemm> runner;
  // Might wanna throw an exception here
  runner.run(stream, hw_info, reinterpret_cast<const ElementA*>(ptr_A),
             reinterpret_cast<const ElementB*>(ptr_B),
             reinterpret_cast<ElementOutput*>(ptr_D), num_rows_per_expert, N, K,
             groups);
}

}  // namespace grouped_gemm
}  // namespace gpu::cutlass_kernel
