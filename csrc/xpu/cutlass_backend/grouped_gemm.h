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

#pragma once

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
bool debug = false;
bool collect_gflops = false;
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
  

  int num_of_expert;

  Options(int64_t * offset, int N, int K, int ne):
    num_of_expert(ne), n(N), k(K), error(false), help(false), alpha(FLT_MAX), beta(FLT_MAX), iterations(100) {
      if (debug) {
        std::cout << "Options()" << std::endl;
      }
    int group_cnt = 0;
    for (int i = 0; i < num_of_expert; ++i){
      if (offset[i] != 0){
        group_cnt++;
      } 
    }
    problem_sizes_host.reserve(group_cnt);
    for (int i = 0; i < num_of_expert; ++i){
      if (offset[i] != 0){
        problem_sizes_host.push_back({static_cast<int>(offset[i]), n, k});
      } 
    }
    groups = group_cnt;
  }
  
  /// Compute performance in GFLOP/s
  double gflops(double runtime_s, std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host) const
  {
    // Number of real-valued multiply-adds
    uint64_t fmas = uint64_t();

    for (auto const & problem : problem_sizes_host) {
      fmas += static_cast<uint64_t>(get<0>(problem)) *
              static_cast<uint64_t>(get<1>(problem)) *
              static_cast<uint64_t>(get<2>(problem));
    }
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * uint64_t(fmas);
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }

};


template <class Gemm>
struct GroupedGemmRunner {
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

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
  using ElementAccumulator = ElementOutput;


  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  // Host-side allocations
  std::vector<int64_t> offset_C;
  // std::vector<int64_t> offset_D;

  std::vector<StrideA> stride_A_host;
  std::vector<StrideB> stride_B_host;
  std::vector<StrideC> stride_C_host;
  std::vector<StrideD> stride_D_host;

  // std::vector<ElementAccumulator> alpha_host;
  // std::vector<ElementAccumulator> beta_host;

  // Device-side allocations
  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

  // This example defines all matrices in a single allocation (e.g. block_A), but this is not a
  // requirement. Matrix base pointers are read from device allocation (e.g. ptr_A)
  cutlass::DeviceAllocation<const ElementC *> ptr_C;
  // cutlass::DeviceAllocation<ElementOutput *> ptr_D;

  cutlass::DeviceAllocation<StrideA> stride_A;
  cutlass::DeviceAllocation<StrideB> stride_B;
  cutlass::DeviceAllocation<StrideC> stride_C;
  cutlass::DeviceAllocation<StrideD> stride_D;

  cutlass::DeviceAllocation<ElementC> block_C;
  // Note, this is an array of pointers to alpha and beta scaling values per group
  // cutlass::DeviceAllocation<ElementAccumulator*> alpha_device;
  // cutlass::DeviceAllocation<ElementAccumulator*> beta_device;
  // cutlass::DeviceAllocation<ElementAccumulator> block_alpha;
  // cutlass::DeviceAllocation<ElementAccumulator> block_beta;

  void release(){
    problem_sizes.release();
    ptr_C.release();
    stride_A.release();
    stride_B.release();
    stride_C.release();
    stride_D.release();
    block_C.release();
  }

  /// Allocates device-side data
void allocate(const Options &options) {
  if (debug){
    std::cout << "void allocate()" << std::endl;
  }
  int64_t total_elements_C = 0;
  // int64_t total_elements_D = 0;
  
  // Compute total allocation sizes across group
  for (int32_t i = 0; i < options.groups; ++i) {
    

    auto problem = options.problem_sizes_host.at(i);
    auto M = get<0>(problem);
    auto N = get<1>(problem);
    auto K = get<2>(problem);

    // Offset into block allocation of each matrix base pointer
    offset_C.push_back(total_elements_C);
    // offset_D.push_back(total_elements_D);

    int64_t elements_C = M * N;
    // int64_t elements_D = M * N;

    total_elements_C += elements_C;
    // total_elements_D += elements_D;

    stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
    stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
    stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1}));
    stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));
  }
  // block_C.reset(total_elements_C);
  // block_alpha.reset(options.groups);
  // block_beta.reset(options.groups);
}

  void initialize(const Options &options) {
    if (debug){
      std::cout << "void initialize()" << std::endl;
    }
    problem_sizes.reset(options.groups);
    problem_sizes.copy_from_host(options.problem_sizes_host.data());
   
    std::vector<ElementC *> ptr_C_host(options.groups);
    // std::vector<ElementC *> ptr_D_host(options.groups);
    // std::vector<ElementAccumulator *> ptr_alpha_host(options.groups);
    // std::vector<ElementAccumulator *> ptr_beta_host(options.groups);

    // Compute offsets, alpha & beta over group on host
    for (int32_t i = 0; i < options.groups; ++i) {
      ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
      // ptr_D_host.at(i) = block_D + offset_D.at(i);
      // Fill host vector of alpha & beta with random values if using per-group values
      // alpha_host.push_back(static_cast<ElementAccumulator>(1));
      // beta_host.push_back(static_cast<ElementAccumulator>(0));
      // // Fill host ptr vectors with offset addresses into device alpha/beta blocks
      // ptr_alpha_host.at(i) = block_alpha.get() + i;
      // ptr_beta_host.at(i) = block_beta.get() + i;
    }

    // // Allocate device memory & copy from host
    ptr_C.reset(options.groups);
    ptr_C.copy_from_host(ptr_C_host.data());

    // ptr_D.reset(options.groups);
    // ptr_D.copy_from_host(ptr_D_host.data());

    stride_A.reset(options.groups);
    stride_A.copy_from_host(stride_A_host.data());

    stride_B.reset(options.groups);
    stride_B.copy_from_host(stride_B_host.data());

    stride_C.reset(options.groups);
    stride_C.copy_from_host(stride_C_host.data());

    stride_D.reset(options.groups);
    stride_D.copy_from_host(stride_D_host.data());

    // Per-group alpha and beta ptrs
    // alpha_device.reset(options.groups);
    // alpha_device.copy_from_host(ptr_alpha_host.data());
    // beta_device.reset(options.groups);
    // beta_device.copy_from_host(ptr_beta_host.data());


    // initialize_block(block_C, 666);
    // Per-group alpha and beta values - note these are not directly passed to kernel - the pointers
    // (alpha_device/beta_device) are passed instead
    // block_alpha.copy_from_host(alpha_host.data());
    // block_beta.copy_from_host(beta_host.data());

  }

  /// Populates a Gemm::Arguments structure from the given commandline options
  typename Gemm::Arguments args_from_options(const Options &options, 
                                             const cutlass::KernelHardwareInfo& hw_info, 
                                             const ElementA ** ptr_A,
                                             const ElementB ** ptr_B,
                                             ElementOutput ** ptr_D,
                                             ElementAccumulator ** ptr_alpha,
                                             ElementAccumulator ** ptr_beta,
                                             bool host_problem_shapes_available = true)
  {
    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args;

    if (options.alpha != FLT_MAX && options.beta != FLT_MAX) {
      // If both alpha/beta are provided (via cmd line args) and are scalar, i.e., same alpha/beta applies to all batches.
      fusion_args.alpha = options.alpha;
      fusion_args.beta = options.beta;
      fusion_args.alpha_ptr = nullptr;
      fusion_args.beta_ptr = nullptr;
      fusion_args.alpha_ptr_array = nullptr;
      fusion_args.beta_ptr_array = nullptr;
      // Single alpha and beta for all groups
      fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
      fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};
    }
    else {
      // If pointers to alpha/beta are provided, i.e., alpha/beta can differ between batches/groups.
      fusion_args.alpha = 0;
      fusion_args.beta = 0;
      fusion_args.alpha_ptr = nullptr;
      fusion_args.beta_ptr = nullptr;
      fusion_args.alpha_ptr_array = ptr_alpha;
      fusion_args.beta_ptr_array = ptr_beta;
      // One alpha and beta per each group
      fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
      fusion_args.dBeta = {cute::_0{}, cute::_0{}, 1};
    }
    using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerXeGroup<ProblemShape>::RasterOrderOptions;

    // Per-GEMM problem shape info may only exist on the device.
    if (host_problem_shapes_available) {
      arguments = typename Gemm::Arguments {
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {options.groups, problem_sizes.get(), options.problem_sizes_host.data()},
        {ptr_A, stride_A.get(), ptr_B, stride_B.get()},
        {fusion_args, ptr_C.get(), stride_C.get(), ptr_D, stride_D.get()},
        hw_info,
        {1, RasterOrderOptions::AlongN}
      };
    }
    else {
      arguments = typename Gemm::Arguments {
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {options.groups, problem_sizes.get(), nullptr},
        {ptr_A, stride_A.get(), ptr_B, stride_B.get()},
        {fusion_args, ptr_C.get(), stride_C.get(), ptr_D, stride_D.get()},
        hw_info,
        {1, RasterOrderOptions::AlongN}
      };
    }

    return arguments;
  }


  cutlass::Status run(
      const Options& options,
      sycl::queue* stream,
      const cutlass::KernelHardwareInfo& hw_info,
      const ElementA** ptr_A,
      const ElementB** ptr_B,
      ElementOutput** ptr_D,
      ElementAccumulator** ptr_alpha,
      ElementAccumulator** ptr_beta) {
    if (debug){
      std::cout << "enter run" << std::endl;
    }

    // std::vector<ElementA *> ptr_AA_host(options.groups);
    
    // stream->memcpy(ptr_AA_host.data(), ptr_AA, options.groups * sizeof(int64_t)).wait();
    // // cutlass::device_memory::copy_from_device(ptr_A_host.data(), ptr_A, options.groups);
    // for (int i = 0; i < options.groups; ++i){
    //   std::cout << "AA ptr:" << ptr_AA_host.at(i) << std::endl;
    // }

    allocate(options);
    initialize(options);
    Gemm gemm_op;

    auto arguments = args_from_options(options, hw_info, 
                                        ptr_A,
                                        ptr_B,
                                        ptr_D,
                                        ptr_alpha,
                                        ptr_beta,
                                        true);

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm_op.can_implement(arguments));

    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
    
    if (debug){
      std::cout << "before run kernel" << std::endl;
    }
    // Run the GEMM
    CUTLASS_CHECK(gemm_op.run());
  
    if (collect_gflops){
      std::cout << "collect_gflops:" << collect_gflops << std::endl;
      GPU_Clock timer;
      timer.start();
      for (int iter = 0; iter < 100; ++iter) {
        CUTLASS_CHECK(gemm_op.run());
      }
      syclcompat::wait();

      float cute_time = timer.seconds() * 1000;
      double cute_average_time = double(cute_time) / double(options.iterations);
      double gflops = options.gflops(cute_average_time / 1000.0, options.problem_sizes_host);
      std::cout << "  Avg runtime : " << cute_average_time << " ms" << std::endl;
      std::cout << "  GFLOPS      : " << gflops << std::endl;

    }
    stream->throw_asynchronous();
    release();
    return cutlass::Status::kSuccess;
  }
};

void kernel_functor(
    sycl::queue* stream,
    void* ptr_A,
    void* ptr_B,
    void* ptr_D,
    void* ptr_alpha,
    void* ptr_beta,
    void* offset,
    int64_t N,
    int64_t K,
    int64_t groups){
 //
  // Run examples
  //
  auto offset_ptr = reinterpret_cast<int64_t*>(offset);
  Options options(offset_ptr, N, K, groups);
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
      options,
      stream,
      hw_info,
      reinterpret_cast<const ElementA**>(ptr_A),
      reinterpret_cast<const ElementB**>(ptr_B),
      reinterpret_cast<ElementOutput**>(ptr_D),
      reinterpret_cast<ElementAccumulator**>(ptr_alpha),
      reinterpret_cast<ElementAccumulator**>(ptr_beta));
 
}

} // namespace grouped_gemm
} // namespace gpu::cutlass_kernel
