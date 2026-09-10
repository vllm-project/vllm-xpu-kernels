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
#include "helper.h"
#include "collective/moe_dtype_policy.hpp"

using namespace cute;
using ProblemShape =
    cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per
                                                             // group

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace gpu::cutlass_kernel {
namespace grouped_gemm {

template <typename Gemm, bool NeedScale>
struct ElementScaleSelector {
  using A = void;
  using B = void;
};

template <typename Gemm>
struct ElementScaleSelector<Gemm, true> {
  using A = typename Gemm::CollectiveMainloop::ElementScaleA;
  using B = typename Gemm::CollectiveMainloop::ElementScaleB;
};

template <class Gemm, bool NeedScale>
struct GroupedGemmRunner {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;

  using CollectiveMainloop = typename Gemm::CollectiveMainloop;
  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementOutput = CollectiveEpilogue::ElementOutput;
  using ElementAccumulator = typename Gemm::ElementAccumulator;

  using ElementScaleA = typename ElementScaleSelector<Gemm, NeedScale>::A;
  using ElementScaleB = typename ElementScaleSelector<Gemm, NeedScale>::B;

  /// Populates a Gemm::Arguments structure from the given commandline options
  typename Gemm::Arguments args_from_options(
      const cutlass::KernelHardwareInfo& hw_info,
      int const* rows_per_expert,
      const ElementA* ptr_A,
      const ElementScaleA* ptr_A_scale,
      const ElementB* ptr_B,
      const ElementScaleB* ptr_B_scale,
      const ElementC* ptr_C,
      ElementOutput* ptr_D,
      int64_t N,
      int64_t K,
      int64_t groups,
      int block_size) {
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
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};
    using RasterOrderOptions = typename cutlass::gemm::kernel::detail::
        PersistentTileSchedulerMoE::RasterOrderOptions;

    // Per-GEMM problem shape info may only exist on the device.
    if constexpr (!NeedScale) {
      arguments = typename Gemm::Arguments{
          cutlass::gemm::GemmUniversalMode::kGrouped,
          {ptr_A, ptr_B},
          {fusion_args, ptr_C, ptr_D},
          rows_per_expert,
          N,
          K,
          groups,
          hw_info,
          {1, RasterOrderOptions::AlongN}};
    } else {
      arguments = typename Gemm::Arguments{
          cutlass::gemm::GemmUniversalMode::kGrouped,
          {ptr_A, ptr_B, ptr_A_scale, ptr_B_scale, block_size},
          {fusion_args, ptr_C, ptr_D},
          rows_per_expert,
          N,
          K,
          groups,
          hw_info,
          {1, RasterOrderOptions::AlongN}};
    }

    return arguments;
  }

  cutlass::Status
  run(sycl::queue& stream,
      const cutlass::KernelHardwareInfo& hw_info,
      int const* rows_per_expert,
      const ElementA* ptr_A,
      const ElementScaleA* ptr_A_scale,
      const ElementB* ptr_B,
      const ElementScaleB* ptr_B_scale,
      const ElementC* ptr_C,
      ElementOutput* ptr_D,
      int64_t N,
      int64_t K,
      int64_t groups,
      int block_size) {
    Gemm gemm_op;

    auto arguments = args_from_options(
        hw_info,
        rows_per_expert,
        ptr_A,
        ptr_A_scale,
        ptr_B,
        ptr_B_scale,
        ptr_C,
        ptr_D,
        N,
        K,
        groups,
        block_size);

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm_op.can_implement(arguments));

    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get(), &stream));

    // Run the GEMM
    CUTLASS_CHECK(gemm_op.run(&stream));
    return cutlass::Status::kSuccess;
  }
};

template <class moe_policy>
void kernel_functor(
    sycl::queue& stream,
    void* ptr_A,
    void* ptr_A_scale,
    void* ptr_B,
    void* ptr_B_scale,
    void* ptr_bias,
    void* ptr_D,
    void* rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t groups) {
  //
  // Run examples
  //
  // The KernelHardwareInfo struct holds the number of EUs on the GPU with a
  // given device ID. This information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;

  // Change device_id to another value if you are running on a machine with
  // multiple GPUs and wish to use a GPU other than that with device ID 0.
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  using Gemm = typename moe_policy::Gemm;
  GroupedGemmRunner<Gemm, moe_policy::NeedScale> runner;

  runner.run(
      stream,
      hw_info,
      reinterpret_cast<const int*>(rows_per_expert),
      reinterpret_cast<const typename moe_policy::ElementA*>(ptr_A),
      reinterpret_cast<const typename moe_policy::ElementScaleA*>(ptr_A_scale),
      reinterpret_cast<const typename moe_policy::ElementB*>(ptr_B),
      reinterpret_cast<const typename moe_policy::ElementScaleB*>(ptr_B_scale),
      reinterpret_cast<const typename moe_policy::ElementAccumulator*>(
          ptr_bias),
      reinterpret_cast<typename moe_policy::ElementOutput*>(ptr_D),
      N,
      K,
      groups,
      moe_policy::BlockSize);
}

#define INSTANTIATE_KERNEL(POLICY)      \
  template void kernel_functor<POLICY>( \
      sycl::queue & stream,             \
      void* ptr_A,                      \
      void* ptr_A_scale,                \
      void* ptr_B,                      \
      void* ptr_B_scale,                \
      void* ptr_bias,                   \
      void* ptr_D,                      \
      void* rows_per_expert,            \
      int64_t N,                        \
      int64_t K,                        \
      int64_t groups);

INSTANTIATE_KERNEL(moe_bf16_policy)
INSTANTIATE_KERNEL(moe_bf16_256x128_policy)
INSTANTIATE_KERNEL(moe_bf16_128x256_policy)
INSTANTIATE_KERNEL(moe_bf16_128x128_policy)
INSTANTIATE_KERNEL(moe_bf16_mid_policy)
INSTANTIATE_KERNEL(moe_bf16_decode_policy)
INSTANTIATE_KERNEL(moe_bf16_decode_k64_policy)
INSTANTIATE_KERNEL(moe_fp16_policy)
INSTANTIATE_KERNEL(moe_fp16_mid_policy)
INSTANTIATE_KERNEL(moe_fp16_decode_policy)
INSTANTIATE_KERNEL(moe_mxfp4_policy)
INSTANTIATE_KERNEL(moe_mxfp4_downproj_wide_policy)
INSTANTIATE_KERNEL(moe_mxfp4_256x128_policy)
INSTANTIATE_KERNEL(moe_mxfp4_128x256_policy)
INSTANTIATE_KERNEL(moe_mxfp4_128x128_policy)
INSTANTIATE_KERNEL(moe_mxfp4_mid_policy)
INSTANTIATE_KERNEL(moe_mxfp4_decode_policy)
INSTANTIATE_KERNEL(moe_mxfp8_policy)
INSTANTIATE_KERNEL(moe_mxfp8_256x128_policy)
INSTANTIATE_KERNEL(moe_mxfp8_128x256_policy)
INSTANTIATE_KERNEL(moe_mxfp8_128x128_policy)
INSTANTIATE_KERNEL(moe_mxfp8_mid_policy)
INSTANTIATE_KERNEL(moe_mxfp8_decode_policy)
INSTANTIATE_KERNEL(moe_w4a8_policy)
INSTANTIATE_KERNEL(moe_w4a8_mid_policy)
INSTANTIATE_KERNEL(moe_fp8block_policy)
INSTANTIATE_KERNEL(moe_fp8block_mid_policy)
INSTANTIATE_KERNEL(moe_fp8block_decode_policy)
INSTANTIATE_KERNEL(moe_fp8pertensor_policy)
INSTANTIATE_KERNEL(moe_fp8pertensor_mid_policy)
INSTANTIATE_KERNEL(moe_fp8pertensor_decode_policy)
INSTANTIATE_KERNEL(moe_fp8pertensor_decode_lowk_policy)
INSTANTIATE_KERNEL(moe_fp8pertensor_decode_narrow_policy)
INSTANTIATE_KERNEL(moe_fp8pertensor_decode_shortk_policy)

}  // namespace grouped_gemm
}  // namespace gpu::cutlass_kernel
