/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
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
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////
namespace cutlass::gemm {

struct KernelMoEArrayCooperative {};

template <int Stages_, class KernelSchedule = KernelMoEArrayCooperative>
struct MainloopMoE16Group {
  constexpr static int Stages = Stages_;
  constexpr static int SubgroupSize = 16;
  using ArchTag = arch::IntelXe;
  using Schedule = KernelSchedule;
  using ClusterShape = Shape<_1, _1, _1>;
};

}  // namespace cutlass::gemm

/////////////////////////////////////////////////////////////////////////////////////////////////
namespace cutlass::gemm::collective {
using namespace cute;
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int Stages,
    class Schedule,
    class TileShape_,
    class ElementA_,
    class StrideA_,
    class ElementB_,
    class StrideB_,
    class TiledMma_,
    class GmemTiledCopyA_,
    class SmemLayoutAtomA_,
    class SmemCopyAtomA_,
    class TransformA_,
    class GmemTiledCopyB_,
    class SmemLayoutAtomB_,
    class SmemCopyAtomB_,
    class TransformB_>
struct CollectiveMma<
    MainloopMoE16Group<Stages, Schedule>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
    : public CollectiveMma<
          MainloopXeL1Staged<Stages, Schedule>,
          TileShape_,
          ElementA_,
          StrideA_,
          ElementB_,
          StrideB_,
          TiledMma_,
          GmemTiledCopyA_,
          SmemLayoutAtomA_,
          SmemCopyAtomA_,
          TransformA_,
          GmemTiledCopyB_,
          SmemLayoutAtomB_,
          SmemCopyAtomB_,
          TransformB_> {
  //
  // Type Aliases
  //
  using Base = CollectiveMma<
      MainloopXeL1Staged<Stages, Schedule>,
      TileShape_,
      ElementA_,
      StrideA_,
      ElementB_,
      StrideB_,
      TiledMma_,
      GmemTiledCopyA_,
      SmemLayoutAtomA_,
      SmemCopyAtomA_,
      TransformA_,
      GmemTiledCopyB_,
      SmemLayoutAtomB_,
      SmemCopyAtomB_,
      TransformB_>;
  using BaseArguments = typename Base::Arguments;
  using BaseParams = typename Base::Params;

  using DispatchPolicy = MainloopMoE16Group<Stages, Schedule>;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using InternalStrideA = typename Base::StrideA;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using InternalStrideB = typename Base::StrideB;

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    ElementB const* ptr_B;
  };

  using Params = Arguments;

  //
  // Methods
  //

  CollectiveMma() = default;

  static constexpr Params to_underlying_arguments(Arguments const& args) {
    return args;
  }

  template <class ProblemShape>
  static bool can_implement(int64_t N, int64_t K, Arguments const& args) {
    constexpr int copy_alignment_bits = 128;
    constexpr int batch_alignment_bits = 512;
    int M, L = 1;
    bool implementable = true;

    constexpr int min_aligned_elements_A =
        copy_alignment_bits / sizeof_bits<ElementA>::value;
    constexpr int min_aligned_elements_B =
        copy_alignment_bits / sizeof_bits<ElementB>::value;
    constexpr int min_batch_aligned_elements_A =
        batch_alignment_bits / sizeof_bits<ElementA>::value;
    constexpr int min_batch_aligned_elements_B =
        batch_alignment_bits / sizeof_bits<ElementB>::value;

    implementable &= cutlass::detail::check_alignment<min_aligned_elements_A>(
        cute::make_shape(M, K, L), InternalStrideA{});
    implementable &= cutlass::detail::check_alignment<min_aligned_elements_B>(
        cute::make_shape(N, K, L), InternalStrideB{});

    if (L > 1) {
      implementable &=
          get<2>(InternalStrideA{}) % min_batch_aligned_elements_A == 0;
      implementable &=
          get<2>(InternalStrideB{}) % min_batch_aligned_elements_B == 0;
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST(
          "  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment "
          "requirements for XE 2D copy.\n");
    }

    return implementable;
  }

  CUTLASS_DEVICE static constexpr BaseArguments
  to_base_arguments(Arguments const& args, int idx) {
    return BaseArguments{
        args.ptr_A[idx], args.dA[idx], args.ptr_B[idx], args.dB[idx]};
  }

  template <typename ProblemShape_MNKL>
  CUTLASS_DEVICE static constexpr BaseArguments to_base_arguments(
      Arguments const& mainloop_params,
      int next_group,
      ProblemShape_MNKL const& problem_shape_mnkl,
      int64_t expert_first_token_offset,
      int64_t /*expert_first_scale_offset*/ = 0) {
    const int32_t M = get<0>(problem_shape_mnkl);
    const int32_t N = get<1>(problem_shape_mnkl);
    const int32_t K = get<2>(problem_shape_mnkl);

    ElementA const* ptr_A_curr_batch =
        static_cast<ElementA const*>(mainloop_params.ptr_A) +
        expert_first_token_offset * K;
    ElementB const* ptr_B_curr_batch =
        static_cast<ElementB const*>(mainloop_params.ptr_B) +
        static_cast<int64_t>(next_group) * N * K;
    StrideA dA = cutlass::make_cute_packed_stride(InternalStrideA{}, {M, K, 1});
    StrideB dB = cutlass::make_cute_packed_stride(InternalStrideB{}, {N, K, 1});

    return BaseArguments{ptr_A_curr_batch, dA, ptr_B_curr_batch, dB};
  }
};

}  // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
