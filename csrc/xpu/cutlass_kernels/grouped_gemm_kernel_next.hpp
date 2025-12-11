/***************************************************************************************************
 * Copyright 2025 Intel corporation. All rights reserved.
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

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/platform/platform.h"
#include "collective/grouped_gemm/grouped_gemm_mainloop.hpp"
#include "collective/grouped_gemm/grouped_gemm_scheduler.hpp"
#include <cute/util/compat.hpp>

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace cutlass::grouped_gemm::kernel {
using namespace cute;

template <typename T, char LayoutKind>
CUTE_DEVICE auto make_moe_tensor(T* ptr, int r, int c) {
  auto shape = make_shape(r, c);
  if constexpr (LayoutKind == 'C')
    return make_tensor(
        make_gmem_ptr(ptr), make_layout(shape, make_stride(_1{}, r)));
  else
    return make_tensor(
        make_gmem_ptr(ptr), make_layout(shape, make_stride(c, _1{})));
}

template <
    class CollectiveMainloop_,
    class CollectiveEpilogue_,
    class TileScheduler_,
    char LayoutKindA,
    char LayoutKindB,
    char LayoutKindD>
class XeGroupedGEMMKernel {
 public:
  using CollectiveMainloop = CollectiveMainloop_;
  using GmemTiledCopyA = typename CollectiveMainloop::GmemTiledCopyA;
  using GmemTiledCopyB = typename CollectiveMainloop::GmemTiledCopyB;
  using GmemTiledCopyC = typename CollectiveMainloop::GmemTiledCopyC;
  using TiledMMA = typename CollectiveMainloop::TiledMMA;
  using ElementA = typename CollectiveMainloop::ElementA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using ElementS = typename CollectiveMainloop::ElementS;
  using ElementBI = typename CollectiveMainloop::ElementBI;
  using ElementD = typename CollectiveMainloop::ElementD;

  using CollectiveEpilogue = CollectiveEpilogue_;
  using TileScheduler = TileScheduler_;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr TiledMMA mma{};
  static constexpr auto wg_tile = mma.tile_mnk();
  static constexpr auto wg_tile_m = get<0>(wg_tile);
  static constexpr auto wg_tile_n = get<1>(wg_tile);

  static constexpr char actual_layout_of_B = LayoutKindB ^ ('R' ^ 'C');
  static constexpr bool is_B_int4 = (std::is_same_v<ElementB, uint8_t>) &&
                                    (!std::is_same_v<ElementS, uint8_t>);
  static constexpr bool is_B_mxfp4 = (std::is_same_v<ElementB, uint8_t>) &&
                                     (std::is_same_v<ElementS, uint8_t>);
  static constexpr bool is_B_4bits = std::is_same_v<ElementB, uint8_t>;

  // Device side arguments
  struct KernelArguments {
    const ElementA* Activations;
    const ElementB* Weights;
    const ElementS* Scales;
    const ElementBI* Bias;
    ElementD* Outputs;
    const int32_t* rows_for_experts;
    const int32_t num_experts;
    const int32_t group_size;
    const int32_t gemm_n;
    const int32_t gemm_k;
    int32_t* atomic_buffer;
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    // MainloopArguments mainloop{};
    // EpilogueArguments epilogue{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel entry point API
  struct Params {
    KernelParams kernel;
    // MainloopParams mainloop;
    // EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return {args.kernel, args.scheduler};
    // CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
    // CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
    // TileScheduler::to_underlying_arguments(args.scheduler)};
  }

  static bool can_implement(Arguments const& args) {
    return true;
    // return CollectiveMainloop::can_implement(args.mainloop) &&
    //    CollectiveEpilogue::can_implement(args.epilogue);
  }

  static int get_workspace_size(Arguments const& args) { return 0; }

  static cutlass::Status initialize_workspace(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr,
      CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape() {
    int sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    auto MaxThreadsPerWorkgroup = size(mma);
    static constexpr int MaxThreadsPerSM = 512;
    TORCH_CHECK(
        MaxThreadsPerSM % MaxThreadsPerWorkgroup == 0,
        "MaxThreadsPerSM must be divisible by MaxThreadsPerWorkgroup")
    return dim3(1, sm_count * MaxThreadsPerSM / MaxThreadsPerWorkgroup, 1);
  }

  static dim3 get_block_shape() {
    auto MaxThreadsPerWorkgroup = size(mma);
    return dim3(MaxThreadsPerWorkgroup, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) const {
    auto& p = params.kernel;

    CollectiveMainloop mainloop{};
    TileScheduler scheduler(params.scheduler, wg_tile_m, wg_tile_n, smem_buf);

    // loop all gemms
    while (true) {
      auto work_gemm_info = scheduler.get_next_gemm();

      int expert_id = get<0>(work_gemm_info);
      int pre_rows = get<1>(work_gemm_info);
      int gemm_m = get<2>(work_gemm_info);

      if (expert_id >= p.num_experts) {
        break;
      }

      int64_t B_offset = static_cast<int64_t>(expert_id) *
                         static_cast<int64_t>(p.gemm_n) *
                         static_cast<int64_t>(p.gemm_k);
      if constexpr (is_B_4bits) {
        B_offset /= 2;
      }

      ElementA* ptr_A_curr_batch =
          const_cast<ElementA*>(p.Activations) + pre_rows * p.gemm_k;
      ElementB* ptr_B_curr_batch = const_cast<ElementB*>(p.Weights) + B_offset;
      ElementD* ptr_D_curr_batch = p.Outputs + pre_rows * p.gemm_n;
      ElementS* ptr_Scales_curr_batch =
          const_cast<ElementS*>(p.Scales) + expert_id;
      if constexpr (is_B_4bits) {
        ptr_Scales_curr_batch =
            const_cast<ElementS*>(p.Scales) + B_offset * 2 / p.group_size;
      }

      ElementBI* ptr_Bias_curr_batch = nullptr;
      if (p.Bias != static_cast<ElementBI*>(nullptr)) {
        ptr_Bias_curr_batch =
            const_cast<ElementBI*>(p.Bias) + expert_id * p.gemm_n;
      }

      auto A_tensor = make_moe_tensor<ElementA, LayoutKindA>(
          ptr_A_curr_batch, gemm_m, p.gemm_k);
      auto B_tensor = [&]() {
        if constexpr (is_B_int4) {
          return make_moe_tensor<int4_t, actual_layout_of_B>(
              reinterpret_cast<int4_t*>(ptr_B_curr_batch), p.gemm_n, p.gemm_k);
        } else if constexpr (is_B_mxfp4) {
          return make_moe_tensor<float_e2m1_t, actual_layout_of_B>(
              reinterpret_cast<float_e2m1_t*>(ptr_B_curr_batch),
              p.gemm_n,
              p.gemm_k);
        } else {
          return make_moe_tensor<ElementB, actual_layout_of_B>(
              ptr_B_curr_batch, p.gemm_n, p.gemm_k);
        }
      }();
      auto D_tensor = make_moe_tensor<ElementD, LayoutKindD>(
          ptr_D_curr_batch, gemm_m, p.gemm_n);

      // loop all tiles of current gemm
      while (true) {
        auto tile_coord = scheduler.get_next_tile_coord();

        if constexpr (is_B_4bits) {
#define XE_GEMM_4BITS_CALLER(GroupSize)    \
  mainloop.template operator()<GroupSize>( \
      A_tensor,                            \
      B_tensor,                            \
      ptr_Scales_curr_batch,               \
      ptr_Bias_curr_batch,                 \
      D_tensor,                            \
      tile_coord,                          \
      mma);

          if (p.group_size == 32) {
            XE_GEMM_4BITS_CALLER(32)
          } else if (p.group_size == 64) {
            XE_GEMM_4BITS_CALLER(64)
          } else if (p.group_size == 128) {
            XE_GEMM_4BITS_CALLER(128)
          } else if (p.group_size == 256) {
            XE_GEMM_4BITS_CALLER(256)
          }

#undef XE_GEMM_4BITS_CALLER
        } else {
          mainloop(
              A_tensor,
              B_tensor,
              ptr_Scales_curr_batch,
              ptr_Bias_curr_batch,
              D_tensor,
              tile_coord,
              mma);
        }

        if (scheduler.is_get_next_gemm()) {
          break;
        }
      }
    }
  }
};

}  // namespace cutlass::grouped_gemm::kernel
