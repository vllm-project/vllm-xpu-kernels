/***************************************************************************************************
 * Copyright (c) 2025 Intel Corporation. All rights reserved.
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
    \brief CUTLASS Intel BMG MoE API example based on sycl-tla Group GEMM

*/
#pragma once

#include <torch/all.h>
#include "utils.h"

#include <cute/tensor.hpp>
#include <random>

#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/initialize_block.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/sycl_event_manager.hpp"

// #include "collective/grouped_gemm/grouped_gemm_policy.hpp"
#include "collective/gemm/moe_gemm_array_cooperative.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace cutlass::grouped_gemm {
using namespace cute;

struct grouped_gemm_args_t {
  void* Activations;
  void* Weights;
  void* Scales;
  void* Bias;
  void* Outputs;
  void* expert_first_token_offset;
  int32_t num_experts;
  int32_t group_size;
  int32_t gemm_n;
  int32_t gemm_k;
  void* atomic_buffer;
};

template <class GroupedGEMMKernel>
struct KernelLauncher {
  using CollectiveMainloop = typename GroupedGEMMKernel::CollectiveMainloop;
  using ElementA = typename CollectiveMainloop::ElementA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using ElementS = typename CollectiveMainloop::ElementS;
  using ElementBI = typename CollectiveMainloop::ElementBI;
  using ElementD = typename CollectiveMainloop::ElementD;

  cutlass::Status
  run(sycl::queue& queue,
      const grouped_gemm_args_t& args,
      const cutlass::KernelHardwareInfo& hw_info) {
    typename GroupedGEMMKernel::Arguments arguments{
        {reinterpret_cast<ElementA*>(args.Activations),
         reinterpret_cast<ElementB*>(args.Weights),
         args.Scales == nullptr ? nullptr
                                : reinterpret_cast<ElementS*>(args.Scales),
         args.Bias == nullptr ? nullptr
                              : reinterpret_cast<ElementBI*>(args.Bias),
         reinterpret_cast<ElementD*>(args.Outputs),
         args.num_experts,
         args.group_size,
         args.gemm_n,
         args.gemm_k},
        {},
        {},
        {reinterpret_cast<int64_t*>(args.expert_first_token_offset),
         args.num_experts,
         args.gemm_n,
         static_cast<int*>(args.atomic_buffer)}};

    auto params = GroupedGEMMKernel::to_underlying_arguments(arguments);

    run(queue, params);

    return cutlass::Status::kSuccess;
  }

  static void
  run(sycl::queue& queue, typename GroupedGEMMKernel::Params params) {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    const auto sycl_block = GroupedGEMMKernel::get_block_shape();
    const auto sycl_grid = GroupedGEMMKernel::get_grid_shape();

    int smem_size = 1 * sizeof(int32_t);

    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<256>};
    compat::experimental::launch_policy launch_pol{
        sycl_grid, sycl_block, launch_props, kernel_props};
    auto event =
        compat::experimental::launch<cutlass::device_kernel<GroupedGEMMKernel>>(
            launch_pol, queue, params);
    EventManager::getInstance().addEvent(event);
  }
};

template <
    char layoutA,
    char layoutB,
    class policy,
    typename ElementA,
    typename ElementB,
    typename ElementS,
    typename ElementBI,
    typename ElementD>
struct GroupedGEMMConfig {
  static void run(sycl::queue& queue, const grouped_gemm_args_t& args) {
    cutlass::KernelHardwareInfo hw_info;

    using ElementA_non_CV = cutlass::platform::remove_cv_t<ElementA>;
    auto op = XE_DPAS_TT<8, float, ElementA_non_CV>{};

    using WGTile = typename policy::WGTile;
    using SGLayout = typename policy::SGLayout;
    using MMA = typename TiledMMAHelper<
        MMA_Atom<decltype(op)>,
        Layout<WGTile>,
        SGLayout>::TiledMMA;

    using GmemTiledCopyA = typename policy::GmemTiledCopyA;
    using GmemTiledCopyB = typename policy::GmemTiledCopyB;
    using GmemTiledCopyD = typename policy::GmemTiledCopyD;

    static constexpr bool is_B_4bits = std::is_same_v<ElementB, uint8_t>;
    using CollectiveMainloop = std::conditional_t<
        is_B_4bits,
        cutlass::grouped_gemm::mainloop::XeGEMMMainloop4Bits<
            GmemTiledCopyA,
            GmemTiledCopyB,
            GmemTiledCopyD,
            MMA,
            ElementA,
            ElementB,
            ElementS,
            ElementBI,
            ElementD>,
        cutlass::grouped_gemm::mainloop::XeGEMMMainloop<
            GmemTiledCopyA,
            GmemTiledCopyB,
            GmemTiledCopyD,
            MMA,
            ElementA,
            ElementB,
            ElementS,
            ElementBI,
            ElementD>>;

    using CollectiveEpilogue =
        cutlass::grouped_gemm::epilogue::GroupedGEMMEpilogue<
            CollectiveMainloop>;
    using TileScheduler =
        cutlass::grouped_gemm::scheduler::XeGroupedGEMMTileScheduler;

    using GroupedGEMMKernel =
        cutlass::grouped_gemm::kernel::XeGroupedGEMMKernel<
            CollectiveMainloop,
            CollectiveEpilogue,
            TileScheduler,
            layoutA,
            layoutB,
            'R'>;
    KernelLauncher<GroupedGEMMKernel> launcher;

    launcher.run(queue, args, hw_info);
  }
};
}  // namespace cutlass::grouped_gemm