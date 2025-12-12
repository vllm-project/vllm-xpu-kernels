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

#include "collective/grouped_gemm/grouped_gemm_policy.hpp"
#include "grouped_gemm_kernel.hpp"

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
  void* rows_for_experts;
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
         reinterpret_cast<int*>(args.rows_for_experts),
         args.num_experts,
         args.group_size,
         args.gemm_n,
         args.gemm_k,
         static_cast<int*>(args.atomic_buffer)},
        {},
        {},
        {reinterpret_cast<int*>(args.rows_for_experts),
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

at::Tensor cutlass_grouped_gemm(
    at::Tensor& ptr_A,
    at::Tensor& ptr_B,
    const c10::optional<at::Tensor>& ptr_scales,
    const c10::optional<at::Tensor>& ptr_bias,
    at::Tensor& ptr_D,
    at::Tensor& num_rows_per_expert_device,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    bool is_B_int4,
    bool is_B_mxfp4) {
  auto& dpcpp_queue =
      at::xpu::getCurrentXPUStream(ptr_A.device().index()).queue();
  auto A_dtype = ptr_A.dtype();
  auto B_dtype = ptr_B.dtype();
  bool is_weight_fp8 =
      ((B_dtype == at::kFloat8_e4m3fn) || (B_dtype == at::kFloat8_e5m2));

  TORCH_CHECK(N % 32 == 0, "N must be divisible by 32");

  TORCH_CHECK(ptr_A.dim() == 2, "ptr_A must be 2D [Total_M, K]");
  TORCH_CHECK(ptr_B.dim() == 3, "ptr_B must be 3D [num_experts, K, N]");
  TORCH_CHECK(ptr_D.dim() == 2, "ptr_D must be 2D [Total_M, N]");
  if (ptr_bias.has_value()) {
    TORCH_CHECK(ptr_bias->dim() == 2, "ptr_bias must be 2D [num_experts, N]");
  }

  TORCH_CHECK(ptr_A.is_contiguous(), "ptr_A must be contiguous");
  TORCH_CHECK(ptr_B.is_contiguous(), "ptr_B must be contiguous");
  TORCH_CHECK(ptr_D.is_contiguous(), "ptr_D must be contiguous");
  if (ptr_bias.has_value()) {
    TORCH_CHECK(ptr_bias->is_contiguous(), "ptr_bias must be contiguous");
  }

  int A_total_M = ptr_A.size(0);
  int A_K = ptr_A.size(1);

  int B_E = ptr_B.size(0);
  int B_K = ptr_B.size(1);
  int B_N = ptr_B.size(2);
  if (is_B_int4 || is_B_mxfp4) {
    B_K = ptr_B.size(2) * 2;
    B_N = ptr_B.size(1);
  }

  int D_total_M = ptr_D.size(0);
  int D_N = ptr_D.size(1);
  int group_size = -1;
  int A_avg_M = A_total_M / num_experts;

  TORCH_CHECK(B_E == num_experts, "ptr_B.size(0) must match num_experts");
  TORCH_CHECK(A_total_M == D_total_M, "ptr_A.size(0) must match ptr_D.size(0)");
  TORCH_CHECK(A_K == B_K && B_K == K, "ptr_A.size(1) must match ptr_B.size(1)");
  TORCH_CHECK(B_N == D_N && D_N == N, "ptr_B.size(2) must match ptr_D.size(1)");
  if (ptr_bias.has_value()) {
    TORCH_CHECK(
        ptr_bias->size(0) == num_experts,
        "ptr_bias.size(0) must match num_experts");
    TORCH_CHECK(ptr_bias->size(1) == N, "ptr_bias.size(1) must match N");
  }

  at::Tensor atomic_buffer =
      at::empty({static_cast<long>(1)}, ptr_A.options().dtype(at::kInt));

  grouped_gemm_args_t args = {
      ptr_A.data_ptr(),
      ptr_B.data_ptr(),
      ptr_scales.has_value() ? ptr_scales->data_ptr() : nullptr,
      ptr_bias.has_value() ? ptr_bias->data_ptr() : nullptr,
      ptr_D.data_ptr(),
      num_rows_per_expert_device.data_ptr(),
      static_cast<int32_t>(num_experts),
      static_cast<int32_t>(group_size),
      static_cast<int32_t>(N),
      static_cast<int32_t>(K),
      static_cast<int32_t*>(atomic_buffer.data_ptr())};

#define MoEGEMMLauncherCallER(                              \
    LayoutA, LayoutB, Policy, ElementA, ElementB, ElementS) \
  GroupedGEMMConfig<                                        \
      LayoutA,                                              \
      LayoutB,                                              \
      Policy,                                               \
      ElementA,                                             \
      ElementB,                                             \
      ElementS,                                             \
      ElementA,                                             \
      ElementA>::run(dpcpp_queue, args);

  if (is_B_int4 || is_B_mxfp4) {
    TORCH_CHECK(ptr_scales.has_value(), "w8a16 grouped gemm must have scales");
    TORCH_CHECK(ptr_scales->is_contiguous(), "ptr_scales must be contiguous");
    TORCH_CHECK(
        ptr_scales->dim() == 3,
        "ptr_scales of int4 must be 3D [num_experts, group_num, N]");
    TORCH_CHECK(
        ptr_scales->size(0) == num_experts,
        "ptr_scales.size(0) of int4 must match num_experts");
    TORCH_CHECK(
        K % ptr_scales->size(2) == 0,
        "ptr_scales.size(2) of int4 must be divisible by K");
    TORCH_CHECK(
        ptr_scales->size(1) == N, "ptr_scales.size(1) of int4 must match N");
    int group_num = ptr_scales->size(2);
    group_size = K / group_num;

    TORCH_CHECK(
        group_size == 32 || group_size == 64 || group_size == 128 ||
            group_size == 256,
        "group_size must be 32, 64, 128 or 256");

    args.group_size = static_cast<int32_t>(group_size);

#define W4A16LauncherCallER(policy)                                         \
  if (is_B_int4) {                                                          \
    if (A_dtype == at::kBFloat16) {                                         \
      using scalar_t = bfloat16_t;                                          \
      MoEGEMMLauncherCallER('R', 'C', policy, scalar_t, uint8_t, scalar_t); \
    } else if (A_dtype == at::kHalf) {                                      \
      using scalar_t = half_t;                                              \
      MoEGEMMLauncherCallER('R', 'C', policy, scalar_t, uint8_t, scalar_t); \
    }                                                                       \
  } else if (is_B_mxfp4) {                                                  \
    if (A_dtype == at::kBFloat16) {                                         \
      using scalar_t = bfloat16_t;                                          \
      MoEGEMMLauncherCallER('R', 'C', policy, scalar_t, uint8_t, uint8_t);  \
    } else if (A_dtype == at::kHalf) {                                      \
      using scalar_t = half_t;                                              \
      MoEGEMMLauncherCallER('R', 'C', policy, scalar_t, uint8_t, uint8_t);  \
    }                                                                       \
  }

    if (A_avg_M <= 32) {
      using policy = cutlass::grouped_gemm::policy::w4a16_policy_m_16;
      W4A16LauncherCallER(policy);
    } else if (A_avg_M <= 128) {
      using policy = cutlass::grouped_gemm::policy::w4a16_policy_m_32;
      W4A16LauncherCallER(policy);
    } else {
      using policy = cutlass::grouped_gemm::policy::w4a16_policy;
      W4A16LauncherCallER(policy);
    }
#undef W4A16LauncherCallER
  } else if (is_weight_fp8) {
    TORCH_CHECK(ptr_scales.has_value(), "w8a16 grouped gemm must have scales");
    TORCH_CHECK(ptr_scales->is_contiguous(), "ptr_scales must be contiguous");
    TORCH_CHECK(
        ptr_scales->dim() == 1, "ptr_scales of fp8 must be 1D [num_experts]");
    TORCH_CHECK(
        ptr_scales->size(0) == num_experts,
        "ptr_scales.size(0) of fp8 must match num_experts");

#define W8A16LauncherCallER(policy)                                            \
  if (B_dtype == at::kFloat8_e4m3fn && A_dtype == at::kHalf) {                 \
    using scalar_t = half_t;                                                   \
    MoEGEMMLauncherCallER('R', 'R', policy, scalar_t, float_e4m3_t, scalar_t); \
  } else if (B_dtype == at::kFloat8_e5m2 && A_dtype == at::kHalf) {            \
    using scalar_t = half_t;                                                   \
    MoEGEMMLauncherCallER('R', 'R', policy, scalar_t, float_e5m2_t, scalar_t); \
  } else if (B_dtype == at::kFloat8_e4m3fn && A_dtype == at::kBFloat16) {      \
    using scalar_t = bfloat16_t;                                               \
    MoEGEMMLauncherCallER('R', 'R', policy, scalar_t, float_e4m3_t, scalar_t); \
  } else if (B_dtype == at::kFloat8_e5m2 && A_dtype == at::kBFloat16) {        \
    using scalar_t = bfloat16_t;                                               \
    MoEGEMMLauncherCallER('R', 'R', policy, scalar_t, float_e5m2_t, scalar_t); \
  }

    if (A_avg_M <= 32) {
      using policy = cutlass::grouped_gemm::policy::w8a16_policy_m_16;
      W8A16LauncherCallER(policy);
    } else if (A_avg_M <= 128) {
      using policy = cutlass::grouped_gemm::policy::w8a16_policy_m_32;
      W8A16LauncherCallER(policy);
    } else {
      using policy = cutlass::grouped_gemm::policy::w8a16_policy;
      W8A16LauncherCallER(policy);
    }
#undef W8A16LauncherCallER
  } else {
    TORCH_CHECK(
        !ptr_scales.has_value(), "w16a16 grouped gemm must not have scales");

#define W16A16LauncherCallER(policy)                                       \
  if (A_dtype == at::kBFloat16) {                                          \
    using scalar_t = bfloat16_t;                                           \
    MoEGEMMLauncherCallER('R', 'R', policy, scalar_t, scalar_t, scalar_t); \
  } else if (A_dtype == at::kHalf) {                                       \
    using scalar_t = half_t;                                               \
    MoEGEMMLauncherCallER('R', 'R', policy, scalar_t, scalar_t, scalar_t); \
  }

    if (A_avg_M <= 4) {
      using policy = cutlass::grouped_gemm::policy::w16a16_policy_m_16;
      W16A16LauncherCallER(policy);
    } else {
      using policy = cutlass::grouped_gemm::policy::w16a16_policy;
      W16A16LauncherCallER(policy);
    }
#undef W16A16LauncherCallER
  }
#undef MoEGEMMLauncherCallER
  return ptr_D;
}
}  // namespace cutlass::grouped_gemm