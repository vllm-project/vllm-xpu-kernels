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

#include "xe_grouped_gemm.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoE {
using namespace cute;

// type tag to define a unique sycl kernel name
template <typename, typename, typename, char, char> class GemmCuteName;

template <char layoutA, char layoutB, typename ElementA, typename ElementB,
          typename ElementS, typename ElementBI,typename ElementD>
void MoEGEMMLauncher(sycl::queue& stream, const ElementA *activations, const ElementB *weights,
                     const ElementS *scales, const ElementBI *bias, ElementD *outputs,
                     const int gemm_n, const int gemm_k,
                     const int *num_rows_per_expert_device,
                     const int num_experts, int32_t* atomic_buffer) {
  using ElementA_non_CV = cutlass::platform::remove_cv_t<ElementA>;
  auto op = XE_DPAS_TT<8, float, ElementA_non_CV, ElementA_non_CV>{};
  using WGTile = Shape<_256, _256, _32>; // 256x256 WG tile size
  using SGLayout =
      Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>; // 8x4 SG tiling, n-major

  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>,
                                      SGLayout>::TiledMMA;
  auto mma = MMA{};

  int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto MaxThreadsPerWorkgroup = size(mma);

  sycl::range<3> local(1, 1, MaxThreadsPerWorkgroup);
  sycl::range<3> global(1, sm_count, 1);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{syclex::sub_group_size<16>,
                                  intelex::grf_size<256>};

  static constexpr bool is_B_fp8_type =
      std::is_same_v<ElementB, cutlass::float_e5m2_t> ||
      std::is_same_v<ElementB, cutlass::float_e4m3_t>;

  using GmemTiledCopyA = XE_LOAD_2D<16, 32, 32, 16>;
  using GmemTiledCopyB = std::conditional_t<is_B_fp8_type, XE_LOAD_2D_VNNI<8, 32, 16, 16>, XE_LOAD_2D_VNNI<16, 32, 16, 16>>;
  using GmemTiledCopyD = XE_STORE_2D<16, 8, 32>;

  auto event = stream.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), cgh);
    cgh.parallel_for<
      GemmCuteName<ElementA, ElementB, ElementD, layoutA, layoutB>>(
      sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
        MoE::MoEGEMM<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD,
                     'R', 'R', 'R'>(activations, weights, scales, bias, outputs, mma,
                                    num_rows_per_expert_device, num_experts,
                                    gemm_n, gemm_k, atomic_buffer, local_mem);
      });
  });
  EventManager::getInstance().addEvent(event);
}

template void MoEGEMMLauncher<'R', 'R'>(sycl::queue& stream, const bfloat16_t *activations, const bfloat16_t *weights,
                     const bfloat16_t *scales, const bfloat16_t *bias, bfloat16_t *outputs,
                     const int gemm_n, const int gemm_k,
                     const int *num_rows_per_expert_device,
                     const int num_experts, int32_t* atomic_buffer);
template void MoEGEMMLauncher<'R', 'R'>(sycl::queue& stream, const half_t *activations, const half_t *weights,
                     const half_t *scales, const half_t *bias, half_t *outputs,
                     const int gemm_n, const int gemm_k,
                     const int *num_rows_per_expert_device,
                     const int num_experts, int32_t* atomic_buffer);
template void MoEGEMMLauncher<'R', 'R'>(sycl::queue& stream, const bfloat16_t *activations, const float_e5m2_t *weights,
                     const bfloat16_t *scales, const bfloat16_t *bias, bfloat16_t *outputs,
                     const int gemm_n, const int gemm_k,
                     const int *num_rows_per_expert_device,
                     const int num_experts, int32_t* atomic_buffer);
template void MoEGEMMLauncher<'R', 'R'>(sycl::queue& stream, const bfloat16_t *activations, const float_e4m3_t *weights,
                     const bfloat16_t *scales, const bfloat16_t *bias, bfloat16_t *outputs,
                     const int gemm_n, const int gemm_k,
                     const int *num_rows_per_expert_device,
                     const int num_experts, int32_t* atomic_buffer);
template void MoEGEMMLauncher<'R', 'R'>(sycl::queue& stream, const half_t *activations, const float_e5m2_t *weights,
                     const half_t *scales, const half_t *bias, half_t *outputs,
                     const int gemm_n, const int gemm_k,
                     const int *num_rows_per_expert_device,
                     const int num_experts, int32_t* atomic_buffer);
template void MoEGEMMLauncher<'R', 'R'>(sycl::queue& stream, const half_t *activations, const float_e4m3_t *weights,
                     const half_t *scales, const half_t *bias, half_t *outputs,
                     const int gemm_n, const int gemm_k,
                     const int *num_rows_per_expert_device,
                     const int num_experts, int32_t* atomic_buffer);

at::Tensor cutlass_xe_grouped_gemm(
    at::Tensor& ptr_A,
    at::Tensor& ptr_B,
    const c10::optional<at::Tensor>& ptr_scales,
    const c10::optional<at::Tensor>& ptr_bias,
    at::Tensor& ptr_D,
    at::Tensor& num_rows_per_expert_device,
    int64_t N,
    int64_t K,
    int64_t groups) {
  auto& dpcpp_queue =
      at::xpu::getCurrentXPUStream(ptr_A.device().index()).queue();
  auto A_dtype = ptr_A.dtype();
  auto B_dtype = ptr_B.dtype();
  bool is_weight_fp8 =
      ((B_dtype == at::kFloat8_e4m3fn) || (B_dtype == at::kFloat8_e5m2));

  TORCH_CHECK(N % 32 == 0, "N must be divisible by 32");

  TORCH_CHECK(ptr_A.dim() == 2, "ptr_A must be 2D [Total_M, K]");
  TORCH_CHECK(ptr_B.dim() == 3, "ptr_B must be 3D [groups, K, N]");
  TORCH_CHECK(ptr_D.dim() == 2, "ptr_D must be 2D [Total_M, N]");
  if(ptr_bias.has_value()){
    TORCH_CHECK(ptr_bias->dim() == 2, "ptr_bias must be 2D [groups, N]");
  }

  TORCH_CHECK(ptr_A.is_contiguous(), "ptr_A must be contiguous");
  TORCH_CHECK(ptr_B.is_contiguous(), "ptr_B must be contiguous");
  TORCH_CHECK(ptr_D.is_contiguous(), "ptr_D must be contiguous");
  if(ptr_bias.has_value()){
    TORCH_CHECK(ptr_bias->is_contiguous(), "ptr_bias must be contiguous");
  }

  int A_total_M = ptr_A.size(0);
  int A_K = ptr_A.size(1);

  int B_E = ptr_B.size(0);
  int B_K = ptr_B.size(1);
  int B_N = ptr_B.size(2);

  int D_total_M = ptr_D.size(0);
  int D_N = ptr_D.size(1);
  
  TORCH_CHECK(B_E == groups, "ptr_B.size(0) must match groups");
  TORCH_CHECK(A_total_M == D_total_M, "ptr_A.size(0) must match ptr_D.size(0)");
  TORCH_CHECK(A_K == B_K && B_K == K, "ptr_A.size(1) must match lptr_B.size(1)");
  TORCH_CHECK(B_N == D_N && D_N == N, "ptr_B.size(2) must match ptr_D.size(1)");
  if(ptr_bias.has_value()){
    TORCH_CHECK(ptr_bias->size(0) == groups, "ptr_bias.size(0) must match groups");
    TORCH_CHECK(ptr_bias->size(1) == N, "ptr_bias.size(1) must match N");
  }

  at::Tensor atomic_buffer =
      at::zeros({static_cast<long>(1)}, ptr_A.options().dtype(at::kInt));

  if (is_weight_fp8) {
    TORCH_CHECK(ptr_scales.has_value(), "w8a16 grouped gemm must have scales");
    TORCH_CHECK(ptr_scales->is_contiguous(), "ptr_scales must be contiguous");
    TORCH_CHECK(ptr_scales->dim() == 1, "ptr_scales of fp8 must be 1D [groups]");
    TORCH_CHECK(ptr_scales->size(0) == groups, "ptr_scales.size(0) of fp8 must match groups");
    if (B_dtype == at::kFloat8_e4m3fn && A_dtype == at::kHalf) {
      using scalar_t = half_t;
      MoEGEMMLauncher<'R', 'R'>(
        dpcpp_queue,
        reinterpret_cast<scalar_t*>(ptr_A.data_ptr()),
        reinterpret_cast<float_e4m3_t*>(ptr_B.data_ptr()),
        ptr_scales.has_value() ? reinterpret_cast<scalar_t*>(ptr_scales->data_ptr())
                         : static_cast<scalar_t*>(nullptr),
        ptr_bias.has_value() ? reinterpret_cast<scalar_t*>(ptr_bias->data_ptr())
                         : static_cast<scalar_t*>(nullptr),
        reinterpret_cast<scalar_t*>(ptr_D.data_ptr()),
        N,
        K,
        reinterpret_cast<int*>(num_rows_per_expert_device.data_ptr()),
        groups,
        static_cast<int*>(atomic_buffer.data_ptr()));
    } else if (B_dtype == at::kFloat8_e5m2 && A_dtype == at::kHalf) {
      using scalar_t = half_t;
      MoEGEMMLauncher<'R', 'R'>(
        dpcpp_queue,
        reinterpret_cast<scalar_t*>(ptr_A.data_ptr()),
        reinterpret_cast<float_e5m2_t*>(ptr_B.data_ptr()),
        ptr_scales.has_value() ? reinterpret_cast<scalar_t*>(ptr_scales->data_ptr())
                         : static_cast<scalar_t*>(nullptr),
        ptr_bias.has_value() ? reinterpret_cast<scalar_t*>(ptr_bias->data_ptr())
                         : static_cast<scalar_t*>(nullptr),
        reinterpret_cast<scalar_t*>(ptr_D.data_ptr()),
        N,
        K,
        reinterpret_cast<int*>(num_rows_per_expert_device.data_ptr()),
        groups,
        static_cast<int*>(atomic_buffer.data_ptr()));
    } else if (B_dtype == at::kFloat8_e4m3fn && A_dtype == at::kBFloat16) {
      using scalar_t = bfloat16_t;
      MoEGEMMLauncher<'R', 'R'>(
        dpcpp_queue,
        reinterpret_cast<scalar_t*>(ptr_A.data_ptr()),
        reinterpret_cast<float_e4m3_t*>(ptr_B.data_ptr()),
        ptr_scales.has_value() ? reinterpret_cast<scalar_t*>(ptr_scales->data_ptr())
                         : static_cast<scalar_t*>(nullptr),
        ptr_bias.has_value() ? reinterpret_cast<scalar_t*>(ptr_bias->data_ptr())
                         : static_cast<scalar_t*>(nullptr),
        reinterpret_cast<scalar_t*>(ptr_D.data_ptr()),
        N,
        K,
        reinterpret_cast<int*>(num_rows_per_expert_device.data_ptr()),
        groups,
        static_cast<int*>(atomic_buffer.data_ptr()));
    } else if (B_dtype == at::kFloat8_e5m2 && A_dtype == at::kBFloat16) {
      using scalar_t = bfloat16_t;
      MoEGEMMLauncher<'R', 'R'>(
        dpcpp_queue,
        reinterpret_cast<scalar_t*>(ptr_A.data_ptr()),
        reinterpret_cast<float_e5m2_t*>(ptr_B.data_ptr()),
        ptr_scales.has_value() ? reinterpret_cast<scalar_t*>(ptr_scales->data_ptr())
                         : static_cast<scalar_t*>(nullptr),
        ptr_bias.has_value() ? reinterpret_cast<scalar_t*>(ptr_bias->data_ptr())
                         : static_cast<scalar_t*>(nullptr),
        reinterpret_cast<scalar_t*>(ptr_D.data_ptr()),
        N,
        K,
        reinterpret_cast<int*>(num_rows_per_expert_device.data_ptr()),
        groups,
        static_cast<int*>(atomic_buffer.data_ptr()));
    }
  } else {
    TORCH_CHECK(!ptr_scales.has_value(), "w16a16 grouped gemm must not have scales");
    if (A_dtype == at::kBFloat16) {
      using scalar_t = bfloat16_t;
      MoEGEMMLauncher<'R', 'R'>(
          dpcpp_queue,
          reinterpret_cast<scalar_t*>(ptr_A.data_ptr()),
          reinterpret_cast<scalar_t*>(ptr_B.data_ptr()),
          static_cast<scalar_t*>(nullptr),
          ptr_bias.has_value() ? reinterpret_cast<scalar_t*>(ptr_bias->data_ptr())
                          : static_cast<scalar_t*>(nullptr),
          reinterpret_cast<scalar_t*>(ptr_D.data_ptr()),
          N,
          K,
          reinterpret_cast<int*>(num_rows_per_expert_device.data_ptr()),
          groups,
          static_cast<int*>(atomic_buffer.data_ptr()));
    } else if (A_dtype == at::kHalf) {
      using scalar_t = half_t;
      MoEGEMMLauncher<'R', 'R'>(
          dpcpp_queue,
          reinterpret_cast<scalar_t*>(ptr_A.data_ptr()),
          reinterpret_cast<scalar_t*>(ptr_B.data_ptr()),
          static_cast<scalar_t*>(nullptr),
          ptr_bias.has_value() ? reinterpret_cast<scalar_t*>(ptr_bias->data_ptr())
                          : static_cast<scalar_t*>(nullptr),
          reinterpret_cast<scalar_t*>(ptr_D.data_ptr()),
          N,
          K,
          reinterpret_cast<int*>(num_rows_per_expert_device.data_ptr()),
          groups,
          static_cast<int*>(atomic_buffer.data_ptr()));
    } else {
      TORCH_CHECK(
          false,
          "cutlass_xe_grouped_gemm only supports BFloat16 and Half dtypes, but got: ",
          A_dtype);
    }
  }
  return ptr_D;
}
} // namespace MoE