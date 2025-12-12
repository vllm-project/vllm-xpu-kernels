/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

namespace cutlass::grouped_gemm::epilogue {
using namespace cute;

template <class CollectiveMainloop_>
class GroupedGEMMEpilogue {
 public:
  using CollectiveMainloop = CollectiveMainloop_;
  using ElementD = typename CollectiveMainloop::ElementD;
  using TiledMMA = typename CollectiveMainloop::TiledMMA;
  using GmemTiledCopyC = typename CollectiveMainloop::GmemTiledCopyC;

  static constexpr auto ATOM_N = CollectiveMainloop{}.ATOM_N;
  static constexpr auto tile_n = CollectiveMainloop{}.tile_n;
  static constexpr auto SG_M = CollectiveMainloop{}.SG_M;
  static constexpr auto SG_N = CollectiveMainloop{}.SG_N;
  static constexpr auto sg_local_range = CollectiveMainloop{}.sg_local_range;

  static constexpr auto wg_tile = TiledMMA{}.tile_mnk();

  // User-facing arguments
  struct Arguments {};

  // Kernel-facing parameters
  using Params = Arguments;

  static constexpr Params to_underlying_arguments(Arguments const& args) {
    return {};
  }

  CUTLASS_HOST_DEVICE static bool can_implement() { return true; }

  CUTLASS_HOST_DEVICE
  GroupedGEMMEpilogue() {}

  template <class SGDTensor, typename ElementBI, typename Coord>
  CUTLASS_DEVICE void
  apply_bias(SGDTensor& Accum, const ElementBI* Bias, Coord& tile_coord) {
    auto wg_n = get<1>(tile_coord);

    auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;

    int sg_local_id = cutlass::get_sub_group_local_id();

    int n_tile_start = wg_n * tile_n;
    int n_sg_start = sg_local_n_coord * SG_N;

    CUTLASS_PRAGMA_UNROLL
    for (int sn = 0; sn < SG_N / sg_local_range; ++sn) {
      int sg_local_n = sn * sg_local_range + sg_local_id;
      float b_float = Bias[n_tile_start + n_sg_start + sg_local_n];
      CUTLASS_PRAGMA_UNROLL
      for (int sm = 0; sm < SG_M; ++sm) {
        Accum(sn * SG_M + sm) += b_float;
      }
    }
  }

  template <class DTensor, class SGCTensor, typename Coord>
  CUTLASS_DEVICE void
  apply_copy(SGCTensor& Accum, DTensor& D_tensor, Coord& tile_coord) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int local_id = item.get_local_linear_id();

    Tensor cC = make_identity_tensor(D_tensor.shape());
    Tensor gC = local_tile(
        cC, wg_tile, tile_coord, Step<_1, _1, X>{});  // (BLK_M,BLK_N)

    auto thr_mma = TiledMMA{}.get_slice(local_id);
    Tensor tCgC = thr_mma.partition_C(gC);
    SubgroupTensor tCrC = thr_mma.partition_sg_fragment_C(gC);

    ElementD tCrC_final_frag[tCrC.size()];
    Tensor tCrC_final_tensor =
        make_tensor(make_rmem_ptr(tCrC_final_frag), tCrC.layout());
    SubgroupTensor tCrC_final_sg_tensor =
        make_subgroup_tensor(tCrC_final_tensor, tCrC.tv_layout());

    auto copy_c = get_block_2d_copy_D<GmemTiledCopyC>(TiledMMA{}, D_tensor);

    reorder(Accum, tCrC_final_sg_tensor);
    copy(copy_c, tCrC_final_sg_tensor, tCgC);
  }

  template <class DTensor, class SGCTensor, typename ElementBI, typename Coord>
  CUTLASS_DEVICE void operator()(
      DTensor& D_tensor,
      SGCTensor& Accum,
      const ElementBI* Bias,
      Coord& tile_coord) {
    if (Bias != nullptr) {
      apply_bias(Accum, Bias, tile_coord);
    }
    apply_copy(Accum, D_tensor, tile_coord);
  }
};
}  // namespace cutlass::grouped_gemm::epilogue
