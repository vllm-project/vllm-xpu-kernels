/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
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

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.h"

namespace cutlass::grouped_gemm::scheduler {

struct XeGroupedGEMMTileScheduler {
 public:
  struct Arguments {
    int* rows_for_experts;
    int num_experts;
    int gemm_n;
    int* atomic_buffer;
  };

  using Params = Arguments;

  Params to_underlying_arguments(Arguments const& args) { return args; }

  CUTLASS_DEVICE
  XeGroupedGEMMTileScheduler(
      Params const& params_, int wg_tile_m_, int wg_tile_n_, char* smem_buf)
      : params(params_), wg_tile_m(wg_tile_m_), wg_tile_n(wg_tile_n_) {
    slm_mem = reinterpret_cast<int32_t*>(smem_buf);

    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    local_id = item.get_local_linear_id();
    group_id = item.get_group_linear_id();
    group_range = item.get_group_range(1);

    gemm_n_pad = (params.gemm_n + wg_tile_n - 1) / wg_tile_n * wg_tile_n;
    group_m_id = (group_id * wg_tile_n) / gemm_n_pad;

    if (group_id == 0 && local_id == 0) {
      auto atm = sycl::atomic_ref<
          int,
          sycl::memory_order::relaxed,
          sycl::memory_scope::device,
          sycl::access::address_space::global_space>(params.atomic_buffer[0]);
      atm.store(0);
    }

    pre_rows = 0;
    pre_tiles = 0;

    cumsum_rows = 0;
    cumsum_tiles = 0;

    next_gemm_id = -1;
  }

  CUTLASS_DEVICE
  auto get_next_gemm() {
    pre_rows = cumsum_rows;
    pre_tiles = cumsum_tiles;

    int gemm_m = 0;

    for (++next_gemm_id; next_gemm_id < params.num_experts; ++next_gemm_id) {
      gemm_m = params.rows_for_experts[next_gemm_id];
      cumsum_rows = gemm_m + pre_rows;
      cumsum_tiles = (gemm_m + wg_tile_m - 1) / wg_tile_m + pre_tiles;

      if (group_m_id >= cumsum_tiles) {
        pre_rows = cumsum_rows;
        pre_tiles = cumsum_tiles;
        continue;
      }

      break;
    }

    return cute::tuple(next_gemm_id, pre_rows, gemm_m);
  }

  CUTLASS_DEVICE
  auto get_next_tile_coord() {
    int n_coord = (group_id * wg_tile_n) % gemm_n_pad / wg_tile_n;
    int m_coord = (group_m_id - pre_tiles);

    return make_coord(m_coord, n_coord, _, 0);
  }

  CUTLASS_DEVICE
  bool is_get_next_gemm() {
    if (local_id == 0) {
      slm_mem[0] = cutlass::atomicAdd(params.atomic_buffer, 1);
    }
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    item.barrier(sycl::access::fence_space::local_space);
    group_id = group_range + slm_mem[0];
    group_m_id = (group_id * wg_tile_n) / gemm_n_pad;

    if (group_m_id < cumsum_tiles) {
      return false;
    }
    return true;
  }

 private:
  Params params;
  int wg_tile_m;
  int wg_tile_n;
  int32_t* slm_mem;

  int local_id;
  int group_id;
  int group_range;

  int gemm_n_pad;
  int group_m_id;

  int pre_rows;
  int pre_tiles;
  int cumsum_rows;
  int cumsum_tiles;

  int next_gemm_id;
};

}  // namespace cutlass::grouped_gemm::scheduler