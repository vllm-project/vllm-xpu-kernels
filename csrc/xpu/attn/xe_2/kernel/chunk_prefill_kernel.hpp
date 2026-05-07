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

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"

#include "cute/util/type_traits.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "csrc/xpu/attn/xe_2/collective/chunk_prefill_mainloop.hpp"
#include "csrc/xpu/attn/xe_2/collective/chunk_prefill_epilogue.hpp"

namespace cutlass::fmha::kernel {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
template <bool IsVarLen_ = false>
struct FMHAProblemShape {
  using SeqLenType = cute::
      conditional_t<IsVarLen_, cutlass::fmha::collective::VariableLength, int>;
  int batch;
  int num_heads_q, num_heads_kv;
  SeqLenType seq_len_qo, seq_len_kv;
  int head_size_qk, head_size_vo;
};

///////////////////////////////////////////////////////////////////////////////

template <
    class ProblemShape_,
    class CollectiveMainloop_,
    class CollectiveEpilogue_,
    class TileScheduler_,
    bool SoftmaxLSE_ = false>
class XeFMHAFwdKernel {
 public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  using VariableLength = cutlass::fmha::collective::VariableLength;
  static constexpr bool is_var_len =
      cutlass::fmha::collective::is_variable_length_v<
          typename ProblemShape::SeqLenType>;
  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using TiledMMAQK = typename CollectiveMainloop::TiledMMAQK;
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using TileShapePV = typename CollectiveMainloop::TileShapePV;
  using SubgroupLayoutQK = typename CollectiveMainloop::SubgroupLayoutQK;
  using ElementQ = typename CollectiveMainloop::TensorQ::element_type;
  using ElementK = typename CollectiveMainloop::TensorK::element_type;
  using ElementV = typename CollectiveMainloop::TensorV::element_type;

  using StrideQ = decltype(stride(typename CollectiveMainloop::TensorQ{}));
  using StrideK = decltype(stride(typename CollectiveMainloop::TensorK{}));
  using StrideV = decltype(stride(typename CollectiveMainloop::TensorV{}));

  using SGPerWG = typename CollectiveMainloop::SGPerWG;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;

  // Tile scheduler derived types
  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;
  using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));

  // Template Features
  static constexpr bool PagedKV = CollectiveMainloop::PagedKV;
  static constexpr bool CausalMask = CollectiveMainloop::CausalMask;
  static constexpr bool LocalMask = CollectiveMainloop::LocalMask;
  static constexpr bool Sink = CollectiveEpilogue::Sink;
  static constexpr bool SoftmaxLSE = SoftmaxLSE_;
  using ElementSink = typename CollectiveEpilogue::ElementSink;

  // Kernel level shared memory storage
  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };

  static constexpr int SharedStorageSize =
      is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);

  // Device side arguments
  struct KernelArguments {
    ProblemShape shape;
    const ElementQ* Q;
    StrideQ dQ;
    const ElementK* K;
    StrideK dK;
    const ElementV* V;
    StrideV dV;
    ElementO* O;
    StrideO dO;

    // softmax sink
    const ElementSink* ptr_S;

    // softmax_lse output [total_seqlen_q, num_heads_q] (nullptr if disabled)
    float* softmax_lse;
    int lse_stride;  // = num_heads_q
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
  };

  // Kernel entry point API
  struct Params {
    KernelParams kernel;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  //
  // Methods
  //

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    return {
        args.kernel,
        CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
        CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
        TileScheduler::to_underlying_arguments(
            args.kernel.shape, args.hw_info, TileShapeO{})};
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.mainloop) &&
           CollectiveEpilogue::can_implement(args.epilogue);
  }

  static int get_workspace_size(Arguments const& args) { return 0; }

  static cutlass::Status initialize_workspace(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr,
      CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(
        params.scheduler);
  }

  static dim3 get_block_shape() {
    return dim3(SGPerWG::value * intel::sg_size, 1, 1);
  }

  CUTLASS_DEVICE
  Shape<int, int> get_sequence_length_shape(
      ProblemShape const& problem_shape, int const& batch) {
    if constexpr (is_var_len) {
      if constexpr (PagedKV) {
        auto q_len = cutlass::fmha::collective::apply_variable_length(
            Shape<VariableLength>{problem_shape.seq_len_qo}, batch);
        return Shape<int, int>{
            get<0>(q_len), problem_shape.seq_len_kv.cumulative_length[batch]};
      } else {
        return cutlass::fmha::collective::apply_variable_length(
            Shape<VariableLength, VariableLength>{
                problem_shape.seq_len_qo, problem_shape.seq_len_kv},
            batch);
      }
    } else {
      return Shape<int, int>{
          problem_shape.seq_len_qo, problem_shape.seq_len_kv};
    }
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    auto& p = params.kernel;
    ProblemShape const& s = p.shape;
    int head_group_q = s.num_heads_q / s.num_heads_kv;

    int thr_id = int(ThreadIdxX());
    int sub_group_id = thr_id / intel::sg_size;
    int q_sg_tile = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})));

    auto cS = make_identity_tensor(take<0, 2>(TiledMMAQK{}.tile_mnk()));
    auto tScS = TiledMMAQK{}.get_slice(thr_id).partition_C(cS);
    auto q_offset_wi = get<0>(tScS(0));
    auto q_offset_sg = group_broadcast(
        sycl::ext::oneapi::this_work_item::get_sub_group(), q_offset_wi, 0);

    TileScheduler tile_scheduler{params.scheduler};

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, head_q, idx_b] =
          tile_scheduler.get_block_coord();  // (Q,V,h,b)
      auto blk_qv = make_coord(blk_q, blk_v);
      int head = head_q / head_group_q;

      auto sequence_length_shape = get_sequence_length_shape(s, idx_b);
      auto [seq_len_qo, seq_len_kv] = sequence_length_shape;
      if (blk_q * get<0>(TileShapeQK{}) >= seq_len_qo) continue;

      auto full_tile_offset = seq_len_kv - seq_len_qo;
      int seq_coord =
          cute::min(seq_len_qo, (blk_q * get<0>(TileShapeQK{}) + q_offset_sg));

      // calc sg level seq_len_kv
      const int seq_len =
          CausalMask
              ? LocalMask
                    ? cute::min(
                          seq_len_kv,
                          full_tile_offset + seq_coord + q_sg_tile +
                              params.mainloop.local_right)
                    : cute::min(
                          seq_len_kv, full_tile_offset + seq_coord + q_sg_tile)
              : seq_len_kv;
      const int k_block0 =
          LocalMask
              ? cute::max(
                    seq_coord + full_tile_offset - params.mainloop.local_left,
                    0) /
                    get<1>(TileShapeQK{})
              : 0;
      const int k_blocks = cute::ceil_div(seq_len, get<1>(TileShapeQK{}));
      const int k_blocks_causal =
          CausalMask ? (seq_coord + full_tile_offset) / get<1>(TileShapeQK{})
                     : 0;

      int offset_q = 0, offset_k = 0, offset_v = 0, offset_o = 0;
      if constexpr (is_var_len) {
        auto qo_cumulative = s.seq_len_qo.cumulative_length;
        auto kv_cumulative = s.seq_len_kv.cumulative_length;
        // Use actual seq strides for offset computation
        offset_q = get<0>(p.dQ) * qo_cumulative[idx_b];
        offset_k = PagedKV ? 0 : get<0>(p.dK) * kv_cumulative[idx_b];
        offset_v = PagedKV ? 0 : get<1>(p.dV) * kv_cumulative[idx_b];
        offset_o = get<0>(p.dO) * qo_cumulative[idx_b];
      }

      auto batch_dim_qo = is_var_len ? 1 : s.batch;
      auto batch_dim_kv = (PagedKV || is_var_len) ? 1 : s.batch;
      auto total_seqlen_kv =
          PagedKV ? params.mainloop.total_seqlen_kv : seq_len_kv;
      auto shape_Q =
          make_shape(seq_len_qo, s.head_size_qk, s.num_heads_q, batch_dim_qo);
      auto shape_K = make_shape(
          total_seqlen_kv, s.head_size_qk, s.num_heads_kv, batch_dim_kv);
      auto shape_V = make_shape(
          s.head_size_vo, total_seqlen_kv, s.num_heads_kv, batch_dim_kv);
      auto shape_O =
          make_shape(seq_len_qo, s.head_size_vo, s.num_heads_q, batch_dim_qo);

      auto dcQ = const_cast<ElementQ*>(p.Q + offset_q);
      auto dcK = const_cast<ElementK*>(p.K + offset_k);
      auto dcV = const_cast<ElementV*>(p.V + offset_v);
      auto dcO = const_cast<ElementO*>(p.O + offset_o);

      auto layout_q = make_layout(shape_Q, p.dQ);
      auto layout_k = make_layout(shape_K, p.dK);
      auto layout_v = make_layout(shape_V, p.dV);
      auto layout_o = make_layout(shape_O, p.dO);

      Tensor Q = make_tensor(make_gmem_ptr(dcQ), layout_q);
      Tensor K = make_tensor(make_gmem_ptr(dcK), layout_k);
      Tensor V = make_tensor(make_gmem_ptr(dcV), layout_v);
      Tensor O = make_tensor(make_gmem_ptr(dcO), layout_o);

      // O accumulator types
      FragA tArA;
      FragARow tA_max, tA_sum;

      // Main loop
      int l_coord_qo = is_var_len ? 0 : idx_b;
      int l_coord_kv = (PagedKV || is_var_len) ? 0 : idx_b;
      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);
      mainloop(
          Q(_, _, head_q, l_coord_qo),
          K(_, _, head, l_coord_kv),
          V(_, _, head, l_coord_kv),
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          idx_b,
          k_block0,
          k_blocks,
          k_blocks_causal,
          thr_id,
          seq_len,
          full_tile_offset);

      // return softmax_lse
      if constexpr (SoftmaxLSE) {
        static_assert(
            size<3>(typename TiledMMAPV::ThrLayoutVMNK{}) == 1,
            "softmax_lse requires ReduceK == 1 in TiledMMAPV");
        if (get<1>(blk_qv) == 0 && (thr_id % intel::sg_size == 0)) {
          using ElementA = typename FragA::value_type;
          constexpr float kLn2 = 0.6931471805599453f;
          int q_tile_start = blk_q * get<0>(TileShapeQK{});
          int qo_cumul = 0;
          if constexpr (is_var_len) {
            qo_cumul = s.seq_len_qo.cumulative_length[idx_b];
          }

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tA_sum.size(); i++) {
            int q_in_batch = q_tile_start + q_offset_sg + i;
            if (q_in_batch < seq_len_qo) {
              ElementA sum_val = tA_sum(i);
              // Include sink contribution for LSE computation
              if constexpr (Sink) {
                constexpr double kLog2e = 1.4426950408889634074;
                ElementSink s_head = p.ptr_S[head_q];
                sum_val += sycl::native::exp2(
                    static_cast<ElementA>(s_head * kLog2e) - tA_max(i));
              }
              float lse = static_cast<float>(tA_max(i)) * kLn2 +
                          sycl::log(static_cast<float>(sum_val));
              int global_q = qo_cumul + q_in_batch;
              p.softmax_lse[global_q * p.lse_stride + head_q] = lse;
            }
          }
        }
      }

      if constexpr (
          !is_empty_v<MainloopSharedStorage> &&
          !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }

      // Epilogue
      CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
      if constexpr (Sink) {
        ElementSink s_head = p.ptr_S[head_q];
        epilogue(
            O(_, _, head_q, l_coord_qo),
            tArA,
            tA_max,
            tA_sum,
            blk_qv,
            s_head,
            thr_id);
      } else {
        epilogue(
            O(_, _, head_q, l_coord_qo),
            tArA,
            tA_max,
            tA_sum,
            blk_qv,
            static_cast<ElementSink>(0),
            thr_id);
      }
    }
  }
};

}  // namespace cutlass::fmha::kernel
