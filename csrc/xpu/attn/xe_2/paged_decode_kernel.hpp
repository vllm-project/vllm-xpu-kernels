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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"

#include "cute/util/type_traits.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "./collective/chunk_prefill_mainloop.hpp"
#include "./collective/chunk_prefill_epilogue.hpp"

namespace cutlass::fmha::kernel {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
template <bool IsVarLen_ = false>
struct DecodeProblemShape {
  using SeqLenType = cute::
      conditional_t<IsVarLen_, cutlass::fmha::collective::VariableLength, int>;
  int batch;
  int num_heads_q, num_heads_kv;
  SeqLenType seq_len_qo, seq_len_kv;
  int head_size_qk, head_size_vo;
};

///////////////////////////////////////////////////////////////////////////////

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class XeFMHAFwdSplitKVKernel {

public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  using VariableLength = cutlass::fmha::collective::VariableLength;
  static constexpr bool is_var_len = cutlass::fmha::collective::is_variable_length_v<typename ProblemShape::SeqLenType>;
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
  using ElementLSE = typename CollectiveEpilogue::ElementLSE;
  using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));

  // Kernel level shared memory storage
  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };

  static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0)
                                                                     : sizeof(SharedStorage);

  static constexpr int max_num_kv_splits = SGPerWG::value * intel::sg_size;
  static constexpr int dpas_max_repeat_count = 8;
  static constexpr bool Sink = CollectiveEpilogue::Sink;
  using ElementSink = typename CollectiveEpilogue::ElementSink;

  // Device side arguments
  struct KernelArguments {
    ProblemShape shape;
    const ElementQ *Q;
    StrideQ dQ;
    const ElementK *K;
    StrideK dK;
    const ElementV *V;
    StrideV dV;
    ElementO *Oaccum;
    StrideO dOaccum;
    ElementLSE *exp_sums;
    StrideO dExp_sums;
    ElementLSE *max_logits;
    StrideO dMax_logits;

    const ElementSink *sm_sink;
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    int num_kv_splits = -1; // no split by default
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

  static Params to_underlying_arguments(Arguments const &args, void *workspace) {
    return {args.kernel,
            CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
            TileScheduler::to_underlying_arguments(args.kernel.shape, args.hw_info, TileShapeO{}, args.num_kv_splits)};
  }

  static bool can_implement(Arguments const &args) {
    if (!is_var_len && args.kernel.shape.seq_len_qo != 1) {
      // decode only
      return false;
    }

    if (args.num_kv_splits > max_num_kv_splits) {
      return false;
    }

    // when GQA packing enabled, limit head group size to 8
    if (args.kernel.shape.num_heads_q / args.kernel.shape.num_heads_kv > dpas_max_repeat_count) {
      return false;
    }

    return CollectiveMainloop::can_implement(args.mainloop)
        && CollectiveEpilogue::can_implement(args.epilogue);
  }

  static int get_workspace_size(Arguments const &args) { return 0; }

  static cutlass::Status initialize_workspace(Arguments const &args, void *workspace = nullptr,
                                              cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const &params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(params.scheduler);
  }

  static dim3 get_block_shape() { return dim3(SGPerWG::value * intel::sg_size, 1, 1); }

  CUTLASS_DEVICE
  Shape<int, int> get_sequence_length_shape(ProblemShape const& problem_shape, int const& batch) {
    if constexpr (is_var_len) {
      return cutlass::fmha::collective::apply_variable_length(Shape<VariableLength, VariableLength>{problem_shape.seq_len_qo, problem_shape.seq_len_kv}, batch);
    } else {
      return Shape<int, int>{problem_shape.seq_len_qo, problem_shape.seq_len_kv};
    }
  }

  CUTLASS_DEVICE
  void operator()(Params const &params, char *smem_buf)
  {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage *>(smem_buf);

    auto &p = params.kernel;
    ProblemShape const& s = p.shape;
    int head_group_q = s.num_heads_q / s.num_heads_kv;

    int thr_id = int(ThreadIdxX());
    int sub_group_id = thr_id / intel::sg_size;
    int q_sg_tile = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})));

    auto cS = make_identity_tensor(take<0,2>(TiledMMAQK{}.tile_mnk()));
    auto tScS = TiledMMAQK{}.get_slice(thr_id).partition_C(cS);
    auto q_offset_wi = get<0>(tScS(0));
    auto q_offset_sg = group_broadcast(sycl::ext::oneapi::this_work_item::get_sub_group(), q_offset_wi, 0);

    TileScheduler tile_scheduler{params.scheduler};
    auto num_kv_splits = params.scheduler.num_kv_splits_;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, head, idx_b, idx_kv_split] = tile_scheduler.get_block_coord(); // (Q,V,h,b,id_split)
      auto blk_qv = make_coord(blk_q, blk_v);
      int head_q_start = head * head_group_q;

      auto sequence_length_shape = get_sequence_length_shape(s, idx_b);
      auto [seq_len_qo, seq_len_kv] = sequence_length_shape;
      if (blk_q * get<0>(TileShapeQK{}) >= seq_len_qo) continue;

      auto offset = cute::min(seq_len_qo, seq_len_kv);
      auto discard_seq_coord = seq_len_qo - offset;
      auto full_tile_offset = seq_len_kv - offset;
      int seq_coord = cute::min(seq_len_qo, (blk_q * get<0>(TileShapeQK{}) + q_offset_sg));

      if (CollectiveMainloop::CausalMask && seq_coord < discard_seq_coord) continue;
      const int seq_len = CollectiveMainloop::CausalMask ? 
                            full_tile_offset 
                            + cute::min(seq_len_kv, seq_coord - discard_seq_coord) 
                            + q_sg_tile 
                            : seq_len_kv;
      const int k_blocks = cute::ceil_div(seq_len, get<1>(TileShapeQK{}));

      int offset_q = 0, offset_k = 0, offset_v = 0, offset_o = 0;
      int offset_exp_sums = 0, offset_max_logits = 0;
      if constexpr (is_var_len) {
        auto qo_cumulative = s.seq_len_qo.cumulative_length;
        
        offset_q = s.num_heads_q * s.head_size_qk * qo_cumulative[idx_b];
        offset_o = s.num_heads_q * s.head_size_vo * num_kv_splits * qo_cumulative[idx_b];
        offset_exp_sums = s.num_heads_q * num_kv_splits * qo_cumulative[idx_b];
        offset_max_logits = s.num_heads_q * num_kv_splits * qo_cumulative[idx_b];

        // for gqa packing, seq_len_qo must be 1
        seq_len_qo = 1;
      }

      // neglect seq_len_qo since it's always 1 for decode
      auto batch_dim = is_var_len ? 1 : s.batch;
      auto shape_Q = make_shape(head_group_q, s.head_size_qk, s.num_heads_kv, batch_dim);
      // shape
      auto total_seqlen_kv = params.mainloop.total_seqlen_kv;
      auto shape_K = make_shape(total_seqlen_kv, s.head_size_qk, s.num_heads_kv, batch_dim);
      auto shape_V = make_shape(s.head_size_vo, total_seqlen_kv, s.num_heads_kv, batch_dim);
      
      auto shape_O = make_shape(head_group_q, s.head_size_vo, s.num_heads_kv, num_kv_splits, batch_dim);
      auto shape_exp_sums = make_shape(head_group_q, num_kv_splits, s.num_heads_kv, batch_dim);
      auto shape_max_logits = make_shape(head_group_q, num_kv_splits, s.num_heads_kv, batch_dim);
      auto shape_sink = make_shape(s.num_heads_kv, head_group_q);

      int num_blocks_per_split = cute::ceil_div(k_blocks, num_kv_splits);
      int kv_split_offset = idx_kv_split * num_blocks_per_split;
      int num_effective_kv_blocks = cute::min(k_blocks - kv_split_offset, num_blocks_per_split);

      if (num_effective_kv_blocks <= 0) {
        // no need computation
        continue;
      }

      auto dcQ = const_cast<ElementQ*>(p.Q + offset_q);
      auto dcK = const_cast<ElementK*>(p.K);
      auto dcV = const_cast<ElementV*>(p.V);
      auto ptrO = p.Oaccum + offset_o;
      auto ptrExp_sums = p.exp_sums + offset_exp_sums;
      auto ptrMax_logits = p.max_logits + offset_max_logits;

      auto layout_q = make_ordered_layout(shape_Q, Step<_1, _0, _2, _3>{});
      auto layout_k = make_ordered_layout(shape_K, Step<_2, _0, _1, _3>{});
      auto layout_v = make_ordered_layout(shape_V, Step<_0, _2, _1, _3>{});
      
      auto layout_o = make_ordered_layout(shape_O, Step<_1, _0, _2, _3, _4>{});
      auto layout_exp_sums = make_ordered_layout(shape_exp_sums, Step<_1, _0, _2, _3>{});
      auto layout_max_logits = make_ordered_layout(shape_max_logits, Step<_1, _0, _2, _3>{});
      auto layout_sink = make_ordered_layout(shape_sink, Step<_1, _0>{});

      Tensor Q = make_tensor(make_gmem_ptr(dcQ), layout_q);
      Tensor K = make_tensor(make_gmem_ptr(dcK), layout_k);
      Tensor V = make_tensor(make_gmem_ptr(dcV), layout_v);
      Tensor O = make_tensor(make_gmem_ptr(ptrO), layout_o);
      Tensor exp_sums = make_tensor(make_gmem_ptr(ptrExp_sums), layout_exp_sums);
      Tensor max_logits = make_tensor(make_gmem_ptr(ptrMax_logits), layout_max_logits);
      Tensor sinks = make_tensor(make_gmem_ptr(const_cast<ElementSink*>(p.sm_sink)), layout_sink);

      // O accumulator types
      FragA tArA;
      FragARow tA_max, tA_sum;

      // Main loop
      int l_coord = is_var_len ? 0 : idx_b;

      int start_blk = kv_split_offset;
      int end_blk = kv_split_offset + num_effective_kv_blocks;

      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);

      mainloop(Q(_,_,head,l_coord),
              K(_,_,head,l_coord),
              V(_,_,head,l_coord),
              tArA, tA_max, tA_sum,
              blk_qv, idx_b, start_blk, end_blk, k_blocks,
              thr_id, seq_len,
              full_tile_offset, discard_seq_coord);

      if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }

      // Epilogue
      CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
      if constexpr (Sink) {
        auto sinks_per_kv = sinks(head, _);
        epilogue(O(_,_,head,idx_kv_split,l_coord),
                  tArA, tA_max, tA_sum,
                  blk_qv, thr_id,
                  exp_sums(_,_,head,l_coord),
                  max_logits(_,_,head,l_coord),
                  idx_kv_split, head_group_q, sinks_per_kv);
      } else {
        epilogue(O(_,_,head,idx_kv_split,l_coord),
                  tArA, tA_max, tA_sum,
                  blk_qv, thr_id,
                  exp_sums(_,_,head,l_coord),
                  max_logits(_,_,head,l_coord),
                  idx_kv_split, head_group_q, sinks);
      }
    }
  }
};

template <class ProblemShape_,
          class TileScheduler_,
          class FMHAKernel_
>
class ReduceSplitK {
public:

  using ProblemShape = ProblemShape_;
  using VariableLength = cutlass::fmha::collective::VariableLength;
  static constexpr bool is_var_len = cutlass::fmha::collective::is_variable_length_v<typename ProblemShape::SeqLenType>;
  using TileScheduler = TileScheduler_;
  static_assert(is_same_v<TileScheduler, cutlass::fmha::kernel::XeReduceSplitKTileScheduler>,
                "ReduceSplitK kernel requires XeReduceSplitKTileScheduler");
  using TileSchedulerParams = typename TileScheduler::Params;

  using ElementO = typename FMHAKernel_::ElementO;
  using StrideO = typename FMHAKernel_::StrideO;
  using TileShapeO = typename FMHAKernel_::TileShapeO;
  using TileShapeQK = typename FMHAKernel_::TileShapeQK;

  using ElementLSE = typename FMHAKernel_::ElementLSE;

  using SGPerWG = typename FMHAKernel_::SGPerWG;

  // num values (head_dim) processed by each thread
  constexpr static int num_vals_per_thread = int(get<1>(TileShapeO{}) / (SGPerWG::value * intel::sg_size));

  //
  // Types
  //

  struct KernelArguments {
    ProblemShape shape;
    // outputs:
    ElementO *O;
    StrideO dO;
    // below are inputs
    // TODO: whether same dtype as output or accum?
    const ElementO *Oaccum;
    StrideO dOaccum;
    const ElementLSE *exp_sums;
    StrideO dExp_sums;
    const ElementLSE *max_logits;
    StrideO dMax_logits;
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    KernelHardwareInfo hw_info{};
    int num_kv_splits = -1; // no split by default
  };

  /// Params structure
  struct Params {
    KernelParams kernel;
    TileSchedulerParams scheduler;
  };

  struct SharedStorage {
    cutlass::Array<ElementLSE, FMHAKernel_::max_num_kv_splits> max_logits_slm_array;
    cutlass::Array<ElementLSE, FMHAKernel_::max_num_kv_splits> exp_sums_slm_array;
    cutlass::Array<ElementLSE, FMHAKernel_::max_num_kv_splits> rescaled_exp_sums_array;
  };

  static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0)
                                                                     : sizeof(SharedStorage);

public:

  static Params to_underlying_arguments(Arguments const &args, void *workspace) {
    return {args.kernel,
            TileScheduler::to_underlying_arguments(args.kernel.shape, args.hw_info, TileShapeO{}, args.num_kv_splits)};
  }

  static bool can_implement(Arguments const &args) {
    // only support decode
    if (!is_var_len && args.kernel.shape.seq_len_qo > 1) {
      return false;
    }

    if (args.num_kv_splits > FMHAKernel_::max_num_kv_splits) {
      return false;
    }
    return true;
  }

  static int get_workspace_size(Arguments const &args) { return 0; }

  static cutlass::Status initialize_workspace(Arguments const &args, void *workspace = nullptr,
                                              cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const &params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(params.scheduler);
  }

  static dim3 get_block_shape() { return dim3(SGPerWG::value * intel::sg_size, 1, 1); }

  CUTLASS_DEVICE
  Shape<int, int> get_sequence_length_shape(ProblemShape const& problem_shape, int const& batch) {
    if constexpr (is_var_len) {
      return cutlass::fmha::collective::apply_variable_length(Shape<VariableLength, VariableLength>{problem_shape.seq_len_qo, problem_shape.seq_len_kv}, batch);
    } else {
      return Shape<int, int>{problem_shape.seq_len_qo, problem_shape.seq_len_kv};
    }
  }

  /// Perform a reduction
  CUTLASS_DEVICE
  void operator()(Params const &params, char *smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage *>(smem_buf);

    auto &p = params.kernel;
    ProblemShape const& s = p.shape;

    int thr_id = int(ThreadIdxX());
    int sub_group_id = thr_id / intel::sg_size;
    int tid_in_sg = thr_id % intel::sg_size;

    TileScheduler tile_scheduler{params.scheduler};
    auto num_kv_splits = params.scheduler.num_kv_splits;

    auto batch_dim = is_var_len ? 1 : s.batch;
    auto num_heads_q = s.num_heads_q;
    auto head_size_vo = s.head_size_vo;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [seq_idx, head_q, idx_b] = tile_scheduler.get_block_coord();

      auto sequence_length_shape = get_sequence_length_shape(s, idx_b);
      auto [seq_len_qo, seq_len_kv] = sequence_length_shape;

      // when varlen enabled, use largest seq_len_qo to decide work group num
      if (seq_idx >= seq_len_qo) continue;

      const int k_blocks = cute::ceil_div(seq_len_kv, get<1>(TileShapeQK{}));
      int num_blocks_per_split = cute::ceil_div(k_blocks, num_kv_splits);

      int offset_o = 0, offset_o_accum = 0;
      int offset_exp_sums = 0, offset_max_logits = 0;

      if constexpr (is_var_len) {
        auto qo_cumulative = s.seq_len_qo.cumulative_length;

        offset_o_accum = s.num_heads_q * s.head_size_vo * num_kv_splits * qo_cumulative[idx_b];
        offset_exp_sums = s.num_heads_q * num_kv_splits * qo_cumulative[idx_b];
        offset_max_logits = s.num_heads_q * num_kv_splits * qo_cumulative[idx_b];

        offset_o = s.num_heads_q * s.head_size_vo * qo_cumulative[idx_b];
      }

      auto shape_O = make_shape(seq_len_qo, head_size_vo, num_heads_q,  batch_dim);
      auto shape_Oaccum = is_var_len ? make_shape(seq_len_qo, head_size_vo, num_heads_q * num_kv_splits, batch_dim) : make_shape(seq_len_qo, head_size_vo, num_heads_q * num_kv_splits, batch_dim);

      auto shape_exp_sums = make_shape(seq_len_qo, num_kv_splits, num_heads_q, batch_dim);
      auto shape_max_logits = make_shape(seq_len_qo, num_kv_splits, num_heads_q, batch_dim);

      auto dcOaccum = const_cast<ElementO*>(p.Oaccum + offset_o_accum);
      auto ptrExp_sums = const_cast<ElementLSE*>(p.exp_sums + offset_exp_sums);
      auto ptrMax_logits = const_cast<ElementLSE*>(p.max_logits + offset_max_logits);
      auto ptrO = p.O + offset_o;

      auto stride_o = is_var_len ? cutlass::make_cute_packed_stride(StrideO{}, shape_O) : p.dO;
      auto stride_o_accum = is_var_len ? cutlass::make_cute_packed_stride(StrideO{}, shape_Oaccum) : p.dOaccum;
      auto stride_exp_sums = is_var_len ? cutlass::make_cute_packed_stride(StrideO{}, shape_exp_sums) : p.dExp_sums;
      auto stride_max_logits = is_var_len ? cutlass::make_cute_packed_stride(StrideO{}, shape_max_logits) : p.dMax_logits;

      Tensor Oaccum = make_tensor(make_gmem_ptr(dcOaccum), make_layout(shape_Oaccum, stride_o_accum));
      Tensor O = make_tensor(make_gmem_ptr(ptrO), make_layout(shape_O, stride_o));

      Tensor exp_sums = make_tensor(make_gmem_ptr(ptrExp_sums), make_layout(shape_exp_sums, stride_exp_sums));
      Tensor max_logits = make_tensor(make_gmem_ptr(ptrMax_logits), make_layout(shape_max_logits, stride_max_logits));

      int l_coord = is_var_len ? 0 : idx_b;

      // Step 1: reduce max logits across different partitions
      // store into SLM for later use

      ElementLSE global_max_logits = cutlass::platform::numeric_limits<ElementLSE>::lowest();
      ElementLSE global_exp_sums {0};
      // only first subgroup participates
      if (thr_id < num_kv_splits && thr_id * num_blocks_per_split < k_blocks)  {
        ElementLSE cur_max_logit = max_logits(seq_idx, thr_id, head_q, l_coord);
        global_max_logits = sycl::max(global_max_logits, cur_max_logit);
        shared_storage.max_logits_slm_array[thr_id] = cur_max_logit;

        ElementLSE cur_exp_sum = exp_sums(seq_idx, thr_id, head_q, l_coord);
        shared_storage.exp_sums_slm_array[thr_id] = cur_exp_sum;
      }

      // barrier for SLM writes finished
      sycl::group_barrier(get_work_group<3>());

      // reduce across wg
      global_max_logits = reduce_over_group(get_work_group<1>(), global_max_logits, sycl::maximum<>());

      // broadcast to all other threads
      global_max_logits = sycl::group_broadcast(get_work_group<1>(), global_max_logits, 0);
      
      // step 2: rescale Oaccum and write back to O
      if (thr_id < num_kv_splits && thr_id * num_blocks_per_split < k_blocks)  {
        
        ElementLSE local_max_logit = shared_storage.max_logits_slm_array[thr_id];
        ElementLSE local_exp_sum = shared_storage.exp_sums_slm_array[thr_id];

        ElementLSE rescale = sycl::native::exp2(local_max_logit - global_max_logits);
        ElementLSE rescaled_exp_sum = local_exp_sum * rescale;
        shared_storage.rescaled_exp_sums_array[thr_id] = rescaled_exp_sum;

        global_exp_sums += rescaled_exp_sum;
      }
      sycl::group_barrier(get_work_group<3>());
      global_exp_sums = reduce_over_group(get_work_group<1>(), global_exp_sums, sycl::plus<>());
      global_exp_sums = sycl::group_broadcast(get_work_group<1>(), global_exp_sums, 0);
      
      ElementLSE inv_global_exp_sums = 1. / global_exp_sums;
      for (int idx = thr_id; idx < s.head_size_vo; idx += SGPerWG::value * intel::sg_size) {
        ElementLSE acc = 0;
        for (int i = 0; i < num_kv_splits; ++i) {
          if (i * num_blocks_per_split >= k_blocks) {
            break;
          }
          // assume seq_len_q == 1
          ElementLSE adjusted_o_accum = static_cast<ElementLSE>(Oaccum(seq_idx, idx, i * num_heads_q + head_q, l_coord)) * 
                                      shared_storage.rescaled_exp_sums_array[i];
          acc += adjusted_o_accum;
        }

        acc *= inv_global_exp_sums;
        O(seq_idx, idx, head_q, l_coord) = static_cast<ElementO>(acc);
      }
    }
  }
};

}  // namespace cutlass::fmha::kernel
