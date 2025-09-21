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
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"

#include "./collective/chunk_prefill_mma.hpp"
namespace cutlass::flash_attention::kernel {

template <class ProblemShape_, class CollectiveMainloop_,
          class CollectiveSoftmaxEpilogue_, class CollectiveEpilogue_,
          class TileScheduler_ = void>
class FMHAPrefillChunk;
///////////////////////////////////////////////////////////////////////////////
template <class ProblemShape_, class CollectiveMainloop_,
          class CollectiveSoftmaxEpilogue_, class CollectiveEpilogue_,
          class TileScheduler_>
class FMHAPrefillChunk {
 public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  // ProblemShape: <batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv,
  // head_size_qk, head_size_vo>
  static_assert(
      rank(ProblemShape{}) == 8,
      "ProblemShape{} should be <batch, num_heads_q, num_heads_kv, seq_len_qo, "
      "seq_len_kv, seq_len_kv_cache, head_size_qk, head_size_vo>");
  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using TileShapePV = typename CollectiveMainloop::TileShapePV;
  using TiledMmaQK = typename CollectiveMainloop::TiledMmaQK;
  using TiledMmaPV = typename CollectiveMainloop::TiledMmaPV;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using ElementQ = typename CollectiveMainloop::ElementQ;
  using StrideQ = typename CollectiveMainloop::StrideQ;
  using ElementK = typename CollectiveMainloop::ElementK;
  using StrideK = typename CollectiveMainloop::StrideK;
  using ElementV = typename CollectiveMainloop::ElementV;
  using StrideV = typename CollectiveMainloop::StrideV;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using CollectiveSoftmaxEpilogue = CollectiveSoftmaxEpilogue_;
  using SoftmaxArguments = typename CollectiveSoftmaxEpilogue::Arguments;
  using SoftmaxParams = typename CollectiveSoftmaxEpilogue::Params;

  static_assert(cute::is_void_v<TileScheduler_> or
                    cute::is_same_v<TileScheduler_, PersistentScheduler> or
                    cute::is_same_v<TileScheduler_, IndividualScheduler>,
                "Unsupported TileScheduler for Intel Xe.");
  using TileSchedulerTag = TileScheduler_;
  using TileScheduler =
      typename detail::TileSchedulerSelector<TileScheduler_,
                                             ArchTag>::Scheduler;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementO = typename CollectiveEpilogue::ElementO;
  using StrideO = typename CollectiveEpilogue::StrideO;
  using ElementLSE = typename CollectiveEpilogue::ElementLSE;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  using TileShapeOutput = typename CollectiveEpilogue::TileShapeOutput;
  using TiledMmaOutput = typename CollectiveEpilogue::TiledMmaOutput;

  static_assert(
      cute::is_same_v<ElementAccumulator,
                      typename CollectiveEpilogue::ElementAccumulator>,
      "Mainloop and epilogue do not agree on accumulator value type.");
  // MSVC requires the cast to fix a warning-as-error.
  static constexpr int SharedStorageSize = 0;

  static constexpr bool CausalMask = CollectiveMainloop::CausalMask;
  static constexpr bool LocalMask = CollectiveMainloop::LocalMask;

  static_assert(!(CausalMask && LocalMask), "Cannot be both causal and local");
  static constexpr bool PagedKV = CollectiveMainloop::PagedKV;

  static constexpr int SubgroupSize =
      CollectiveMainloop::SubgroupSize;  // sub_group size
  static constexpr uint32_t MaxThreadsPerBlock =
      CollectiveMainloop::MaxThreadsPerBlock;
  using MmaAtomShape = typename CollectiveMainloop::MmaAtomShape;  // 8,16,16

  static constexpr int QK_BLK_M = CollectiveMainloop::QK_BLK_M;
  static constexpr int QK_BLK_N = CollectiveMainloop::QK_BLK_N;
  static constexpr int QK_BLK_K = CollectiveMainloop::QK_BLK_K;

  static constexpr int QK_ATOM_N = CollectiveMainloop::QK_ATOM_N;
  static constexpr int QK_ATOM_K = CollectiveMainloop::QK_ATOM_K;

  static constexpr int QK_SG_M = CollectiveMainloop::QK_SG_M;

  static constexpr int Epilogue_BLK_N = get<1>(TileShapeOutput{});
  static constexpr int Epilogue_BLK_K = get<2>(TileShapeOutput{});

  static constexpr int PV_ATOM_M = CollectiveMainloop::PV_ATOM_M;
  static constexpr int PV_ATOM_N = CollectiveMainloop::PV_ATOM_N;
  static constexpr int PV_ATOM_K = CollectiveMainloop::PV_ATOM_K;

  static constexpr auto Num_SGs = PV_ATOM_N * PV_ATOM_M * PV_ATOM_K;
  static constexpr int Vec = CollectiveMainloop::Vec;
  static constexpr int FragsM = CollectiveMainloop::FragsM;
  // The FragsN here used for Creation of S matrix so we use the FragsN for S
  // shape
  static constexpr int FragsN = CollectiveMainloop::FragsNS;

  static constexpr int VSlicer =
      get<1>(TileShapeOutput{}) /
      (get<1>(TileShapePV{}) * PV_ATOM_N);  // ceil_div(FragsNOut,FragsNS);
  using AccumeShape = decltype(make_shape(
      Int<Vec>{}, Int<FragsM>{}, get<1>(TileShapePV{}) / get<1>(MmaAtomShape()),
      Int<VSlicer>{}));

  static constexpr bool is_var_len = CollectiveMainloop::is_var_len;
  // Kernel level shared memory storage
  struct SharedStorage {
    using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;
    EpilogueTensorStorage epilogue;
  };

  // Device side arguments
  struct Arguments {
    gemm::GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    SoftmaxArguments softmax{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
  };

  // Kernel entry point API
  struct Params {
    gemm::GemmUniversalMode mode;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    SoftmaxParams softmax;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the
  // aliased type.
  static Params to_underlying_arguments(Arguments const& args,
                                        void* workspace) {
    (void)workspace;
    return {args.mode,
            args.problem_shape,
            CollectiveMainloop::to_underlying_arguments(
                args.problem_shape, args.mainloop, workspace),
            CollectiveSoftmaxEpilogue::to_underlying_arguments(args.softmax),
            CollectiveEpilogue::to_underlying_arguments(
                args.problem_shape, args.epilogue, workspace),
            TileScheduler::to_underlying_arguments(
                args.problem_shape, args.hw_info, TileShapeOutput{})};
  }

  static bool can_implement(Arguments const& args) {
    bool mode_implementable = args.mode == gemm::GemmUniversalMode::kGemm or
                              (args.mode == gemm::GemmUniversalMode::kBatched &&
                               rank(ProblemShape{}) == 4);
    return mode_implementable;
  }

  static int get_workspace_size(Arguments const& args) { return 0; }

  static cutlass::Status initialize_workspace(
      Arguments const& args, void* workspace = nullptr,
      cudaStream_t stream = nullptr, CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::template get_grid_shape<Num_SGs>(params.scheduler);
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTLASS_DEVICE
  Shape<int, int, int> get_sequence_length_shape(
      ProblemShape const& problem_shape, int const& batch) {
    if constexpr (is_var_len) {
      return cutlass::fmha::collective::apply_variable_length(
          select<3, 4, 5>(problem_shape), batch);
    } else {
      return select<3, 4, 5>(problem_shape);
    }
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    // Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShapeQK>::value);
    CUTE_STATIC_ASSERT(is_static<TileShapePV>::value);
    // Separate out problem shape for convenience

    // "ProblemShape{} should be <batch, num_heads_q, num_heads_kv, seq_len_qo,
    // seq_len_kv, head_size_qk, head_size_vo>");
    auto batch = get<0>(params.problem_shape);
    auto num_heads_q = get<1>(params.problem_shape);
    auto num_heads_kv = get<2>(params.problem_shape);
    auto q_group_size = num_heads_q / num_heads_kv;

    // auto &seq_len_qo = get<3>(params.problem_shape); // seq_len_qo

    auto& head_size_qk = get<6>(params.problem_shape);
    auto& head_size_vo = get<7>(params.problem_shape);
    // Preconditions
    static_assert(cute::rank(StrideQ{}) == 3,
                  "StrideQ must be rank-3: [seq_len_qo, head_size_qk, batch * "
                  "num_heads_q].");
    static_assert(cute::rank(StrideK{}) == 3,
                  "StrideK must be rank-3: [head_size_qk, seq_len_kv, batch * "
                  "num_heads_kv].");
    static_assert(cute::rank(StrideV{}) == 3,
                  "StrideV must be rank-3: [seq_len_kv, head_size_vo, batch * "
                  "num_heads_kv].");

    int thread_idx = int(ThreadIdxX());
    auto sub_group_id = get_sub_group_id();
    auto local_id = get_sub_group_local_id();

    TileScheduler tile_scheduler{params.scheduler};
    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto blk_coord =
          tile_scheduler
              .get_block_coord();  // head_size_blk_idx, seq_len_blk_idx,
                                   // batch_blk_idx, num_heads_blk_idx

      auto blk_m_coord = get<0>(blk_coord);   // seq_len_blk_idx
      auto blk_n_coord = 0;                   // nums_head_blk_idx
      auto q_head_coord = get<1>(blk_coord);  // q_head_idx

      auto batch_coord = get<2>(blk_coord);  // batch_blk_idx

      // For variable sequence length case, batch is considered to be 1 (same
      // as group gemm). For fixed sequence length case, the l_coord is the
      // weighted sum of both batch_coord and num_heads_coord. Flash Attention
      // implementation combines batch and num_heads to calculate the total
      // batch_size. iff is_var_len: batch_size = num_heads (as each batch
      // would have it's own seq_len_qo and seq_len_kv) iff !is_var_len:
      // batch_size = batch * num_heads
      // auto blk_l_coord = q_group_coord;

      // Get problem shape for the current batch_blk_idx. For variable
      // sequence length, it loads the sequence length from Global memory for
      // the given batch_blk_idx and returns the appropriate problem_shape.
      // For fixed sequence length, sequence_length_shape == select<3, 4,
      // 5>(params.problem_shape). sequence_length_shape = [batch,
      // num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, seq_len_kv_cache,
      // head_size_qk, head_size_vo]
      auto sequence_length_shape =
          get_sequence_length_shape(params.problem_shape, batch_coord);
      get<1>(sequence_length_shape) = 0;

      auto [seq_len_qo, seq_len_kv, seq_len_kv_cache] = sequence_length_shape;
      // int seq_len_kv_total = seq_len_kv_cache + seq_len_kv;
      // For variable sequence length case, batch is considered to be 1 (same
      // as group gemm). For fixed sequence length case, the l_coord is the
      // weighted sum of both batch_coord and num_heads_coord. Flash Attention
      // implementation combines batch and num_heads to calculate the total
      // batch_size. iff is_var_len: batch_size = num_heads (as each batch
      // would have it's own seq_len_qo and seq_len_kv) iff !is_var_len:
      // batch_size = batch * num_heads

      // Calculate the seq_len_idx (blk_m_coord * get<0>(TileShapeOutput{}))
      // and check if it is still within bounds of the actual seq_len_qo
      // (get<0>(sequence_length_shape)).
      if (blk_m_coord * get<0>(TileShapeOutput{}) >= seq_len_qo) {
        continue;
      }

      const int seq_len = seq_len_kv;

      const int kv_splits_new = cute::ceil_div(seq_len, QK_BLK_N);
      const int kv_splits_cache = cute::ceil_div(seq_len_kv_cache, QK_BLK_N);
      const int kv_splits = kv_splits_cache + kv_splits_new;

      int tiles_per_page = params.mainloop.page_size / QK_BLK_N;

      Tensor mQ_mkl = cute::get_xe_tensor(
          make_shape(seq_len_qo, head_size_qk, 1));  //(m,k,l)

      Tensor mK_nkl = cute::get_xe_tensor(
          make_shape(seq_len_kv, head_size_qk, 1));  //(n,k,l)
      Tensor mV_nkl = cute::get_xe_tensor(
          make_shape(head_size_vo, seq_len_kv, 1));  //(n,k,l)
      Tensor mK_cache_nkl = cute::get_xe_tensor(
          make_shape(seq_len_kv_cache, head_size_qk, 1));  // (n_cache,k,l)
      Tensor mV_cache_nkl = cute::get_xe_tensor(
          make_shape(head_size_vo, seq_len_kv_cache, 1));  // (n_cache,k,l)

      Tensor mQ_mk = mQ_mkl(_, _, 0);

      Tensor mK_nk = mK_nkl(_, _, 0);  // (n,k)
      Tensor mV_nk = mV_nkl(_, _, 0);

      Tensor mK_cache_nk = mK_cache_nkl(_, _, 0);  // (n_cache, k)
      Tensor mV_cache_nk = mV_cache_nkl(_, _, 0);  // (n_cache, k)

      auto gQ = local_tile(mQ_mk, TileShapeQK{}, make_coord(blk_m_coord, _, _),
                           Step<_1, X, _1>{});
      auto gK = local_tile(mK_nk, TileShapeQK{}, make_coord(_, _, _),
                           Step<X, _1, _1>{});

      auto gV = local_tile(mV_nk, TileShapeOutput{},
                           make_coord(_, blk_n_coord, _), Step<X, _1, _1>{});
      auto gK_cache = local_tile(mK_cache_nk, TileShapeQK{},
                                 make_coord(_, _, _), Step<X, _1, _1>{});
      auto gV_cache =
          local_tile(mV_cache_nk, TileShapeOutput{},
                     make_coord(_, blk_n_coord, _), Step<X, _1, _1>{});

      auto mainloop_params = CollectiveMainloop::get_updated_copies(
          params.mainloop, params.problem_shape, sequence_length_shape,
          batch_coord, q_head_coord);

      // we limit the horizontal size to two subgroup, the empirical results
      // show that reading the two cacheline side by side in gives better
      // performance and anything after that does not have an effect on
      // performance. // (64 here for float b float when possible and loop over
      // to cover all the data needed)
      auto tiled_prefetch_q = cute::prefetch_selector<
          Shape<Int<QK_BLK_M>, Int<cute::max(cute::gcd(QK_BLK_K, 64), 32)>>,
          Num_SGs>(mainloop_params.gmem_tiled_copy_q);
      auto tiled_prefetch_k = cute::prefetch_selector<
          Shape<Int<QK_BLK_N>, Int<cute::max(cute::gcd(QK_BLK_K, 64), 32)>>,
          Num_SGs>(mainloop_params.gmem_tiled_copy_k);
      auto tiled_prefetch_v = cute::prefetch_selector<
          Shape<Int<cute::max(cute::gcd(Epilogue_BLK_N, 64), 32)>,
                Int<Epilogue_BLK_K>>,
          Num_SGs>(mainloop_params.gmem_tiled_copy_v);
      auto tiled_prefetch_k_cache = cute::prefetch_selector<
          Shape<Int<QK_BLK_N>, Int<cute::max(cute::gcd(QK_BLK_K, 64), 32)>>,
          Num_SGs>(mainloop_params.gmem_tiled_copy_k_cache);
      auto tiled_prefetch_v_cache = cute::prefetch_selector<
          Shape<Int<cute::max(cute::gcd(Epilogue_BLK_N, 64), 32)>,
                Int<Epilogue_BLK_K>>,
          Num_SGs>(mainloop_params.gmem_tiled_copy_v_cache);
      auto thr_prefetch_Q = tiled_prefetch_q.get_slice(thread_idx);
      auto thr_prefetch_K = tiled_prefetch_k.get_slice(thread_idx);
      auto thr_prefetch_V = tiled_prefetch_v.get_slice(thread_idx);
      auto pQgQ = thr_prefetch_Q.partition_S(gQ);
      auto pKgK = thr_prefetch_K.partition_S(gK);
      auto pVgV = thr_prefetch_V.partition_S(gV);
      // assuming the copy function is the same otherwise this need to have its
      // own tile_prefetch
      auto pKgK_cache = thr_prefetch_K.partition_S(gK_cache);
      auto pVgV_cache = thr_prefetch_V.partition_S(gV_cache);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<3>(pQgQ); i++) {
        prefetch(tiled_prefetch_q, pQgQ(_, _, _, i));
      }
      auto& prefetch_K =
          (seq_len_kv_cache == 0) ? tiled_prefetch_k : tiled_prefetch_k_cache;
      auto& pKgK1_ = (seq_len_kv_cache == 0) ? pKgK : pKgK_cache;

      int cached_nblock = 0;
      if constexpr (PagedKV) {
        if (seq_len_kv_cache != 0) {
          // int curr_batch_pages =
          //     is_var_len
          //         ? mainloop_params.num_pages_per_seq[batch_coord + 1] -
          //               mainloop_params.num_pages_per_seq[batch_coord]
          //         : ceil_div(seq_len_kv_cache, mainloop_params.page_size);
          // int batch_offset =
          //     is_var_len ? mainloop_params.num_pages_per_seq[batch_coord]
          //                : batch_coord * curr_batch_pages;
          int batch_offset = batch_coord * mainloop_params.max_pages_per_seq;
          cached_nblock =
              mainloop_params
                  .ptr_page_table[batch_offset  // page table for this batch
          ] * tiles_per_page;  // base block idx of physical page
        }
      }
      // The headsize for both cached and non-cached version is the same
      for (int j = 0; j < size<4>(pKgK1_); j++) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = cached_nblock; i < cached_nblock + DispatchPolicy::Stages;
             i++) {
          prefetch(prefetch_K, pKgK1_(_, _, _, i, j));
        }
      }

      // Allocate the tiled_mma and the accumulators for the (M,N)
      // workgroup_shape
      Tensor out_reg = make_tensor<ElementAccumulator>(AccumeShape{});

      // There are 16 workitem and 16 max per subgroup, each worktime contain 1
      // max and cumulatively, they calculate the max per subgroup
      ElementAccumulator max_reg{-INFINITY};
      // The sum reg each contains a 2d tesnor for 8 x 2 This is number of
      // sequence length process per subgroup
      Tensor sum_reg =
          make_tensor<ElementAccumulator>(Shape<Int<Vec>, Int<FragsM>>{});

      clear(sum_reg);
      clear(out_reg);
      // Perform the collective scoped MMA
      CollectiveMainloop collective_mma;
      // when causal mask is true. It is not possible to set the scope
      // of the barrier to workgroup level as the number n block is
      // different for each subgroup due to triangular nature of causal based
      // operation
      // static constexpr int barrier_scope = CausalMask ? 3 : 2;
      static constexpr int barrier_scope = 2;

      int q_start_coord = blk_m_coord * QK_BLK_M;
      int q_end_coord = cute::min(q_start_coord + QK_BLK_M, seq_len_qo);
      int seq_diff = seq_len_kv_cache - seq_len_qo;

      const int seq_coord = cute::min(
          seq_len_qo,
          (blk_m_coord * QK_BLK_M + (sub_group_id / PV_ATOM_N) * QK_SG_M) %
              seq_len_qo);

      CUTLASS_PRAGMA_UNROLL
      for (int split = 0; split < kv_splits; split++) {
        barrier_arrive(barrier_scope);

        int kv_start_coord = split * QK_BLK_N;

        if constexpr (CausalMask) {
          if (kv_start_coord >= q_end_coord + seq_diff) break;
        }

        bool is_KV_cache = split < kv_splits_cache;
        // 1) Load KV (performed inside mmaQK)
        auto gK_ = is_KV_cache ? gK_cache(_, _, cached_nblock, _)
                               : gK(_, _, split - kv_splits_cache, _);
        auto gV_ = is_KV_cache ? gV_cache(_, _, cached_nblock)
                               : gV(_, _, split - kv_splits_cache);
        // 2) Create Tensor S
        Tensor tSr = make_tensor<ElementAccumulator>(
            Shape<Int<Vec>, Int<FragsM>, Int<FragsN>>{});
        clear(tSr);
        // 3) Perform GEMM S = Q*K
        // Then modify layout to LayoutQ = ((seq_leq_q, group_head_q),
        // head_size_qk, batch* num_heads_q / group_head_q), which can be merged
        // into one gemm for (int i = 0; i < q_group_size; ++i) {
        collective_mma.mmaQK(tSr, gQ, gK_, tSr,
                             ceil_div(head_size_qk, QK_BLK_K), mainloop_params,
                             is_KV_cache);

        // mask padding
        int col_start = local_id + kv_start_coord;
        int col_end = col_start + (FragsN - 1) * get<1>(MmaAtomShape());
        if (col_end >= seq_len_kv_cache) {
          int col_idx = col_start;
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < FragsN;
               n++, col_idx += get<1>(MmaAtomShape())) {  // 4
            if (col_idx >= seq_len_kv_cache) {
              CUTLASS_PRAGMA_UNROLL
              for (int m = 0; m < FragsM; m++) {  // 2
                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < Vec; row++) {  // 8
                  tSr(row, m, n) = ElementAccumulator{-INFINITY};
                }
              }
            }
          }
        }

        if constexpr (CausalMask) {
          int row_start = q_start_coord + sub_group_id * QK_SG_M;
          if (row_start + seq_diff < col_end) {
            int col_idx = col_start;
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < FragsN;
                 n++, col_idx += get<1>(MmaAtomShape())) {  // 4
              if (col_idx > row_start + seq_diff) {
                CUTLASS_PRAGMA_UNROLL
                for (int m = 0; m < FragsM; m++) {  // 2
                  CUTLASS_PRAGMA_UNROLL
                  for (int row = 0; row < Vec; row++) {  // 8
                    int row_idx = row_start + m * Vec + row;
                    if (row_idx + seq_diff < col_idx)
                      tSr(row, m, n) = ElementAccumulator{-INFINITY};
                  }
                }
              }
            }
          }
        }

        if constexpr (LocalMask) {
          // mask the elements of each tile where j - left > i || j + right < i
          const int item_id = thread_idx % SubgroupSize;
          int col_idx = item_id;
          if (split < kv_splits_cache) {
            col_idx += split * cute::min(QK_BLK_N, seq_len_kv_cache);
          } else {
            col_idx += seq_len_kv_cache + (split - kv_splits_cache) *
                                              cute::min(QK_BLK_N, seq_len_kv);
          }

          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < FragsN;
               n++, col_idx += get<1>(MmaAtomShape())) {  // 4
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < FragsM; m++) {  // 2
              int row_idx = m * Vec + seq_coord;
              CUTLASS_PRAGMA_UNROLL
              for (int row = 0; row < Vec; row++) {  // 8
                bool left_mask =
                    col_idx < cute::max(0, row + row_idx + seq_len_kv_cache -
                                               mainloop_params.window_left);
                bool right_mask =
                    col_idx > cute::min(seq_len_kv_cache + seq_len_kv,
                                        row + row_idx + seq_len_kv_cache +
                                            mainloop_params.window_right);
                if (left_mask || right_mask) {
                  tSr(row, m, n) = ElementAccumulator{-INFINITY};
                }
              }
            }
          }
        }

        auto& tiled_prefetch_v_ =
            is_KV_cache ? tiled_prefetch_v_cache : tiled_prefetch_v;
        auto& pVgV_ = is_KV_cache ? pVgV_cache : pVgV;
        int v_prefetch_idx = is_KV_cache ? PagedKV ? cached_nblock : split
                                         : split - kv_splits_cache;
        for (int i = 0; i < size<1>(pVgV_); i++) {
          prefetch(tiled_prefetch_v_, pVgV_(_, i, _, v_prefetch_idx));
        }
        int next_cached_nblock = split + 1;
        bool is_next_KV_cache = next_cached_nblock < kv_splits_cache;
        if constexpr (PagedKV) {
          if (is_next_KV_cache) {
            // int curr_batch_pages =
            //     is_var_len
            //         ? mainloop_params.num_pages_per_seq[batch_coord + 1] -
            //               mainloop_params.num_pages_per_seq[batch_coord]
            //         : ceil_div(seq_len_kv_cache, mainloop_params.page_size);
            int curr_batch_pages = mainloop_params.max_pages_per_seq;
            int next_page_logical_idx =
                next_cached_nblock * QK_BLK_N / params.mainloop.page_size;
            // int batch_offset =
            //     is_var_len ? mainloop_params.num_pages_per_seq[batch_coord]
            //                : batch_coord * curr_batch_pages;
            int batch_offset = batch_coord * mainloop_params.max_pages_per_seq;
            bool valid_page = next_page_logical_idx < curr_batch_pages;
            // get physical page idx from page table
            if (valid_page) {
              next_cached_nblock =
                  params.mainloop.ptr_page_table
                          [batch_offset +         // page table for this batch
                           next_page_logical_idx  // split (tile idx) to logical
                                                  // page idx
              ] * tiles_per_page +  // base block idx of physical page
                  next_cached_nblock % tiles_per_page;  // offset within page
            } else {
              next_cached_nblock =
                  curr_batch_pages *
                  tiles_per_page;  // push idx out of bounds to respect the
                                   // boundary between batches
            }
          }
        }

        // 4) Fused softmax
        CollectiveSoftmaxEpilogue softmax(params.softmax);
        softmax(split == 0, tSr, max_reg, sum_reg, out_reg);
        // if (cute::thread(0, 0)) {
        //   ElementAccumulator temp_reg{-INFINITY};
        //   temp_reg = temp_reg - 0.1f;
        //   print("test1: "); print(temp_reg); print("\n");
        //   temp_reg = sycl::native::exp2(temp_reg);
        //   print("test2: "); print(temp_reg); print("\n");
        // }
        // 5) Perform GEMM O = S*V
        collective_mma.template mmaPV<VSlicer>(out_reg, tSr, gV_, out_reg,
                                               mainloop_params, is_KV_cache);
        // ... prefetch next tile ...
        // Prefetch the next Q tile
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<3>(pQgQ); i++) {
          prefetch(tiled_prefetch_q, pQgQ(_, _, _, i));
        }

        is_KV_cache = is_next_KV_cache;
        cached_nblock = next_cached_nblock;
        // Prefetch the next K tile
        // there is no need to guard it with if statement as prefetch will
        // ignore out of bound reading

        bool sel_prefetch_k =
            (split + DispatchPolicy::Stages) < kv_splits_cache;
        auto& prefetch_k_selector =
            sel_prefetch_k ? tiled_prefetch_k_cache : tiled_prefetch_k;
        auto& pKgK_ = sel_prefetch_k ? pKgK_cache : pKgK;
        int k_prefetch_idx =
            sel_prefetch_k
                ? PagedKV ? cached_nblock : split + DispatchPolicy::Stages
                : split + DispatchPolicy::Stages - kv_splits_cache;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size<4>(pKgK_); j++) {
          prefetch(prefetch_k_selector, pKgK_(_, _, _, k_prefetch_idx, j));
        }
        barrier_wait(barrier_scope);
      }

      // Epilogue
      auto epilogue_params =
          CollectiveEpilogue::template get_updated_copies<is_var_len>(
              params.epilogue, params.problem_shape, sequence_length_shape,
              batch_coord, q_head_coord);
      CollectiveEpilogue epilogue{epilogue_params, shared_storage.epilogue};
      auto blk_coord_mnkl = make_coord(blk_m_coord, blk_n_coord, _, 0);
      epilogue(params.problem_shape, sequence_length_shape, blk_coord_mnkl,
               out_reg, max_reg, sum_reg);
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::flash_attention::kernel
