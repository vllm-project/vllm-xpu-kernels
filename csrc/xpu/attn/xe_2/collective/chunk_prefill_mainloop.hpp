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

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"

namespace cutlass::fmha {

template <int Stages>
class XeDefault {};  // Default FMHA mainloop, P in registers.

};  // namespace cutlass::fmha

namespace cutlass::fmha::collective {

static inline void sbarrier_wait() { asm volatile("sbarrier.wait\n"); }

static inline void sbarrier_signal() { asm volatile("sbarrier.signal\n"); }

static inline void gfence() { asm volatile("lsc_fence.ugm.none.group\n"); }

static inline void barrier() {
  asm volatile("lsc_fence.ugm.none.group\n");
  asm volatile("barrier\n");
}

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class DispatchPolicy_,
    bool CausalMask_,
    bool LocalMask_,
    bool PagedKV_,
    class TiledMMAQK_,  // Tiling for Q*K GEMM
    class TiledMMAPV_,  // Tiling for P*V GEMM
    int VTiles_,        // # of tiles in V dimension
    class TensorQ_,     // Global Q/K/V tensors
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_ = void,  // Optional TiledCopy for loading Q
    class TiledCopyK_ = void,  // Optional TiledCopy for loading K
    class TiledCopyV_ = void>  // Optional TiledCopy for loading V
struct FMHAFwdMainloop {
  static_assert(
      cutlass::detail::dependent_false<DispatchPolicy_>,
      "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int Stages,
    bool CausalMask_,
    bool LocalMask_,
    bool PagedKV_,
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    class TensorQ_,
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_,
    class TiledCopyK_,
    class TiledCopyV_>
struct FMHAFwdMainloop<
    XeDefault<Stages>,
    CausalMask_,
    LocalMask_,
    PagedKV_,
    TiledMMAQK_,
    TiledMMAPV_,
    VTiles_,
    TensorQ_,
    TensorK_,
    TensorV_,
    TiledCopyQ_,
    TiledCopyK_,
    TiledCopyV_> {
  //
  // Type Aliases
  //
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(
      take<1, 4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;
  using TensorV = TensorV_;

  using ElementQ = typename TensorQ::engine_type::value_type;
  using ElementK = typename TensorK::engine_type::value_type;

  using TensorQ2D =
      decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_, _), 0)));

  // 2D per-block types used for TiledCopy derivation (block 2D copy is 2D).
  using TensorK2D =
      decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _), 0)));
  using TensorV2D =
      decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _), 0)));

  // Operator tensor types: 3D for PagedKV (block_size, head_size, block_nums),
  // 2D for non-paged (seq_len, head_size).
  using TensorKND = conditional_t<
      PagedKV_,
      decltype(TensorK_{}(make_coord(_, _, 0, _))),
      TensorK2D>;
  using TensorVND = conditional_t<
      PagedKV_,
      decltype(TensorV_{}(make_coord(_, _, 0, _))),
      TensorV2D>;

  using TiledCopyQ = conditional_t<
      is_void_v<TiledCopyQ_>,
      decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})),
      TiledCopyQ_>;
  // TiledCopy always uses 2D types. For PagedKV, per-block copies are
  // constructed at runtime to rebase the base address per physical block.
  using TiledCopyK = conditional_t<
      is_void_v<TiledCopyK_>,
      decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})),
      TiledCopyK_>;
  using TiledCopyV = conditional_t<
      is_void_v<TiledCopyV_>,
      decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})),
      TiledCopyV_>;

  //
  // Accumulator types
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using FragSCol = decltype(reduce<0>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  using SingleFragA = FragC<TiledMMAPV>;  // (atom val,q',v')
  using FragA =
      expand_sg_fragment_t<SingleFragA, 1, VTiles>;  // (atom val,q',v',VV)
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  using ElementA = typename TiledMMAPV::ValTypeD;

  static constexpr bool Fp8KV =
      is_any_of_v<ElementK, float_e5m2_t, float_e4m3_t>;
  static constexpr bool CausalMask = CausalMask_;
  static constexpr bool LocalMask = LocalMask_;
  static constexpr bool PagedKV = PagedKV_;

  // User-facing arguments
  struct Arguments {
    ElementS const scale;
    void* const scale_k;
    void* const scale_v;

    // Paged KV Cache (BNHS layout: [block_nums, num_heads, block_size, head_size])
    const int* ptr_page_table;
    int page_size;
    int max_pages_per_seq;
    int total_seqlen_kv;
    int block_nums;
    // Local Mask
    int local_left, local_right;
  };

  // Kernel-facing parameters
  using Params = Arguments;

  // SLM data
  struct SharedStorage {};

  Params params;

  //
  // Methods
  //

  FMHAFwdMainloop(Params const& params_, SharedStorage&) : params(params_) {}

  static constexpr Params
  to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    constexpr double kLog2e = 1.4426950408889634074;  // log_2(e)
    ElementS val = args.scale * static_cast<ElementS>(kLog2e);
    return Params{
        val,
        args.scale_k,
        args.scale_v,
        args.ptr_page_table,
        args.page_size,
        args.max_pages_per_seq,
        args.total_seqlen_kv,
        args.block_nums,
        args.local_left,
        args.local_right};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  CUTLASS_DEVICE int get_paged_block(int block_logical, int idx_b) {
    int b_offset = idx_b * params.max_pages_per_seq;
    int page_local_idx = block_logical;
    if (page_local_idx >= params.max_pages_per_seq) {
      page_local_idx = params.max_pages_per_seq - 1;
    }
    return params.ptr_page_table[b_offset + page_local_idx];
  }

  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,  // (q,d)
      TensorKND const& K_ND,  // PagedKV: (block_size,d,block_nums); else (k,d)
      TensorVND const& V_ND,  // PagedKV: (d,block_size,block_nums); else (d,k)
      FragA& tArA,            // Output accumulator (q,v)
      FragARow& tA_max,       // Softmax row-wise max accumulator
      FragARow& tA_sum,       // Softmax row-wise sum accumulator
      QVCoord blk_qv,         // WG tile indices: (Q,V)
      int const& idx_b,       // WG tile indices: (B)
      int blk_k0,             // K block range: [K0,K1)
      int blk_k1,
      int blk_k1_causal,
      int thr_id,
      int seq_len,
      int full_tile_offset,
      int head = 0) {         // KV head index for BNHS addressing
    using namespace sycl::ext::oneapi::this_work_item;

    constexpr int tile_k = get<1>(TileShapeQK{});

    /* Create proxy coordinate tensors for Q and P (common) */
    Tensor cQ = make_identity_tensor(Q_2D.shape());
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));

    Tensor gQ = local_tile(
        cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});

    /* Common MMA and Q copy setup */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};
    TiledCopyQ copy_q{Q_2D};

    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    auto tQgQ = thr_copy_q.partition_S(gQ);
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));
    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);

    /* Masking scalars (common) */
    int lane_id = thr_id % intel::sg_size;
    constexpr int sg_tile_q = get<0>(TileShapeQK{}) / SGPerWG::value;
    int row_base = get<0>(blk_qv) * get<0>(TileShapeQK{}) +
                   (thr_id / intel::sg_size) * sg_tile_q;

    if constexpr (PagedKV) {
      // ── BNHS PagedKV path ──────────────────────────────────────
      // K_ND: (block_size, head_size, block_nums) — 3D after head selection
      // V_ND: (head_size, block_size, block_nums) — 3D after head selection
      // Blocks are non-contiguous; we use per-block 2D copies. For each
      // physical block, a TiledCopyK/V is constructed (just 1 HW payload
      // create instruction) from the 2D per-block slice.
      int tiles_per_block = params.page_size / tile_k;

      // 2D per-block reference (block 0) for identity tensor and partitions
      auto K_ref = K_ND(_, _, 0);  // 2D: (block_size, head_size)
      auto V_ref = V_ND(_, _, 0);  // 2D: (head_size, block_size)

      Tensor cK = make_identity_tensor(K_ref.shape());
      Tensor cV = make_identity_tensor(V_ref.shape());

      /* Tile K within one block: (block_size, head_size) →
         (tile_k, tile_d, Kw, D) where Kw = block_size/tile_k */
      Tensor gK = local_tile(
          cK, TileShapeQK{}, make_coord(_, _, _),
          Step<X, _1, _1>{});
      /* Single-level V tiling within one block:
         (head_size, block_size) → (v_tile, k_tile, V_all, K_within) */
      auto tile_v_paged = make_shape(
          get<1>(TileShapePV{}), get<2>(TileShapePV{}));
      Tensor gV = local_tile(cV, tile_v_paged, make_coord(_, _));

      TiledCopyK copy_k_ref{K_ref};
      TiledCopyV copy_v_ref{V_ref};
      auto thr_copy_k = copy_k_ref.get_slice(thr_id);
      auto thr_copy_v = copy_v_ref.get_slice(thr_id);

      auto tKgK = thr_copy_k.partition_S(gK);
      auto tVgV = thr_copy_v.partition_S(gV);

      auto tKrK = thr_copy_k.partition_sg_fragment_D(
          gK(_, _, 0, 0));
      auto tSrK = thr_mma_qk.partition_sg_fragment_B(
          gK(_, _, 0, 0));
      auto tVrV = thr_copy_v.partition_sg_fragment_D(
          gV(_, _, 0, 0));
      auto tArV = thr_mma_pv.partition_sg_fragment_B(
          gV(_, _, 0, 0));

      // V head tile offset for output head dimension
      int vv_base = get<1>(blk_qv) * VTiles;

      const int barrier_scope = 2;

      /* Q prefetch */
      {
        auto prefetch_q = make_block_2d_prefetch(copy_q);
        auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
        for (int D = 0; D < size<3>(pQgQ); D++) {
          prefetch(prefetch_q, pQgQ(_, _, _, D));
        }
      }

      /* K prefetch for first Stages blocks */
      for (int S = 0; S < Stages; S++) {
        int pre_K = blk_k0 + S;
        int pre_blk = pre_K / tiles_per_block;
        int pre_tw = pre_K % tiles_per_block;
        int pre_bp = get_paged_block(pre_blk, idx_b);
        auto prefetch_k_s = make_block_2d_prefetch(
            TiledCopyK{K_ND(_, _, pre_bp)});
        auto pKgK_s =
            prefetch_k_s.get_slice(thr_id).partition_S(gK);
        for (int D = 0; D < size<4>(pKgK_s); D++) {
          prefetch(prefetch_k_s,
                   pKgK_s(_, _, _, pre_tw, D));
        }
      }

      clear(tArA);
      fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
      clear(tA_sum);

      bool check_remainder_k = (seq_len % tile_k != 0);

      /* Main loop, blocked in k. */
      for (int K = blk_k0; K < blk_k1; K++) {

        bool need_causal = false;
        if constexpr (CausalMask) {
          need_causal = K >= blk_k1_causal;
        }

        int block_logical = K / tiles_per_block;
        int tile_within = K % tiles_per_block;
        int block_phys = get_paged_block(block_logical, idx_b);

        int pre_K = K + Stages;
        int pre_blk = pre_K / tiles_per_block;
        int pre_tw = pre_K % tiles_per_block;
        int pre_bp = get_paged_block(pre_blk, idx_b);

        barrier_arrive(barrier_scope);

        // Per-block 2D copies (rebase HW payload to this block)
        TiledCopyK copy_k_blk{K_ND(_, _, block_phys)};
        TiledCopyV copy_v_blk{V_ND(_, _, block_phys)};

      {
        // prefetch V for Gemm PV
        auto prefetch_v_blk = make_block_2d_prefetch(copy_v_blk);
        auto pVgV_blk =
            prefetch_v_blk.get_slice(thr_id).partition_S(gV);
        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          prefetch(prefetch_v_blk,
                    pVgV_blk(_, _, _, vv_base + VV, tile_within));
        }
      }

        /* GEMM 1: S = Q * K^T */
        clear(tSrS);
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < size<4>(tKgK); D++) {
          copy(copy_q, tQgQ(_, _, _, D), tQrQ);
          copy(copy_k_blk,
               tKgK(_, _, _, tile_within, D), tKrK);
          reorder(tQrQ, tSrQ);
          reorder(tKrK, tSrK);
          cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
        }

        /* Causal masking and k remainder masking */
        {
          bool need_k_rem = check_remainder_k && K == blk_k1 - 1;
          if (need_causal || need_k_rem) {
            constexpr int kTileK = tile_k;
            constexpr int n_reps = kTileK / intel::sg_size;
            constexpr int elems_per_n = tSrS.size() / n_reps;
            int k_base = K * kTileK;
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < n_reps; n++) {
              int col = k_base + n * intel::sg_size + lane_id;
              if (need_k_rem && col >= seq_len) {
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < elems_per_n; j++) {
                  tSrS(n * elems_per_n + j) = ElementS(-INFINITY);
                }
              } else if constexpr (CausalMask) {
                if (need_causal) {
                  int causal_bound =
                      col - full_tile_offset - row_base;
                  CUTLASS_PRAGMA_UNROLL
                  for (int j = 0; j < elems_per_n; j++) {
                    if (j < causal_bound) {
                      tSrS(n * elems_per_n + j) =
                          ElementS(-INFINITY);
                    }
                  }
                }
              }
            }
          }
        }
        /* Local masking */
        if constexpr (LocalMask) {
          Tensor cPgP =
              make_identity_tensor(make_shape(seq_len, seq_len));
          Tensor gP = local_tile(
              cPgP, take<0, 2>(TileShapeQK{}),
              make_coord(get<0>(blk_qv), K));
          auto cS_thread = thr_mma_qk.partition_C(gP);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            int row_idx = get<0>(cS_thread(i));
            int col_idx = get<1>(cS_thread(i)) - full_tile_offset;
            bool left_mask =
                col_idx < row_idx - params.local_left;
            bool right_mask =
                col_idx > row_idx + params.local_right;
            if (left_mask || right_mask) {
              tSrS(i) = ElementS(-INFINITY);
            }
          }
        }

       /* Apply softmax and scaling (tA rescaling fused into GEMM2 VTile loop) */
        auto rescale = softmax(K == blk_k0, tSrS, tA_max, tA_sum);
        reorder(tSrS, tArP);

        /* GEMM 2: A += P * V, split in v dimension */
        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          copy(copy_v_blk,
               tVgV(_, _, _, vv_base + VV, tile_within), tVrV);
          reorder(tVrV, tArV);
          if (K != blk_k0) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tArA.size() / VTiles; i++)
              tArA(_,_,_,VV)(i) *= broadcast<0>(rescale, tArA, i);
          }
          cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
        }

        /* K prefetch for next iteration */
        {
          auto prefetch_k_next = make_block_2d_prefetch(
              TiledCopyK{K_ND(_, _, pre_bp)});
          auto pKgK_next =
              prefetch_k_next.get_slice(thr_id).partition_S(gK);
          for (int D = 0; D < size<4>(pKgK_next); D++) {
            prefetch(prefetch_k_next,
                     pKgK_next(_, _, _, pre_tw, D));
          }
        }

        barrier_wait(barrier_scope);
      }
    } else {
      // ── Non-paged path: K/V are 2D (original) ─────────────────
      auto tile_shape_v = make_shape(
          get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

      Tensor cK = make_identity_tensor(K_ND.shape());
      Tensor cV = make_identity_tensor(V_ND.shape());

      Tensor gK = local_tile(
          cK, TileShapeQK{}, make_coord(_, _, _),
          Step<X, _1, _1>{});
      Tensor gV = local_tile(
          cV, tile_shape_v, make_coord(get<1>(blk_qv), _));
      Tensor gV_split = local_tile(
          gV, TileShapePV{}, make_coord(_, _, 0),
          Step<X, _1, _1>{});

      TiledCopyK copy_k{K_ND};
      TiledCopyV copy_v{V_ND};
      auto thr_copy_k = copy_k.get_slice(thr_id);
      auto thr_copy_v = copy_v.get_slice(thr_id);

      auto tKgK = thr_copy_k.partition_S(gK);
      auto tVgV = thr_copy_v.partition_S(gV_split);

      auto tKrK = thr_copy_k.partition_sg_fragment_D(
          gK(_, _, 0, 0));
      auto tSrK = thr_mma_qk.partition_sg_fragment_B(
          gK(_, _, 0, 0));
      auto tVrV = thr_copy_v.partition_sg_fragment_D(
          gV_split(_, _, 0, 0));
      auto tArV = thr_mma_pv.partition_sg_fragment_B(
          gV_split(_, _, 0, 0));

      auto prefetch_k = make_block_2d_prefetch(copy_k);
      auto prefetch_v = make_block_2d_prefetch(copy_v);
      auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
      auto pVgV =
          prefetch_v.get_slice(thr_id).partition_S(gV_split);

      const int barrier_scope = 2;

      {
        auto prefetch_q = make_block_2d_prefetch(copy_q);
        auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
        for (int D = 0; D < size<3>(pQgQ); D++) {
          prefetch(prefetch_q, pQgQ(_, _, _, D));
        }
      }

      for (int D = 0; D < size<4>(pKgK); D++) {
        for (int S = 0; S < Stages; S++) {
          prefetch(prefetch_k, pKgK(_, _, _, blk_k0 + S, D));
        }
      }

      clear(tArA);
      fill(tA_max,
           cutlass::platform::numeric_limits<ElementA>::lowest());
      clear(tA_sum);

      bool check_remainder_k = (seq_len % tile_k != 0);

      for (int K = blk_k0; K < blk_k1; K++) {
        bool need_causal = false;
        if constexpr (CausalMask) {
          need_causal = K >= blk_k1_causal;
        }

#if SYNC
        barrier_arrive(barrier_scope);
#endif

        clear(tSrS);
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < size<4>(tKgK); D++) {
          copy(copy_q, tQgQ(_, _, _, D), tQrQ);
          copy(copy_k, tKgK(_, _, _, K, D), tKrK);
          reorder(tQrQ, tSrQ);
          reorder(tKrK, tSrK);
          cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
        }

        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          prefetch(prefetch_v, pVgV(_, _, _, VV, K));
        }

        {
          bool need_k_rem = check_remainder_k && K == blk_k1 - 1;
          if (need_causal || need_k_rem) {
            constexpr int kTileK = tile_k;
            constexpr int n_reps = kTileK / intel::sg_size;
            constexpr int elems_per_n = tSrS.size() / n_reps;
            int k_base = K * kTileK;
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < n_reps; n++) {
              int col = k_base + n * intel::sg_size + lane_id;
              if (need_k_rem && col >= seq_len) {
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < elems_per_n; j++) {
                  tSrS(n * elems_per_n + j) = ElementS(-INFINITY);
                }
              } else if constexpr (CausalMask) {
                if (need_causal) {
                  int causal_bound =
                      col - full_tile_offset - row_base;
                  CUTLASS_PRAGMA_UNROLL
                  for (int j = 0; j < elems_per_n; j++) {
                    if (j < causal_bound) {
                      tSrS(n * elems_per_n + j) =
                          ElementS(-INFINITY);
                    }
                  }
                }
              }
            }
          }
        }
        if constexpr (LocalMask) {
          Tensor cPgP =
              make_identity_tensor(make_shape(seq_len, seq_len));
          Tensor gP = local_tile(
              cPgP, take<0, 2>(TileShapeQK{}),
              make_coord(get<0>(blk_qv), K));
          auto cS_thread = thr_mma_qk.partition_C(gP);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            int row_idx = get<0>(cS_thread(i));
            int col_idx = get<1>(cS_thread(i)) - full_tile_offset;
            bool left_mask =
                col_idx < row_idx - params.local_left;
            bool right_mask =
                col_idx > row_idx + params.local_right;
            if (left_mask || right_mask) {
              tSrS(i) = ElementS(-INFINITY);
            }
          }
        }

        /* Apply softmax and scaling (tA rescaling fused into GEMM2 VTile loop) */
        auto rescale = softmax(K == blk_k0, tSrS, tA_max, tA_sum);
        reorder(tSrS, tArP);

        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          copy(copy_v, tVgV(_, _, _, VV, K), tVrV);
          reorder(tVrV, tArV);
          if (K != blk_k0) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tArA.size() / VTiles; i++)
              tArA(_,_,_,VV)(i) *= broadcast<0>(rescale, tArA, i);
          }
          cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
        }

        for (int D = 0; D < size<4>(pKgK); D++) {
          prefetch(prefetch_k, pKgK(_, _, _, K + Stages, D));
        }

#if SYNC
        barrier_wait(barrier_scope);
#endif
      }
    }
  }

  // Single step of blocked softmax.
  CUTLASS_DEVICE
  FragSRow softmax(
      bool first_block,  // First softmax block?
      FragS& tS,         // Softmax src/dst block
      FragSRow& tS_max,  // Softmax row-wise max accumulator
      FragSRow& tS_sum) {  // Softmax row-wise sum accumulator 

    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    FragSRow rescale;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      ElementS new_max = sycl::max(tS_max(i), params.scale * tS_bmax(i));
      rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
      tS_max(i) = new_max;
    }

    /* Scale S and subtract maxima, then exponentiate */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++) {
      tS(i) = sycl::native::exp2(params.scale * tS(i) - broadcast<0>(tS_max, tS, i));
    }

    /* Rescale existing S sums */
    if (!first_block) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_sum.size(); i++) {
        tS_sum(i) *= rescale(i);
      }
    }

    /* Update sums */
    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_sum.size(); i++) {
      tS_sum(i) += tS_bsum(i);
    }
    return rescale;
  }
};

template <
    class DispatchPolicy_,
    bool PagedKV_,
    bool CausalMask_,
    class TiledMMAQK_,  // Tiling for Q*K GEMM
    class TiledMMAPV_,  // Tiling for P*V GEMM
    int VTiles_,        // # of tiles in V dimension
    class TensorQ_,     // Global Q/K/V tensors
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_ = void,  // Optional TiledCopy for loading Q
    class TiledCopyK_ = void,  // Optional TiledCopy for loading K
    class TiledCopyV_ = void,  // Optional TiledCopy for loading V
    bool LocalMask_ = false>
struct DecodeFwdMainloop {
  static_assert(
      cutlass::detail::dependent_false<DispatchPolicy_>,
      "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int Stages,
    bool PagedKV_,
    bool CausalMask_,
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    class TensorQ_,
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_,
    class TiledCopyK_,
    class TiledCopyV_,
    bool LocalMask_>
struct DecodeFwdMainloop<
    XeDefault<Stages>,
    PagedKV_,
    CausalMask_,
    TiledMMAQK_,
    TiledMMAPV_,
    VTiles_,
    TensorQ_,
    TensorK_,
    TensorV_,
    TiledCopyQ_,
    TiledCopyK_,
    TiledCopyV_,
    LocalMask_> {
  //
  // Type Aliases
  //
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(
      take<1, 4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;
  using TensorV = TensorV_;

  using ElementQ = typename TensorQ::engine_type::value_type;
  using ElementK = typename TensorK::engine_type::value_type;

  using TensorQ2D =
      decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_, _), 0)));
  using TensorK2D =
      decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _), 0)));
  using TensorV2D =
      decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _), 0)));

  using TiledCopyQ = conditional_t<
      is_void_v<TiledCopyQ_>,
      decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})),
      TiledCopyQ_>;
  using TiledCopyK = conditional_t<
      is_void_v<TiledCopyK_>,
      decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})),
      TiledCopyK_>;
  using TiledCopyV = conditional_t<
      is_void_v<TiledCopyV_>,
      decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})),
      TiledCopyV_>;

  // TODO: static_asserts on TiledMMAPV here...

  //
  // Accumulator types
  //
  // FragS:    accumulator for Q*K MMA
  // FragO:    accumulator for P*V MMAs.
  //           Note: v mode may be split into multiple pieces
  //             to reduce register pressure.
  // Frag*Row types are reductions of the corresponding Frag* types
  //   over rows.
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using FragSCol = decltype(reduce<0>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  using SingleFragA = FragC<TiledMMAPV>;  // (atom val,q',v')
  using FragA =
      expand_sg_fragment_t<SingleFragA, 1, VTiles>;  // (atom val,q',v',VV)
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  // static_assert(is_same_v<decltype(FragSRow{}.shape()), float>, "dtype
  // mismatched");
  using ElementA = typename TiledMMAPV::ValTypeD;

  static constexpr bool PagedKV = PagedKV_;
  static constexpr bool CausalMask = CausalMask_;
  static constexpr bool Fp8KV =
      is_any_of_v<ElementK, float_e5m2_t, float_e4m3_t>;
  static constexpr bool LocalMask = LocalMask_;

  // User-facing arguments
  struct Arguments {
    ElementS const scale;
    void* const scale_k;
    void* const scale_v;
    // Paged KV Cache
    int const* ptr_page_table;
    int page_size;
    int max_pages_per_seq;
    int total_seqlen_kv;
    // Local Mask
    int window_size_left;
    int window_size_right;
  };

  // Kernel-facing parameters
  using Params = Arguments;

  // SLM data
  struct SharedStorage {};

  Params params;

  //
  // Methods
  //

  DecodeFwdMainloop(Params const& params_, SharedStorage&) : params(params_) {}

  static constexpr Params
  to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    constexpr double kLog2e = 1.4426950408889634074;  // log_2(e)
    ElementS val = args.scale * static_cast<ElementS>(kLog2e);
    return Params{
        val,
        args.scale_k,
        args.scale_v,
        args.ptr_page_table,
        args.page_size,
        args.max_pages_per_seq,
        args.total_seqlen_kv,
        args.window_size_left,
        args.window_size_right};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,  // (q,d)
      TensorK2D const& K_2D,  // (k,d)
      TensorV2D const& V_2D,  // (d,k)
      FragA& tArA,            // Output accumulator (q,v)
      FragARow& tA_max,       // Softmax row-wise max accumulator
      FragARow& tA_sum,       // Softmax row-wise sum accumulator
      QVCoord blk_qv,         // WG tile indices: (Q,V)
      int const& idx_b,       // WG tile indices: (B)
      int blk_k0,             // K block range: [K0,K1)
      int blk_k1,
      int total_blk,  // Total # of K blocks
      int thr_id,
      int seq_len,
      int full_tile_offset,
      int discard_seq_coord) {
    using namespace sycl::ext::oneapi::this_work_item;

    // Short dimension names:
    //    q = sequence len dimension for Q
    //    k = sequence len dimension for K
    //    d = head size dimension for K/Q
    //    v = head size dimension for V
    //   VV = MMA tile indices for V
    // Capital letters (Q, K, ...) refer to WG block indices.
    // Primed letters (q', k', ...) refer to atom block indices.

    auto tile_shape_v =
        make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    /* Create proxy coordinate tensors for Q/K/P/V */
    Tensor cQ = make_identity_tensor(Q_2D.shape());               // (q,d)
    Tensor cK = make_identity_tensor(K_2D.shape());               // (k,d)
    Tensor cV = make_identity_tensor(V_2D.shape());               // (v,k)
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));  // (q,k)

    /* Partition global tensors into workgroup tiles */
    Tensor gQ = local_tile(
        cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});  // (q,d,D)
    Tensor gK = local_tile(
        cK,
        TileShapeQK{},
        make_coord(_, _, _),
        Step<X, _1, _1>{});  // (k,d,K,D)
    Tensor gV =
        local_tile(cV, tile_shape_v, make_coord(get<1>(blk_qv), _));  // (v,k,K)
    Tensor gV_split = local_tile(
        gV,
        TileShapePV{},
        make_coord(_, _, 0),
        Step<X, _1, _1>{});  // (v,k,VV,K)

    /* Create global -> register copies */
    TiledCopyQ copy_q{Q_2D};
    TiledCopyK copy_k{K_2D};
    TiledCopyV copy_v{V_2D};

    /* Create MMAs */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    auto copyQ = make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{});

    /* Slice TiledCopy/TiledMMA operations down to to work-item level */
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    /* Partition coordinate tensors for copy */
    auto tQgQ = thr_copy_q.partition_S(gQ);        // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);        // (atom_val,k',d',K,D)
    auto tVgV = thr_copy_v.partition_S(gV_split);  // (atom_val,v',k',VV,K)

    /* Create register fragments for MMA and copies */
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));

    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_, _, 0, 0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_, _, 0, 0));

    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);

    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_, _, 0, 0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_, _, 0, 0));

    /* Create TiledCopy objects for prefetches */
    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_k = make_block_2d_prefetch(copy_k);
    auto prefetch_v =
        make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D);

    /* Partition global tensors for prefetch */
    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV);

    // ------
    // Kernel
    // ------

    // PagedKV
    int tiles_per_page = params.page_size / get<1>(TileShapeQK{});
    int tile_idx = blk_k0;
    int b_offset = idx_b * params.max_pages_per_seq;
    if constexpr (PagedKV) {
      int page_local_idx = tile_idx * get<1>(TileShapeQK{}) / params.page_size;
      tile_idx =
          params.ptr_page_table[b_offset + page_local_idx] * tiles_per_page +
          tile_idx % tiles_per_page;
    }

    /* Initialization steps for first block: Q/K prefetch, O init */
    /* TODO: limit D prefetch for large head size, and reorder K prefetches */
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_, _, _, D));
    }

    for (int D = 0; D < size<4>(pKgK); D++) {
      prefetch(prefetch_k, pKgK(_, _, _, tile_idx, D));
    }

    clear(tArA);
    fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    clear(tA_sum);

    /* Check if */
    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);

    /* Pre-compute lane_id for k remainder scalar arithmetic */
    int lane_id = thr_id % intel::sg_size;
    // FP8 KV Scale: Currently we only support per-tensor scale for KV
    float scale_k = 1.f, scale_v = 1.f;
    if constexpr (Fp8KV) {
      scale_k = *static_cast<const float*>(params.scale_k);
      scale_v = *static_cast<const float*>(params.scale_v);
    }

    /* Main loop, blocked in k. */
    int next_tile_idx;
    for (int K = blk_k0; K < blk_k1; K++) {
      /* Split barrier to keep threads together */
      // barrier_arrive(ScopeWorkgroup);

      auto tKgK_cache =
          PagedKV ? tKgK(_, _, _, tile_idx, _) : tKgK(_, _, _, K, _);
      auto tVgV_cache =
          PagedKV ? tVgV(_, _, _, _, tile_idx) : tVgV(_, _, _, _, K);

      /* GEMM 1: S = K * Q */
      clear(tSrS); /* TODO: fuse w/ initial gemm call */
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        copy(copy_k, tKgK_cache(_, _, _, D), tKrK);

        reorder(tQrQ, tSrQ);
        reorder(tKrK, tSrK);
        if constexpr (Fp8KV) {
          for (int i = 0; i < tSrK.size(); ++i) {
            tSrK(i) =
                static_cast<ElementQ>(scale_k * static_cast<float>(tSrK(i)));
          }
        }

        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      /* V prefetch for GEMM 2 */
      prefetch(prefetch_v, pVgV(_, _, _, tile_idx));

      /* Causal masking */
      // No Causal masking in decoding
      // if constexpr (CausalMask) {
      //   if (K == blk_k1 - 1) {
      //     // Need to get global col and row indices to mask the elements
      //     Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
      //     Tensor gP = local_tile(cPgP, take<0,2>(TileShapeQK{}),
      //     make_coord(get<0>(blk_qv), K)); auto cS_thread =
      //     thr_mma_qk.partition_C(gP); CUTLASS_PRAGMA_UNROLL for (int i = 0; i
      //     < tSrS.size(); ++i) {
      //       int row_idx = get<0>(cS_thread(i));
      //       int col_idx = get<1>(cS_thread(i));
      //       if (col_idx - full_tile_offset > row_idx - discard_seq_coord) {
      //         tSrS(i) = ElementS(-INFINITY);
      //       }
      //     }
      //   }
      // }

      /* Local/sliding window masking */
      if constexpr (LocalMask) {
        // For decode, all packed GQA heads share the same KV position
        // (seq_len_kv - 1). Use a fixed decode row for all elements.
        int decode_row = seq_len - 1 - full_tile_offset;
        Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
        Tensor gP = local_tile(
            cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
        auto cS_thread = thr_mma_qk.partition_C(gP);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); ++i) {
          int col_idx = get<1>(cS_thread(i)) - full_tile_offset;
          bool left_mask = col_idx < decode_row - params.window_size_left;
          bool right_mask = col_idx > decode_row + params.window_size_right;
          if (left_mask || right_mask) {
            tSrS(i) = ElementS(-INFINITY);
          }
        }
      }

      /* k masking for remainder tiles — scalar arithmetic */
      if (check_remainder_k && K == blk_k1 - 1) {
        constexpr int kTileK = get<1>(TileShapeQK{});
        constexpr int n_reps = kTileK / intel::sg_size;
        int elems_per_n = tSrS.size() / n_reps;
        int k_base = K * kTileK;
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < n_reps; n++) {
          if (k_base + n * intel::sg_size + lane_id >= seq_len) {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < elems_per_n; j++) {
              tSrS(n * elems_per_n + j) = ElementS(-INFINITY);
            }
          }
        }
      }

      /* Apply softmax and scaling */
      softmax(K == blk_k0, tSrS, tA_max, tA_sum, tArA);
      reorder(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v, tVgV_cache(_, _, _, VV), tVrV);
        reorder(tVrV, tArV);
        if constexpr (Fp8KV) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tArV.size(); ++i) {
            tArV(i) =
                static_cast<ElementQ>(scale_v * static_cast<float>(tArV(i)));
          }
        }
        cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
      }

      barrier();

      // next tile_idx
      next_tile_idx = K + 1;
      if constexpr (PagedKV) {
        int next_page_local_idx =
            next_tile_idx * get<1>(TileShapeQK{}) / params.page_size;
        if (next_page_local_idx < params.max_pages_per_seq) {
          next_tile_idx =
              params.ptr_page_table[b_offset + next_page_local_idx] *
                  tiles_per_page +
              next_tile_idx % tiles_per_page;
        } else {
          // set to last page
          next_tile_idx = params.max_pages_per_seq * tiles_per_page - 1;
        }
      }
      tile_idx = next_tile_idx;

      /* K prefetch */
      for (int D = 0; D < size<4>(pKgK); D++) {
        prefetch(prefetch_k, pKgK(_, _, _, tile_idx, D));
      }

      // barrier_wait(ScopeWorkgroup);
    }
  }

  // Single step of blocked softmax.
  CUTLASS_DEVICE
  void softmax(
      bool first_block,  // First softmax block?
      FragS& tS,         // Softmax src/dst block
      FragSRow& tS_max,  // Softmax row-wise max accumulator
      FragSRow& tS_sum,  // Softmax row-wise sum accumulator
      FragA& tA) {       // O accumulator (for rescaling)

    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    /* Update (scaled) maxima */
    auto tS_prev_max = tS_max;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      tS_max(i) = sycl::max(tS_max(i), params.scale * tS_bmax(i));
    }

    /* Scale S and subtract maxima, then exponentiate */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(
          params.scale * tS(i) - broadcast<0>(tS_max, tS, i));

    /* Rescale existing S sums and O accumulator */
    if (!first_block) {
      FragSRow rescale;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_max.size(); i++) {
        rescale(i) = sycl::native::exp2(tS_prev_max(i) - tS_max(i));
        tS_sum(i) *= rescale(i);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tA.size(); i++)
        tA(i) *= broadcast<0>(rescale, tA, i);
    }

    /* Update sums */
    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    for (int i = 0; i < tS_sum.size(); i++)
      tS_sum(i) += tS_bsum(i);
  }
};

template <typename SGLayoutQK>
CUTLASS_HOST_DEVICE constexpr auto get_sg_layout_pv(SGLayoutQK const&) {
  return make_layout(
      get<0>(SGLayoutQK{}), Layout<_1, _0>{}, get<1>(SGLayoutQK{}));
}

// Get a P*V TiledMMA given K*Q tile size and SG configuration, for mainloops
//   not supporting S data interchange among subgroups (e.g. XeDefault).
template <
    typename MMAOp,
    typename WGTileQK,
    typename SGLayoutQK,
    typename TileV>
CUTLASS_HOST_DEVICE constexpr auto get_tiled_mma_pv(
    MMAOp const&,
    WGTileQK const& wg_tile_qk,
    SGLayoutQK const& sg_layout_qk,
    TileV const&) {
  using TileQ = decltype(get<0>(wg_tile_qk));
  using TileK = decltype(get<1>(wg_tile_qk));

  using WGTilePV = Shape<TileQ, TileV, TileK>;
  using SGLayoutPV = decltype(get_sg_layout_pv(sg_layout_qk));

  static_assert(
      size(SGLayoutPV{}) == size(SGLayoutQK{}),
      "Q*K cannot be parallelized in the head size dimension");

  return TiledMMAHelper<MMAOp, WGTilePV, SGLayoutPV>{};
}

}  // namespace cutlass::fmha::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
