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

#include <array>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/util/sycl_vec.hpp"
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

#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
CUTE_DEVICE
void cvt_f32x2_to_bf16x2_bias(
    float const& src0, float const& src1, cute::intel::uint2& tmp) {
  asm("{\n"
      ".decl IN_UD0 v_type=G type=UD num_elts=16 alias=<%1,0>\n"
      ".decl IN_UD1 v_type=G type=UD num_elts=16 alias=<%2,0>\n"
      ".decl TMP_UD v_type=G type=UD num_elts=32 alias=<%0,0>\n"
      "add (M1_NM, 16) TMP_UD(0,0)<1> IN_UD0(0,0)<1;1,0> 0x8000:uw\n"
      "add (M1_NM, 16) TMP_UD(1,0)<1> IN_UD1(0,0)<1;1,0> 0x8000:uw\n"
      "}\n"
      : "=rw"(tmp)
      : "rw"(src0), "rw"(src1));
}

CUTE_DEVICE
void cvt_f32x2_to_bf16x2_pack(
    cute::intel::uint2 const& tmp, cute::intel::ushort2& dst) {
  asm("{\n"
      ".decl TMP_UD v_type=G type=UD num_elts=32 alias=<%1,0>\n"
      ".decl TMP_UW v_type=G type=UW num_elts=64 alias=<TMP_UD,0>\n"
      ".decl OUT_UW v_type=G type=UW num_elts=32 alias=<%0,0>\n"
      "mov (M1_NM, 32) OUT_UW(0,0)<1> TMP_UW(0,1)<2;1,0>\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(tmp));
}
#else
CUTE_DEVICE
void cvt_f32x2_to_bf16x2_bias(
    float const& /*src0*/, float const& /*src1*/, cute::intel::uint2& /*tmp*/) {
  CUTE_INVALID_CONTROL_PATH(
      "cvt_f32x2_to_bf16x2_bias requires Intel Xe SYCL device target");
}
CUTE_DEVICE
void cvt_f32x2_to_bf16x2_pack(
    cute::intel::uint2 const& /*tmp*/, cute::intel::ushort2& /*dst*/) {
  CUTE_INVALID_CONTROL_PATH(
      "cvt_f32x2_to_bf16x2_pack requires Intel Xe SYCL device target");
}
#endif

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
  // Head size (VTiles * PV N-tile) must be divisible by the QK K-tile so that
  // the GEMM1 reduction dimension splits evenly into DTiles register tiles.
  static_assert(
      (VTiles * decltype(get<1>(TileShapePV{}))::value) %
              decltype(get<2>(TileShapeQK{}))::value ==
          0,
      "Head size (VTiles * PV N-tile) must be divisible by the QK "
      "K-tile (BLK_QK_D); check the ShapeQK/ShapePV tile "
      "configuration for this head dimension.");
  static constexpr int DTiles = VTiles *
                                decltype(get<1>(TileShapePV{}))::value /
                                decltype(get<2>(TileShapeQK{}))::value;
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
  // Per-lane partial row sum: vertical (in-register) reduction only, deferring
  // the cross-lane horizontal hreduce of the softmax denominator to the
  // epilogue.
  using FragSPartialRow =
      decltype(reduce<1, ReduceMode::Vertical>(FragS{}, sycl::plus<void>{}));
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

    // Paged KV Cache
    const int* ptr_page_table;
    int page_size;
    int max_pages_per_seq;
    int total_seqlen_kv;
    // Local Mask
    int local_left, local_right;
    // Interleaved KV Cache
    bool is_interleaved_kv_cache;
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
        args.local_left,
        args.local_right,
        args.is_interleaved_kv_cache};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  CUTLASS_DEVICE int get_paged_idx(int K, int idx_b) {
    int tiles_per_page = params.page_size / get<1>(TileShapeQK{});
    int b_offset = idx_b * params.max_pages_per_seq;
    int page_local_idx = K * get<1>(TileShapeQK{}) / params.page_size;

    // Clamp page_local_idx to the valid range [0, max_pages_per_seq - 1]
    if (page_local_idx >= params.max_pages_per_seq) {
      page_local_idx = params.max_pages_per_seq - 1;
    }

    int block_idx = params.ptr_page_table[b_offset + page_local_idx];
    if (params.is_interleaved_kv_cache) block_idx *= 2;
    return block_idx * tiles_per_page + K % tiles_per_page;
  }

  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,    // (q,d)
      TensorK2D const& K_2D,    // (k,d)
      TensorV2D const& V_2D,    // (d,k)
      FragA& tArA,              // Output accumulator (q,v)
      FragARow& tA_max,         // Softmax row-wise max accumulator
      FragSPartialRow& tA_sum,  // Softmax row-wise partial sum (per-lane)
      QVCoord blk_qv,           // WG tile indices: (Q,V)
      int const& idx_b,         // WG tile indices: (B)
      int blk_k0,               // K block range: [K0,K1)
      int blk_k1,
      int blk_k1_causal,
      int thr_id,
      int seq_len,
      int full_tile_offset,
      int blk_local_l_safe,
      int blk_local_r_safe) {
    using namespace sycl::ext::oneapi::this_work_item;

    // Short dimension names:
    //    q = sequence len dimension for Q
    //    k = sequence len dimension for K
    //    d = head size dimension for K/Q
    //    v = head size dimension for V
    //   VV = MMA tile indices for V
    // Capital letters (Q, K, ...) refer to WG block indices.
    // Primed letters (q', k', ...) refer to atom block indices.

    constexpr int tile_k = get<1>(TileShapeQK{});

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
    auto tile_shape_k =
        make_shape(get<1>(TileShapeQK{}), get<2>(TileShapeQK{}) * C<DTiles>{});
    Tensor gK_prefetch =
        local_tile(cK, tile_shape_k, make_coord(_, 0));  // (k,d,K)

    /* Create global -> register copies */
    TiledCopyQ copy_q{Q_2D};
    TiledCopyK copy_k{K_2D};
    TiledCopyV copy_v{V_2D};

    /* Create MMAs */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    /* Slice TiledCopy/TiledMMA operations down to work-item level */
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
    [[maybe_unused]] auto tSrQ =
        thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));
    std::array<decltype(tSrQ), DTiles> tSrQ_arr;
    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_, _, 0, 0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_, _, 0, 0));
    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);
    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_, _, 0, 0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_, _, 0, 0));

    /* Create TiledCopy objects for prefetches */
    auto prefetch_k =
        make_block_2d_prefetch<SGPerWG::value>(tile_shape_k, K_2D);
    auto prefetch_v =
        make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D);
    /* Partition global tensors for prefetch */
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK_prefetch);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV);

    // ------
    // Kernel
    // ------

    /* Initialization steps for first block: Q/K prefetch, O init */
    using PreparedK_t =
        decltype(prepare_payloads(copy_k, tKgK(_, _, _, 0, 0), tKrK));
    using PreparedV_t =
        decltype(prepare_payloads(copy_v, tVgV(_, _, _, 0, 0), tVrV));
    std::array<PreparedK_t, DTiles> prepared_k;
    std::array<PreparedV_t, VTiles> prepared_v;

    /* Preload + reorder Q once; reused across all K iterations. */
    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < DTiles; d++) {
      copy(copy_q, tQgQ(_, _, _, d), tQrQ);
      reorder(tQrQ, tSrQ_arr[d]);
    }

    /* Masking scalars */
    int lane_id = thr_id % intel::sg_size;
    constexpr int sg_tile_q = get<0>(TileShapeQK{}) / SGPerWG::value;
    int row_base = get<0>(blk_qv) * get<0>(TileShapeQK{}) +
                   (thr_id / intel::sg_size) * sg_tile_q;

    // PagedKV
    constexpr int kv_stride = get<1>(TileShapeQK{});
    int page_idx, next_page_idx = blk_k0, page_offset = kv_stride;
    if constexpr (PagedKV) {
      next_page_idx = get_paged_idx(blk_k0, idx_b);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int D = 0; D < DTiles; D++) {
      prepared_k[D] =
          prepare_payloads(copy_k, tKgK(_, _, _, next_page_idx, D), tKrK);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int VV = 0; VV < VTiles; VV++) {
      prepared_v[VV] =
          prepare_payloads(copy_v, tVgV(_, _, _, VV, next_page_idx), tVrV);
    }

    auto prepared_pk = prepare_payloads(
        prefetch_k, pKgK(_, _, _, next_page_idx), pKgK(_, _, _, next_page_idx));
    auto prepared_pv = prepare_payloads(
        prefetch_v, pVgV(_, _, _, next_page_idx), pVgV(_, _, _, next_page_idx));

    prefetch_with_payloads(prefetch_k, prepared_pk, shape(pKgK(_, _, _, 0)));
    prefetch_with_payloads(prefetch_v, prepared_pv, shape(pVgV(_, _, _, 0)));

    clear(tArA);
    fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    clear(tA_sum);

    bool check_remainder_k = (seq_len % tile_k != 0);

    // FP8 KV Scale: Currently we only support per-tensor scale for KV
    float scale_k = 1.f, scale_v = 1.f;
    if constexpr (Fp8KV) {
      scale_k = *static_cast<const float*>(params.scale_k);
      scale_v = *static_cast<const float*>(params.scale_v);
    }

    constexpr int kAtomsPerD =
        decltype(get<2>(TileShapeQK{}))::value /
        decltype(get<2>(typename TiledMMAQK::AtomShape_MNK{}))::value;

    /* Main loop, blocked in k. */
    for (int K = blk_k0; K < blk_k1; K++) {
      bool need_causal = false;
      if constexpr (CausalMask) {
        need_causal = K >= blk_k1_causal;
      }

      page_idx = next_page_idx;
      next_page_idx = K + 1;
      // next paged_idx
      if constexpr (PagedKV) {
        next_page_idx = get_paged_idx(next_page_idx, idx_b);
        page_offset = (next_page_idx - page_idx) * kv_stride;
      }

      auto tKgK_cache =
          PagedKV ? tKgK(_, _, _, page_idx, _) : tKgK(_, _, _, K, _);
      auto tVgV_cache =
          PagedKV ? tVgV(_, _, _, _, page_idx) : tVgV(_, _, _, _, K);

      /* V prefetch for next iter */
      update_payloads(prepared_pv, page_offset);
      prefetch_with_payloads(prefetch_v, prepared_pv, shape(pVgV(_, _, _, 0)));

      /* GEMM 1: S = Q * K^T */
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < DTiles; D++) {
        copy_with_multi_payloads(copy_k, prepared_k[D], tKrK);
        update_payloads(prepared_k[D], page_offset);
        reorder(tKrK, tSrK);

        if constexpr (Fp8KV) {
          dequantize(tSrK, scale_k);
        }

        auto const& tSrQ_d = tSrQ_arr[D];
        if (D == 0) {
          cute::gemm<true>(mma_qk, tSrQ_d(_, _, 0), tSrK(_, _, 0), tSrS);
          CUTLASS_PRAGMA_UNROLL
          for (int k = 1; k < kAtomsPerD; k++) {
            cute::gemm(mma_qk, tSrQ_d(_, _, k), tSrK(_, _, k), tSrS);
          }
        } else {
          cute::gemm(mma_qk, tSrQ_d, tSrK, tSrS);
        }
      }

      /* K prefetch for next iteration */
      update_payloads(prepared_pk, page_offset);
      prefetch_with_payloads(prefetch_k, prepared_pk, shape(pKgK(_, _, _, 0)));

      /* Early V-load for VV=0: overlap memory latency with softmax scalar work
       */
      copy_with_multi_payloads(copy_v, prepared_v[0], tVrV);
      update_payloads(prepared_v[0], page_offset);

      /* Causal masking and k remainder masking */
      if constexpr (CausalMask) {
        if (need_causal) {
          constexpr int kTileK = tile_k;
          constexpr int n_reps = kTileK / intel::sg_size;
          constexpr int elems_per_n = tSrS.size() / n_reps;
          int k_base = K * kTileK;
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < n_reps; n++) {
            int col = k_base + n * intel::sg_size + lane_id;
            int causal_bound = col - full_tile_offset - row_base;
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < elems_per_n; j++) {
              if (j < causal_bound) {
                tSrS(n * elems_per_n + j) = ElementS(-INFINITY);
              }
            }
          }
        }
      }
      /* k masking for remainder tiles */
      if (check_remainder_k && K == blk_k1 - 1) {
        FragSCol k_rem_mask;
        int k = get<0>(tKgK(0, 0, 0, K, 0)) + get_sub_group().get_local_id()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
          k_rem_mask(i) =
              (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
        }
      }
      /* Local masking */
      if constexpr (LocalMask) {
        const bool need_left = (K < blk_local_l_safe);
        const bool need_right = (K > blk_local_r_safe);
        if (need_left || need_right) {
          Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
          Tensor gP = local_tile(
              cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
          auto cS_thread = thr_mma_qk.partition_C(gP);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            int row_idx = get<0>(cS_thread(i));
            int col_idx = get<1>(cS_thread(i)) - full_tile_offset;
            bool left_mask = col_idx < row_idx - params.local_left;
            bool right_mask = col_idx > row_idx + params.local_right;
            if (left_mask || right_mask) {
              tSrS(i) = ElementS(-INFINITY);
            }
          }
        }
      }

      /* Apply softmax (deferred row-sum hreduce) */
      auto [rescale, tS_partial_sum] = softmax(tSrS, tA_max, tA_sum);
      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      constexpr int kSumSize = decltype(tA_sum.size())::value;
      constexpr bool kSumDivVT = (kSumSize % VTiles == 0);
      constexpr int kSumPerVT = kSumDivVT ? (kSumSize / VTiles) : 0;

      using ElementP = typename TiledMMAPV::ValTypeA;
      if constexpr (std::is_same_v<ElementP, bfloat16_t>) {
        static_assert(
            decltype(tArP.size())::value % 2 == 0,
            "tArP per-WI element count must be even for f32x2->bf16x2 packing");
        constexpr int kCvtPairs = decltype(tSrS.size())::value / 2;
        cute::intel::uint2 cvt_tmp[kCvtPairs];
        CUTLASS_PRAGMA_UNROLL
        for (int p = 0; p < kCvtPairs; p++) {
          cvt_f32x2_to_bf16x2_bias(tSrS(2 * p), tSrS(2 * p + 1), cvt_tmp[p]);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int p = 0; p < kCvtPairs; p++) {
          cvt_f32x2_to_bf16x2_pack(
              cvt_tmp[p], reinterpret_cast<cute::intel::ushort2&>(tArP(2 * p)));
        }
      } else {
        reorder(tSrS, tArP);
      }

      /* GEMM 2: A += P * V, split in v dimension.
       * VV=0 pre-loaded before softmax; VV>0 loaded inline. */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        if (VV > 0) {
          copy_with_multi_payloads(copy_v, prepared_v[VV], tVrV);
          update_payloads(prepared_v[VV], page_offset);
        }
        reorder(tVrV, tArV);

        CUTLASS_PRAGMA_UNROLL
        for (int i = tArA.size() / VTiles - 1; i >= 0; i--)
          tArA(_, _, _, VV)(i) *= broadcast<0>(rescale, tArA, i);
        /* Rescale + accumulate per-lane partial row sums, fused into V loop. */
        if constexpr (kSumDivVT) {
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < kSumPerVT; j++) {
            int const i = VV * kSumPerVT + j;
            tA_sum(i) = tA_sum(i) * group_broadcast(sg, rescale(0), i) +
                        tS_partial_sum(i);
          }
        }
        if constexpr (Fp8KV) {
          dequantize(tArV, scale_v);
        }
        cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
      }
    }
  }

  // Single step of blocked softmax.
  // Returns (rescale, per-lane vertical partial sum). The cross-lane horizontal
  // reduction of the row-sum is deferred: tS_sum keeps per-lane partial sums
  // and is collapsed once in the epilogue (reduce<0, Horizontal>).
  CUTLASS_DEVICE
  auto softmax(
      FragS& tS,                  // Softmax src/dst block
      FragARow& tA_max,           // Softmax row-wise max accumulator
      FragSPartialRow& tA_sum) {  // Softmax row-wise partial sum (per-lane)

    /* Compute row-wise maxima for this block */
    auto tS_bmax =
        reduce<1, ReduceMode::Full, /*EnableFast64Rows=*/!CausalMask>(
            tS, sycl::maximum<void>{});

    FragARow rescale;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tA_max.size(); i++) {
      ElementS new_max = sycl::max(tA_max(i), params.scale * tS_bmax(i));
      rescale(i) = sycl::native::exp2(tA_max(i) - new_max);
      tA_max(i) = new_max;
    }

    /* Scale S and subtract maxima, then exponentiate */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = params.scale * tS(i) - broadcast<0>(tA_max, tS, i);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(tS(i));

    /* Per-lane vertical partial sum (deferred horizontal reduction) */
    auto tS_partial_sum =
        reduce<1, ReduceMode::Vertical>(tS, sycl::plus<void>{});

    constexpr int kSumSize = decltype(tA_sum.size())::value;
    constexpr bool kSumDivVT = (kSumSize % VTiles == 0);

    // When tS_sum.size() does not divide VTiles, rescale + accumulate the
    // partial sums once here instead of fusing per VTile in the PV loop.
    if constexpr (!kSumDivVT) {
      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kSumSize; i++) {
        tA_sum(i) =
            tA_sum(i) * group_broadcast(sg, rescale(0), i) + tS_partial_sum(i);
      }
    }

    return cute::make_tuple(rescale, tS_partial_sum);
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
    // Interleaved KV Cache
    bool is_interleaved_kv_cache;
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
        args.window_size_right,
        args.is_interleaved_kv_cache};
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
      int block_idx = params.ptr_page_table[b_offset + page_local_idx];
      if (params.is_interleaved_kv_cache) block_idx *= 2;
      tile_idx = block_idx * tiles_per_page + tile_idx % tiles_per_page;
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

      /* k masking for remainder tiles */
      if (check_remainder_k && K == blk_k1 - 1) {
        FragSCol k_rem_mask;
        int k = get<0>(tKgK(0, 0, 0, K, 0)) + get_sub_group().get_local_id()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
          k_rem_mask(i) =
              (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
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
          int next_block_idx =
              params.ptr_page_table[b_offset + next_page_local_idx];
          if (params.is_interleaved_kv_cache) next_block_idx *= 2;
          next_tile_idx =
              next_block_idx * tiles_per_page + next_tile_idx % tiles_per_page;
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
