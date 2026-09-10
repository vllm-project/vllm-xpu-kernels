/***************************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
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
#include "cutlass/fp8_to_fp16.h"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////
namespace cutlass::gemm {

// PerTensor selects the scale-granularity of the e4m3 GEMM:
//   false (default): 128x128 block-wise scales (A: [M, K/128], B: [N/128,
//          K/128] per expert) -- the original behavior.
//   true: per-tensor scales (A: a single scalar [1], B: one scalar per expert
//          [E]) -- every block reads the same scalar.
template <
    int Stages_,
    class KernelSchedule = KernelMoEArrayCooperative,
    bool PerTensor_ = false>
struct MainloopIntelXeXMX16BlockFp8 {
  constexpr static int Stages = Stages_;
  constexpr static int SubgroupSize = 16;
  constexpr static bool PerTensor = PerTensor_;
  using ArchTag = arch::IntelXe;
  using Schedule = KernelSchedule;
  using ClusterShape = Shape<_1, _1, _1>;
};

}  // namespace cutlass::gemm
/////////////////////////////////////////////////////////////////////////////////////////////////
namespace cutlass::gemm::collective {
using namespace cute;
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int Stages,
    bool PerTensor,
    class TileShape_,
    class ElementPairA_,
    class StridePairA_,
    class ElementPairB_,
    class StridePairB_,
    class TiledMma_,
    class GmemTiledCopyPairA_,
    class SmemLayoutAtomA_,
    class SmemCopyAtomA_,
    class TransformA_,
    class GmemTiledCopyPairB_,
    class SmemLayoutAtomB_,
    class SmemCopyAtomB_,
    class TransformB_>
struct CollectiveMma<
    MainloopIntelXeXMX16BlockFp8<Stages, KernelMoEArrayCooperative, PerTensor>,
    TileShape_,
    ElementPairA_,
    StridePairA_,
    ElementPairB_,
    StridePairB_,
    TiledMma_,
    GmemTiledCopyPairA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyPairB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_> {
 public:
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopIntelXeXMX16BlockFp8<
      Stages,
      KernelMoEArrayCooperative,
      PerTensor>;
  using WorkgroupTileShape = TileShape_;

  using GmemTiledCopyPairA = GmemTiledCopyPairA_;
  using GmemTiledCopyPairB = GmemTiledCopyPairB_;

  using TiledMma = TiledMma_;
  using ElementPairA = ElementPairA_;
  using ElementPairB = ElementPairB_;
  using ElementAMma = typename TiledMma::ValTypeA;
  using ElementBMma = typename TiledMma::ValTypeB;
  using StridePairA = StridePairA_;
  using StridePairB = StridePairB_;

  using ElementMMA = typename TiledMma_::ValTypeA;

  // A and B matrices
  using ElementA = remove_cvref_t<decltype(get<0>(ElementPairA{}))>;
  using StrideA =
      cute::remove_pointer_t<remove_cvref_t<decltype(get<0>(StridePairA{}))>>;

  using ElementB = remove_cvref_t<decltype(get<0>(ElementPairB{}))>;
  using StrideB =
      cute::remove_pointer_t<remove_cvref_t<decltype(get<0>(StridePairB{}))>>;

  // SFA and SFB
  using ElementSF = remove_cvref_t<decltype(get<1>(ElementPairA{}))>;
  using StrideScaleA =
      cute::remove_pointer_t<remove_cvref_t<decltype(get<1>(StridePairA{}))>>;
  using StrideScaleB =
      cute::remove_pointer_t<remove_cvref_t<decltype(get<1>(StridePairB{}))>>;

  using ElementScaleA = ElementSF;
  using ElementScaleB = ElementSF;

  using ElementAccumulator = typename TiledMma::ValTypeC;

  using GmemTiledCopyA =
      typename std::tuple_element<0, GmemTiledCopyPairA>::type;
  using GmemTiledCopyB =
      typename std::tuple_element<0, GmemTiledCopyPairB>::type;
  using GmemTiledCopyScaleA =
      typename std::tuple_element<1, GmemTiledCopyPairA>::type;
  using GmemTiledCopyScaleB =
      typename std::tuple_element<1, GmemTiledCopyPairB>::type;

  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;
  using MmaType =
      typename TiledMma::ValTypeA;  // ValTypeA and ValTypeB are always same and
                                    // reflects MMA type on intel Xe

  static constexpr bool kSupportedElementA =
      cute::is_same_v<ElementA, float> ||
      cute::is_same_v<ElementA, cutlass::half_t> ||
      cute::is_same_v<ElementA, cutlass::float_e5m2_t> ||
      cute::is_same_v<ElementA, cutlass::float_e4m3_t> ||
      cute::is_same_v<ElementA, cutlass::float_e2m1_t>;

  static constexpr bool kSupportedElementB =
      cute::is_same_v<ElementB, cutlass::float_e5m2_t> ||
      cute::is_same_v<ElementB, cutlass::float_e4m3_t> ||
      cute::is_same_v<ElementB, cutlass::float_e2m1_t>;

  static_assert(
      std::is_same_v<TransformA, cute::identity>,
      "Transformation for A is not currently supported on Intel PVC");
  static_assert(
      std::is_same_v<TransformB, cute::identity>,
      "Transformation for B is not currently supported on Intel PVC");
  static_assert(
      kSupportedElementA && kSupportedElementB,
      "Intel Xe blockscaled MMA only supports bf8 and hf8 operand types.");

 public:
  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

  using MmaAtomShape = typename TiledMma::AtomShape_MNK;

  static constexpr int BLK_M = get<0>(WorkgroupTileShape{});
  static constexpr int BLK_N = get<1>(WorkgroupTileShape{});
  static constexpr int BLK_K = get<2>(WorkgroupTileShape{});

  static constexpr int SG_NUMS_M =
      get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr int SG_NUMS_N =
      get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr int SG_NUMS_K =
      get<3>(typename TiledMma::ThrLayoutVMNK{}.shape());

  static constexpr int MMA_M = get<0>(typename TiledMma::Shape_MNK{});
  static constexpr int MMA_N = get<1>(typename TiledMma::Shape_MNK{});
  static constexpr int MMA_K = get<2>(typename TiledMma::Shape_MNK{});

  static constexpr int SG_M = ceil_div(BLK_M, SG_NUMS_M);
  static constexpr int SG_N = ceil_div(BLK_N, SG_NUMS_N);
  static constexpr int SG_K = ceil_div(BLK_K, SG_NUMS_K);
  using SubgroupTileShape = Shape<C<SG_M>, C<SG_N>, C<SG_K>>;

  static constexpr auto GROUP_K = 128;

  static constexpr auto Num_SGs = SG_NUMS_N * SG_NUMS_M * SG_NUMS_K;
  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});

  using CopyThreadShape = Shape<_1, Int<SubgroupSize>>;
  using CopyThreadShapeRev = decltype(cute::reverse(CopyThreadShape{}));

  // Helper to get tensor types
  template <class Element, class Stride>
  using TensorType = decltype(make_tensor(
      make_gmem_ptr(static_cast<Element const*>(nullptr)),
      make_layout(make_shape(int{}, int{}, int{}), Stride{})));

  template <class Element, class Stride>
  using TensorScaleType = decltype(make_tensor(
      make_gmem_ptr(static_cast<Element const*>(nullptr)),
      make_layout(make_shape(
          make_shape(int{}, int{}), make_shape(int{}, int{}), int{}))));

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    ElementScaleA const* ptr_SA = nullptr;
    StrideScaleA dSA{};
    ElementScaleB const* ptr_SB = nullptr;
    StrideScaleB dSB{};
    int group_size = 128;
  };

  struct Params {
    TensorType<ElementA, StrideA> mA_mkl;
    TensorType<ElementB, StrideB> mB_nkl;
    ElementScaleA const* ptr_SA = nullptr;
    ElementScaleB const* ptr_SB = nullptr;
    int group_size;
  };

  //
  // Methods
  //

  CollectiveMma() = default;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace) {
    (void)workspace;

    auto [M, N, K, L] = problem_shape;

    auto mA_mkl = make_tensor(
        make_gmem_ptr(args.ptr_A), make_layout(make_shape(M, K, L), args.dA));

    auto ptr_B = [&]() {
      return make_gmem_ptr(static_cast<ElementB const*>(args.ptr_B));
    }();

    auto mB_nkl = make_tensor(ptr_B, make_layout(make_shape(N, K, L), args.dB));

    auto mScaleA = make_tensor(
        make_gmem_ptr(static_cast<ElementScaleA const*>(args.ptr_SA)),
        make_layout(
            make_shape(
                make_shape(1, cute::ceil_div(M, 1)),
                make_shape(K, cute::ceil_div(K, GROUP_K)),
                L),
            args.dSA));
    auto mScaleB = make_tensor(
        make_gmem_ptr(static_cast<ElementScaleB const*>(args.ptr_SB)),
        make_layout(
            make_shape(
                make_shape(GROUP_K, cute::ceil_div(N, GROUP_K)),
                make_shape(GROUP_K, cute::ceil_div(K, GROUP_K)),
                L),
            args.dSB));
    return Params{mA_mkl, mB_nkl, args.ptr_SA, args.ptr_SB, GROUP_K};
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape problem_shapes, Arguments const& args) {
    constexpr int copy_alignment_bits = 128;
    constexpr int batch_alignment_bits = 512;
    auto problem_shape_MNKL = append<4>(problem_shapes, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    bool implementable = true;

    constexpr int min_aligned_elements_A =
        copy_alignment_bits / sizeof_bits<ElementA>::value;
    implementable &= cutlass::detail::check_alignment<min_aligned_elements_A>(
        cute::make_shape(M, K, L), args.dA);
    constexpr int min_aligned_elements_B =
        copy_alignment_bits / sizeof_bits<ElementB>::value;
    implementable &= cutlass::detail::check_alignment<min_aligned_elements_B>(
        cute::make_shape(N, K, L), args.dB);

    if (L > 1) {
      constexpr int min_batch_aligned_elements_A =
          batch_alignment_bits / sizeof_bits<ElementA>::value;
      implementable &= get<2>(args.dA) % min_batch_aligned_elements_A == 0;
      constexpr int min_batch_aligned_elements_B =
          batch_alignment_bits / sizeof_bits<ElementB>::value;
      implementable &= get<2>(args.dB) % min_batch_aligned_elements_B == 0;
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST(
          "  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment "
          "requirements for XE 2D copy.\n");
    }

    return implementable;
  }

  /// Perform a subgroup-scoped matrix multiply-accumulate
  template <
      class FrgTensorD,
      class TensorA,
      class TensorB,
      class FrgTensorC,
      class KTileIterator,
      class BlkCoord>
  CUTLASS_DEVICE void operator()(
      FrgTensorD& accum,
      TensorA gA,
      TensorB gB,
      FrgTensorC const& src_accum,
      KTileIterator k_tile_iter,
      int k_tile_count,
      BlkCoord const& blk_coord,
      int const& K_start,
      int thread_idx,
      Params const& mainloop) {
    static_assert(
        is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(
        is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    // Partition the copying of A and B tiles across the threads
    (void)blk_coord;
    auto batch_idx = get<3>(blk_coord);
    auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(
        TiledMma{}, mainloop.mA_mkl(_, _, batch_idx));
    auto copy_b = get_block_2d_copy_B<GmemTiledCopyB>(
        TiledMma{}, mainloop.mB_nkl(_, _, batch_idx));

    auto thr_copy_a = copy_a.get_slice(thread_idx);
    auto thr_copy_b = copy_b.get_slice(thread_idx);

    // Instantiate the MMA object and get thread slice
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thread_idx);

    /* Register fragments for MMA */
    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

    /* Register fragments for copies */
    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

    /* Partition global tensor (proxies) for copies */
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);

    /* Create prefetch TiledCopy instances */
    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);

    auto thr_prefetch_A = prefetch_a.get_slice(thread_idx);
    auto thr_prefetch_B = prefetch_b.get_slice(thread_idx);

    /* Partition global tensor (proxies) for prefetch */
    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    auto [m_idx, n_idx, k_idx, l_idx] = blk_coord;
    const int m_coord = m_idx * BLK_M + (get_sub_group_id() / SG_NUMS_N) * SG_M;
    const int n_coord = n_idx * BLK_N + (get_sub_group_id() % SG_NUMS_N) * SG_N;
    const int l_coord = l_idx;

#define LOG_THREAD 0
#define LOG_GROUP 0
#define CUTLASS_ENABLE_DEBUG_PRINTS 0

#if CUTLASS_ENABLE_DEBUG_PRINTS
  #define PRINT(x)  \
    print(#x ": "); \
    print(x);       \
    print("\n");
    if (cute::thread(LOG_THREAD, LOG_GROUP)) {
      PRINT(BLK_M);
      PRINT(BLK_N);
      PRINT(BLK_K);

      PRINT(SG_NUMS_M);
      PRINT(SG_NUMS_N);
      PRINT(SG_NUMS_K);

      PRINT(MMA_M);
      PRINT(MMA_N);
      PRINT(MMA_K);

      PRINT(SG_M);
      PRINT(SG_N);
      PRINT(SG_K);

      PRINT(blk_coord);
      PRINT(thread_idx);

      print("======================= A: \n");
      PRINT(gA);
      PRINT(tAgA);

      PRINT(tCrA);
      PRINT(tArA);

      print("======================= B: \n");
      PRINT(gB);
      PRINT(tBgB);

      PRINT(tCrB);
      PRINT(tBrB);
    }
  #undef PRINT
#endif

    using mma_M = Int<decltype(size<1>(tCrA.shape()))::value>;
    using mma_N = Int<decltype(size<1>(tCrB.shape()))::value>;
    using mma_K = Int<decltype(size<2>(tCrB.shape()))::value>;

    auto gemm_m_offsets = make_tensor<uint8_t>(
        Layout<Shape<_1, mma_M, _1>, Stride<_0, _1, _0>>{});
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < mma_M::value; ++m) {
      gemm_m_offsets(m) =
          sizeof_bits_v<ElementA> < 8 ? (m / 2) * 32 + (m % 2) * 8 : m * 8;
    }

    auto gemm_n_offsets = make_tensor<uint8_t>(
        Layout<Shape<_1, mma_N, _1>, Stride<_0, _1, _0>>{});
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < mma_N::value; ++n) {
      gemm_n_offsets(n) = sizeof_bits_v<ElementB> < 8 ? n * 32 : n * 16;
    }

    constexpr SPIRVScope barrier_scope = ScopeWorkgroup;

    constexpr int k_reload_factor = cute::max(GROUP_K / BLK_K, 1);

    const int k_start_idx = crd2idx((*k_tile_iter), make_shape(K_start));
    int prefetch_k = k_start_idx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < DispatchPolicy::Stages; i++, prefetch_k++) {
      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
      prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
    }
    //
    // Mainloop
    //
    constexpr int acc_m0 = decltype(size<0>(accum))::value;
    constexpr int acc_m1 = decltype(size<1>(accum))::value;
    constexpr int sg_m_rows = acc_m0 * acc_m1;
    for (int k_tile = k_start_idx; k_tile < k_tile_count + k_start_idx;
         k_tile++, prefetch_k++) {
      barrier_arrive(barrier_scope);

      copy(copy_a, tAgA(_, _, _, k_tile), tArA);
      copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

      if (prefetch_k < k_tile_count) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
      }

      reorder(tArA, tCrA);
      reorder(tBrB, tCrB);

      if constexpr (DispatchPolicy::PerTensor) {
        cute::gemm(tiled_mma, tCrA, tCrB, accum);
      } else {
        Tensor scaler = make_tensor_like(accum);
        int M = shape<0>(mainloop.mA_mkl);
        int K = shape<1>(mainloop.mA_mkl);
        int K_groups = cute::ceil_div(K, GROUP_K);
        int n_scale_idx = int(n_coord / 128);
        int k_group_idx = int(k_tile * SG_K / GROUP_K);
        float scaleB = mainloop.ptr_SB[n_scale_idx * K_groups + k_group_idx];
        float combined_scale[sg_m_rows];
        CUTLASS_PRAGMA_UNROLL
        for (int i1 = 0; i1 < acc_m1; ++i1) {
          CUTLASS_PRAGMA_UNROLL
          for (int i0 = 0; i0 < acc_m0; ++i0) {
            int row = i1 * acc_m0 + i0;
            combined_scale[row] =
                mainloop.ptr_SA[(m_coord + row) * K_groups + k_group_idx] *
                scaleB;
          }
        }

        cute::gemm(tiled_mma, tCrA, tCrB, scaler);

        CUTLASS_PRAGMA_UNROLL
        for (int i0 = 0; i0 < acc_m0; ++i0) {
          CUTLASS_PRAGMA_UNROLL
          for (int i1 = 0; i1 < acc_m1; ++i1) {
            int row = i1 * acc_m0 + i0;
            CUTLASS_PRAGMA_UNROLL
            for (int i2 = 0; i2 < size<2>(accum); ++i2) {
              accum[make_coord(i0, i1, i2)] +=
                  scaler[make_coord(i0, i1, i2)] * combined_scale[row];
            }
          }
        }

        clear(scaler);
      }

      barrier_wait(barrier_scope);
    }

    if constexpr (DispatchPolicy::PerTensor) {
      float scaleAB = mainloop.ptr_SA[0] * mainloop.ptr_SB[0];
      CUTLASS_PRAGMA_UNROLL
      for (int i0 = 0; i0 < acc_m0; ++i0) {
        CUTLASS_PRAGMA_UNROLL
        for (int i1 = 0; i1 < acc_m1; ++i1) {
          CUTLASS_PRAGMA_UNROLL
          for (int i2 = 0; i2 < size<2>(accum); ++i2) {
            accum[make_coord(i0, i1, i2)] *= scaleAB;
          }
        }
      }
    }
  }
};

}  // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
