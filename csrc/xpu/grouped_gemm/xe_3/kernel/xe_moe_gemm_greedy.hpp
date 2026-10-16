// SPDX-License-Identifier: Apache-2.0
//
// Xe3 greedy MoE grouped GEMM kernel.
//
// Code structure mirrors the SYCL-TLA MoE grouped GEMM benchmark:
//
//   runner/greedy_dispatch.hpp    -- launch API shared with the torch wrapper
//   runner/moe_types.hpp          -- dtype configuration, plain type bags
//   runner/greedy_<dtype>.cpp     -- one explicit instantiation per dtype
//   kernel/xe_moe_greedy_plan.hpp -- per-expert M split, buckets, schedulers
//   kernel/xe_moe_gemm_greedy.hpp -- tile policies, tile dispatch, launcher
//   mainloop/xe_moe_mainloop.hpp  -- CuTe mainloops + epilogue
//
// A workgroup is persistent: it builds the tile plan in SLM, then pulls
// (expert, M-slice, N-tile) work items from a scheduler and runs one CuTe GEMM
// per item. Every bucket shares the same 512-thread workgroup so a single
// nd_range can host all tile shapes.
#pragma once

#include <c10/util/Exception.h>

#include "../runner/greedy_dispatch.hpp"
#include "../mainloop/xe_moe_mainloop.hpp"
#include "../mainloop/xe_moe_scale_copy.hpp"
#include "../runner/moe_types.hpp"
#include "xe_moe_greedy_plan.hpp"

namespace gpu::cutlass_kernel::grouped_gemm {

// How a tile reaches global memory / consumes its scales:
//   Native        -- alignment-safe scale copies, bias supported
//   NativeNoBias  -- same, statically known to have no bias
//   CheckAlignment-- pick the benchmark pipeline per tile when scales allow it
//   Aligned       -- always the benchmark pipeline
//   Uniform       -- benchmark pipeline only for large uniform-expert launches
enum class GreedyStore {
  Native,
  CheckAlignment,
  Aligned,
  Uniform,
  NativeNoBias
};

template <GreedyDtype>
struct GreedyTraits;

template <class Config, int TileK_, int MinK_, int MidMax_>
struct GreedyFamily {
  using Base = Config;
  static constexpr int TileK = TileK_, MinK = MinK_, MidMax = MidMax_;
  static constexpr int NormalN = 512, MidN = 256;
  static constexpr bool AllowMid384 = true, SplitTinyM = false;
};

// Per-bucket tile configuration: workgroup tile, subgroup grid and the derived
// TiledMMA / scale copy atoms. This is the only place tile geometry is defined.
template <GreedyDtype Dtype, int M, int N, int K, bool Tiny>
struct GreedyTilePolicy {
  using Base = typename GreedyTraits<Dtype>::Base;
  static constexpr bool MX = Base::ScaleKind == MoEScaleKind::MXBlock;
  static constexpr int SGM =
      Tiny ? ((N < 512 || (GreedyTraits<Dtype>::SplitTinyM && M >= 32)) ? 2 : 1)
           : 4;
  static constexpr int SGN = 32 / SGM;
  // W4A8 uses BF16 BDPAS: the pinned inner-K scale indexing only covers
  // one 32-element scale block per tile when MMA_K is 16.
  static_assert(Dtype != GreedyDtype::W4A8 || K == 32);
  static_assert(Dtype != GreedyDtype::FP8Block || 128 % K == 0);
  static_assert(
      Dtype != GreedyDtype::FP8Block || 128 % (N / SGN) == 0,
      "A block-FP8 subgroup must not cross a B-scale block");
  static constexpr int TileM = M;
  static constexpr bool TinyBucket = Tiny;
  using TileShape = Shape<Int<M>, Int<N>, Int<K>>;
  using SGLayout =
      Layout<Shape<Int<SGM>, Int<SGN>, _1>, Stride<Int<SGN>, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      typename Base::MmaAtom,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  using ScaleCopyA =
      cute::MoEGreedyScaleLoad<(K + 31) / 32, (M / SGM <= 32 ? 32 : 64)>;
  // Keep W4A8 weight scales in a full GRF for both N16 MMA steps.
  using ScaleCopyB = cute::MoEGreedyScaleLoad<
      (K + 31) / 32,
      (Dtype == GreedyDtype::W4A8 || N / SGN > 32 ? 64 : 32)>;
  static_assert(
      size(TiledMma{}) == 512,
      "All greedy buckets must use the same physical workgroup");
};

// Run one (expert slice, N tile) GEMM with CuTe.
//
// All pointers are advanced to the tile's expert and first row here, so the
// mainloop always starts at M row 0 and only the N tile index varies.
template <class Policy, GreedyStore Direct>
CUTLASS_DEVICE void greedy_tile(
    GreedyArguments const& args,
    greedy::Work const& work,
    bool use_direct = true) {
  using Cfg = typename Policy::Base;
  using ElementA = typename Cfg::ElementA;
  using ElementB = typename Cfg::ElementB;
  using ElementOutput = typename Cfg::ElementOutput;
  using ElementBias = typename Cfg::ElementBias;
  using TiledMma = typename Policy::TiledMma;
  // Bias is only ever requested on the alignment-safe path.
  constexpr bool MaybeBias = Direct == GreedyStore::Native;
  // The block-scaled gather pipeline keeps the collective's per-K-tile
  // workgroup barriers, which hold the subgroups' 2D block loads together and
  // are worth several percent on every decode and mid-size tile. The largest
  // Mid buckets are the exception: their K loops are long enough to hide the
  // loads on their own, and there the barrier-free pipeline measures faster.
  constexpr bool Barrier =
      !Cfg::WideTileBarrierFree || Policy::TinyBucket || Policy::TileM < 192;

  const int m = work.rows - work.tile.offset;
  const int n = args.n;
  const int k = args.k;
  const int64_t row = work.row_offset + work.tile.offset;
  const int64_t expert = work.expert;

  // Sub-byte operands are packed two per byte.
  constexpr int64_t kPackA = cute::sizeof_bits_v<ElementA> == 4 ? 2 : 1;
  constexpr int64_t kPackB = cute::sizeof_bits_v<ElementB> == 4 ? 2 : 1;

  auto mA_mkl = make_tensor(
      make_gmem_ptr(static_cast<ElementA const*>(args.a) + row * k / kPackA),
      make_layout(
          make_shape(m, k, 1),
          cutlass::make_cute_packed_stride(
              typename Cfg::StrideA{}, cute::make_shape(m, k, 1))));
  auto mB_nkl = make_tensor(
      make_gmem_ptr(
          static_cast<ElementB const*>(args.b) + expert * n * k / kPackB),
      make_layout(
          make_shape(n, k, 1),
          cutlass::make_cute_packed_stride(
              typename Cfg::StrideB{}, cute::make_shape(n, k, 1))));
  auto mD_mnl = make_tensor(
      make_gmem_ptr(static_cast<ElementOutput*>(args.d) + row * n),
      make_layout(
          make_shape(m, n, 1),
          cutlass::make_cute_packed_stride(
              typename Cfg::StrideD{}, cute::make_shape(m, n, 1))));
  auto mC_mnl = make_tensor(
      make_gmem_ptr(
          static_cast<ElementBias const*>(args.bias) +
          (args.bias ? row * n : 0)),
      make_layout(
          make_shape(m, n, 1),
          cutlass::make_cute_packed_stride(
              typename Cfg::StrideC{}, cute::make_shape(m, n, 1))));
  auto mA = mA_mkl(_, _, 0);
  auto mB = mB_nkl(_, _, 0);
  auto mD = mD_mnl(_, _, 0);
  auto mC = mC_mnl(_, _, 0);
  const bool has_bias = MaybeBias && args.bias != nullptr;

  if constexpr (Cfg::ScaleKind == MoEScaleKind::MXBlock) {
    using ElementScaleA = typename Cfg::ElementScaleA;
    using ElementScaleB = typename Cfg::ElementScaleB;
    const int64_t scale_k =
        (int64_t(k) + Cfg::ScaleBlock - 1) / Cfg::ScaleBlock;
    // The A scales are MN-major with each expert's rows padded to a multiple of
    // 4. A tile slice moves the scale origin, not the expert's pitch.
    const int64_t pitch = greedy::padded_rows(work.rows);
    auto mSA = make_tensor(
        make_gmem_ptr(
            static_cast<ElementScaleA const*>(args.sa) +
            work.scale_offset * scale_k + work.tile.offset),
        make_layout(
            make_shape(m, int(scale_k), 1),
            make_stride(_1{}, pitch, pitch * scale_k)));
    auto mSB = make_tensor(
        make_gmem_ptr(
            static_cast<ElementScaleB const*>(args.sb) + expert * n * scale_k),
        make_layout(
            make_shape(n, int(scale_k), 1),
            make_stride(_1{}, int64_t(n), int64_t(n) * scale_k)));

    if constexpr (
        Direct != GreedyStore::Native && Direct != GreedyStore::NativeNoBias) {
      // The benchmark scale pipeline requires genuinely aligned scale bases;
      // padded4 expert segments can land mid-cache-line, so fall back to the
      // gather-capable copies when they do.
      if (use_direct &&
          (Direct == GreedyStore::Aligned ||
           ((reinterpret_cast<uintptr_t>(raw_pointer_cast(mSA.data())) & 63) ==
                0 &&
            (reinterpret_cast<uintptr_t>(raw_pointer_cast(mSB.data())) & 63) ==
                0))) {
        moe_greedy_gemm<Cfg, true, false, true, void, void>(
            mA, mB, mSA, mSB, mC, mD, work.n_tile, TiledMma{}, false);
        return;
      }
    }
    if constexpr (Direct != GreedyStore::Aligned) {
      moe_greedy_gemm<
          Cfg,
          false,
          MaybeBias,
          Barrier,
          typename Policy::ScaleCopyA,
          typename Policy::ScaleCopyB>(
          mA, mB, mSA, mSB, mC, mD, work.n_tile, TiledMma{}, has_bias);
    }
  } else if constexpr (Cfg::ScaleKind == MoEScaleKind::Fp8Block) {
    const int64_t scale_k = (int64_t(k) + 127) / 128;
    const int64_t scale_n = (int64_t(n) + 127) / 128;
    auto ptr_SA = static_cast<float const*>(args.sa) + row * scale_k;
    auto ptr_SB =
        static_cast<float const*>(args.sb) + expert * scale_n * scale_k;
    moe_greedy_gemm<Cfg, false, MaybeBias, true, void, void>(
        mA, mB, ptr_SA, ptr_SB, mC, mD, work.n_tile, TiledMma{}, has_bias);
  } else if constexpr (Cfg::ScaleKind == MoEScaleKind::Fp8Tensor) {
    // One global scalar for A, one scalar per expert for B.
    auto ptr_SA = static_cast<float const*>(args.sa);
    auto ptr_SB = static_cast<float const*>(args.sb) + expert;
    moe_greedy_gemm<Cfg, false, MaybeBias, true, void, void>(
        mA, mB, ptr_SA, ptr_SB, mC, mD, work.n_tile, TiledMma{}, has_bias);
  } else {
    moe_greedy_gemm<Cfg, false, MaybeBias, true, void, void>(
        mA,
        mB,
        static_cast<float const*>(nullptr),
        static_cast<float const*>(nullptr),
        mC,
        mD,
        work.n_tile,
        TiledMma{},
        has_bias);
  }
}

template <GreedyDtype Dtype, int MidN, int TinyK, GreedyStore Direct>
class GreedyKernel;

template <
    GreedyDtype Dtype,
    int MidN,
    int TinyK,
    GreedyStore Direct = GreedyStore::Native>
void launch_greedy_config(sycl::queue& stream, GreedyArguments args) {
  constexpr int K = GreedyTraits<Dtype>::TileK;
  constexpr bool OptimizeUniform =
      (Dtype == GreedyDtype::MXFP8 || Dtype == GreedyDtype::MXFP4) &&
      Direct != GreedyStore::Native;
  constexpr GreedyStore TinyDirect =
      Direct == GreedyStore::Uniform ? GreedyStore::Native : Direct;
  constexpr int NormalN = GreedyTraits<Dtype>::NormalN;
  using T8 = GreedyTilePolicy<Dtype, 8, 512, TinyK, true>;
  using T32 = GreedyTilePolicy<Dtype, 32, 512, TinyK, true>;
  using T64 = GreedyTilePolicy<Dtype, 64, 512, TinyK, true>;
  using T128 = GreedyTilePolicy<Dtype, 128, 512, TinyK, true>;
  using N32 = GreedyTilePolicy<Dtype, 32, NormalN, K, true>;
  using N192 = GreedyTilePolicy<Dtype, 192, NormalN, K, false>;
  using N256 = GreedyTilePolicy<Dtype, 256, NormalN, K, false>;
  using M192 = GreedyTilePolicy<Dtype, 192, MidN, K, false>;
  using M256 = GreedyTilePolicy<Dtype, 256, MidN, K, false>;
  using M320 = GreedyTilePolicy<Dtype, 320, MidN, K, false>;
  greedy::MidWindow window =
      args.k >= GreedyTraits<Dtype>::MinK
          ? greedy::MidWindow{128, GreedyTraits<Dtype>::MidMax}
          : greedy::MidWindow{};
  if constexpr (
      Dtype == GreedyDtype::BF16 || Dtype == GreedyDtype::MXFP8 ||
      Dtype == GreedyDtype::MXFP4 || Dtype == GreedyDtype::FP8Tensor) {
    // Retain the SYCL-TLA window for the previously tuned 1536x1536 cell.
    if (args.n == 1536 && args.k == 1536) window = {128, 576};
  }
  const auto uniform_window = [&] {
    if constexpr (OptimizeUniform)
      return GreedyTraits<Dtype>::uniform_mid_window(args.n, args.k, window);
    else
      return window;
  }();
  const auto device = stream.get_device();
  const int workers =
      device.get_info<sycl::ext::intel::info::device::gpu_slices>() *
      device
          .get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
  TORCH_CHECK(workers > 0, "Cannot determine the Xe3 greedy persistent grid");
  const bool fast_decode = greedy::is_decode_shape(args.total_rows);
  // The CuTe epilogue keeps everything in registers, so SLM holds the tile plan
  // and nothing else.
  const size_t shared_bytes = fast_decode
                                  ? greedy::decode_shared_bytes(args.groups)
                                  : greedy::plan_bytes(args.groups);
  namespace oneapi = sycl::ext::oneapi::experimental;
  namespace intel = sycl::ext::intel::experimental;
  const oneapi::properties properties{
      oneapi::sub_group_size<16>, intel::grf_size<VLLM_GRF_SIZE>};
  stream.submit([&](sycl::handler& handler) {
    sycl::local_accessor<char, 1> storage(
        sycl::range<1>(shared_bytes), handler);
    handler.parallel_for<GreedyKernel<Dtype, MidN, TinyK, Direct>>(
        sycl::nd_range<3>({1, 1, size_t(workers) * 512}, {1, 1, 512}),
        properties,
        [=](sycl::nd_item<3> item) {
          const auto subgroup = item.get_sub_group();
          char* shared =
              storage.template get_multi_ptr<sycl::access::decorated::no>()
                  .get();
          // Decode uses a scheduler that knows every expert is Tiny, so it
          // needs neither the tile prefix nor the search structure the general
          // path builds.
          if (fast_decode) {
            int* records = reinterpret_cast<int*>(
                shared + greedy::decode_records_offset(args.groups));
            greedy::build_decode_plan(
                greedy::plan_storage(shared, args.groups),
                records,
                args.counts,
                args.groups,
                item);
            greedy::DecodeScheduler scheduler(
                records,
                args.groups,
                args.n,
                int(item.get_group_linear_id()),
                workers);
            for (auto work = scheduler.next(); work.valid();
                 work = scheduler.next()) {
              switch (work.tile.bucket) {
                case greedy::Bucket::Tiny8:
                  greedy_tile<T8, TinyDirect>(args, work);
                  break;
                case greedy::Bucket::Tiny32:
                  greedy_tile<T32, TinyDirect>(args, work);
                  break;
                case greedy::Bucket::Tiny64:
                  greedy_tile<T64, TinyDirect>(args, work);
                  break;
                default:
                  greedy_tile<T128, TinyDirect>(args, work);
                  break;
              }
            }
            return;
          }
          greedy::PlanStorage plan = greedy::plan_storage(shared, args.groups);
          greedy::build_plan<MidN, NormalN>(
              plan, args.counts, args.groups, args.n, window, item);
          const int* counts = args.counts;
          const int first = counts[0];
          bool same = true;
          for (int64_t expert = subgroup.get_local_linear_id();
               expert < args.groups;
               expert += 16) {
            same &= counts[expert] == first;
          }
          const bool uniform = sycl::all_of_group(subgroup, same);
          greedy::Scheduler<MidN, NormalN, OptimizeUniform> scheduler(
              plan,
              counts,
              args.groups,
              args.n,
              uniform && first > 128 ? uniform_window : window,
              int(item.get_group_linear_id()),
              workers,
              uniform);
          const bool use_direct =
              Direct != GreedyStore::Uniform || (uniform && first > 128);
          for (auto work = scheduler.next(); work.valid();
               work = scheduler.next()) {
            switch (work.tile.bucket) {
              case greedy::Bucket::Tiny8:
                greedy_tile<T8, TinyDirect>(args, work);
                break;
              case greedy::Bucket::Tiny32:
                greedy_tile<T32, TinyDirect>(args, work);
                break;
              case greedy::Bucket::Tiny64:
                greedy_tile<T64, TinyDirect>(args, work);
                break;
              case greedy::Bucket::Tiny128:
                greedy_tile<T128, TinyDirect>(args, work);
                break;
              case greedy::Bucket::Normal32:
                greedy_tile<N32, Direct>(args, work, use_direct);
                break;
              case greedy::Bucket::Normal192:
                greedy_tile<N192, Direct>(args, work, use_direct);
                break;
              case greedy::Bucket::Normal256:
                greedy_tile<N256, Direct>(args, work, use_direct);
                break;
              case greedy::Bucket::Mid192:
                greedy_tile<M192, Direct>(args, work, use_direct);
                break;
              case greedy::Bucket::Mid256:
                greedy_tile<M256, Direct>(args, work, use_direct);
                break;
              case greedy::Bucket::Mid320:
                greedy_tile<M320, Direct>(args, work, use_direct);
                break;
            }
          }
        });
  });
}

template <GreedyDtype Dtype>
void launch_greedy(sycl::queue& stream, GreedyArguments args) {
  constexpr int K = GreedyTraits<Dtype>::TileK;
  constexpr int MidN = GreedyTraits<Dtype>::MidN;
  constexpr bool AllowMid384 = GreedyTraits<Dtype>::AllowMid384;
  const bool mid384 = (args.n == 1536 || args.n == 3072) && AllowMid384;
  if constexpr (Dtype == GreedyDtype::MXFP8) {
    if (!args.bias) {
      // Padded4 prefixes and all greedy M origins preserve alignment when
      // scale_K is divisible by 16. Compile out the fallback in that case.
      const bool aligned = (((int64_t(args.k) + 31) / 32) % 16 == 0) &&
                           args.n % 64 == 0 &&
                           (reinterpret_cast<uintptr_t>(args.sa) & 63) == 0 &&
                           (reinterpret_cast<uintptr_t>(args.sb) & 63) == 0;
      if (aligned) {
        if (mid384)
          launch_greedy_config<Dtype, 384, K, GreedyStore::Aligned>(
              stream, args);
        else
          launch_greedy_config<Dtype, 256, K, GreedyStore::Aligned>(
              stream, args);
        return;
      }
      if (!mid384) {
        launch_greedy_config<Dtype, 256, K, GreedyStore::CheckAlignment>(
            stream, args);
        return;
      }
      // Mixing both MidN384 mainloops introduces private scratch storage.
      launch_greedy_config<Dtype, 384, K, GreedyStore::NativeNoBias>(
          stream, args);
      return;
    }
  }
  if constexpr (Dtype == GreedyDtype::MXFP4) {
    if (!args.bias &&
        greedy::may_have_large_uniform_experts(args.total_rows, args.groups)) {
      // Match the benchmark mainloop/store only for large uniform experts;
      // retain the alignment-safe path for small, nonuniform, or unaligned
      // jobs.
      launch_greedy_config<Dtype, MidN, K, GreedyStore::Uniform>(stream, args);
      return;
    }
  }
  if constexpr (AllowMid384) {
    if (mid384) {
      launch_greedy_config<Dtype, 384, K>(stream, args);
      return;
    }
  }
  launch_greedy_config<Dtype, MidN, K>(stream, args);
}

}  // namespace gpu::cutlass_kernel::grouped_gemm
