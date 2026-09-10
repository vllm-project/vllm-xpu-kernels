#pragma once
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include <cfloat>

#include "moe_array_mma.hpp"
#include "moe_array_mma_mxfp.hpp"
#include "moe_array_mma_fp8block.hpp"
#include "moe_array_epilogue.hpp"
#include "moe_callbacks.hpp"
#include "moe_gemm_array_cooperative.hpp"
#include "moe_tile_scheduler.hpp"

using namespace cute;
using ProblemShape =
    cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per
                                                             // group

namespace gpu::cutlass_kernel {
namespace grouped_gemm {

template <
    class ElementA,
    class ElementScaleA,
    class ElementB,
    class ElementScaleB,
    class ElementAccumulator,
    class ElementOutput,
    class LayoutA,
    class LayoutB,
    class LayoutC,
    class LayoutD,
    class StrideScale,
    class TiledMma,
    class TileShape,
    class GEMMDispatchPolicy,
    class EpilogueDispatchPolicy,
    class EpilogueOp,
    class GmemTiledCopyA,
    class GmemTiledCopyScaleA,
    class GmemTiledCopyB,
    class GmemTiledCopyScaleB,
    bool NeedScale>
class GenerateGemm {
  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy,
      EpilogueOp,
      TileShape,
      decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      void,
      ElementAccumulator,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      FusionCallBacks,
      void,
      void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      std::conditional_t<
          NeedScale,
          cute::tuple<ElementA, ElementScaleA>,
          ElementA>,
      std::conditional_t<
          NeedScale,
          cute::tuple<cutlass::gemm::TagToStrideA_t<LayoutA>, StrideScale>,
          cutlass::gemm::TagToStrideA_t<LayoutA>>,
      std::conditional_t<
          NeedScale,
          cute::tuple<ElementB, ElementScaleB>,
          ElementB>,
      std::conditional_t<
          NeedScale,
          cute::tuple<cutlass::gemm::TagToStrideB_t<LayoutB>, StrideScale>,
          cutlass::gemm::TagToStrideB_t<LayoutB>>,
      TiledMma,
      std::conditional_t<
          NeedScale,
          cute::tuple<GmemTiledCopyA, GmemTiledCopyScaleA>,
          GmemTiledCopyA>,
      void,
      void,
      cute::identity,  // A
      std::conditional_t<
          NeedScale,
          cute::tuple<GmemTiledCopyB, GmemTiledCopyScaleB>,
          GmemTiledCopyB>,
      void,
      void,
      cute::identity  // B
      >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::GroupScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

#define CALL_GENERATE_GEMM()  \
  using Gemm = GenerateGemm<  \
      ElementA,               \
      ElementScaleA,          \
      ElementB,               \
      ElementScaleB,          \
      ElementAccumulator,     \
      ElementOutput,          \
      LayoutA,                \
      LayoutB,                \
      LayoutC,                \
      LayoutD,                \
      StrideScale,            \
      TiledMma,               \
      TileShape,              \
      GEMMDispatchPolicy,     \
      EpilogueDispatchPolicy, \
      EpilogueOp,             \
      GmemTiledCopyA,         \
      GmemTiledCopyScaleA,    \
      GmemTiledCopyB,         \
      GmemTiledCopyScaleB,    \
      NeedScale>::Gemm;

class moe_policy_base {
 public:
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementA = float;
  using ElementB = float;
  using ElementOutput = float;
  using ElementScaleA = void;
  using ElementScaleB = void;
  using StrideScale = void;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  using GmemTiledCopyScaleA = void;
  using GmemTiledCopyScaleB = void;

  static constexpr int PipelineStages = 4;
  using EpilogueDispatchPolicy = cutlass::epilogue::MoE16Group;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      float_t,
      ElementComputeEpilogue,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;
  static constexpr bool NeedScale = false;
  static constexpr int BlockSize = -1;
};

// BF16 / FP16 policies

class moe_bf16_policy : public moe_policy_base {
 public:
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementOutput = cutlass::bfloat16_t;

  using TileShape = Shape<_256, _512, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopMoE16Group<PipelineStages>;
  CALL_GENERATE_GEMM();
};

// BF16 tile variants selected by pick_bf16_variant() for wave-quantization.
class moe_bf16_256x128_policy : public moe_bf16_policy {
 public:
  using TileShape = Shape<_256, _128, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  CALL_GENERATE_GEMM();
};

class moe_bf16_128x256_policy : public moe_bf16_policy {
 public:
  using TileShape = Shape<_128, _256, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  CALL_GENERATE_GEMM();
};

class moe_bf16_128x128_policy : public moe_bf16_policy {
 public:
  using TileShape = Shape<_128, _128, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  CALL_GENERATE_GEMM();
};

class moe_fp16_policy : public moe_policy_base {
 public:
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;

  using TileShape = Shape<_256, _256, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopMoE16Group<PipelineStages>;
  CALL_GENERATE_GEMM();
};

class moe_bf16_mid_policy : public moe_bf16_policy {
 public:
  using TileShape = Shape<_128, _128, _32>;
  using SGLayout = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  CALL_GENERATE_GEMM();
};

class moe_fp16_mid_policy : public moe_fp16_policy {
 public:
  using TileShape = Shape<_128, _128, _32>;
  using SGLayout = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  CALL_GENERATE_GEMM();
};

// Decode regime (avg M per expert <= 4, total_M tiny): the case is memory-bound
// on the B-weight load and the only parallelism is groups * N-tiles. Use a
// small (8,64,32) tile so the N dimension is carved into many narrow tiles,
// maximizing the number of concurrent workgroups to saturate DRAM. Widening
// BLK_N (e.g. 256) or BLK_M starves occupancy and regresses bandwidth;
// shrinking both is the win. Decode, short contraction (e.g. down_proj K=768):
// widen BLK_N 64 -> 128 so the E=2 launch collapses to a single 32-core wave
// (N*E/128 <= 32 tiles), and lift the prefetch pipeline 2 -> 4 stages to hide
// weight-DRAM latency. BLK_K stays 32 because k64 regresses at short K (see k64
// policy below).
class moe_bf16_decode_policy : public moe_bf16_policy {
 public:
  using TileShape = Shape<_8, _128, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  static constexpr int PipelineStages = 4;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopMoE16Group<PipelineStages>;
  CALL_GENERATE_GEMM();
};

// Decode with a long contraction (e.g. gate_up_proj K=2048): widen BLK_N 64 ->
// 128 (single 32-core wave for E=2) and BLK_K 32 -> 64 so the B 2D-block loads
// become larger contiguous DRAM bursts and the K-loop trip count halves, plus a
// 4-stage prefetch pipeline to hide the long-K latency. Register footprint
// stays tiny. Only profitable when K is large enough to fill the pipeline;
// short-K decode (down_proj K=768) regresses, so dispatch selects this only for
// K >= 1024.
class moe_bf16_decode_k64_policy : public moe_bf16_policy {
 public:
  using TileShape = Shape<_8, _128, _64>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  static constexpr int PipelineStages = 4;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopMoE16Group<PipelineStages>;
  CALL_GENERATE_GEMM();
};

class moe_fp16_decode_policy : public moe_fp16_policy {
 public:
  using TileShape = Shape<_8, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  CALL_GENERATE_GEMM();
};

// MXFP8
class moe_mxfp8_policy : public moe_policy_base {
 public:
  static constexpr bool NeedScale = true;
  static constexpr int BlockSize = 32;
  using ElementType = cutlass::mx_float8_t<float_e4m3_t>;
  using MmaType = typename ElementType::DataType;

  using ElementA = typename ElementType::DataType;
  using ElementB = typename ElementType::DataType;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementScaleA = typename ElementType::ScaleFactorType;
  using ElementScaleB = typename ElementType::ScaleFactorType;
  using StrideScale = cute::Stride<_1, int64_t, int64_t>;
  using TileShape = Shape<_256, _512, _64>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_BDPAS_TT<8, float, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;

  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopMXFPXGroup<PipelineStages>;
  CALL_GENERATE_GEMM();
};

// MXFP8 prefill tile variants selected by pick_prefill_tile() to keep the
// 32 Xe cores fully packed (wave quantization). Mirrors the bf16 variants.
class moe_mxfp8_256x128_policy : public moe_mxfp8_policy {
 public:
  using TileShape = Shape<_256, _128, _64>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_BDPAS_TT<8, float, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  CALL_GENERATE_GEMM();
};

class moe_mxfp8_128x256_policy : public moe_mxfp8_policy {
 public:
  using TileShape = Shape<_128, _256, _64>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_BDPAS_TT<8, float, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  CALL_GENERATE_GEMM();
};

class moe_mxfp8_128x128_policy : public moe_mxfp8_policy {
 public:
  using TileShape = Shape<_128, _128, _64>;
  using SGLayout = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_BDPAS_TT<8, float, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  CALL_GENERATE_GEMM();
};

class moe_mxfp8_mid_policy : public moe_mxfp8_policy {
 public:
  using TileShape = Shape<_128, _128, _64>;
  using SGLayout = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_BDPAS_TT<8, float, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  CALL_GENERATE_GEMM();
};

// Decode regime (avg M per expert <= 4): the mid tile's BLK_M=128 makes the
// systolic array grind 128 masked rows for a single real token, so keep BLK_M=8
// (the DPAS atom minimum) to stop wasting passes on the few real rows. The
// production decode shapes are *not* tiny -- gate_up is N=1536,K=2048 and
// down_proj is N=2048,K=768 -- i.e. a long, weight-DRAM-bound GEMV. Tuning for
// those: BLK_N=128 keeps E=2 to a single 32-core wave (N*E/128 <= 32 tiles) and
// BLK_K=64 makes the B 2D-block loads 8 KB contiguous bursts instead of 2 KB,
// which together lifted gate_up from ~429 to ~524 GB/s. A 4-stage prefetch
// pipeline (vs 2) hides the 64-k-tile DRAM latency; deeper gives no further
// win.
class moe_mxfp8_decode_policy : public moe_mxfp8_policy {
 public:
  using TileShape = Shape<_8, _128, _64>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_BDPAS_TT<8, float, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  static constexpr int PipelineStages = 4;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopMXFPXGroup<PipelineStages>;
  CALL_GENERATE_GEMM();
};

// MXFP4
class moe_mxfp4_policy : public moe_policy_base {
 public:
  static constexpr bool NeedScale = true;
  static constexpr int BlockSize = 32;
  using ElementType = cutlass::mx_float4_t<float_e2m1_t>;
  using MmaType = typename ElementType::DataType;

  using ElementA = typename ElementType::DataType;
  using ElementB = typename ElementType::DataType;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementScaleA = typename ElementType::ScaleFactorType;
  using ElementScaleB = typename ElementType::ScaleFactorType;
  using LayoutB = cutlass::layout::ColumnMajor;

  using StrideScale = cute::Stride<_1, int64_t, int64_t>;

  using TileShape = Shape<_256, _256, _128>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_BDPAS_TT<8, float, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;

  using GEMMDispatchPolicy = cutlass::gemm::MainloopMXFPXGroup<PipelineStages>;
  CALL_GENERATE_GEMM();
};

// Short-K, wide-N policy for larger down-projection M. The wider N tile halves
// tile count and a two-stage pipeline avoids underfill in the six-iteration
// K loop.
class moe_mxfp4_downproj_wide_policy : public moe_mxfp4_policy {
 public:
  using TileShape = Shape<_256, _512, _128>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_BDPAS_TT<8, float, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopMXFPXGroup<PipelineStages>;
  CALL_GENERATE_GEMM();
};

// MXFP4 prefill tile variants selected by pick_prefill_tile().
// Tuned short-K tile with 16 subgroups and PipelineStages 2.
// The 4x4 subgroup grid preserves the 64x64 per-subgroup output of the former
// 512x256/8x4 policy while reducing the workgroup to 256 threads. This permits
// two resident workgroups per Xe core to better hide the load/DPAS pipeline.
class moe_mxfp4_256x128_policy : public moe_mxfp4_policy {
 public:
  using TileShape = Shape<_256, _256, _128>;
  using SGLayout = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_BDPAS_TT<8, float, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopMXFPXGroup<PipelineStages>;
  CALL_GENERATE_GEMM();
};

class moe_mxfp4_128x256_policy : public moe_mxfp4_policy {
 public:
  using TileShape = Shape<_128, _256, _128>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_BDPAS_TT<8, float, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  CALL_GENERATE_GEMM();
};

class moe_mxfp4_128x128_policy : public moe_mxfp4_policy {
 public:
  using TileShape = Shape<_128, _128, _128>;
  using SGLayout = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_BDPAS_TT<8, float, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  CALL_GENERATE_GEMM();
};

class moe_mxfp4_mid_policy : public moe_mxfp4_policy {
 public:
  using TileShape = Shape<_128, _128, _128>;
  using SGLayout = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_BDPAS_TT<8, float, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  CALL_GENERATE_GEMM();
};

// Decode regime (avg M per expert <= 4): same memory-bound single-token GEMV as
// mxfp8. Widen BLK_N 64 -> 128 so E=2 collapses to a single 32-core wave and
// BLK_K 32 -> 64 for larger contiguous weight bursts; the inherited 4-stage
// pipeline hides DRAM latency (Stages=2 regressed FP4 here).
class moe_mxfp4_decode_policy : public moe_mxfp4_policy {
 public:
  using TileShape = Shape<_8, _128, _64>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_BDPAS_TT<8, float, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  CALL_GENERATE_GEMM();
};

// W4A8: MXFP8 activation (A) x MXFP4 weight (B).
//
// A is mx_float8 (e4m3 data + e8m0 block scale) and B is mx_float4 (e2m1 data +
// e8m0 block scale). The Xe hardware block-scaled DPAS has no native fp8 x fp4
// op, so the mainloop computes in e4m3: A is consumed natively while the e2m1
// weight is upconverted to e4m3 by reorder() before the MMA. The upconvert is
// lossless (every e2m1 value is exactly representable in e4m3) and the per-32
// block scales are still applied independently by the block-scaled DPAS, so the
// result is bit-equivalent to a true mxfp4 weight. The native block-scaled
// mainloop requires GroupSize == 32 whenever either operand is e2m1, which
// holds here (BlockSize == 32). The mxfp4 weight is column-major like
// moe_mxfp4_policy.
class moe_w4a8_policy : public moe_policy_base {
 public:
  static constexpr bool NeedScale = true;
  static constexpr int BlockSize = 32;
  using ElementTypeA = cutlass::mx_float8_t<float_e4m3_t>;
  using ElementTypeB = cutlass::mx_float4_t<float_e2m1_t>;
  // Compute in bf16: the native block-scaled mainloop has direct optimized
  // e4m3->bf16 and e2m1->bf16 reorders (there is no direct e2m1->e4m3 reorder,
  // which sends e4m3 down a recursive conversion path the SYCL kernel cannot
  // compile). bf x bf BDPAS still consumes the e8m0 block scales in hardware.
  using MmaType = cutlass::bfloat16_t;

  using ElementA = typename ElementTypeA::DataType;
  using ElementB = typename ElementTypeB::DataType;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementScaleA = typename ElementTypeA::ScaleFactorType;
  using ElementScaleB = typename ElementTypeB::ScaleFactorType;
  using LayoutB = cutlass::layout::ColumnMajor;

  using StrideScale = cute::Stride<_1, int64_t, int64_t>;

  // bf16 BDPAS has MMA_K = 16; with a 32-element block-scale group each scale
  // spans two MMA-K steps (k_reload_factor = 2), so keep K a multiple of 32.
  using TileShape = Shape<_256, _256, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_BDPAS_TT<8, float, MmaType>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;

  using GEMMDispatchPolicy = cutlass::gemm::MainloopMXFPXGroup<PipelineStages>;
  CALL_GENERATE_GEMM();
};

class moe_w4a8_mid_policy : public moe_w4a8_policy {
 public:
  using TileShape = Shape<_128, _128, _32>;
  using SGLayout = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_BDPAS_TT<8, float, MmaType>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  CALL_GENERATE_GEMM();
};

// FP8 block-scaled
class moe_fp8block_policy : public moe_policy_base {
 public:
  static constexpr bool NeedScale = true;
  static constexpr int BlockSize = 128;
  using ElementType = cutlass::float_e4m3_t;

  using ElementA = ElementType;
  using ElementB = ElementType;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementScaleA = float;
  using ElementScaleB = float;

  using StrideScale = cute::Stride<_1, int64_t, int64_t>;

  using TileShape = Shape<_128, _128, _32>;
  using SGLayout = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;

  using GEMMDispatchPolicy =
      cutlass::gemm::MainloopFP8BlockGroup<PipelineStages>;
  CALL_GENERATE_GEMM();
};

class moe_fp8block_mid_policy : public moe_fp8block_policy {
 public:
  using TileShape = Shape<_32, _128, _32>;
  using SGLayout = Layout<Shape<_2, _8, _1>, Stride<_8, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  CALL_GENERATE_GEMM();
};

class moe_fp8block_decode_policy : public moe_fp8block_policy {
 public:
  using TileShape = Shape<_16, _128, _32>;
  using SGLayout = Layout<Shape<_1, _8, _1>, Stride<_8, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  CALL_GENERATE_GEMM();
};

// FP8 per-tensor: A scaled by a single global scalar (shape [1]), each
// expert's B by one scalar (shape [E]). Same e4m3 DPAS as the block path; only
// the scale granularity differs (selected via MainloopFP8PerTensorGroup).
class moe_fp8pertensor_policy : public moe_policy_base {
 public:
  static constexpr bool NeedScale = true;
  static constexpr int BlockSize = 128;
  using ElementType = cutlass::float_e4m3_t;

  using ElementA = ElementType;
  using ElementB = ElementType;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementScaleA = float;
  using ElementScaleB = float;

  using StrideScale = cute::Stride<_1, int64_t, int64_t>;

  using TileShape = Shape<_128, _256, _32>;
  using SGLayout = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;

  static constexpr int PipelineStages = 3;
  using GEMMDispatchPolicy =
      cutlass::gemm::MainloopFP8PerTensorGroup<PipelineStages>;
  CALL_GENERATE_GEMM();
};

class moe_fp8pertensor_mid_policy : public moe_fp8pertensor_policy {
 public:
  using TileShape = Shape<_32, _128, _32>;
  using SGLayout = Layout<Shape<_2, _8, _1>, Stride<_8, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy =
      cutlass::gemm::MainloopFP8PerTensorGroup<PipelineStages>;
  CALL_GENERATE_GEMM();
};

class moe_fp8pertensor_decode_policy : public moe_fp8pertensor_policy {
 public:
  using TileShape = Shape<_8, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy =
      cutlass::gemm::MainloopFP8PerTensorGroup<PipelineStages>;
  CALL_GENERATE_GEMM();
};

class moe_fp8pertensor_decode_lowk_policy
    : public moe_fp8pertensor_decode_policy {
 public:
  static constexpr int PipelineStages = 1;
  using GEMMDispatchPolicy =
      cutlass::gemm::MainloopFP8PerTensorGroup<PipelineStages>;
  CALL_GENERATE_GEMM();
};

class moe_fp8pertensor_decode_narrow_policy : public moe_fp8pertensor_policy {
 public:
  using TileShape = Shape<_8, _32, _64>;
  using SGLayout = Layout<Shape<_1, _2, _1>, Stride<_2, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy =
      cutlass::gemm::MainloopFP8PerTensorGroup<PipelineStages>;
  CALL_GENERATE_GEMM();
};

class moe_fp8pertensor_decode_shortk_policy : public moe_fp8pertensor_policy {
 public:
  using TileShape = Shape<_8, _128, _128>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, ElementA>>,
      Layout<TileShape>,
      SGLayout>::TiledMMA;
  static constexpr int PipelineStages = 1;
  using GEMMDispatchPolicy =
      cutlass::gemm::MainloopFP8PerTensorGroup<PipelineStages>;
  CALL_GENERATE_GEMM();
};

}  // namespace grouped_gemm
}  // namespace gpu::cutlass_kernel
