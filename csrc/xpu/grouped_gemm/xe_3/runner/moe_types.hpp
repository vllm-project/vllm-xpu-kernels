// SPDX-License-Identifier: Apache-2.0
//
// Type configuration for the Xe3 MoE grouped GEMM.
//
// Every entry here is a pure type bag: element types, tensor strides, the MMA
// atom and the pipeline depth that the CuTe mainloops in
// `mainloop/xe_moe_mainloop.hpp` consume. There is no CollectiveMma /
// CollectiveEpilogue / GemmUniversal machinery -- the kernels are written
// directly in CuTe, so a "policy" is nothing more than the data the device
// code needs.
#pragma once

#include <cute/tensor.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/float8.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

#include "../mainloop/xe_moe_store_copy.hpp"

namespace gpu::cutlass_kernel::grouped_gemm {

using namespace cute;

// How the mainloop has to fold scales into the accumulator.
enum class MoEScaleKind {
  None,       // plain 16-bit GEMM
  MXBlock,    // e8m0 block scales consumed by the block-scaled DPAS
  Fp8Block,   // fp32 128x128 block scales folded in the K loop
  Fp8Tensor,  // one fp32 scalar per operand, folded once after the K loop
};

// Shared defaults. A/C/D are row-major (M,K)/(M,N); B is (N,K) dense along N
// ("RowMajor" in the CUTLASS tag world), which is what the torch wrapper
// validates. Sub-byte weights override StrideB to be dense along K.
struct MoEConfigBase {
  using ElementAccumulator = float;
  // The optional bias is always expanded to fp32 [total_M, N] by the caller.
  using ElementBias = float;

  using StrideA = cute::Stride<int64_t, _1, int64_t>;
  using StrideB = cute::Stride<_1, int64_t, int64_t>;
  using StrideC = cute::Stride<int64_t, _1, int64_t>;
  using StrideD = cute::Stride<int64_t, _1, int64_t>;
  using StrideScale = cute::Stride<_1, int64_t, int64_t>;

  // Store cache policy override for D (void keeps the CuTe default).
  using StoreCache = void;

  static constexpr MoEScaleKind ScaleKind = MoEScaleKind::None;
  static constexpr int ScaleBlock = 0;
  static constexpr int Stages = 4;
  // Set by the dtypes whose widest tiles run their K loop faster without the
  // per-K-tile workgroup barrier.
  static constexpr bool WideTileBarrierFree = false;
};

struct MoEBf16Config : MoEConfigBase {
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementMma = cutlass::bfloat16_t;
  using ElementScaleA = void;
  using ElementScaleB = void;
  using ElementOutput = cutlass::bfloat16_t;
  using MmaAtom = MMA_Atom<XE_DPAS_TT<8, float, ElementMma>>;
  static constexpr int Stages = 2;
};

struct MoEFp16Config : MoEConfigBase {
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementMma = cutlass::half_t;
  using ElementScaleA = void;
  using ElementScaleB = void;
  using ElementOutput = cutlass::half_t;
  using MmaAtom = MMA_Atom<XE_DPAS_TT<8, float, ElementMma>>;
  static constexpr int Stages = 4;
};

// MXFP8: e4m3 data with e8m0 scales every 32 elements on both operands.
struct MoEMxfp8Config : MoEConfigBase {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementMma = cutlass::float_e4m3_t;
  using ElementScaleA = cutlass::float_ue8m0_t;
  using ElementScaleB = cutlass::float_ue8m0_t;
  using ElementOutput = cutlass::bfloat16_t;
  using MmaAtom = MMA_Atom<XE_BDPAS_TT<8, float, ElementMma>>;
  static constexpr MoEScaleKind ScaleKind = MoEScaleKind::MXBlock;
  static constexpr int ScaleBlock = 32;
  static constexpr int Stages = 2;
};

// MXFP4: 4-bit e2m1 operands, weights dense along K.
struct MoEMxfp4Config : MoEConfigBase {
  using ElementA = cutlass::float_e2m1_t;
  using ElementB = cutlass::float_e2m1_t;
  using ElementMma = cutlass::float_e2m1_t;
  using ElementScaleA = cutlass::float_ue8m0_t;
  using ElementScaleB = cutlass::float_ue8m0_t;
  using ElementOutput = cutlass::bfloat16_t;
  using MmaAtom = MMA_Atom<XE_BDPAS_TT<8, float, ElementMma>>;
  using StrideB = cute::Stride<int64_t, _1, int64_t>;
  using StoreCache = cute::C<XeStoreCachePolicy::kWB_UC_UC>;
  static constexpr MoEScaleKind ScaleKind = MoEScaleKind::MXBlock;
  static constexpr int ScaleBlock = 32;
  static constexpr int Stages = 4;
  static constexpr bool WideTileBarrierFree = true;
};

// W4A8: e4m3 activations x e2m1 weights. Both operands are reordered to bf16
// before the block-scaled DPAS (there is no direct e2m1 -> e4m3 reorder), and
// the hardware still applies the e8m0 block scales, so the result matches a
// native mxfp4 weight.
struct MoEW4a8Config : MoEConfigBase {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e2m1_t;
  using ElementMma = cutlass::bfloat16_t;
  using ElementScaleA = cutlass::float_ue8m0_t;
  using ElementScaleB = cutlass::float_ue8m0_t;
  using ElementOutput = cutlass::bfloat16_t;
  using MmaAtom = MMA_Atom<XE_BDPAS_TT<8, float, ElementMma>>;
  using StrideB = cute::Stride<int64_t, _1, int64_t>;
  static constexpr MoEScaleKind ScaleKind = MoEScaleKind::MXBlock;
  static constexpr int ScaleBlock = 32;
  static constexpr int Stages = 4;
};

// FP8 with 128x128 block scales: plain e4m3 DPAS, scales folded in the K loop.
struct MoEFp8BlockConfig : MoEConfigBase {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementMma = cutlass::float_e4m3_t;
  using ElementScaleA = float;
  using ElementScaleB = float;
  using ElementOutput = cutlass::bfloat16_t;
  using MmaAtom = MMA_Atom<XE_DPAS_TT<8, float, ElementMma>>;
  static constexpr MoEScaleKind ScaleKind = MoEScaleKind::Fp8Block;
  static constexpr int ScaleBlock = 128;
  static constexpr int Stages = 4;
};

// FP8 per-tensor: same e4m3 DPAS, one scalar for A and one per expert for B.
struct MoEFp8TensorConfig : MoEFp8BlockConfig {
  static constexpr MoEScaleKind ScaleKind = MoEScaleKind::Fp8Tensor;
  static constexpr int Stages = 3;
};

}  // namespace gpu::cutlass_kernel::grouped_gemm
