#pragma once

#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

namespace gpu::cutlass_kernel {
namespace grouped_gemm {

class moe_policy_base {
 public:
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementA = float;
  using ElementB = float;
  using ElementOutput = float;
  using ElementScale = float;
  using GmemTiledCopyA = cute::XE_2D_TF32x32x16_LD_N;
  using GmemTiledCopyB = cute::XE_2D_U32x32x16_LD_N;
  using MMAOperation = cute::XE_8x16x8_F32TF32TF32F32_TT;
};

class moe_bf16_policy : public moe_policy_base {
 public:
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementScale = cutlass::bfloat16_t;
  using GmemTiledCopyA = cute::XE_2D_U16x32x32_LD_N;
  using GmemTiledCopyB = cute::XE_2D_U16x32x32_LD_V;
  using MMAOperation = cute::XE_8x16x16_F32BF16BF16F32_TT;
};

class moe_fp16_policy : public moe_policy_base {
 public:
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementScale = cutlass::half_t;
  using GmemTiledCopyA = cute::XE_2D_U16x32x32_LD_N;
  using GmemTiledCopyB = cute::XE_2D_U16x32x32_LD_V;
  using MMAOperation = cute::XE_8x16x16_F32F16F16F32_TT;
};

class moe_e4m3fp16_policy : public moe_policy_base {
 public:
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementOutput = cutlass::half_t;
  using ElementScale = cutlass::half_t;
  using GmemTiledCopyA = cute::XE_2D_U16x32x32_LD_N;
  using GmemTiledCopyB = cute::XE_2D_U8x32x32_LD_V;
  using MMAOperation = cute::XE_8x16x16_F32F16F16F32_TT;
};

class moe_e5m2fp16_policy : public moe_policy_base {
 public:
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::float_e5m2_t;
  using ElementOutput = cutlass::half_t;
  using ElementScale = cutlass::half_t;
  using GmemTiledCopyA = cute::XE_2D_U16x32x32_LD_N;
  using GmemTiledCopyB = cute::XE_2D_U8x32x32_LD_V;
  using MMAOperation = cute::XE_8x16x16_F32F16F16F32_TT;
};

class moe_e4m3bf16_policy : public moe_policy_base {
 public:
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementScale = cutlass::bfloat16_t;
  using GmemTiledCopyA = cute::XE_2D_U16x32x32_LD_N;
  using GmemTiledCopyB = cute::XE_2D_U8x32x32_LD_V;
  using MMAOperation = cute::XE_8x16x16_F32BF16BF16F32_TT;
};

class moe_e5m2bf16_policy : public moe_policy_base {
 public:
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::float_e5m2_t;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementScale = cutlass::bfloat16_t;
  using GmemTiledCopyA = cute::XE_2D_U16x32x32_LD_N;
  using GmemTiledCopyB = cute::XE_2D_U8x32x32_LD_V;
  using MMAOperation = cute::XE_8x16x16_F32BF16BF16F32_TT;
};

}  // namespace grouped_gemm
}  // namespace gpu::cutlass_kernel
