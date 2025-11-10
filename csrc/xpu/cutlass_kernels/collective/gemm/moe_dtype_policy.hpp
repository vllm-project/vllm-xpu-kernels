#pragma once

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
};

class moe_bf16_policy : public moe_policy_base {
 public:
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementScale = cutlass::bfloat16_t;
};

class moe_fp16_policy : public moe_policy_base {
 public:
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementScale = cutlass::half_t;
};

}  // namespace grouped_gemm
}  // namespace gpu::cutlass_kernel
