#pragma once
#include "torch/all.h"

#define HEAD_SIZE_LIMIT_0 64
#define HEAD_SIZE_LIMIT_1 128
#define HEAD_SIZE_LIMIT_2 256
#define HEAD_SIZE_LIMIT_3 512

enum class CutlassType {
  half,
  bfloat16,
};

inline CutlassType aten_to_Cutlass_dtype(const at::Tensor& input) {
  CutlassType cuType;
  if (input.scalar_type() == torch::kHalf) {
    cuType = CutlassType::half;
  } else if (input.scalar_type() == torch::kBFloat16) {
    cuType = CutlassType::bfloat16;
  } else {
    TORCH_INTERNAL_ASSERT(
        false,
        "");
  }
  return cuType;
}
