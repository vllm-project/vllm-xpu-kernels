#pragma once
#include "torch/all.h"
#include <cute/tensor.hpp>

#define HEAD_SIZE_LIMIT_0 64
#define HEAD_SIZE_LIMIT_1 128
#define HEAD_SIZE_LIMIT_2 192
#define HEAD_SIZE_LIMIT_3 256
#define HEAD_SIZE_LIMIT_4 512

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
        false, "Current cutlass kernel only support half/bf16 data type.");
  }
  return cuType;
}

using namespace cute;
struct chunk_policy_head64 {
  using ShapeQK = Shape<_128, _32, _32>;
  using ShapePV = Shape<_128, _32, _32>;
  using ShapeOut = Shape<_128, _64>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
};

struct chunk_policy_head96 {
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOut = Shape<_128, _96>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
};

struct chunk_policy_head128 {
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOut = Shape<_128, _128>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
};

struct chunk_policy_head192 {
  using ShapeQK = Shape<_256, _32, _32>;
  using ShapePV = Shape<_256, _32, _32>;
  using ShapeOut = Shape<_256, _192>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
};

struct chunk_policy_head256 {
  using ShapeQK = Shape<_256, _32, _32>;
  using ShapePV = Shape<_256, _32, _32>;
  using ShapeOut = Shape<_256, _256>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
};

// define marco for decode policy
#define DECODE_NUM_SG _16
#define DECODE_KV_TILE _64 // KV tile size is set to 64 for page size is 64
#define DECODE_Q_PACKED_TILE _8

struct decode_policy_head64 {
  using ShapeQK = Shape<DECODE_Q_PACKED_TILE, DECODE_KV_TILE, _64>;
  using ShapePV = Shape<DECODE_Q_PACKED_TILE, _32, DECODE_KV_TILE>;
  using ShapeOut = Shape<DECODE_Q_PACKED_TILE, _64>;
  using SubgroupLayoutQK = Layout<Shape<_1, DECODE_NUM_SG, _1>>;
};

struct decode_policy_head96 {
  using ShapeQK = Shape<DECODE_Q_PACKED_TILE, DECODE_KV_TILE, _64>;
  using ShapePV = Shape<DECODE_Q_PACKED_TILE, _32, DECODE_KV_TILE>;
  using ShapeOut = Shape<DECODE_Q_PACKED_TILE, _96>;
  using SubgroupLayoutQK = Layout<Shape<_1, DECODE_NUM_SG, _1>>;
};

struct decode_policy_head128 {
  using ShapeQK = Shape<DECODE_Q_PACKED_TILE, DECODE_KV_TILE, _64>;
  using ShapePV = Shape<DECODE_Q_PACKED_TILE, _32, DECODE_KV_TILE>;
  using ShapeOut = Shape<DECODE_Q_PACKED_TILE, _128>;
  using SubgroupLayoutQK = Layout<Shape<_1, DECODE_NUM_SG, _1>>;
};

struct decode_policy_head192 {
  using ShapeQK = Shape<DECODE_Q_PACKED_TILE, DECODE_KV_TILE, _64>;
  using ShapePV = Shape<DECODE_Q_PACKED_TILE, _32, DECODE_KV_TILE>;
  using ShapeOut = Shape<DECODE_Q_PACKED_TILE, _192>;
  using SubgroupLayoutQK = Layout<Shape<_1, DECODE_NUM_SG, _1>>;
};

struct decode_policy_head256 {
  using ShapeQK = Shape<DECODE_Q_PACKED_TILE, DECODE_KV_TILE, _64>;
  using ShapePV = Shape<DECODE_Q_PACKED_TILE, _32, DECODE_KV_TILE>;
  using ShapeOut = Shape<DECODE_Q_PACKED_TILE, _256>;
  using SubgroupLayoutQK = Layout<Shape<_1, DECODE_NUM_SG, _1>>;
};
