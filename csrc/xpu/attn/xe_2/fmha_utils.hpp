#pragma once
#include "torch/all.h"
#include <cute/tensor.hpp>

#define HEAD_SIZE_LIMIT_0 64
#define HEAD_SIZE_LIMIT_1 96
#define HEAD_SIZE_LIMIT_2 128
#define HEAD_SIZE_LIMIT_3 192
#define HEAD_SIZE_LIMIT_4 256

enum class CutlassDType { half, bfloat16, float8_e4m3, float8_e5m2 };

// Struct to carry separate Q and K dtypes without breaking existing API
struct CutlassQKType {
  CutlassDType q_type;
  CutlassDType k_type;

  // Convenience: construct with identical types
  explicit CutlassQKType(CutlassDType t) : q_type(t), k_type(t) {}
  CutlassQKType(CutlassDType q_t, CutlassDType k_t)
      : q_type(q_t), k_type(k_t) {}
};

inline CutlassDType aten_to_dtype(const at::ScalarType st) {
  if (st == torch::kHalf) {
    return CutlassDType::half;
  } else if (st == torch::kBFloat16) {
    return CutlassDType::bfloat16;
  } else if (st == torch::kFloat8_e4m3fn) {
    return CutlassDType::float8_e4m3;
  } else if (st == torch::kFloat8_e5m2) {
    return CutlassDType::float8_e5m2;
  }
  TORCH_INTERNAL_ASSERT(
      false,
      "Unsupported dtype: only half/bfloat16/float8_e4m3/float8_e5m2 supported "
      "for Q/K.");
}

inline CutlassDType aten_to_dtype(const at::Tensor& t) {
  return aten_to_dtype(t.scalar_type());
}

// Helper to build Q/K dtype pair from tensors
inline CutlassQKType
aten_to_Cutlass_qk_dtype(const at::Tensor& q, const at::Tensor& k) {
  return CutlassQKType(aten_to_dtype(q), aten_to_dtype(k));
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

// define macro for decode policy
#define DECODE_NUM_SG _4
#define DECODE_KV_TILE _64  // KV tile size is set to 64 for page size is 64

template <class q_packed, class head_dim>
struct decode_policy_qpacked_head {
  using ShapeQK = Shape<q_packed, DECODE_KV_TILE, _64>;
  using ShapePV = Shape<q_packed, _32, DECODE_KV_TILE>;
  using ShapeOut = Shape<q_packed, head_dim>;
  using SubgroupLayoutQK = Layout<Shape<_1, DECODE_NUM_SG, _1>>;
};
