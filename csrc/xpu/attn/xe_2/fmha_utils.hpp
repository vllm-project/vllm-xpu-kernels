#pragma once
#include "torch/all.h"
#include <cute/tensor.hpp>

#define HEAD_SIZE_LIMIT_0 64
#define HEAD_SIZE_LIMIT_1 96
#define HEAD_SIZE_LIMIT_2 128
#define HEAD_SIZE_LIMIT_3 192
#define HEAD_SIZE_LIMIT_4 256
#define HEAD_SIZE_LIMIT_5 512

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

inline bool is_interleaved_kv_cache(
    const at::Tensor& key_cache, const at::Tensor& value_cache) {
  auto value_stride = value_cache.strides();

  if (key_cache.dim() == value_cache.dim() &&
      key_cache.sizes() == value_cache.sizes() &&
      key_cache.strides() == value_stride &&
      key_cache.element_size() == value_cache.element_size() &&
      key_cache.storage().data_ptr() == value_cache.storage().data_ptr() &&
      key_cache.storage_offset() == 0 &&
      value_cache.storage_offset() == value_stride[0] / 2) {
    return true;
  }

  return false;
}

using namespace cute;
struct chunk_policy_head64 {
  using ShapeQK = Shape<_128, _32, _32>;
  using ShapePV = Shape<_128, _32, _32>;
  using ShapeOut = Shape<_128, _64>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
};

struct chunk_policy_head96 {
  using ShapeQK = Shape<_128, _32, _32>;
  using ShapePV = Shape<_128, _32, _32>;
  using ShapeOut = Shape<_128, _96>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
};

struct chunk_policy_head128 {
  using ShapeQK = Shape<_256, _32, _32>;
  using ShapePV = Shape<_256, _32, _32>;
  using ShapeOut = Shape<_256, _128>;
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

struct chunk_policy_head512 {
  using ShapeQK = Shape<_256, _32, _32>;
  using ShapePV = Shape<_256, _32, _32>;
  using ShapeOut = Shape<_256, _256>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
};

// chunk_prefill policies with TileShapeQK[1] = 16 (for block_size = 16).
// These mirror the head-size policies above but halve the K-dim sub-tile so
// that page_size=16 satisfies tiles_per_page = page_size / TileShapeQK[1] = 1.
struct chunk_policy_head64_b16 {
  using ShapeQK = Shape<_128, _16, _32>;
  using ShapePV = Shape<_128, _32, _16>;
  using ShapeOut = Shape<_128, _64>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
};

struct chunk_policy_head96_b16 {
  using ShapeQK = Shape<_128, _16, _32>;
  using ShapePV = Shape<_128, _32, _16>;
  using ShapeOut = Shape<_128, _96>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
};

struct chunk_policy_head128_b16 {
  using ShapeQK = Shape<_256, _16, _32>;
  using ShapePV = Shape<_256, _32, _16>;
  using ShapeOut = Shape<_256, _128>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
};

struct chunk_policy_head192_b16 {
  using ShapeQK = Shape<_256, _16, _32>;
  using ShapePV = Shape<_256, _32, _16>;
  using ShapeOut = Shape<_256, _192>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
};

struct chunk_policy_head256_b16 {
  using ShapeQK = Shape<_256, _16, _32>;
  using ShapePV = Shape<_256, _32, _16>;
  using ShapeOut = Shape<_256, _256>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
};

struct chunk_policy_head512_b16 {
  using ShapeQK = Shape<_256, _16, _32>;
  using ShapePV = Shape<_256, _32, _16>;
  using ShapeOut = Shape<_256, _256>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
};

// define decode policy
template <typename q_packed, typename head_dim, typename kv_tile>
struct decode_policy_qpacked_head {
  static_assert(
      cute::is_same_v<kv_tile, _16> || cute::is_same_v<kv_tile, _32> ||
          cute::is_same_v<kv_tile, _64> || cute::is_same_v<kv_tile, _128>,
      "Unsupported kv_tile for decode_policy_qpacked_head "
      "(supported: _16, _32, _64, _128)");
};

// kv_tile == _16 (block_size == 16)
template <typename q_packed, typename head_dim>
struct decode_policy_qpacked_head<q_packed, head_dim, _16> {
  using ShapeQK = Shape<q_packed, _16, _64>;
  using ShapePV = Shape<q_packed, _32, _16>;
  using ShapeOut = Shape<q_packed, head_dim>;
  using SubgroupLayoutQK = Layout<Shape<_1, _1, _1>>;
};

// kv_tile == _32 (block_size == 32)
template <typename q_packed, typename head_dim>
struct decode_policy_qpacked_head<q_packed, head_dim, _32> {
  using ShapeQK = Shape<q_packed, _32, _64>;
  using ShapePV = Shape<q_packed, _32, _32>;
  using ShapeOut = Shape<q_packed, head_dim>;
  using SubgroupLayoutQK = Layout<Shape<_1, _2, _1>>;
};

// kv_tile == _64
// Also services any block_size that is a positive multiple of 64
// (e.g. 64, 128, 192, 256, 320, ...). The mainloop iterates
// page_size / 64 sub-tiles per page via the page-table indirection.
template <typename q_packed, typename head_dim>
struct decode_policy_qpacked_head<q_packed, head_dim, _64> {
  using ShapeQK = Shape<q_packed, _64, _64>;
  using ShapePV = Shape<q_packed, _32, _64>;
  using ShapeOut = Shape<q_packed, head_dim>;
  using SubgroupLayoutQK = Layout<Shape<_1, _4, _1>>;
};

// kv_tile == _128
// NOTE: Currently UNUSED. The dispatcher in paged_decode_utils.hpp routes
// page_size that is a multiple of 128 through the kv_tile=_64 policy because
// this _128 policy uses SubgroupLayoutQK<_1,_8,_1> (ReduceK=8), which
// triggers a wrong-result bug in the cross-SG SLM reduction
// (chunk_prefill_epilogue.hpp::reduce_A) when SGTileShapeO collapses to
// (1, 32). See dispatch_by_page_size for details. Kept here so it can be
// re-enabled once the upstream ReduceK=8 reduction path is fixed.
template <typename q_packed, typename head_dim>
struct decode_policy_qpacked_head<q_packed, head_dim, _128> {
  using ShapeQK = Shape<q_packed, _128, _64>;
  using ShapePV = Shape<q_packed, _32, _128>;
  using ShapeOut = Shape<q_packed, head_dim>;
  using SubgroupLayoutQK = Layout<Shape<_1, _8, _1>>;
};
