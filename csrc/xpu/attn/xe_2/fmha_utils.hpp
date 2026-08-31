#pragma once
#include "xpu/attn/paged_kv_utils.h"
#include "torch/all.h"
#include <cute/tensor.hpp>

#define HEAD_SIZE_LIMIT_0 64
#define HEAD_SIZE_LIMIT_1 96
#define HEAD_SIZE_LIMIT_2 128
#define HEAD_SIZE_LIMIT_3 192
#define HEAD_SIZE_LIMIT_4 256
#define HEAD_SIZE_LIMIT_5 512
#define HEAD_SIZE_LIMIT_6 576

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

// Maximum V-dim (head_size_vo) a single decode work-group may own.
//
// The softmax/output accumulator held in registers is
// (q_packed x ShapeOut[1]) floats, and the epilogue's cross-SG reduction
// buffer is q_packed * ShapeOut[1] * SGPerWG * sizeof(float) of SLM. At
// head_dim 512/576 that accumulator needs the entire Xe2 register file (256
// GRF at -cl-intel-256-GRF-per-thread), so the kernel spills it to memory on
// *every* KV tile: measured ~65 GB/s at head 512 and ~30 GB/s at head 576
// (MLA) versus ~440 GB/s at head 256 on the same card. With q_packed 16 the
// kernel does not even launch (UR_RESULT_ERROR_OUT_OF_RESOURCES at head 512,
// 128 KiB of SLM at head 576).
//
// Capping the V-dim makes the decode scheduler split V across grid.x (it
// derives grid.x = ceil_div(head_size_vo, ShapeOut[1]); the mainloop and
// epilogue already tile V by blk_v, and the chunk-prefill head-512 policy
// uses the same 256-wide split). The split costs an extra K read per V tile
// but restores head-256-class memory-level parallelism, and it keeps the SLM
// reduction buffer within the per-WG cap.
//
// Because the accumulator is q_packed x ShapeOut[1], the cap has to scale
// with q_packed: a fixed 256 leaves q_packed 16 spilling nearly as badly as
// before (measured 4.5x slower than q_packed 8 on MLA, tracking q_packed
// almost exactly). kDecodeAccBudget bounds the product instead, giving every
// q_packed the same accumulator footprint that makes q_packed 8 fast.
// q_packed is only ever _8 or _16 (see paged_decode_xe2.cpp: GQA ratios above
// 8 are split across work-groups using the _16 tile), so this resolves to
// V=256 for _8 and V=128 for _16.
//
// The budget was tuned by sweeping it on MLA (head_size_qk 576) decode:
//
//   budget   q8 V / q16 V   q8 kv=54k   q16 kv=54k
//   8192     256 / 256      0.335 ms    1.492 ms
//   2048     256 / 128      0.336 ms    0.499 ms   <- chosen
//   1024     128 /  64      0.436 ms    0.523 ms
//
// 8192 (no q_packed scaling) leaves q_packed 16 spilling; 1024 over-splits,
// costing q_packed 8 30% because each extra V tile re-reads K.
//
// Only head_dim > 256 is affected; smaller head dims already fit and are left
// bit-identical so existing tuned configs do not change.
//
// Note head_dim here is the *Q/K* head size bucket; the V split is driven by
// the runtime head_size_vo, which differs for MLA (head_size_qk 576 with
// head_size_vo 512).
static constexpr int kDecodeMaxShapeOutV = 256;
static constexpr int kDecodeAccBudget = 2048;

template <typename q_packed, typename head_dim>
constexpr int decode_shapeout_v_value() {
  if constexpr (head_dim::value <= kDecodeMaxShapeOutV) {
    return head_dim::value;
  } else {
    constexpr int budgeted = kDecodeAccBudget / q_packed::value;
    return budgeted < kDecodeMaxShapeOutV ? budgeted : kDecodeMaxShapeOutV;
  }
}

template <typename q_packed, typename head_dim>
using decode_shapeout_v =
    cute::Int<decode_shapeout_v_value<q_packed, head_dim>()>;

// kv_tile == _16 (block_size == 16)
template <typename q_packed, typename head_dim>
struct decode_policy_qpacked_head<q_packed, head_dim, _16> {
  using HeadDim = head_dim;
  using ShapeQK = Shape<q_packed, _16, _64>;
  using ShapePV = Shape<q_packed, _32, _16>;
  using ShapeOut = Shape<q_packed, decode_shapeout_v<q_packed, head_dim>>;
  using SubgroupLayoutQK = Layout<Shape<_1, _1, _1>>;
};

// kv_tile == _32 (block_size == 32)
template <typename q_packed, typename head_dim>
struct decode_policy_qpacked_head<q_packed, head_dim, _32> {
  using HeadDim = head_dim;
  using ShapeQK = Shape<q_packed, _32, _64>;
  using ShapePV = Shape<q_packed, _32, _32>;
  using ShapeOut = Shape<q_packed, decode_shapeout_v<q_packed, head_dim>>;
  using SubgroupLayoutQK = Layout<Shape<_1, _2, _1>>;
};

// kv_tile == _64
// Also services any block_size that is a positive multiple of 64
// (e.g. 64, 128, 192, 256, 320, ...). The mainloop iterates
// page_size / 64 sub-tiles per page via the page-table indirection.
template <typename q_packed, typename head_dim>
struct decode_policy_qpacked_head<q_packed, head_dim, _64> {
  using HeadDim = head_dim;
  using ShapeQK = Shape<q_packed, _64, _64>;
  using ShapePV = Shape<q_packed, _32, _64>;
  using ShapeOut = Shape<q_packed, decode_shapeout_v<q_packed, head_dim>>;
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
  using HeadDim = head_dim;
  using ShapeQK = Shape<q_packed, _128, _64>;
  using ShapePV = Shape<q_packed, _32, _128>;
  using ShapeOut = Shape<q_packed, decode_shapeout_v<q_packed, head_dim>>;
  using SubgroupLayoutQK = Layout<Shape<_1, _8, _1>>;
};
