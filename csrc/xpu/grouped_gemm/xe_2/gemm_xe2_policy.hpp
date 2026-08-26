#pragma once

#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

namespace MoE {
using namespace cute;

class xe_gemm_policy_base {
 public:
  using WGTile = Shape<_256, _256, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

  // Copy can be turned for better performance
  using GmemTiledCopyA = void;  // same as make_block_2d_copy_A
  using GmemTiledCopyB = void;  // same as make_block_2d_copy_B
  using GmemTiledCopyD = void;  // same as make_block_2d_copy_D
};

class w16a16_policy : public xe_gemm_policy_base {
 public:
  using GmemTiledCopyD = XE_STORE_2D<16, 8, 32>;
};

class w16a16_policy_n_128 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_256, _128, _32>;
  using SGLayout = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;
};

class w16a16_policy_n_64 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_256, _64, _32>;
  using SGLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _0>>;
};

class w16a16_policy_m_8 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_8, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class w16a16_policy_m_16 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_16, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class w16a16_policy_m_32 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_32, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class w8a16_policy : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_128, _128, _16>;
  using SGLayout = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;

  using GmemTiledCopyD = XE_STORE_2D<16, 8, 32>;
};

class w8a16_policy_m_8 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_8, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class w8a16_policy_m_16 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_16, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class w8a16_policy_m_32 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_32, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class w4a16_policy : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_128, _256, _32>;
  using SGLayout = Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>;

  using GmemTiledCopyD = XE_STORE_2D<16, 8, 32>;
};

class w4a16_policy_m_8 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_8, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class w4a16_policy_m_16 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_16, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class w4a16_policy_m_32 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_32, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

// NVFP4 ladder. These mirror the w4a16 policies above with the K tile halved
// to 16 so that tile_k == group_size for NVFP4's 16-element block scales.
// One tile then covers exactly one scale group, which keeps the existing
// scale-reload gate in xe_gemm_4bits (`k_tile * tile_k % group_size == 0`)
// correct without touching the mainloop. Halving tile_k also halves the work
// per K iteration, so these are not interchangeable with the tile_k=32
// variants on performance grounds.
class w4a16_policy_k16 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_128, _256, _16>;
  using SGLayout = Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>;

  using GmemTiledCopyD = XE_STORE_2D<16, 8, 32>;
};

class w4a16_policy_m_8_k16 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_8, _64, _16>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class w4a16_policy_m_16_k16 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_16, _64, _16>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class w4a16_policy_m_32_k16 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_32, _64, _16>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

}  // namespace MoE