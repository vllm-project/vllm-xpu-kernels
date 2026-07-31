#pragma once
#include <Python.h>
#include <sycl/sycl.hpp>
#include <torch/all.h>
#include <c10/xpu/XPUStream.h>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>

#include "utils.h"

using namespace cute;

namespace syclex = sycl::ext::oneapi::experimental;
namespace sycli = sycl::ext::intel::experimental;

using bf16_t = cutlass::bfloat16_t;
using fp16_t = cutlass::half_t;

// DPAS tile dimensions
static constexpr int TM = 8;
static constexpr int TN = 16;
static constexpr int TK = 16;
static constexpr int SG_COUNT = 8;
static constexpr int SG_SIZE = 16;
static constexpr int WG_SIZE = SG_COUNT * SG_SIZE;
static constexpr int N_TM = 4;
static constexpr int N_TK = 2;
static constexpr int N_TN = 2;
static constexpr int SLM_STRIDE = SG_COUNT * TK;

static inline float silu_apply(float v) { return v / (1.0f + sycl::exp(-v)); }

// SLM-to-register TiledCopy for A (s2r)
template <class TiledMMA>
CUTE_HOST_DEVICE auto make_block_2d_slm_copy_A(TiledMMA const& mma) {
  using ValType = typename TiledMMA::ValTypeA;
  using CopyType = cute::uint_bit_t<cute::sizeof_bits_v<ValType>>;
  auto tile_mk = select<0, 2>(mma.tile_mnk());
  auto thrfrg = mma.thrfrg_A(make_layout(tile_mk));
  auto thrfrg_T = layout<0>(thrfrg);
  auto thrfrg_V = layout<1>(thrfrg);
  auto shape_vmnk = shape(mma.get_thr_layout_vmnk());
  auto s2r_T = make_layout(
      make_shape(shape<0>(thrfrg_T), get<2>(shape_vmnk), shape<1>(thrfrg_T)),
      make_stride(stride<0>(thrfrg_T), _0{}, stride<1>(thrfrg_T)));
  auto s2r_tv = make_layout(s2r_T, thrfrg_V);
  auto atom = Copy_Atom<UniversalCopy<CopyType>, ValType>{};
  return cute::TiledCopy<decltype(atom), decltype(s2r_tv), decltype(tile_mk)>{};
}

// ─────────────────────────────────────────────────────────────────────────────
// FusedMoeCommon — data members and methods shared by FusedMoeDecode and
// FusedMoePrefill.  Neither decode nor prefill method sets are included here;
// see fused_moe_decode.hpp and fused_moe_prefill.hpp.
// ─────────────────────────────────────────────────────────────────────────────
template <
    bool HAS_W13_BIAS,
    bool HAS_W2_BIAS,
    bool HAS_CLAMP_LIMIT,
    bool IS_DECODE,
    typename DTYPE,
    typename INTER_DTYPE,
    int TM,
    int N_TM,
    int N_TK,
    int N_TN>
struct FusedMoeCommon {
  const DTYPE* tokens;    // [NT, H]
  const DTYPE* w13;       // [E, H, 2*I]
  const DTYPE* w2;        // [E, I, H]
  const float* w13_bias;  // [E, 2*I] or nullptr
  const float* w2_bias;   // [E, H]   or nullptr
  DTYPE* output;          // [NT, H]
  int H;
  int I;
  INTER_DTYPE* intermediate;  // [NT*K, H]
  int32_t* row_counter;       // [NT]
  int K;
  sycl::local_accessor<DTYPE, 1> slm;
  const int64_t* expert_offset;  // [E+1]
  int num_experts;
  int total_tokens;
  const int32_t* source_row;  // [T]
  const float* topk_weights;  // [T]
  int num_tokens;
  const int32_t* topk_ids;  // [T]
  float gemm1_clamp_limit;

  static auto make_gate_up_tiled_mma() {
    using Op = XE_DPAS_TT<TM, float, DTYPE, DTYPE>;
    using WGTile = Shape<Int<N_TM * TM>, Int<SG_COUNT * TN>, Int<N_TK * TK>>;
    using SGLayout = Layout<Shape<_1, Int<SG_COUNT>, _1>, Stride<_0, _1, _0>>;
    using MMA =
        typename TiledMMAHelper<MMA_Atom<Op>, Layout<WGTile>, SGLayout>::
            TiledMMA;
    return MMA{};
  }

  static auto make_down_tiled_mma() {
    using Op = XE_DPAS_TT<TM, float, DTYPE, DTYPE>;
    using WGTile =
        Shape<Int<N_TM * TM>, Int<N_TN * SG_COUNT * TN>, Int<N_TK * TK>>;
    using SGLayout = Layout<Shape<_1, Int<SG_COUNT>, _1>, Stride<_0, _1, _0>>;
    using MMA =
        typename TiledMMAHelper<MMA_Atom<Op>, Layout<WGTile>, SGLayout>::
            TiledMMA;
    return MMA{};
  }

  static auto make_reduce_tiled_mma() {
    constexpr int TK_ = (cute::sizeof_bits_v<INTER_DTYPE> == 16) ? TK : TK / 2;
    using DPAS_DTYPE = std::conditional_t<
        std::is_same_v<INTER_DTYPE, float>,
        tfloat32_t,
        INTER_DTYPE>;
    using Op = XE_DPAS_TT<1, float, DPAS_DTYPE, DPAS_DTYPE, float>;
    using WGTile = Shape<_1, Int<TN>, Int<TK_>>;
    using SGLayout = Layout<Shape<_1, _1, _1>, Stride<_0, _1, _0>>;
    using MMA =
        typename TiledMMAHelper<MMA_Atom<Op>, Layout<WGTile>, SGLayout>::
            TiledMMA;
    return MMA{};
  }

  auto get(syclex::properties_tag) const {
    return syclex::properties(
        sycli::grf_size<256>, syclex::sub_group_size<SG_SIZE>);
  }

  template <int HEIGHT, int WIDTH = TN, typename TCAccum>
  __attribute__((always_inline)) void reduce_accumulate(
      int out_row,
      int sg_id,
      int lane,
      TCAccum& tCrOut,
      const int offset,
      int k_start,
      int k_values) const {
    using InType = decltype(*intermediate);
    constexpr int N_OUT = WIDTH / SG_SIZE;
    const int64_t base_addr = ((int64_t)out_row * k_values + k_start) * H;

    auto coord_reduce =
        make_identity_tensor(make_shape(Int<WIDTH>{}, Int<HEIGHT>{}));

    const int offset_next = offset + SG_COUNT * WIDTH;
    if (offset_next < H) {
      auto gReduce_next = make_tensor(
          make_gmem_ptr(intermediate + base_addr + offset_next + sg_id * WIDTH),
          make_layout(
              make_shape(Int<WIDTH>{}, Int<HEIGHT>{}), make_stride(_1{}, H)));
      auto pref_next = make_block_2d_prefetch(make_block_2d_copy(
          XE_LOAD_2D<cute::sizeof_bits_v<InType>, HEIGHT, WIDTH>{},
          gReduce_next));
      prefetch(pref_next, pref_next.get_slice(lane).partition_S(coord_reduce));
    }

    auto gReduce = make_tensor(
        make_gmem_ptr(intermediate + base_addr + offset + sg_id * WIDTH),
        make_layout(
            make_shape(Int<WIDTH>{}, Int<HEIGHT>{}), make_stride(_1{}, H)));
    auto copy_reduce = make_block_2d_copy(
        XE_LOAD_2D<cute::sizeof_bits_v<InType>, HEIGHT, WIDTH>{}, gReduce);
    auto thr_copy_reduce = copy_reduce.get_slice(lane);
    auto tRrReduce = thr_copy_reduce.partition_fragment_D(coord_reduce);
    copy(copy_reduce, thr_copy_reduce.partition_S(coord_reduce), tRrReduce);

    if constexpr (HEIGHT == 1) {
      CUTE_UNROLL
      for (int n = 0; n < N_OUT; ++n)
        tCrOut(n) = float(tCrOut(n)) + float(tRrReduce(n));
    } else {
      CUTE_UNROLL
      for (int n = 0; n < N_OUT; ++n) {
        float sum = 0.0f;
        CUTE_UNROLL
        for (int k = 0; k < HEIGHT; ++k)
          sum += float(tRrReduce(k * N_OUT + n));
        tCrOut(n) = float(tCrOut(n)) + sum;
      }
    }
  }

  template <int HEIGHT, typename TCAccum, typename TCrA>
  __attribute__((always_inline)) void reduce_accumulate_dpas(
      int out_row,
      int sg_id,
      int lane,
      TCAccum& tCrOut,
      const TCrA& tCrA_in,
      const int offset,
      int k_start,
      int k_values) const {
    // TK_: DPAS K-dimension. bf16 → TK=16, tf32/float32 → TK/2=8.
    constexpr int TK_ = (cute::sizeof_bits_v<INTER_DTYPE> == 16) ? TK : TK / 2;
    static_assert(
        HEIGHT <= TK_, "reduce_accumulate_dpas: HEIGHT must be <= TK_");

    constexpr int WIDTH = TN;
    const int64_t base_addr = ((int64_t)out_row * k_values + k_start) * H;
    const int sg_offset = offset + sg_id * WIDTH;

    const int offset_next = offset + SG_COUNT * WIDTH;
    if (offset_next < H) {
      auto coord_pf =
          make_identity_tensor(make_shape(Int<WIDTH>{}, Int<HEIGHT>{}));
      auto gPref = make_tensor(
          make_gmem_ptr(intermediate + base_addr + offset_next + sg_id * WIDTH),
          make_layout(
              make_shape(Int<WIDTH>{}, Int<HEIGHT>{}), make_stride(_1{}, H)));
      auto pref = make_block_2d_prefetch(make_block_2d_copy(
          XE_LOAD_2D<cute::sizeof_bits_v<INTER_DTYPE>, HEIGHT, WIDTH>{},
          gPref));
      prefetch(pref, pref.get_slice(lane).partition_S(coord_pf));
    }

    auto mma = make_reduce_tiled_mma();
    auto thr_mma = mma.get_slice(lane);
    auto cBktile = make_identity_tensor(select<1, 2>(mma.tile_mnk()));

    auto gB = make_tensor(
        make_gmem_ptr(intermediate + base_addr + sg_offset),
        make_layout(
            make_shape(Int<WIDTH>{}, Int<TK_>{}), make_stride(_1{}, H)));
    auto copy_B = make_block_2d_copy_B(mma, gB);
    auto thr_copy_B = copy_B.get_slice(lane);
    auto tCrB = thr_mma.partition_sg_fragment_B(cBktile);
    auto tBrB = thr_copy_B.partition_sg_fragment_D(cBktile);
    copy(copy_B, thr_copy_B.partition_S(cBktile), tBrB);
    reorder(tBrB, tCrB);

    cute::gemm(mma, tCrA_in, tCrB, tCrOut);
  }

  __attribute__((always_inline)) void
  reduce_one_row(int out_row, int sg_id, int lane, int k_values) const {
    constexpr int WIDTH = TN;
    constexpr int count = WIDTH / SG_SIZE;
    const int blocks = H / (SG_COUNT * WIDTH);
    auto coord_out = make_identity_tensor(make_shape(_1{}, Int<WIDTH>{}));

    constexpr int TK_ = (cute::sizeof_bits_v<INTER_DTYPE> == 16) ? TK : TK / 2;
    using DPAS_DTYPE = std::conditional_t<
        std::is_same_v<INTER_DTYPE, float>,
        tfloat32_t,
        INTER_DTYPE>;
    auto mma = make_reduce_tiled_mma();
    auto thr_mma = mma.get_slice(lane);
    auto tCrA_full = thr_mma.partition_sg_fragment_A(
        make_identity_tensor(select<0, 2>(mma.tile_mnk())));
    tCrA_full(0) = DPAS_DTYPE(1.0f);
    auto tCrA_half = thr_mma.partition_sg_fragment_A(
        make_identity_tensor(select<0, 2>(mma.tile_mnk())));
    tCrA_half(0) = DPAS_DTYPE(lane < TK_ / 2 ? 1.0f : 0.0f);

    for (int block_idx = 0; block_idx < blocks; ++block_idx) {
      const int offset = block_idx * SG_COUNT * WIDTH;
      auto gOut = make_tensor(
          make_gmem_ptr(output + (int64_t)out_row * H + offset + sg_id * WIDTH),
          make_layout(make_shape(_1{}, Int<WIDTH>{}), make_stride(H, _1{})));
      auto copy_out = make_block_2d_copy(
          XE_STORE_2D<cute::sizeof_bits_v<DTYPE>, 1, WIDTH>{}, gOut);
      auto thr_copy_out = copy_out.get_slice(lane);

      auto tCrOut = partition_fragment_C(mma, select<0, 1>(mma.tile_mnk()));
      clear(tCrOut);
      int k_start = 0;
      const int k_end = k_values;
      for (; k_start + TK_ <= k_end; k_start += TK_)
        reduce_accumulate_dpas<TK_>(
            out_row, sg_id, lane, tCrOut, tCrA_full, offset, k_start, k_values);
      for (; k_start + TK_ / 2 <= k_end; k_start += TK_ / 2)
        reduce_accumulate_dpas<TK_ / 2>(
            out_row, sg_id, lane, tCrOut, tCrA_half, offset, k_start, k_values);
      for (; k_start + 4 <= k_end; k_start += 4)
        reduce_accumulate<4>(
            out_row, sg_id, lane, tCrOut, offset, k_start, k_values);
      for (; k_start + 2 <= k_end; k_start += 2)
        reduce_accumulate<2>(
            out_row, sg_id, lane, tCrOut, offset, k_start, k_values);
      for (; k_start < k_end; ++k_start)
        reduce_accumulate<1>(
            out_row, sg_id, lane, tCrOut, offset, k_start, k_values);
      auto tCrOut_ = thr_copy_out.partition_fragment_S(coord_out);
      CUTE_UNROLL
      for (int n = 0; n < count; ++n)
        tCrOut_(n) = DTYPE(float(tCrOut(n)));
      copy(copy_out, tCrOut_, thr_copy_out.partition_D(coord_out));
    }
  }
};
