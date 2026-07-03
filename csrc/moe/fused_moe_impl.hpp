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

template <
    bool HAS_W13_BIAS,
    bool HAS_W2_BIAS,
    bool HAS_CLAMP_LIMIT = false,
    bool IS_DECODE = false,
    typename DTYPE = bf16_t,
    int TM = (IS_DECODE ? 1 : ::TM),
    int N_TM = (IS_DECODE ? 1 : ::N_TM),
    int N_TK = ::N_TK,
    int N_TN = ::N_TN>
struct FusedMoe {
  const DTYPE* tokens;    // [NT, H]
  const DTYPE* w13;       // [E, H, 2*I]
  const DTYPE* w2;        // [E, I, H]
  const float* w13_bias;  // [E, 2*I] or nullptr
  const float* w2_bias;   // [E, H]   or nullptr
  DTYPE* output;          // [NT, H]
  int H;
  int I;
  float* intermediate;   // [NT*K, H] -- float32 expert results
  int32_t* row_counter;  // [NT]
  int K;
  sycl::local_accessor<DTYPE, 1> slm;
  const int64_t* expert_offset;  // [E+1]
  int num_experts;
  int total_tokens;
  const int32_t* source_row;  // [T] sorted source rows
  const float*
      topk_weights;  // [T] decode: flat weights; prefill: sorted weights
  int num_tokens;
  const int32_t*
      topk_ids;  // [T] decode: flat expert IDs; prefill: sorted slot IDs
  float gemm1_clamp_limit;  // 0 = disabled; > 0: clamp gate to [-inf, limit],
                            // up to [-limit, limit]

  static sycl::nd_range<2>
  get_nd_range(int total_tokens, int num_experts, int K = 0) {
    if constexpr (IS_DECODE) {
      return sycl::nd_range<2>(
          sycl::range<2>(1, (size_t)(K * WG_SIZE)),
          sycl::range<2>(1, (size_t)WG_SIZE));
    } else {
      const int num_blocks = (total_tokens + TM - 1) / TM;
      const int64_t super_m_blocks =
          ((int64_t)num_blocks + N_TM - 1) / N_TM + num_experts;
      return sycl::nd_range<2>(
          sycl::range<2>(1, (size_t)(super_m_blocks * WG_SIZE)),
          sycl::range<2>(1, (size_t)WG_SIZE));
    }
  }

  __attribute__((always_inline)) int compute_tile_info(
      int wg_id,
      bool tile_active[N_TM],
      int tile_token_offset[N_TM],
      int tile_token_count[N_TM]) const {
    if constexpr (IS_DECODE) {
      const int expert_id = (int)topk_ids[wg_id];
      tile_active[0] = true;
      tile_token_offset[0] = wg_id;
      tile_token_count[0] = 1;
      return expert_id;
    } else {
      int expert_id = -1;
      int expert_block = 0;
      int expert_remainer = wg_id;
      for (int e_idx = 0; e_idx < num_experts; ++e_idx) {
        const int expert_num_tokens =
            (int)(expert_offset[e_idx + 1] - expert_offset[e_idx]);
        const int m_blocks_expert = (expert_num_tokens + TM - 1) / TM;
        const int super_blocks_expert = (m_blocks_expert + N_TM - 1) / N_TM;
        if (expert_remainer < super_blocks_expert) {
          expert_id = e_idx;
          expert_block = m_blocks_expert;
          break;
        }
        expert_remainer -= super_blocks_expert;
      }
      const int first_m_block = expert_remainer * N_TM;
      CUTE_UNROLL
      for (int mt = 0; mt < N_TM; ++mt) {
        const int m_block_idx = first_m_block + mt;
        tile_active[mt] = (expert_id >= 0) && (m_block_idx < expert_block);
        if (tile_active[mt]) {
          const int tok_start = (int)expert_offset[expert_id];
          const int tok_count =
              (int)(expert_offset[expert_id + 1] - expert_offset[expert_id]);
          tile_token_offset[mt] = tok_start + m_block_idx * TM;
          tile_token_count[mt] = sycl::min(TM, tok_count - m_block_idx * TM);
        } else {
          tile_token_offset[mt] = 0;
          tile_token_count[mt] = 0;
        }
      }
      return expert_id;
    }
  }

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

  auto get(syclex::properties_tag) const {
    return syclex::properties(
        sycli::grf_size<256>, syclex::sub_group_size<SG_SIZE>);
  }

  __attribute__((always_inline)) void load_hidden_states(
      int sg_id,
      int lane,
      bool const tile_active[N_TM],
      int const tile_token_offset[N_TM],
      int k_idx,
      DTYPE* slm_ptr) const {
    const bool is_partial = (k_idx + SG_COUNT * TK > H);

    CUTE_UNROLL
    for (int tm_idx = 0; tm_idx < N_TM; ++tm_idx) {
      const int slm_row_idx = tm_idx * TM + sg_id;
      const int topk_row_idx = tile_token_offset[tm_idx] + sg_id;
      const bool valid =
          tile_active[tm_idx] && (sg_id < TM) && (topk_row_idx < total_tokens);
      if (valid) {
        const int row_idx = (int)source_row[topk_row_idx];
        if (!is_partial) {
          auto slm_coord =
              make_identity_tensor(make_shape(Int<TK>{}, Int<SG_COUNT>{}));
          auto gA = make_tensor(
              make_gmem_ptr(tokens + (int64_t)row_idx * H + k_idx),
              make_layout(
                  make_shape(Int<TK>{}, Int<SG_COUNT>{}),
                  make_stride(_1{}, Int<TK>{})));
          auto load_copy = make_block_2d_copy(
              XE_LOAD_2D<cute::sizeof_bits_v<DTYPE>, SG_COUNT, TK>{}, gA);
          auto thr_lc = load_copy.get_slice(lane);
          auto tSrS = thr_lc.partition_fragment_D(slm_coord);
          const int next_tm_idx = tm_idx + 1;
          if (next_tm_idx < N_TM && tile_active[next_tm_idx]) {
            const int next_topk_row = tile_token_offset[next_tm_idx] + sg_id;
            if (next_topk_row < total_tokens) {
              const int next_row_idx = (int)source_row[next_topk_row];
              auto gA_next = make_tensor(
                  make_gmem_ptr(tokens + (int64_t)next_row_idx * H + k_idx),
                  make_layout(
                      make_shape(Int<TK>{}, Int<SG_COUNT>{}),
                      make_stride(_1{}, Int<TK>{})));
              auto pref_copy = make_block_2d_prefetch(make_block_2d_copy(
                  XE_LOAD_2D<cute::sizeof_bits_v<DTYPE>, SG_COUNT, TK>{},
                  gA_next));
              prefetch(
                  pref_copy, pref_copy.get_slice(lane).partition_S(slm_coord));
            }
          }
          copy(load_copy, thr_lc.partition_S(slm_coord), tSrS);
          CUTE_UNROLL
          for (int j = 0; j < SG_COUNT; ++j)
            slm_ptr[slm_row_idx * SLM_STRIDE + j * TK + lane] =
                static_cast<DTYPE>(tSrS(j));
        } else {
          CUTE_UNROLL
          for (int j = 0; j < SG_COUNT * TK / SG_SIZE; ++j) {
            const int src_col = k_idx + j * SG_SIZE + lane;
            slm_ptr[slm_row_idx * SLM_STRIDE + j * SG_SIZE + lane] =
                (src_col < H) ? tokens[(int64_t)row_idx * H + src_col]
                              : DTYPE(0);
          }
        }
      } else if (sg_id < TM) {
        CUTE_UNROLL
        for (int j = 0; j < SG_COUNT * TK / SG_SIZE; ++j)
          slm_ptr[slm_row_idx * SLM_STRIDE + j * SG_SIZE + lane] = DTYPE(0);
      }
    }
  }

  template <typename TiledMMA>
  __attribute__((always_inline)) void compute_gemm13_silu(
      sycl::nd_item<2> const& it,
      TiledMMA const& mma,
      int local_id,
      int sg_id,
      int lane,
      int n_idx,
      int expert_id,
      const DTYPE* w13_expert,
      bool const tile_active[N_TM],
      int const tile_token_offset[N_TM],
      DTYPE* slm_ptr) const {
    Tensor cW1 = make_tensor(
        make_gmem_ptr(w13_expert + n_idx),
        make_layout(
            make_shape(Int<SG_COUNT * TN>{}, H),
            make_stride(_1{}, (int)(2 * I))));
    Tensor cW3 = make_tensor(
        make_gmem_ptr(w13_expert + I + n_idx),
        make_layout(
            make_shape(Int<SG_COUNT * TN>{}, H),
            make_stride(_1{}, (int)(2 * I))));

    auto wg_tile = mma.tile_mnk();
    auto tCrGate = partition_fragment_C(mma, select<0, 1>(wg_tile));
    auto tCrUp = partition_fragment_C(mma, select<0, 1>(wg_tile));
    clear(tCrGate);
    clear(tCrUp);

    auto cBw_ktile = make_identity_tensor(select<1, 2>(wg_tile));
    auto cBw1_full = make_identity_tensor(make_shape(Int<SG_COUNT * TN>{}, H));
    auto cBw3_full = make_identity_tensor(make_shape(Int<SG_COUNT * TN>{}, H));
    auto gBw1_coord =
        local_tile(cBw1_full, select<1, 2>(wg_tile), make_coord(Int<0>{}, _));
    auto gBw3_coord =
        local_tile(cBw3_full, select<1, 2>(wg_tile), make_coord(Int<0>{}, _));
    auto copy_w1 = make_block_2d_copy_B(mma, cW1);
    auto copy_w3 = make_block_2d_copy_B(mma, cW3);
    auto prefetch_w1 = make_block_2d_prefetch(copy_w1);
    auto prefetch_w3 = make_block_2d_prefetch(copy_w3);
    auto thr_mma = mma.get_slice(local_id);
    auto thr_copy_w1 = copy_w1.get_slice(local_id);
    auto thr_copy_w3 = copy_w3.get_slice(local_id);
    auto tCrW1 = thr_mma.partition_sg_fragment_B(cBw_ktile);
    auto tCrW3 = thr_mma.partition_sg_fragment_B(cBw_ktile);
    auto tBrBw1 = thr_copy_w1.partition_sg_fragment_D(cBw_ktile);
    auto tBrBw3 = thr_copy_w3.partition_sg_fragment_D(cBw_ktile);
    auto tBgBw1 = thr_copy_w1.partition_S(gBw1_coord);
    auto tBgBw3 = thr_copy_w3.partition_S(gBw3_coord);
    auto pBgBw1 = prefetch_w1.get_slice(local_id).partition_S(gBw1_coord);
    auto pBgBw3 = prefetch_w3.get_slice(local_id).partition_S(gBw3_coord);

    if constexpr (IS_DECODE) {
      const int out_row = tile_token_offset[0] / K;
      const DTYPE* row_ptr = tokens + (int64_t)out_row * H;
      auto cA = make_tensor(
          make_gmem_ptr(row_ptr),
          make_layout(make_shape(_1{}, H), make_stride(H, _1{})));
      auto cA_full = make_identity_tensor(make_shape(_1{}, H));
      auto gA =
          local_tile(cA_full, select<0, 2>(mma.tile_mnk()), make_coord(_, _));
      auto copy_A = make_block_2d_copy_A(mma, cA);
      auto thr_copy_A = copy_A.get_slice(local_id);
      auto tArA = thr_copy_A.partition_sg_fragment_D(gA(_, _, _, 0));
      auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0, 0));
      auto tAgA = thr_copy_A.partition_S(gA);

      const int k_tile_count = H / (N_TK * TK);
      constexpr int DECODE_PREFETCH_COUNT = 4;
      CUTE_UNROLL
      for (int k_pref_tile_idx = 0; k_pref_tile_idx < DECODE_PREFETCH_COUNT;
           ++k_pref_tile_idx) {
        prefetch(
            make_block_2d_prefetch(copy_A),
            make_block_2d_prefetch(copy_A).get_slice(local_id).partition_S(gA)(
                _, _, _, _, k_pref_tile_idx));
        prefetch(prefetch_w1, pBgBw1(_, _, _, k_pref_tile_idx));
        prefetch(prefetch_w3, pBgBw3(_, _, _, k_pref_tile_idx));
      }
      CUTE_UNROLL
      for (int k_tile_idx = 0; k_tile_idx < k_tile_count; ++k_tile_idx) {
        const int k_pref_tile_idx = k_tile_idx + DECODE_PREFETCH_COUNT;
        if (k_pref_tile_idx < k_tile_count) {
          prefetch(
              make_block_2d_prefetch(copy_A),
              make_block_2d_prefetch(copy_A).get_slice(local_id).partition_S(
                  gA)(_, _, _, _, k_pref_tile_idx));
          prefetch(prefetch_w1, pBgBw1(_, _, _, k_pref_tile_idx));
          prefetch(prefetch_w3, pBgBw3(_, _, _, k_pref_tile_idx));
        }
        copy(copy_A, tAgA(_, _, _, _, k_tile_idx), tArA);
        reorder(tArA, tCrA);
        copy(copy_w1, tBgBw1(_, _, _, k_tile_idx), tBrBw1);
        reorder(tBrBw1, tCrW1);
        cute::gemm(mma, tCrA, tCrW1, tCrGate);
        copy(copy_w3, tBgBw3(_, _, _, k_tile_idx), tBrBw3);
        reorder(tBrBw3, tCrW3);
        cute::gemm(mma, tCrA, tCrW3, tCrUp);
      }
      {
        auto cC_full =
            make_identity_tensor(make_shape(_1{}, Int<SG_COUNT * TN>{}));
        auto tCcC = thr_mma.partition_C(cC_full);
        CUTE_UNROLL
        for (int i = 0; i < size(tCrGate); ++i) {
          const int n = cute::get<1>(tCcC(i));
          float g = float(tCrGate(i));
          float u = float(tCrUp(i));
          if constexpr (HAS_W13_BIAS) {
            g += w13_bias[(int64_t)expert_id * (2 * I) + n_idx + n];
            u += w13_bias[(int64_t)expert_id * (2 * I) + I + n_idx + n];
          }
          if constexpr (HAS_CLAMP_LIMIT) {
            g = sycl::fmin(g, gemm1_clamp_limit);
            u = sycl::fmax(
                sycl::fmin(u, gemm1_clamp_limit), -gemm1_clamp_limit);
          }
          slm_ptr[n] = DTYPE(silu_apply(g) * u);
        }
      }
      it.barrier(sycl::access::fence_space::local_space);
    } else {
      auto copy_slm = make_block_2d_slm_copy_A(mma);
      auto thr_copy_slm = copy_slm.get_slice(local_id);

      auto cSlm_full = make_tensor(
          make_smem_ptr(slm_ptr),
          make_shape(Int<N_TM * TM>{}, Int<SG_COUNT * TK>{}),
          make_stride(Int<SG_COUNT * TK>{}, _1{}));
      Tensor gSlm =
          local_tile(cSlm_full, select<0, 2>(wg_tile), make_coord(0, _));

      constexpr int K_STEPS = SG_COUNT / N_TK;
      const int k1_tile_count = H / (SG_COUNT * TK);
      const int k_tiles_total = k1_tile_count * K_STEPS;

      for (int k1_tile_idx = 0; k1_tile_idx < k1_tile_count; ++k1_tile_idx) {
        const int col_base = k1_tile_idx * (SG_COUNT * TK);
        load_hidden_states(
            sg_id, lane, tile_active, tile_token_offset, col_base, slm_ptr);
        it.barrier(sycl::access::fence_space::local_space);
        CUTE_UNROLL
        for (int k_step = 0; k_step < K_STEPS; ++k_step) {
          const int k_step_idx = k1_tile_idx * K_STEPS + k_step;
          auto gSlm_k = gSlm(_, _, k_step);
          auto tCrSlm = thr_mma.partition_sg_fragment_A(gSlm_k);
          auto tCrSlm_in = thr_copy_slm.retile_D(tCrSlm);
          auto tAsSlm_in = thr_copy_slm.partition_S(gSlm_k);
          copy(copy_slm, tAsSlm_in, tCrSlm_in);
          if (k_step_idx + 1 < k_tiles_total) {
            prefetch(prefetch_w1, pBgBw1(_, _, _, k_step_idx + 1));
            prefetch(prefetch_w3, pBgBw3(_, _, _, k_step_idx + 1));
          }
          copy(copy_w1, tBgBw1(_, _, _, k_step_idx), tBrBw1);
          reorder(tBrBw1, tCrW1);
          cute::gemm(mma, tCrSlm, tCrW1, tCrGate);
          copy(copy_w3, tBgBw3(_, _, _, k_step_idx), tBrBw3);
          reorder(tBrBw3, tCrW3);
          cute::gemm(mma, tCrSlm, tCrW3, tCrUp);
        }
        it.barrier(sycl::access::fence_space::local_space);
      }
      {
        auto cC_full = make_identity_tensor(
            make_shape(Int<N_TM * TM>{}, Int<SG_COUNT * TN>{}));
        auto tCcC = thr_mma.partition_C(cC_full);
        if constexpr (HAS_W13_BIAS) {
          CUTE_UNROLL
          for (int i = 0; i < size(tCrGate); ++i) {
            const int row = cute::get<0>(tCcC(i));
            const int col = cute::get<1>(tCcC(i));
            const float gate_bias =
                w13_bias[(int64_t)expert_id * (2 * I) + n_idx + col];
            const float up_bias =
                w13_bias[(int64_t)expert_id * (2 * I) + I + n_idx + col];
            float g = float(tCrGate(i)) + gate_bias;
            float u = float(tCrUp(i)) + up_bias;
            if constexpr (HAS_CLAMP_LIMIT) {
              g = sycl::fmin(g, gemm1_clamp_limit);
              u = sycl::fmax(
                  sycl::fmin(u, gemm1_clamp_limit), -gemm1_clamp_limit);
            }
            slm_ptr[row * SLM_STRIDE + col] = DTYPE(silu_apply(g) * u);
          }
        } else {
          CUTE_UNROLL
          for (int i = 0; i < size(tCrGate); ++i) {
            const int row = cute::get<0>(tCcC(i));
            const int col = cute::get<1>(tCcC(i));
            float g = float(tCrGate(i));
            float u = float(tCrUp(i));
            if constexpr (HAS_CLAMP_LIMIT) {
              g = sycl::fmin(g, gemm1_clamp_limit);
              u = sycl::fmax(
                  sycl::fmin(u, gemm1_clamp_limit), -gemm1_clamp_limit);
            }
            slm_ptr[row * SLM_STRIDE + col] = DTYPE(silu_apply(g) * u);
          }
        }
      }
      it.barrier(sycl::access::fence_space::local_space);
    }
  }

  template <typename TAccum, typename TCoord>
  __attribute__((always_inline)) void store_output(
      TAccum const& tCrDown,
      TCoord const& tCcC,
      int sg_id,
      int lane,
      int h_tile_idx,
      int expert_id,
      int n_idx,
      bool const tile_active[N_TM],
      int const tile_token_offset[N_TM],
      int const tile_token_count[N_TM]) const {
    CUTE_UNROLL
    for (int i = 0; i < size(tCrDown); ++i) {
      const int row = cute::get<0>(tCcC(i));
      const int col = cute::get<1>(tCcC(i));
      const int mt = row / TM;
      const int m = row % TM;
      if (!tile_active[mt] || m >= tile_token_count[mt]) continue;

      int topk_row_idx, slot_id;
      if constexpr (IS_DECODE) {
        topk_row_idx = tile_token_offset[mt];
        slot_id = topk_row_idx;
      } else {
        topk_row_idx = tile_token_offset[mt] + m;
        slot_id = topk_ids[topk_row_idx];
      }
      const float weight = topk_weights[topk_row_idx];

      const int abs_col = h_tile_idx + col;
      float val = float(tCrDown(i)) * weight;
      if constexpr (HAS_W2_BIAS)
        if (n_idx == 0)
          val += w2_bias[(int64_t)expert_id * H + abs_col] * weight;
      intermediate[(int64_t)slot_id * H + abs_col] += val;
    }
  }

  template <typename TiledMMA>
  __attribute__((always_inline)) void compute_gemm2(
      sycl::nd_item<2> const& it,
      TiledMMA const& mma,
      int local_id,
      int expert_id,
      bool const tile_active[N_TM],
      int const tile_token_offset[N_TM],
      int const tile_token_count[N_TM],
      const DTYPE* w2_expert,
      DTYPE* act_slm_ptr,
      int n_idx) const {
    auto wg_tile = mma.tile_mnk();
    const int sg_id = local_id / SG_SIZE;
    const int lane = local_id % SG_SIZE;
    auto thr_mma = mma.get_slice(local_id);

    auto copy_slm = make_block_2d_slm_copy_A(mma);
    auto thr_s2r_A = copy_slm.get_slice(local_id);
    auto cSlm = make_tensor(
        make_smem_ptr(act_slm_ptr),
        make_shape(Int<N_TM * TM>{}, Int<SG_COUNT * TK>{}),
        make_stride(Int<SG_COUNT * TK>{}, _1{}));
    Tensor gSlm = local_tile(cSlm, select<0, 2>(wg_tile), make_coord(0, _));

    constexpr int WG_N_TILE = N_TN * SG_COUNT * TN;  // 256
    constexpr int K_STEPS = SG_COUNT / N_TK;         // 4

    auto cBw2_ktile = make_identity_tensor(select<1, 2>(wg_tile));
    auto cBw2_full = make_identity_tensor(
        make_shape(Int<WG_N_TILE>{}, Int<SG_COUNT * TK>{}));
    auto gBw2_coord =
        local_tile(cBw2_full, select<1, 2>(wg_tile), make_coord(Int<0>{}, _));

    for (int h_tile_idx = 0; h_tile_idx < H; h_tile_idx += WG_N_TILE) {
      Tensor cW2 = make_tensor(
          make_gmem_ptr(w2_expert + h_tile_idx),
          make_layout(
              make_shape(Int<WG_N_TILE>{}, Int<SG_COUNT * TK>{}),
              make_stride(_1{}, (int)H)));
      auto copy_w2 = make_block_2d_copy_B(mma, cW2);
      auto thr_copy_w2 = copy_w2.get_slice(local_id);
      auto tCrW2 = thr_mma.partition_sg_fragment_B(cBw2_ktile);
      auto tBrBw2 = thr_copy_w2.partition_sg_fragment_D(cBw2_ktile);
      auto tBgBw2 = thr_copy_w2.partition_S(gBw2_coord);

      const int h_next = h_tile_idx + WG_N_TILE;
      if (h_next < H) {
        Tensor cW2_next = make_tensor(
            make_gmem_ptr(w2_expert + h_next),
            make_layout(
                make_shape(Int<WG_N_TILE>{}, Int<SG_COUNT * TK>{}),
                make_stride(_1{}, (int)H)));
        auto pref_w2 =
            make_block_2d_prefetch(make_block_2d_copy_B(mma, cW2_next));
        auto pW2gW2 = pref_w2.get_slice(local_id).partition_S(gBw2_coord);
        CUTE_UNROLL
        for (int k_pf = 0; k_pf < K_STEPS; ++k_pf)
          prefetch(pref_w2, pW2gW2(_, _, _, k_pf));
      }

      auto tCrDown = partition_fragment_C(mma, select<0, 1>(wg_tile));
      clear(tCrDown);

      CUTE_UNROLL
      for (int k_step = 0; k_step < K_STEPS; ++k_step) {
        auto gSlm_k = gSlm(_, _, k_step);
        auto tCrSlm = thr_mma.partition_sg_fragment_A(gSlm_k);
        auto tCrSlm_in = thr_s2r_A.retile_D(tCrSlm);
        auto tAsSlm_in = thr_s2r_A.partition_S(gSlm_k);
        copy(copy_slm, tAsSlm_in, tCrSlm_in);
        copy(copy_w2, tBgBw2(_, _, _, k_step), tBrBw2);
        reorder(tBrBw2, tCrW2);
        cute::gemm(mma, tCrSlm, tCrW2, tCrDown);
      }

      {
        auto cC_id = make_identity_tensor(select<0, 1>(wg_tile));
        auto tCcC = thr_mma.partition_C(cC_id);
        store_output(
            tCrDown,
            tCcC,
            sg_id,
            lane,
            h_tile_idx,
            expert_id,
            n_idx,
            tile_active,
            tile_token_offset,
            tile_token_count);
      }
    }

    it.barrier(sycl::access::fence_space::local_space);
  }

  template <int HEIGHT, int WIDTH = TN, typename TCAccum>
  __attribute__((always_inline)) void reduce_accumulate(
      int out_row,
      int sg_id,
      int lane,
      TCAccum& tCrOut,
      const int offset,
      int k_start) const {
    constexpr int N_OUT = WIDTH / SG_SIZE;
    const int64_t base_addr = ((int64_t)out_row * K + k_start) * H;

    auto coord_reduce =
        make_identity_tensor(make_shape(Int<WIDTH>{}, Int<HEIGHT>{}));

    const int offset_next = offset + SG_COUNT * WIDTH;

    if (offset_next < H) {
      auto gReduce_next = make_tensor(
          make_gmem_ptr(intermediate + base_addr + offset_next + sg_id * WIDTH),
          make_layout(
              make_shape(Int<WIDTH>{}, Int<HEIGHT>{}), make_stride(_1{}, H)));
      auto pref_next = make_block_2d_prefetch(make_block_2d_copy(
          XE_LOAD_2D<cute::sizeof_bits_v<float>, HEIGHT, WIDTH>{},
          gReduce_next));
      prefetch(pref_next, pref_next.get_slice(lane).partition_S(coord_reduce));
    }

    auto gReduce = make_tensor(
        make_gmem_ptr(intermediate + base_addr + offset + sg_id * WIDTH),
        make_layout(
            make_shape(Int<WIDTH>{}, Int<HEIGHT>{}), make_stride(_1{}, H)));
    auto copy_reduce = make_block_2d_copy(
        XE_LOAD_2D<cute::sizeof_bits_v<float>, HEIGHT, WIDTH>{}, gReduce);
    auto thr_copy_reduce = copy_reduce.get_slice(lane);
    auto tRrReduce = thr_copy_reduce.partition_fragment_D(coord_reduce);
    copy(copy_reduce, thr_copy_reduce.partition_S(coord_reduce), tRrReduce);

    if constexpr (HEIGHT == 1) {
      CUTE_UNROLL
      for (int n = 0; n < N_OUT; ++n) {
        tCrOut(n) = DTYPE(float(tCrOut(n)) + float(tRrReduce(n)));
      }
    } else {
      CUTE_UNROLL
      for (int n = 0; n < N_OUT; ++n) {
        float sum = 0.0f;
        CUTE_UNROLL
        for (int k = 0; k < HEIGHT; ++k)
          sum += float(tRrReduce(k * N_OUT + n));
        tCrOut(n) = DTYPE(float(tCrOut(n)) + sum);
      }
    }
  }

  __attribute__((always_inline)) void
  reduce_one_row(int out_row, int sg_id, int lane) const {
    constexpr int WIDTH = TN;
    constexpr int N_OUT = WIDTH / SG_SIZE;
    const int n_blocks = H / (SG_COUNT * WIDTH);
    auto coord_out = make_identity_tensor(make_shape(_1{}, Int<WIDTH>{}));

    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
      const int offset = block_idx * SG_COUNT * WIDTH;
      auto gOut = make_tensor(
          make_gmem_ptr(output + (int64_t)out_row * H + offset + sg_id * WIDTH),
          make_layout(make_shape(_1{}, Int<WIDTH>{}), make_stride(H, _1{})));
      auto copy_out = make_block_2d_copy(
          XE_STORE_2D<cute::sizeof_bits_v<DTYPE>, 1, WIDTH>{}, gOut);
      auto thr_copy_out = copy_out.get_slice(lane);
      auto tCrOut = thr_copy_out.partition_fragment_S(coord_out);

      clear(tCrOut);

      int k_start = 0;
      const int k_end = K;
      for (; k_start + 8 <= k_end; k_start += 8)
        reduce_accumulate<8>(out_row, sg_id, lane, tCrOut, offset, k_start);
      for (; k_start + 4 <= k_end; k_start += 4)
        reduce_accumulate<4>(out_row, sg_id, lane, tCrOut, offset, k_start);
      for (; k_start + 2 <= k_end; k_start += 2)
        reduce_accumulate<2>(out_row, sg_id, lane, tCrOut, offset, k_start);
      for (; k_start < k_end; ++k_start)
        reduce_accumulate<1>(out_row, sg_id, lane, tCrOut, offset, k_start);

      copy(copy_out, tCrOut, thr_copy_out.partition_D(coord_out));
    }
  }

  __attribute__((always_inline)) void reduce_output(
      sycl::nd_item<2> const& it,
      int local_id,
      int sg_id,
      int lane,
      bool const tile_active[N_TM],
      int const tile_token_offset[N_TM],
      int const tile_token_count[N_TM]) const {
    if constexpr (IS_DECODE) {
      const int out_row = 0;
      int count = 0;
      if (local_id == 0) {
        sycl::atomic_ref<
            int32_t,
            sycl::memory_order_acq_rel,
            sycl::memory_scope_device,
            sycl::access::address_space::global_space>
            ctr(row_counter[out_row]);
        count = ctr.fetch_add(1);
      }
      count = sycl::group_broadcast(it.get_group(), count, 0);
      if (count != K - 1) return;
      reduce_one_row(out_row, sg_id, lane);
    } else {
      CUTE_UNROLL
      for (int mt = 0; mt < N_TM; ++mt) {
        if (!tile_active[mt]) continue;
        for (int m = 0; m < tile_token_count[mt]; ++m) {
          int count = 0;
          if (local_id == 0) {
            sycl::atomic_ref<
                int32_t,
                sycl::memory_order_acq_rel,
                sycl::memory_scope_device,
                sycl::access::address_space::global_space>
                ctr(row_counter[(int)source_row[tile_token_offset[mt] + m]]);
            count = ctr.fetch_add(1);
          }
          count = sycl::group_broadcast(it.get_group(), count, 0);
          if (count != K - 1) continue;
          const int out_row = (int)source_row[tile_token_offset[mt] + m];
          reduce_one_row(out_row, sg_id, lane);
        }
      }
    }
  }

  void operator()(sycl::nd_item<2> it) const {
    const int wg_id = (int)it.get_group(1);

    if constexpr (IS_DECODE) {
      if (wg_id >= K) return;
    }

    auto sg = it.get_sub_group();
    const int sg_id = (int)sg.get_group_id()[0];
    const int lane = (int)sg.get_local_id()[0];
    const int local_id = sg_id * SG_SIZE + lane;

    bool tile_active[N_TM];
    int tile_token_offset[N_TM], tile_token_count[N_TM];
    int expert_id = compute_tile_info(
        wg_id, tile_active, tile_token_offset, tile_token_count);
    if (expert_id < 0) return;

    DTYPE* slm_ptr =
        slm.template get_multi_ptr<sycl::access::decorated::no>().get();
    const DTYPE* w13_expert = w13 + (int64_t)expert_id * H * (2 * I);
    const DTYPE* w2_expert = w2 + (int64_t)expert_id * I * H;

    auto gate_up_mma = make_gate_up_tiled_mma();
    auto down_mma = make_down_tiled_mma();

    for (int n_idx = 0; n_idx < I; n_idx += SG_COUNT * TK) {
      compute_gemm13_silu(
          it,
          gate_up_mma,
          local_id,
          sg_id,
          lane,
          n_idx,
          expert_id,
          w13_expert,
          tile_active,
          tile_token_offset,
          slm_ptr);
      compute_gemm2(
          it,
          down_mma,
          local_id,
          expert_id,
          tile_active,
          tile_token_offset,
          tile_token_count,
          w2_expert + (int64_t)n_idx * H,
          slm_ptr,
          n_idx);
    }

    reduce_output(
        it,
        local_id,
        sg_id,
        lane,
        tile_active,
        tile_token_offset,
        tile_token_count);
  }
};
