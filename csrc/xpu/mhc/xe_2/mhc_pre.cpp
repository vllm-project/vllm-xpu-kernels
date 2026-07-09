// DeepSeek V4 MHC pre-processing kernel with Split-K optimization.
//
// Two-path design:
//   Small M (< DISPATCH_THRESHOLD): vector dot-product path
//     → standalone Stage 2
//
//   Large M (≥ DISPATCH_THRESHOLD): Split-K DPAS GEMM + fused Reduce+Stage2
//     Pass 1: 3D nd_range GEMM kernel (N-groups, M-groups, n_splits)
//       - Each WG computes partial GEMM + partial sqrsum over K/n_splits tiles
//       - Writes partial_C to workspace_c[n_splits × M_padded, 24]
//       - Writes partial_sqrsum to workspace_sqr[n_splits × M_padded]
//     Pass 2: Fused Reduce + Stage 2 (1 WG per token, 256 threads)
//       - Phase 0: reduce partial_C and partial_sqrsum, apply RMS-norm → mixes
//       in SLM
//       - Phase 1: sigmoid + Sinkhorn (SG0 only)
//       - Phase 2: layer_input = sum(pre_mix * residual, dim=HC)
//
// Key advantages vs non-split-K:
//   - Split-K fills GPU cores for all M values (20 XE-cores on BMG)
//   - Fused reduce+Stage2 eliminates intermediate rms_mixes global memory
//   traffic
//   - Policy (BLK_M=16) maximizes occupancy along M dimension
//
// Data flow:
//   residual[M,HC,H] bf16 --+-- SplitK GEMM --> workspace_c[n_splits*M,24] fp32
//   fn[24,K] fp32 ----------+                   workspace_sqr[n_splits*M] fp32
//                                                       |
//   residual[M,HC,H] bf16 ---- Fused Reduce+Stage2 --> post_mix, comb_mix,
//   layer_input

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "utils.h"

#include "cutlass/device_kernel.h"
#include "cute/util/compat/device.hpp"
#include "cute/util/compat/dims.hpp"
#include "cute/util/compat/launch_policy.hpp"

#include <cute/tensor.hpp>
#include <cute/algorithm/subgroup_algorithms.hpp>
#include <cute/numeric/arithmetic_tuple.hpp>
#include <cutlass/kernel_hardware_info.h>

using namespace cute;

namespace {

// ===========================================================================
// Policy: BLK_M=16, BLK_N=32, BLK_K=32, 1×2 SGs (32 threads), pf=3
// Maximizes occupancy along M dimension for split-K GEMM.
// ===========================================================================
struct SplitKPolicy {
  using WGTile = Shape<_16, _32, _32>;
  using SGLayout = Layout<Shape<_1, _2, _1>, Stride<_2, _1, _0>>;
  static constexpr int prefetch_dist = 3;
};

// ===========================================================================
// Auto split-K selection heuristic
//
// Three regimes based on num_wg_m vs sm_count:
//   num_wg_m ≤ sm_count           → 32 splits (GPU severely underoccupied)
//   num_wg_m ≤ 4 * sm_count       → 4 splits  (moderate occupancy)
//   num_wg_m > 4 * sm_count       → 8 splits  (well-occupied, cache benefit)
//
// TODO: this heuristic is tuned on B60, may need adjustment for other GPUs.
// ===========================================================================
static int choose_n_splits(int M, int BLK_M) {
  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);
  TORCH_CHECK(
      hw_info.sm_count > 0, "Failed to query device multiprocessor count");

  int num_wg_m = (M + BLK_M - 1) / BLK_M;

  if (num_wg_m <= hw_info.sm_count)
    return 32;
  else if (num_wg_m <= 4 * hw_info.sm_count)
    return 4;
  else
    return 8;
}

// ===========================================================================
// Split-K GEMM + sqrsum device function
//
// Each WG computes partial GEMM and partial sqrsum over [k_start, k_end).
// Results are written to workspace buffers for later reduction.
//
// Uses 3D nd_range: dim0=N-groups(=1), dim1=M-groups, dim2=split-K groups.
// ===========================================================================
template <
    int PrefetchDist,
    class ATensor,
    class BTensor,
    class WSTensor,
    class TiledMMA>
void mhc_pre_splitk_device(
    ATensor const& A,  // (M, K) bf16
    BTensor const& B,  // (N, K) fp32
    WSTensor& WS,      // (n_splits * M_padded, N) fp32
    float* ws_sqr,     // [n_splits * M_padded] fp32
    TiledMMA const& mma,
    int n_splits,
    int num_wg_m,
    int M) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
  auto wg_n = int(BlockIdxX());      // N direction (always 0)
  auto wg_m = int(BlockIdxY());      // M direction
  auto split_id = int(BlockIdxZ());  // split-K direction
  auto local_id = int(ThreadIdxY());

  /* Compute K-tile range for this split */
  auto wg_tile = mma.tile_mnk();
  int total_k_tiles = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tiles_per_split = total_k_tiles / n_splits;
  int k_remainder = total_k_tiles % n_splits;
  int k_start, k_end;
  if (split_id < k_remainder) {
    k_start = split_id * (k_tiles_per_split + 1);
    k_end = k_start + k_tiles_per_split + 1;
  } else {
    k_start = k_remainder * (k_tiles_per_split + 1) +
              (split_id - k_remainder) * k_tiles_per_split;
    k_end = k_start + k_tiles_per_split;
  }

  /* Workspace WG-M coordinate: splits are stacked along M */
  int wg_m_ws = split_id * num_wg_m + wg_m;
  int BLK_M = int(get<0>(wg_tile));
  int M_padded = num_wg_m * BLK_M;

  /* Create proxy coordinate tensors */
  Tensor cA = make_identity_tensor(A.shape());    // (M, K)
  Tensor cB = make_identity_tensor(B.shape());    // (N, K)
  Tensor cWS = make_identity_tensor(WS.shape());  // (n_splits * M_padded, N)

  /* Tile A and B using actual wg_m (same rows for all splits) */
  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));

  /* Tile workspace using wg_m_ws (offset by split_id) */
  auto wg_coord_ws = make_coord(wg_m_ws, wg_n, 0);
  Tensor gWS = local_tile(cWS, wg_tile, wg_coord_ws, Step<_1, _1, X>{});

  /* Create block 2D TiledCopies */
  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_b = make_block_2d_copy_B(mma, B);
  auto copy_c = make_block_2d_copy_D(mma, WS);

  /* Slice TiledCopy/TiledMMA to thread level */
  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  /* Register fragments for MMA */
  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  /* Register fragments for copies */
  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  /* Partition global tensor for copies */
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  /* Partition workspace output */
  Tensor tCrC = partition_fragment_C(mma, select<0, 1>(wg_tile));
  Tensor tCgC = thr_mma.partition_C(gWS);

  /* Prefetch setup */
  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  constexpr int prefetch_dist = PrefetchDist;

  // sqrsum: per M-row accumulation from copy-A fragment
  auto local_sqr_sum = cute::make_subgroup_tensor(
      make_tensor<float>(tArA.layout()), tArA.tv_layout());
  CUTE_UNROLL
  for (int i = 0; i < local_sqr_sum.size(); i++) {
    local_sqr_sum(i) = 0.f;
  }

  // GEMM K-loop + sqrsum accumulation
  static constexpr SPIRVScope barrier_scope = ScopeWorkgroup;

  clear(tCrC);

  /* Prefetch warmup from k_start */
  int k_tile_prefetch = k_start;
  CUTE_UNROLL
  for (int i = 0; i < prefetch_dist && k_tile_prefetch < total_k_tiles;
       i++, k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  /* Main loop over K tiles [k_start, k_end) */
  for (int k_tile = k_start; k_tile < k_end; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    if (k_tile_prefetch < total_k_tiles) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    // sqrsum: accumulate squared values from copy-A fragment
    CUTE_UNROLL
    for (int frag_idx = 0; frag_idx < tArA.size(); frag_idx++) {
      float val = static_cast<float>(tArA(frag_idx));
      local_sqr_sum(frag_idx) += val * val;
    }

    gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);
  }

  /* Write partial GEMM result to workspace */
  copy(copy_c, tCrC, tCgC);

  /* Reduce sqrsum along K (mode 1) → 1 value per M-row per lane */
  constexpr auto tv_layout = local_sqr_sum.tv_layout();
  constexpr auto coshape = atuple_coshape(tv_layout);
  auto tmp = cute::make_subgroup_tensor(
      make_tensor<float>(local_sqr_sum.layout()),
      make_layout(coshape, make_stride(E<0>{}, E<1>{})));
  reorder(local_sqr_sum, tmp);
  auto reduced = cute::reduce<1>(tmp, sycl::plus<void>{});

  /* Write partial sqrsum to workspace (SG 0 only) */
  int sg_id = sg.get_group_id();
  int lane = sg.get_local_id()[0];
  if (sg_id == 0) {
    int m_row = wg_m * BLK_M + lane;
    if (m_row < M) {
      ws_sqr[split_id * M_padded + m_row] = reduced(0);
    }
  }
}

class MhcPreSplitKGemm;
class MhcPreFusedReduceStage2;

using bf16 = sycl::ext::oneapi::bfloat16;

using vllm::xpu::aligned_vec;

class MhcPreStage1Vector;

static inline float sigmoid(float x) {
  return 1.f / (1.f + sycl::native::exp(-x));
}

class MhcPreStage2;

class MhcPreStage1VectorFunctor {
 public:
  static constexpr int HC = 4;
  static constexpr int BLOCK_M = 2;
  static constexpr int BLOCK_N = 12;
  static constexpr int VEC_SIZE = 4;
  static constexpr int SG_SIZE = 16;
  static constexpr int WG_SIZE = 256;
  static constexpr int NUM_SG = WG_SIZE / SG_SIZE;
  static constexpr int HC_MULT3 = HC * (2 + HC);
  static_assert(HC_MULT3 % BLOCK_N == 0);
  static constexpr int NUM_N_BLOCKS = HC_MULT3 / BLOCK_N;
  static constexpr int NUM_REDUCE_VALUES = BLOCK_N + 1;

  struct Params {
    const bf16* residual;
    const float* fn;
    float* rms_mixes;
    int N;
    int H;
    int k_size;
    float rms_eps;
  };

  CUTLASS_DEVICE
  void operator()(const Params& p, char* smem_buf) const {
    using vec_bf16_t = aligned_vec<bf16, VEC_SIZE>;
    using vec_f32_t = aligned_vec<float, VEC_SIZE>;

    float* red_scratch = reinterpret_cast<float*>(smem_buf);

    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int wg_id = int(BlockIdxX());
    const int tid = int(ThreadIdxY());
    auto sg = item.get_sub_group();
    const int sg_id = sg.get_group_id()[0];
    const int sg_lane_id = sg.get_local_id()[0];
    const int token_base = wg_id * BLOCK_M;

#pragma unroll
    for (int block_idx = 0; block_idx < NUM_N_BLOCKS; ++block_idx) {
      float local_mixes[BLOCK_M][BLOCK_N];
      float local_sqrsum[BLOCK_M];
#pragma unroll
      for (int t = 0; t < BLOCK_M; ++t) {
        local_sqrsum[t] = 0.f;
#pragma unroll
        for (int j = 0; j < BLOCK_N; ++j)
          local_mixes[t][j] = 0.f;
      }

      for (int k_idx = tid * VEC_SIZE; k_idx < p.k_size;
           k_idx += WG_SIZE * VEC_SIZE) {
        vec_f32_t fn_tile[BLOCK_N];
#pragma unroll
        for (int j = 0; j < BLOCK_N; ++j) {
          fn_tile[j] = *reinterpret_cast<const vec_f32_t*>(
              p.fn + (block_idx * BLOCK_N + j) * p.k_size + k_idx);
        }

#pragma unroll
        for (int t = 0; t < BLOCK_M; ++t) {
          int token_idx = token_base + t;
          if (token_idx >= p.N) break;

          auto residual_vec = *reinterpret_cast<const vec_bf16_t*>(
              p.residual + token_idx * p.k_size + k_idx);
          float x_vec[VEC_SIZE];
#pragma unroll
          for (int v = 0; v < VEC_SIZE; ++v)
            x_vec[v] = float(residual_vec.val[v]);

#pragma unroll
          for (int v = 0; v < VEC_SIZE; ++v)
            local_sqrsum[t] += x_vec[v] * x_vec[v];

#pragma unroll
          for (int j = 0; j < BLOCK_N; ++j) {
            float acc = 0.f;
#pragma unroll
            for (int v = 0; v < VEC_SIZE; ++v)
              acc += x_vec[v] * fn_tile[j].val[v];
            local_mixes[t][j] += acc;
          }
        }
      }

#pragma unroll
      for (int t = 0; t < BLOCK_M; ++t) {
        int token_idx = token_base + t;
        if (token_idx >= p.N) break;

        float* token_scratch = &red_scratch[t * NUM_REDUCE_VALUES * NUM_SG];
        float sg_sum =
            sycl::reduce_over_group(sg, local_sqrsum[t], sycl::plus<float>());
        if (sg_lane_id == 0) token_scratch[sg_id * NUM_REDUCE_VALUES] = sg_sum;

#pragma unroll
        for (int j = 0; j < BLOCK_N; ++j) {
          float sg_mix = sycl::reduce_over_group(
              sg, local_mixes[t][j], sycl::plus<float>());
          if (sg_lane_id == 0)
            token_scratch[sg_id * NUM_REDUCE_VALUES + (j + 1)] = sg_mix;
        }
        sycl::group_barrier(item.get_group());

        if (sg_id == 0 && sg_lane_id == 0) {
#pragma unroll
          for (int j = 0; j < NUM_REDUCE_VALUES; ++j) {
            float s = 0.f;
#pragma unroll
            for (int sg_idx = 0; sg_idx < NUM_SG; ++sg_idx)
              s += token_scratch[sg_idx * NUM_REDUCE_VALUES + j];
            token_scratch[j] = s;
          }
        }
        sycl::group_barrier(item.get_group());

        float inv_rms =
            sycl::native::rsqrt(token_scratch[0] / p.k_size + p.rms_eps);
        if (tid < BLOCK_N)
          p.rms_mixes[token_idx * HC_MULT3 + block_idx * BLOCK_N + tid] =
              token_scratch[tid + 1] * inv_rms;
      }
    }
  }
};

template <class ATensor, class BTensor, class WSTensor, class TiledMMA>
class MhcPreSplitKGemmFunctor {
 public:
  struct Params {
    ATensor A_cute;
    BTensor B_cute;
    WSTensor WS_cute;
    float* ws_sqr;
    TiledMMA mma;
    int n_splits;
    int num_wg_m;
    int M;
  };

  CUTLASS_DEVICE
  void operator()(const Params& p, char*) const {
    mhc_pre_splitk_device<SplitKPolicy::prefetch_dist>(
        p.A_cute,
        p.B_cute,
        p.WS_cute,
        p.ws_sqr,
        p.mma,
        p.n_splits,
        p.num_wg_m,
        p.M);
  }
};

}  // namespace

// ===========================================================================
// Fused Reduce + Stage 2
//
// Combines split-K reduction with post-processing in a single kernel.
// Phase 0: ALL 256 threads load workspace → SLM, then 25 threads reduce
// Phase 1: sigmoid + Sinkhorn (SG0 only)
// Phase 2: layer_input = sum(pre_mix * residual, dim=HC)
//
// Grid: 1 WG per token, 256 threads per WG.
// ===========================================================================

void launch_mhc_pre_fused_reduce_stage2(
    sycl::queue& q,
    const float* ws_c,      // [n_splits * M_padded, HC3] fp32
    const float* ws_sqr,    // [n_splits * M_padded] fp32
    const bf16* residual,   // [N, HC, H] bf16
    const float* hc_scale,  // [3]
    const float* hc_base,   // [HC3]
    float* post_mix,        // [N, HC]
    float* comb_mix,        // [N, HC*HC]
    bf16* layer_input,      // [N, H]
    int num_tokens,
    int hidden_size,
    int n_splits,
    int M_padded,
    int K,
    float rms_eps,
    float hc_pre_eps,
    float hc_sinkhorn_eps,
    float hc_post_mult_value,
    int sinkhorn_repeat) {
  static constexpr int SG_SIZE = 16;
  static constexpr int WG_THREADS = 256;
  static constexpr int VEC = 8;
  static constexpr int HC = 4;
  static constexpr int HC3 = HC * (2 + HC);   // 24
  static constexpr int HC3_PLUS1 = HC3 + 1;   // 25 (24 mixes + 1 sqrsum)
  static constexpr int COMB_LANES = HC * HC;  // 16
  using vec_bf16_t = aligned_vec<bf16, VEC>;

  // Max supported n_splits for SLM sizing.
  static constexpr int MAX_SPLITS = 32;

  sycl::range<1> global(static_cast<size_t>(num_tokens) * WG_THREADS);
  sycl::range<1> local(WG_THREADS);

  q.submit([&](sycl::handler& h) {
    // SLM layout:
    //   [0 .. HC3]           → mixes_slm: 25 floats (24 mixes + 1 rms_coeff)
    //   [HC3+1 .. HC3+1 + MAX_SPLITS*HC3_PLUS1 - 1] → reduce_buf
    sycl::local_accessor<float, 1> slm(HC3_PLUS1 + MAX_SPLITS * HC3_PLUS1, h);

    h.parallel_for<MhcPreFusedReduceStage2>(
        sycl::nd_range<1>(global, local),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]]
        {
          const int tok = item.get_group(0);
          const int tid = item.get_local_id(0);
          auto sg = item.get_sub_group();
          int sg_id = sg.get_group_id();

          constexpr int REDUCE_BUF_OFF = HC3_PLUS1;

          // ============================================================
          // Phase 0, Step 1: ALL 256 threads load workspace → SLM
          // ============================================================
          const int total_items = n_splits * HC3_PLUS1;
          for (int i = tid; i < total_items; i += WG_THREADS) {
            int s = i / HC3_PLUS1;
            int c = i % HC3_PLUS1;
            float val;
            if (c < HC3) {
              val = ws_c[(s * M_padded + tok) * HC3 + c];
            } else {
              val = ws_sqr[s * M_padded + tok];
            }
            slm[REDUCE_BUF_OFF + i] = val;
          }
          sycl::group_barrier(item.get_group());

          // ============================================================
          // Phase 0, Step 2: Reduce across splits + apply RMS-norm
          // ============================================================
          float my_reduce_sum = 0.f;
          if (tid < HC3_PLUS1) {
            for (int s = 0; s < n_splits; s++) {
              my_reduce_sum += slm[REDUCE_BUF_OFF + s * HC3_PLUS1 + tid];
            }
            if (tid == HC3) {
              // Thread 24: sqrsum → rms_coeff
              slm[HC3] = sycl::native::rsqrt(
                  my_reduce_sum / static_cast<float>(K) + rms_eps);
            }
          }
          sycl::group_barrier(item.get_group());

          if (tid < HC3) {
            slm[tid] = my_reduce_sum * slm[HC3];
          }
          sycl::group_barrier(item.get_group());

          // ============================================================
          // Phase 1: sigmoid + Sinkhorn (SG0 only)
          // ============================================================

          if (sg_id == 0) {
            const int lane = sg.get_local_id()[0];

            if (lane < HC) {
              float pre_logits = slm[lane] * hc_scale[0] + hc_base[lane];
              float pre_mix_val = sigmoid(pre_logits) + hc_pre_eps;
              slm[lane] = pre_mix_val;

              float post_logits =
                  slm[lane + HC] * hc_scale[1] + hc_base[lane + HC];
              post_mix[tok * HC + lane] =
                  sigmoid(post_logits) * hc_post_mult_value;
            }

            int comb_idx = lane % COMB_LANES;
            float comb_logits = slm[comb_idx + 2 * HC] * hc_scale[2] +
                                hc_base[comb_idx + 2 * HC];

            float vmax = comb_logits;
#pragma unroll
            for (int off = 1; off < HC; off <<= 1) {
              float tmp = sycl::permute_group_by_xor(sg, vmax, off);
              vmax = sycl::max(vmax, tmp);
            }
            float comb_val = sycl::native::exp(comb_logits - vmax);

            float rsum = comb_val;
#pragma unroll
            for (int off = 1; off < HC; off <<= 1) {
              rsum += sycl::permute_group_by_xor(sg, rsum, off);
            }
            comb_val *= sycl::native::recip(rsum);
            comb_val += hc_sinkhorn_eps;

#pragma unroll
            for (int it_sk = 0; it_sk < sinkhorn_repeat; ++it_sk) {
              float col_sum = comb_val;
#pragma unroll
              for (int off = HC; off < COMB_LANES; off <<= 1) {
                col_sum += sycl::permute_group_by_xor(sg, col_sum, off);
              }
              comb_val *= sycl::native::recip(col_sum + hc_sinkhorn_eps);

              if (it_sk < sinkhorn_repeat - 1) {
                float row_sum = comb_val;
#pragma unroll
                for (int off = 1; off < HC; off <<= 1) {
                  row_sum += sycl::permute_group_by_xor(sg, row_sum, off);
                }
                comb_val *= sycl::native::recip(row_sum + hc_sinkhorn_eps);
              }
            }

            if (lane < COMB_LANES) {
              comb_mix[tok * COMB_LANES + lane] = comb_val;
            }
          }

          sycl::group_barrier(item.get_group());

          // ============================================================
          // Phase 2: layer_input = sum(pre_mix * residual, dim=HC)
          // ============================================================

          float pre_mix[HC];
#pragma unroll
          for (int m = 0; m < HC; ++m)
            pre_mix[m] = slm[m];

          for (int k = tid * VEC; k < hidden_size; k += WG_THREADS * VEC) {
            float acc[VEC];
#pragma unroll
            for (int v = 0; v < VEC; ++v)
              acc[v] = 0.f;

#pragma unroll
            for (int m = 0; m < HC; ++m) {
              auto rv = *reinterpret_cast<const vec_bf16_t*>(
                  residual + (tok * HC + m) * hidden_size + k);
#pragma unroll
              for (int v = 0; v < VEC; ++v)
                acc[v] += pre_mix[m] * float(rv.val[v]);
            }

            vec_bf16_t ov;
#pragma unroll
            for (int v = 0; v < VEC; ++v)
              ov.val[v] = static_cast<bf16>(acc[v]);
            *reinterpret_cast<vec_bf16_t*>(
                layer_input + tok * hidden_size + k) = ov;
          }
        });
  });
}

void launch_mhc_pre_stage1_vector(
    sycl::queue& q,
    const bf16* residual,
    const float* fn,
    float* rms_mixes,
    int N,
    int H,
    float rms_eps) {
  static constexpr int HC = 4;
  static constexpr int BLOCK_M = 2;
  static constexpr int BLOCK_N = 12;
  static constexpr int VEC_SIZE = 4;
  static constexpr int SG_SIZE = 16;
  static constexpr int WG_SIZE = 256;
  static constexpr int NUM_SG = WG_SIZE / SG_SIZE;
  static constexpr int HC_MULT3 = HC * (2 + HC);
  static_assert(HC_MULT3 % BLOCK_N == 0);
  static constexpr int NUM_N_BLOCKS = HC_MULT3 / BLOCK_N;
  static constexpr int NUM_REDUCE_VALUES = BLOCK_N + 1;
  const int k_size = HC * H;

  const size_t num_wgs = static_cast<size_t>((N + BLOCK_M - 1) / BLOCK_M);

  MhcPreStage1VectorFunctor::Params params{
      residual, fn, rms_mixes, N, H, k_size, rms_eps};

  const int smem_size = static_cast<int>(
      MhcPreStage1VectorFunctor::NUM_REDUCE_VALUES *
      MhcPreStage1VectorFunctor::NUM_SG * MhcPreStage1VectorFunctor::BLOCK_M *
      sizeof(float));

  const auto sycl_block = compat::dim3(1, WG_SIZE, 1);
  const auto sycl_grid = compat::dim3(static_cast<unsigned int>(num_wgs), 1, 1);

  compat::experimental::launch_properties launch_props{
      sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
  };
  compat::experimental::kernel_properties kernel_props{
      sycl::ext::oneapi::experimental::sub_group_size<16>};
  compat::experimental::launch_policy policy{
      sycl_grid, sycl_block, launch_props, kernel_props};
  compat::experimental::launch<
      cutlass::device_kernel<MhcPreStage1VectorFunctor>,
      MhcPreStage1VectorFunctor>(policy, q, params);
}

void launch_mhc_pre_stage2(
    sycl::queue& q,
    const float* rms_mixes,
    const bf16* residual,
    const float* hc_scale,
    const float* hc_base,
    float* post_mix,
    float* comb_mix,
    bf16* layer_input,
    int num_tokens,
    int hidden_size,
    float hc_pre_eps,
    float hc_sinkhorn_eps,
    float hc_post_mult_value,
    int sinkhorn_repeat) {
  static constexpr int SG_SIZE = 16;
  static constexpr int WG_THREADS = 256;
  static constexpr int VEC = 8;
  static constexpr int HC = 4;
  static constexpr int HC3 = HC * (2 + HC);
  static constexpr int COMB_LANES = HC * HC;
  using vec_bf16_t = aligned_vec<bf16, VEC>;

  sycl::range<1> global(static_cast<size_t>(num_tokens) * WG_THREADS);
  sycl::range<1> local(WG_THREADS);

  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> mixes_slm(HC3, h);

    h.parallel_for<MhcPreStage2>(
        sycl::nd_range<1>(global, local),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
          const int tok = item.get_group(0);
          const int tid = item.get_local_id(0);
          auto sg = item.get_sub_group();
          int sg_id = sg.get_group_id();

          if (tid < HC3) mixes_slm[tid] = rms_mixes[tok * HC3 + tid];
          sycl::group_barrier(item.get_group());

          if (sg_id == 0) {
            const int lane = sg.get_local_id()[0];
            if (lane < HC) {
              float pre_logits = mixes_slm[lane] * hc_scale[0] + hc_base[lane];
              float pre_mix_val = sigmoid(pre_logits) + hc_pre_eps;
              mixes_slm[lane] = pre_mix_val;

              float post_logits =
                  mixes_slm[lane + HC] * hc_scale[1] + hc_base[lane + HC];
              post_mix[tok * HC + lane] =
                  sigmoid(post_logits) * hc_post_mult_value;
            }

            int comb_idx = lane % COMB_LANES;
            float comb_logits = mixes_slm[comb_idx + 2 * HC] * hc_scale[2] +
                                hc_base[comb_idx + 2 * HC];

            float vmax = comb_logits;
#pragma unroll
            for (int off = 1; off < HC; off <<= 1) {
              float tmp = sycl::permute_group_by_xor(sg, vmax, off);
              vmax = sycl::max(vmax, tmp);
            }
            float comb_val = sycl::native::exp(comb_logits - vmax);

            float rsum = comb_val;
#pragma unroll
            for (int off = 1; off < HC; off <<= 1)
              rsum += sycl::permute_group_by_xor(sg, rsum, off);
            comb_val *= sycl::native::recip(rsum);
            comb_val += hc_sinkhorn_eps;

#pragma unroll
            for (int it_sk = 0; it_sk < sinkhorn_repeat; ++it_sk) {
              float col_sum = comb_val;
#pragma unroll
              for (int off = HC; off < COMB_LANES; off <<= 1)
                col_sum += sycl::permute_group_by_xor(sg, col_sum, off);
              comb_val *= sycl::native::recip(col_sum + hc_sinkhorn_eps);

              if (it_sk < sinkhorn_repeat - 1) {
                float row_sum = comb_val;
#pragma unroll
                for (int off = 1; off < HC; off <<= 1)
                  row_sum += sycl::permute_group_by_xor(sg, row_sum, off);
                comb_val *= sycl::native::recip(row_sum + hc_sinkhorn_eps);
              }
            }

            if (lane < COMB_LANES) comb_mix[tok * COMB_LANES + lane] = comb_val;
          }

          sycl::group_barrier(item.get_group());

          float pre_mix[HC];
#pragma unroll
          for (int m = 0; m < HC; ++m)
            pre_mix[m] = mixes_slm[m];

          for (int k = tid * VEC; k < hidden_size; k += WG_THREADS * VEC) {
            float acc[VEC];
#pragma unroll
            for (int v = 0; v < VEC; ++v)
              acc[v] = 0.f;

#pragma unroll
            for (int m = 0; m < HC; ++m) {
              auto rv = *reinterpret_cast<const vec_bf16_t*>(
                  residual + (tok * HC + m) * hidden_size + k);
#pragma unroll
              for (int v = 0; v < VEC; ++v)
                acc[v] += pre_mix[m] * float(rv.val[v]);
            }

            vec_bf16_t ov;
#pragma unroll
            for (int v = 0; v < VEC; ++v)
              ov.val[v] = static_cast<bf16>(acc[v]);
            *reinterpret_cast<vec_bf16_t*>(
                layer_input + tok * hidden_size + k) = ov;
          }
        });
  });
}

// Returns {n_splits, M_padded, K, N_gemm} for workspace allocation.
std::tuple<int, int, int, int> mhc_pre_splitk_params(int M, int H) {
  constexpr int HC_VAL = 4;
  constexpr int BLK_M = 16;  // SplitKPolicy WGTile M
  const int K = HC_VAL * H;
  const int N_gemm = HC_VAL * 2 + HC_VAL * HC_VAL;  // 24
  const int num_wg_m = (M + BLK_M - 1) / BLK_M;
  const int M_padded = num_wg_m * BLK_M;
  int n_splits = choose_n_splits(M, BLK_M);
  return {n_splits, M_padded, K, N_gemm};
}

void launch_mhc_pre_splitk_gemm(
    sycl::queue& queue,
    const bf16* residual,
    const float* fn,
    float* ws_c,    // [n_splits * M_padded, N_gemm]
    float* ws_sqr,  // [n_splits * M_padded]
    int M,
    int H,
    int n_splits,
    int M_padded) {
  constexpr int HC_VAL = 4;
  const int K = HC_VAL * H;
  const int N_gemm = HC_VAL * 2 + HC_VAL * HC_VAL;  // 24
  const int num_wg_m = M_padded / 16;               // BLK_M = 16

  using Policy = SplitKPolicy;
  using MMAOp = XE_DPAS_TT<8, float, tfloat32_t>;
  using MMAAtom = MMA_Atom<MMAOp>;
  using TiledMMA_t = typename TiledMMAHelper<
      MMAAtom,
      Layout<typename Policy::WGTile>,
      typename Policy::SGLayout>::TiledMMA;
  TiledMMA_t mma;

  auto wg_tile = mma.tile_mnk();

  auto A_cute = make_tensor(
      make_gmem_ptr(reinterpret_cast<const bfloat16_t*>(residual)),
      make_layout(make_shape(M, K), make_stride(K, Int<1>{})));

  auto B_cute = make_tensor(
      make_gmem_ptr(fn),
      make_layout(make_shape(N_gemm, K), make_stride(K, Int<1>{})));

  auto WS_cute = make_tensor(
      make_gmem_ptr(ws_c),
      make_layout(
          make_shape(n_splits * M_padded, N_gemm),
          make_stride(N_gemm, Int<1>{})));

  const auto block_x = static_cast<unsigned int>(size(mma));
  const auto grid_x =
      static_cast<unsigned int>(ceil_div(shape<0>(B_cute), get<1>(wg_tile)));
  const auto grid_y = static_cast<unsigned int>(num_wg_m);
  const auto grid_z = static_cast<unsigned int>(n_splits);

  typename MhcPreSplitKGemmFunctor<
      decltype(A_cute),
      decltype(B_cute),
      decltype(WS_cute),
      TiledMMA_t>::Params params{
      A_cute, B_cute, WS_cute, ws_sqr, mma, n_splits, num_wg_m, M};

  const auto sycl_block = compat::dim3(1, block_x, 1);
  const auto sycl_grid = compat::dim3(grid_x, grid_y, grid_z);

  compat::experimental::launch_properties launch_props{
      sycl::ext::oneapi::experimental::work_group_scratch_size(0),
  };
  compat::experimental::kernel_properties kernel_props{
      sycl::ext::oneapi::experimental::sub_group_size<16>,
      sycl::ext::intel::experimental::grf_size<256>};
  compat::experimental::launch_policy policy{
      sycl_grid, sycl_block, launch_props, kernel_props};
  compat::experimental::launch<
      cutlass::device_kernel<MhcPreSplitKGemmFunctor<
          decltype(A_cute),
          decltype(B_cute),
          decltype(WS_cute),
          TiledMMA_t>>,
      MhcPreSplitKGemmFunctor<
          decltype(A_cute),
          decltype(B_cute),
          decltype(WS_cute),
          TiledMMA_t>>(policy, queue, params);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> mhc_pre(
    const at::Tensor& residual,
    const at::Tensor& fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    double rms_eps,
    double hc_pre_eps,
    double hc_sinkhorn_eps,
    double hc_post_mult_value,
    int64_t sinkhorn_repeat) {
  TORCH_CHECK(
      residual.is_xpu() && fn.is_xpu() && hc_scale.is_xpu() && hc_base.is_xpu(),
      "mhc_pre: tensors must be on XPU");
  TORCH_CHECK(
      residual.scalar_type() == at::kBFloat16, "residual must be bfloat16");
  TORCH_CHECK(fn.scalar_type() == at::kFloat, "fn must be float32");
  TORCH_CHECK(hc_scale.scalar_type() == at::kFloat, "hc_scale must be float32");
  TORCH_CHECK(hc_base.scalar_type() == at::kFloat, "hc_base must be float32");

  auto residual_c = residual.contiguous();
  auto fn_c = fn.contiguous();

  const int64_t HC = residual_c.size(-2);
  const int64_t H = residual_c.size(-1);
  TORCH_CHECK(
      H % 8 == 0,
      "mhc_pre: hidden_size must be a multiple of 8 for vectorized "
      "loads/stores");
  const int64_t HC2 = HC * HC;
  const int64_t HC3 = HC * 2 + HC2;

  TORCH_CHECK(HC == 4, "mhc_pre: only hc_mult=4 is supported");
  TORCH_CHECK(
      fn_c.size(0) == HC3 && fn_c.size(1) == HC * H, "fn shape mismatch");
  TORCH_CHECK(hc_scale.numel() == 3, "hc_scale must have 3 elements");
  TORCH_CHECK(hc_base.numel() == HC3, "hc_base must have HC3 elements");

  auto outer_shape = residual_c.sizes().slice(0, residual_c.dim() - 2).vec();
  auto residual_flat = residual_c.view({-1, HC, H});
  const int64_t num_tokens = residual_flat.size(0);

  auto opts_f32 = residual_c.options().dtype(at::kFloat);
  auto opts_bf16 = residual_c.options().dtype(at::kBFloat16);

  auto post_mix = at::empty({num_tokens, HC}, opts_f32);
  auto comb_mix = at::empty({num_tokens, HC2}, opts_f32);
  auto layer_input = at::empty({num_tokens, H}, opts_bf16);

  if (num_tokens == 0) {
    std::vector<int64_t> ps = outer_shape;
    ps.push_back(HC);
    ps.push_back(1);
    std::vector<int64_t> cs = outer_shape;
    cs.push_back(HC);
    cs.push_back(HC);
    std::vector<int64_t> ls = outer_shape;
    ls.push_back(H);
    return {post_mix.view(ps), comb_mix.view(cs), layer_input.view(ls)};
  }

  auto& queue = vllm::xpu::vllmGetQueue();
  const int M_GEMM = static_cast<int>(num_tokens);
  const int K_GEMM = static_cast<int>(HC * H);
  const int N_GEMM = static_cast<int>(HC3);
  (void)K_GEMM;

  // =====================================================================
  // Dispatch: vector path for small M, split-K DPAS for large M
  // =====================================================================

  static constexpr int DISPATCH_THRESHOLD = 128;
  const bool use_tf32 = vllm::xpu::mhc_use_tf32();

  if (M_GEMM < DISPATCH_THRESHOLD || !use_tf32) {
    // --- Small M: vector dot-product path → standalone Stage 2 ---
    auto rms_mixes = at::empty({M_GEMM, N_GEMM}, opts_f32);

    launch_mhc_pre_stage1_vector(
        queue,
        reinterpret_cast<const bf16*>(residual_flat.data_ptr()),
        fn_c.data_ptr<float>(),
        rms_mixes.data_ptr<float>(),
        M_GEMM,
        static_cast<int>(H),
        static_cast<float>(rms_eps));

    launch_mhc_pre_stage2(
        queue,
        rms_mixes.data_ptr<float>(),
        reinterpret_cast<const bf16*>(residual_flat.data_ptr()),
        hc_scale.data_ptr<float>(),
        hc_base.data_ptr<float>(),
        post_mix.data_ptr<float>(),
        comb_mix.data_ptr<float>(),
        reinterpret_cast<bf16*>(layer_input.data_ptr()),
        static_cast<int>(num_tokens),
        static_cast<int>(H),
        static_cast<float>(hc_pre_eps),
        static_cast<float>(hc_sinkhorn_eps),
        static_cast<float>(hc_post_mult_value),
        static_cast<int>(sinkhorn_repeat));
  } else {
    // --- Large M: Split-K DPAS GEMM → Fused Reduce+Stage2 ---
    auto [n_splits, M_padded, K_GEMM_val, N_GEMM_val] =
        mhc_pre_splitk_params(M_GEMM, static_cast<int>(H));
    auto workspace_c = at::empty({n_splits * M_padded, N_GEMM_val}, opts_f32);
    auto workspace_sqr = at::empty({n_splits * M_padded}, opts_f32);

    launch_mhc_pre_splitk_gemm(
        queue,
        reinterpret_cast<const bf16*>(residual_flat.data_ptr()),
        fn_c.data_ptr<float>(),
        workspace_c.data_ptr<float>(),
        workspace_sqr.data_ptr<float>(),
        M_GEMM,
        static_cast<int>(H),
        n_splits,
        M_padded);

    launch_mhc_pre_fused_reduce_stage2(
        queue,
        workspace_c.data_ptr<float>(),
        workspace_sqr.data_ptr<float>(),
        reinterpret_cast<const bf16*>(residual_flat.data_ptr()),
        hc_scale.data_ptr<float>(),
        hc_base.data_ptr<float>(),
        post_mix.data_ptr<float>(),
        comb_mix.data_ptr<float>(),
        reinterpret_cast<bf16*>(layer_input.data_ptr()),
        M_GEMM,
        static_cast<int>(H),
        n_splits,
        M_padded,
        K_GEMM_val,
        static_cast<float>(rms_eps),
        static_cast<float>(hc_pre_eps),
        static_cast<float>(hc_sinkhorn_eps),
        static_cast<float>(hc_post_mult_value),
        static_cast<int>(sinkhorn_repeat));
  }

  std::vector<int64_t> ps = outer_shape;
  ps.push_back(HC);
  ps.push_back(1);
  std::vector<int64_t> cs = outer_shape;
  cs.push_back(HC);
  cs.push_back(HC);
  std::vector<int64_t> ls = outer_shape;
  ls.push_back(H);
  return {post_mix.view(ps), comb_mix.view(cs), layer_input.view(ls)};
}
