// DeepSeek V4 MHC pre-processing kernel with Split-K optimization.
//
// Two-path design; both feed the same split-K workspace and Pass 2:
//   Small M (< DISPATCH_THRESHOLD): split-K vector dot-product path
//     Pass 1: 2D nd_range GEMV kernel (M-groups, n_splits)
//
//   Large M (≥ DISPATCH_THRESHOLD): Split-K DPAS GEMM
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
  // Cache the multiprocessor count: the runtime query enumerates the device
  // and is comparatively expensive, so it must not run on every invocation.
  static const int sm_count = [] {
    cutlass::KernelHardwareInfo hw_info;
    int count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
        hw_info.device_id);
    TORCH_CHECK(count > 0, "Failed to query device multiprocessor count");
    return count;
  }();

  int num_wg_m = (M + BLK_M - 1) / BLK_M;

  if (num_wg_m <= sm_count)
    return 32;
  else if (num_wg_m <= 4 * sm_count)
    return 4;
  else
    return 8;
}

// ===========================================================================
// Split-K selection for the vector (GEMV) stage 1.
//
// The vector path has no N-direction parallelism at all, so the token blocks
// alone give 1-2 work-groups at decode shapes. Aim for ~4 work-groups per
// Xe-core, then clamp so each split still has at least one full WG_SIZE
// iteration of work, and so n_splits stays within the MAX_SPLITS = 32 SLM
// budget of launch_mhc_pre_fused_reduce_stage2.
// ===========================================================================
static int choose_n_splits_vector(int num_wg_m, int k_vecs, int wg_size) {
  // Cached for the same reason as in choose_n_splits().
  static const int sm_count = [] {
    cutlass::KernelHardwareInfo hw_info;
    int count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
        hw_info.device_id);
    TORCH_CHECK(count > 0, "Failed to query device multiprocessor count");
    return count;
  }();

  constexpr int MAX_SPLITS = 32;  // matches launch_mhc_pre_fused_reduce_stage2

  int n_splits = (4 * sm_count + num_wg_m - 1) / num_wg_m;
  int cap = k_vecs / wg_size;
  if (n_splits > cap) n_splits = cap;
  if (n_splits > MAX_SPLITS) n_splits = MAX_SPLITS;
  if (n_splits < 1) n_splits = 1;
  return n_splits;
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

// ===========================================================================
// Split-K vector (GEMV) stage 1 for the small-M path.
//
// Computes the same partial products as the DPAS split-K GEMM and writes the
// same workspace layout, so both paths share launch_mhc_pre_fused_reduce_stage2
// for the cross-split reduction, the RMS-norm and stage 2.
//
// Parallelism: 2D grid of (token blocks) x (K splits). Without the K split the
// grid is ceil(N / BLOCK_M) work-groups, which is 1-2 at decode shapes and
// leaves ~90% of the Xe-cores idle on a 20-core part.
//
// BLOCK_N covers all HC3 output columns in one pass so `fn` (24 x HC*H fp32,
// 2.75 MB) is streamed exactly once per token block instead of once per
// N-block. `fn` is loaded one column at a time: holding the whole column tile
// in registers on top of the BLOCK_M x BLOCK_N accumulators overflows the GRF.
// ===========================================================================
class MhcPreStage1VectorFunctor {
 public:
  static constexpr int HC = 4;
  static constexpr int BLOCK_M = 4;
  static constexpr int VEC_SIZE = 4;
  static constexpr int SG_SIZE = 16;
  static constexpr int WG_SIZE = 256;
  static constexpr int NUM_SG = WG_SIZE / SG_SIZE;
  static constexpr int HC_MULT3 = HC * (2 + HC);
  static constexpr int BLOCK_N = HC_MULT3;
  // Per token: BLOCK_N mixes followed by one sqrsum.
  static constexpr int NUM_REDUCE_VALUES = BLOCK_N + 1;
  static constexpr int SCRATCH_PER_SG = BLOCK_M * NUM_REDUCE_VALUES;
  static constexpr int SCRATCH_FLOATS = NUM_SG * SCRATCH_PER_SG;
  static_assert(SCRATCH_PER_SG <= WG_SIZE, "final reduce needs one thread per "
                                           "reduced value");

  struct Params {
    const bf16* residual;
    const float* fn;
    float* ws_c;    // [n_splits * M_padded, HC_MULT3] fp32
    float* ws_sqr;  // [n_splits * M_padded] fp32
    int N;
    int H;
    int k_size;
    int n_splits;
    int M_padded;
  };

  CUTLASS_DEVICE
  void operator()(const Params& p, char* smem_buf) const {
    using vec_bf16_t = aligned_vec<bf16, VEC_SIZE>;
    using vec_f32_t = aligned_vec<float, VEC_SIZE>;

    float* red_scratch = reinterpret_cast<float*>(smem_buf);

    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int wg_m = int(BlockIdxX());
    const int split_id = int(BlockIdxY());
    const int tid = int(ThreadIdxY());
    auto sg = item.get_sub_group();
    const int sg_id = sg.get_group_id()[0];
    const int sg_lane_id = sg.get_local_id()[0];

    const int token_base = wg_m * BLOCK_M;
    const int num_tok =
        (p.N - token_base) < BLOCK_M ? (p.N - token_base) : BLOCK_M;

    /* K range of this split, in VEC_SIZE-element units; the first kv_rem
       splits take one extra vector. */
    const int k_vecs = p.k_size / VEC_SIZE;
    const int kv_per = k_vecs / p.n_splits;
    const int kv_rem = k_vecs % p.n_splits;
    const int kv_start =
        split_id * kv_per + (split_id < kv_rem ? split_id : kv_rem);
    const int kv_end = kv_start + kv_per + (split_id < kv_rem ? 1 : 0);

    float local_mixes[BLOCK_M][BLOCK_N];
    float local_sqrsum[BLOCK_M];
#pragma unroll
    for (int t = 0; t < BLOCK_M; ++t) {
      local_sqrsum[t] = 0.f;
#pragma unroll
      for (int j = 0; j < BLOCK_N; ++j)
        local_mixes[t][j] = 0.f;
    }

    for (int kv = kv_start + tid; kv < kv_end; kv += WG_SIZE) {
      const int k_idx = kv * VEC_SIZE;

      float x_vec[BLOCK_M][VEC_SIZE];
#pragma unroll
      for (int t = 0; t < BLOCK_M; ++t) {
        if (t >= num_tok) break;
        auto residual_vec = *reinterpret_cast<const vec_bf16_t*>(
            p.residual + (token_base + t) * p.k_size + k_idx);
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v)
          x_vec[t][v] = float(residual_vec.val[v]);
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v)
          local_sqrsum[t] += x_vec[t][v] * x_vec[t][v];
      }

      // One fn column at a time: only a single vec_f32_t is live per step.
#pragma unroll
      for (int j = 0; j < BLOCK_N; ++j) {
        const vec_f32_t fn_vec =
            *reinterpret_cast<const vec_f32_t*>(p.fn + j * p.k_size + k_idx);
#pragma unroll
        for (int t = 0; t < BLOCK_M; ++t) {
          if (t >= num_tok) break;
          float acc = 0.f;
#pragma unroll
          for (int v = 0; v < VEC_SIZE; ++v)
            acc += x_vec[t][v] * fn_vec.val[v];
          local_mixes[t][j] += acc;
        }
      }
    }

    /* Sub-group reduce, then one barrier for the whole work-group. The
       reduce_over_group calls must stay convergent, so the inactive token
       slots participate too (their accumulators are zero). */
#pragma unroll
    for (int t = 0; t < BLOCK_M; ++t) {
      float* dst = &red_scratch[sg_id * SCRATCH_PER_SG + t * NUM_REDUCE_VALUES];
#pragma unroll
      for (int j = 0; j < BLOCK_N; ++j) {
        float sg_mix =
            sycl::reduce_over_group(sg, local_mixes[t][j], sycl::plus<float>());
        if (sg_lane_id == 0) dst[j] = sg_mix;
      }
      float sg_sqr =
          sycl::reduce_over_group(sg, local_sqrsum[t], sycl::plus<float>());
      if (sg_lane_id == 0) dst[BLOCK_N] = sg_sqr;
    }

    sycl::group_barrier(item.get_group());

    /* One thread per (token, reduced value) sums across the NUM_SG partials. */
    if (tid < BLOCK_M * NUM_REDUCE_VALUES) {
      const int t = tid / NUM_REDUCE_VALUES;
      const int j = tid - t * NUM_REDUCE_VALUES;
      if (t < num_tok) {
        float s = 0.f;
#pragma unroll
        for (int sg_idx = 0; sg_idx < NUM_SG; ++sg_idx)
          s += red_scratch[sg_idx * SCRATCH_PER_SG + t * NUM_REDUCE_VALUES + j];

        const int row = split_id * p.M_padded + token_base + t;
        if (j < BLOCK_N)
          p.ws_c[row * HC_MULT3 + j] = s;
        else
          p.ws_sqr[row] = s;
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
// Shared tuning constants for the layer_input write path.
//
// MAX_TILES bounds the per-thread element count for the *fused-norm* path
// only: it covers hidden_size up to WG_THREADS * VEC * MAX_TILES (= 8192 for
// 256x8x4), which covers the range used by current models (4096 needs 2
// tiles, 7168 needs 4). Keeping this as tight as possible is critical for
// MBU: every extra tile is VEC live bf16 registers per thread held across the
// reduction barrier, which inflates GRF pressure and drops occupancy in the
// fused-norm path. If a larger hidden size is ever needed for the fused path,
// template this kernel on the tile count and select it on the host from
// hidden_size (in-lambda branching does not reduce the GRF footprint). The
// host guards hidden_size <= WG_THREADS * VEC * MAX_TILES for the fused path.
//
// The non-fused path does NOT use this bound: it stores straight to HBM with
// no register stash, so it uses an unbounded strided loop that handles any
// hidden_size.
// ===========================================================================
static constexpr int MHC_PRE_OUT_MAX_TILES = 4;

// ===========================================================================
// Pre-barrier prefetch state for the layer_input write path.
//
// Holds the loads that do NOT depend on pre_mix (and therefore do not depend
// on the Sinkhorn phase), so they can be issued before the work-group barrier
// that separates Phase 1 from Phase 2:
//   - residual tile 0: only the *multiply* needs pre_mix, the load does not.
//     Restricting the early load to a single tile keeps the extra register
//     pressure at HC*VEC bf16 (32 values) instead of HC*VEC*MAX_TILES.
//   - norm_weight tiles: token-independent, so the whole set is hoisted and
//     its latency is hidden behind Sinkhorn + the sumsq work-group reduction.
// ===========================================================================
template <int HC_, int VEC_, int MAX_TILES_, int WG_THREADS_>
struct MhcPreOutPrefetch {
  using vec_bf16_t = aligned_vec<bf16, VEC_>;
  vec_bf16_t res0[HC_];        // residual tile 0 (k = tid*VEC)
  vec_bf16_t wgt[MAX_TILES_];  // norm_weight tiles (fused-norm path only)
  bool has_res0;
};

// Issue the pre_mix-independent loads. Call this BEFORE the Phase-1 barrier so
// the memory latency overlaps with the SG0-only Sinkhorn computation, during
// which the remaining sub-groups would otherwise sit idle on the barrier.
template <int HC_, int VEC_, int MAX_TILES_, int WG_THREADS_>
static inline void mhc_pre_prefetch_out_tiles(
    int tid,
    const bf16* residual,
    const bf16* norm_weight,
    int tok,
    int hidden_size,
    MhcPreOutPrefetch<HC_, VEC_, MAX_TILES_, WG_THREADS_>& pf) {
  using vec_bf16_t = aligned_vec<bf16, VEC_>;
  constexpr int STRIDE = WG_THREADS_ * VEC_;

  const int k0 = tid * VEC_;
  pf.has_res0 = (k0 < hidden_size);
  if (pf.has_res0) {
#pragma unroll
    for (int m = 0; m < HC_; ++m) {
      pf.res0[m] = *reinterpret_cast<const vec_bf16_t*>(
          residual + (tok * HC_ + m) * hidden_size + k0);
    }
  }

  if (norm_weight != nullptr) {
#pragma unroll
    for (int t = 0; t < MAX_TILES_; ++t) {
      const int k = k0 + t * STRIDE;
      if (k >= hidden_size) break;
      pf.wgt[t] = *reinterpret_cast<const vec_bf16_t*>(norm_weight + k);
    }
  }
}

// ===========================================================================
// layer_input write path, with optional fused trailing RMSNorm.
//
// Baseline (norm_weight == nullptr):
//   out = bf16(sum_hc(pre_mix * residual))
//   Uses an unbounded strided loop — no register stash, no GRF/MAX_TILES
//   budget constraint, supports any hidden_size.
//
// Fused (norm_weight != nullptr) — folds in the RMSNorm (attn_norm/ffn_norm)
// that would otherwise run as a separate kernel:
//   ol   = bf16(sum_hc(pre_mix * residual))   (matches the bf16 tensor a
//                                              standalone RMSNorm would see)
//   rms  = rsqrt(mean(ol^2) + norm_eps)
//   out  = ol * rms * norm_weight
//   Uses a MAX_TILES-bounded unrolled loop; the unnormalized bf16 result is
//   stashed in registers so residual is read exactly once from HBM with no
//   SLM round-trip. The host guards hidden_size <= WG_THREADS*VEC*MAX_TILES.
//
// Both paths are software-pipelined: the residual for tile t+1 is issued
// before the multiply-accumulate for tile t (cur/nxt double buffer), and tile
// 0 comes from the pre-barrier prefetch. This is the equivalent of a
// num_stages=2 pipeline and costs only HC*VEC extra bf16 registers, so it does
// not disturb the MAX_TILES GRF budget of the fused-norm path.
//
// Tuned for Intel Xe (B70): 256-thread work-group, one work-group per token,
// sub-group + SLM two-level reduction, 128-bit vectorized HBM traffic.
// ===========================================================================
template <int HC_, int VEC_, int MAX_TILES_, int WG_THREADS_, int SG_SIZE_>
static inline void mhc_pre_write_layer_input(
    const sycl::nd_item<1>& item,
    const sycl::sub_group& sg,
    const bf16* residual,
    const bf16* norm_weight,
    bf16* layer_input,
    float* norm_red,
    const float (&pre_mix)[HC_],
    int tok,
    int hidden_size,
    float norm_eps,
    const MhcPreOutPrefetch<HC_, VEC_, MAX_TILES_, WG_THREADS_>& pf) {
  using vec_bf16_t = aligned_vec<bf16, VEC_>;
  constexpr int NUM_SG = WG_THREADS_ / SG_SIZE_;
  constexpr int STRIDE = WG_THREADS_ * VEC_;

  const int tid = static_cast<int>(item.get_local_id(0));

  // Double-buffered residual registers: `cur` is the tile being consumed,
  // `nxt` is the tile whose load has already been issued.
  vec_bf16_t cur[HC_];
  vec_bf16_t nxt[HC_];
  if (pf.has_res0) {
#pragma unroll
    for (int m = 0; m < HC_; ++m)
      cur[m] = pf.res0[m];
  }

  // -------------------------------------------------------------------------
  // Non-fused path: unbounded strided loop — no register stash needed, so
  // there is no GRF / MAX_TILES budget constraint.  Supports any hidden_size.
  // -------------------------------------------------------------------------
  if (norm_weight == nullptr) {
    for (int k = tid * VEC_; k < hidden_size; k += STRIDE) {
      const int k_next = k + STRIDE;
      if (k_next < hidden_size) {
#pragma unroll
        for (int m = 0; m < HC_; ++m) {
          nxt[m] = *reinterpret_cast<const vec_bf16_t*>(
              residual + (tok * HC_ + m) * hidden_size + k_next);
        }
      }

      float acc[VEC_];
#pragma unroll
      for (int v = 0; v < VEC_; ++v)
        acc[v] = 0.f;
#pragma unroll
      for (int m = 0; m < HC_; ++m) {
#pragma unroll
        for (int v = 0; v < VEC_; ++v)
          acc[v] += pre_mix[m] * float(cur[m].val[v]);
      }

      vec_bf16_t ov;
#pragma unroll
      for (int v = 0; v < VEC_; ++v)
        ov.val[v] = static_cast<bf16>(acc[v]);
      *reinterpret_cast<vec_bf16_t*>(layer_input + tok * hidden_size + k) = ov;

      if (k_next < hidden_size) {
#pragma unroll
        for (int m = 0; m < HC_; ++m)
          cur[m] = nxt[m];
      }
    }
    return;
  }

  // -------------------------------------------------------------------------
  // Fused-norm path: MAX_TILES-bounded unrolled loop. Stashes the bf16-
  // rounded result in registers and accumulates the squared sum from the
  // *rounded* value so the numerics match the unfused "write bf16 -> RMSNorm
  // reads it back" sequence exactly.
  // hidden_size <= WG_THREADS * VEC * MAX_TILES is enforced on the host.
  // -------------------------------------------------------------------------
  const int sg_id = static_cast<int>(sg.get_group_id()[0]);
  const int lane = static_cast<int>(sg.get_local_id()[0]);
  bf16 stash[MAX_TILES_ * VEC_];
  float partial_sumsq = 0.f;

#pragma unroll
  for (int t = 0; t < MAX_TILES_; ++t) {
    const int k = tid * VEC_ + t * STRIDE;
    if (k >= hidden_size) break;

    // Issue the next tile's loads before consuming the current one.
    const int k_next = k + STRIDE;
    if (k_next < hidden_size) {
#pragma unroll
      for (int m = 0; m < HC_; ++m) {
        nxt[m] = *reinterpret_cast<const vec_bf16_t*>(
            residual + (tok * HC_ + m) * hidden_size + k_next);
      }
    }

    float acc[VEC_];
#pragma unroll
    for (int v = 0; v < VEC_; ++v)
      acc[v] = 0.f;

#pragma unroll
    for (int m = 0; m < HC_; ++m) {
#pragma unroll
      for (int v = 0; v < VEC_; ++v)
        acc[v] += pre_mix[m] * float(cur[m].val[v]);
    }

#pragma unroll
    for (int v = 0; v < VEC_; ++v) {
      bf16 b = static_cast<bf16>(acc[v]);
      stash[t * VEC_ + v] = b;
      float bf = static_cast<float>(b);
      partial_sumsq += bf * bf;
    }

    if (k_next < hidden_size) {
#pragma unroll
      for (int m = 0; m < HC_; ++m)
        cur[m] = nxt[m];
    }
  }

  // Work-group reduction of the squared sum: sub-group reduce, then a single
  // barrier; every thread re-reads the NUM_SG partials and computes rnorm
  // locally (cheap) so no second barrier is needed.
  float sg_sum =
      sycl::reduce_over_group(sg, partial_sumsq, sycl::plus<float>());
  if (lane == 0) norm_red[sg_id] = sg_sum;
  sycl::group_barrier(item.get_group());
  float tot = 0.f;
#pragma unroll
  for (int s = 0; s < NUM_SG; ++s)
    tot += norm_red[s];
  const float rnorm =
      sycl::native::rsqrt(tot / static_cast<float>(hidden_size) + norm_eps);

  // Pass 2: scale the register stash by rms * norm_weight and store to HBM.
  // norm_weight already resides in registers from the pre-barrier prefetch.
#pragma unroll
  for (int t = 0; t < MAX_TILES_; ++t) {
    const int k = tid * VEC_ + t * STRIDE;
    if (k >= hidden_size) break;
    const vec_bf16_t wv = pf.wgt[t];
    vec_bf16_t outv;
#pragma unroll
    for (int v = 0; v < VEC_; ++v) {
      float o = static_cast<float>(stash[t * VEC_ + v]);
      outv.val[v] = static_cast<bf16>(o * rnorm * float(wv.val[v]));
    }
    *reinterpret_cast<vec_bf16_t*>(layer_input + tok * hidden_size + k) = outv;
  }
}

// ===========================================================================
// Fused Reduce + Stage 2
//
// Combines split-K reduction with post-processing in a single kernel.
// Phase 0: ALL 256 threads load workspace → SLM, then 25 threads reduce
// Phase 1: sigmoid + Sinkhorn (SG0 only), overlapped with the pre_mix-
//          independent residual / norm_weight loads issued by every thread
// Phase 2: layer_input = sum(pre_mix * residual, dim=HC) [+ fused RMSNorm]
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
    int sinkhorn_repeat,
    const bf16* norm_weight,
    float norm_eps) {
  static constexpr int SG_SIZE = 16;
  static constexpr int WG_THREADS = 256;
  static constexpr int VEC = 8;
  static constexpr int HC = 4;
  static constexpr int HC3 = HC * (2 + HC);   // 24
  static constexpr int HC3_PLUS1 = HC3 + 1;   // 25 (24 mixes + 1 sqrsum)
  static constexpr int COMB_LANES = HC * HC;  // 16
  static constexpr int MAX_TILES = MHC_PRE_OUT_MAX_TILES;

  // Max supported n_splits for SLM sizing.
  static constexpr int MAX_SPLITS = 32;

  sycl::range<1> global(static_cast<size_t>(num_tokens) * WG_THREADS);
  sycl::range<1> local(WG_THREADS);

  q.submit([&](sycl::handler& h) {
    // SLM layout:
    //   [0 .. HC3]           → mixes_slm: 25 floats (24 mixes + 1 rms_coeff)
    //   [HC3+1 .. HC3+1 + MAX_SPLITS*HC3_PLUS1 - 1] → reduce_buf
    sycl::local_accessor<float, 1> slm(HC3_PLUS1 + MAX_SPLITS * HC3_PLUS1, h);
    // Fused-RMSNorm scratch: per-sub-group squared-sum reduction buffer.
    sycl::local_accessor<float, 1> norm_red(WG_THREADS / SG_SIZE, h);

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

          // ------------------------------------------------------------
          // Pre-barrier prefetch: issue the residual tile-0 and the
          // norm_weight loads now. They do not depend on the Sinkhorn
          // result, so their latency is absorbed by the SG0-only Phase 1
          // below plus the barrier that follows it.
          // ------------------------------------------------------------
          MhcPreOutPrefetch<HC, VEC, MAX_TILES, WG_THREADS> pf;
          mhc_pre_prefetch_out_tiles<HC, VEC, MAX_TILES, WG_THREADS>(
              tid, residual, norm_weight, tok, hidden_size, pf);

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
          //   with optional fused trailing RMSNorm.
          // ============================================================

          float pre_mix[HC];
#pragma unroll
          for (int m = 0; m < HC; ++m)
            pre_mix[m] = slm[m];

          mhc_pre_write_layer_input<HC, VEC, MAX_TILES, WG_THREADS, SG_SIZE>(
              item,
              sg,
              residual,
              norm_weight,
              layer_input,
              &norm_red[0],
              pre_mix,
              tok,
              hidden_size,
              norm_eps,
              pf);
        });
  });
}

// Returns {n_splits, M_padded} for the vector stage-1 workspace allocation.
std::tuple<int, int> mhc_pre_vector_splitk_params(int M, int H) {
  using F = MhcPreStage1VectorFunctor;
  const int k_size = F::HC * H;
  const int k_vecs = k_size / F::VEC_SIZE;
  const int num_wg_m = (M + F::BLOCK_M - 1) / F::BLOCK_M;
  const int M_padded = num_wg_m * F::BLOCK_M;
  const int n_splits = choose_n_splits_vector(num_wg_m, k_vecs, F::WG_SIZE);
  return {n_splits, M_padded};
}

void launch_mhc_pre_stage1_vector(
    sycl::queue& q,
    const bf16* residual,
    const float* fn,
    float* ws_c,    // [n_splits * M_padded, HC3]
    float* ws_sqr,  // [n_splits * M_padded]
    int N,
    int H,
    int n_splits,
    int M_padded) {
  using F = MhcPreStage1VectorFunctor;
  const int k_size = F::HC * H;
  const int num_wg_m = M_padded / F::BLOCK_M;

  F::Params params{
      residual, fn, ws_c, ws_sqr, N, H, k_size, n_splits, M_padded};

  const int smem_size =
      static_cast<int>(F::SCRATCH_FLOATS * sizeof(float));

  const auto sycl_block = compat::dim3(1, F::WG_SIZE, 1);
  const auto sycl_grid = compat::dim3(
      static_cast<unsigned int>(num_wg_m),
      static_cast<unsigned int>(n_splits),
      1);

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
    int64_t sinkhorn_repeat,
    const std::optional<at::Tensor>& norm_weight,
    double norm_eps) {
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

  // Optional trailing RMSNorm weight (attn_norm / ffn_norm) to fuse into the
  // layer_input write path. nullptr keeps the standalone (non-fused) behavior.
  const bf16* norm_weight_ptr = nullptr;
  at::Tensor norm_weight_c;
  if (norm_weight.has_value() && norm_weight->defined()) {
    norm_weight_c = norm_weight->contiguous();
    TORCH_CHECK(
        norm_weight_c.scalar_type() == at::kBFloat16,
        "mhc_pre: norm_weight must be bfloat16");
    TORCH_CHECK(
        norm_weight_c.numel() == H,
        "mhc_pre: norm_weight must have hidden_size elements");
    TORCH_CHECK(
        H <= 8192,
        "mhc_pre: fused norm supports hidden_size <= 8192 "
        "(WG_THREADS * VEC * MAX_TILES); template the stash tile count for "
        "larger sizes");
    norm_weight_ptr = reinterpret_cast<const bf16*>(norm_weight_c.data_ptr());
  }

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

  // =====================================================================
  // Dispatch: vector path for small M, split-K DPAS for large M
  // =====================================================================

  static constexpr int DISPATCH_THRESHOLD = 128;
  const bool use_tf32 = vllm::xpu::mhc_use_tf32();

  if (M_GEMM < DISPATCH_THRESHOLD || !use_tf32) {
    // --- Small M: split-K vector dot-product path → Fused Reduce+Stage2 ---
    auto [n_splits, M_padded] =
        mhc_pre_vector_splitk_params(M_GEMM, static_cast<int>(H));
    auto workspace_c = at::empty({n_splits * M_padded, N_GEMM}, opts_f32);
    auto workspace_sqr = at::empty({n_splits * M_padded}, opts_f32);

    launch_mhc_pre_stage1_vector(
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
        K_GEMM,
        static_cast<float>(rms_eps),
        static_cast<float>(hc_pre_eps),
        static_cast<float>(hc_sinkhorn_eps),
        static_cast<float>(hc_post_mult_value),
        static_cast<int>(sinkhorn_repeat),
        norm_weight_ptr,
        static_cast<float>(norm_eps));
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
        static_cast<int>(sinkhorn_repeat),
        norm_weight_ptr,
        static_cast<float>(norm_eps));
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