/***************************************************************************************************
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Impl3: shape-specialized fused MoE forward (routed + shared expert) for the
 * Qwen3.5-35B-A3B / TP2 decode ULT (H=2048, I_local=256, E=256, top_k=8,
 * fp16 act x fp8-e4m3 per-tensor(per-expert) weights, SiLU).
 *
 * Design (vs the generic `cutlass_grouped_gemm_xe2_impl` path used by
 * XpuFusedMoe):
 *   - ONE torch op (`moe_qwen35_sp::fused_forward`) does the whole forward in
 *     8 device kernel launches (vs impl2's 13 op dispatches) -> attacks the
 *     ~250us host-bound floor of the standard-interface composition.
 *   - The grouped GEMMs reuse the UNCHANGED sycl-tla mainloop `MoE::xe_gemm`
 *     (cute tensors, block-2D copies, XE_DPAS_TT atoms) but replace the
 *     persistent whole-GPU + global-atomic work-stealing scheduler
 *     (`MoE::MoEGEMM`'s O(E) per-work-group expert rescan + atomicAdd per
 *     stolen tile) with a DIRECT-INDEXED compact tile map:
 *       kernel 1 (map)   : topk_ids -> per-m-tile (expert, row_base, rows),
 *                          route->row perm table, permuted routing weights.
 *                          Single work-group.
 *       kernel 2 (gather): x -> expert-major A_perm (work-proportional).
 *       gemm work-group b decodes its tile DIRECTLY as
 *          (m_tile, n_tile) = (b / n_tiles, b % n_tiles), then
 *          (expert, row0, rows) = mtile_[m_tile] -- no scan, no atomics.
 *       Grid is the host-computable upper bound min(R, E) m-tiles x n_tiles;
 *       surplus work-groups exit on one cached scalar read.
 *   - The routing weight (topk_weights) is applied in the finalize kernel
 *     (impl1 applies it in its down kernel; impl2 at moe_gather) -- this
 *     keeps `xe_gemm` byte-for-byte unmodified (its epilogue has no per-row
 *     scalar hook; adding one would mean editing the shared mainloop file).
 *   - Shared expert (num_shared_experts=1, DeepSeek/Qwen3-Next convention):
 *     dense GEMMs through the SAME tile launcher in SINGLE_GROUP mode
 *     (one expert panel, A = x / shared act directly -- no map/gather), and
 *     its activation rows live in the same buffers as the routed ones so one
 *     silu_mul launch covers both. gate = sigmoid(dot(x, gate_vec)) is
 *     computed redundantly-but-parallel in finalize (impl1's down_finalize
 *     does the same; its separate down_gate_precompute kernel with 1
 *     work-item/token costs ~8us).
 *
 * Interface contract (Python side, fused_moe_interface_specialized.py):
 * XpuFusedMoeSpecialized is a drop-in for XpuFusedMoe -- same 15-param
 * __init__ prefix, same apply(output, hidden_states, topk_weights, topk_ids,
 * expert_map=None, a1q_scale=None) semantics (weighted+reduced result
 * written into the pre-allocated output; no return). Shared-expert weights
 * are accepted as optional trailing kwargs (None -> routed-only, exactly
 * impl2's behavior).
 **************************************************************************************************/
#include <torch/all.h>
#include <c10/xpu/XPUStream.h>

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"

#include "gemm_xe2_policy.hpp"
#include "grouped_gemm_xe2.hpp"  // MoE::make_moe_tensor + MoE::xe_gemm (unchanged mainloop)

namespace moe_sp {
using namespace cute;

namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

using ElementA = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementS = float;
using ElementBI = cutlass::half_t;

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

constexpr int MAX_E = 512;  // routing supports up to 512 experts (E=256 here)
constexpr int MAX_R =
    512;  // upper bound on routed rows handled by the map kernel
constexpr int GEMM_WG = 64;  // size(mma) for both m_8 and m_16 policies
constexpr int MAP_WG = 64;

inline bool num_experts_static_ok(int64_t e) { return e <= MAX_E; }

// ---------------------------------------------------------------------------
// Kernel name tags (unique types per instantiation)
// ---------------------------------------------------------------------------
template <typename IDT>
class SpMapName;
class SpGatherName;
class SpSiluMulName;
class SpFinalizeName;
template <class Policy, bool SingleGroup>
class SpGemmName;

// ---------------------------------------------------------------------------
// Kernel 1: map build. ONE work-group of MAP_WG lanes.
//   outputs: mt_expert[umt], mt_row_base[umt], mt_rows[umt], total_mt[1],
//            perm_pos[R] (route -> expert-major row), w_perm[R] (fp32).
// local mem: rows[MAX_E], cursor[MAX_E], offs[MAX_E+1], tbase[MAX_E+1].
// ---------------------------------------------------------------------------
template <typename IDT>
void launch_map_build(
    sycl::queue& q,
    const IDT* ids,
    const float* weights,
    int num_tokens,
    int top_k,
    int num_experts,
    int tile_m,
    int umt,
    int32_t* mt_expert,
    int32_t* mt_row_base,
    int32_t* mt_rows,
    int32_t* total_mt,
    int32_t* perm_pos,
    float* w_perm) {
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> slm(
        sycl::range<1>(MAX_R + 2 * MAX_E + 2 * (MAX_E + 1)), cgh);
    cgh.parallel_for<SpMapName<IDT>>(
        sycl::nd_range<1>(sycl::range<1>(MAP_WG), sycl::range<1>(MAP_WG)),
        [=](sycl::nd_item<1> item) {
          int32_t* ids_slm =
              slm.get_multi_ptr<sycl::access::decorated::no>().get();
          int32_t* rows = ids_slm + MAX_R;
          int32_t* cursor = rows + MAX_E;
          int32_t* offs = cursor + MAX_E;
          int32_t* tbase = offs + (MAX_E + 1);
          const int R = num_tokens * top_k;
          const int lane = item.get_local_linear_id();

          // Cooperative coalesced load of topk_ids into local memory first:
          // the naive per-lane rescans of global ids[] serialize one ~600ns
          // memory round-trip per element and cost ~40us at R=16.
          for (int r = lane; r < R; r += MAP_WG)
            ids_slm[r] = (int32_t)ids[r];
          sycl::group_barrier(item.get_group());

          const int tm_shift = (tile_m == 8) ? 3 : 4;  // tile_m is 2^k; GPU
          // integer division is emulated (~100 cycles) and the serial prefix
          // does one per expert -- 256 divs cost ~12us, shifts cost nothing.
          for (int e = lane; e < num_experts; e += MAP_WG) {
            int c = 0;
            for (int r = 0; r < R; ++r)
              c += (ids_slm[r] == e) ? 1 : 0;
            rows[e] = c;
            cursor[e] = 0;
          }
          sycl::group_barrier(item.get_group());

          if (lane == 0) {
            int off = 0, tb = 0;
            offs[0] = 0;
            tbase[0] = 0;
            for (int e = 0; e < num_experts; ++e) {
              const int r_e = rows[e];
              off += r_e;
              tb += (r_e > 0) ? ((r_e + tile_m - 1) >> tm_shift) : 0;
              offs[e + 1] = off;
              tbase[e + 1] = tb;
            }
            total_mt[0] = tb;
          }
          sycl::group_barrier(item.get_group());

          for (int e = lane; e < num_experts; e += MAP_WG) {
            const int r = rows[e];
            const int b = tbase[e];
            for (int t = 0; t < ((r + tile_m - 1) >> tm_shift); ++t) {
              mt_expert[b + t] = e;
              mt_row_base[b + t] = offs[e] + t * tile_m;
              const int rem = r - t * tile_m;
              mt_rows[b + t] = rem < tile_m ? rem : tile_m;
            }
          }

          for (int r = lane; r < R; r += MAP_WG) {
            const int e = ids_slm[r];
            sycl::atomic_ref<
                int,
                sycl::memory_order::relaxed,
                sycl::memory_scope::work_group,
                sycl::access::address_space::local_space>
                cur(cursor[e]);
            const int p = offs[e] + cur.fetch_add(1);
            perm_pos[r] = p;
            w_perm[p] = weights[r];
          }

          // Shared-expert gate: sigmoid(dot(x[m], gate_vec)) per token,
          // vectorized (16 halves / 32B per load), computed here so the
          // finalize kernel is a pure weighted sum. Skipped entirely when
          // there is no shared expert (gate_out stays unwritten then).
        });
  });
}

// ---------------------------------------------------------------------------
// Kernel 2: gather. x[token(r)] -> A_perm[perm_pos[r]], chunked over K.
// ---------------------------------------------------------------------------
class SpGatherLauncher {
 public:
  static void launch(
      sycl::queue& q,
      const ElementA* x,
      const int32_t* perm_pos,
      int R,
      int K,
      int top_k,
      ElementA* a_perm) {
    // One element per work-item, 2D grid (route, K-chunk of 4) so the
    // work-item indices need no integer division; R*K fully-coalesced.
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<SpGatherName>(
          sycl::nd_range<2>(sycl::range<2>(R, K / 4), sycl::range<2>(1, 64)),
          [=](sycl::nd_item<2> item) {
            const int r = item.get_global_id(0);
            const int c = item.get_global_id(1) * 4;
            const int dst = perm_pos[r];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              a_perm[(int64_t)dst * K + c + j] =
                  x[(int64_t)(r / top_k) * K + c + j];
            }
          });
    });
  }
};

// ---------------------------------------------------------------------------
// Kernels 3/5/6/8: the specialized grouped-GEMM tile launcher. Reuses the
// unchanged `MoE::xe_gemm` mainloop; replaces the persistent+atomic
// scheduler with direct tile-map indexing.
//   SINGLE_GROUP=false: routed grouped GEMM over the tile map.
//   SINGLE_GROUP=true : one dense panel (shared expert); A is already
//                       contiguous, expert id 0, grid = exact tile count.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Merged gemm launcher: routed tiles (direct-indexed map) + shared panel
// (single group) in ONE grid, ONE submit per gemm stage. The two segments
// share the same N/K/policy but differ in A/B/D/scale pointers.
// ---------------------------------------------------------------------------
template <class Policy, typename ElementB>
void launch_sp_gemm_merged(
    sycl::queue& q,
    const void* A_routed,
    const void* B_routed,
    const ElementS* scales_routed,
    void* D_routed,
    const int32_t* mt_expert,
    const int32_t* mt_row_base,
    const int32_t* mt_rows,
    const int32_t* total_mt,
    int umt,
    const void* A_shared,
    const void* B_shared,
    const ElementS* scales_shared,
    void* D_shared,
    int gemm_m_shared,
    int N,
    int K,
    bool has_shared) {
  auto op = XE_DPAS_TT<8, float, ElementA>{};
  using WGTile = typename Policy::WGTile;
  using SGLayout = typename Policy::SGLayout;
  using MMA = typename TiledMMAHelper<
      MMA_Atom<decltype(op)>,
      Layout<WGTile>,
      SGLayout>::TiledMMA;
  MMA mma{};

  const int wg_tile_m = get<0>(WGTile{});
  const int wg_tile_n = get<1>(WGTile{});
  const int n_tiles = ceil_div(N, wg_tile_n);
  const int shared_m_tiles =
      has_shared ? ceil_div(gemm_m_shared, wg_tile_m) : 0;
  const int total_wgs = umt * n_tiles + shared_m_tiles * n_tiles;

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<SpGemmName<Policy, false>>(
        sycl::nd_range<3>(
            sycl::range<3>(1, total_wgs, GEMM_WG),
            sycl::range<3>(1, 1, GEMM_WG)),
        syclex::properties{syclex::sub_group_size<16>, intelex::grf_size<256>},
        [=](sycl::nd_item<3> item) {
          const int tid = item.get_group_linear_id();
          // Grid layout: [routed upper-bound umt*n_tiles | shared n_tiles].
          // WGs between the ACTUAL routed tile count (device-known
          // total_mt[0]) and the umt upper bound must EXIT -- they are
          // surplus routed tiles, NOT shared tiles (mishandling them read
          // out of bounds and broke correctness).
          const int shared_base = umt * n_tiles;
          const int routed_live = total_mt[0] * n_tiles;
          if (tid < shared_base && tid >= routed_live)
            return;  // surplus routed
          const bool is_routed = tid < shared_base;
          const int seg_tile = is_routed ? tid : tid - shared_base;
          const int m_tile = seg_tile / n_tiles;
          const int n_tile = seg_tile % n_tiles;

          const void* A = is_routed ? A_routed : A_shared;
          const void* B = is_routed ? B_routed : B_shared;
          const ElementS* sc = is_routed ? scales_routed : scales_shared;
          void* D = is_routed ? D_routed : D_shared;
          int e = 0, row0 = 0, rows;
          if (is_routed) {
            e = mt_expert[m_tile];
            row0 = mt_row_base[m_tile];
            rows = mt_rows[m_tile];
          } else {
            rows = gemm_m_shared;
          }

          auto A_t = MoE::make_moe_tensor<ElementA, 'R'>(
              (ElementA*)A + (int64_t)row0 * K, rows, K);
          auto B_t = MoE::make_moe_tensor<ElementB, 'C'>(
              (ElementB*)B + (int64_t)e * N * K, N, K);
          auto D_t = MoE::make_moe_tensor<ElementD, 'R'>(
              (ElementD*)D + (int64_t)row0 * N, rows, N);
          auto tile_coord = make_coord(0, n_tile, _, 0);

          MoE::xe_gemm<void, void, void>(
              A_t,
              B_t,
              sc + e,
              (const ElementBI*)nullptr,
              D_t,
              tile_coord,
              mma);
        });
  });
}

// ---------------------------------------------------------------------------
// Kernel 6: SiLU-and-mul over a COMBINED buffer of [R + M] rows (routed act
// rows first, shared-expert rows after) -- one launch for both activations.
//   in  [rows, 2*ic] -> out [rows, ic] : silu(in[:, :ic]) * in[:, ic:]
// ---------------------------------------------------------------------------
class SpSiluMulLauncher {
 public:
  // Silu-and-mul over a COMBINED buffer of [R + M] rows, plus (when the
  // shared expert is present) the shared-expert gate dots for the M tokens,
  // all in ONE launch: work-groups [0, silu_wgs) do silu rows, work-groups
  // [silu_wgs, silu_wgs + num_tokens) each cooperatively reduce one token's
  // gate dot (64 lanes x vec16 loads + SLM reduce). Folding the gate here
  // (a) removes ~2-8us from the single-work-group map kernel and (b) keeps
  // the kernel count at 6.
  static void launch(
      sycl::queue& q,
      const ElementA* in,
      ElementA* out,
      int rows,
      int ic,
      const ElementA* x,
      const ElementA* gate_vec,
      float* gate_out,
      int num_tokens,
      int h_size,
      bool has_shared) {
    constexpr int VEC = 8;
    const int cols = ic / VEC;
    const int total = rows * cols;
    const int silu_wgs = (total + 63) / 64;
    const int wgs = silu_wgs + (has_shared ? num_tokens : 0);
    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> red(sycl::range<1>(64), cgh);
      cgh.parallel_for<SpSiluMulName>(
          sycl::nd_range<1>(sycl::range<1>(wgs * 64), sycl::range<1>(64)),
          [=](sycl::nd_item<1> item) {
            const int g = item.get_group_linear_id();
            const int lane = item.get_local_linear_id();
            if (g < silu_wgs) {
              const int i = g * 64 + lane;
              if (i >= total) return;
              const int row = i / cols;
              const int c = (i % cols) * VEC;
              const ElementA* gp = in + (int64_t)row * (2 * ic) + c;
              const ElementA* u = gp + ic;
              ElementA* o = out + (int64_t)row * ic + c;
#pragma unroll
              for (int j = 0; j < VEC; ++j) {
                const float gv = (float)gp[j];
                const float sv = gv / (1.f + sycl::exp(-gv));
                o[j] = ElementA(sv * (float)u[j]);
              }
            } else {
              // cooperative gate dot for token (g - silu_wgs)
              struct alignas(32) h16 {
                ElementA v[16];
              };
              const int m = g - silu_wgs;
              const h16* xv =
                  reinterpret_cast<const h16*>(x + (int64_t)m * h_size);
              const h16* wv = reinterpret_cast<const h16*>(gate_vec);
              const int n_blocks = h_size / 16;
              float part = 0.f;
              for (int idx = lane; idx < n_blocks; idx += 64) {
                const h16 a = xv[idx];
                const h16 b = wv[idx];
#pragma unroll
                for (int t = 0; t < 16; ++t) {
                  part += (float)a.v[t] * (float)b.v[t];
                }
              }
              red[lane] = part;
              sycl::group_barrier(item.get_group());
              if (lane == 0) {
                float acc = 0.f;
                for (int l = 0; l < 64; ++l)
                  acc += red[l];
                gate_out[m] = 1.f / (1.f + sycl::exp(-acc));
              }
            }
          });
    });
  }
};

// ---------------------------------------------------------------------------
// Kernel 8: finalize. out[m, c:c+8] = sum_k w_perm[perm_pos[m,k]] *
// D2[perm_pos[m,k], c:c+8] + sigmoid(dot(x[m], gate_vec)) * sh2[m, c:c+8]
// ---------------------------------------------------------------------------
class SpFinalizeLauncher {
 public:
  static void launch(
      sycl::queue& q,
      const float* gate,
      const int32_t* perm_pos,
      const float* w_perm,
      const ElementA* d2,
      const ElementA* sh2,
      ElementD* out,
      int num_tokens,
      int H,
      int top_k,
      bool has_shared) {
    constexpr int VEC = 8;
    const int cols = H / VEC;
    const int total = num_tokens * cols;
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<SpFinalizeName>(
          sycl::nd_range<1>(
              sycl::range<1>(((total + 63) / 64) * 64), sycl::range<1>(64)),
          [=](sycl::nd_item<1> item) {
            const int i = item.get_global_linear_id();
            if (i >= total) return;
            const int m = i / cols;
            const int c = (i % cols) * VEC;

            // Gate was precomputed by the map kernel (vectorized dot there);
            // this kernel is a pure weighted sum + gated shared add.

            ElementD* o = out + (int64_t)m * H + c;
            if (has_shared) {
#pragma unroll
              for (int j = 0; j < VEC; ++j) {
                float acc = 0.f;
                for (int kk = 0; kk < top_k; ++kk) {
                  const int p = perm_pos[m * top_k + kk];
                  acc += w_perm[p] * (float)d2[(int64_t)p * H + c + j];
                }
                o[j] = ElementD(
                    acc + gate[m] * (float)sh2[(int64_t)m * H + c + j]);
              }
            } else {
#pragma unroll
              for (int j = 0; j < VEC; ++j) {
                float acc = 0.f;
                for (int kk = 0; kk < top_k; ++kk) {
                  const int p = perm_pos[m * top_k + kk];
                  acc += w_perm[p] * (float)d2[(int64_t)p * H + c + j];
                }
                o[j] = ElementD(acc);
              }
            }
          });
    });
  }
};

// ---------------------------------------------------------------------------
// Host wrapper: the whole fused forward in one op.
// ---------------------------------------------------------------------------
inline int32_t* iptr(uint8_t* base, int64_t byte_off) {
  return reinterpret_cast<int32_t*>(base + byte_off);
}
inline int64_t align256(int64_t b) { return (b + 255) / 256 * 256; }

torch::Tensor moe_sp_fused_forward(
    torch::Tensor output,
    torch::Tensor x,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor w13,
    torch::Tensor w13_scale,
    torch::Tensor w2,
    torch::Tensor w2_scale,
    c10::optional<torch::Tensor> shared_w13,
    c10::optional<torch::Tensor> shared_w13_scale,
    c10::optional<torch::Tensor> shared_w2,
    c10::optional<torch::Tensor> shared_w2_scale,
    c10::optional<torch::Tensor> shared_gate,
    torch::Tensor ws,
    int64_t policy_id) {
  TORCH_CHECK(
      x.scalar_type() == at::kHalf && x.is_contiguous() && x.dim() == 2);
  TORCH_CHECK(topk_ids.dim() == 2 && topk_ids.is_contiguous());
  TORCH_CHECK(
      topk_weights.scalar_type() == at::kFloat && topk_weights.is_contiguous());
  TORCH_CHECK(output.scalar_type() == at::kHalf && output.is_contiguous());
  TORCH_CHECK(w13.is_contiguous() && w2.is_contiguous());
  TORCH_CHECK(w13_scale.scalar_type() == at::kFloat && w13_scale.dim() == 1);
  TORCH_CHECK(w2_scale.scalar_type() == at::kFloat && w2_scale.dim() == 1);

  const int num_tokens = x.size(0);
  const int H = x.size(1);
  const int top_k = topk_ids.size(1);
  const int R = num_tokens * top_k;
  const int E = w13.size(0);
  const int K1 = w13.size(1);
  const int N1 = w13.size(2);
  const int K2 = w2.size(1);
  const int N2 = w2.size(2);
  TORCH_CHECK(num_experts_static_ok(E), "E > 512 unsupported");
  const bool has_shared = shared_w13.has_value();
  TORCH_CHECK(
      !has_shared ||
          (shared_w13->is_contiguous() && shared_w2->is_contiguous() &&
           shared_w13_scale->scalar_type() == at::kFloat &&
           shared_w2_scale->scalar_type() == at::kFloat &&
           shared_gate->scalar_type() == at::kHalf),
      "shared expert weight/scale dtypes");

  const int umt = R < E ? R : E;
  const int M_s = num_tokens;

  // ---- ONE flat byte workspace (allocated once per batch size by the
  // Python side): [int32 map arrays][w_perm fp32][gate fp32][fp16 buffers]
  uint8_t* wsb = reinterpret_cast<uint8_t*>(ws.data_ptr());
  const int64_t o_int = 0;
  const int64_t o_wperm = align256(o_int + 4 * (3 * umt + 1 + R));
  const int64_t o_gate = align256(o_wperm + 4 * R);
  int64_t o_ap = align256(o_gate + 4 * M_s);
  int64_t o_d1 = o_ap + (int64_t)R * K1 * 2;
  int64_t o_act = o_d1 + (int64_t)(R + M_s) * N1 * 2;
  int64_t o_d2 = o_act + (int64_t)(R + M_s) * K2 * 2;
  int64_t o_sh2 = o_d2 + (int64_t)R * N2 * 2;
  int64_t htotal = o_sh2 + (int64_t)M_s * N2 * 2;
  TORCH_CHECK(ws.numel() >= htotal, "workspace too small");

  int32_t* mt_expert = iptr(wsb, o_int);
  int32_t* mt_row_base = iptr(wsb, o_int + 4 * umt);
  int32_t* mt_rows = iptr(wsb, o_int + 8 * umt);
  int32_t* total_mt = iptr(wsb, o_int + 12 * umt);
  int32_t* perm_pos = iptr(wsb, o_int + 4 * (3 * umt + 1));
  float* w_perm = reinterpret_cast<float*>(wsb + o_wperm);
  float* gate_out = reinterpret_cast<float*>(wsb + o_gate);
  ElementA* a_perm = reinterpret_cast<ElementA*>(wsb + o_ap);
  ElementA* d1 = reinterpret_cast<ElementA*>(wsb + o_d1);
  ElementA* act = reinterpret_cast<ElementA*>(wsb + o_act);
  ElementA* d2 = reinterpret_cast<ElementA*>(wsb + o_d2);
  ElementA* sh2 = reinterpret_cast<ElementA*>(wsb + o_sh2);

  auto& q = c10::xpu::getCurrentXPUStream(x.device().index()).queue();
  const int tile_m = (policy_id == 1) ? 8 : 16;

  // 1: map + gate (single work-group), 2: gather
  if (topk_ids.scalar_type() == at::kLong) {
    launch_map_build<int64_t>(
        q,
        topk_ids.data_ptr<int64_t>(),
        topk_weights.data_ptr<float>(),
        num_tokens,
        top_k,
        E,
        tile_m,
        umt,
        mt_expert,
        mt_row_base,
        mt_rows,
        total_mt,
        perm_pos,
        w_perm);
  } else {
    TORCH_CHECK(topk_ids.scalar_type() == at::kInt);
    launch_map_build<int32_t>(
        q,
        topk_ids.data_ptr<int32_t>(),
        topk_weights.data_ptr<float>(),
        num_tokens,
        top_k,
        E,
        tile_m,
        umt,
        mt_expert,
        mt_row_base,
        mt_rows,
        total_mt,
        perm_pos,
        w_perm);
  }
  SpGatherLauncher::launch(
      q, (ElementA*)x.data_ptr(), perm_pos, R, K1, top_k, a_perm);

  // 3: gemm1 (routed tiles + shared panel in one grid)
  if (policy_id == 1) {
    launch_sp_gemm_merged<MoE::w8a16_policy_m_8, cutlass::float_e4m3_t>(
        q,
        a_perm,
        w13.data_ptr(),
        w13_scale.data_ptr<float>(),
        d1,
        mt_expert,
        mt_row_base,
        mt_rows,
        total_mt,
        umt,
        (ElementA*)x.data_ptr(),
        shared_w13.value_or(w13).data_ptr(),
        shared_w13_scale.value_or(w13_scale).data_ptr<float>(),
        d1 + (int64_t)R * N1,
        M_s,
        N1,
        K1,
        has_shared);
  } else {
    launch_sp_gemm_merged<MoE::w8a16_policy_m_16, cutlass::float_e4m3_t>(
        q,
        a_perm,
        w13.data_ptr(),
        w13_scale.data_ptr<float>(),
        d1,
        mt_expert,
        mt_row_base,
        mt_rows,
        total_mt,
        umt,
        (ElementA*)x.data_ptr(),
        shared_w13.value_or(w13).data_ptr(),
        shared_w13_scale.value_or(w13_scale).data_ptr<float>(),
        d1 + (int64_t)R * N1,
        M_s,
        N1,
        K1,
        has_shared);
  }

  // 4: combined activation (routed rows + shared rows)
  SpSiluMulLauncher::launch(
      q,
      d1,
      act,
      R + (has_shared ? M_s : 0),
      K2,
      (ElementA*)x.data_ptr(),
      has_shared ? (ElementA*)shared_gate->data_ptr() : (ElementA*)x.data_ptr(),
      gate_out,
      num_tokens,
      H,
      has_shared);

  // 5: gemm2 (routed tiles + shared panel in one grid)
  if (policy_id == 1) {
    launch_sp_gemm_merged<MoE::w8a16_policy_m_8, cutlass::float_e4m3_t>(
        q,
        act,
        w2.data_ptr(),
        w2_scale.data_ptr<float>(),
        d2,
        mt_expert,
        mt_row_base,
        mt_rows,
        total_mt,
        umt,
        act + (int64_t)R * K2,
        shared_w2.value_or(w2).data_ptr(),
        shared_w2_scale.value_or(w2_scale).data_ptr<float>(),
        sh2,
        M_s,
        N2,
        K2,
        has_shared);
  } else {
    launch_sp_gemm_merged<MoE::w8a16_policy_m_16, cutlass::float_e4m3_t>(
        q,
        act,
        w2.data_ptr(),
        w2_scale.data_ptr<float>(),
        d2,
        mt_expert,
        mt_row_base,
        mt_rows,
        total_mt,
        umt,
        act + (int64_t)R * K2,
        shared_w2.value_or(w2).data_ptr(),
        shared_w2_scale.value_or(w2_scale).data_ptr<float>(),
        sh2,
        M_s,
        N2,
        K2,
        has_shared);
  }

  // 6: finalize (pure weighted sum + gated shared add) into output
  SpFinalizeLauncher::launch(
      q,
      gate_out,
      perm_pos,
      w_perm,
      d2,
      sh2,
      (ElementD*)output.data_ptr(),
      num_tokens,
      H,
      top_k,
      has_shared);
  return output;
}

}  // namespace moe_sp

TORCH_LIBRARY(moe_qwen35_sp, m) {
  m.def(
      "fused_forward(Tensor output, Tensor x, Tensor topk_ids, Tensor "
      "topk_weights, Tensor w13, Tensor w13_scale, Tensor w2, Tensor "
      "w2_scale, Tensor? shared_w13, Tensor? shared_w13_scale, Tensor? "
      "shared_w2, Tensor? shared_w2_scale, Tensor? shared_gate, Tensor "
      "ws, int policy_id=0) -> Tensor");
}

TORCH_LIBRARY_IMPL(moe_qwen35_sp, XPU, m) {
  m.impl("fused_forward", &moe_sp::moe_sp_fused_forward);
}
