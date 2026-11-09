/***************************************************************************************************
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Fused routed + shared-expert MoE forward for small-M decode, INT4-AWQ
 * recipe: FP16/BF16 activations, int8-packed 4-bit per-expert weights
 * with per-group scales (same dtype as activations), SiLU, one sigmoid-gated
 * shared expert. Per-group scales apply in-register in the shared
 * xe_gemm_4bits mainloop.
 *
 * One op submits six kernels with no per-call allocation:
 *   1. map     : topk_ids -> compact per-expert tile map, route->row
 *                permutation and permuted routing weights (one work-group).
 *   2. gather  : x -> expert-major A_perm.
 *   3. gemm1   : routed tiles (direct-indexed tile map) + shared-expert panel
 *                in ONE grid; mainloop is the sycl-tla MoE::xe_gemm_4bits
 *                INT4 grouped GEMM.
 *   4. act     : SiLU-and-mul over routed + shared rows, plus the shared
 *                expert sigmoid gate dot per token, in one launch.
 *   5. gemm2   : routed tiles + shared panel in one grid.
 *   6. finalize: out = sum_k w_k * routed_k + sigmoid(gate) * shared.
 *
 * Output is the per-rank partial sum (TP all-reduce is left to the caller).
 *
 * Padding expert IDs (-1) contribute zero to the routed experts; valid IDs
 * must be distinct within each token's top-k routes.
 *
 * Shared experts are optional: pass None for all five shared tensors to
 * run routed-only (the caller then runs the dense shared MLP separately).
 *
 * H is a positive multiple of 256 and the packed I/2 a multiple of 4, so
 * compact routed row bases remain 64-byte aligned for block-2D copies.
 * Weight layouts (all contiguous, [E, N, K] storage order with K packed,
 * same convention as the grouped int4 path, with 64-byte-aligned bases):
 *   w13        [E, 2I, H/2] int8-packed 4-bit, w13_scale [E, 2I, H/G]
 *   w2         [E, H, I/2]  int8-packed 4-bit, w2_scale  [E, H, I/G]
 *   shared_w13 [2I, H/2]   int8-packed, shared_w13_scale [1, 2I, H/G]
 *   shared_w2  [H, I/2]    int8-packed, shared_w2_scale  [1, H, I/G]
 *   scales have the activation dtype; G = group size (32/64/128/256)
 *   shared_gate[H] (or [1, H]) matches the input dtype
 * ws is a caller-owned uint8 workspace sized by the Python helper
 * (moe_shared_fused_int4_workspace_bytes); it must be 64-byte aligned and
 * disjoint from all other arguments. The op checks these requirements.
 * tile_m selects the GEMM M-tile (kMoeSharedFusedInt4TileM8 or
 * kMoeSharedFusedInt4TileM16); the interface resolves 0 (auto) before calling.
 **************************************************************************************************/
#include <torch/all.h>
#include <ATen/MemoryOverlap.h>
#include <c10/xpu/XPUStream.h>
#include <cstdint>

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"

#include "moe_shared_fused_decode_int4_xe2.h"

#include "gemm_xe2_policy.hpp"
#include "grouped_gemm_xe2.hpp"  // MoE::make_moe_tensor + MoE::xe_gemm_4bits

namespace moe_shared_fused_int4 {
using namespace cute;

namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

constexpr int MAX_E = 512;  // routing supports up to 512 experts
constexpr int MAX_R =
    512;  // upper bound on routed rows handled by the map kernel
constexpr int GEMM_WG = 64;  // size(mma) for both m_8 and m_16 policies
constexpr int MAP_WG = 64;

constexpr int64_t MAX_BLOCK_2D_BYTES = (1 << 24) - 1;

// ---------------------------------------------------------------------------
// Kernel name tags (unique types per instantiation)
// ---------------------------------------------------------------------------
template <typename IDT>
class SpMapName;
template <typename ElementA>
class SpGatherName;
template <typename ElementA>
class SpSiluMulName;
template <typename ElementA>
class SpFinalizeName;
template <class Policy, typename ElementA>
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
          for (int r = lane; r < R; r += MAP_WG) {
            // Check the original width before narrowing int64 IDs.
            const IDT id = ids[r];
            ids_slm[r] = (id >= 0 && id < num_experts) ? (int32_t)id : -1;
            // vLLM marks padded tokens with expert ID -1.
            perm_pos[r] = -1;
          }
          sycl::group_barrier(item.get_group());

          if (R <= MAP_WG && num_tokens <= tile_m) {
            // Decode fast path: every expert receives at most num_tokens <=
            // tile_m rows, i.e. at most one tile, so the map follows from
            // ranks among the R routes (no per-expert scan / serial prefix):
            //   perm_pos[r] = #routes with a smaller id (+ earlier same id)
            //   tile(e)     = #distinct ids smaller than e
            int32_t* first = rows;  // first[r]: route r is its id's first
            if (lane < R) {
              const int e = ids_slm[lane];
              int f = (e >= 0 && e < num_experts);
              for (int s = 0; s < lane; ++s)
                f &= ids_slm[s] != e;
              first[lane] = f;
            }
            sycl::group_barrier(item.get_group());
            if (lane < R && ids_slm[lane] >= 0 && ids_slm[lane] < num_experts) {
              const int e = ids_slm[lane];
              int less = 0, before = 0, cnt = 0, tile = 0;
              for (int s = 0; s < R; ++s) {
                const int es = ids_slm[s];
                less += es >= 0 && es < num_experts && es < e;
                cnt += es == e;
                before += (es == e) & (s < lane);
                tile += (es < e) & first[s];
              }
              const int p = less + before;
              perm_pos[lane] = p;
              w_perm[p] = weights[lane];
              if (first[lane]) {
                mt_expert[tile] = e;
                mt_row_base[tile] = less;
                mt_rows[tile] = cnt;
              }
            }
            if (lane == 0) {
              int n = 0;
              for (int s = 0; s < R; ++s)
                n += first[s];
              total_mt[0] = n;
            }
            return;
          }

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
            if (e < 0 || e >= num_experts) continue;
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
        });
  });
}

// ---------------------------------------------------------------------------
// Kernel 2: gather. x[token(r)] -> A_perm[perm_pos[r]], chunked over K.
// ---------------------------------------------------------------------------
template <typename ElementA>
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
      cgh.parallel_for<SpGatherName<ElementA>>(
          sycl::nd_range<2>(
              sycl::range<2>(R, ceil_div(K / 4, 64) * 64),
              sycl::range<2>(1, 64)),
          [=](sycl::nd_item<2> item) {
            const int r = item.get_global_id(0);
            const int c = item.get_global_id(1) * 4;
            if (c >= K) return;
            const int dst = perm_pos[r];
            if (dst < 0) return;
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
// Kernels 3 and 5: merged grouped-GEMM tile launcher (gemm1, gemm2).
// Reuses the `MoE::xe_gemm_4bits` INT4 mainloop; replaces the
// persistent+atomic scheduler with direct tile-map indexing. Routed tiles
// (direct-indexed map) + shared panel (one dense panel; A is already
// contiguous, expert id 0, grid = exact tile count) run in ONE grid,
// ONE submit per gemm stage. The two segments share the same N/K/policy
// but differ in A/B/D/scale pointers.
// ---------------------------------------------------------------------------
template <class Policy, typename ElementA>
void launch_sp_gemm_merged_int4(
    sycl::queue& q,
    const void* A_routed,
    const void* B_routed,
    const ElementA* scales_routed,
    void* D_routed,
    const int32_t* mt_expert,
    const int32_t* mt_row_base,
    const int32_t* mt_rows,
    const int32_t* total_mt,
    int umt,
    const void* A_shared,
    const void* B_shared,
    const ElementA* scales_shared,
    void* D_shared,
    int gemm_m_shared,
    int N,
    int K,
    bool has_shared,
    int64_t group_size,
    const uint8_t* zps_routed = nullptr,
    const uint8_t* zps_shared = nullptr) {
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
    cgh.parallel_for<SpGemmName<Policy, ElementA>>(
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
          const ElementA* sc_base = is_routed ? scales_routed : scales_shared;
          const int64_t ngroups = K / group_size;
          void* D = is_routed ? D_routed : D_shared;
          int e = 0, row0 = 0, rows;
          if (is_routed) {
            e = mt_expert[m_tile];
            row0 = mt_row_base[m_tile];
            rows = mt_rows[m_tile];
          } else {
            // Shared panel: one dense GEMM split into M-tiles of wg_tile_m.
            row0 = m_tile * wg_tile_m;
            const int rem = gemm_m_shared - row0;
            rows = rem < wg_tile_m ? rem : wg_tile_m;
          }

          auto A_t = MoE::make_moe_tensor<ElementA, 'R'>(
              (ElementA*)A + (int64_t)row0 * K, rows, K);
          auto B_t = MoE::make_moe_tensor<int4_t, 'R'>(
              reinterpret_cast<int4_t*>((uint8_t*)B + (int64_t)e * N * K / 2),
              N, K);
          auto D_t = MoE::make_moe_tensor<ElementA, 'R'>(
              (ElementA*)D + (int64_t)row0 * N, rows, N);
          auto tile_coord = make_coord(0, n_tile, _, 0);

          const ElementA* sc_e = sc_base + (int64_t)e * N * ngroups;
          // Asymmetric zero-points ride alongside the scales ([E, N, K/G]
          // raw uint8, same indexing). A null segment pointer selects the
          // symmetric instantiation, which is today's code unchanged.
          const uint8_t* zp_base = is_routed ? zps_routed : zps_shared;
          const uint8_t* zp_e =
              (zp_base == nullptr)
                  ? nullptr
                  : zp_base + (int64_t)(is_routed ? e : 0) * N * ngroups;
          if (group_size == 32) {
            if (zp_e != nullptr) {
              MoE::xe_gemm_4bits<void, void, void, MoE::B_DTYPE::INT4, 32,
                                 true>(
                  A_t, B_t, sc_e, (const ElementA*)nullptr, D_t, tile_coord,
                  mma, zp_e);
            } else {
              MoE::xe_gemm_4bits<void, void, void, MoE::B_DTYPE::INT4, 32,
                                 false>(
                  A_t, B_t, sc_e, (const ElementA*)nullptr, D_t, tile_coord,
                  mma);
            }
          } else if (group_size == 64) {
            if (zp_e != nullptr) {
              MoE::xe_gemm_4bits<void, void, void, MoE::B_DTYPE::INT4, 64,
                                 true>(
                  A_t, B_t, sc_e, (const ElementA*)nullptr, D_t, tile_coord,
                  mma, zp_e);
            } else {
              MoE::xe_gemm_4bits<void, void, void, MoE::B_DTYPE::INT4, 64,
                                 false>(
                  A_t, B_t, sc_e, (const ElementA*)nullptr, D_t, tile_coord,
                  mma);
            }
          } else if (group_size == 128) {
            if (zp_e != nullptr) {
              MoE::xe_gemm_4bits<void, void, void, MoE::B_DTYPE::INT4, 128,
                                 true>(
                  A_t, B_t, sc_e, (const ElementA*)nullptr, D_t, tile_coord,
                  mma, zp_e);
            } else {
              MoE::xe_gemm_4bits<void, void, void, MoE::B_DTYPE::INT4, 128,
                                 false>(
                  A_t, B_t, sc_e, (const ElementA*)nullptr, D_t, tile_coord,
                  mma);
            }
          } else {
            if (zp_e != nullptr) {
              MoE::xe_gemm_4bits<void, void, void, MoE::B_DTYPE::INT4, 256,
                                 true>(
                  A_t, B_t, sc_e, (const ElementA*)nullptr, D_t, tile_coord,
                  mma, zp_e);
            } else {
              MoE::xe_gemm_4bits<void, void, void, MoE::B_DTYPE::INT4, 256,
                                 false>(
                  A_t, B_t, sc_e, (const ElementA*)nullptr, D_t, tile_coord,
                  mma);
            }
          }
        });
  });
}

// ---------------------------------------------------------------------------
// Kernel 4: SiLU-and-mul over a COMBINED buffer of [R + M] rows (routed act
// rows first, shared-expert rows after) -- one launch for both activations.
//   in  [rows, 2*ic] -> out [rows, ic] : silu(in[:, :ic]) * in[:, ic:]
// ---------------------------------------------------------------------------
template <typename ElementA>
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
      cgh.parallel_for<SpSiluMulName<ElementA>>(
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
// Kernel 6: finalize. out[m, c:c+8] = sum_k w_perm[perm_pos[m,k]] *
// D2[perm_pos[m,k], c:c+8] + sigmoid(dot(x[m], gate_vec)) * sh2[m, c:c+8]
// ---------------------------------------------------------------------------
template <typename ElementA>
class SpFinalizeLauncher {
 public:
  static void launch(
      sycl::queue& q,
      const float* gate,
      const int32_t* perm_pos,
      const float* w_perm,
      const ElementA* d2,
      const ElementA* sh2,
      ElementA* out,
      int num_tokens,
      int H,
      int top_k,
      bool has_shared) {
    constexpr int VEC = 8;
    const int cols = H / VEC;
    const int total = num_tokens * cols;
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<SpFinalizeName<ElementA>>(
          sycl::nd_range<1>(
              sycl::range<1>(((total + 63) / 64) * 64), sycl::range<1>(64)),
          [=](sycl::nd_item<1> item) {
            const int i = item.get_global_linear_id();
            if (i >= total) return;
            const int m = i / cols;
            const int c = (i % cols) * VEC;

            // Gate dots were computed by the SiLU kernel (cooperative dot
            // there); this kernel is a pure weighted sum + gated shared add.

            ElementA* o = out + (int64_t)m * H + c;
#pragma unroll
            for (int j = 0; j < VEC; ++j) {
              float acc = 0.f;
              for (int kk = 0; kk < top_k; ++kk) {
                const int p = perm_pos[m * top_k + kk];
                if (p >= 0)
                  acc += w_perm[p] * (float)d2[(int64_t)p * H + c + j];
              }
              o[j] = ElementA(
                  acc +
                  (has_shared ? gate[m] * (float)sh2[(int64_t)m * H + c + j]
                              : 0.f));
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

}  // namespace moe_shared_fused_int4

template <typename ElementA>
void moe_shared_fused_decode_int4_xe2_impl(
    torch::Tensor& output,
    const torch::Tensor& x,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    const torch::Tensor& w13,
    const torch::Tensor& w13_scale,
    const torch::Tensor& w2,
    const torch::Tensor& w2_scale,
    const torch::Tensor& shared_w13_t,
    const torch::Tensor& shared_w13_scale_t,
    const torch::Tensor& shared_w2_t,
    const torch::Tensor& shared_w2_scale_t,
    const torch::Tensor& shared_gate_t,
    const torch::Tensor& w13_zps,
    const torch::Tensor& w2_zps,
    const torch::Tensor& shared_w13_zps,
    const torch::Tensor& shared_w2_zps,
    torch::Tensor& ws,
    int64_t tile_m) {
  using namespace moe_shared_fused_int4;
  constexpr const char* kRecipe =
      "moe_shared_fused_decode_int4 only supports fp16/bf16 activations + "
      "int8-packed 4-bit weights with per-group scales: ";
  TORCH_CHECK(x.is_xpu(), "x must be an XPU tensor");
  const auto check_device = [&x](
                                const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(
        tensor.is_xpu() && tensor.device() == x.device(),
        name,
        " must be on the same XPU device as x");
  };
  // Shared experts are optional (Tensor? in the schema): pass None for
  // all five shared tensors to run routed-only, and the caller runs the
  // dense shared MLP separately.
  const bool has_shared = shared_w13_t.defined();
  TORCH_CHECK(
      has_shared == shared_w13_scale_t.defined() &&
          has_shared == shared_w2_t.defined() &&
          has_shared == shared_w2_scale_t.defined() &&
          has_shared == shared_gate_t.defined(),
      "shared_w13, shared_w13_scale, shared_w2, shared_w2_scale and "
      "shared_gate must be either all defined or all undefined");
  check_device(output, "output");
  check_device(topk_ids, "topk_ids");
  check_device(topk_weights, "topk_weights");
  check_device(w13, "w13");
  check_device(w13_scale, "w13_scale");
  check_device(w2, "w2");
  check_device(w2_scale, "w2_scale");
  if (has_shared) {
    check_device(shared_w13_t, "shared_w13");
    check_device(shared_w13_scale_t, "shared_w13_scale");
    check_device(shared_w2_t, "shared_w2");
    check_device(shared_w2_scale_t, "shared_w2_scale");
    check_device(shared_gate_t, "shared_gate");
  }
  check_device(ws, "ws");
  TORCH_CHECK(
      w13.dim() == 3 && w2.dim() == 3, "routed weights must be rank three");
  TORCH_CHECK(
      w13_scale.is_contiguous() && w2_scale.is_contiguous(),
      "routed scales must be contiguous");
  TORCH_CHECK(ws.is_contiguous(), "workspace must be contiguous");
  TORCH_CHECK(
      (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) &&
          output.scalar_type() == x.scalar_type(),
      kRecipe,
      "x and output must have the same float16 or bfloat16 dtype");
  TORCH_CHECK(
      w13.scalar_type() == at::kChar && w2.scalar_type() == at::kChar,
      kRecipe,
      "routed weights must be int8-packed 4-bit");
  TORCH_CHECK(
      !has_shared ||
          (shared_w13_t.scalar_type() == at::kChar &&
           shared_w2_t.scalar_type() == at::kChar),
      kRecipe,
      "shared weights must be int8-packed 4-bit");
  TORCH_CHECK(
      w13_scale.scalar_type() == x.scalar_type() && w13_scale.dim() == 3 &&
          w2_scale.scalar_type() == x.scalar_type() && w2_scale.dim() == 3 &&
          w13_scale.size(0) == w13.size(0) &&
          w2_scale.size(0) == w2.size(0) && w13_scale.size(1) == w13.size(1) &&
          w2_scale.size(1) == w2.size(1),
      kRecipe,
      "routed scales must be [E, N, K/G] in the activation dtype");
  TORCH_CHECK(
      !has_shared ||
          (shared_w13_scale_t.scalar_type() == x.scalar_type() &&
           shared_w13_scale_t.dim() == 3 &&
           shared_w13_scale_t.size(0) == 1 &&
           shared_w13_scale_t.size(1) == w13.size(1) &&
           shared_w13_scale_t.size(2) == w13_scale.size(2) &&
           shared_w2_scale_t.scalar_type() == x.scalar_type() &&
           shared_w2_scale_t.dim() == 3 &&
           shared_w2_scale_t.size(0) == 1 &&
           shared_w2_scale_t.size(1) == w2.size(1) &&
           shared_w2_scale_t.size(2) == w2_scale.size(2)),
      kRecipe,
      "shared scales must be [1, N, K/G] in the activation dtype");
  // Asymmetric zero-points ride alongside the scales (raw uint8, same
  // [E, N, K/G] geometry). Mixed setups are rejected: when any zp is
  // given, every supplied weight set needs its pair.
  const bool has_zps = w13_zps.defined();
  TORCH_CHECK(
      has_zps == w2_zps.defined(),
      kRecipe,
      "routed zero-points must be both defined or both undefined");
  TORCH_CHECK(
      has_shared ? (has_zps == shared_w13_zps.defined() &&
                    has_zps == shared_w2_zps.defined())
                 : (!shared_w13_zps.defined() &&
                    !shared_w2_zps.defined()),
      kRecipe,
      "shared zero-points must be defined exactly when routed ones are "
      "(routed-only runs take no shared zero-points)");
  if (has_zps) {
    check_device(w13_zps, "w13_zps");
    check_device(w2_zps, "w2_zps");
    TORCH_CHECK(
        w13_zps.scalar_type() == at::kByte && w13_zps.dim() == 3 &&
            w13_zps.sizes() == w13_scale.sizes() &&
            w2_zps.scalar_type() == at::kByte && w2_zps.dim() == 3 &&
            w2_zps.sizes() == w2_scale.sizes(),
        kRecipe,
        "routed zero-points must be uint8 [E, N, K/G] matching the scales");
    TORCH_CHECK(
        w13_zps.is_contiguous() && w2_zps.is_contiguous(),
        "routed zero-points must be contiguous");
    if (has_shared) {
      check_device(shared_w13_zps, "shared_w13_zps");
      check_device(shared_w2_zps, "shared_w2_zps");
      TORCH_CHECK(
          shared_w13_zps.scalar_type() == at::kByte &&
              shared_w13_zps.dim() == 3 &&
              shared_w13_zps.sizes() == shared_w13_scale_t.sizes() &&
              shared_w2_zps.scalar_type() == at::kByte &&
              shared_w2_zps.dim() == 3 &&
              shared_w2_zps.sizes() == shared_w2_scale_t.sizes(),
          kRecipe,
          "shared zero-points must be uint8 [1, N, K/G] matching the "
          "scales");
      TORCH_CHECK(
          shared_w13_zps.is_contiguous() && shared_w2_zps.is_contiguous(),
          "shared zero-points must be contiguous");
    }
  }
  TORCH_CHECK(
      !has_shared || shared_gate_t.scalar_type() == x.scalar_type(),
      kRecipe,
      "shared_gate must have the input dtype");
  TORCH_CHECK(
      topk_weights.scalar_type() == at::kFloat, "topk_weights must be float32");
  TORCH_CHECK(
      topk_ids.scalar_type() == at::kInt || topk_ids.scalar_type() == at::kLong,
      "topk_ids must be int32 or int64");
  TORCH_CHECK(ws.scalar_type() == at::kByte, "ws must be uint8");
  TORCH_CHECK(
      tile_m == kMoeSharedFusedInt4TileM8 || tile_m == kMoeSharedFusedInt4TileM16,
      "tile_m must be 8 or 16");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous() && output.is_contiguous());
  TORCH_CHECK(output.sizes() == x.sizes(), "output must match x");
  TORCH_CHECK(topk_ids.dim() == 2 && topk_ids.is_contiguous());
  TORCH_CHECK(topk_ids.size(0) == x.size(0), "routing rows must match x");
  TORCH_CHECK(
      topk_weights.sizes() == topk_ids.sizes() && topk_weights.is_contiguous());
  TORCH_CHECK(w13.is_contiguous() && w2.is_contiguous());
  TORCH_CHECK(
      !has_shared ||
          (shared_w13_t.is_contiguous() && shared_w2_t.is_contiguous() &&
           shared_gate_t.is_contiguous()),
      "shared weights must be contiguous");
  TORCH_CHECK(
      w13.size(2) * 2 == x.size(1) && w2.size(1) == x.size(1) &&
          w13.size(1) == 4 * w2.size(2),
      "routed weight shapes must be w13 [E, 2I, H/2], w2 [E, H, I/2] packed");
  TORCH_CHECK(
      !has_shared ||
          (shared_w13_t.dim() == 2 && shared_w13_t.size(0) == w13.size(1) &&
           shared_w13_t.size(1) == w13.size(2) && shared_w2_t.dim() == 2 &&
           shared_w2_t.size(0) == w2.size(1) &&
           shared_w2_t.size(1) == w2.size(2) &&
           shared_gate_t.numel() == x.size(1)),
      "shared weight shapes must be [N1, K1/2], [N2, K2/2] packed and gate [H]");
  TORCH_CHECK(
      x.size(1) > 0 && x.size(1) % 256 == 0 &&
          x.size(1) * 2 <= MAX_BLOCK_2D_BYTES && w2.size(2) * 2 > 0 &&
          w2.size(2) % 4 == 0 && w2.size(2) * 8 <= MAX_BLOCK_2D_BYTES,
      "H must be a positive multiple of 256 and packed I/2 a multiple of 4, "
      "within block-2D limits");

  const auto check_alignment =
      [](const torch::Tensor& tensor, const char* name, uintptr_t bytes) {
        TORCH_CHECK(
            reinterpret_cast<uintptr_t>(tensor.data_ptr()) % bytes == 0,
            name,
            " must have a ",
            bytes,
            "-byte-aligned base address");
      };
  check_alignment(x, "x", 64);
  check_alignment(w13, "w13", 64);
  check_alignment(w2, "w2", 64);
  if (has_shared) {
    check_alignment(shared_w13_t, "shared_w13", 64);
    check_alignment(shared_w2_t, "shared_w2", 64);
    check_alignment(shared_gate_t, "shared_gate", 32);
  }
  check_alignment(ws, "workspace", 64);

  // Scratch is mutable and must not overlap any user tensor.
  const torch::Tensor* inputs[] = {
      &output, &x, &topk_ids, &topk_weights, &w13, &w13_scale, &w2, &w2_scale};
  for (const auto* tensor : inputs) {
    at::assert_no_overlap(ws, *tensor);
  }
  if (has_shared) {
    const torch::Tensor* shared_inputs[] = {
        &shared_w13_t,
        &shared_w13_scale_t,
        &shared_w2_t,
        &shared_w2_scale_t,
        &shared_gate_t};
    for (const auto* tensor : shared_inputs) {
      at::assert_no_overlap(ws, *tensor);
    }
  }

  // Check 64-bit metadata before converting dimensions or multiplying.
  TORCH_CHECK(
      x.size(0) > 0 && x.size(0) <= MAX_R && w13.size(0) > 0 &&
          w13.size(0) <= MAX_E && topk_ids.size(1) > 0 &&
          topk_ids.size(1) <= w13.size(0),
      "require positive token count <= 512 and 1 <= top_k <= E <= 512");
  TORCH_CHECK(
      x.size(0) * topk_ids.size(1) <= MAX_R,
      "num_tokens * top_k exceeds ",
      MAX_R);
  const int num_tokens = x.size(0);
  const int H = x.size(1);
  const int top_k = topk_ids.size(1);
  const int R = num_tokens * top_k;
  const int E = w13.size(0);
  // [E, N, K] storage order, K packed: size(1) is logical N,
  // size(2) is K/2 packed bytes.
  const int K1 = w13.size(2) * 2;
  const int N1 = w13.size(1);
  const int K2 = w2.size(2) * 2;
  const int N2 = w2.size(1);
  const int64_t ngroups1 = w13_scale.size(2);
  const int64_t ngroups2 = w2_scale.size(2);
  const int64_t group_size = K1 / ngroups1;
  TORCH_CHECK(
      K1 % ngroups1 == 0 && K2 % ngroups2 == 0 && K1 / ngroups1 == K2 / ngroups2,
      kRecipe, "scales K/G must divide K with one group size for w13 and w2");
  TORCH_CHECK(
      group_size == 32 || group_size == 64 || group_size == 128 ||
          group_size == 256,
      kRecipe, "group size must be 32, 64, 128 or 256");

  // Routed-only<<< (all shared tensors undefined): data_ptr() on an
  // undefined tensor throws, so resolve the shared pointers once here
  // under has_shared. The merged grid launches zero shared tiles when
  // has_shared is false, so these fallbacks are never read.
  const void* shared_w13_ptr =
      has_shared ? shared_w13_t.data_ptr() : w13.data_ptr();
  const ElementA* shared_w13_scale_ptr =
      has_shared
          ? reinterpret_cast<const ElementA*>(shared_w13_scale_t.data_ptr())
          : reinterpret_cast<const ElementA*>(w13_scale.data_ptr());
  const void* shared_w2_ptr =
      has_shared ? shared_w2_t.data_ptr() : w2.data_ptr();
  const ElementA* shared_w2_scale_ptr =
      has_shared
          ? reinterpret_cast<const ElementA*>(shared_w2_scale_t.data_ptr())
          : reinterpret_cast<const ElementA*>(w2_scale.data_ptr());
  // Null selects the symmetric instantiation in the merged launcher.
  const uint8_t* w13_zps_ptr =
      has_zps ? w13_zps.data_ptr<uint8_t>() : nullptr;
  const uint8_t* w2_zps_ptr = has_zps ? w2_zps.data_ptr<uint8_t>() : nullptr;
  const uint8_t* shared_w13_zps_ptr =
      (has_shared && has_zps) ? shared_w13_zps.data_ptr<uint8_t>() : nullptr;
  const uint8_t* shared_w2_zps_ptr =
      (has_shared && has_zps) ? shared_w2_zps.data_ptr<uint8_t>() : nullptr;

  // Each routed row can require its own tile; skewed routes may exceed E.
  const int umt = R;
  const int M_s = num_tokens;

  // ---- ONE flat byte workspace (allocated once per batch size by the
  // Python side): [int32 map arrays][w_perm fp32][gate fp32][16-bit buffers]
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

  // 1: tile map (single work-group), 2: gather
  if (topk_ids.scalar_type() == at::kLong) {
    launch_map_build<int64_t>(
        q,
        topk_ids.data_ptr<int64_t>(),
        topk_weights.data_ptr<float>(),
        num_tokens,
        top_k,
        E,
        tile_m,
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
        mt_expert,
        mt_row_base,
        mt_rows,
        total_mt,
        perm_pos,
        w_perm);
  }
  SpGatherLauncher<ElementA>::launch(
      q, (ElementA*)x.data_ptr(), perm_pos, R, K1, top_k, a_perm);

  // 3: gemm1 (routed tiles + shared panel in one grid)
  if (tile_m == kMoeSharedFusedInt4TileM8) {
    launch_sp_gemm_merged_int4<
        MoE::w4a16_policy_m_8,
        ElementA>(
        q,
        a_perm,
        w13.data_ptr(),
        reinterpret_cast<const ElementA*>(w13_scale.data_ptr()),
        d1,
        mt_expert,
        mt_row_base,
        mt_rows,
        total_mt,
        umt,
        (ElementA*)x.data_ptr(),
        shared_w13_ptr,
        shared_w13_scale_ptr,
        d1 + (int64_t)R * N1,
        M_s,
        N1,
        K1,
        has_shared,
        group_size,
        w13_zps_ptr,
        shared_w13_zps_ptr);
  } else {
    launch_sp_gemm_merged_int4<
        MoE::w4a16_policy_m_16,
        ElementA>(
        q,
        a_perm,
        w13.data_ptr(),
        reinterpret_cast<const ElementA*>(w13_scale.data_ptr()),
        d1,
        mt_expert,
        mt_row_base,
        mt_rows,
        total_mt,
        umt,
        (ElementA*)x.data_ptr(),
        shared_w13_ptr,
        shared_w13_scale_ptr,
        d1 + (int64_t)R * N1,
        M_s,
        N1,
        K1,
        has_shared,
        group_size,
        w13_zps_ptr,
        shared_w13_zps_ptr);
  }

  // 4: combined activation (routed rows + shared rows)
  SpSiluMulLauncher<ElementA>::launch(
      q,
      d1,
      act,
      R + (has_shared ? M_s : 0),
      K2,
      (ElementA*)x.data_ptr(),
      has_shared ? (ElementA*)shared_gate_t.data_ptr() : (ElementA*)x.data_ptr(),
      gate_out,
      num_tokens,
      H,
      has_shared);

  // 5: gemm2 (routed tiles + shared panel in one grid)
  if (tile_m == kMoeSharedFusedInt4TileM8) {
    launch_sp_gemm_merged_int4<
        MoE::w4a16_policy_m_8,
        ElementA>(
        q,
        act,
        w2.data_ptr(),
        reinterpret_cast<const ElementA*>(w2_scale.data_ptr()),
        d2,
        mt_expert,
        mt_row_base,
        mt_rows,
        total_mt,
        umt,
        act + (int64_t)R * K2,
        shared_w2_ptr,
        shared_w2_scale_ptr,
        sh2,
        M_s,
        N2,
        K2,
        has_shared,
        group_size,
        w2_zps_ptr,
        shared_w2_zps_ptr);
  } else {
    launch_sp_gemm_merged_int4<
        MoE::w4a16_policy_m_16,
        ElementA>(
        q,
        act,
        w2.data_ptr(),
        reinterpret_cast<const ElementA*>(w2_scale.data_ptr()),
        d2,
        mt_expert,
        mt_row_base,
        mt_rows,
        total_mt,
        umt,
        act + (int64_t)R * K2,
        shared_w2_ptr,
        shared_w2_scale_ptr,
        sh2,
        M_s,
        N2,
        K2,
        has_shared,
        group_size,
        w2_zps_ptr,
        shared_w2_zps_ptr);
  }

  // 6: finalize (pure weighted sum + gated shared add) into output
  SpFinalizeLauncher<ElementA>::launch(
      q,
      gate_out,
      perm_pos,
      w_perm,
      d2,
      sh2,
      (ElementA*)output.data_ptr(),
      num_tokens,
      H,
      top_k,
      has_shared);
}

void moe_shared_fused_decode_int4_xe2(
    torch::Tensor& output,
    const torch::Tensor& x,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    const torch::Tensor& w13,
    const torch::Tensor& w13_scale,
    const torch::Tensor& w2,
    const torch::Tensor& w2_scale,
    const torch::Tensor& shared_w13_t,
    const torch::Tensor& shared_w13_scale_t,
    const torch::Tensor& shared_w2_t,
    const torch::Tensor& shared_w2_scale_t,
    const torch::Tensor& shared_gate_t,
    const torch::Tensor& w13_zps,
    const torch::Tensor& w2_zps,
    const torch::Tensor& shared_w13_zps,
    const torch::Tensor& shared_w2_zps,
    torch::Tensor& ws,
    int64_t tile_m) {
  if (x.scalar_type() == at::kBFloat16)
    return moe_shared_fused_decode_int4_xe2_impl<cutlass::bfloat16_t>(
        output,
        x,
        topk_ids,
        topk_weights,
        w13,
        w13_scale,
        w2,
        w2_scale,
        shared_w13_t,
        shared_w13_scale_t,
        shared_w2_t,
        shared_w2_scale_t,
        shared_gate_t,
        w13_zps,
        w2_zps,
        shared_w13_zps,
        shared_w2_zps,
        ws,
        tile_m);
  return moe_shared_fused_decode_int4_xe2_impl<cutlass::half_t>(
      output,
      x,
      topk_ids,
      topk_weights,
      w13,
      w13_scale,
      w2,
      w2_scale,
      shared_w13_t,
      shared_w13_scale_t,
      shared_w2_t,
      shared_w2_scale_t,
      shared_gate_t,
      w13_zps,
      w2_zps,
      shared_w13_zps,
      shared_w2_zps,
      ws,
      tile_m);
}
