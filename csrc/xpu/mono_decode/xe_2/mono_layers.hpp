// SPDX-License-Identifier: Apache-2.0
//
// Decode-only "mono" (one resident kernel per decoder layer) for Qwen3.5 /
// Qwen3.6 MoE (35B-A3B shapes) at TP2 on two Intel Arc Pro B70 (Xe2, 32 Xe
// cores): M = 1 token, fp8 e4m3 per-tensor weights (vLLM online fp8), fp16
// activations / KV cache, fp16 conv state, fp32 recurrent state. The
// tensor-parallel all-reduces are done inside the kernel by pushing (value |
// call) words into the peer card's mailbox (PCIe P2P stores).
//
// Two layer kernels share one tail:
//   GDN  front (linear_attention): input norm, in_proj q|k|v|z + b|a, causal
//   conv1d, gated delta rule,
//        gated RMSNorm -> o (2048, SLM)
//   ATTN front (full_attention):   input norm, qkv (q|gate per head, k, v), q/k
//   norm + RoPE, paged attention
//        split along the context with DPAS, merge with the new token, sigmoid
//        gate, KV write -> o (2048)
//   tail: out_proj / o_proj (64 rows per WG) -> all-reduce #1 -> residual ->
//   router / shared-expert partials of
//        the WG's own 64 rows -> fixed-order reduction -> top-8 -> routed
//        gate|up (vLLM XPU layout [E][K][N]) + shared expert -> down of the
//        WG's 64 output rows -> all-reduce #2
//
// Grid: 32 WG x 512 work-items (32 SIMD16 sub-groups), 256 GRF -> one WG per Xe
// core, all resident. Hand-offs between stages are monotonic per-stage arrival
// counters (target = 32 * call). `call` is a device-side epoch, so launches can
// be recorded into a graph. Every launch on a scratch must be matched by one
// launch on the peer card.
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <sycl/ext/oneapi/experimental/prefetch.hpp>
#include <cstdint>
#include <cmath>
#include <type_traits>

namespace mono_decode {

using half = sycl::half;
using h2 = sycl::vec<half, 2>;
namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

constexpr int NWG = 32, WGS = 512, NSG = 32;
constexpr int HID = 2048, NE = 256, TOPK = 8,
              INTER = 256;  // per rank (TP2): expert / shared inter 512 / 2
constexpr float EPS = 1e-6f;
// GDN (per rank): 8 k heads, 16 v heads, 128 dims
constexpr int NKH = 8, NVH = 16, DK = 128, DV = 128;
constexpr int KD = NKH * DK, VD = NVH * DV, CONVD = 2 * KD + VD,
              QKVZ = CONVD + VD;
// full attention (per rank): 8 q heads, 1 kv head, head dim 256, partial RoPE
// 64 dims
constexpr int NQ = 8, HD = 256, QKV_ROWS = NQ * 2 * HD + 2 * HD;  // 4608
constexpr int GROWS = NQ * HD + 2 * HD;  // gate + k + v rows
constexpr int KROW = 2 * HD;  // halves per token row of the KV cache (K | V)
constexpr float SM_SCALE = 0.0625f * 1.4426950408889634f, THETA = 1.0e7f;
constexpr int GKV_MID = 2, GKV_ROWS = 2,
              GKV_CHUNKS = (GROWS - GKV_MID * NWG * NSG) / GKV_ROWS;  // 256
constexpr int SPW = INTER / NWG;  // shared inter rows per WG (8)
constexpr int SHR =
    2 * INTER / NSG;  // shared gate|up rows per SG in the partial (16)

enum {
  C_A = 0,
  C_B,
  C_C,
  C_O,
  C_GU,
  C_TKT,
  NCNT
};  // GDN: A = qkvz, B = core ; ATTN: A = q, B = att, C = merge
constexpr int EPOCH = NCNT * 16;
// mailbox tag counters, one per kernel configuration (rows 1, 2, 4, 8 at MBT +
// 0 .. 3; rows 3, 6 at MBT2 + 0, 1), each with its own mailbox region:
// consecutive uses of a region carry consecutive tags, so a word left from an
// older use never matches the 16-bit tag
constexpr int MBT = EPOCH + 4, MBT2 = EPOCH + 16;

// ----------------------------------------------------------------------------
// arguments
struct Ptrs {
  // common
  const half *hs_in,
      *res_in;  // layer input hidden / residual (zeros for layer 0)
  half *hid_out,
      *res_out;  // MoE output (after all-reduce #2), residual after the mixer
  const half *in_norm_w, *post_norm_w, *router_w, *sgate_w;
  const uint8_t *out_w, *w13, *w2, *sw13,
      *sw2;  // out_proj / o_proj [2048][2048], w13 [E][2048][512],
             // w2 [E][256][2048], shared w13 [512][2048], shared w2 [2048][256]
  const float *out_s, *w13_s, *w2_s, *sw13_s, *sw2_s;
  // GDN
  const uint8_t *qkvz_w, *ba_w;  // [6144][2048] (q|k|v|z), [32][2048] (b|a)
  const float *qkvz_s, *ba_s, *A_log;
  const half *conv_w, *dt_bias, *gnorm_w;  // conv_w [4096][4]
  half* conv_cache;  // [slots][3][4096] (slot stride conv_stride)
  float* ssm_cache;  // [slots][16][128 v][128 k] (slot stride ssm_stride)
  int64_t conv_stride, ssm_stride;
  const int* state_idx;  // [1] slot of this sequence (< 0: padding)
  // ATTN
  const uint8_t*
      qkv_w;  // [4608][2048]: per head q (256) | gate (256), then k, v
  const float* qkv_s;
  const half *q_norm_w, *k_norm_w;
  half* kv;                           // [blocks][1][bs][K 256 | V 256]
  int64_t blk_stride;                 // halves per block
  int bs;                             // tokens per block (multiple of 64)
  const int *block_table, *seq_lens;  // row 0 / [0] of this sequence
  int bt_stride;                      // batched: ints per block-table row
  const int64_t *slot, *pos;          // slot_mapping[0], positions[0]
};

struct Scratch {
  uint32_t* cnt;     // EPOCH + 32
  float* qkv_raw;    // ATTN: QKV_ROWS fp32 GEMV outputs
  float* part_ml;    // ATTN: NWG * NQ * 2
  float* part_o;     // ATTN: NWG * NQ * HD
  half* attn_out;    // ATTN: NQ * HD (gated attention output)
  half* qkv;         // GDN : CONVD
  half* core;        // GDN : VD
  half* zbuf;        // GDN : VD
  half* ba;          // GDN : 2 * NVH
  float* part_rt;    // NWG * NE
  float* part_sh;    // NWG * 2 * INTER
  float* part_sg;    // NWG * 2
  half* hbuf;        // (TOPK + 1) * INTER
  half* dummy_conv;  // GDN padding slot: 3 * CONVD
  float* dummy_ssm;  // GDN padding slot: NVH * DV * DK
  uint64_t* mbox;    // 2 * HID words written by the peer card
  uint64_t* peer;    // the peer card's mbox (P2P mapping)
};

// bytes of one layer's scratch (everything except mbox / peer), 256-B aligned
// pieces
inline size_t scratch_bytes() {
  size_t n = 0;
  auto add = [&](size_t b) { n += (b + 255) & ~size_t(255); };
  add((EPOCH + 32) * 4);
  add(QKV_ROWS * 4);
  add(NWG * NQ * 2 * 4);
  add((size_t)NWG * NQ * HD * 4);
  add(NQ * HD * 2);
  add(CONVD * 2);
  add(VD * 2);
  add(VD * 2);
  add(2 * NVH * 2);
  add((size_t)NWG * NE * 4);
  add((size_t)NWG * 2 * INTER * 4);
  add(NWG * 2 * 4);
  add((TOPK + 1) * INTER * 2);
  add(3 * CONVD * 2);
  add((size_t)NVH * DV * DK * 4);
  return n;
}
// carve a zero-initialized device buffer of scratch_bytes() into the Scratch
// pointers
inline Scratch scratch_carve(void* base, uint64_t* mbox, uint64_t* peer) {
  Scratch S{};
  uint8_t* p = static_cast<uint8_t*>(base);
  auto take = [&](size_t b) {
    void* r = p;
    p += (b + 255) & ~size_t(255);
    return r;
  };
  S.cnt = (uint32_t*)take((EPOCH + 32) * 4);
  S.qkv_raw = (float*)take(QKV_ROWS * 4);
  S.part_ml = (float*)take(NWG * NQ * 2 * 4);
  S.part_o = (float*)take((size_t)NWG * NQ * HD * 4);
  S.attn_out = (half*)take(NQ * HD * 2);
  S.qkv = (half*)take(CONVD * 2);
  S.core = (half*)take(VD * 2);
  S.zbuf = (half*)take(VD * 2);
  S.ba = (half*)take(2 * NVH * 2);
  S.part_rt = (float*)take((size_t)NWG * NE * 4);
  S.part_sh = (float*)take((size_t)NWG * 2 * INTER * 4);
  S.part_sg = (float*)take(NWG * 2 * 4);
  S.hbuf = (half*)take((TOPK + 1) * INTER * 2);
  S.dummy_conv = (half*)take(3 * CONVD * 2);
  S.dummy_ssm = (float*)take((size_t)NVH * DV * DK * 4);
  S.mbox = mbox;
  S.peer = peer;
  return S;
}
constexpr size_t MBOX_WORDS = 2 * HID;  // per layer, rows-1 region

// ----------------------------------------------------------------------------
// helpers
using mb_ref = sycl::atomic_ref<
    uint64_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::system,
    sycl::access::address_space::global_space>;
// A peer write may land as two 4-byte halves (e.g. GPUs on different CPU root
// ports): each half carries the 16-bit tag (word = tag|hi16, tag|lo16), so a
// word that mixes two writes never matches.
inline uint64_t mb_word(uint32_t v, uint32_t tag) {
  const uint64_t t = tag & 0xffffu;
  return (t << 48) | ((uint64_t)(v >> 16) << 32) | (t << 16) | (v & 0xffffu);
}
inline bool mb_match(uint64_t w, uint32_t tag) {
  const uint32_t t = tag & 0xffffu;
  return (uint32_t)(w >> 48) == t && ((uint32_t)(w >> 16) & 0xffffu) == t;
}
inline uint32_t mb_val(uint64_t w) {
  return ((uint32_t)(w >> 32) << 16) | ((uint32_t)w & 0xffffu);
}
inline void mb_send(uint64_t* peer, float v, uint32_t tag) {
  mb_ref(*peer).store(mb_word(sycl::bit_cast<uint32_t>(v), tag));
}
inline float mb_recv(uint64_t* mine, uint32_t tag) {
  uint64_t w;
  do {
    w = mb_ref(*mine).load();
  } while (!mb_match(w, tag));
  return sycl::bit_cast<float>(mb_val(w));
}

inline uint32_t fp8x2_lo(uint32_t w) {  // bytes 0,2 -> two fp16 = e4m3 / 256
  uint32_t t = w & 0x00ff00ffu;
  return ((t & 0x00800080u) << 8) | ((t & 0x007f007fu) << 7);
}
inline uint32_t fp8x2_hi(uint32_t w) {  // bytes 1,3
  uint32_t t = (w >> 8) & 0x00ff00ffu;
  return ((t & 0x00800080u) << 8) | ((t & 0x007f007fu) << 7);
}
inline h2 as_h2(uint32_t u) { return sycl::bit_cast<h2>(u); }

// 16 fp8 weights . 16 activations stored permuted (x0,x2,x1,x3 per 4) in SLM.
inline float dot16_fp8(sycl::uint4 w, const uint32_t* xp) {
  sycl::uint4 xa = *reinterpret_cast<const sycl::uint4*>(xp);
  sycl::uint4 xb = *reinterpret_cast<const sycl::uint4*>(xp + 4);
  h2 acc = as_h2(fp8x2_lo(w.x())) * as_h2(xa.x());
  acc += as_h2(fp8x2_hi(w.x())) * as_h2(xa.y());
  acc += as_h2(fp8x2_lo(w.y())) * as_h2(xa.z());
  acc += as_h2(fp8x2_hi(w.y())) * as_h2(xa.w());
  acc += as_h2(fp8x2_lo(w.z())) * as_h2(xb.x());
  acc += as_h2(fp8x2_hi(w.z())) * as_h2(xb.y());
  acc += as_h2(fp8x2_lo(w.w())) * as_h2(xb.z());
  acc += as_h2(fp8x2_hi(w.w())) * as_h2(xb.w());
  return (float)acc[0] + (float)acc[1];
}
// 16 fp8 weights . 16 activations in natural order (registers)
inline float dot16_fp8_nat(sycl::uint4 w, const sycl::vec<half, 16>& x) {
  h2 acc = as_h2(fp8x2_lo(w.x())) * h2(x[0], x[2]);
  acc += as_h2(fp8x2_hi(w.x())) * h2(x[1], x[3]);
  acc += as_h2(fp8x2_lo(w.y())) * h2(x[4], x[6]);
  acc += as_h2(fp8x2_hi(w.y())) * h2(x[5], x[7]);
  acc += as_h2(fp8x2_lo(w.z())) * h2(x[8], x[10]);
  acc += as_h2(fp8x2_hi(w.z())) * h2(x[9], x[11]);
  acc += as_h2(fp8x2_lo(w.w())) * h2(x[12], x[14]);
  acc += as_h2(fp8x2_hi(w.w())) * h2(x[13], x[15]);
  return (float)acc[0] + (float)acc[1];
}
// acc[0..15] += (16 fp8 weights, natural column order) * x   (fp16 products,
// fp32 accumulation)
inline void axpy16_fp8(float (&acc)[16], sycl::uint4 w, half x) {
  const h2 xx(x, x);
  const uint32_t ww[4] = {w.x(), w.y(), w.z(), w.w()};
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const h2 lo = as_h2(fp8x2_lo(ww[q])) * xx;  // columns 4q, 4q+2
    const h2 hi = as_h2(fp8x2_hi(ww[q])) * xx;  // columns 4q+1, 4q+3
    acc[4 * q] += (float)lo[0];
    acc[4 * q + 2] += (float)lo[1];
    acc[4 * q + 1] += (float)hi[0];
    acc[4 * q + 3] += (float)hi[1];
  }
}
template <int KCH>
inline void load_row(sycl::uint4 (&w)[KCH], const uint8_t* row, int lane) {
#pragma unroll
  for (int j = 0; j < KCH; ++j)
    w[j] = *reinterpret_cast<const sycl::uint4*>(row + j * 256 + lane * 16);
}
template <int KCH>
inline float dot_row_regs(
    sycl::sub_group sg,
    int lane,
    const sycl::uint4 (&w)[KCH],
    const uint32_t* xp) {
  float sacc = 0.f;
#pragma unroll
  for (int j = 0; j < KCH; ++j)
    sacc += dot16_fp8(w[j], xp + (j * 256 + lane * 16) / 2);
  return sycl::reduce_over_group(sg, sacc, sycl::plus<float>());
}
// Software-pipelined row GEMV: NB batches of RPB fp8 rows (K = KCH*256 B);
// batch b+1's loads are issued before batch b is computed. out(b, r, v) gets
// the full (sub-group reduced) sum.
template <int NB, int RPB, int KCH, typename RowP, typename XP, typename Out>
inline void gemv_pipe(sycl::sub_group sg, int lane, RowP rowp, XP xp, Out out) {
  sycl::uint4 buf[2][RPB][KCH];
#pragma unroll
  for (int r = 0; r < RPB; ++r)
#pragma unroll
    for (int j = 0; j < KCH; ++j)
      buf[0][r][j] = *reinterpret_cast<const sycl::uint4*>(
          rowp(0, r) + j * 256 + lane * 16);
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    if (b + 1 < NB) {
#pragma unroll
      for (int r = 0; r < RPB; ++r)
#pragma unroll
        for (int j = 0; j < KCH; ++j)
          buf[(b + 1) & 1][r][j] = *reinterpret_cast<const sycl::uint4*>(
              rowp(b + 1, r) + j * 256 + lane * 16);
    }
#pragma unroll
    for (int r = 0; r < RPB; ++r) {
      const uint32_t* x = xp(b, r);
      float sacc = 0.f;
#pragma unroll
      for (int j = 0; j < KCH; ++j)
        sacc += dot16_fp8(buf[b & 1][r][j], x + (j * 256 + lane * 16) / 2);
      out(b, r, sycl::reduce_over_group(sg, sacc, sycl::plus<float>()));
    }
  }
}
// x[k] -> permuted slot (x0,x2,x1,x3 per 4): swap bits 0 and 1 of k.
inline void store_perm(half* xp, int k, half v) {
  xp[(k & ~3) | ((k & 1) << 1) | ((k >> 1) & 1)] = v;
}

inline uint32_t uc_load32(const uint32_t* p) {  // L1-uncached plain load
  syclex::annotated_ptr<
      const uint32_t,
      decltype(syclex::properties{intelex::read_hint<intelex::cache_control<
          intelex::cache_mode::uncached,
          syclex::cache_level::L1>>})>
      a(p);
  return *a;
}
// state store with L1 streaming + L3 write-back (IGC LSC builtin)
typedef unsigned int lsc_u4 __attribute__((ext_vector_type(4)));
#ifdef __SYCL_DEVICE_ONLY__
extern "C" SYCL_EXTERNAL void __builtin_IB_lsc_store_global_uint4(
    __attribute__((opencl_global)) lsc_u4* base,
    int off,
    lsc_u4 val,
    int cache);
#endif
inline void st_store8(float* p, const sycl::vec<float, 8>& v) {
#ifdef __SYCL_DEVICE_ONLY__
  auto* g = (__attribute__((opencl_global))
             lsc_u4*)(__attribute__((opencl_global)) void*)(p);
  lsc_u4 a = {
      sycl::bit_cast<unsigned>(v[0]),
      sycl::bit_cast<unsigned>(v[1]),
      sycl::bit_cast<unsigned>(v[2]),
      sycl::bit_cast<unsigned>(v[3])};
  lsc_u4 b = {
      sycl::bit_cast<unsigned>(v[4]),
      sycl::bit_cast<unsigned>(v[5]),
      sycl::bit_cast<unsigned>(v[6]),
      sycl::bit_cast<unsigned>(v[7])};
  __builtin_IB_lsc_store_global_uint4(g, 0, a, 6);  // LSC_STCC_L1S_L3WB
  __builtin_IB_lsc_store_global_uint4(g, 1, b, 6);
#else
  (void)p;
  (void)v;
#endif
}

using cnt_ref = sycl::atomic_ref<
    uint32_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>;
// NOTE: the release fence also waits for the thread's outstanding LOADS ->
// issue preloads after signal().
inline void signal(sycl::nd_item<1> it, uint32_t* c) {
  sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
  sycl::group_barrier(it.get_group());
  if (it.get_local_linear_id() == 0)
    cnt_ref(*c).fetch_add(1u, sycl::memory_order::release);
}
// No device-scope acquire fence after the barrier: it lowers to an L1
// invalidate on every thread, which delays the hand-off. Not needed here: a WG
// reads another WG's hand-off data only after the matching wait and never
// touches those lines earlier in the same launch, and L1 starts clean at every
// launch.
// The counters and targets wrap at 2^32 (32 producers a call: after 2^27
// calls of a layer), so compare the signed difference.
inline void wait_for(sycl::nd_item<1> it, uint32_t* c, uint32_t target) {
  if (it.get_local_linear_id() == 0)
    while ((int32_t)(cnt_ref(*c).load(sycl::memory_order::relaxed) - target) <
           0) {
    }
  sycl::group_barrier(it.get_group());
}
// in-WG barrier with SLM-only memory semantics (group_barrier also fences
// global memory = waits for loads)
inline void wg_bar(sycl::group<1> g) {
#if defined(__SYCL_DEVICE_ONLY__)
  __spirv_ControlBarrier(
      __spv::Scope::Workgroup,
      __spv::Scope::Workgroup,
      __spv::MemorySemanticsMask::AcquireRelease |
          __spv::MemorySemanticsMask::WorkgroupMemory);
#else
  sycl::group_barrier(g);
#endif
}

// ---- Xe2 DPAS + 2D block loads (inline vISA / IGC builtins, as the sycl-tla
// arch primitives): joint_matrix element access (get_wi_data) does not build
// with IGC 2.38.
typedef float v8f __attribute__((ext_vector_type(8)));
typedef unsigned int v8u __attribute__((ext_vector_type(8)));
typedef unsigned int v4u __attribute__((ext_vector_type(4)));
typedef int v2i __attribute__((ext_vector_type(2)));
#ifdef __SYCL_DEVICE_ONLY__
extern "C" SYCL_EXTERNAL v8u
__builtin_IB_subgroup_block_read_flat_transpose_u32_k8(
    intptr_t base, int width_m1, int height_m1, int pitch_m1, v2i xy);
extern "C" SYCL_EXTERNAL v8u
__builtin_IB_subgroup_block_read_flat_transform_u16_k16(
    intptr_t base, int width_m1, int height_m1, int pitch_m1, v2i xy);
#endif
// d[8 rows x 16 cols] (lane = column, element i = row i) += a[8 x 16 half] .
// b[16 x 16 half]
//   a: dpas A layout, lane l element j = (A[2j + l/8][2(l%8)], A[2j +
//   l/8][2(l%8) + 1]) b: VNNI, lane n element p = (B[2p][n], B[2p + 1][n])
inline void dpas_hf(v8f& d, const v4u& a, const v8u& b) {
#ifdef __SYCL_DEVICE_ONLY__
  asm("{\n"
      ".decl DST     v_type=G type=f num_elts=128 alias=<%0,0>\n"
      ".decl SRC1_UD v_type=G type=UD num_elts=128 alias=<%2,0>\n"
      ".decl SRC2_UD v_type=G type=UD num_elts=64 alias=<%1,0>\n"
      "dpas.hf.hf.8.8 (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD(0,0)\n"
      "}\n"
      : "+rw"(d)
      : "rw"(a), "rw"(b));
#else
  (void)d;
  (void)a;
  (void)b;
#endif
}
// K tile: 16 token rows x 16 dims at dims dw*2.. (dw in dwords) -> VNNI B of
// K^T (lane = token)
inline v8u load_k_tile(const half* row0, int dw) {
#ifdef __SYCL_DEVICE_ONLY__
  return __builtin_IB_subgroup_block_read_flat_transpose_u32_k8(
      (intptr_t)row0, HD * 2 - 1, 15, KROW * 2 - 1, v2i{dw, 0});
#else
  (void)row0;
  (void)dw;
  return v8u{};
#endif
}
// V tile: 16 token rows x 16 dims (VNNI-transformed) of a 128-dim half
inline v8u load_v_tile(const half* row0, int col) {
#ifdef __SYCL_DEVICE_ONLY__
  return __builtin_IB_subgroup_block_read_flat_transform_u16_k16(
      (intptr_t)row0, 128 * 2 - 1, 15, KROW * 2 - 1, v2i{col, 0});
#else
  (void)row0;
  (void)col;
  return v8u{};
#endif
}
inline uint32_t pack_h2(half a, half b) {
  return (uint32_t)sycl::bit_cast<uint16_t>(a) |
         ((uint32_t)sycl::bit_cast<uint16_t>(b) << 16);
}

// one head (256) -> (GemmaRMS + NeoX RoPE on dims 0..63 if nw) -> fp16 dst (and
// dst2).
inline void head_norm_rope(
    sycl::sub_group sg,
    int lane,
    const float* src,
    const half* nw,
    const float* cs,
    half* dst,
    half* dst2) {
  float y[16];
#pragma unroll
  for (int i = 0; i < 16; ++i)
    y[i] = (float)(half)src[lane * 16 + i];
  if (nw) {
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i)
      ss += y[i] * y[i];
    ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
    const float rs = sycl::rsqrt(ss / HD + EPS);
#pragma unroll
    for (int i = 0; i < 16; ++i)
      y[i] = y[i] * rs * (1.f + (float)nw[lane * 16 + i]);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float other = sycl::permute_group_by_xor(sg, y[i], 2);
      if (lane < 4) {
        const int fi = (lane & 1) * 16 + i;
        y[i] = lane < 2 ? y[i] * cs[fi] - other * cs[32 + fi]
                        : y[i] * cs[fi] + other * cs[32 + fi];
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    dst[lane * 16 + i] = (half)y[i];
    if (dst2) dst2[lane * 16 + i] = (half)y[i];
  }
}

// ----------------------------------------------------------------------------
// SLM layout GDN
constexpr int G_X = 0, G_XN = 4096, G_QK = 8192, G_RED = G_QK + 2048,
              G_ROUTE = G_RED + 512, G_AO = G_ROUTE + 128, G_MR = G_AO + 8192,
              G_BYTES = G_MR + 16384;
// ATTN
constexpr int A_X = 0, A_XN = 8192, A_Q = A_XN + 4096, A_KV = A_Q + 8192,
              A_RED = A_KV + 2048, A_ROUTE = A_RED + 512, A_CS = A_ROUTE + 128,
              A_ST = A_CS + 256, A_QPK = A_ST + 2048, A_AO = A_QPK + 4096,
              A_MR = A_AO + 8192, A_BYTES = A_AO + 65536;
static_assert(
    A_MR + 16384 <= A_BYTES, "tail SLM inside the attention partial area");

struct Slm {  // tail view
  half* xp;
  const uint32_t* xp32;
  half* xn;
  float* red;
  int* route_e;
  float* route_w;
  float* sgate;
  uint8_t* ao;
  float* mr;
  float* qk;
};

// ----------------------------------------------------------------------------
// tail Entry: the mixer output o (2048 fp16) is in SLM xp (permuted), the
// out_proj rows ow[] of this SG are in registers (issued after the previous
// signal), prt_r / prt_s the router / shared gate|up column blocks of this WG.
inline void tail(
    sycl::nd_item<1> it,
    const Ptrs& P,
    const Scratch& S,
    const Slm& L,
    uint32_t call,
    int c_o,
    const sycl::uint4 (&ow)[2][8],
    const sycl::uint4 (&prt_r)[4],
    const sycl::uint4 (&prt_s)[SHR / 4]) {
  const uint32_t mtag = uc_load32(S.cnt + MBT) + 1u;  // rows-1 mailbox region
  const int wg = it.get_group(0);
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  auto grp = it.get_group();
  {
    const float sc = 256.f * P.out_s[0];
    const int r0 = wg * 64 + s * 2;
    const float o0 = dot_row_regs<8>(sg, lane, ow[0], L.xp32),
                o1 = dot_row_regs<8>(sg, lane, ow[1], L.xp32);
    float ov = (lane == 0 ? o0 : o1) * sc;
    {  // all-reduce #1 over this WG's 64 rows, one contiguous block of 64
       // work-items. a + b == b + a
      float* ob = reinterpret_cast<float*>(L.ao + 1536);
      if (lane < 2) ob[s * 2 + lane] = ov;
      wg_bar(grp);
      if (lid < 64) {
        float v = ob[lid];
        mb_send(S.peer + wg * 64 + lid, v, mtag);
        v += mb_recv(S.mbox + wg * 64 + lid, mtag);
        ob[lid] = v;
      }
      wg_bar(grp);
      if (lane < 2) ov = ob[s * 2 + lane];
    }
    float* rblk = reinterpret_cast<float*>(L.ao);     // this WG's 64 res2 rows
    half* xbn = reinterpret_cast<half*>(L.ao + 256);  // (1+w) * res2, natural
    half* xbp = reinterpret_cast<half*>(L.ao + 384);  // same, permuted
    if (lane < 2) {
      const int r = r0 + lane;
      const float res = (float)(half)((float)P.hs_in[r] + (float)P.res_in[r]);
      const half r2 = (half)(ov + res);
      P.res_out[r] = r2;
      rblk[s * 2 + lane] = (float)r2;
    }
    wg_bar(grp);
    if (lid < 64)
      xbn[lid] =
          (half)(rblk[lid] * (1.f + (float)P.post_norm_w[wg * 64 + lid]));
    if (s == 2) {
      float q = rblk[lane * 4] * rblk[lane * 4] +
                rblk[lane * 4 + 1] * rblk[lane * 4 + 1] +
                rblk[lane * 4 + 2] * rblk[lane * 4 + 2] +
                rblk[lane * 4 + 3] * rblk[lane * 4 + 3];
      q = sycl::reduce_over_group(sg, q, sycl::plus<float>());
      if (lane == 0) S.part_sg[wg * 2 + 1] = q;
    }
    wg_bar(grp);
    if (lid < 64) store_perm(xbp, lid, xbn[lid]);
    wg_bar(grp);
    {  // router partials: 8 experts per SG
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const sycl::vec<half, 8> wh =
            sycl::bit_cast<sycl::vec<half, 8>>(prt_r[t]);
        float v = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
          v += (float)wh[i] * (float)xbn[(lane & 7) * 8 + i];
        v += sycl::permute_group_by_xor(sg, v, 4);
        v += sycl::permute_group_by_xor(sg, v, 2);
        v += sycl::permute_group_by_xor(sg, v, 1);
        if ((lane & 7) == 0)
          S.part_rt[wg * NE + s * 8 + 2 * t + (lane >> 3)] = v;
      }
      const uint32_t* xb32 = reinterpret_cast<const uint32_t*>(xbp);
#pragma unroll
      for (int t = 0; t < SHR / 4;
           ++t) {  // shared gate|up partials: SHR rows per SG
        float v = dot16_fp8(prt_s[t], xb32 + (lane & 3) * 8);
        v += sycl::permute_group_by_xor(sg, v, 2);
        v += sycl::permute_group_by_xor(sg, v, 1);
        if ((lane & 3) == 0)
          S.part_sh[wg * 2 * INTER + s * SHR + 4 * t + (lane >> 2)] = v;
      }
      if (s == NSG - 1) {  // shared-expert gate partial
        float v = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i)
          v += (float)P.sgate_w[wg * 64 + lane * 4 + i] *
               (float)xbn[lane * 4 + i];
        v = sycl::reduce_over_group(sg, v, sycl::plus<float>());
        if (lane == 0) S.part_sg[wg * 2] = v;
      }
    }
  }
  signal(it, S.cnt + c_o * 16);

  // ---------------- S5': reduce the partials (fixed order): rs, x2, logits,
  // shared slice, gate -----
  wait_for(it, S.cnt + c_o * 16, NWG * call);
  float* lgs = reinterpret_cast<float*>(L.ao + 512);  // 256 logits
  {
    float ss = 0.f;
#pragma unroll
    for (int w2i = 0; w2i < NWG; ++w2i)
      ss += S.part_sg[w2i * 2 + 1];
    const float rs = sycl::rsqrt(ss / HID + EPS);
    half x2[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      x2[i] = (half)((float)P.res_out[lid * 4 + i] * rs *
                     (1.f + (float)P.post_norm_w[lid * 4 + i]));
    *reinterpret_cast<sycl::vec<half, 4>*>(L.xn + lid * 4) =
        sycl::vec<half, 4>(x2[0], x2[1], x2[2], x2[3]);
    if (lid < NE) {
      float a = 0.f;
#pragma unroll
      for (int w2i = 0; w2i < NWG; ++w2i)
        a += S.part_rt[w2i * NE + lid];
      lgs[lid] = a * rs;
    } else if (lid < NE + SPW) {
      const int i = wg * SPW + lid - NE;
      float g = 0.f, u = 0.f;
#pragma unroll
      for (int w2i = 0; w2i < NWG; ++w2i) {
        g += S.part_sh[w2i * 2 * INTER + i];
        u += S.part_sh[w2i * 2 * INTER + INTER + i];
      }
      const float sc = 256.f * P.sw13_s[0] * rs;
      g *= sc;
      u *= sc;
      S.hbuf[TOPK * INTER + i] = (half)(g / (1.f + sycl::exp(-g)) * u);
    } else if (lid == NE + 32) {
      float a = 0.f;
#pragma unroll
      for (int w2i = 0; w2i < NWG; ++w2i)
        a += S.part_sg[w2i * 2];
      L.sgate[0] = 1.f / (1.f + sycl::exp(-a * rs));
    }
    wg_bar(grp);
  }
  // top-8 + softmax over the selected logits (ties: lower index first)
  if (s == 0) {
    float lg[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
      lg[i] = lgs[lane * 16 + i];
    float lmax = -INFINITY, sel_v[TOPK];
    int seli[TOPK];
    for (int k = 0; k < TOPK; ++k) {
      float bv = -INFINITY;
      int bi = 1 << 20;
#pragma unroll
      for (int i = 0; i < 16; ++i)
        if (lg[i] > bv) {
          bv = lg[i];
          bi = lane * 16 + i;
        }
      const float mv = sycl::reduce_over_group(sg, bv, sycl::maximum<float>());
      const int mi = sycl::reduce_over_group(
          sg, bv == mv ? bi : (1 << 20), sycl::minimum<int>());
#pragma unroll
      for (int i = 0; i < 16; ++i)
        if (lane * 16 + i == mi) lg[i] = -INFINITY;
      sel_v[k] = mv;
      seli[k] = mi;
      if (k == 0) lmax = mv;
    }
    float sum = 0.f, e[TOPK];
    for (int k = 0; k < TOPK; ++k) {
      e[k] = sycl::exp(sel_v[k] - lmax);
      sum += e[k];
    }
    if (lane == 0)
      for (int k = 0; k < TOPK; ++k) {
        L.route_e[k] = seli[k] & (NE - 1);
        L.route_w[k] = e[k] / sum;
      }
  }
  wg_bar(grp);

  // ---------------- S6: routed gate|up, layout w13[E][K = 2048][N = 512]
  // --------------------------- WG = (slot j = wg/4, quarter q4): gate columns
  // q4*64..+64 and the same up columns (+256). SG s: k = s*64..+64; per load
  // lanes 0..7 take row k0, lanes 8..15 row k0+1; lane&3 = 16-column group,
  // lane&4 = gate / up. All 32 loads of the SG in flight at once; the down
  // segments of this WG's 64 output rows (routed slot s/4, k = (s&3)*64 + 4t +
  // lane/4; shared rows wg*64 + 2s + rr) refill the freed registers.
  sycl::uint4 wd[16], wsd[2];
  {
    const int j = wg >> 2, q4 = wg & 3, e = L.route_e[j];
    const uint8_t* base = P.w13 + (size_t)e * HID * (2 * INTER) +
                          ((lane & 4) ? INTER : 0) + q4 * 64 + (lane & 3) * 16;
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
      acc[i] = 0.f;
    {
      sycl::uint4 gw[32];
#pragma unroll
      for (int t = 0; t < 32; ++t)
        gw[t] = *reinterpret_cast<const sycl::uint4*>(
            base + (size_t)(s * 64 + 2 * t + (lane >> 3)) * (2 * INTER));
      const int jd = s >> 2, ed = L.route_e[jd];
      const uint8_t* based =
          P.w2 + (size_t)ed * INTER * HID + wg * 64 + (lane & 3) * 16;
#pragma unroll
      for (int t = 0; t < 32; ++t) {
        axpy16_fp8(acc, gw[t], L.xn[s * 64 + 2 * t + (lane >> 3)]);
        if (t & 1)
          wd[t >> 1] = *reinterpret_cast<const sycl::uint4*>(
              based +
              (size_t)((s & 3) * 64 + 4 * (t >> 1) + (lane >> 2)) * HID);
      }
#pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        wsd[rr] = *reinterpret_cast<const sycl::uint4*>(
            P.sw2 + (size_t)(wg * 64 + 2 * s + rr) * INTER + lane * 16);
    }
#pragma unroll
    for (int i = 0; i < 16; ++i)
      acc[i] += sycl::permute_group_by_xor(sg, acc[i], 8);
    if (lane < 8) {
#pragma unroll
      for (int i = 0; i < 16; ++i)
        L.mr[s * 128 + lane * 16 + i] = acc[i];
    }
    wg_bar(grp);
    if (lid < 128) {
      float a = 0.f;
#pragma unroll
      for (int w2i = 0; w2i < NSG; ++w2i)
        a += L.mr[w2i * 128 + lid];
      L.qk[lid] = a * 256.f * P.w13_s[e];  // lid 0..63: gate column q4*64 + lid
                                           // ; 64..127: the same up column
    }
    wg_bar(grp);
    if (lid < 64) {
      const float g = L.qk[lid], u = L.qk[64 + lid];
      S.hbuf[j * INTER + q4 * 64 + lid] = (half)(g / (1.f + sycl::exp(-g)) * u);
    }
  }
  signal(it, S.cnt + C_GU * 16);

  // ---------------- S7: down for this WG's 64 rows ; all-reduce #2
  // -----------------------------
  wait_for(it, S.cnt + C_GU * 16, NWG * call);
  if (wg == 0 && lid == 0) {  // every WG has read the epoch and the tag
    cnt_ref(S.cnt[EPOCH]).store(call);
    cnt_ref(S.cnt[MBT]).store(mtag);
  }
  {
    const int j = s >> 2, e = L.route_e[j];
    const float coef = L.route_w[j] * 256.f * P.w2_s[e];
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
      acc[i] = 0.f;
#pragma unroll
    for (int t = 0; t < 16; ++t)
      axpy16_fp8(
          acc, wd[t], S.hbuf[j * INTER + (s & 3) * 64 + 4 * t + (lane >> 2)]);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      acc[i] += sycl::permute_group_by_xor(sg, acc[i], 4);
      acc[i] += sycl::permute_group_by_xor(sg, acc[i], 8);
    }
    if (lane < 4) {
#pragma unroll
      for (int i = 0; i < 16; ++i)
        L.mr[s * 64 + lane * 16 + i] = acc[i] * coef;
    }
    const sycl::vec<half, 16> hx =
        *reinterpret_cast<const sycl::vec<half, 16>*>(
            S.hbuf + TOPK * INTER + lane * 16);
    const float cs = L.sgate[0] * 256.f * P.sw2_s[0];
    float d0 = dot16_fp8_nat(wsd[0], hx), d1 = dot16_fp8_nat(wsd[1], hx);
    d0 = sycl::reduce_over_group(sg, d0, sycl::plus<float>());
    d1 = sycl::reduce_over_group(sg, d1, sycl::plus<float>());
    if (lane < 2) L.red[2 * s + lane] = (lane == 0 ? d0 : d1) * cs;
    wg_bar(grp);
    if (lid < 64) {
      float a = 0.f;
#pragma unroll
      for (int w2i = 0; w2i < NSG; ++w2i)
        a += L.mr[w2i * 64 + lid];
      a += L.red[lid];
      mb_send(S.peer + HID + wg * 64 + lid, a, mtag);  // all-reduce #2
      a += mb_recv(S.mbox + HID + wg * 64 + lid, mtag);
      P.hid_out[wg * 64 + lid] = (half)a;
    }
  }
}

// PRT column blocks of this WG (router fp16, shared gate|up fp8) -> registers
inline void load_prt(
    const Ptrs& P,
    int wg,
    int s,
    int lane,
    sycl::uint4 (&prt_r)[4],
    sycl::uint4 (&prt_s)[SHR / 4]) {
#pragma unroll
  for (int t = 0; t < 4;
       ++t)  // experts s*8 + 2t + (lane>>3), halves (lane&7)*8..+8
    prt_r[t] = *reinterpret_cast<const sycl::uint4*>(
        P.router_w + (size_t)(s * 8 + 2 * t + (lane >> 3)) * HID + wg * 64 +
        (lane & 7) * 8);
#pragma unroll
  for (int t = 0; t < SHR / 4;
       ++t)  // rows s*SHR + 4t + (lane>>2), bytes (lane&3)*16..+16
    prt_s[t] = *reinterpret_cast<const sycl::uint4*>(
        P.sw13 + (size_t)(s * SHR + 4 * t + (lane >> 2)) * HID + wg * 64 +
        (lane & 3) * 16);
}

// res = h + r ; x = GemmaRMS(res) -> xn (natural) and xp (permuted)
inline void input_norm(sycl::nd_item<1> it, const Ptrs& P, half* xn, half* xp) {
  const int lid = it.get_local_linear_id();
  float v[4], ss = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int k = lid * 4 + i;
    v[i] = (float)(half)((float)P.hs_in[k] + (float)P.res_in[k]);
    ss += v[i] * v[i];
  }
  ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
  const float rs = sycl::rsqrt(ss / HID + EPS);
  half x[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    x[i] = (half)(v[i] * rs * (1.f + (float)P.in_norm_w[lid * 4 + i]));
  if (xn)
    *reinterpret_cast<sycl::vec<half, 4>*>(xn + lid * 4) =
        sycl::vec<half, 4>(x[0], x[1], x[2], x[3]);
  *reinterpret_cast<sycl::vec<half, 4>*>(xp + lid * 4) =
      sycl::vec<half, 4>(x[0], x[2], x[1], x[3]);
}

// ----------------------------------------------------------------------------
// GDN layer
inline void
gdn_layer(sycl::nd_item<1> it, uint8_t* slm, const Ptrs& P, const Scratch& S) {
  const int wg = it.get_group(0);
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  auto grp = it.get_group();
  // epoch: plain L1-uncached load (an atomic load is an atomic_or RMW: the 1024
  // threads would serialize)
  const uint32_t call = uc_load32(S.cnt + EPOCH) + 1u;

  Slm L;
  L.xp = reinterpret_cast<half*>(slm + G_X);
  L.xp32 = reinterpret_cast<const uint32_t*>(slm + G_X);
  L.xn = reinterpret_cast<half*>(slm + G_XN);
  L.qk = reinterpret_cast<float*>(slm + G_QK);
  L.red = reinterpret_cast<float*>(slm + G_RED);
  L.route_e = reinterpret_cast<int*>(slm + G_ROUTE);
  L.route_w = reinterpret_cast<float*>(slm + G_ROUTE + 32);
  L.sgate = reinterpret_cast<float*>(slm + G_ROUTE + 64);
  L.ao = slm + G_AO;
  L.mr = reinterpret_cast<float*>(slm + G_MR);
  float* qk = L.qk;  // q[0..127], k[128..255], v[256..319], beta / g [320..321]
  float* red = L.red;

  const int h = wg >> 1, c = wg & 1,
            kh = h >> 1;  // S2 role: v head h (k head h/2), v rows c*64..+64
  const int sidx = P.state_idx[0];
  half* conv =
      sidx >= 0 ? P.conv_cache + (size_t)sidx * P.conv_stride : S.dummy_conv;
  float* ssm =
      sidx >= 0 ? P.ssm_cache + (size_t)sidx * P.ssm_stride : S.dummy_ssm;

  // ---------------- kernel start: inputs of S2 that do not depend on this
  // call's activations ----------
  sycl::vec<float, 8>
      st[2];  // recurrent state block: rows v = c*64 + 2s + rr, k = lane*8..+8
#pragma unroll
  for (int rr = 0; rr < 2; ++rr)
    st[rr] = *reinterpret_cast<const sycl::vec<float, 8>*>(
        ssm + ((size_t)h * DV + c * 64 + 2 * s + rr) * DK + lane * 8);
  const int cch = lid < 128   ? kh * DK + lid
                  : lid < 256 ? KD + kh * DK + (lid - 128)
                              : 2 * KD + h * DV + c * 64 + (lid - 256);
  half cst[3], cwt[4];
  if (lid < 320) {
#pragma unroll
    for (int i = 0; i < 3; ++i)
      cst[i] = conv[i * CONVD + cch];
    const sycl::vec<half, 4> w4 = *reinterpret_cast<const sycl::vec<half, 4>*>(
        P.conv_w + (size_t)cch * 4);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      cwt[i] = w4[i];
  }
  sycl::uint4 baw[8];  // b (SG 0) and a (SG 1) rows of v head h
  if (s < 2) load_row<8>(baw, P.ba_w + (size_t)(s * NVH + h) * HID, lane);

  input_norm(it, P, L.xn, L.xp);
  wg_bar(grp);

  // ---------------- S1: q|k|v rows of in_proj_qkvz (4 per SG) ----------------
  {
    const float sc = 256.f * P.qkvz_s[0];
    const int r0 = (wg * NSG + s) * 4;
    sycl::uint4 w[4][8];
#pragma unroll
    for (int b = 0; b < 4; ++b)
      load_row<8>(w[b], P.qkvz_w + (size_t)(r0 + b) * HID, lane);
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const float v = dot_row_regs<8>(sg, lane, w[b], L.xp32);
      if (lane == 0) S.qkv[r0 + b] = (half)(v * sc);
    }
  }
  signal(it, S.cnt + C_A * 16);
  sycl::uint4 zw[2][8];  // z rows of this WG: channel h*128 + c*64 + 2s + rr
                         // (consumed after the recurrence)
#pragma unroll
  for (int rr = 0; rr < 2; ++rr)
    load_row<8>(
        zw[rr],
        P.qkvz_w + (size_t)(CONVD + h * DV + c * 64 + 2 * s + rr) * HID,
        lane);
  if (s < 2) {  // b, a of v head h (fp16, as vLLM's GEMM output)
    const float v = dot_row_regs<8>(sg, lane, baw, L.xp32) * 256.f * P.ba_s[0];
    if (lane == 0) red[s] = (float)(half)v;
  }

  // ---------------- S2: conv1d + SiLU, L2 norm, gated delta rule -------------
  wait_for(it, S.cnt + C_A * 16, NWG * call);
  if (lid < 320) {
    const half xnew = S.qkv[cch];
    float acc = (float)cst[0] * (float)cwt[0];
    acc += (float)cst[1] * (float)cwt[1];
    acc += (float)cst[2] * (float)cwt[2];
    acc += (float)xnew * (float)cwt[3];
    qk[lid] = (float)(half)(acc / (1.f + sycl::exp(-acc)));
    // conv state <- (s1, s2, x): q/k channels by the (even h, c = 0) WG of the
    // k head, v channels by the owner
    if (lid >= 256 || ((h & 1) == 0 && c == 0)) {
      conv[cch] = cst[1];
      conv[CONVD + cch] = cst[2];
      conv[2 * CONVD + cch] = xnew;
    }
  }
  if (lid == 0) {  // beta, g of v head h
    const float b = red[0], a = red[1] + (float)P.dt_bias[h];
    const float sp = a < 20.f ? sycl::log(1.f + sycl::exp(a)) : a;
    qk[320] = 1.f / (1.f + sycl::exp(-b));
    qk[321] = sycl::exp(-sycl::exp(P.A_log[h]) * sp);
  }
  wg_bar(grp);
  if (s < 16) {  // L2 norms: q = SGs 0..7, k = SGs 8..15
    const float v = qk[lid];
    const float ss = sycl::reduce_over_group(sg, v * v, sycl::plus<float>());
    if (lane == 0) red[8 + s] = ss;
  }
  wg_bar(grp);
  {
    float qn = 0.f, kn = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      qn += red[8 + i];
      kn += red[16 + i];
    }
    const float iq = 1.f / sycl::sqrt(qn + 1e-6f),
                ik = 1.f / sycl::sqrt(kn + 1e-6f);
    const float scale = 0.08838834764831845f;  // 1 / sqrt(128)
    float qv[8], kv8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      qv[i] = qk[lane * 8 + i] * iq * scale;
      kv8[i] = qk[128 + lane * 8 + i] * ik;
    }
    const float beta = qk[320], g = qk[321];
    float kvm[2], o[2];
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      float d = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        st[rr][i] *= g;
        d += st[rr][i] * kv8[i];
      }
      kvm[rr] = d;
    }
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
      kvm[rr] = sycl::reduce_over_group(sg, kvm[rr], sycl::plus<float>());
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const float delta = (qk[256 + 2 * s + rr] - kvm[rr]) * beta;
      float d = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        st[rr][i] += kv8[i] * delta;
        d += st[rr][i] * qv[i];
      }
      o[rr] = d;
    }
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
      o[rr] = sycl::reduce_over_group(sg, o[rr], sycl::plus<float>());
    if (lane < 2)
      S.core[h * DV + c * 64 + 2 * s + lane] = (half)(lane == 0 ? o[0] : o[1]);
  }
  {  // z rows (registers since S1)
    const float sc = 256.f * P.qkvz_s[0];
    const float z0 = dot_row_regs<8>(sg, lane, zw[0], L.xp32),
                z1 = dot_row_regs<8>(sg, lane, zw[1], L.xp32);
    if (lane < 2)
      S.zbuf[h * DV + c * 64 + 2 * s + lane] =
          (half)((lane == 0 ? z0 : z1) * sc);
  }
  signal(it, S.cnt + C_B * 16);
#pragma unroll
  for (int rr = 0; rr < 2; ++rr)  // state write-back after the hand-off
    st_store8(
        ssm + ((size_t)h * DV + c * 64 + 2 * s + rr) * DK + lane * 8, st[rr]);
  sycl::uint4 ow[2][8], prt_r[4], prt_s[SHR / 4];
  load_row<8>(ow[0], P.out_w + (size_t)(wg * 64 + s * 2) * VD, lane);
  load_row<8>(ow[1], P.out_w + (size_t)(wg * 64 + s * 2 + 1) * VD, lane);
  load_prt(P, wg, s, lane, prt_r, prt_s);

  // ---------------- S3: gated RMSNorm (all heads) -> o in SLM ; tail
  // -------------------------
  wait_for(it, S.cnt + C_B * 16, NWG * call);
  {
    const sycl::vec<half, 4> o4 =
        *reinterpret_cast<const sycl::vec<half, 4>*>(S.core + lid * 4);
    const sycl::vec<half, 4> z4 =
        *reinterpret_cast<const sycl::vec<half, 4>*>(S.zbuf + lid * 4);
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      ss += (float)o4[i] * (float)o4[i];
    ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
    if (lane == 0) red[64 + s] = ss;  // head = 2 SGs
    wg_bar(grp);
    const int hh = s >> 1;
    const float rs =
        sycl::rsqrt((red[64 + 2 * hh] + red[64 + 2 * hh + 1]) / DV + EPS);
    half y[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float zf = (float)z4[i];
      y[i] = (half)((float)o4[i] * rs *
                    (float)P.gnorm_w[(lid * 4 + i) & (DV - 1)] *
                    (zf / (1.f + sycl::exp(-zf))));
    }
    *reinterpret_cast<sycl::vec<half, 4>*>(L.xp + lid * 4) =
        sycl::vec<half, 4>(y[0], y[2], y[1], y[3]);
    wg_bar(grp);
  }
  tail(it, P, S, L, call, C_O, ow, prt_r, prt_s);
}

// ----------------------------------------------------------------------------
// full-attention layer
#define MONO_ATTN_STEP(tb, NT)                                                \
  do {                                                                        \
    const half* kb[NT];                                                       \
    _Pragma("unroll") for (int tt = 0; tt < NT; ++tt) kb[tt] =                \
        trow((tb) + tt * 16);                                                 \
    v8f S4[NT];                                                               \
    _Pragma("unroll") for (int tt = 0; tt < NT; ++tt) {                       \
      S4[tt] = v8f(0.f);                                                      \
      _Pragma("unroll") for (int kt = 0; kt < 16; ++kt) {                     \
        const v8u bk = load_k_tile(kb[tt], kt * 8);                           \
        const v4u qa =                                                        \
            *reinterpret_cast<const v4u*>(qpk + (kt * 16 + lane) * 4);        \
        dpas_hf(S4[tt], qa, bk);                                              \
      }                                                                       \
    }                                                                         \
    float sv[NT][8];                                                          \
    _Pragma("unroll") for (int tt = 0; tt < NT; ++tt) {                       \
      const bool ok = (tb) + tt * 16 + lane < lc; /* token = column = lane */ \
      _Pragma("unroll") for (int i = 0; i < 8; ++i) sv[tt][i] =               \
          ok ? S4[tt][i] * SM_SCALE : -INFINITY;                              \
    }                                                                         \
    float alpha[8];                                                           \
    _Pragma("unroll") for (int i = 0; i < 8; ++i) {                           \
      float lm = sv[0][i];                                                    \
      _Pragma("unroll") for (int tt = 1; tt < NT; ++tt) lm =                  \
          sycl::fmax(lm, sv[tt][i]);                                          \
      const float mn = sycl::fmax(                                            \
          m[i], sycl::reduce_over_group(sg, lm, sycl::maximum<float>()));     \
      alpha[i] = mn == -INFINITY ? 1.f : sycl::native::exp2(m[i] - mn);       \
      m[i] = mn;                                                              \
      lp[i] *= alpha[i];                                                      \
    }                                                                         \
    _Pragma("unroll") for (int nt = 0; nt < 8; ++nt)                          \
        _Pragma("unroll") for (int i = 0; i < 8; ++i) O[nt][i] *= alpha[i];   \
    _Pragma("unroll") for (int tt = 0; tt < NT; ++tt) {                       \
      half pv[8];                                                             \
      _Pragma("unroll") for (int i = 0; i < 8; ++i) {                         \
        const float p =                                                       \
            m[i] == -INFINITY ? 0.f : sycl::native::exp2(sv[tt][i] - m[i]);   \
        lp[i] += p;                                                           \
        pv[i] = (half)p;                                                      \
      }                                                                       \
      /* P (lane = token, element = head) -> dpas A: lane l, element j = P[2j \
       * + l/8][2(l%8) + {0,1}] */                                            \
      v4u pa;                                                                 \
      _Pragma("unroll") for (int j = 0; j < 4; ++j) {                         \
        const int t0_ = 2 * (lane & 7);                                       \
        const half a0 = sycl::select_from_group(sg, pv[2 * j], t0_);          \
        const half a1 = sycl::select_from_group(sg, pv[2 * j + 1], t0_);      \
        const half b0 = sycl::select_from_group(sg, pv[2 * j], t0_ + 1);      \
        const half b1 = sycl::select_from_group(sg, pv[2 * j + 1], t0_ + 1);  \
        pa[j] = (lane < 8) ? pack_h2(a0, b0) : pack_h2(a1, b1);               \
      }                                                                       \
      const half* vb = kb[tt] + HD + dh * 128;                                \
      _Pragma("unroll") for (int nt = 0; nt < 8; ++nt)                        \
          dpas_hf(O[nt], pa, load_v_tile(vb, nt * 16));                       \
    }                                                                         \
  } while (0)

inline void
attn_layer(sycl::nd_item<1> it, uint8_t* slm, const Ptrs& P, const Scratch& S) {
  const int wg = it.get_group(0);
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  auto grp = it.get_group();
  const uint32_t call = uc_load32(S.cnt + EPOCH) + 1u;

  Slm L;
  L.xp = reinterpret_cast<half*>(slm + A_X);
  L.xp32 = reinterpret_cast<const uint32_t*>(slm + A_X);
  L.xn = reinterpret_cast<half*>(slm + A_XN);
  L.qk = reinterpret_cast<float*>(slm + A_KV);
  L.red = reinterpret_cast<float*>(slm + A_RED);
  L.route_e = reinterpret_cast<int*>(slm + A_ROUTE);
  L.route_w = reinterpret_cast<float*>(slm + A_ROUTE + 32);
  L.sgate = reinterpret_cast<float*>(slm + A_ROUTE + 64);
  L.ao = slm + A_AO;
  L.mr = reinterpret_cast<float*>(slm + A_MR);
  half* xp = L.xp;
  const uint32_t* xp32 = L.xp32;
  half* qs = reinterpret_cast<half*>(slm + A_Q);
  half* kvn = reinterpret_cast<half*>(slm + A_KV);
  float* red = L.red;
  float* st = reinterpret_cast<float*>(slm + A_ST);
  half* ao = reinterpret_cast<half*>(slm + A_AO);
  float* rope_cs = reinterpret_cast<float*>(slm + A_CS);
  auto qrow = [](int qi) { return (qi >> 8) * 512 + (qi & 255); };
  auto grow = [](int gi) {
    return gi < NQ * HD ? (gi >> 8) * 512 + 256 + (gi & 255)
                        : NQ * 512 + (gi - NQ * HD);
  };

  const int seqlen = P.seq_lens[0];
  const int lc =
      seqlen - 1;  // cached tokens (the new one is merged separately)
  const int bs = P.bs;
  // SG pair p = wg*16 + s/2 owns the cached tokens [p*T, p*T + T) of every
  // pass of APASS1 = 40960 tokens: 5 tiles of 16 a pass. The first pass is one
  // straight-line body (a loop around it is much slower; the 5th tile only
  // runs past 32K tokens); longer contexts take further passes in a loop.
  constexpr int T = 80;
  constexpr int APASS1 = NWG * 16 * T;
  const int t0 = (wg * 16 + (s >> 1)) * T;
  auto trow = [&](int t) {  // K row of token t (a 16-token tile never crosses a
                            // block: bs % 16 == 0); past the
    if (t >= lc) t = 0;     // context: token 0 (masked by the score select)
    return P.kv + (size_t)P.block_table[t / bs] * P.blk_stride +
           (size_t)(t % bs) * KROW;
  };

  input_norm(it, P, nullptr, xp);
  wg_bar(grp);

  // ---------------- S1: q rows (2 per SG) ------------------------------------
  {
    const float sc = 256.f * P.qkv_s[0];
    float* qr = S.qkv_raw;
    const int q0 = (wg * NSG + s) * 2;
    gemv_pipe<2, 1, 8>(
        sg,
        lane,
        [&](int b, int) { return P.qkv_w + (size_t)qrow(q0 + b) * HID; },
        [&](int, int) { return xp32; },
        [&](int b, int, float v) {
          if (lane == 0) qr[qrow(q0 + b)] = v * sc;
        });
  }
  signal(it, S.cnt + C_A * 16);
  {  // the q hand-off + q norm leave DRAM idle: warm L2 with the first K tile
     // of this SG pair
    if (t0 < lc) {
      const half* kb = trow(t0);
#pragma unroll
      for (int u = lane; u < 8 * 8; u += 16)
        syclex::prefetch(
            (void*)(kb + (size_t)((s & 1) * 8 + u / 8) * KROW + (u % 8) * 32),
            syclex::properties{syclex::prefetch_hint_L2});
    }
  }
  if (lid < 32) {  // RoPE table: Cody-Waite reduction by 2*pi, then cos/sin of
                   // |r| <= pi
    const float inv = 1.0f / sycl::pow(THETA, (float)(2 * lid) / 64.f);
    const float f = (float)P.pos[0] * inv;
    const float k = sycl::rint(f * 0.15915494309189535f);
    float r = sycl::fma(-k, 6.28125f, f);
    r = sycl::fma(-k, 1.9353071795864769e-3f, r);
    rope_cs[lid] = sycl::cos(r);
    rope_cs[32 + lid] = sycl::sin(r);
  }

  // ---------------- S2: q norm + RoPE, attention split along the context -----
  wait_for(it, S.cnt + C_A * 16, NWG * call);
  if (s < NQ)
    head_norm_rope(
        sg,
        lane,
        S.qkv_raw + s * 512,
        P.q_norm_w,
        rope_cs,
        qs + s * HD,
        nullptr);
  wg_bar(grp);
  // q in dpas A order: qpk[(kt*16 + l)*4 + j] = (q[2j + l/8][kt*16 + 2(l%8)],
  // .. + 1)
  uint32_t* qpk = reinterpret_cast<uint32_t*>(slm + A_QPK);
  for (int u = lid; u < 16 * 16 * 4; u += WGS) {
    const int j = u & 3, l = (u >> 2) & 15, kt = u >> 6;
    const half* src = qs + (2 * j + (l >> 3)) * HD + kt * 16 + 2 * (l & 7);
    qpk[u] = pack_h2(src[0], src[1]);
  }
  wg_bar(grp);
  {
    // SG unit: dh = s&1 (output dims half), chunks (wg*16 + s/2) + 512 i.
    // S^T = q[8 heads x 256] . K^T (A = q, B = K rows col_major); O[8 x 128] +=
    // P . V (B row_major). acc / A / B layout on Xe2: lane = column, element i
    // = row i.
    const int dh = s & 1;

    v8f O[8];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
      O[nt] = v8f(0.f);
    float m[8], lp[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      m[i] = -INFINITY;
      lp[i] = 0.f;
    }
    // In the one partial tile the owner zeroes the stale V rows first (a reused
    // block may hold NaN and 0 * NaN poisons P.V; the rows belong to this
    // sequence's block and are written by later tokens anyway).
    if (lc > 0 && t0 < lc) {
      if ((lc & 15) && (lc & ~15) >= t0 && (lc & ~15) < t0 + T) {
        half* vz = const_cast<half*>(trow(lc & ~15)) + HD + dh * 128;
        for (int r = lc & 15; r < 16; ++r)
          *reinterpret_cast<sycl::vec<half, 8>*>(
              vz + (size_t)r * KROW + lane * 8) = sycl::vec<half, 8>(0.f);
        sycl::atomic_fence(
            sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
      }
      // one flash step over NT tiles starting at token tb (a macro, not a
      // lambda: matrices captured by reference become generic pointers, which
      // IGC 2.38 cannot lower)
      MONO_ATTN_STEP(t0, 4);
      if (t0 + 64 < lc) MONO_ATTN_STEP(t0 + 64, 1);
    }
    // further passes (contexts past APASS1), one tile a step: a second copy of
    // the first pass's body would slow the common case
#pragma unroll 1
    for (int pb = t0 + APASS1; pb < lc; pb += APASS1) {
#pragma unroll 1
      for (int tb = pb; tb < pb + T && tb < lc; tb += 16) {
        if ((lc & 15) && (lc & ~15) == tb) {  // the partial tile's stale V rows
          half* vz = const_cast<half*>(trow(tb)) + HD + dh * 128;
          for (int r = lc & 15; r < 16; ++r)
            *reinterpret_cast<sycl::vec<half, 8>*>(
                vz + (size_t)r * KROW + lane * 8) = sycl::vec<half, 8>(0.f);
          sycl::atomic_fence(
              sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
        }
        MONO_ATTN_STEP(tb, 1);
      }
    }
    sycl::uint4 gmid[GKV_MID][8];  // gate/k/v rows of this SG, consumed after
                                   // the in-WG combine
#pragma unroll
    for (int r = 0; r < GKV_MID; ++r)
      load_row<8>(
          gmid[r],
          P.qkv_w + (size_t)grow((wg * NSG + s) * GKV_MID + r) * HID,
          lane);
    {  // SG partial -> SLM: ao[s][head i][128 dims] (normalized fp16), st[s][i]
       // = (m, l)
      float il[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float l = sycl::reduce_over_group(sg, lp[i], sycl::plus<float>());
        il[i] = l > 0.f ? 1.f / l : 0.f;
        if (lane == 0) {
          st[(s * 8 + i) * 2] = m[i];
          st[(s * 8 + i) * 2 + 1] = l;
        }
      }
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int i = 0; i < 8; ++i)
          ao[(s * 8 + i) * 128 + nt * 16 + lane] = (half)(O[nt][i] * il[i]);
    }
    wg_bar(grp);
    if (s < 2 * NQ) {  // in-WG combine: SG -> (head hh, dims half dh2); sources
                       // s' = gg*2 + dh2
      const int hh = s >> 1, dh2 = s & 1;
      float M = -INFINITY;
#pragma unroll
      for (int gg = 0; gg < 16; ++gg)
        M = sycl::fmax(M, st[((gg * 2 + dh2) * 8 + hh) * 2]);
      float Lw = 0.f, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      if (M != -INFINITY) {
#pragma unroll
        for (int gg = 0; gg < 16; ++gg) {
          const int ss2 = gg * 2 + dh2;
          const float wgt = st[(ss2 * 8 + hh) * 2 + 1] *
                            sycl::native::exp2(st[(ss2 * 8 + hh) * 2] - M);
          Lw += wgt;
          const half* src = ao + (ss2 * 8 + hh) * 128 + lane * 8;
#pragma unroll
          for (int i = 0; i < 8; ++i)
            acc[i] += wgt * (float)src[i];
        }
      }
      const float iL = Lw > 0.f ? 1.f / Lw : 0.f;
      float* dst =
          S.part_o + ((size_t)wg * NQ + hh) * HD + dh2 * 128 + lane * 8;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        dst[i] = acc[i] * iL;
      if (dh2 == 0 && lane == 0) {
        S.part_ml[(wg * NQ + hh) * 2] = M;
        S.part_ml[(wg * NQ + hh) * 2 + 1] = Lw;
      }
    }
    {
      const float sc = 256.f * P.qkv_s[0];
#pragma unroll
      for (int r = 0; r < GKV_MID; ++r) {
        const float v = dot_row_regs<8>(sg, lane, gmid[r], xp32);
        if (lane == 0) S.qkv_raw[grow((wg * NSG + s) * GKV_MID + r)] = v * sc;
      }
    }
  }
  // remaining gate / k / v rows taken dynamically by sub-groups that finish
  // early (results land at fixed rows). The ticket counter is monotonic: every
  // call consumes exactly GKV_CHUNKS + NSG*NWG tickets.
  {
    const float sc = 256.f * P.qkv_s[0];
    float* qr = S.qkv_raw;
    const uint32_t base = (call - 1) * ((uint32_t)GKV_CHUNKS + NSG * NWG);
    while (true) {
      uint32_t t = 0;
      if (lane == 0)
        t = cnt_ref(S.cnt[C_TKT * 16])
                .fetch_add(1u, sycl::memory_order::relaxed);
      const int chunk = (int)(sycl::group_broadcast(sg, t, 0) - base);
      if (chunk >= GKV_CHUNKS) break;
      const int g0 = GKV_MID * NSG * NWG + chunk * GKV_ROWS;
      gemv_pipe<1, GKV_ROWS, 8>(
          sg,
          lane,
          [&](int, int r) { return P.qkv_w + (size_t)grow(g0 + r) * HID; },
          [&](int, int) { return xp32; },
          [&](int, int r, float v) {
            if (lane == 0) qr[grow(g0 + r)] = v * sc;
          });
    }
  }
  signal(it, S.cnt + C_B * 16);
  sycl::uint4 ow[2][8];  // o_proj rows of this SG -> registers, consumed after
                         // the merge
  load_row<8>(ow[0], P.out_w + (size_t)(wg * 64 + s * 2) * (NQ * HD), lane);
  load_row<8>(ow[1], P.out_w + (size_t)(wg * 64 + s * 2 + 1) * (NQ * HD), lane);

  // ---------------- S3: merge partials + new token, gate ; WG = (head, 64
  // dims) -----------------
  wait_for(it, S.cnt + C_B * 16, NWG * call);
  {
    const int hh = (wg * 64) / HD, d0 = (wg * 64) % HD;
    float* mrg = reinterpret_cast<float*>(slm + A_AO);  // [32][64]
    {  // new token: k (norm + rope) and v; WG 0 writes them to the cache
      const int64_t slot = P.slot[0];
      half* row = slot >= 0 ? P.kv + (size_t)(slot / bs) * P.blk_stride +
                                  (size_t)(slot % bs) * KROW
                            : nullptr;
      if (s == 0)
        head_norm_rope(
            sg,
            lane,
            S.qkv_raw + NQ * 512,
            P.k_norm_w,
            rope_cs,
            kvn,
            wg == 0 ? row : nullptr);
      else if (s == 1)
        head_norm_rope(
            sg,
            lane,
            S.qkv_raw + NQ * 512 + HD,
            nullptr,
            rope_cs,
            kvn + HD,
            (wg == 0 && row) ? row + HD : nullptr);
      wg_bar(grp);
    }
    if (s == 0) {  // new-token score
      float d = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i)
        d += (float)qs[hh * HD + lane * 16 + i] * (float)kvn[lane * 16 + i];
      d = sycl::reduce_over_group(sg, d, sycl::plus<float>());
      if (lane == 0) red[96] = d * SM_SCALE;
    }
    if (lane == 0) red[s] = S.part_ml[(s * NQ + hh) * 2];
    wg_bar(grp);
    float M = red[96];
    for (int i = 0; i < 32; ++i)
      M = sycl::fmax(M, red[i]);
    const float wgt =
        S.part_ml[(s * NQ + hh) * 2 + 1] * sycl::native::exp2(red[s] - M);
    const float* src = S.part_o + ((size_t)s * NQ + hh) * HD + d0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      mrg[s * 64 + lane * 4 + i] = wgt * src[lane * 4 + i];
    if (lane == 0) red[32 + s] = wgt;
    wg_bar(grp);
    if (lid < 64) {
      const int d = d0 + lid;
      float Lw = sycl::native::exp2(red[96] - M), acc = Lw * (float)kvn[HD + d];
      for (int i = 0; i < 32; ++i) {
        acc += mrg[i * 64 + lid];
        Lw += red[32 + i];
      }
      const float o = acc / Lw;
      const float gate = S.qkv_raw[hh * 512 + 256 + d];
      S.attn_out[hh * HD + d] = (half)(o / (1.f + sycl::exp(-gate)));
    }
  }
  signal(it, S.cnt + C_C * 16);
  sycl::uint4 prt_r[4], prt_s[SHR / 4];
  load_prt(P, wg, s, lane, prt_r, prt_s);

  // ---------------- S4: o (all heads) -> SLM ; tail
  // -------------------------------------------
  wait_for(it, S.cnt + C_C * 16, NWG * call);
  for (int k = lid; k < NQ * HD; k += WGS)
    store_perm(xp, k, S.attn_out[k]);
  wg_bar(grp);
  tail(it, P, S, L, call, C_O, ow, prt_r, prt_s);
}

// ----------------------------------------------------------------------------
// launch One template instance per layer kind: each translation unit
// instantiates only the kernel it launches (an inline non-template launcher
// with per-file bodies would be an ODR violation).
template <bool ATTN>
void launch_t(sycl::queue& q, const Ptrs& P, const Scratch& S) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(
        sycl::range<1>(ATTN ? A_BYTES : G_BYTES), h);
    syclex::properties props{intelex::grf_size<256>};
    h.parallel_for(
        sycl::nd_range<1>(NWG * WGS, WGS),
        props,
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
          if constexpr (ATTN)
            attn_layer(
                it,
                slm.get_multi_ptr<sycl::access::decorated::no>().get(),
                P,
                S);
          else
            gdn_layer(
                it,
                slm.get_multi_ptr<sycl::access::decorated::no>().get(),
                P,
                S);
        });
  });
}
inline void launch(sycl::queue& q, bool attn, const Ptrs& P, const Scratch& S) {
  if (attn)
    launch_t<true>(q, P, S);
  else
    launch_t<false>(q, P, S);
}

}  // namespace mono_decode
