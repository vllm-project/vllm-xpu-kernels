// SYCL implementation (non-ESIMD) of fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn
//
// Mirrors the Triton kernel with FP4 quantization logic from fp4_quant.cpp
// One work-item group per token, using subgroups for quantization parallelism
//
// Hard-coded for DeepSeek-V4 indexer MXFP4 path:
//   HEAD_SIZE     = 128
//   ROPE_HEAD_DIM = 64
//   QUANT_BLOCK   = 32
//   TOKEN_STRIDE  = 64  (packed bytes / token)
//   SCALE_DIM     = 4   (ue8m0 bytes / token)

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

#ifdef __SYCL_DEVICE_ONLY__
template <int N> using uvec = unsigned int __attribute__((ext_vector_type(N)));

SYCL_EXTERNAL extern "C" int* __builtin_IB_subgroup_createBlock2DAddressPayload(
    long base, int w1, int h1, int p1, int bx, int by, int bw, int bh, int nb);
#endif

static constexpr int HEAD_SIZE = 128;
static constexpr int SG_SIZE   = 32;

// SIMD32 loader: 128 contiguous bf16 -> 4 bf16/lane (2 pairs).
inline void block2d_load_sg32(const bf16* base, bf16* out) {
#ifdef __SYCL_DEVICE_ONLY__
  uvec<2> raw;
  int* pay = __builtin_IB_subgroup_createBlock2DAddressPayload(
      (long)(uintptr_t)base, 64 - 1, 4 - 1, 64 - 1, 0, 0, 32, 4, 1);
  asm volatile("lsc_load_block2d.ugm (M1, 1)  %0:d16.1x32x4nn flat[%1+(0,0)]"
      : "=rw"(raw) : "rw.u"(pay));
  const bf16* s = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
  for (int k = 0; k < 4; ++k) out[k] = s[k];
#else
  for (int k = 0; k < 4; ++k) out[k] = base[k];  // host fallback (unused)
#endif
}

// Unified SIMD32 loader: n_pos positions' kv[128]+score[128] each.
// Output layout: [kv0[4], score0[4], kv1[4], score1[4], ...] (n_pos*8 bf16).
// When contiguous, uses single block2d load; otherwise n_pos*2 separate loads.
inline void block2d_load_kv_score_sg32(const bf16* base,
    int64_t state_width, int64_t pos_stride, int n_pos, bf16* out) {
#ifdef __SYCL_DEVICE_ONLY__
  constexpr int wb = SG_SIZE * (int)sizeof(bf16);       // 64
  constexpr int elems = HEAD_SIZE / SG_SIZE;            // 4
  bool contiguous = (state_width == HEAD_SIZE) &&
                    (n_pos == 1 || pos_stride == 2 * HEAD_SIZE);
  if (contiguous && n_pos == 2) {
    constexpr int total = (HEAD_SIZE / SG_SIZE) * 4;    // 16
    uvec<8> raw;
    int* pay = __builtin_IB_subgroup_createBlock2DAddressPayload(
        (long)(uintptr_t)base, wb - 1, total - 1, wb - 1, 0, 0, SG_SIZE, total, 1);
    asm volatile("lsc_load_block2d.ugm (M1, 1)  %0:d16.1x%2x%3nn flat[%1+(0,0)]"
        : "=rw"(raw) : "rw.u"(pay), "P"(SG_SIZE), "P"(total));
    const bf16* s = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
    for (int i = 0; i < 16; ++i) out[i] = s[i];
  } else if (contiguous && n_pos == 1) {
    constexpr int total = (HEAD_SIZE / SG_SIZE) * 2;    // 8
    uvec<4> raw;
    int* pay = __builtin_IB_subgroup_createBlock2DAddressPayload(
        (long)(uintptr_t)base, wb - 1, total - 1, wb - 1, 0, 0, SG_SIZE, total, 1);
    asm volatile("lsc_load_block2d.ugm (M1, 1)  %0:d16.1x%2x%3nn flat[%1+(0,0)]"
        : "=rw"(raw) : "rw.u"(pay), "P"(SG_SIZE), "P"(total));
    const bf16* s = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = s[i];
  } else {
    for (int p = 0; p < n_pos; ++p) {
      block2d_load_sg32(base + p * pos_stride, out + p * 2 * elems);
      block2d_load_sg32(base + p * pos_stride + state_width, out + p * 2 * elems + elems);
    }
  }
#endif
}

// SIMD16 loader: 128 contiguous bf16 -> 8 bf16/lane (4 pairs).
inline void block2d_load_sg16(const bf16* base, bf16* out) {
#ifdef __SYCL_DEVICE_ONLY__
  uvec<4> raw;
  int* pay = __builtin_IB_subgroup_createBlock2DAddressPayload(
      (long)(uintptr_t)base, 64 - 1, 4 - 1, 64 - 1, 0, 0, 32, 4, 1);
  asm volatile("lsc_load_block2d.ugm (M1, 1)  %0:d16.1x32x4nn flat[%1+(0,0)]"
      : "=rw"(raw) : "rw.u"(pay));
  const bf16* s = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
  for (int k = 0; k < 8; ++k) out[k] = s[k];
#else
  for (int k = 0; k < 8; ++k) out[k] = base[k];  // host fallback (unused)
#endif
}

inline void simd_load_sg32(const bf16* base, bf16* out, unsigned int lane) {
#ifdef __SYCL_DEVICE_ONLY__
  long addr0 = (long)(uintptr_t)base + (long)lane * 4;        // first 64 bf16
  long addr1 = (long)(uintptr_t)base + 128 + (long)lane * 4;  // next 64 bf16
  unsigned int r0, r1;
  asm volatile("lsc_load.ugm (M1, 32) %0:d32x1 flat[%1]:a64"
      : "=rw"(r0) : "rw"(addr0));
  asm volatile("lsc_load.ugm (M1, 32) %0:d32x1 flat[%1]:a64"
      : "=rw"(r1) : "rw"(addr1));
  const bf16* s0 = reinterpret_cast<const bf16*>(&r0);
  const bf16* s1 = reinterpret_cast<const bf16*>(&r1);
  out[0] = s0[0];
  out[1] = s0[1];
  out[2] = s1[0];
  out[3] = s1[1];
#endif
}

inline void simd_load_sg16(const bf16* base, bf16* out, unsigned int lane) {
#ifdef __SYCL_DEVICE_ONLY__
  // 128 bf16 / 16 lanes = 8 bf16 per lane, loaded via 4 × d32 instructions.
  // Each load covers 32 bf16 (16 lanes × 2 bf16/lane = one pair).
  unsigned int r0, r1, r2, r3;
  long addr0 = (long)(uintptr_t)base +       (long)lane * 4;  // bf16[0..31]
  long addr1 = (long)(uintptr_t)base +  64 + (long)lane * 4;  // bf16[32..63]
  long addr2 = (long)(uintptr_t)base + 128 + (long)lane * 4;  // bf16[64..95]
  long addr3 = (long)(uintptr_t)base + 192 + (long)lane * 4;  // bf16[96..127]
  asm volatile("lsc_load.ugm (M1, 16) %0:d32x1 flat[%1]:a64"
      : "=rw"(r0) : "rw"(addr0));
  asm volatile("lsc_load.ugm (M1, 16) %0:d32x1 flat[%1]:a64"
      : "=rw"(r1) : "rw"(addr1));
  asm volatile("lsc_load.ugm (M1, 16) %0:d32x1 flat[%1]:a64"
      : "=rw"(r2) : "rw"(addr2));
  asm volatile("lsc_load.ugm (M1, 16) %0:d32x1 flat[%1]:a64"
      : "=rw"(r3) : "rw"(addr3));
  const bf16* s0 = reinterpret_cast<const bf16*>(&r0);
  const bf16* s1 = reinterpret_cast<const bf16*>(&r1);
  const bf16* s2 = reinterpret_cast<const bf16*>(&r2);
  const bf16* s3 = reinterpret_cast<const bf16*>(&r3);
  out[0] = s0[0]; out[1] = s0[1];  // pair 0
  out[2] = s1[0]; out[3] = s1[1];  // pair 1
  out[4] = s2[0]; out[5] = s2[1];  // pair 2
  out[6] = s3[0]; out[7] = s3[1];  // pair 3
#endif
}


// Constants
static constexpr int ROPE_HEAD_DIM = 64;
static constexpr int QUANT_BLOCK   = 32;
static constexpr int TOKEN_STRIDE  = 64;
static constexpr int SCALE_DIM     = 4;
static constexpr int N_QUANT_BLOCKS = HEAD_SIZE / QUANT_BLOCK;  // 4
static constexpr int HALF_BLOCK    = QUANT_BLOCK / 2;  // 16
static constexpr int HALF_ROPE     = ROPE_HEAD_DIM / 2;
static constexpr int NOPE_HEAD_DIM = HEAD_SIZE - ROPE_HEAD_DIM;
static constexpr int NUM_PAIRS     = HEAD_SIZE / 2;
static constexpr int NOPE_PAIRS    = NOPE_HEAD_DIM / 2;
// Number of tokens processed per work-group. Each token is handled by exactly
// one subgroup (SG_SIZE lanes), so the work-group size is TOKENS_PER_WG * SG_SIZE.
// Packing several tokens per work-group improves EU occupancy and amortizes
// launch overhead. No group barriers are used, so subgroups (tokens) run and
// early-exit fully independently. Overridable via -DTOKENS_PER_WG=N for tuning.
// 16 was found to be the occupancy sweet spot on the target XPU (see sweep:
// it beats 8 and 32 across most token counts / compress ratios).
#ifndef TOKENS_PER_WG
#define TOKENS_PER_WG 32
#endif
static constexpr int kTokensPerWg = TOKENS_PER_WG;
static constexpr float FP4_MAX     = 6.0f;
static constexpr float INV_FP4_MAX = 1.0f / 6.0f;

// E2M1 encode: Convert float to 4-bit E2M1 format
inline uint8_t encode_fp4(float x) {
  uint8_t sign = (x < 0.0f) ? 0x8u : 0x0u;
  float a = sycl::fabs(x);
  uint8_t code = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) +
                 (a > 2.5f)  + (a >= 3.5f)  + (a > 5.0f);
  return code | sign;
}

// Fast scale calculation using bit manipulation
inline float fast_scale(float amax, uint8_t& ue8m0) {
  float ratio = amax * INV_FP4_MAX;
  uint32_t bits;
  __builtin_memcpy(&bits, &ratio, 4);
  if ((bits & 0x7F800000u) == 0) bits = 0x00800000u;
  if (bits & 0x7FFFFFu)          bits = (bits + 0x800000u) & 0xFF800000u;
  float scale;
  __builtin_memcpy(&scale, &bits, 4);
  ue8m0 = static_cast<uint8_t>((bits >> 23) & 0xFFu);
  return scale;
}

// Quantize a pair of floats to packed 4-bit
inline uint8_t quant_pair(float x0, float x1, float inv_s) {
  float q0 = sycl::fmax(-FP4_MAX, sycl::fmin(x0 * inv_s, FP4_MAX));
  float q1 = sycl::fmax(-FP4_MAX, sycl::fmin(x1 * inv_s, FP4_MAX));
  return ((encode_fp4(q1) & 0xFu) << 4) | (encode_fp4(q0) & 0xFu);
}

// ============================================================================
// FusedMxfp4SmallKernel: Specialized kernel for small compress ratios (CR <= 8)
// Key optimizations:
//   - Higher tokens per work-group (16 vs 8) for better EU occupancy
//   - Compile-time N_GATHER <= 16 enables aggressive unrolling
//   - Register blocking optimized for smaller gather loop
//   - Uses SG_SIZE=32 for better register utilization on small workloads
// ============================================================================
template <int COMPRESS_RATIO, int OVERLAP>
class FusedMxfp4SmallKernel {
 public:
  static_assert(COMPRESS_RATIO <= 8, "FusedMxfp4SmallKernel is for CR <= 8");
  static constexpr int N_GATHER = (1 + OVERLAP) * COMPRESS_RATIO;
  static constexpr int SMALL_TOKENS_PER_WG = 8;  // Higher occupancy for small CR
  static constexpr int SMALL_SG_SIZE = 32;       // Subgroup size: change to 16 or 32
  static constexpr int SMALL_ELEMENTS_PER_LANE = HEAD_SIZE / SMALL_SG_SIZE;
  // Derived quant constants (adapt automatically to SMALL_SG_SIZE)
  static constexpr int SMALL_PAIR_STRIDE = 2 * SMALL_SG_SIZE;  // elements per pair
  static constexpr int SMALL_LANES_PER_QBLOCK = QUANT_BLOCK / 2;  // 16
  static constexpr int SMALL_QBLOCKS_PER_PAIR = SMALL_PAIR_STRIDE / QUANT_BLOCK;
  int32_t num_tokens;
  
  FusedMxfp4SmallKernel(
      const bf16* state_cache,
      int64_t state_cache_s0,
      int64_t state_cache_s1,
      const int32_t* token_to_req,
      const int64_t* positions,
      const int64_t* slot_mapping,
      const int32_t* block_table,
      int64_t block_table_stride,
      int32_t block_size,
      const bf16* rms_weight,
      float rms_eps,
      const float* cos_sin_cache,
      int64_t cos_sin_stride,
      uint8_t* kv_cache,
      const int64_t* kv_slot_mapping,
      int32_t kv_block_size,
      int64_t kv_block_stride,
      int64_t state_width,
      int32_t num_tokens)
      : state_cache_ptr(state_cache),
        state_cache_stride0(state_cache_s0),
        state_cache_stride1(state_cache_s1),
        token_to_req_ptr(token_to_req),
        positions_ptr(positions),
        slot_mapping_ptr(slot_mapping),
        block_table_ptr(block_table),
        block_table_stride(block_table_stride),
        block_size(block_size),
        rms_weight_ptr(rms_weight),
        rms_eps(rms_eps),
        cos_sin_ptr(cos_sin_cache),
        cos_sin_stride(cos_sin_stride),
        kv_cache_ptr(kv_cache),
        kv_slot_ptr(kv_slot_mapping),
        kv_block_size(kv_block_size),
        kv_block_stride(kv_block_stride),
        state_width(state_width),
        num_tokens(num_tokens) {}

  [[sycl::reqd_sub_group_size(SMALL_SG_SIZE)]]
  void operator()(sycl::nd_item<2> item) const {
    // ============================================================================
    // Optimized for small CR (4 or 8): N_GATHER is at most 16
    // Uses SG_SIZE=32, same data layout as FusedMxfp4Kernel
    // ============================================================================
    
    const int wg_id = item.get_group(0);
    const int sg_id = item.get_sub_group().get_group_id()[0];
    const int lane = item.get_local_id(1) % SMALL_SG_SIZE;
    const int wg_token_start = wg_id * SMALL_TOKENS_PER_WG;

    auto sg = item.get_sub_group();

    const int my_token_global = wg_token_start + sg_id;
    
    if (my_token_global >= num_tokens) return;
    
    // Load token metadata
    int slot_id = 0;
    int position = 0;
    int req_idx = 0;
    
    if (lane == 0) {
      slot_id = slot_mapping_ptr[my_token_global];
      position = positions_ptr[my_token_global];
      req_idx = token_to_req_ptr[my_token_global];
    }
    
    slot_id = sycl::group_broadcast(sg, slot_id, 0);
    position = sycl::group_broadcast(sg, position, 0);
    req_idx = sycl::group_broadcast(sg, req_idx, 0);
    
    if (slot_id < 0) return;
    
    const int start = position - N_GATHER + 1;
    const int64_t bt_row = static_cast<int64_t>(req_idx) * block_table_stride;
    
    // ============================================================================
    // PHASE 1: Online Softmax Compression (optimized for small N_GATHER)
    // ============================================================================
    
    constexpr float NEG_LARGE = -std::numeric_limits<float>::infinity();
    float m_run[SMALL_ELEMENTS_PER_LANE];
    float s_run[SMALL_ELEMENTS_PER_LANE];
    float acc[SMALL_ELEMENTS_PER_LANE];
    
#pragma unroll
    for (int i = 0; i < SMALL_ELEMENTS_PER_LANE; ++i) {
      m_run[i] = NEG_LARGE;
      s_run[i] = 0.0f;
      acc[i] = 0.0f;
    }

    // Small CR: N_GATHER <= 16, fully unrolled
#pragma unroll
    for (int r = 0; r < N_GATHER; ++r) {
      int p = start + r;
      bool valid = (p >= 0);
      
      int blk_idx = 0, blk_num = 0, blk_off = 0;
      int hoff = (r >= COMPRESS_RATIO) ? HEAD_SIZE : 0;
      const bf16* row_ptr = nullptr;
      
      if (valid) {
        blk_idx = p / block_size;
        blk_num = block_table_ptr[bt_row + blk_idx];
        blk_off = p % block_size;
        int64_t row_base = static_cast<int64_t>(blk_num) * state_cache_stride0
                         + static_cast<int64_t>(blk_off) * state_cache_stride1
                         + hoff;
        row_ptr = state_cache_ptr + row_base;
      }
      
      // Block 2D load: 128 contiguous bf16 -> 8 bf16/lane (paired layout).
      // local index a (0..7), pair p=a/2: global element = 32*p + 2*lane + (a&1)
      if (valid) {
        bf16 kv_buf[SMALL_ELEMENTS_PER_LANE];
        bf16 score_buf[SMALL_ELEMENTS_PER_LANE];
        if constexpr (SMALL_SG_SIZE == 32) {
          simd_load_sg32(row_ptr, kv_buf, lane);
          simd_load_sg32(row_ptr + state_width, score_buf, lane);
        } else {
          simd_load_sg16(row_ptr, kv_buf, lane);
          simd_load_sg16(row_ptr + state_width, score_buf, lane);
        }
        
#pragma unroll
        for (int i = 0; i < SMALL_ELEMENTS_PER_LANE; ++i) {
          float kv = static_cast<float>(kv_buf[i]);
          float score = static_cast<float>(score_buf[i]);
          
          // Online softmax update
          const float new_m = sycl::fmax(m_run[i], score);
          const float alpha = sycl::exp(m_run[i] - new_m);
          const float p_exp = sycl::exp(score - new_m);
          s_run[i] = s_run[i] * alpha + p_exp;
          acc[i] = acc[i] * alpha + p_exp * kv;
          m_run[i] = new_m;
        }
      }
    }
  
    // Final compressed values
    float compressed[SMALL_ELEMENTS_PER_LANE];
#pragma unroll
    for (int i = 0; i < SMALL_ELEMENTS_PER_LANE; ++i) {
      compressed[i] = acc[i] / s_run[i];
    }
  
    // ============================================================================
    // PHASE 2: RMSNorm
    // ============================================================================
    
    float my_var = 0.0f;
#pragma unroll
    for (int i = 0; i < SMALL_ELEMENTS_PER_LANE; ++i) {
      my_var += compressed[i] * compressed[i];
    }
    
    float total_var = sycl::reduce_over_group(sg, my_var, sycl::plus<float>());
    total_var /= HEAD_SIZE;
    float rrms = sycl::rsqrt(total_var + rms_eps);
    
    float normed[SMALL_ELEMENTS_PER_LANE];
#pragma unroll
    for (int i = 0; i < SMALL_ELEMENTS_PER_LANE; ++i) {
      int elem_idx = SMALL_PAIR_STRIDE * (i / 2) + 2 * lane + (i & 1);
      float w = static_cast<float>(rms_weight_ptr[elem_idx]);
      normed[i] = compressed[i] * rrms * w;
    }
    
    // ============================================================================
    // PHASE 3: GPT-J RoPE (paired layout: each pair is lane-local, no shuffle)
    // ============================================================================
    
    int compressed_pos = (position / COMPRESS_RATIO) * COMPRESS_RATIO;
    const float* cs_base = cos_sin_ptr + compressed_pos * cos_sin_stride;
    
#pragma unroll
    for (int pp = 0; pp < SMALL_ELEMENTS_PER_LANE / 2; ++pp) {
      int even_elem = SMALL_PAIR_STRIDE * pp + 2 * lane;
      int pair_idx = even_elem / 2;  // = SMALL_SG_SIZE*pp + lane
      float even_val = normed[2 * pp];
      float odd_val  = normed[2 * pp + 1];
      
      if (pair_idx >= NOPE_PAIRS) {
        int rope_pair_idx = pair_idx - NOPE_PAIRS;
        float cos_v = cs_base[rope_pair_idx];
        float sin_v = cs_base[HALF_ROPE + rope_pair_idx];
        float r_even = even_val * cos_v - odd_val * sin_v;
        float r_odd  = odd_val  * cos_v + even_val * sin_v;
        normed[2 * pp]     = static_cast<float>(static_cast<bf16>(r_even));
        normed[2 * pp + 1] = static_cast<float>(static_cast<bf16>(r_odd));
      } else {
        normed[2 * pp]     = static_cast<float>(static_cast<bf16>(even_val));
        normed[2 * pp + 1] = static_cast<float>(static_cast<bf16>(odd_val));
      }
    }
    
    // ============================================================================
    // PHASE 4: FP4 Quantization and KV Cache Write
    // SG_SIZE=32: each pair pp spans 64 elements (32 lanes × 2), containing
    // 2 QUANT_BLOCKs. Use XOR butterfly within 16-lane halves for amax.
    // ============================================================================
    
    int kv_slot_idx = 0;
    if (lane == 0) {
      kv_slot_idx = kv_slot_ptr[my_token_global];
    }
    kv_slot_idx = sycl::group_broadcast(sg, kv_slot_idx, 0);
    
    if (kv_slot_idx < 0) return;
    
    const int kv_block_idx = kv_slot_idx / kv_block_size;
    const int kv_pos_in_block = kv_slot_idx % kv_block_size;
    
    uint8_t* cache_block = kv_cache_ptr + static_cast<int64_t>(kv_block_idx) * kv_block_stride;
    uint8_t* val_ptr = cache_block + kv_pos_in_block * TOKEN_STRIDE;
    uint8_t* scale_ptr = cache_block + static_cast<int64_t>(kv_block_size) * TOKEN_STRIDE
                       + kv_pos_in_block * SCALE_DIM;
    
#pragma unroll
    for (int pp = 0; pp < SMALL_ELEMENTS_PER_LANE / 2; ++pp) {
      float a = sycl::fmax(sycl::fabs(normed[2 * pp]), sycl::fabs(normed[2 * pp + 1]));
#pragma unroll
      for (int mask = 1; mask < SMALL_LANES_PER_QBLOCK; mask <<= 1) {
        a = sycl::fmax(a, sycl::permute_group_by_xor(sg, a, mask));
      }
      
      uint8_t ue8m0;
      float inv_scale = 1.0f / fast_scale(a, ue8m0);
      
      int block_id = SMALL_QBLOCKS_PER_PAIR * pp + lane / SMALL_LANES_PER_QBLOCK;
      if ((lane & (SMALL_LANES_PER_QBLOCK - 1)) == 0) {
        scale_ptr[block_id] = ue8m0;
      }
      
      uint8_t packed = quant_pair(normed[2 * pp], normed[2 * pp + 1], inv_scale);
      val_ptr[SMALL_SG_SIZE * pp + lane] = packed;
    }
  }

 private:
  const bf16* state_cache_ptr;
  int64_t state_cache_stride0;
  int64_t state_cache_stride1;
  const int32_t* token_to_req_ptr;
  const int64_t* positions_ptr;
  const int64_t* slot_mapping_ptr;
  const int32_t* block_table_ptr;
  int64_t block_table_stride;
  int32_t block_size;
  const bf16* rms_weight_ptr;
  float rms_eps;
  const float* cos_sin_ptr;
  int64_t cos_sin_stride;
  uint8_t* kv_cache_ptr;
  const int64_t* kv_slot_ptr;
  int32_t kv_block_size;
  int64_t kv_block_stride;
  int64_t state_width;
};

// Launch function for small kernel
template <int CR, int OV>
void launch_small_kernel(sycl::queue& q, FusedMxfp4SmallKernel<CR, OV> kernel) {
  const int num_tokens = kernel.num_tokens;
  if (num_tokens == 0) return;

  constexpr int SMALL_TOKENS_PER_WG = FusedMxfp4SmallKernel<CR, OV>::SMALL_TOKENS_PER_WG;
  constexpr int SMALL_SG_SIZE = FusedMxfp4SmallKernel<CR, OV>::SMALL_SG_SIZE;
  const size_t groups0 =
      (static_cast<size_t>(num_tokens) + SMALL_TOKENS_PER_WG - 1) / SMALL_TOKENS_PER_WG;
  const size_t global0 = groups0 * SMALL_TOKENS_PER_WG;

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>({global0, (size_t)SMALL_SG_SIZE},
                          {(size_t)SMALL_TOKENS_PER_WG, (size_t)SMALL_SG_SIZE}),
        kernel);
  }).wait();
}

template <int COMPRESS_RATIO, int OVERLAP>
class FusedMxfp4Kernel {
 public:
  static constexpr int N_GATHER = (1 + OVERLAP) * COMPRESS_RATIO;
  int32_t num_tokens;  // Made public for launch_kernel access
  
  FusedMxfp4Kernel(
      const bf16* state_cache,
      int64_t state_cache_s0,
      int64_t state_cache_s1,
      const int32_t* token_to_req,
      const int64_t* positions,
      const int64_t* slot_mapping,
      const int32_t* block_table,
      int64_t block_table_stride,
      int32_t block_size,
      const bf16* rms_weight,
      float rms_eps,
      const float* cos_sin_cache,
      int64_t cos_sin_stride,
      uint8_t* kv_cache,
      const int64_t* kv_slot_mapping,
      int32_t kv_block_size,
      int64_t kv_block_stride,
      int64_t state_width,
      int32_t num_tokens)
      : state_cache_ptr(state_cache),
        state_cache_stride0(state_cache_s0),
        state_cache_stride1(state_cache_s1),
        token_to_req_ptr(token_to_req),
        positions_ptr(positions),
        slot_mapping_ptr(slot_mapping),
        block_table_ptr(block_table),
        block_table_stride(block_table_stride),
        block_size(block_size),
        rms_weight_ptr(rms_weight),
        rms_eps(rms_eps),
        cos_sin_ptr(cos_sin_cache),
        cos_sin_stride(cos_sin_stride),
        kv_cache_ptr(kv_cache),
        kv_slot_ptr(kv_slot_mapping),
        kv_block_size(kv_block_size),
        kv_block_stride(kv_block_stride),
        state_width(state_width),
        num_tokens(num_tokens) {}

  [[sycl::reqd_sub_group_size(SG_SIZE)]]
  void operator()(sycl::nd_item<2> item) const {
    // ============================================================================
    // Single-subgroup-per-token design: no SLM, no group barrier
    // Each subgroup (32 lanes) handles one token end-to-end:
    //   Online Softmax Compression → RMSNorm → RoPE → FP4 Quantization
    // Each lane processes HEAD_SIZE/SG_SIZE = 128/32 = 4 elements
    // ============================================================================
    
    const int wg_id = item.get_group(0);
    const int sg_id = item.get_sub_group().get_group_id()[0];
    const int lane = item.get_local_id(1) % SG_SIZE;
    const int wg_token_start = wg_id * kTokensPerWg;

    auto sg = item.get_sub_group();

    // Each lane handles 4 elements (HEAD_SIZE=128, SG_SIZE=32)
    constexpr int ELEMENTS_PER_LANE = HEAD_SIZE / SG_SIZE;  // 4

    // This subgroup processes one token
    const int my_token_global = wg_token_start + sg_id;
    
    // Early exit for out-of-bounds tokens
    if (my_token_global >= num_tokens) return;
    
    // Load token metadata (lane 0 loads, then broadcast)
    int slot_id = 0;
    int position = 0;
    int req_idx = 0;
    
    if (lane == 0) {
      slot_id = slot_mapping_ptr[my_token_global];
      position = positions_ptr[my_token_global];
      req_idx = token_to_req_ptr[my_token_global];
    }
    
    slot_id = sycl::group_broadcast(sg, slot_id, 0);
    position = sycl::group_broadcast(sg, position, 0);
    req_idx = sycl::group_broadcast(sg, req_idx, 0);
    
    // Skip invalid slots
    if (slot_id < 0) return;
    
    const int start = position - N_GATHER + 1;
    const int64_t bt_row = static_cast<int64_t>(req_idx) * block_table_stride;
    
    // ============================================================================
    // PHASE 1: Online Softmax Compression
    // Each lane processes 4 elements, maintaining running max/sum/accumulator
    // Batched by 2: when consecutive positions are in the same page block,
    // load 2 positions with a single d16.1x32x16nn instruction (16 rows).
    // ============================================================================
    
    constexpr float NEG_LARGE = -std::numeric_limits<float>::infinity();
    float m_run[ELEMENTS_PER_LANE];
    float s_run[ELEMENTS_PER_LANE];
    float acc[ELEMENTS_PER_LANE];
    
#pragma unroll
    for (int i = 0; i < ELEMENTS_PER_LANE; ++i) {
      m_run[i] = NEG_LARGE;
      s_run[i] = 0.0f;
      acc[i] = 0.0f;
    }

    int r = 0;
    // Batched loop: process 2 positions per iteration when possible
    for (; r + 1 < N_GATHER; r += 2) {
      int p0 = start + r;
      int p1 = start + r + 1;
      bool valid0 = (p0 >= 0);
      bool valid1 = (p1 >= 0);
      
      // hoff is always 0 for FusedMxfp4Kernel (overlap=0, r < CR always)
      int hoff0 = (r >= COMPRESS_RATIO) ? HEAD_SIZE : 0;
      int hoff1 = (r + 1 >= COMPRESS_RATIO) ? HEAD_SIZE : 0;
      
      // Check if both positions are in the same page block and contiguous
      int blk_idx0 = 0, blk_idx1 = 0;
      int blk_num0 = 0, blk_num1 = 0;
      int blk_off0 = 0, blk_off1 = 0;
      const bf16* row_ptr0 = nullptr;
      const bf16* row_ptr1 = nullptr;
      
      if (valid0) {
        blk_idx0 = p0 / block_size;
        blk_num0 = block_table_ptr[bt_row + blk_idx0];
        blk_off0 = p0 % block_size;
        int64_t row_base0 = static_cast<int64_t>(blk_num0) * state_cache_stride0
                          + static_cast<int64_t>(blk_off0) * state_cache_stride1
                          + hoff0;
        row_ptr0 = state_cache_ptr + row_base0;
      }
      if (valid1) {
        blk_idx1 = p1 / block_size;
        blk_num1 = block_table_ptr[bt_row + blk_idx1];
        blk_off1 = p1 % block_size;
        int64_t row_base1 = static_cast<int64_t>(blk_num1) * state_cache_stride0
                          + static_cast<int64_t>(blk_off1) * state_cache_stride1
                          + hoff1;
        row_ptr1 = state_cache_ptr + row_base1;
      }
      
      // Batched path: both valid, same page block, consecutive offsets, same hoff
      bool can_batch = valid0 && valid1 && (blk_idx0 == blk_idx1)
                     && (blk_off1 == blk_off0 + 1) && (hoff0 == hoff1);
      
      if (can_batch) {
        // Load 2 positions with single d16.1x32x16nn (512 bf16 contiguous)
        bf16 buf2[ELEMENTS_PER_LANE * 4];
        block2d_load_kv_score_sg32(row_ptr0, state_width, state_cache_stride1, 2, buf2);
        bf16* kv1_buf = buf2;
        bf16* score1_buf = buf2 + ELEMENTS_PER_LANE;
        bf16* kv2_buf = buf2 + 2 * ELEMENTS_PER_LANE;
        bf16* score2_buf = buf2 + 3 * ELEMENTS_PER_LANE;
        
        // Online softmax for position 1
#pragma unroll
        for (int i = 0; i < ELEMENTS_PER_LANE; ++i) {
          float kv = static_cast<float>(kv1_buf[i]);
          float score = static_cast<float>(score1_buf[i]);
          const float new_m = sycl::fmax(m_run[i], score);
          const float alpha = sycl::exp(m_run[i] - new_m);
          const float p_exp = sycl::exp(score - new_m);
          s_run[i] = s_run[i] * alpha + p_exp;
          acc[i] = acc[i] * alpha + p_exp * kv;
          m_run[i] = new_m;
        }
        // Online softmax for position 2
#pragma unroll
        for (int i = 0; i < ELEMENTS_PER_LANE; ++i) {
          float kv = static_cast<float>(kv2_buf[i]);
          float score = static_cast<float>(score2_buf[i]);
          const float new_m = sycl::fmax(m_run[i], score);
          const float alpha = sycl::exp(m_run[i] - new_m);
          const float p_exp = sycl::exp(score - new_m);
          s_run[i] = s_run[i] * alpha + p_exp;
          acc[i] = acc[i] * alpha + p_exp * kv;
          m_run[i] = new_m;
        }
      } else {
        // Fallback: load and process individually
        if (valid0) {
          bf16 buf0[ELEMENTS_PER_LANE * 2];
          block2d_load_kv_score_sg32(row_ptr0, state_width, 0, 1, buf0);
#pragma unroll
          for (int i = 0; i < ELEMENTS_PER_LANE; ++i) {
            float kv = static_cast<float>(buf0[i]);
            float score = static_cast<float>(buf0[ELEMENTS_PER_LANE + i]);
            const float new_m = sycl::fmax(m_run[i], score);
            const float alpha = sycl::exp(m_run[i] - new_m);
            const float p_exp = sycl::exp(score - new_m);
            s_run[i] = s_run[i] * alpha + p_exp;
            acc[i] = acc[i] * alpha + p_exp * kv;
            m_run[i] = new_m;
          }
        }
        if (valid1) {
          bf16 buf1[ELEMENTS_PER_LANE * 2];
          block2d_load_kv_score_sg32(row_ptr1, state_width, 0, 1, buf1);
#pragma unroll
          for (int i = 0; i < ELEMENTS_PER_LANE; ++i) {
            float kv = static_cast<float>(buf1[i]);
            float score = static_cast<float>(buf1[ELEMENTS_PER_LANE + i]);
            const float new_m = sycl::fmax(m_run[i], score);
            const float alpha = sycl::exp(m_run[i] - new_m);
            const float p_exp = sycl::exp(score - new_m);
            s_run[i] = s_run[i] * alpha + p_exp;
            acc[i] = acc[i] * alpha + p_exp * kv;
            m_run[i] = new_m;
          }
        }
      }
    }
    
    // Handle remaining odd iteration (if N_GATHER is odd)
    if (r < N_GATHER) {
      int p = start + r;
      bool valid = (p >= 0);
      if (valid) {
        int blk_idx = p / block_size;
        int blk_num = block_table_ptr[bt_row + blk_idx];
        int blk_off = p % block_size;
        int hoff = (r >= COMPRESS_RATIO) ? HEAD_SIZE : 0;
        int64_t row_base = static_cast<int64_t>(blk_num) * state_cache_stride0
                         + static_cast<int64_t>(blk_off) * state_cache_stride1
                         + hoff;
        const bf16* row_ptr = state_cache_ptr + row_base;
        
        bf16 buf[ELEMENTS_PER_LANE * 2];
        block2d_load_kv_score_sg32(row_ptr, state_width, 0, 1, buf);
#pragma unroll
        for (int i = 0; i < ELEMENTS_PER_LANE; ++i) {
          float kv = static_cast<float>(buf[i]);
          float score = static_cast<float>(buf[ELEMENTS_PER_LANE + i]);
          const float new_m = sycl::fmax(m_run[i], score);
          const float alpha = sycl::exp(m_run[i] - new_m);
          const float p_exp = sycl::exp(score - new_m);
          s_run[i] = s_run[i] * alpha + p_exp;
          acc[i] = acc[i] * alpha + p_exp * kv;
          m_run[i] = new_m;
        }
      }
    }
    
#pragma unroll
    for (int i = 0; i < ELEMENTS_PER_LANE; ++i) {
      m_run[i] = acc[i] / s_run[i];
    }
    
    // ============================================================================
    // PHASE 2: RMSNorm
    // Reduce variance across all 128 elements (32 lanes × 4 elements)
    // ============================================================================
    
    float my_var = 0.0f;
#pragma unroll
    for (int i = 0; i < ELEMENTS_PER_LANE; ++i) {
      my_var += m_run[i] * m_run[i];
    }
    
    float total_var = sycl::reduce_over_group(sg, my_var, sycl::plus<float>());
    total_var /= HEAD_SIZE;
    float rrms = sycl::rsqrt(total_var + rms_eps);
    
#pragma unroll
    for (int i = 0; i < ELEMENTS_PER_LANE; ++i) {
      // paired layout: global element index = 64*(i/2) + 2*lane + (i&1)
      int elem_idx = 64 * (i / 2) + 2 * lane + (i & 1);
      float w = static_cast<float>(rms_weight_ptr[elem_idx]);
      m_run[i] = m_run[i] * rrms * w;
    }
    
    // ============================================================================
    // PHASE 3: GPT-J RoPE (paired layout: each pair is lane-local, no shuffle)
    // pair pp=0 -> elements 0..63 (nope), pair pp=1 -> elements 64..127 (rope)
    // ============================================================================
    
    int compressed_pos = (position / COMPRESS_RATIO) * COMPRESS_RATIO;
    const float* cs_base = cos_sin_ptr + compressed_pos * cos_sin_stride;
    
#pragma unroll
    for (int pp = 0; pp < ELEMENTS_PER_LANE / 2; ++pp) {
      int even_elem = 64 * pp + 2 * lane;  // even global index
      int pair_idx = even_elem / 2;        // = 32*pp + lane
      float even_val = m_run[2 * pp];
      float odd_val  = m_run[2 * pp + 1];
      
      if (pair_idx >= NOPE_PAIRS) {
        int rope_pair_idx = pair_idx - NOPE_PAIRS;
        float cos_v = cs_base[rope_pair_idx];
        float sin_v = cs_base[HALF_ROPE + rope_pair_idx];
        float r_even = even_val * cos_v - odd_val * sin_v;
        float r_odd  = odd_val  * cos_v + even_val * sin_v;
        m_run[2 * pp]     = static_cast<float>(static_cast<bf16>(r_even));
        m_run[2 * pp + 1] = static_cast<float>(static_cast<bf16>(r_odd));
      } else {
        m_run[2 * pp]     = static_cast<float>(static_cast<bf16>(even_val));
        m_run[2 * pp + 1] = static_cast<float>(static_cast<bf16>(odd_val));
      }
    }
    
    // ============================================================================
    // PHASE 4: FP4 Quantization and KV Cache Write
    // Paired layout: a QUANT_BLOCK (32 elems) = one pair pp over 16 lanes (half
    // subgroup). amax is reduced within each 16-lane half via XOR butterfly
    // (masks 1,2,4,8 stay inside the 16-lane group, bit-4 separates the halves).
    // ============================================================================
    
    int kv_slot_idx = 0;
    if (lane == 0) {
      kv_slot_idx = kv_slot_ptr[my_token_global];
    }
    kv_slot_idx = sycl::group_broadcast(sg, kv_slot_idx, 0);
    
    if (kv_slot_idx < 0) return;
    
    const int kv_block_idx = kv_slot_idx / kv_block_size;
    const int kv_pos_in_block = kv_slot_idx % kv_block_size;
    
    uint8_t* cache_block = kv_cache_ptr + static_cast<int64_t>(kv_block_idx) * kv_block_stride;
    uint8_t* val_ptr = cache_block + kv_pos_in_block * TOKEN_STRIDE;
    uint8_t* scale_ptr = cache_block + static_cast<int64_t>(kv_block_size) * TOKEN_STRIDE
                       + kv_pos_in_block * SCALE_DIM;
    
#pragma unroll
    for (int pp = 0; pp < ELEMENTS_PER_LANE / 2; ++pp) {
      // amax over this pair within the 16-lane half (QUANT_BLOCK = 32 elements)
      float a = sycl::fmax(sycl::fabs(m_run[2 * pp]), sycl::fabs(m_run[2 * pp + 1]));
#pragma unroll
      for (int mask = 1; mask < SG_SIZE / 2; mask <<= 1) {
        a = sycl::fmax(a, sycl::permute_group_by_xor(sg, a, mask));
      }
      
      uint8_t ue8m0;
      float inv_scale = 1.0f / fast_scale(a, ue8m0);
      
      // block id: pair pp, lower half (lanes 0..15) -> 2*pp, upper half -> 2*pp+1
      int block_id = 2 * pp + (lane >> 4);
      // one lane per half writes the scale (lane 0 for lower, lane 16 for upper)
      if ((lane & (SG_SIZE / 2 - 1)) == 0) {
        scale_ptr[block_id] = ue8m0;
      }
      
      // pack lane-local pair; output byte index = even_elem/2 = 32*pp + lane
      uint8_t packed = quant_pair(m_run[2 * pp], m_run[2 * pp + 1], inv_scale);
      val_ptr[32 * pp + lane] = packed;
    }
  }

 private:
  const bf16* state_cache_ptr;
  int64_t state_cache_stride0;
  int64_t state_cache_stride1;
  const int32_t* token_to_req_ptr;
  const int64_t* positions_ptr;
  const int64_t* slot_mapping_ptr;
  const int32_t* block_table_ptr;
  int64_t block_table_stride;
  int32_t block_size;
  const bf16* rms_weight_ptr;
  float rms_eps;
  const float* cos_sin_ptr;
  int64_t cos_sin_stride;
  uint8_t* kv_cache_ptr;
  const int64_t* kv_slot_ptr;
  int32_t kv_block_size;
  int64_t kv_block_stride;
  int64_t state_width;
};

template <int CR, int OV>
void launch_kernel(sycl::queue& q, FusedMxfp4Kernel<CR, OV> kernel) {
  const int num_tokens = kernel.num_tokens;
  if (num_tokens == 0) return;

  // Each token is processed by one subgroup (SG_SIZE lanes); pack TOKENS_PER_WG
  // tokens into a single work-group to raise occupancy. dim0 is rounded up to a
  // multiple of TOKENS_PER_WG (extra subgroups early-exit via the bounds check).
  const size_t groups0 =
      (static_cast<size_t>(num_tokens) + TOKENS_PER_WG - 1) / TOKENS_PER_WG;
  const size_t global0 = groups0 * TOKENS_PER_WG;

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>({global0, (size_t)SG_SIZE},
                          {(size_t)TOKENS_PER_WG, (size_t)SG_SIZE}),
        kernel);
  }).wait();
}

void fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn_impl(
    torch::Tensor state_cache,
    torch::Tensor token_to_req_indices,
    torch::Tensor positions,
    torch::Tensor slot_mapping,
    torch::Tensor block_table,
    int64_t block_size,
    int64_t state_width,
    torch::Tensor rms_norm_weight,
    double rms_norm_eps,
    torch::Tensor cos_sin_cache,
    torch::Tensor kv_cache,
    torch::Tensor kv_slot_mapping,
    int64_t kv_cache_block_size,
    int64_t head_dim,
    int64_t rope_head_dim,
    int64_t compress_ratio,
    int64_t overlap,
    int64_t quant_block) {
  
  TORCH_CHECK(state_cache.is_xpu() && state_cache.dtype() == at::kBFloat16);
  TORCH_CHECK(rms_norm_weight.is_xpu() && rms_norm_weight.dtype() == at::kBFloat16);
  TORCH_CHECK(cos_sin_cache.is_xpu() && cos_sin_cache.dtype() == at::kFloat);
  TORCH_CHECK(kv_cache.is_xpu() && kv_cache.dtype() == at::kByte);
  TORCH_CHECK(head_dim == 128 && rope_head_dim == 64 && quant_block == 32);
  // block2d loaders now handle arbitrary state_width via fallback path

  const int num_tokens = static_cast<int>(positions.numel());
  if (num_tokens == 0) return;

  auto& queue = c10::xpu::getCurrentXPUStream().queue();

  const bf16* state_ptr = reinterpret_cast<const bf16*>(state_cache.const_data_ptr<at::BFloat16>());
  const int32_t* token_to_req = token_to_req_indices.const_data_ptr<int32_t>();
  const int64_t* pos_ptr = positions.const_data_ptr<int64_t>();
  const int64_t* slot_ptr = slot_mapping.const_data_ptr<int64_t>();
  const int32_t* btable = block_table.const_data_ptr<int32_t>();
  const bf16* rms_ptr = reinterpret_cast<const bf16*>(rms_norm_weight.const_data_ptr<at::BFloat16>());
  const float* cs_ptr = cos_sin_cache.const_data_ptr<float>();
  uint8_t* kv_ptr = kv_cache.data_ptr<uint8_t>();
  const int64_t* kv_slot_ptr = kv_slot_mapping.const_data_ptr<int64_t>();

// Macro for small CR kernel (CR <= 8)
#define LAUNCH_SMALL_CASE(CR, OV)                                         \
  launch_small_kernel<CR, OV>(queue, FusedMxfp4SmallKernel<CR, OV>(        \
      state_ptr, state_cache.stride(0), state_cache.stride(1),           \
      token_to_req, pos_ptr, slot_ptr, btable, block_table.stride(0),    \
      static_cast<int32_t>(block_size), rms_ptr,                         \
      static_cast<float>(rms_norm_eps), cs_ptr, cos_sin_cache.stride(0), \
      kv_ptr, kv_slot_ptr, static_cast<int32_t>(kv_cache_block_size),    \
      kv_cache.stride(0), state_width, num_tokens))

// Macro for large CR kernel (CR > 8)
#define LAUNCH_CASE(CR, OV)                                             \
  launch_kernel<CR, OV>(queue, FusedMxfp4Kernel<CR, OV>(                \
      state_ptr, state_cache.stride(0), state_cache.stride(1),         \
      token_to_req, pos_ptr, slot_ptr, btable, block_table.stride(0),  \
      static_cast<int32_t>(block_size), rms_ptr,                       \
      static_cast<float>(rms_norm_eps), cs_ptr, cos_sin_cache.stride(0), \
      kv_ptr, kv_slot_ptr, static_cast<int32_t>(kv_cache_block_size),  \
      kv_cache.stride(0), state_width, num_tokens))

  // Use FusedMxfp4SmallKernel for CR <= 8 (optimized for small gather loops)
  if (compress_ratio == 4 && overlap == 0) {
    LAUNCH_SMALL_CASE(4, 0);
  } else if (compress_ratio == 4 && overlap == 1) {
    LAUNCH_SMALL_CASE(4, 1);
  } else if (compress_ratio == 8 && overlap == 0) {
    LAUNCH_SMALL_CASE(8, 0);
  } else if (compress_ratio == 8 && overlap == 1) {
    LAUNCH_SMALL_CASE(8, 1);
  } else if (compress_ratio == 16 && overlap == 0) {
    LAUNCH_CASE(16, 0);
  } else if (compress_ratio == 16 && overlap == 1) {
    LAUNCH_CASE(16, 1);
  } else if (compress_ratio == 32 && overlap == 0) {
    LAUNCH_CASE(32, 0);
  } else if (compress_ratio == 32 && overlap == 1) {
    LAUNCH_CASE(32, 1);
  } else if (compress_ratio == 64 && overlap == 0) {
    LAUNCH_CASE(64, 0);
  } else if (compress_ratio == 64 && overlap == 1) {
    LAUNCH_CASE(64, 1);
  } else if (compress_ratio == 128 && overlap == 0) {
    LAUNCH_CASE(128, 0);
  } else if (compress_ratio == 128 && overlap == 1) {
    LAUNCH_CASE(128, 1);
  } else {
    TORCH_CHECK(false, "Unsupported compress_ratio/overlap combination");
  }

#undef LAUNCH_SMALL_CASE
#undef LAUNCH_CASE
}

}  // anonymous namespace

void fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn(
    torch::Tensor state_cache,
    torch::Tensor token_to_req_indices,
    torch::Tensor positions,
    torch::Tensor slot_mapping,
    torch::Tensor block_table,
    int64_t block_size,
    int64_t state_width,
    torch::Tensor rms_norm_weight,
    double rms_norm_eps,
    torch::Tensor cos_sin_cache,
    torch::Tensor kv_cache,
    torch::Tensor kv_slot_mapping,
    int64_t kv_cache_block_size,
    int64_t head_dim,
    int64_t rope_head_dim,
    int64_t compress_ratio,
    int64_t overlap,
    int64_t quant_block) {
  fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn_impl(
      state_cache, token_to_req_indices, positions, slot_mapping, block_table,
      block_size, state_width, rms_norm_weight, rms_norm_eps, cos_sin_cache,
      kv_cache, kv_slot_mapping, kv_cache_block_size, head_dim, rope_head_dim,
      compress_ratio, overlap, quant_block);
}
