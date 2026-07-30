// SYCL implementation (non-ESIMD) of
// fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn
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

#include "utils.h"
#include "quantization/fp4/mxfp4_quant.h"
#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

static constexpr int HEAD_SIZE = 128;

inline void simd_load_sg32(const bf16* base, bf16* out, unsigned int lane) {
#ifdef __SYCL_DEVICE_ONLY__
  long addr0 = (long)(uintptr_t)base + (long)lane * 4;        // first 64 bf16
  long addr1 = (long)(uintptr_t)base + 128 + (long)lane * 4;  // next 64 bf16
  unsigned int r0, r1;
  asm volatile("lsc_load.ugm (M1, 32) %0:d32x1 flat[%1]:a64"
               : "=rw"(r0)
               : "rw"(addr0));
  asm volatile("lsc_load.ugm (M1, 32) %0:d32x1 flat[%1]:a64"
               : "=rw"(r1)
               : "rw"(addr1));
  const bf16* s0 = reinterpret_cast<const bf16*>(&r0);
  const bf16* s1 = reinterpret_cast<const bf16*>(&r1);
  out[0] = s0[0];
  out[1] = s0[1];
  out[2] = s1[0];
  out[3] = s1[1];
#endif
}

// Constants
static constexpr int ROPE_HEAD_DIM = 64;
static constexpr int QUANT_BLOCK = 32;
static constexpr int TOKEN_STRIDE = 64;
static constexpr int SCALE_DIM = 4;
static constexpr int HALF_ROPE = ROPE_HEAD_DIM / 2;
static constexpr int NOPE_HEAD_DIM = HEAD_SIZE - ROPE_HEAD_DIM;
static constexpr int NOPE_PAIRS = NOPE_HEAD_DIM / 2;

// ============================================================================
// FusedMxfp4SmallKernel: Specialized kernel for small compress ratios (CR <= 8)
// Key optimizations:
//   - Tuned tokens per work-group for small CR to improve EU occupancy
//   - Compile-time N_GATHER <= 16 enables aggressive unrolling
//   - Register blocking optimized for smaller gather loop
//   - Uses SG_SIZE=32 for better register utilization on small workloads
// ============================================================================
template <int COMPRESS_RATIO, int OVERLAP>
class FusedMxfp4SmallKernel {
 public:
  static_assert(COMPRESS_RATIO == 4, "FusedMxfp4SmallKernel is for CR == 4");
  static_assert(OVERLAP == 1, "FusedMxfp4SmallKernel requires OVERLAP == 1");
  static constexpr int N_GATHER = (1 + OVERLAP) * COMPRESS_RATIO;
  static constexpr int SMALL_TOKENS_PER_WG =
      8;                                    // Higher occupancy for small CR
  static constexpr int SMALL_SG_SIZE = 32;  // Subgroup size (fixed at 32)
  static constexpr int SMALL_ELEMENTS_PER_LANE = HEAD_SIZE / SMALL_SG_SIZE;
  // Derived quant constants (adapt automatically to SMALL_SG_SIZE)
  static constexpr int SMALL_PAIR_STRIDE =
      2 * SMALL_SG_SIZE;  // elements per pair
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
    // Uses SG_SIZE=32; one subgroup per token
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
        int64_t row_base = static_cast<int64_t>(blk_num) * state_cache_stride0 +
                           static_cast<int64_t>(blk_off) * state_cache_stride1 +
                           hoff;
        row_ptr = state_cache_ptr + row_base;
      }

      // Block 2D load: 128 contiguous bf16 -> 8 bf16/lane (paired layout).
      // local index a (0..7), pair p=a/2: global element = 32*p + 2*lane +
      // (a&1)
      if (valid) {
        bf16 kv_buf[SMALL_ELEMENTS_PER_LANE];
        bf16 score_buf[SMALL_ELEMENTS_PER_LANE];
        simd_load_sg32(row_ptr, kv_buf, lane);
        simd_load_sg32(row_ptr + state_width, score_buf, lane);

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
      float odd_val = normed[2 * pp + 1];

      if (pair_idx >= NOPE_PAIRS) {
        int rope_pair_idx = pair_idx - NOPE_PAIRS;
        float cos_v = cs_base[rope_pair_idx];
        float sin_v = cs_base[HALF_ROPE + rope_pair_idx];
        float r_even = even_val * cos_v - odd_val * sin_v;
        float r_odd = odd_val * cos_v + even_val * sin_v;
        normed[2 * pp] = static_cast<float>(static_cast<bf16>(r_even));
        normed[2 * pp + 1] = static_cast<float>(static_cast<bf16>(r_odd));
      } else {
        normed[2 * pp] = static_cast<float>(static_cast<bf16>(even_val));
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

    uint8_t* cache_block =
        kv_cache_ptr + static_cast<int64_t>(kv_block_idx) * kv_block_stride;
    uint8_t* val_ptr = cache_block + kv_pos_in_block * TOKEN_STRIDE;
    uint8_t* scale_ptr = cache_block +
                         static_cast<int64_t>(kv_block_size) * TOKEN_STRIDE +
                         kv_pos_in_block * SCALE_DIM;

#pragma unroll
    for (int pp = 0; pp < SMALL_ELEMENTS_PER_LANE / 2; ++pp) {
      float a = sycl::fmax(
          sycl::fabs(normed[2 * pp]), sycl::fabs(normed[2 * pp + 1]));
#pragma unroll
      for (int mask = 1; mask < SMALL_LANES_PER_QBLOCK; mask <<= 1) {
        a = sycl::fmax(a, sycl::permute_group_by_xor(sg, a, mask));
      }

      auto scale_info = vllm::mxfp4::compute_ue8m0_scale(a);
      float inv_scale = 1.0f / scale_info.scale;

      int block_id =
          SMALL_QBLOCKS_PER_PAIR * pp + lane / SMALL_LANES_PER_QBLOCK;
      if ((lane & (SMALL_LANES_PER_QBLOCK - 1)) == 0) {
        scale_ptr[block_id] = scale_info.ue8m0;
      }

      uint8_t packed =
          vllm::mxfp4::quantize_pair_to_mxfp4<vllm::mxfp4::Fp4TieBreak::ToEven>(
              normed[2 * pp], normed[2 * pp + 1], inv_scale);
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

  constexpr int SMALL_TOKENS_PER_WG =
      FusedMxfp4SmallKernel<CR, OV>::SMALL_TOKENS_PER_WG;
  constexpr int SMALL_SG_SIZE = FusedMxfp4SmallKernel<CR, OV>::SMALL_SG_SIZE;
  const size_t groups0 =
      (static_cast<size_t>(num_tokens) + SMALL_TOKENS_PER_WG - 1) /
      SMALL_TOKENS_PER_WG;
  const size_t global0 = groups0 * SMALL_TOKENS_PER_WG;

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>(
            {global0, (size_t)SMALL_SG_SIZE},
            {(size_t)SMALL_TOKENS_PER_WG, (size_t)SMALL_SG_SIZE}),
        kernel);
  });
}

void fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn_impl(
    const torch::Tensor& state_cache,
    const torch::Tensor& token_to_req_indices,
    const torch::Tensor& positions,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& block_table,
    int64_t block_size,
    int64_t state_width,
    const torch::Tensor& rms_norm_weight,
    double rms_norm_eps,
    const torch::Tensor& cos_sin_cache,
    torch::Tensor& kv_cache,
    const torch::Tensor& kv_slot_mapping,
    int64_t kv_cache_block_size,
    int64_t head_dim,
    int64_t rope_head_dim,
    int64_t compress_ratio,
    int64_t overlap,
    int64_t quant_block) {
  TORCH_CHECK(state_cache.is_xpu() && state_cache.dtype() == at::kBFloat16);
  TORCH_CHECK(
      token_to_req_indices.is_xpu() &&
      token_to_req_indices.dtype() == at::kInt);
  TORCH_CHECK(positions.is_xpu() && positions.dtype() == at::kLong);
  TORCH_CHECK(slot_mapping.is_xpu() && slot_mapping.dtype() == at::kLong);
  TORCH_CHECK(block_table.is_xpu() && block_table.dtype() == at::kInt);
  TORCH_CHECK(
      rms_norm_weight.is_xpu() && rms_norm_weight.dtype() == at::kBFloat16);
  TORCH_CHECK(cos_sin_cache.is_xpu() && cos_sin_cache.dtype() == at::kFloat);
  TORCH_CHECK(kv_cache.is_xpu() && kv_cache.dtype() == at::kByte);
  TORCH_CHECK(kv_slot_mapping.is_xpu() && kv_slot_mapping.dtype() == at::kLong);
  TORCH_CHECK(head_dim == 128 && rope_head_dim == 64 && quant_block == 32);
  // simd_load_sg32 handles arbitrary state_width

  const int num_tokens = static_cast<int>(positions.numel());
  if (num_tokens == 0) return;

  auto& queue = vllm::xpu::vllmGetQueue();

  const bf16* state_ptr =
      reinterpret_cast<const bf16*>(state_cache.data_ptr<at::BFloat16>());
  const int32_t* token_to_req = token_to_req_indices.data_ptr<int32_t>();
  const int64_t* pos_ptr = positions.data_ptr<int64_t>();
  const int64_t* slot_ptr = slot_mapping.data_ptr<int64_t>();
  const int32_t* btable = block_table.data_ptr<int32_t>();
  const bf16* rms_ptr =
      reinterpret_cast<const bf16*>(rms_norm_weight.data_ptr<at::BFloat16>());
  const float* cs_ptr = cos_sin_cache.data_ptr<float>();
  uint8_t* kv_ptr = kv_cache.data_ptr<uint8_t>();
  const int64_t* kv_slot_ptr = kv_slot_mapping.data_ptr<int64_t>();

// Macro for the CR==4 / overlap==1 indexer kernel (the only configuration used
// by the DeepSeek-V4 MXFP4 indexer path).
#define LAUNCH_SMALL_CASE(CR, OV)                    \
  launch_small_kernel<CR, OV>(                       \
      queue,                                         \
      FusedMxfp4SmallKernel<CR, OV>(                 \
          state_ptr,                                 \
          state_cache.stride(0),                     \
          state_cache.stride(1),                     \
          token_to_req,                              \
          pos_ptr,                                   \
          slot_ptr,                                  \
          btable,                                    \
          block_table.stride(0),                     \
          static_cast<int32_t>(block_size),          \
          rms_ptr,                                   \
          static_cast<float>(rms_norm_eps),          \
          cs_ptr,                                    \
          cos_sin_cache.stride(0),                   \
          kv_ptr,                                    \
          kv_slot_ptr,                               \
          static_cast<int32_t>(kv_cache_block_size), \
          kv_cache.stride(0),                        \
          state_width,                               \
          num_tokens))

  // DeepSeek-V4 only exercises the MXFP4 indexer with compress_ratio == 4 and
  // overlap == 1 (the indexer layer is created solely for the CR==4 case, and
  // DeepseekCompressor sets overlap = (compress_ratio == 4)). Only that
  // configuration is supported here.
  TORCH_CHECK(
      compress_ratio == 4 && overlap == 1,
      "fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn only supports "
      "compress_ratio == 4 and overlap == 1");
  LAUNCH_SMALL_CASE(4, 1);

#undef LAUNCH_SMALL_CASE
}

}  // anonymous namespace

void fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn(
    const torch::Tensor& state_cache,
    const torch::Tensor& token_to_req_indices,
    const torch::Tensor& positions,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& block_table,
    int64_t block_size,
    int64_t state_width,
    const torch::Tensor& rms_norm_weight,
    double rms_norm_eps,
    const torch::Tensor& cos_sin_cache,
    torch::Tensor& kv_cache,
    const torch::Tensor& kv_slot_mapping,
    int64_t kv_cache_block_size,
    int64_t head_dim,
    int64_t rope_head_dim,
    int64_t compress_ratio,
    int64_t overlap,
    int64_t quant_block) {
  fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn_impl(
      state_cache,
      token_to_req_indices,
      positions,
      slot_mapping,
      block_table,
      block_size,
      state_width,
      rms_norm_weight,
      rms_norm_eps,
      cos_sin_cache,
      kv_cache,
      kv_slot_mapping,
      kv_cache_block_size,
      head_dim,
      rope_head_dim,
      compress_ratio,
      overlap,
      quant_block);
}
