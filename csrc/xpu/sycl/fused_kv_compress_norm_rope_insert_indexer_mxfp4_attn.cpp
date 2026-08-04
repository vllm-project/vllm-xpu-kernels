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

// File-scope model/layout constants for this fixed DeepSeek-V4 path.
static constexpr int HEAD_SIZE = 128;
static constexpr int ROPE_HEAD_DIM = 64;
static constexpr int QUANT_BLOCK = 32;
static constexpr int TOKEN_STRIDE = 64;
static constexpr int SCALE_DIM = 4;
static constexpr int HALF_ROPE = ROPE_HEAD_DIM / 2;
static constexpr int NOPE_HEAD_DIM = HEAD_SIZE - ROPE_HEAD_DIM;
static constexpr int NOPE_PAIRS = NOPE_HEAD_DIM / 2;

inline void load_head_paired(const bf16* base, bf16* out, unsigned int lane) {
  out[0] = base[2 * lane];
  out[1] = base[2 * lane + 1];
  out[2] = base[HEAD_SIZE / 2 + 2 * lane];
  out[3] = base[HEAD_SIZE / 2 + 2 * lane + 1];
}

// ============================================================================
// FusedMxfp4Cr4Kernel: Specialized kernel for DeepSeek-V4 CR=4, overlap=1.
// Key optimizations:
//   - Tuned tokens per work-group for this fixed CR=4 path.
//   - Compile-time N_GATHER enables aggressive unrolling.
//   - Register blocking for this indexer gather pattern.
//   - Uses SG_SIZE=32 for the target occupancy/register tradeoff.
// ============================================================================
template <int COMPRESS_RATIO, int OVERLAP>
class FusedMxfp4Cr4Kernel {
 public:
  static_assert(COMPRESS_RATIO == 4, "FusedMxfp4Cr4Kernel is for CR == 4");
  static_assert(OVERLAP == 1, "FusedMxfp4Cr4Kernel requires OVERLAP == 1");
  static constexpr int N_GATHER = (1 + OVERLAP) * COMPRESS_RATIO;
  static constexpr int CR4_TOKENS_PER_WG = 8;
  static constexpr int CR4_SG_SIZE = 32;
  static constexpr int CR4_ELEMENTS_PER_LANE = HEAD_SIZE / CR4_SG_SIZE;
  static constexpr int CR4_PAIR_STRIDE = 2 * CR4_SG_SIZE;  // elements per pair
  static constexpr int CR4_LANES_PER_QBLOCK = QUANT_BLOCK / 2;  // 16
  static constexpr int CR4_QBLOCKS_PER_PAIR = CR4_PAIR_STRIDE / QUANT_BLOCK;

  FusedMxfp4Cr4Kernel(
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

  [[sycl::reqd_sub_group_size(CR4_SG_SIZE)]]
  void operator()(sycl::nd_item<2> item) const {
    // ============================================================================
    // Fixed CR=4, overlap=1 execution path:
    // - one subgroup per token
    // - CR4_SG_SIZE fixed to 32
    // - N_GATHER resolved at compile time
    // ============================================================================

    const int wg_id = item.get_group(0);
    const int sg_id = item.get_sub_group().get_group_id()[0];
    const int lane = item.get_local_id(1) % CR4_SG_SIZE;
    const int wg_token_start = wg_id * CR4_TOKENS_PER_WG;

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
    // PHASE 1: Online Softmax Compression (fixed compile-time N_GATHER)
    // ============================================================================

    constexpr float NEG_LARGE = -std::numeric_limits<float>::infinity();
    float m_run[CR4_ELEMENTS_PER_LANE];
    float s_run[CR4_ELEMENTS_PER_LANE];
    float acc[CR4_ELEMENTS_PER_LANE];

#pragma unroll
    for (int i = 0; i < CR4_ELEMENTS_PER_LANE; ++i) {
      m_run[i] = NEG_LARGE;
      s_run[i] = 0.0f;
      acc[i] = 0.0f;
    }

    // N_GATHER is compile-time constant for this specialization.
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

      // Block load over one 32-lane subgroup: 128 contiguous bf16 total,
      // 4 bf16 per lane in paired layout.
      // local index a (0..3): global element = 32*(a/2) + 2*lane + (a&1)
      if (valid) {
        bf16 kv_buf[CR4_ELEMENTS_PER_LANE];
        bf16 score_buf[CR4_ELEMENTS_PER_LANE];
        load_head_paired(row_ptr, kv_buf, lane);
        load_head_paired(row_ptr + state_width, score_buf, lane);

#pragma unroll
        for (int i = 0; i < CR4_ELEMENTS_PER_LANE; ++i) {
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
    float compressed[CR4_ELEMENTS_PER_LANE];
#pragma unroll
    for (int i = 0; i < CR4_ELEMENTS_PER_LANE; ++i) {
      compressed[i] = acc[i] / s_run[i];
    }

    // ============================================================================
    // PHASE 2: RMSNorm
    // ============================================================================

    float my_var = 0.0f;
#pragma unroll
    for (int i = 0; i < CR4_ELEMENTS_PER_LANE; ++i) {
      my_var += compressed[i] * compressed[i];
    }

    float total_var = sycl::reduce_over_group(sg, my_var, sycl::plus<float>());
    total_var /= HEAD_SIZE;
    float rrms = sycl::rsqrt(total_var + rms_eps);

    float normed[CR4_ELEMENTS_PER_LANE];
#pragma unroll
    for (int i = 0; i < CR4_ELEMENTS_PER_LANE; ++i) {
      int elem_idx = CR4_PAIR_STRIDE * (i / 2) + 2 * lane + (i & 1);
      float w = static_cast<float>(rms_weight_ptr[elem_idx]);
      normed[i] = compressed[i] * rrms * w;
    }

    // ============================================================================
    // PHASE 3: GPT-J RoPE (paired layout: each pair is lane-local, no shuffle)
    // ============================================================================

    int compressed_pos = (position / COMPRESS_RATIO) * COMPRESS_RATIO;
    const float* cs_base = cos_sin_ptr + compressed_pos * cos_sin_stride;

#pragma unroll
    for (int pp = 0; pp < CR4_ELEMENTS_PER_LANE / 2; ++pp) {
      int even_elem = CR4_PAIR_STRIDE * pp + 2 * lane;
      int pair_idx = even_elem / 2;
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
    for (int pp = 0; pp < CR4_ELEMENTS_PER_LANE / 2; ++pp) {
      float a = sycl::fmax(
          sycl::fabs(normed[2 * pp]), sycl::fabs(normed[2 * pp + 1]));
#pragma unroll
      for (int mask = 1; mask < CR4_LANES_PER_QBLOCK; mask <<= 1) {
        a = sycl::fmax(a, sycl::permute_group_by_xor(sg, a, mask));
      }

      auto scale_info = vllm::mxfp4::compute_ue8m0_scale(a);
      float inv_scale = 1.0f / scale_info.scale;

      int block_id = CR4_QBLOCKS_PER_PAIR * pp + lane / CR4_LANES_PER_QBLOCK;
      if ((lane & (CR4_LANES_PER_QBLOCK - 1)) == 0) {
        scale_ptr[block_id] = scale_info.ue8m0;
      }

      uint8_t packed =
          vllm::mxfp4::quantize_pair_to_mxfp4<vllm::mxfp4::Fp4TieBreak::ToEven>(
              normed[2 * pp], normed[2 * pp + 1], inv_scale);
      val_ptr[CR4_SG_SIZE * pp + lane] = packed;
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
  int32_t num_tokens;
};

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

  // DeepSeek-V4 only exercises the MXFP4 indexer with compress_ratio == 4 and
  // overlap == 1 (the indexer layer is created solely for the CR==4 case, and
  // DeepseekCompressor sets overlap = (compress_ratio == 4)). Only that
  // configuration is supported here.
  TORCH_CHECK(
      compress_ratio == 4 && overlap == 1,
      "fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn only supports "
      "compress_ratio == 4 and overlap == 1");

  using Kernel = FusedMxfp4Cr4Kernel<4, 1>;
  const size_t groups =
      (static_cast<size_t>(num_tokens) + Kernel::CR4_TOKENS_PER_WG - 1) /
      Kernel::CR4_TOKENS_PER_WG;

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>(
            {groups * Kernel::CR4_TOKENS_PER_WG,
             static_cast<size_t>(Kernel::CR4_SG_SIZE)},
            {static_cast<size_t>(Kernel::CR4_TOKENS_PER_WG),
             static_cast<size_t>(Kernel::CR4_SG_SIZE)}),
        Kernel(
            state_ptr,
            state_cache.stride(0),
            state_cache.stride(1),
            token_to_req,
            pos_ptr,
            slot_ptr,
            btable,
            block_table.stride(0),
            static_cast<int32_t>(block_size),
            rms_ptr,
            static_cast<float>(rms_norm_eps),
            cs_ptr,
            cos_sin_cache.stride(0),
            kv_ptr,
            kv_slot_ptr,
            static_cast<int32_t>(kv_cache_block_size),
            kv_cache.stride(0),
            state_width,
            num_tokens));
  });
}
