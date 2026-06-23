#pragma once

#include "cutlass/bfloat16.h"

#include <cstdint>

#ifndef FLASH_NAMESPACE
#define FLASH_NAMESPACE vllm::xpu::attn::mla
#endif

namespace FLASH_NAMESPACE {

#ifndef FLASH_MLA_PREFILL_B_TOPK
#define FLASH_MLA_PREFILL_B_TOPK 32
#endif

static constexpr float LOG_2_E = 1.4426950408889634f;
static constexpr float LOG_E_2 = 0.6931471805599453f;

// specific for DeepSeek V4
static constexpr int SPARSE_MLA_FP8_NOPE_BYTES = 448;
static constexpr int SPARSE_MLA_FP8_ROPE_DIM = 64;
static constexpr int SPARSE_MLA_FP8_DATA_BYTES_PER_TOKEN = 576;
static constexpr int SPARSE_MLA_FP8_SCALE_BYTES_PER_TOKEN = 8;
static constexpr int SPARSE_MLA_FP8_HEAD_BYTES = 584;

// Below are for FlashMLA definitions

// Same as https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/params.h#L145
struct SparseAttnFwdParams {
    int s_q, s_kv, h_q, h_kv, d_qk, d_v, topk;
    float sm_scale, sm_scale_div_log2;

    // Input tensors
    cutlass::bfloat16_t* __restrict__ q;    // [s_q, h_q, d_qk]
    cutlass::bfloat16_t* __restrict__ kv;   // [s_kv, h_kv, d_qk]
    int* __restrict__ indices;   // [s_q, h_kv, topk]
    float* __restrict__ attn_sink;   // [h_q], may be nullptr
    int* __restrict__ topk_length;    // [s_q], may be nullptr

    // Strides
    int stride_q_s_q; int stride_q_h_q;
    int stride_kv_s_kv; int stride_kv_h_kv;
    int stride_indices_s_q; int stride_indices_h_kv;

    // Internal gathered dense KV workspace.
    cutlass::bfloat16_t* __restrict__ gathered_k;
    int* __restrict__ gathered_valid_mask;

    // Gathered workspace strides.
    int stride_gathered_k_s_q; int stride_gathered_k_topk;
    int stride_gathered_mask_s_q;

    // Output tensors
    cutlass::bfloat16_t* __restrict__ out;   // [s_q, h_q, d_v]
    float* __restrict__ max_logits; // [s_q, h_q], may be nullptr
    float* __restrict__ lse; // [s_q, h_q], may be nullptr
    bool return_softmax_lse;

    int num_sm;
};

// WA: avoid issue of not supported device-only copy
struct XPUSparseAttnFwdParams : public SparseAttnFwdParams {
    sycl::queue queue;
};

struct SparseAttnDecodeParams {
    int b, s_q;
    int h_q, h_kv;
    int d_qk, d_v;
    float sm_scale, sm_scale_div_log2;
    int num_blocks, page_block_size, topk;
    int gathered_topk;

    void* __restrict__ q;   // [b, s_q, h_q, d_qk], bf16
    uint8_t* __restrict__ kv;  // packed fp8 KV cache, [num_blocks, page_block_size, h_kv=1, head_bytes]
    int* __restrict__ indices;   // [b, s_q, topk]
    int* __restrict__ topk_length;  // [b], may be nullptr
    float* __restrict__ attn_sink;  // [h_q], may be nullptr
    cutlass::bfloat16_t* __restrict__ gathered_k;  // [b, s_q, gathered_topk, d_qk]
    int* __restrict__ gathered_valid_mask;  // [b, s_q, gathered_topk]

    float* __restrict__ lse;    // [b, s_q, h_q]
    cutlass::bfloat16_t* __restrict__ out;   // [b, s_q, h_q, d_v]
    
    int extra_num_blocks, extra_page_block_size, extra_topk;
  uint8_t* __restrict__ extra_kv;  // packed fp8 KV cache, may be nullptr
    int* __restrict__ extra_indices;   // [b, s_q, extra_topk]
  int* __restrict__ extra_topk_length;  // [b], may be nullptr
    
    int stride_q_b, stride_q_s_q, stride_q_h_q;
    int stride_kv_block, stride_kv_row, stride_kv_head;
    int stride_indices_b, stride_indices_s_q;
    int stride_topk_length_b, stride_topk_length_s_q;  // stride_topk_length_s_q is unused for decode (1D topk_length)
    int stride_gathered_k_b, stride_gathered_k_s_q, stride_gathered_k_topk;
    int stride_gathered_mask_b, stride_gathered_mask_s_q;
    int stride_lse_b, stride_lse_s_q;
    int stride_o_b, stride_o_s_q, stride_o_h_q;
    int stride_extra_kv_block, stride_extra_kv_row, stride_extra_kv_head;
    int stride_extra_indices_b, stride_extra_indices_s_q;
    int stride_extra_topk_length_b, stride_extra_topk_length_s_q;  // stride_extra_topk_length_s_q is unused for decode (1D extra_topk_length)
    
    // cudaStream_t stream;
    
    // SplitKV-related parameters
    float* __restrict__ lse_accum;  // [num_splits, s_q, h_q]
    float* __restrict__ o_accum;    // [num_splits, s_q, h_q, d_v]
    int stride_lse_accum_split, stride_lse_accum_s_q;
    int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;
    void* __restrict__ tile_scheduler_metadata_ptr; // unused by no-split sparse decode for now
    int* __restrict__ num_splits_ptr; // [batch_size+1, ], contiguous
    int num_sm_parts;
    int num_sm;
};

// WA: avoid issue of not supported device-only copy
struct XPUSparseDecodeAttnFwdParams : public SparseAttnDecodeParams {
    sycl::queue queue;
};

template<int D_QK, bool HAVE_TOPK_LENGTH>
void launch_sparse_mla_prefill_gather_kernel(const XPUSparseAttnFwdParams& params);

template<int D_QK, bool HAVE_TOPK_LENGTH, bool HAS_ATTN_SINK>
void launch_sparse_mla_prefill_fwd_kernel(const XPUSparseAttnFwdParams& params);

template<int D_QK, bool HAS_TOPK_LENGTH, bool HAS_ATTN_SINK>
void launch_sparse_mla_decode_fp8_fwd_kernel(const XPUSparseDecodeAttnFwdParams& params);

} // namespace FLASH_NAMESPACE
