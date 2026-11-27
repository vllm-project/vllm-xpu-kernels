// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace vllm::minicpm5 {
inline constexpr int MaxContext = 512 * 1024;

template <typename Scalar>
struct WeightPointers {
  const Scalar *input_norm, *post_norm;
  const uint8_t *qkv, *out, *w13, *w2, *shared13, *shared2;
  const float *qkv_scale, *out_scale, *w13_scale, *w2_scale;
  const Scalar* router;
  const float *bias, *shared13_scale, *shared2_scale;

  template <typename Other>
  WeightPointers<Other> as() const {
    return {
        reinterpret_cast<const Other*>(input_norm),
        reinterpret_cast<const Other*>(post_norm),
        qkv,
        out,
        w13,
        w2,
        shared13,
        shared2,
        qkv_scale,
        out_scale,
        w13_scale,
        w2_scale,
        reinterpret_cast<const Other*>(router),
        bias,
        shared13_scale,
        shared2_scale};
  }
};

template <typename Scalar>
struct WorkspacePointers {
  Scalar *normed, *qkv, *attention, *intermediate, *expert_out;
  float *attention_partial, *router, *routing, *gate_up_partial;
  int* experts;

  template <typename Other>
  WorkspacePointers<Other> as() const {
    return {
        reinterpret_cast<Other*>(normed),
        reinterpret_cast<Other*>(qkv),
        reinterpret_cast<Other*>(attention),
        reinterpret_cast<Other*>(intermediate),
        reinterpret_cast<Other*>(expert_out),
        attention_partial,
        router,
        routing,
        gate_up_partial,
        experts};
  }
};

template <typename Scalar>
struct StepPointers {
  const Scalar *hidden, *residual;
  Scalar *output, *res_out, *cache;
  const int64_t *positions, *slots;
  const int *blocks, *lengths;
  int block_size, block_table_stride;
  int64_t cache_blocks, block_table_cols;
  int64_t cache_block_stride, cache_head_stride, cache_token_stride;
  uint32_t* control;
  uint64_t *mailbox, *peer;
  uint32_t* health = nullptr;
  const Scalar* rope_cache = nullptr;
  int64_t rope_cache_rows = 0;

  template <typename Other>
  StepPointers<Other> as() const {
    return {
        reinterpret_cast<const Other*>(hidden),
        reinterpret_cast<const Other*>(residual),
        reinterpret_cast<Other*>(output),
        reinterpret_cast<Other*>(res_out),
        reinterpret_cast<Other*>(cache),
        positions,
        slots,
        blocks,
        lengths,
        block_size,
        block_table_stride,
        cache_blocks,
        block_table_cols,
        cache_block_stride,
        cache_head_stride,
        cache_token_stride,
        control,
        mailbox,
        peer,
        health,
        reinterpret_cast<const Other*>(rope_cache),
        rope_cache_rows};
  }
};

inline constexpr int DraftRows = 7;

#if defined(MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED) && \
    defined(MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED)
template <typename Scalar>
struct DraftWeights {
  const Scalar *input_norm, *post_norm, *q_norm, *k_norm;
  const Scalar *qkv, *out, *gate_up, *down;
};
#endif

namespace mono_fp16 {
#if defined(MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED) && \
    defined(MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED)
void dispatch_draft(
    sycl::queue& queue,
    DraftWeights<sycl::half> weights,
    WorkspacePointers<sycl::half> work,
    StepPointers<sycl::half> step,
    int tp);
#endif
void dispatch(
    sycl::queue& queue,
    WeightPointers<sycl::half> weights,
    WorkspacePointers<sycl::half> work,
    StepPointers<sycl::half> step,
    int tp,
    int rows,
    bool dense,
    bool ignore_padding);
#if defined(MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED) &&  \
    defined(MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED) && \
    defined(MINICPM5_MONO_REFERENCE_MOE_ENABLED) &&        \
    defined(MINICPM5_MONO_DPAS_ENABLED)
void dispatch_tail_rows4(
    sycl::queue& queue,
    WeightPointers<sycl::half> weights,
    WorkspacePointers<sycl::half> work,
    StepPointers<sycl::half> step);

void dispatch_tail(
    sycl::queue& queue,
    WeightPointers<sycl::half> weights,
    WorkspacePointers<sycl::half> work,
    StepPointers<sycl::half> step);

void dispatch_pre_o(
    sycl::queue& queue,
    WeightPointers<sycl::half> weights,
    WorkspacePointers<sycl::half> work,
    StepPointers<sycl::half> step);

void dispatch_pre_o_rows4(
    sycl::queue& queue,
    WeightPointers<sycl::half> weights,
    WorkspacePointers<sycl::half> work,
    StepPointers<sycl::half> step);

void dispatch_post_o(
    sycl::queue& queue,
    WeightPointers<sycl::half> weights,
    WorkspacePointers<sycl::half> work,
    StepPointers<sycl::half> step);
#endif
void dispatch_tiled_qkv(
    sycl::queue& queue,
    WeightPointers<sycl::half> weights,
    WorkspacePointers<sycl::half> work,
    StepPointers<sycl::half> step,
    int tp,
    bool dense);
}  // namespace mono_fp16
}  // namespace vllm::minicpm5
