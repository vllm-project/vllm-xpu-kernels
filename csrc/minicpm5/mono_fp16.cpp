// SPDX-License-Identifier: Apache-2.0
#define MINICPM5_MONO_FP16 1
#include "minicpm5/mono.h"

namespace vllm::minicpm5::mono_fp16 {
template <int TP>
void dispatch_rows(
    sycl::queue& queue,
    Weights weights,
    Workspace work,
    Step step,
    int rows,
    bool dense,
    bool ignore_padding) {
  switch (rows) {
    case 1:
      dispatch_layer<TP, 1>(queue, weights, work, step, dense, ignore_padding);
      break;
    case 2:
      dispatch_layer<TP, 2>(queue, weights, work, step, dense, ignore_padding);
      break;
    case 4:
      dispatch_layer<TP, 4>(queue, weights, work, step, dense, ignore_padding);
      break;
    case 3:
      dispatch_layer<TP, 3>(queue, weights, work, step, dense, ignore_padding);
      break;
    case 5:
      dispatch_layer<TP, 5>(queue, weights, work, step, dense, ignore_padding);
      break;
    case 6:
      dispatch_layer<TP, 6>(queue, weights, work, step, dense, ignore_padding);
      break;
    case 7:
      dispatch_layer<TP, 7>(queue, weights, work, step, dense, ignore_padding);
      break;
    case 8:
#if defined(MINICPM5_MONO_REFERENCE_MOE_ENABLED) && \
    defined(MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED)
      if (!dense && ignore_padding) {
  #if defined(MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED) && \
      defined(MINICPM5_MONO_DPAS_ENABLED)
        constexpr int PageTokens = 64;
        if (step.block_size == PageTokens &&
            (step.cache_token_stride == 2 * D ||
             step.cache_token_stride == (2 / TP) * (2 * D)) &&
            step.cache_block_stride == PageTokens * (2 / TP) * (2 * D)) {
          if constexpr (TP == 1) {
            if (step.block_table_stride != 0) {
              launch<TP, 8, false, true, false, true, true, true>(
                  queue, weights, work, step);
              break;
            }
          }
          launch<TP, 8, false, true, false, true, true>(
              queue, weights, work, step);
          break;
        }
  #endif
        launch<TP, 8, false, true, false, true>(queue, weights, work, step);
        break;
      }
#endif
      dispatch_layer<TP, 8>(queue, weights, work, step, dense, ignore_padding);
      break;
  }
}

void dispatch(
    sycl::queue& queue,
    Weights weights,
    Workspace work,
    Step step,
    int tp,
    int rows,
    bool dense,
    bool ignore_padding) {
  if (tp == 1)
    dispatch_rows<1>(queue, weights, work, step, rows, dense, ignore_padding);
  else
    dispatch_rows<2>(queue, weights, work, step, rows, dense, ignore_padding);
}
#if defined(MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED) &&  \
    defined(MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED) && \
    defined(MINICPM5_MONO_REFERENCE_MOE_ENABLED) &&        \
    defined(MINICPM5_MONO_DPAS_ENABLED)
void dispatch_tail(
    sycl::queue& queue, Weights weights, Workspace work, Step step) {
  launch<
      1,
      8,
      false,
      true,
      false,
      true,
      true,
      true,
      LayerInput::PostAttentionNorm>(queue, weights, work, step);
}

void dispatch_pre_o(
    sycl::queue& queue, Weights weights, Workspace work, Step step) {
  launch<
      2,
      8,
      false,
      true,
      false,
      true,
      true,
      false,
      LayerInput::GatedAttention>(queue, weights, work, step);
}

void dispatch_post_o(
    sycl::queue& queue, Weights weights, Workspace work, Step step) {
  launch<
      2,
      8,
      false,
      true,
      false,
      true,
      true,
      false,
      LayerInput::OutputProjection>(queue, weights, work, step);
}
#endif

void dispatch_tiled_qkv(
    sycl::queue& queue,
    Weights weights,
    Workspace work,
    Step step,
    int tp,
    bool dense) {
  if (tp == 1) {
    if (dense)
      launch<1, 1, true, false, true>(queue, weights, work, step);
    else
      launch<1, 1, false, false, true>(queue, weights, work, step);
  } else {
    if (dense)
      launch<2, 1, true, false, true>(queue, weights, work, step);
    else
      launch<2, 1, false, false, true>(queue, weights, work, step);
  }
}
}  // namespace vllm::minicpm5::mono_fp16
