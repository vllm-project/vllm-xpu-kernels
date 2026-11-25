// SPDX-License-Identifier: Apache-2.0
#define MINICPM5_MONO_FP16 1
#include "minicpm5/mono.h"

namespace vllm::minicpm5::mono_fp16 {

#if defined(MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED) &&  \
    defined(MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED) && \
    defined(MINICPM5_MONO_REFERENCE_MOE_ENABLED) &&        \
    defined(MINICPM5_MONO_DPAS_ENABLED)
void dispatch_tail_rows4(
    sycl::queue& queue, Weights weights, Workspace work, Step step) {
  launch<
      1,
      4,
      false,
      true,
      false,
      false,
      false,
      false,
      LayerInput::PostAttentionNormRows4>(queue, weights, work, step);
}
#endif

}  // namespace vllm::minicpm5::mono_fp16
