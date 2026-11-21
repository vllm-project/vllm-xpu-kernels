// SPDX-License-Identifier: Apache-2.0
#include "../kernel/xe_moe_gemm_greedy.hpp"

namespace gpu::cutlass_kernel::grouped_gemm {
template <>
struct GreedyTraits<GreedyDtype::FP8Tensor>
    : GreedyFamily<MoEFp8TensorConfig, 32, 2048, 512> {
  static constexpr int NormalN = 256;
  static constexpr bool AllowMid384 = false, SplitTinyM = true;
};

template void
launch_greedy<GreedyDtype::FP8Tensor>(sycl::queue&, GreedyArguments);
}  // namespace gpu::cutlass_kernel::grouped_gemm
