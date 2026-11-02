// SPDX-License-Identifier: Apache-2.0
#include "../kernel/xe_moe_gemm_greedy.hpp"

namespace gpu::cutlass_kernel::grouped_gemm {
template <>
struct GreedyTraits<GreedyDtype::W4A8>
    : GreedyFamily<MoEW4a8Config, 32, 2048, 512> {
  static constexpr int NormalN = 256;
  static constexpr bool AllowMid384 = false;
};

template void launch_greedy<GreedyDtype::W4A8>(sycl::queue&, GreedyArguments);
}  // namespace gpu::cutlass_kernel::grouped_gemm
