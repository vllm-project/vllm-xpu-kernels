// SPDX-License-Identifier: Apache-2.0
#include "../kernel/xe_moe_gemm_greedy.hpp"

namespace gpu::cutlass_kernel::grouped_gemm {
template <>
struct GreedyTraits<GreedyDtype::BF16>
    : GreedyFamily<MoEBf16Config, 32, 0, 768> {};

template void launch_greedy<GreedyDtype::BF16>(sycl::queue&, GreedyArguments);
}  // namespace gpu::cutlass_kernel::grouped_gemm
