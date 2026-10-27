// SPDX-License-Identifier: Apache-2.0
#include "../kernel/xe_moe_gemm_greedy.hpp"

namespace gpu::cutlass_kernel::grouped_gemm {
template <>
struct GreedyTraits<GreedyDtype::MXFP8>
    : GreedyFamily<MoEMxfp8Config, 64, 2048, 512> {
  static constexpr greedy::MidWindow
  uniform_mid_window(int n, int k, greedy::MidWindow fallback) {
    // Match the SYCL-TLA benchmark's measured uniform-expert M windows.
    constexpr greedy::MidOverride overrides[] = {
        {1536, 1536, {128, 576}},
        {1536, 4096, {128, 704}},
        {2048, 4096, {128, 576, 278, 528}},
        {2560, 1536, {128, 832}},
        {2560, 3584, {128, 662}},
        {2560, 4096, {128, 704, 320, 576}},
        {3072, 1536, {128, 704, 320, 454}},
        {3072, 4096, {128, 704, 506, 528}},
        {3584, 1280, {128, 278}},
        {4096, 1536, {}},
        {4096, 2048, {}},
        {4096, 4096, {128, 449, 330, 416}},
    };
    return greedy::mid_window_for(overrides, n, k, fallback);
  }
};

template void launch_greedy<GreedyDtype::MXFP8>(sycl::queue&, GreedyArguments);
}  // namespace gpu::cutlass_kernel::grouped_gemm
