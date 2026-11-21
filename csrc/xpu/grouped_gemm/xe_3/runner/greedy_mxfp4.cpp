// SPDX-License-Identifier: Apache-2.0
#include "../kernel/xe_moe_gemm_greedy.hpp"

namespace gpu::cutlass_kernel::grouped_gemm {
template <>
struct GreedyTraits<GreedyDtype::MXFP4>
    : GreedyFamily<MoEMxfp4Config, 128, 1280, 768> {
  static constexpr int NormalN = 256;
  static constexpr bool AllowMid384 = false;

  static constexpr greedy::MidWindow
  uniform_mid_window(int n, int k, greedy::MidWindow fallback) {
    // Keep our narrow family at N=1536/3072; the benchmark uses MidN384
    // there. The remaining measured cells use the same MidN256 family.
    constexpr greedy::MidOverride overrides[] = {
        {2048, 4096, {128, 832}},
        {2560, 3584, {128, 832}},
        {3584, 1280, {128, 662, 405, 469}},
        {4096, 1536, {128, 576, 288, 470}},
        {4096, 2048, {128, 448}},
        {4096, 4096, {128, 470}},
    };
    return greedy::mid_window_for(overrides, n, k, fallback);
  }
};

template void launch_greedy<GreedyDtype::MXFP4>(sycl::queue&, GreedyArguments);
}  // namespace gpu::cutlass_kernel::grouped_gemm
