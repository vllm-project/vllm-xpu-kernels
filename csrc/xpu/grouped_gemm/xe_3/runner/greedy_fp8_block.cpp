// SPDX-License-Identifier: Apache-2.0
#include "../kernel/xe_moe_gemm_greedy.hpp"

namespace gpu::cutlass_kernel::grouped_gemm {
template <>
struct GreedyTraits<GreedyDtype::FP8Block>
    // TileK 32 -> 64: K = 3584 drops from 112 mainloop iterations to 56,
    // halving the per-iteration barrier pair, copies, prefetches and reorders
    // for identical DPAS work. k_tiles_per_group becomes 2. This depends on
    // the lane-packed A scales in the mainloop: without the registers those
    // free, the doubled A/B fragments have nowhere to go.
    : GreedyFamily<MoEFp8BlockConfig, 64, 2048, 512> {
  // MidN was 128, which cost 1.5x on Mid-bucket shapes. 256 matches per-tensor
  // FP8. The 320-row Mid tile is no longer capped: it used to spill 88
  // registers at N=256 in the single kernel every bucket shares, but packing
  // the A scales across lanes drops the live scale state from sg_m_rows GRF to
  // ceil(sg_m_rows / 16), so the tall tile now fits. These traits are
  // consequently identical to GreedyTraits<FP8Tensor>, which has always run
  // Mid320x256 without spilling.
  static constexpr int NormalN = 256, MidN = 256;
  static constexpr bool AllowMid384 = false, SplitTinyM = true;
};

template void
launch_greedy<GreedyDtype::FP8Block>(sycl::queue&, GreedyArguments);
}  // namespace gpu::cutlass_kernel::grouped_gemm
