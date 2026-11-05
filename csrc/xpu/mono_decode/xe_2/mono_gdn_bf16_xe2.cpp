// SPDX-License-Identifier: Apache-2.0
//
// Linear-attention (GatedDeltaNet) + MoE decoder layer as one kernel (see
// mono_layers.hpp), bf16. Its own translation unit: the two layer kernels in
// one device image run slower (code layout); so do the two dtypes.
#include "mono_layers_b.hpp"

namespace mono_decode {
template void launch_gdn<bf16>(
    sycl::queue&, int, const Ptrs&, const Scratch&, const ScratchB&);
}  // namespace mono_decode
