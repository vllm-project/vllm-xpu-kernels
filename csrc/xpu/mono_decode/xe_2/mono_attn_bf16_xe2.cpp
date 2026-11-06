// SPDX-License-Identifier: Apache-2.0
//
// Full-attention + MoE decoder layer as one kernel (see mono_layers.hpp), bf16.
// Its own translation unit, like the GDN layer.
#include "mono_layers_b.hpp"

namespace mono_decode {
template void launch_attn<bf16>(
    sycl::queue&, int, const Ptrs&, const Scratch&, const ScratchB&);
}  // namespace mono_decode
