// SPDX-License-Identifier: Apache-2.0
//
// Speculative-decoding verify: linear-attention (GatedDeltaNet) + MoE layer
// for NB = requests x T tokens (see mono_layers_s.hpp), bf16. Its own
// translation unit, like the decode kernels.
#include "mono_layers_s.hpp"

namespace mono_decode {
template bool launch_gdn_spec<bf16>(
    sycl::queue&, int, int, const Ptrs&, const Scratch&, const ScratchB&);
}  // namespace mono_decode
