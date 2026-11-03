// SPDX-License-Identifier: Apache-2.0
//
// Full-attention + MoE decoder layer as one kernel (see mono_layers.hpp).
#include "mono_layers_b.hpp"

namespace mono_decode {
void launch_attn(sycl::queue& q, const Ptrs& P, const Scratch& S) {
  launch_t<true>(q, P, S);
}
void launch_attn_b(
    sycl::queue& q,
    int nb,
    const Ptrs& P,
    const Scratch& S,
    const ScratchB& SB) {
  if (nb == 2)
    launch_attn_b<2>(q, P, S, SB);
  else if (nb == 4)
    launch_attn_b<4>(q, P, S, SB);
  else
    launch_attn_b<8>(q, P, S, SB);
}
}  // namespace mono_decode
