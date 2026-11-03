// SPDX-License-Identifier: Apache-2.0
//
// Linear-attention (GatedDeltaNet) + MoE decoder layer as one kernel (see
// mono_layers.hpp). Its own translation unit: the two layer kernels in one
// device image run slower (code layout).
#include "mono_layers_b.hpp"

namespace mono_decode {
void launch_gdn(sycl::queue& q, const Ptrs& P, const Scratch& S) {
  launch_t<false>(q, P, S);
}
void launch_gdn_b(
    sycl::queue& q,
    int nb,
    const Ptrs& P,
    const Scratch& S,
    const ScratchB& SB) {
  if (nb == 2)
    launch_gdn_b<2>(q, P, S, SB);
  else if (nb == 4)
    launch_gdn_b<4>(q, P, S, SB);
  else
    launch_gdn_b<8>(q, P, S, SB);
}
}  // namespace mono_decode
