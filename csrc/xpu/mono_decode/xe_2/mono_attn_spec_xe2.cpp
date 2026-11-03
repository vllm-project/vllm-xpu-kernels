// SPDX-License-Identifier: Apache-2.0
//
// Speculative-decoding verify: full-attention + MoE layer
// for NB = requests x T tokens (see mono_layers_s.hpp). Its own translation
// unit, like the decode kernels.
#include "mono_layers_s.hpp"

namespace mono_decode {
bool launch_attn_s(
    sycl::queue& q,
    int nb,
    int t,
    const Ptrs& P,
    const Scratch& S,
    const ScratchB& SB) {
  if (nb == 2 && t == 2)
    launch_attn_s<2, 2>(q, P, S, SB);
  else if (nb == 3 && t == 3)
    launch_attn_s<3, 3>(q, P, S, SB);
  else if (nb == 4 && t == 2)
    launch_attn_s<4, 2>(q, P, S, SB);
  else if (nb == 4 && t == 4)
    launch_attn_s<4, 4>(q, P, S, SB);
  else if (nb == 6 && t == 3)
    launch_attn_s<6, 3>(q, P, S, SB);
  else if (nb == 8 && t == 2)
    launch_attn_s<8, 2>(q, P, S, SB);
  else if (nb == 8 && t == 4)
    launch_attn_s<8, 4>(q, P, S, SB);
  else if (nb == 8 && t == 8)
    launch_attn_s<8, 8>(q, P, S, SB);
  else
    return false;
  return true;
}
}  // namespace mono_decode
