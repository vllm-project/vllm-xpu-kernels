// SPDX-License-Identifier: Apache-2.0
//
// MTP drafter prologue (norms, fc) as one kernel (see mono_mtp.hpp).
#include "mono_mtp.hpp"

namespace mono_decode {
bool launch_mtp(
    sycl::queue& q, int nb, const Ptrs& P, const Scratch& S, int rank) {
  if (nb == 1)
    launch_mtp_prologue<1>(q, P, S, rank);
  else if (nb == 2)
    launch_mtp_prologue<2>(q, P, S, rank);
  else if (nb == 3)
    launch_mtp_prologue<3>(q, P, S, rank);
  else if (nb == 4)
    launch_mtp_prologue<4>(q, P, S, rank);
  else if (nb == 6)
    launch_mtp_prologue<6>(q, P, S, rank);
  else if (nb == 8)
    launch_mtp_prologue<8>(q, P, S, rank);
  else
    return false;
  return true;
}
}  // namespace mono_decode
