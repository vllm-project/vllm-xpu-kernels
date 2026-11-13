// SPDX-License-Identifier: Apache-2.0
//
// MTP drafter prologue (norms, fc) as one kernel (see mono_mtp.hpp), fp16.
#include "mono_mtp.hpp"

namespace mono_decode {
template bool
launch_mtp<half>(sycl::queue&, int, const Ptrs&, const Scratch&, int);
}  // namespace mono_decode
