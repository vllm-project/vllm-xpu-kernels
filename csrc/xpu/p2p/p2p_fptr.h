#pragma once

#include <cstdint>

namespace vllm {
namespace xpu {
namespace p2p {

// Device addresses cross the torch op boundary as int64_t, the way vLLM's
// custom_all_reduce passes `fptr_t`.  On XPU they do not fit in one: Level
// Zero hands out USM device pointers above 2**63, so what actually arrives
// is the two's-complement bit pattern and the int64_t is negative.
//
// Converting a negative integer straight to a pointer with reinterpret_cast
// is implementation-defined.  Going through uintptr_t first is well defined
// for every value a caller can produce -- the signed-to-unsigned conversion
// is modular, so it reproduces the bit pattern exactly -- and it states the
// intent: reinterpret the bits, do not sign-extend.
//
// vllm_xpu_kernels/p2p.py::as_fptr is the matching conversion on the way in.
template <typename T>
inline T* from_fptr(int64_t fptr) {
  return reinterpret_cast<T*>(static_cast<uintptr_t>(fptr));
}

// The same reinterpretation on the way out, for an address this code hands
// back.  reinterpret_cast<int64_t> directly has the same problem in reverse:
// the address need not be representable.  C++20 makes the narrowing modular,
// so this round-trips through from_fptr exactly.
template <typename T>
inline int64_t to_fptr(T* p) {
  return static_cast<int64_t>(reinterpret_cast<uintptr_t>(p));
}

}  // namespace p2p
}  // namespace xpu
}  // namespace vllm
