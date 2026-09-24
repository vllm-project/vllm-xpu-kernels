#pragma once

#include <cstdint>

namespace vllm {
namespace xpu {
namespace p2p {

// Device addresses cross the torch op boundary as int64_t, like vLLM's
// custom_all_reduce `fptr_t`.  Level Zero hands out USM device pointers above
// 2**63, which torch's int schema type cannot carry, so callers pass the
// two's-complement bit pattern (vllm_xpu_kernels/p2p.py::as_fptr).  Going
// through uintptr_t reinterprets those bits exactly, where converting a
// negative integer straight to a pointer is implementation-defined.
template <typename T>
inline T* from_fptr(int64_t fptr) {
  return reinterpret_cast<T*>(static_cast<uintptr_t>(fptr));
}

template <typename T>
inline int64_t to_fptr(T* p) {
  return static_cast<int64_t>(reinterpret_cast<uintptr_t>(p));
}

// Bytes of a region for a staging slot of slot_bytes, laid out as documented
// in p2p_collective.cpp.
int64_t region_bytes(int64_t slot_bytes);

}  // namespace p2p
}  // namespace xpu
}  // namespace vllm
