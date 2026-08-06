#pragma once

// Per-architecture inline namespace selection.
//
// The shared sources under common/ are compiled once into every backend library
// (fused_moe_xe2, and later fused_moe_xe3). To avoid exporting identical
// mangled symbols from more than one library -- which would let the dynamic
// linker interpose one backend's kernels over another's -- the concrete kernel
// entry points are placed in a per-arch inline namespace (FusedMOE::xe2::...,
// FusedMOE::xe3::...). Because it is an inline namespace, unqualified and
// `FusedMOE::`-qualified references keep resolving to the current backend.
//
// Each backend library is compiled with exactly one of the VLLM_XPU_ENABLE_*
// macros (added PRIVATE by add_xe2_kernel_library), so a given translation unit
// only ever sees a single arch namespace.

#if defined(VLLM_XPU_ENABLE_XE3)
  #define FUSED_MOE_ARCH_NS xe3
#elif defined(VLLM_XPU_ENABLE_XE2)
  #define FUSED_MOE_ARCH_NS xe2
#else
  #define FUSED_MOE_ARCH_NS generic
#endif

#define FUSED_MOE_NS_BEGIN \
  namespace FusedMOE {     \
  inline namespace FUSED_MOE_ARCH_NS {
#define FUSED_MOE_NS_END \
  }                      \
  }
