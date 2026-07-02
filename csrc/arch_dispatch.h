// SPDX-License-Identifier: Apache-2.0
// Architecture dispatch macros for multi-AOT fat binaries.
//
// A source file that uses ARCH_FUNC() can be compiled more than once with a
// different ARCH_SUFFIX so its public symbols do not collide between the
// fallback path (compiled into the _C extension) and an arch-specialized shared
// library (e.g. lib_C_xe3asm.so). The suffix is injected at compile time:
//
//   * fallback / default path : no define -> ARCH_SUFFIX defaults to _default
//   * xe3 asm path            : -DARCH_SUFFIX=_xe3asm -DVLLM_XE3_ASM
//
// Usage: wrap public function names with ARCH_FUNC(name).
//   void ARCH_FUNC(reshape_and_cache_flash)(...) { ... }
// expands to reshape_and_cache_flash_default or reshape_and_cache_flash_xe3asm
// depending on the compilation target. Inside the body, guard the specialized
// implementation with #ifdef VLLM_XE3_ASM.

#pragma once

#define _ARCH_CAT(a, b) a##b
#define ARCH_CAT(a, b) _ARCH_CAT(a, b)

// ARCH_SUFFIX is normally injected via the build system (-DARCH_SUFFIX=...).
// Fall back to the generic suffix when it is not provided.
#ifndef ARCH_SUFFIX
#define ARCH_SUFFIX _default
#endif

#define ARCH_FUNC(name) ARCH_CAT(name, ARCH_SUFFIX)
