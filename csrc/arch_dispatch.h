// SPDX-License-Identifier: Apache-2.0
// Architecture dispatch macros for multi-AOT fat binary.
//
// When a source file is compiled multiple times with different
// SYCL_INTEL_TARGET values, these macros rename public functions with an arch
// suffix to avoid symbol conflicts between per-arch shared libraries.
//
// Usage: wrap public function names with ARCH_FUNC(name).
//   void ARCH_FUNC(reshape_and_cache_flash)(...) { ... }
// This expands to reshape_and_cache_flash_xe2 (or other arch suffix)
// depending on the compilation target.

#pragma once

#define _ARCH_CAT(a, b) a##b
#define ARCH_CAT(a, b) _ARCH_CAT(a, b)

#define ARCH_SUFFIX _xe2

#define ARCH_FUNC(name) ARCH_CAT(name, ARCH_SUFFIX)
