// SPDX-License-Identifier: Apache-2.0
// Runtime dispatch wrappers for cache operations.
//
// This file is compiled once into the _C extension. It defines the public cache
// ops that torch_bindings.cpp registers, and forwards each call to an
// arch-specific implementation:
//   * <op>_default : fallback path, compiled into _C, valid on every arch.
//   * <op>_xe3asm  : xe3 asm-path implementation, compiled into lib_C_xe3asm.so
//                    and only linked/dispatched when VLLM_BUILD_XE3ASM is set.

#include <torch/torch.h>
#include <optional>
#include <string>

#include "ops.h"
#include "utils.h"

// Declare the arch-suffixed implementations for a given suffix in one place so
// the _default and _xe3asm variants cannot drift apart.
#define VLLM_DECLARE_CACHE_ARCH_VARIANTS(SUF)                              \
  extern void reshape_and_cache_flash##SUF(                                \
      torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&,      \
      torch::Tensor&, const std::string&, torch::Tensor&, torch::Tensor&);
  
VLLM_DECLARE_CACHE_ARCH_VARIANTS(_default)
#ifdef VLLM_BUILD_XE3ASM
VLLM_DECLARE_CACHE_ARCH_VARIANTS(_xe3asm)
#endif

#undef VLLM_DECLARE_CACHE_ARCH_VARIANTS

#ifdef VLLM_BUILD_XE3ASM
// Prefer the xe3 asm path when it was built in and the running device is xe3,
// unless the user forces the fallback via VLLM_XPU_FORCE_XE_DEFAULT_KERNEL.
static inline bool use_xe3asm() {
  return vllm::xpu::is_xe3_arch() && !vllm::xpu::force_xe_default_kernel();
}
#endif

// Dispatch wrappers — these are the symbols that torch_bindings.cpp references.

void reshape_and_cache_flash(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
#ifdef VLLM_BUILD_XE3ASM
  if (use_xe3asm()) {
    reshape_and_cache_flash_xe3asm(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale);
    return;
  }
#endif
  reshape_and_cache_flash_default(
      key,
      value,
      key_cache,
      value_cache,
      slot_mapping,
      kv_cache_dtype,
      k_scale,
      v_scale);
}
