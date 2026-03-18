#pragma once

#include "chunk_prefill.hpp"

// =============================================================================
// Configuration: Add new policies here
// =============================================================================
// To add a new head size policy, simply add it to this X-Macro list:
// X(policy_name)
#define CHUNK_POLICY_LIST(X) \
  X(chunk_policy_head128)    \

// =============================================================================
// Automatic extern template declarations for all policy combinations
// =============================================================================
// Only Paged is a compile-time variant; Causal, Local, Sink are disabled
// (hardcoded to false). This gives 2 instantiations per policy.

#define DECLARE_POLICY_DISPATCH_EXTERN(POLICY, PAGED)       \
  extern template void                                      \
  policy_dispatch_impl<POLICY, PAGED>(                      \
      sycl::queue & queue,                                  \
      CutlassQKType & cuQKType,                             \
      const chunk_prefill_args_t& args);

#define DECLARE_ALL_PAGED_COMBINATIONS(POLICY) \
  DECLARE_POLICY_DISPATCH_EXTERN(POLICY, true)

CHUNK_POLICY_LIST(DECLARE_ALL_PAGED_COMBINATIONS)

#undef DECLARE_ALL_PAGED_COMBINATIONS
#undef DECLARE_POLICY_DISPATCH_EXTERN
