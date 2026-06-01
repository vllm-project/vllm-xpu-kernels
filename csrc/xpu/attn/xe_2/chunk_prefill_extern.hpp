#pragma once
#include "chunk_prefill.hpp"
#define CHUNK_POLICY_LIST(X) X(chunk_policy_head128)
#define DECLARE_POLICY_DISPATCH_EXTERN(POLICY, PAGED, CAUSAL, LOCAL, SINK, LSE) \
  extern template void policy_dispatch_impl<POLICY, PAGED, CAUSAL, LOCAL, SINK, LSE>( \
      sycl::queue & queue, CutlassQKType & cuQKType, const chunk_prefill_args_t& args);
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head128, false, true, false, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head128, false, true, false, false, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head128, false, false, false, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head128, false, false, false, false, true)
#undef DECLARE_POLICY_DISPATCH_EXTERN
