#pragma once

#include "chunk_prefill.hpp"

// Prevent implicit instantiation of policy_dispatch in translation units
// that include chunk_prefill.hpp. Each specialization is explicitly
// instantiated in its own .cpp file.

// Macro to generate extern template declarations
#define DECLARE_POLICY_DISPATCH_EXTERN(POLICY, PAGED, CAUSAL, LOCAL, SINK)  \
  extern template void policy_dispatch<POLICY, PAGED, CAUSAL, LOCAL, SINK>( \
      sycl::queue & queue,                                                  \
      CutlassType cuType,                                                   \
      const chunk_prefill_args_t& args);

// chunk_policy_head64
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head64, false, false, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head64, false, false, false, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head64, false, false, true, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head64, false, false, true, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head64, false, true, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head64, false, true, false, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head64, false, true, true, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head64, false, true, true, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head64, true, false, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head64, true, false, false, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head64, true, false, true, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head64, true, false, true, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head64, true, true, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head64, true, true, false, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head64, true, true, true, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head64, true, true, true, true)

// chunk_policy_head96
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head96, false, false, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head96, false, false, false, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head96, false, false, true, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head96, false, false, true, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head96, false, true, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head96, false, true, false, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head96, false, true, true, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head96, false, true, true, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head96, true, false, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head96, true, false, false, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head96, true, false, true, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head96, true, false, true, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head96, true, true, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head96, true, true, false, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head96, true, true, true, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head96, true, true, true, true)

// chunk_policy_head128
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head128, false, false, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head128, false, false, false, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head128, false, false, true, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head128, false, false, true, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head128, false, true, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head128, false, true, false, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head128, false, true, true, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head128, false, true, true, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head128, true, false, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head128, true, false, false, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head128, true, false, true, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head128, true, false, true, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head128, true, true, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head128, true, true, false, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head128, true, true, true, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head128, true, true, true, true)

// chunk_policy_head192
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head192, false, false, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head192, false, false, false, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head192, false, false, true, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head192, false, false, true, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head192, false, true, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head192, false, true, false, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head192, false, true, true, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head192, false, true, true, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head192, true, false, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head192, true, false, false, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head192, true, false, true, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head192, true, false, true, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head192, true, true, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head192, true, true, false, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head192, true, true, true, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head192, true, true, true, true)

// chunk_policy_head256
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head256, false, false, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head256, false, false, false, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head256, false, false, true, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head256, false, false, true, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head256, false, true, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head256, false, true, false, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head256, false, true, true, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head256, false, true, true, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head256, true, false, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head256, true, false, false, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head256, true, false, true, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head256, true, false, true, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head256, true, true, false, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head256, true, true, false, true)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head256, true, true, true, false)
DECLARE_POLICY_DISPATCH_EXTERN(chunk_policy_head256, true, true, true, true)

#undef DECLARE_POLICY_DISPATCH_EXTERN
