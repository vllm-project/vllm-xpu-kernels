// SPDX-License-Identifier: Apache-2.0
#pragma once

// SwiGLU limit folded into the fused silu_and_mul + quant kernels. Matches the
// eager reference on the activation tensor:
//   gate.clamp_(max=limit); up.clamp_(min=-limit, max=limit)

#include <c10/util/Exception.h>

#include <cmath>
#include <limits>
#include <optional>
#include <type_traits>

namespace vllm {

// Clamping against +inf is a no-op, so +inf marks the limit as disabled. A
// positive limit that rounds to 0 in the activation dtype stays enabled and
// clamps to 0, like the eager clamp_.
inline constexpr float kSwigluLimitDisabled =
    std::numeric_limits<float>::infinity();

// Validated limit. None or 0 disables it, following vLLM's swiglu_limit
// convention.
inline float swiglu_limit_value(const std::optional<double>& swiglu_limit) {
  if (!swiglu_limit.has_value()) return kSwigluLimitDisabled;
  const double limit = *swiglu_limit;
  TORCH_CHECK(
      limit >= 0.0 && !std::isnan(limit),
      "swiglu_limit must be non-negative, got ",
      limit);
  return limit == 0.0 ? kSwigluLimitDisabled : static_cast<float>(limit);
}

// The eager clamp_ writes scalar_t(limit), so clamp against the limit rounded
// to the activation dtype to stay bitwise identical for any limit. A limit
// that overflows scalar_t becomes +inf, a no-op, as it is for clamp_.
template <typename scalar_t>
inline float swiglu_limit_as(float limit) {
  return limit < kSwigluLimitDisabled
             ? static_cast<float>(static_cast<scalar_t>(limit))
             : kSwigluLimitDisabled;
}

// NaN-propagating, like torch.clamp. Clamping against kSwigluLimitDisabled
// (+inf) is a no-op, so callers may skip these or apply them unconditionally.
inline float swiglu_clamp_gate(float g, float limit) {
  return g > limit ? limit : g;
}

inline float swiglu_clamp_up(float u, float limit) {
  return u > limit ? limit : (u < -limit ? -limit : u);
}

// Calls fn(std::true_type) when the limit is enabled, else fn(std::false_type),
// so kernels can drop the clamp at compile time on the default path.
template <typename Fn>
inline void dispatch_swiglu_limit(float limit, Fn&& fn) {
  if (limit < kSwigluLimitDisabled) {
    fn(std::true_type{});
  } else {
    fn(std::false_type{});
  }
}

}  // namespace vllm
