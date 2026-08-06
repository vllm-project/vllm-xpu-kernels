#pragma once

#include <sycl/sycl.hpp>

// Shared KDA gate activation.
//
// KDA supports two gate parameterisations, selected per model by the HF config
// field `linear_attn_config.gate_lower_bound`:
//
//   softplus (unbounded, Kimi-Linear-48B-A3B, `gate_lower_bound` unset)
//     g = -exp(A_log[h]) * softplus(raw_gate + dt_bias)          in (-inf, 0]
//
//   sigmoid (bounded, Kimi-K3, `gate_lower_bound = -5.0`)
//     g = lower_bound * sigmoid(exp(A_log[h]) * (raw_gate + dt_bias))
//                                                                in (lb, 0)
//
// Both produce a non-positive log-domain decay, so the chunked prefill path can
// keep clamping the running cumsum at `g_floor` (max(x + d, floor) still
// composes because `d <= 0`).
//
// The mode is carried through the kernels as a single float: any value < 0 is a
// valid lower bound and selects the sigmoid gate, while `no_lower_bound` (0)
// selects softplus. This avoids a second flag on every kernel functor, and the
// branch is uniform across the whole launch.

namespace kda_gate {

// Sentinel for "no lower bound" -> softplus gate.
static constexpr float no_lower_bound = 0.0f;

inline bool use_lower_bound(float lower_bound) { return lower_bound < 0.0f; }

inline float softplus(float x) {
  return x < 20.0f ? sycl::log(1.0f + sycl::exp(x)) : x;
}

// Branch-free so the `prepare` cumsum unrolls cleanly: for x >= 20 the
// exponential saturates to +inf and log(1 + inf) = inf, which the select then
// discards, so no NaN can escape.
inline float native_softplus(float x) {
  const float saturated = sycl::native::log(1.0f + sycl::native::exp(x));
  return x < 20.0f ? saturated : x;
}

inline float sigmoid(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

inline float native_sigmoid(float x) {
  return sycl::native::recip(1.0f + sycl::native::exp(-x));
}

// `x` is `raw_gate + dt_bias`, `head_a` is `-exp(A_log[head])` (so `-head_a` is
// the positive decay coefficient the sigmoid form scales its input by).
inline float log_gate(float x, float head_a, float lower_bound) {
  return use_lower_bound(lower_bound) ? lower_bound * sigmoid(-head_a * x)
                                      : head_a * softplus(x);
}

// Same, built from the hardware-accelerated transcendentals. The result feeds
// exp(g) and is consumed at activation precision, so the extra ulps of the
// native ops are irrelevant here.
inline float native_log_gate(float x, float head_a, float lower_bound) {
  return use_lower_bound(lower_bound)
             ? lower_bound * native_sigmoid(-head_a * x)
             : head_a * native_softplus(x);
}

}  // namespace kda_gate
