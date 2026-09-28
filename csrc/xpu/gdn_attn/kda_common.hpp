#pragma once

#include <sycl/sycl.hpp>

#include "kda_gate.hpp"

// Constants and small helpers shared by the KDA causal-conv1d kernels
// (kda_causal_conv1d.hpp) and the recurrent gated-delta-rule backend
// (kda_recurrent_opt.hpp).

namespace kda {

// Both kernels are written against a 32-wide sub-group: the conv kernels map
// one lane per channel, and the recurrent kernel splits a head's key dimension
// across the lanes of one sub-group so its reductions stay intra-sub-group.
static constexpr int sub_group_size = 32;
static constexpr int work_group_size = 256;
// Value rows a single sub-group of the recurrent kernel keeps in registers.
static constexpr int values_per_sub_group = 4;
static constexpr int values_per_work_group =
    work_group_size / sub_group_size * values_per_sub_group;
static constexpr float l2norm_eps = 0.000001f;
// vLLM marks unused cache slots with -1; kernels must not read or write them.
static constexpr int pad_slot_id = -1;

inline float silu(float x) { return x / (1.0f + sycl::exp(-x)); }

using kda_gate::no_lower_bound;

}  // namespace kda
