// SPDX-License-Identifier: Apache-2.0
#pragma once

// Adapt the reference PRT's K-sliced router to the post-norm activations.
// Accumulate the projection and apply sigmoid in FP32.
inline void reference_b1_router_preload(
    sycl::nd_item<1> item, Weights weights, sycl::uint4 (&raw)[4]) {
  const auto sg = item.get_sub_group();
  const int s = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
  if (s * 8 < E)
#pragma unroll
    for (int t = 0; t < 4; ++t)
      raw[t] = *reinterpret_cast<const sycl::uint4*>(
          weights.router + size_t(s * 8 + 2 * t + (lane >> 3)) * H +
          item.get_group_linear_id() * 64 + (lane & 7) * 8);
}

inline void reference_b1_router_partial(
    sycl::nd_item<1> item,
    Workspace work,
    const scalar_t* input,
    const sycl::uint4 (&raw)[4]) {
  const auto sg = item.get_sub_group();
  const int s = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
  const int wg = item.get_group_linear_id();
  if (s * 8 < E) {
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const auto wh = sycl::bit_cast<sycl::vec<uint16_t, 8>>(raw[t]);
      float value = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        value += float(sycl::bit_cast<scalar_t>(wh[i])) *
                 float(input[wg * 64 + (lane & 7) * 8 + i]);
      value += sycl::permute_group_by_xor(sg, value, 4);
      value += sycl::permute_group_by_xor(sg, value, 2);
      value += sycl::permute_group_by_xor(sg, value, 1);
      if ((lane & 7) == 0)
        work.attention_partial[wg * E + s * 8 + 2 * t + (lane >> 3)] = value;
    }
  }
  reference_wg_bar(item.get_group());
}
template <int Rows>
inline void reference_router_preload(
    sycl::nd_item<1> item, Weights weights, UInt8 (&first)[2]) {
  const int wg = item.get_group_linear_id();
  const int expert = item.get_local_linear_id();
  if (expert < E)
#pragma unroll
    for (int chunk = 0; chunk < 2; ++chunk)
      first[chunk] = *reinterpret_cast<const UInt8*>(
          weights.router + expert * H + wg * 64 + chunk * 16);
}

template <int Rows>
inline void reference_router_partial(
    sycl::nd_item<1> item,
    Weights weights,
    Workspace work,
    const scalar_t* input,
    uint32_t* packed,
    const UInt8 (&first)[2]) {
  static_assert(H / Groups == 64 && E % 16 == 0);
  const int wg = item.get_group_linear_id();
  const auto sg = item.get_sub_group();
  const int s = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
  const int expert = s * 16 + lane;
  UInt8 registers[4];
  registers[0] = first[0];
  registers[1] = first[1];
  if (expert < E)
#pragma unroll
    for (int chunk = 2; chunk < 4; ++chunk)
      registers[chunk] = *reinterpret_cast<const UInt8*>(
          weights.router + expert * H + wg * 64 + chunk * 16);
  pack_projection_input<Rows, 64>(item, input + wg * 64, H, packed);
  reference_wg_bar(item.get_group());
  if (expert < E) {
    Float8 values{};
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk)
      dpas_rows<Rows>(
          values, projection_a<Rows>(packed, chunk, lane), registers[chunk]);
#pragma unroll
    for (int r = 0; r < Rows; ++r)
      // All attention readers completed the phase4 handoff before AR1.
      // Keep these partials separate from the later routed activations so
      // every WG can finish/select locally without another global handoff.
      work.attention_partial[(wg * Rows + r) * E + expert] = values[r];
  }
  reference_wg_bar(item.get_group());
}

template <
    int Rows,
    bool B1Unroll = false,
    bool OwnerRow = false,
    bool IgnoreInactive = false>
inline bool reference_router_finish(
    sycl::nd_item<1> item,
    Weights weights,
    Workspace work,
    Step step,
    uint32_t call,
    float* local_scores) {
  bool finite = true;
  for (int index = item.get_local_linear_id(); index < Rows * E;
       index += Local) {
    const int r = index / E, e = index % E;
    if constexpr (OwnerRow)
      if (item.get_group_linear_id() != r) continue;
    if constexpr (IgnoreInactive) {
      // A stock Graph producer need only fill live query rows.
      if (step.slots[r] < 0) {
        local_scores[index] = 0.5f;
        if (OwnerRow || item.get_group_linear_id() == 0)
          work.router[index] = 0.5f;
        continue;
      }
    }
    float value = 0.f;
    if constexpr (B1Unroll) {
      // mono_layers.hpp tail S5': issue the fixed-order WG reduction as
      // a fully unrolled sequence.
#pragma unroll
      for (int wg = 0; wg < Groups; ++wg)
        value += work.attention_partial[(wg * Rows + r) * E + e];
    } else {
#pragma unroll 1
      for (int wg = 0; wg < Groups; ++wg)
        value += work.attention_partial[(wg * Rows + r) * E + e];
    }
    if (!sycl::isfinite(value) || !sycl::isfinite(weights.bias[e])) {
      finite = false;
      fail(step.control, 4, 7, call, index, step.health);
    }
    const float score = 1.f / (1.f + sycl::exp(-value));
    if (OwnerRow || item.get_group_linear_id() == 0) work.router[index] = score;
    local_scores[index] = score;
  }
  // The routing subgroup consumes this WG's completed rows from SLM.
  reference_wg_bar(item.get_group());
  return sycl::all_of_group(item.get_group(), finite);
}

constexpr int RouterSortSize = 256;

inline void reference_router_select_row(
    sycl::nd_item<1> item,
    Weights weights,
    Workspace work,
    Step step,
    uint32_t call,
    const float* scores,
    uint64_t* exchange) {
  static_assert(Top == 16 && E <= RouterSortSize && RouterSortSize <= Local);
  const int expert = item.get_local_linear_id();
  const int row = item.get_group_linear_id();
  const auto sg = item.get_sub_group();
  uint64_t key = 0;
  if (expert < E) {
    float value = scores[expert] + weights.bias[expert];
    if (value == 0.f) value = 0.f;
    const uint32_t bits = sycl::bit_cast<uint32_t>(value);
    const uint32_t ordered = bits ^ (bits >> 31 ? 0xffffffffu : 0x80000000u);
    key = (uint64_t(ordered) << 32) | uint32_t(E - expert);
  }
  int buffer = 0;
#pragma unroll
  for (int width = 2; width <= RouterSortSize; width *= 2) {
#pragma unroll
    for (int stride = width / 2; stride != 0; stride /= 2) {
      uint64_t partner = 0;
      if (stride < 16) {
        partner = sycl::permute_group_by_xor(sg, key, stride);
      } else {
        if (expert < RouterSortSize) exchange[buffer + expert] = key;
        reference_wg_bar(item.get_group());
        if (expert < RouterSortSize)
          partner = exchange[buffer + (expert ^ stride)];
        buffer ^= RouterSortSize;
      }
      const bool larger = ((expert & width) == 0) == ((expert & stride) == 0);
      key = larger ? sycl::max(key, partner) : sycl::min(key, partner);
    }
  }
  if (expert < Top) {
    const int chosen = E - int(uint32_t(key));
    const float score = scores[chosen];
    float total = 0.f;
#pragma unroll
    for (int k = 0; k < Top; ++k)
      total += sycl::select_from_group(sg, score, k);
    if (!(total > 0.f) || !sycl::isfinite(total)) {
      if (expert == 0) fail(step.control, 4, 7, call, row, step.health);
      return;
    }
    work.experts[row * Top + expert] = chosen;
    work.routing[row * Top + expert] = score * (3.66f / total);
  }
}
