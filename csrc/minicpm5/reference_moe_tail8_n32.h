// SPDX-License-Identifier: Apache-2.0
#pragma once

inline void reference_moe_down_tail8_two_rows_n32(
    const uint32_t* input,
    const uint8_t* weight,
    uint32_t (&blocks)[2][16],
    int column,
    int lane,
    uint32_t mask,
    int live,
    const int* choices,
    float scale,
    scalar_t* output) {
  constexpr int Rows = 8, I = 512;
  const int row0 = sycl::ctz(mask);
  const uint32_t rest = mask & (mask - 1);
  const int row1 = rest ? sycl::ctz(rest) : row0;
  const int row = lane < 8 ? row0 : row1;
  Float8 out[2]{};
  for (int first = 0; first < I; first += 64) {
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const int k = first + t * 32;
      UInt4 a0{}, a1{};
      a0[0] = input[(k / 16 * Rows + row) * 8 + lane % 8];
      a1[0] = input[((k / 16 + 1) * Rows + row) * 8 + lane % 8];
#pragma unroll
      for (int col = 0; col < 2; ++col) {
        UInt8 even, odd;
        const auto* source = blocks[t] + col * 8;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          even[2 * p] = fp8_pair_rows<Rows>(source[p]);
          even[2 * p + 1] = fp8_pair_rows<Rows>(source[p] >> 16);
          odd[2 * p] = fp8_pair_rows<Rows>(source[4 + p]);
          odd[2 * p + 1] = fp8_pair_rows<Rows>(source[4 + p] >> 16);
        }
        dpas_rows<2>(out[col], a0, even);
        dpas_rows<2>(out[col], a1, odd);
      }
      if (k + 64 < I)
        reference_load_kn<32>(blocks[t], weight, H, I, column, k + 64);
    }
  }
#pragma unroll
  for (int r = 0; r < 2; ++r)
    if (r < live) {
      const int output_row = r == 0 ? row0 : row1;
#pragma unroll
      for (int col = 0; col < 2; ++col)
        output
            [(output_row * Top + choices[output_row]) * 64 + col * 16 + lane] =
                scalar_t(out[col][r] * scale);
    }
}

inline bool reference_moe_down_tail8_n32(
    sycl::nd_item<1> item,
    Weights weights,
    Workspace work,
    const float* routing,
    const int* experts,
    const int* choices,
    int count,
    scalar_t* local_output,
    float (&routed)[2],
    Step step,
    uint32_t call) {
  constexpr int Rows = 8, I = 512, ExpertStride = Subgroups / 2;
  const auto sg = item.get_sub_group();
  const int subgroup = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  const int tile_offset = subgroup % 2 * 32;
  const int column = item.get_group_linear_id() * 64 + tile_offset;
  const auto* activations =
      reinterpret_cast<const uint32_t*>(work.gate_up_partial);
  auto* tile_output = local_output + tile_offset;
  uint32_t blocks[2][16];
  const int initial = subgroup / 2;
  if (initial < count)
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const auto* weight = weights.w2 + size_t(experts[initial]) * I * H;
      reference_load_kn<32>(blocks[t], weight, H, I, column, t * 32);
    }
  if (!grid_wait<Groups * Local>(item, step.control, 9, call, step.health))
    return false;

  for (int u = initial; u < count; u += ExpertStride) {
    const int expert = experts[u];
    const auto* weight = weights.w2 + size_t(expert) * I * H;
    const auto* input = activations + size_t(u) * (I / 16) * Rows * 8;
    const int* row_choices = choices + u * Rows;
    const float scale = fp8_scale(weights.w2_scale[expert]);
    uint32_t mask = 0;
#pragma unroll
    for (int row = 0; row < Rows; ++row)
      if (row_choices[row] >= 0) mask |= 1u << row;
    const int live = sycl::popcount(mask);
    if (live <= 2) {
      reference_moe_down_tail8_two_rows_n32(
          input,
          weight,
          blocks,
          column,
          lane,
          mask,
          live,
          row_choices,
          scale,
          tile_output);
    } else {
      Float8 out[2]{};
      for (int first = 0; first < I; first += 64) {
#pragma unroll
        for (int t = 0; t < 2; ++t) {
          const int k = first + t * 32;
#pragma unroll
          for (int col = 0; col < 2; ++col)
            reference_mm_kn32<Rows>(
                out[col], blocks[t] + col * 8, input, k, lane);
          if (k + 64 < I)
            reference_load_kn<32>(blocks[t], weight, H, I, column, k + 64);
        }
      }
#pragma unroll
      for (int row = 0; row < Rows; ++row)
        if (row_choices[row] >= 0)
#pragma unroll
          for (int col = 0; col < 2; ++col)
            tile_output[(row * Top + row_choices[row]) * 64 + col * 16 + lane] =
                scalar_t(out[col][row] * scale);
    }
    if (u + ExpertStride < count)
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        const auto* next =
            weights.w2 + size_t(experts[u + ExpertStride]) * I * H;
        reference_load_kn<32>(blocks[t], next, H, I, column, t * 32);
      }
  }

  reference_wg_bar(item.get_group());
  const int lid = item.get_local_linear_id();
  if (lid < Rows * 32) {
    const int row = lid / 32, pair = lid % 32;
    float values[2]{};
    const bool live = step.slots[row] >= 0;
    for (int k = 0; k < Top && live; ++k)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        values[j] += routing[row * Top + k] *
                     float(local_output[(row * Top + k) * 64 + 2 * pair + j]);
#pragma unroll
    for (int j = 0; j < 2; ++j)
      routed[j] = float(scalar_t(values[j]));
  }
  reference_wg_bar(item.get_group());
  return true;
}
