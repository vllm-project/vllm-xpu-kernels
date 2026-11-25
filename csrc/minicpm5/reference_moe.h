// SPDX-License-Identifier: Apache-2.0
#pragma once

// Adapted from mono_layers_b.hpp expert union, S6 and S7 block-load pipeline,
// revision 922432bc4288f7a323cd550b2599fb318d4568a9. MiniCPM retains BF16
// gate/up/down rounding and applies routing weights after each expert down.

#ifdef __SYCL_DEVICE_ONLY__
extern "C" SYCL_EXTERNAL void
_Z51intel_sub_group_2d_block_read_transform_8b_32r16x2cPU3AS1viiiDv2_iPj(
    __attribute__((opencl_global)) void*,
    int,
    int,
    int,
    Int2,
    __attribute__((opencl_private)) unsigned int*);
extern "C" SYCL_EXTERNAL void
_Z51intel_sub_group_2d_block_read_transform_8b_32r16x4cPU3AS1viiiDv2_iPj(
    __attribute__((opencl_global)) void*,
    int,
    int,
    int,
    Int2,
    __attribute__((opencl_private)) unsigned int*);
#endif

template <int Columns>
inline void reference_load_kn(
    uint32_t (&out)[Columns / 2],
    const uint8_t* weight,
    int n,
    int k,
    int col,
    int row) {
  static_assert(Columns == 32 || Columns == 64);
#ifdef __SYCL_DEVICE_ONLY__
  if constexpr (Columns == 32)
    _Z51intel_sub_group_2d_block_read_transform_8b_32r16x2cPU3AS1viiiDv2_iPj(
        (__attribute__((opencl_global)) void*)weight,
        n,
        k,
        n,
        Int2{col, row},
        (__attribute__((opencl_private)) unsigned int*)out);
  else
    _Z51intel_sub_group_2d_block_read_transform_8b_32r16x4cPU3AS1viiiDv2_iPj(
        (__attribute__((opencl_global)) void*)weight,
        n,
        k,
        n,
        Int2{col, row},
        (__attribute__((opencl_private)) unsigned int*)out);
#endif
}

template <int Rows>
inline void reference_mm_kn32(
    Float8& result,
    const uint32_t* weight,
    const uint32_t* packed,
    int k,
    int lane) {
  UInt8 even, odd;
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    even[2 * p] = fp8_pair_rows<Rows>(weight[p]);
    even[2 * p + 1] = fp8_pair_rows<Rows>(weight[p] >> 16);
    odd[2 * p] = fp8_pair_rows<Rows>(weight[4 + p]);
    odd[2 * p + 1] = fp8_pair_rows<Rows>(weight[4 + p] >> 16);
  }
  dpas_rows<Rows>(result, projection_a<Rows>(packed, k / 16, lane), even);
  dpas_rows<Rows>(result, projection_a<Rows>(packed, k / 16 + 1, lane), odd);
}

template <int Rows, bool Preselected = false, bool IgnorePadding = false>
inline void reference_expert_union(
    sycl::nd_item<1> item,
    Workspace work,
    int* selected,
    int* experts,
    int* choices,
    int* count,
    const int64_t* slots) {
  const int lid = item.get_local_linear_id();
  auto sg = item.get_sub_group();
  const int lane = sg.get_local_linear_id();
  if constexpr (!Preselected)
    if (lid < Rows * Top) selected[lid] = work.experts[lid];
  reference_wg_bar(item.get_group());
  if (sg.get_group_linear_id() == 0) {
    uint32_t mask = 0;
#pragma unroll
    for (int j = 0; j < Rows * Top; ++j)
      if ((!IgnorePadding || slots[j / Top] >= 0) && selected[j] / 16 == lane)
        mask |= 1u << (selected[j] % 16);
    const int size = sycl::popcount(mask);
    int pos = sycl::exclusive_scan_over_group(sg, size, sycl::plus<int>());
    if (lane == 15) count[0] = pos + size;
#pragma unroll
    for (int bit = 0; bit < 16; ++bit)
      if ((mask >> bit) & 1u) experts[pos++] = lane * 16 + bit;
  }
  reference_wg_bar(item.get_group());
  for (int index = lid; index < count[0] * Rows; index += Local) {
    const int u = index / Rows, r = index % Rows;
    int choice = -1;
#pragma unroll
    for (int k = 0; k < Top; ++k)
      if ((!IgnorePadding || slots[r] >= 0) &&
          selected[r * Top + k] == experts[u])
        choice = k;
    choices[index] = choice;
  }
  reference_wg_bar(item.get_group());
}

template <int TP, int Rows>
constexpr bool WideMoeGateUp = TP == 1 && Rows > 2;

#ifdef MINICPM5_MONO_FP16
template <int Rows, bool IgnorePadding>
inline void reference_expert_union_indexed(
    sycl::nd_item<1> item,
    const int* selected,
    int* experts,
    int* choices,
    int* count,
    const int64_t* slots,
    int* row_choices) {
  static_assert(E % 16 == 0 && E <= Local && E / 16 <= 16);
  const int lid = item.get_local_linear_id();
  auto sg = item.get_sub_group();
  const int s = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
  // Index each row by expert ID, then compact the active experts in ID order.
  int* subgroup_offsets = row_choices + Rows * E;
  for (int i = lid; i < Rows * E; i += Local)
    row_choices[i] = -1;
  reference_wg_bar(item.get_group());
  if (lid < Rows * Top) {
    const int row = lid / Top;
    if (!IgnorePadding || slots[row] >= 0)
      row_choices[row * E + selected[lid]] = lid % Top;
  }
  reference_wg_bar(item.get_group());
  int selected_choices[Rows];
  bool active = false;
  if (lid < E) {
  #pragma unroll
    for (int row = 0; row < Rows; ++row) {
      selected_choices[row] = row_choices[row * E + lid];
      active |= selected_choices[row] >= 0;
    }
  }
  const int pos =
      sycl::exclusive_scan_over_group(sg, int(active), sycl::plus<int>());
  if (lane == 15 && s < E / 16) subgroup_offsets[s] = pos + int(active);
  reference_wg_bar(item.get_group());
  if (s == 0) {
    const int size = lane < E / 16 ? subgroup_offsets[lane] : 0;
    const int offset =
        sycl::exclusive_scan_over_group(sg, size, sycl::plus<int>());
    if (lane < E / 16) subgroup_offsets[lane] = offset;
    if (lane == 15) count[0] = offset + size;
  }
  reference_wg_bar(item.get_group());
  if (active) {
    const int index = subgroup_offsets[s] + pos;
    experts[index] = lid;
  #pragma unroll
    for (int row = 0; row < Rows; ++row)
      choices[index * Rows + row] = selected_choices[row];
  }
  reference_wg_bar(item.get_group());
}
#endif

template <int TP, int Rows, int TileColumns = 64, bool AlternatePartial = false>
inline void reference_moe_gateup_wide(
    sycl::nd_item<1> item,
    Weights weights,
    Workspace work,
    const scalar_t* input,
    uint32_t* packed,
    float* partial,
    const int* experts,
    const int* choices,
    int count) {
  static_assert(WideMoeGateUp<TP, Rows>);
  constexpr int I = 512 / TP;
#ifdef MINICPM5_MONO_FP16
  constexpr int KParts = Rows <= 2 || (TP == 1 && Rows == 8) ? 8 : 16;
#else
  constexpr int KParts = Rows <= 2 ? 8 : 16;
#endif
  constexpr int Span = H / KParts, Blocks = Span / 32;
  static_assert(
      TileColumns == 64 || (TileColumns == 32 && TP == 1 && Rows == 8));
  constexpr int Outputs = TileColumns / 16;
  constexpr int PartialWords = Subgroups * Outputs * Rows * 16;
  static_assert(
      !AlternatePartial || (TP == 1 && Rows == 4 && TileColumns == 64));
  constexpr int TileGroups = Subgroups / (2 * KParts);
  constexpr int GroupColumns = TileGroups * TileColumns,
                Columns = I / GroupColumns;
  static_assert(Subgroups % (2 * KParts) == 0 && I % GroupColumns == 0);
  const int wg = item.get_group_linear_id(), lid = item.get_local_linear_id();
  auto sg = item.get_sub_group();
  const int s = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
  const int tile_group = s / (2 * KParts), gu = (s / KParts) % 2,
            ks = s % KParts;
  pack_projection_input<Rows, H>(item, input, H, packed);
  reference_wg_bar(item.get_group());
  auto* activations = reinterpret_cast<uint32_t*>(work.gate_up_partial);
  const int tasks = count * Columns;
  auto source = [&](int task) {
    return weights.w13 + size_t(experts[task / Columns]) * H * 2 * I;
  };
  auto column = [&](int task) {
    return gu * I + task % Columns * GroupColumns + tile_group * TileColumns;
  };
  uint32_t registers[2][TileColumns / 2];
  if (wg < tasks)
#pragma unroll
    for (int k = 0; k < 2; ++k)
      reference_load_kn<TileColumns>(
          registers[k], source(wg), 2 * I, H, column(wg), ks * Span + k * 32);
  for (int task = wg; task < tasks; task += Groups) {
    const int u = task / Columns, n0 = task % Columns * GroupColumns,
              e = experts[u];
    float* task_partial = partial;
    if constexpr (AlternatePartial)
      task_partial += ((task / Groups) & 1) * PartialWords;
    Float8 out[Outputs]{};
#pragma unroll 1
    for (int block = 0; block < Blocks; block += 2) {
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int offset = block + k;
#pragma unroll
        for (int col = 0; col < Outputs; ++col)
          reference_mm_kn32<Rows>(
              out[col],
              registers[k] + col * 8,
              packed,
              ks * Span + offset * 32,
              lane);
        if (offset + 2 < Blocks)
          reference_load_kn<TileColumns>(
              registers[k],
              source(task),
              2 * I,
              H,
              column(task),
              ks * Span + (offset + 2) * 32);
        else if (task + Groups < tasks)
          reference_load_kn<TileColumns>(
              registers[k],
              source(task + Groups),
              2 * I,
              H,
              column(task + Groups),
              ks * Span + (offset - (Blocks - 2)) * 32);
      }
    }
#pragma unroll
    for (int tile = 0; tile < Outputs; ++tile)
#pragma unroll
      for (int r = 0; r < Rows; ++r)
        task_partial[((s * Outputs + tile) * Rows + r) * 16 + lane] =
            out[tile][r];
    reference_wg_bar(item.get_group());
    if (lid < Rows * (GroupColumns / 2)) {
      const int r = lid / (GroupColumns / 2),
                col = lid % (GroupColumns / 2) * 2;
      scalar_t value[2] = {scalar_t(0.f), scalar_t(0.f)};
      if (r < Rows && choices[u * Rows + r] >= 0) {
        float gates[2]{}, ups[2]{};
        if constexpr (WideMoeGateUp<TP, Rows>) {
#pragma unroll
          for (int k = 0; k < KParts; ++k)
#pragma unroll
            for (int half = 0; half < 2; ++half) {
              const int c = (col + half) % TileColumns;
              const int group = (col + half) / TileColumns * (2 * KParts);
              gates[half] += task_partial
                  [(((group + k) * Outputs + c / 16) * Rows + r) * 16 + c % 16];
              ups[half] += task_partial
                  [(((group + KParts + k) * Outputs + c / 16) * Rows + r) * 16 +
                   c % 16];
            }
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          float g = 0.f, up = 0.f;
          if constexpr (WideMoeGateUp<TP, Rows>) {
            g = gates[half];
            up = ups[half];
          } else {
            const int c = (col + half) % TileColumns;
            const int group = (col + half) / TileColumns * (2 * KParts);
#pragma unroll
            for (int k = 0; k < KParts; ++k) {
              g += task_partial
                  [(((group + k) * Outputs + c / 16) * Rows + r) * 16 + c % 16];
              up += task_partial
                  [(((group + KParts + k) * Outputs + c / 16) * Rows + r) * 16 +
                   c % 16];
            }
          }
          g = float(scalar_t(g * fp8_scale(weights.w13_scale[e])));
          up = float(scalar_t(up * fp8_scale(weights.w13_scale[e])));
          value[half] = scalar_t(g / (1.f + sycl::exp(-g)) * up);
        }
      }
      const int c = n0 + col;
      const size_t index = size_t(u) * (I / 16) * Rows * 8 +
                           (c / 16 * Rows + r) * 8 + c % 16 / 2;
      activations[index] = pack16(value[0], value[1]);
    }
    // Task t+1 publishes through its producer barrier before task t+2
    // can reuse the plane read by task t. Phase 9 publishes the final GU pack.
    if constexpr (!AlternatePartial) reference_wg_bar(item.get_group());
  }
}

template <int TP, int Rows, bool AlternatePartial = false>
inline void reference_moe_gateup(
    sycl::nd_item<1> item,
    Weights weights,
    Workspace work,
    uint32_t* packed,
    float* partial,
    const int* experts,
    const int* choices,
    int count) {
  constexpr int I = 512 / TP;
  constexpr int KParts = Rows <= 2 ? 8 : 16;
  constexpr int Span = H / KParts, Blocks = Span / 32;
  constexpr int TileColumns = Rows <= 2 ? 64 : 32;
  constexpr int Outputs = TileColumns / 16;
  constexpr int TileGroups = Subgroups / (2 * KParts);
  constexpr int GroupColumns = TileGroups * TileColumns,
                Columns = I / GroupColumns;
  constexpr int PartialWords = Subgroups * Outputs * Rows * 16;
  static_assert(Subgroups % (2 * KParts) == 0 && I % GroupColumns == 0);
  static_assert(
      !AlternatePartial ||
      (TP == 2 && Rows == 8 &&
       PartialWords * sizeof(float) == Rows * H * sizeof(scalar_t)));
  const int wg = item.get_group_linear_id(), lid = item.get_local_linear_id();
  auto sg = item.get_sub_group();
  const int s = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
  const int tile_group = s / (2 * KParts), gu = (s / KParts) % 2,
            ks = s % KParts;
  pack_projection_input<Rows, H>(item, work.normed, H, packed);
  reference_wg_bar(item.get_group());
  auto* activations = reinterpret_cast<uint32_t*>(work.gate_up_partial);
  const int tasks = count * Columns;
  auto source = [&](int task) {
    return weights.w13 + size_t(experts[task / Columns]) * H * 2 * I;
  };
  auto column = [&](int task) {
    return gu * I + task % Columns * GroupColumns + tile_group * TileColumns;
  };
  uint32_t registers[2][TileColumns / 2];
  if (wg < tasks)
#pragma unroll
    for (int k = 0; k < 2; ++k)
      reference_load_kn<TileColumns>(
          registers[k], source(wg), 2 * I, H, column(wg), ks * Span + k * 32);
  for (int task = wg; task < tasks; task += Groups) {
    const int u = task / Columns, n0 = task % Columns * GroupColumns,
              e = experts[u];
    float* task_partial = partial;
    if constexpr (AlternatePartial)
      task_partial += ((task / Groups) & 1) * PartialWords;
    Float8 out[Outputs]{};
#pragma unroll 1
    for (int block = 0; block < Blocks; block += 2) {
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int offset = block + k;
#pragma unroll
        for (int col = 0; col < Outputs; ++col)
          reference_mm_kn32<Rows>(
              out[col],
              registers[k] + col * 8,
              packed,
              ks * Span + offset * 32,
              lane);
        if (offset + 2 < Blocks)
          reference_load_kn<TileColumns>(
              registers[k],
              source(task),
              2 * I,
              H,
              column(task),
              ks * Span + (offset + 2) * 32);
        else if (task + Groups < tasks)
          reference_load_kn<TileColumns>(
              registers[k],
              source(task + Groups),
              2 * I,
              H,
              column(task + Groups),
              ks * Span + (offset - (Blocks - 2)) * 32);
      }
    }
#pragma unroll
    for (int tile = 0; tile < Outputs; ++tile)
#pragma unroll
      for (int r = 0; r < Rows; ++r)
        task_partial[((s * Outputs + tile) * Rows + r) * 16 + lane] =
            out[tile][r];
    reference_wg_bar(item.get_group());
    if (lid < Rows * (GroupColumns / 2)) {
      const int r = lid / (GroupColumns / 2),
                col = lid % (GroupColumns / 2) * 2;
      scalar_t value[2] = {scalar_t(0.f), scalar_t(0.f)};
      if (r < Rows && choices[u * Rows + r] >= 0) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int c = (col + half) % TileColumns;
          const int group = (col + half) / TileColumns * (2 * KParts);
          float g = 0.f, up = 0.f;
#pragma unroll
          for (int k = 0; k < KParts; ++k) {
            g += task_partial
                [(((group + k) * Outputs + c / 16) * Rows + r) * 16 + c % 16];
            up += task_partial
                [(((group + KParts + k) * Outputs + c / 16) * Rows + r) * 16 +
                 c % 16];
          }
          g = float(scalar_t(g * fp8_scale(weights.w13_scale[e])));
          up = float(scalar_t(up * fp8_scale(weights.w13_scale[e])));
          value[half] = scalar_t(g / (1.f + sycl::exp(-g)) * up);
        }
      }
      const int c = n0 + col;
      const size_t index = size_t(u) * (I / 16) * Rows * 8 +
                           (c / 16 * Rows + r) * 8 + c % 16 / 2;
      activations[index] = pack16(value[0], value[1]);
    }
    // The producer barrier of task t+1 closes task t's reads before t+2
    // reuses that plane. Phase 9 still releases the final global activation.
    if constexpr (!AlternatePartial) reference_wg_bar(item.get_group());
  }
}

template <int TP, int Rows>
inline void reference_moe_down_two_rows(
    Weights weights,
    const uint32_t* input,
    uint32_t (&blocks)[2][32],
    int expert,
    int column,
    int lane,
    int row0,
    int row1,
    int live,
    const int* choices,
    scalar_t* local_output) {
  constexpr int I = 512 / TP;
  const auto* weight = weights.w2 + size_t(expert) * I * H;
  Float8 out[4]{};
  for (int first = 0; first < I; first += 64) {
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const int k = first + t * 32;
      UInt4 a0{}, a1{};
      const int row = lane < 8 ? row0 : row1;
      a0[0] = input[(k / 16 * Rows + row) * 8 + lane % 8];
      a1[0] = input[((k / 16 + 1) * Rows + row) * 8 + lane % 8];
#pragma unroll
      for (int col = 0; col < 4; ++col) {
        UInt8 even, odd;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          const auto* source = blocks[t] + col * 8;
          even[2 * p] = fp8_pair_rows<Rows>(source[p]);
          even[2 * p + 1] = fp8_pair_rows<Rows>(source[p] >> 16);
          odd[2 * p] = fp8_pair_rows<Rows>(source[4 + p]);
          odd[2 * p + 1] = fp8_pair_rows<Rows>(source[4 + p] >> 16);
        }
        dpas_rows<2>(out[col], a0, even);
        dpas_rows<2>(out[col], a1, odd);
      }
      if (k + 64 < I)
        reference_load_kn<64>(blocks[t], weight, H, I, column, k + 64);
    }
  }
  // This only compacts independent DPAS rows. Keep the original K order,
  // scale, per-expert rounding and original row/top-k output locations.
#pragma unroll
  for (int r = 0; r < 2; ++r)
    if (r < live) {
      const int row = r == 0 ? row0 : row1;
#pragma unroll
      for (int col = 0; col < 4; ++col)
        local_output[(row * Top + choices[row]) * 64 + col * 16 + lane] =
            scalar_t(out[col][r] * fp8_scale(weights.w2_scale[expert]));
    }
}

template <int TP, int Rows, bool IgnorePadding = false>
inline bool reference_moe_down(
    sycl::nd_item<1> item,
    Weights weights,
    Workspace work,
    const float* routing,
    const int* experts,
    const int* choices,
    int count,
    scalar_t* local_output,
    float (&routed)[2],
    float* partial,
    Step step,
    uint32_t call) {
  constexpr int I = 512 / TP;
  static_assert(Top * 64 <= H);
  const int wg = item.get_group_linear_id();
  auto sg = item.get_sub_group();
  const int s = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
  const auto* activations =
      reinterpret_cast<const uint32_t*>(work.gate_up_partial);
  constexpr int Parts = Rows == 1 ? 2 : 1;
  constexpr int Span = I / Parts;
  const int initial = s / Parts, begin = (s % Parts) * Span;
#if defined(MINICPM5_MONO_FP16) || defined(MINICPM5_MONO_BF16_FP8_HALF_ENABLED)
  constexpr int Buffers = 2;
#else
  constexpr int Buffers = 4;
#endif
  uint32_t blocks[Buffers][32];
  if (initial < count)
#pragma unroll
    for (int t = 0; t < 2; ++t)
      reference_load_kn<64>(
          blocks[t],
          weights.w2 + size_t(experts[initial]) * I * H,
          H,
          I,
          wg * 64,
          begin + t * 32);
#ifdef MINICPM5_MONO_FP16
  if (!grid_wait < Groups * Local,
      TP == 2 > (item, step.control, 9, call, step.health))
    return false;
#else
  if (!grid_wait < Groups * Local,
      TP == 2 && Rows == 1 > (item, step.control, 9, call, step.health))
    return false;
#endif
  if constexpr (Rows == 1) {
    // Two K partitions keep all 32 subgroups active for the 16 experts.
    // Merge the FP32 partials before the canonical per-expert BF16 rounding.
    const int e = experts[initial], n0 = wg * 64;
    const auto* weight = weights.w2 + size_t(e) * I * H;
    const auto* input = activations + size_t(initial) * (I / 16) * Rows * 8;
    Float8 out[4]{};
#pragma unroll
    for (int t = 2; t < Buffers; ++t)
      reference_load_kn<64>(blocks[t], weight, H, I, n0, begin + t * 32);
    for (int first = begin; first < begin + Span; first += 32 * Buffers) {
#pragma unroll
      for (int t = 0; t < Buffers; ++t) {
        const int k = first + t * 32;
#pragma unroll
        for (int col = 0; col < 4; ++col)
          reference_mm_kn32<Rows>(
              out[col], blocks[t] + col * 8, input, k, lane);
        if (k + 32 * Buffers < begin + Span)
          reference_load_kn<64>(blocks[t], weight, H, I, n0, k + 32 * Buffers);
      }
    }
#pragma unroll
    for (int col = 0; col < 4; ++col)
      partial[s * 64 + col * 16 + lane] = out[col][0];
    reference_wg_bar(item.get_group());
    for (int index = item.get_local_linear_id(); index < Top * 64;
         index += Local) {
      const int u = index / 64, col = index % 64;
      const float value =
          partial[2 * u * 64 + col] + partial[(2 * u + 1) * 64 + col];
      local_output[choices[u] * 64 + col] =
          scalar_t(value * fp8_scale(weights.w2_scale[experts[u]]));
    }
  } else {
    for (int u = s; u < count; u += Subgroups) {
      const int e = experts[u], n0 = wg * 64;
      const auto* weight = weights.w2 + size_t(e) * I * H;
      const auto* input = activations + size_t(u) * (I / 16) * Rows * 8;
#ifdef MINICPM5_MONO_FP16
      if constexpr (Rows > 4) {
        uint32_t mask = 0;
  #pragma unroll
        for (int r = 0; r < Rows; ++r)
          if (choices[u * Rows + r] >= 0) mask |= 1u << r;
        const int live = sycl::popcount(mask);
        if (live <= 2) {
          const int row0 = sycl::ctz(mask);
          const uint32_t rest = mask & (mask - 1);
          const int row1 = rest ? sycl::ctz(rest) : row0;
          reference_moe_down_two_rows<TP, Rows>(
              weights,
              input,
              blocks,
              e,
              n0,
              lane,
              row0,
              row1,
              live,
              choices + u * Rows,
              local_output);
          if (u + Subgroups < count)
  #pragma unroll
            for (int t = 0; t < 2; ++t)
              reference_load_kn<64>(
                  blocks[t],
                  weights.w2 + size_t(experts[u + Subgroups]) * I * H,
                  H,
                  I,
                  n0,
                  t * 32);
          continue;
        }
      }
#endif
      Float8 out[4]{};
#pragma unroll
      for (int t = 2; t < Buffers; ++t)
        reference_load_kn<64>(blocks[t], weight, H, I, n0, t * 32);
      for (int first = 0; first < I; first += 32 * Buffers) {
#pragma unroll
        for (int t = 0; t < Buffers; ++t) {
          const int k = first + t * 32;
#pragma unroll
          for (int col = 0; col < 4; ++col)
            reference_mm_kn32<Rows>(
                out[col], blocks[t] + col * 8, input, k, lane);
          if (k + 32 * Buffers < I)
            reference_load_kn<64>(
                blocks[t], weight, H, I, n0, k + 32 * Buffers);
        }
      }
#pragma unroll
      for (int r = 0; r < Rows; ++r)
        if (choices[u * Rows + r] >= 0)
#pragma unroll
          for (int col = 0; col < 4; ++col)
            local_output
                [(r * Top + choices[u * Rows + r]) * 64 + col * 16 + lane] =
                    scalar_t(out[col][r] * fp8_scale(weights.w2_scale[e]));
      if (u + Subgroups < count)
#pragma unroll
        for (int t = 0; t < 2; ++t)
          reference_load_kn<64>(
              blocks[t],
              weights.w2 + size_t(experts[u + Subgroups]) * I * H,
              H,
              I,
              n0,
              t * 32);
    }
  }
  // Reuse norm SLM after expert GEMMs. Preserve each expert's BF16 output
  // and the top-k accumulation order before shared-down reuses this storage.
  reference_wg_bar(item.get_group());
  const int lid = item.get_local_linear_id();
  if (lid < Rows * 32) {
    const int r = lid / 32, pair = lid % 32;
    float values[2]{};
    // Padding has no expert outputs in the opt-in masked union. Do not read
    // its unwritten SLM entries; live rows retain the original top-k order.
    const bool live = !IgnorePadding || step.slots[r] >= 0;
    for (int k = 0; k < Top && live; ++k)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        values[j] += routing[r * Top + k] *
                     float(local_output[(r * Top + k) * 64 + 2 * pair + j]);
#pragma unroll
    for (int j = 0; j < 2; ++j)
      routed[j] = float(scalar_t(values[j]));
  }
  reference_wg_bar(item.get_group());
  return true;
}
