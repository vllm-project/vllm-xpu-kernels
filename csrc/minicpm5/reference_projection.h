// SPDX-License-Identifier: Apache-2.0
#pragma once

// Adapted from mono_layers_b.hpp a_put_nat, ld_rows, mm_rows and the S1/S4
// projection schedule, revision 922432bc4288f7a323cd550b2599fb318d4568a9.
// BF16 A/B operands preserve the MiniCPM W8A16 range and rounding contract.

// B1 GEMV port of mono_layers.hpp dot16_fp8(_nat)/gemv_pipe,
// revision 922432bc4288f7a323cd550b2599fb318d4568a9.
// Exact E4M3 NaNs and MiniCPM projection-dtype rounding are retained.
inline float reference_b1_dot16(sycl::uint4 w, const uint32_t* xp) {
  using Half2 = sycl::vec<sycl::half, 2>;
  const auto xa = *reinterpret_cast<const sycl::uint4*>(xp);
  const auto xb = *reinterpret_cast<const sycl::uint4*>(xp + 4);
  const auto lo = [](uint32_t v) {
    return sycl::bit_cast<Half2>(fp8_half_lanes_multiply(v & 0x00ff00ffu));
  };
  const auto hi = [](uint32_t v) {
    return sycl::bit_cast<Half2>(
        fp8_half_lanes_multiply((v >> 8) & 0x00ff00ffu));
  };
  Half2 acc = lo(w.x()) * sycl::bit_cast<Half2>(xa.x());
  acc += hi(w.x()) * sycl::bit_cast<Half2>(xa.y());
  acc += lo(w.y()) * sycl::bit_cast<Half2>(xa.z());
  acc += hi(w.y()) * sycl::bit_cast<Half2>(xa.w());
  acc += lo(w.z()) * sycl::bit_cast<Half2>(xb.x());
  acc += hi(w.z()) * sycl::bit_cast<Half2>(xb.y());
  acc += lo(w.w()) * sycl::bit_cast<Half2>(xb.z());
  acc += hi(w.w()) * sycl::bit_cast<Half2>(xb.w());
  return float(acc[0]) + float(acc[1]);
}

inline float reference_b1_dot16_nat(sycl::uint4 w, const scalar_t* x) {
  using Half2 = sycl::vec<sycl::half, 2>;
  const auto lo = [](uint32_t v) {
    return sycl::bit_cast<Half2>(fp8_half_lanes_multiply(v & 0x00ff00ffu));
  };
  const auto hi = [](uint32_t v) {
    return sycl::bit_cast<Half2>(
        fp8_half_lanes_multiply((v >> 8) & 0x00ff00ffu));
  };
  const auto xx = [&](int a, int b) {
    return Half2(sycl::half(float(x[a])), sycl::half(float(x[b])));
  };
  Half2 acc = lo(w.x()) * xx(0, 2);
  acc += hi(w.x()) * xx(1, 3);
  acc += lo(w.y()) * xx(4, 6);
  acc += hi(w.y()) * xx(5, 7);
  acc += lo(w.z()) * xx(8, 10);
  acc += hi(w.z()) * xx(9, 11);
  acc += lo(w.w()) * xx(12, 14);
  acc += hi(w.w()) * xx(13, 15);
  return float(acc[0]) + float(acc[1]);
}

inline void reference_b1_output_preload(
    sycl::nd_item<1> item, const uint8_t* weight, sycl::uint4 (&raw)[2][8]) {
  const auto sg = item.get_sub_group();
  const int row =
      item.get_group_linear_id() * 64 + sg.get_group_linear_id() * 2;
  const int lane = sg.get_local_linear_id();
#pragma unroll
  for (int r = 0; r < 2; ++r)
#pragma unroll
    for (int j = 0; j < 8; ++j)
      raw[r][j] = *reinterpret_cast<const sycl::uint4*>(
          weight + size_t(row + r) * 2048 + j * 256 + lane * 16);
}

inline bool reference_b1_output_gemv(
    sycl::nd_item<1> item,
    const sycl::uint4 (&raw)[2][8],
    const scalar_t* input,
    float scale,
    uint32_t* packed,
    float* output,
    Step step,
    uint32_t call) {
  if (!grid_wait<Groups * Local, true>(
          item, step.control, 4, call, step.health))
    return false;
  const int lid = item.get_local_linear_id();
  auto* xp = reinterpret_cast<sycl::half*>(packed);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int k = lid * 4 + i;
    xp[(k & ~3) | ((k & 1) << 1) | ((k >> 1) & 1)] =
        sycl::half(float(input[k]));
  }
  reference_wg_bar(item.get_group());
  const auto sg = item.get_sub_group();
  const int lane = sg.get_local_linear_id();
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      sum += reference_b1_dot16(raw[r][j], packed + (j * 256 + lane * 16) / 2);
    sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
    if (lane == 0)
      output[sg.get_group_linear_id() * 2 + r] =
          float(scalar_t(sum * (256.f * scale)));
  }
  return true;
}

// mono_layers.hpp load_prt / tail shared column partials, TP2 INTER=256.
// MiniCPM retains canonical normalized inputs and rounds gate/up before SiLU.
inline void reference_b1_shared_preload(
    sycl::nd_item<1> item, Weights weights, sycl::uint4 (&raw)[4]) {
  const auto sg = item.get_sub_group();
  const int s = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
#pragma unroll
  for (int t = 0; t < 4; ++t)
    raw[t] = *reinterpret_cast<const sycl::uint4*>(
        weights.shared13 + size_t(s * 16 + 4 * t + (lane >> 2)) * H +
        item.get_group_linear_id() * 64 + (lane & 3) * 16);
}

inline void reference_b1_shared_partial(
    sycl::nd_item<1> item,
    Workspace work,
    const scalar_t* input,
    const sycl::uint4 (&raw)[4]) {
  const auto sg = item.get_sub_group();
  const int s = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
  const int wg = item.get_group_linear_id();
  float* partial = work.attention_partial + Groups * E;
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    float value =
        reference_b1_dot16_nat(raw[t], input + wg * 64 + (lane & 3) * 16);
    value += sycl::permute_group_by_xor(sg, value, 2);
    value += sycl::permute_group_by_xor(sg, value, 1);
    if ((lane & 3) == 0)
      partial[wg * 512 + s * 16 + 4 * t + (lane >> 2)] = value;
  }
}

inline void reference_b1_shared_finish(
    sycl::nd_item<1> item, Weights weights, Workspace work) {
  const int n = item.get_group_linear_id() * 16 + item.get_local_linear_id();
  if (item.get_group_linear_id() < 16 && item.get_local_linear_id() < 16) {
    const float* partial = work.attention_partial + Groups * E;
    float gate = 0.f, up = 0.f;
#pragma unroll
    for (int wg = 0; wg < Groups; ++wg) {
      gate += partial[wg * 512 + n];
      up += partial[wg * 512 + 256 + n];
    }
    const float scale = 256.f * weights.shared13_scale[0];
    gate = float(scalar_t(gate * scale));
    up = float(scalar_t(up * scale));
    work.intermediate[Top * 256 + n] =
        scalar_t(gate / (1.f + sycl::exp(-gate)) * up);
  }
}

#if defined(MINICPM5_MONO_FP16) || defined(MINICPM5_MONO_BF16_FP8_HALF_ENABLED)
constexpr bool ProjectionPermuted = true;
#else
constexpr bool ProjectionPermuted = false;
#endif

template <int Rows, int K, bool Permuted = false>
inline void pack_projection_input(
    sycl::nd_item<1> item,
    const scalar_t* input,
    int stride,
    uint32_t* packed) {
  for (int index = item.get_local_linear_id(); index < (K / 16) * Rows * 8;
       index += Local) {
    const int row = index / 8 % Rows, tile = index / (Rows * 8);
    const int pair = 2 * (index % 8);
    const int position = Permuted ? (pair & ~3) | ((pair >> 1) & 1) : pair;
    const int column = tile * 16 + position;
    packed[index] = pack16(
        input[row * stride + column],
        input[row * stride + column + (Permuted ? 2 : 1)]);
  }
}

template <int Rows>
inline UInt4 projection_a(const uint32_t* packed, int tile, int lane) {
  UInt4 result{};
  // DPAS consumes a subgroup matrix. Keep operand loads converged: repeat1
  // ignores the second row, whose lanes can safely mirror row0 here.
#pragma unroll
  for (int word = 0; word < (Rows + 1) / 2; ++word) {
    const int row = operand_row(lane, word) % Rows;
    result[word] = packed[tile * Rows * 8 + row * 8 + lane % 8];
  }
  return result;
}

template <int NC, int ChunkStride = 16>
inline void reference_load_rows(sycl::uint4 (&weight)[NC], const uint8_t* row) {
#pragma unroll
  for (int tile = 0; tile < NC; ++tile)
    weight[tile] =
        *reinterpret_cast<const sycl::uint4*>(row + tile * ChunkStride);
}

// PR batched Q / O preload: 64 output columns per WG, a full K slice per SG.
// MiniCPM FP16 retains its two-pass layout; both projections have K=H at TP2.
template <int NC, int K = H>
inline void reference_batched_projection_preload(
    sycl::nd_item<1> item, const uint8_t* weight, sycl::uint4 (&raw)[NC]) {
  constexpr int Slices = K / (NC * 16);
  static_assert(Subgroups % Slices == 0 && K % (NC * 16) == 0);
  const auto sg = item.get_sub_group();
  const int s = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
  const int n = item.get_group_linear_id() * 64 + s / Slices * 16 + lane;
  reference_load_rows(raw, weight + size_t(n) * K + s % Slices * NC * 16);
}

template <int Rows>
inline UInt8 reference_fp8_tile(sycl::uint4 value) {
  UInt8 result;
#pragma unroll
  for (int word = 0; word < 4; ++word) {
    if constexpr (ProjectionPermuted) {
      result[2 * word] = fp8_lanes_rows<Rows>(value[word] & 0x00ff00ffu);
      result[2 * word + 1] =
          fp8_lanes_rows<Rows>((value[word] >> 8) & 0x00ff00ffu);
    } else {
      result[2 * word] = fp8_pair_rows<Rows>(value[word]);
      result[2 * word + 1] = fp8_pair_rows<Rows>(value[word] >> 16);
    }
  }
  return result;
}

template <int Rows, int NC>
inline void reference_mm_rows(
    Float8& out,
    const sycl::uint4 (&weight)[NC],
    const uint32_t* packed,
    int first,
    int lane) {
#pragma unroll
  for (int tile = 0; tile < NC; ++tile)
    dpas_rows<Rows>(
        out,
        projection_a<Rows>(packed, first + tile, lane),
        reference_fp8_tile<Rows>(weight[tile]));
}

template <int Chunks>
constexpr int ProjectionPreload =
#if defined(MINICPM5_MONO_FP16) || defined(MINICPM5_MONO_BF16_FP8_HALF_ENABLED)
    Chunks > 2 ? 2 : Chunks;
#else
    Chunks;
#endif

template <int Rows, int Chunks, int Depth, int ChunkStride = 16>
inline void reference_mm_rows_stream(
    Float8& out,
    sycl::uint4 (&current)[Depth],
    const uint8_t* row,
    const uint32_t* packed,
    int first,
    int lane) {
  static_assert(Chunks % Depth == 0);
  if constexpr (Chunks == Depth) {
    reference_mm_rows<Rows>(out, current, packed, first, lane);
  } else {
    // Keep both weight banks scalarized. An unrolled full row lets the
    // scheduler retain too many converted DPAS operands at the same time.
#pragma unroll 1
    for (int tile = 0; tile < Chunks; tile += 2 * Depth) {
      sycl::uint4 next[Depth];
      if (tile + Depth < Chunks)
        reference_load_rows<Depth, ChunkStride>(
            next, row + (tile + Depth) * ChunkStride);
      reference_mm_rows<Rows>(out, current, packed, first + tile, lane);
      if (tile + 2 * Depth < Chunks)
        reference_load_rows<Depth, ChunkStride>(
            current, row + (tile + 2 * Depth) * ChunkStride);
      if (tile + Depth < Chunks)
        reference_mm_rows<Rows>(out, next, packed, first + tile + Depth, lane);
    }
  }
}

template <
    int Rows,
    int K,
    int N,
    int PackedK = (K > 4096 ? 4096 : K),
    int WaitPhase = -1,
    bool QueriesOnly = false,
    bool FullWeightPreload = false,
    bool ExternalPreload = false,
    bool BalancedOutput = false,
    bool TiledWeight = false>
inline bool reference_projection_nk(
    sycl::nd_item<1> item,
    const uint8_t* weight,
    const scalar_t* input,
    int input_stride,
    scalar_t* output,
    int output_stride,
    float scale,
    uint32_t* packed,
    float* partial,
    uint32_t* control = nullptr,
    uint32_t call = 0,
    uint32_t* health = nullptr,
    const sycl::uint4* initial_weights = nullptr) {
  static_assert(WaitPhase < 0 || K <= PackedK);
  static_assert(!ExternalPreload || (FullWeightPreload && K <= PackedK));
  static_assert(
      !BalancedOutput ||
      (Rows == 1 && K == H && N == 4352 && !QueriesOnly && !ExternalPreload));
  scale = fp8_scale(scale);
  constexpr int Tiles = N / 16;
  constexpr int GroupTiles =
      (Tiles + Groups - 1) / Groups < 4 ? (Tiles + Groups - 1) / Groups : 4;
#ifdef MINICPM5_MONO_FP16
  // Bound live weight registers while preserving this WG's contiguous
  // output ownership, required by its paired in-kernel all-reduce.
  constexpr int PassTiles = GroupTiles > 2 ? 2 : GroupTiles;
#else
  constexpr int PassTiles = GroupTiles;
#endif
  constexpr int Passes = GroupTiles / PassTiles;
  constexpr int Slices = Subgroups / PassTiles;
  constexpr int NC = K / Slices / 16;
  static_assert(
      N % 16 == 0 && Subgroups % PassTiles == 0 &&
      GroupTiles % PassTiles == 0 && K % (Slices * 16) == 0);
  const int lid = item.get_local_linear_id(), wg = item.get_group_linear_id();
  auto sg = item.get_sub_group();
  const int s = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
  const int tile = s / Slices, slice = s % Slices;
  const auto physical_row = [](int n) {
    return QueriesOnly ? (n / D) * 2 * D + n % D : n;
  };
  static_assert(!TiledWeight || BalancedOutput);
  if constexpr (BalancedOutput) {
    // Stripe independent (output tile, K slice) jobs across subgroups.
    // Every output keeps the original eight contiguous K slices and
    // ascending FP32 partial reduction, including the ninth output tile.
    static_assert(Rows == 1 && Slices == 8 && NC == 16);
    const int count = Tiles / Groups + int(wg < Tiles % Groups);
    const int begin = wg * (Tiles / Groups) + sycl::min(wg, Tiles % Groups);
    constexpr int WeightStride = TiledWeight ? 256 : 16;
    const auto row_pointer = [&](int n, int part) {
      if constexpr (TiledWeight)
        return weight + (size_t(n / 16) * (K / 16) + part * NC) * 256 +
               (n % 16) * 16;
      else
        return weight + size_t(n) * K + part * NC * 16;
    };
    sycl::uint4 raw[ProjectionPreload<NC>];
    const int initial_n = (begin + s / Slices) * 16 + lane;
    reference_load_rows<ProjectionPreload<NC>, WeightStride>(
        raw, row_pointer(initial_n, s % Slices));
    pack_projection_input<Rows, K, ProjectionPermuted>(
        item, input, input_stride, packed);
    reference_wg_bar(item.get_group());
#pragma unroll 1
    for (int task = s; task < count * Slices; task += Subgroups) {
      const int n = (begin + task / Slices) * 16 + lane;
      const int part = task % Slices;
      Float8 values{};
      reference_mm_rows_stream<Rows, NC, ProjectionPreload<NC>, WeightStride>(
          values, raw, row_pointer(n, part), packed, part * NC, lane);
      partial[task * 16 + lane] = values[0];
      const int next = task + Subgroups;
      if (next < count * Slices)
        reference_load_rows<ProjectionPreload<NC>, WeightStride>(
            raw,
            row_pointer((begin + next / Slices) * 16 + lane, next % Slices));
    }
    reference_wg_bar(item.get_group());
    if (lid < count * 16) {
      const int tile_index = lid / 16, column = lid % 16;
      float sum = 0.f;
#pragma unroll
      for (int k = 0; k < Slices; ++k)
        sum += partial[(tile_index * Slices + k) * 16 + column];
      output[(begin + tile_index) * 16 + column] = scalar_t(sum * scale);
    }
    reference_wg_bar(item.get_group());
    return true;
  }
  if constexpr (K > PackedK) {
    static_assert(K % PackedK == 0 && N == H);
    constexpr int SegmentChunks = PackedK / Slices / 16;
#pragma unroll 1
    for (int pass = 0; pass < Passes; ++pass) {
      const int n = (wg * GroupTiles + pass * PassTiles + tile) * 16 + lane;
      Float8 values{};
#pragma unroll
      for (int begin = 0; begin < K; begin += PackedK) {
        sycl::uint4 registers[ProjectionPreload<SegmentChunks>];
        reference_load_rows(
            registers,
            weight + size_t(physical_row(n)) * K + begin +
                slice * SegmentChunks * 16);
        pack_projection_input<Rows, PackedK, ProjectionPermuted>(
            item, input + begin, input_stride, packed);
        reference_wg_bar(item.get_group());
        reference_mm_rows_stream<Rows, SegmentChunks>(
            values,
            registers,
            weight + size_t(physical_row(n)) * K + begin +
                slice * SegmentChunks * 16,
            packed,
            slice * SegmentChunks,
            lane);
        reference_wg_bar(item.get_group());
      }
#pragma unroll
      for (int r = 0; r < Rows; ++r)
        partial[(s * Rows + r) * 16 + lane] = values[r];
      reference_wg_bar(item.get_group());
      for (int index = lid; index < Rows * PassTiles * 16; index += Local) {
        const int r = index / (PassTiles * 16), col = index % (PassTiles * 16);
        float sum = 0.f;
#pragma unroll
        for (int k = 0; k < Slices; ++k)
          sum += partial[((col / 16 * Slices + k) * Rows + r) * 16 + col % 16];
        output
            [r * output_stride +
             physical_row((wg * GroupTiles + pass * PassTiles) * 16 + col)] =
                scalar_t(sum * scale);
      }
      reference_wg_bar(item.get_group());
    }
  } else {
    // PR mono_layers_b.hpp ld_rows<16>/mm_rows<16>: keep the complete
    // raw weight slice resident before packing inputs or waiting on OP.
    // The accumulator order and dtype rounding stay unchanged.
    sycl::uint4 registers[FullWeightPreload ? NC : ProjectionPreload<NC>];
    const int first_output = (wg * GroupTiles + tile) * 16 + lane;
    if (first_output < N) {
      if constexpr (ExternalPreload) {
#pragma unroll
        for (int c = 0; c < NC; ++c)
          registers[c] = initial_weights[c];
      } else {
        reference_load_rows(
            registers,
            weight + size_t(physical_row(first_output)) * K + slice * NC * 16);
      }
    }
    if constexpr (WaitPhase >= 0)
      if (!grid_wait(item, control, WaitPhase, call, health)) return false;
    pack_projection_input<Rows, K, ProjectionPermuted>(
        item, input, input_stride, packed);
    reference_wg_bar(item.get_group());
    for (int first = wg * GroupTiles; first < Tiles;
         first += Groups * GroupTiles) {
#pragma unroll 1
      for (int pass = 0; pass < Passes; ++pass) {
        const int n = (first + pass * PassTiles + tile) * 16 + lane;
        Float8 values{};
        if (n < N)
          reference_mm_rows_stream<Rows, NC>(
              values,
              registers,
              weight + size_t(physical_row(n)) * K + slice * NC * 16,
              packed,
              slice * NC,
              lane);
#pragma unroll
        for (int r = 0; r < Rows; ++r)
          partial[(s * Rows + r) * 16 + lane] = values[r];
        reference_wg_bar(item.get_group());
        for (int index = lid; index < Rows * PassTiles * 16; index += Local) {
          const int r = index / (PassTiles * 16),
                    col = index % (PassTiles * 16);
          const int out = (first + pass * PassTiles) * 16 + col;
          if (out < N) {
            float sum = 0.f;
#pragma unroll
            for (int k = 0; k < Slices; ++k)
              sum +=
                  partial[((col / 16 * Slices + k) * Rows + r) * 16 + col % 16];
            output[r * output_stride + physical_row(out)] =
                scalar_t(sum * scale);
          }
        }
        const int next = pass + 1 < Passes
                             ? n + PassTiles * 16
                             : (first + Groups * GroupTiles + tile) * 16 + lane;
        if (next < N)
          reference_load_rows(
              registers,
              weight + size_t(physical_row(next)) * K + slice * NC * 16);
        reference_wg_bar(item.get_group());
      }
    }
  }
  return true;
}

template <int Rows, int Intermediate>
inline void reference_shared_gateup(
    sycl::nd_item<1> item,
    const uint8_t* weight,
    const scalar_t* input,
    int input_stride,
    scalar_t* output,
    int output_stride,
    float scale,
    uint32_t* packed,
    float* partial) {
  static_assert(Intermediate % 16 == 0);
  scale = fp8_scale(scale);
  constexpr int Chunks = H / Subgroups / 16;
  const int wg = item.get_group_linear_id(), lid = item.get_local_linear_id();
  const auto sg = item.get_sub_group();
  const int s = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
  const int n = wg * 16 + lane;
  sycl::uint4 gate[Chunks], up[Chunks];
  if (n < Intermediate) {
    reference_load_rows(gate, weight + size_t(n) * H + s * Chunks * 16);
    reference_load_rows(
        up, weight + size_t(Intermediate + n) * H + s * Chunks * 16);
  }
  pack_projection_input<Rows, H, ProjectionPermuted>(
      item, input, input_stride, packed);
  reference_wg_bar(item.get_group());
  if (wg * 16 >= Intermediate) return;
  Float8 g{}, u{};
  reference_mm_rows<Rows>(g, gate, packed, s * Chunks, lane);
  reference_mm_rows<Rows>(u, up, packed, s * Chunks, lane);
#pragma unroll
  for (int r = 0; r < Rows; ++r) {
    partial[(s * Rows + r) * 16 + lane] = g[r];
    partial[((Subgroups + s) * Rows + r) * 16 + lane] = u[r];
  }
  reference_wg_bar(item.get_group());
  for (int index = lid; index < Rows * 16; index += Local) {
    const int r = index / 16, col = index % 16;
    float gv = 0.f, uv = 0.f;
#pragma unroll
    for (int k = 0; k < Subgroups; ++k) {
      gv += partial[(k * Rows + r) * 16 + col];
      uv += partial[((Subgroups + k) * Rows + r) * 16 + col];
    }
    gv = float(scalar_t(gv * scale));
    uv = float(scalar_t(uv * scale));
    output[r * output_stride + wg * 16 + col] =
        scalar_t(gv / (1.f + sycl::exp(-gv)) * uv);
  }
  reference_wg_bar(item.get_group());
}

#ifdef MINICPM5_MONO_FP16
// Balanced output ownership keeps each FP16 column's original sixteen K128
// partials, DPAS order and ascending partial reduction. No reassociation.
template <int N>
inline void reference_fp16_tiled_qkv(
    sycl::nd_item<1> item,
    const uint8_t* weight,
    const scalar_t* input,
    scalar_t* output,
    float scale,
    uint32_t* packed,
    float* partial) {
  static_assert(N == 4352 || N == 8704);
  constexpr int Tiles = N / 16, Slices = 16, NC = 8;
  constexpr int Depth = ProjectionPreload<NC>;
  const int lid = item.get_local_linear_id(), wg = item.get_group_linear_id();
  const auto sg = item.get_sub_group();
  const int s = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
  const int count = Tiles / Groups + int(wg < Tiles % Groups);
  const int begin = wg * (Tiles / Groups) + sycl::min(wg, Tiles % Groups);
  const auto row_pointer = [&](int n, int part) {
    return weight + (size_t(n / 16) * (H / 16) + part * NC) * 256 +
           (n % 16) * 16;
  };
  scale = fp8_scale(scale);
  sycl::uint4 registers[Depth];
  reference_load_rows<Depth, 256>(
      registers, row_pointer((begin + s / Slices) * 16 + lane, s % Slices));
  pack_projection_input<1, H, ProjectionPermuted>(item, input, H, packed);
  reference_wg_bar(item.get_group());
  #pragma unroll 1
  for (int task = s; task < count * Slices; task += Subgroups) {
    const int n = (begin + task / Slices) * 16 + lane;
    const int part = task % Slices;
    Float8 values{};
    reference_mm_rows_stream<1, NC, Depth, 256>(
        values, registers, row_pointer(n, part), packed, part * NC, lane);
    partial[task * 16 + lane] = values[0];
    const int next = task + Subgroups;
    if (next < count * Slices)
      reference_load_rows<Depth, 256>(
          registers,
          row_pointer((begin + next / Slices) * 16 + lane, next % Slices));
  }
  reference_wg_bar(item.get_group());
  if (lid < count * 16) {
    const int output_tile = lid / 16, column = lid % 16;
    float sum = 0.f;
  #pragma unroll
    for (int part = 0; part < Slices; ++part)
      sum += partial[(output_tile * Slices + part) * 16 + column];
    output[(begin + output_tile) * 16 + column] = scalar_t(sum * scale);
  }
  reference_wg_bar(item.get_group());
}
#endif
