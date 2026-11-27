// SPDX-License-Identifier: Apache-2.0
#pragma once

// The single-row GEMV/AXPY schedule from mono_layers.hpp gemv_pipe and S6/S7,
// revision 922432bc4288f7a323cd550b2599fb318d4568a9. Exact BF16 decoding and
// FP32 products replace scaled FP16 scratch to preserve the model contract.

inline void
reference_axpy16(float (&out)[16], sycl::uint4 weight, float input) {
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    const uint32_t pair = fp8_pair_bf16(weight[p / 2] >> (16 * (p % 2)));
    out[2 * p] += sycl::bit_cast<float>((pair & 0xffffu) << 16) * input;
    out[2 * p + 1] += sycl::bit_cast<float>(pair & 0xffff0000u) * input;
  }
}

// K8192 cannot hold both reference GEMV buffers in 256 GRFs. Distribute
// K2048 slices across adjacent subgroups, then reduce their scalar results
// in SLM. Each slice retains the reference fixed-NB, two-buffer schedule.
template <int K, int N>
inline void reference_gemv_split_nk(
    sycl::nd_item<1> item,
    const uint8_t* weight,
    const scalar_t* input,
    scalar_t* output,
    float scale,
    float* partial) {
  constexpr int SliceK = 2048, Slices = K / SliceK;
  constexpr int Outputs = Subgroups / Slices, Workers = Groups * Outputs;
  constexpr int Chunks = SliceK / 256, NB = N / Workers;
  static_assert(K % SliceK == 0 && Subgroups % Slices == 0 && N % Workers == 0);
  auto sg = item.get_sub_group();
  const int lane = sg.get_local_linear_id(), s = sg.get_group_linear_id();
  const int slice = s % Slices, wg = item.get_group_linear_id();
  const int sid = wg * Outputs + s / Slices, lid = item.get_local_linear_id();
  sycl::uint4 buffer[2][Chunks];
#pragma unroll
  for (int k = 0; k < Chunks; ++k)
    buffer[0][k] = *reinterpret_cast<const sycl::uint4*>(
        weight + size_t(sid) * K + slice * SliceK + k * 256 + lane * 16);
#pragma clang loop unroll(full)
  for (int turn = 0; turn < NB; ++turn) {
    const int n = sid + turn * Workers;
    if (turn + 1 < NB)
#pragma unroll
      for (int k = 0; k < Chunks; ++k)
        buffer[(turn + 1) & 1][k] = *reinterpret_cast<const sycl::uint4*>(
            weight + size_t(n + Workers) * K + slice * SliceK + k * 256 +
            lane * 16);
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < Chunks; ++k) {
      const auto current = buffer[turn & 1][k];
#pragma unroll
      for (int p = 0; p < 8; ++p) {
        const uint32_t pair = fp8_pair_bf16(current[p / 2] >> (16 * (p % 2)));
        const int offset = slice * SliceK + k * 256 + lane * 16 + 2 * p;
        sum += sycl::bit_cast<float>((pair & 0xffffu) << 16) *
               float(input[offset]);
        sum += sycl::bit_cast<float>(pair & 0xffff0000u) *
               float(input[offset + 1]);
      }
    }
    sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
    if (lane == 0) partial[s] = sum;
    reference_wg_bar(item.get_group());
    if (lid < Outputs) {
      float value = 0.f;
#pragma unroll
      for (int j = 0; j < Slices; ++j)
        value += partial[lid * Slices + j];
      output[wg * Outputs + lid + turn * Workers] = scalar_t(value * scale);
    }
    reference_wg_bar(item.get_group());
  }
}

template <int K, int N>
inline void reference_gemv_nk(
    sycl::nd_item<1> item,
    const uint8_t* weight,
    const scalar_t* input,
    scalar_t* output,
    float scale,
    float* partial = nullptr) {
  if constexpr (K > 4096) {
    reference_gemv_split_nk<K, N>(item, weight, input, output, scale, partial);
    return;
  }
  static_assert(K % 256 == 0);
  auto sg = item.get_sub_group();
  const int lane = sg.get_local_linear_id();
  const int sid =
      sg.get_group_linear_id() * Groups + item.get_group_linear_id();
  constexpr int Chunks = K / 256, Workers = Groups * Subgroups;
  constexpr int NB = (N + Workers - 1) / Workers;
  sycl::uint4 buffer[2][Chunks];
  if (sid >= N) return;
#pragma unroll
  for (int k = 0; k < Chunks; ++k)
    buffer[0][k] = *reinterpret_cast<const sycl::uint4*>(
        weight + size_t(sid) * K + k * 256 + lane * 16);
#pragma clang loop unroll(full)
  for (int turn = 0; turn < NB; ++turn) {
    const int n = sid + turn * Workers;
    if (n >= N) continue;
    if (n + Workers < N)
#pragma unroll
      for (int k = 0; k < Chunks; ++k)
        buffer[(turn + 1) & 1][k] = *reinterpret_cast<const sycl::uint4*>(
            weight + size_t(n + Workers) * K + k * 256 + lane * 16);
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < Chunks; ++k) {
      const auto current = buffer[turn & 1][k];
#pragma unroll
      for (int p = 0; p < 8; ++p) {
        const uint32_t pair = fp8_pair_bf16(current[p / 2] >> (16 * (p % 2)));
        const int offset = k * 256 + lane * 16 + 2 * p;
        sum += sycl::bit_cast<float>((pair & 0xffffu) << 16) *
               float(input[offset]);
        sum += sycl::bit_cast<float>(pair & 0xffff0000u) *
               float(input[offset + 1]);
      }
    }
    sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
    if (lane == 0) output[n] = scalar_t(sum * scale);
  }
}

template <int TP>
inline void reference_moe_gateup_m1(
    sycl::nd_item<1> item,
    Weights weights,
    Workspace work,
    const scalar_t* input,
    float* partial) {
  constexpr int I = 512 / TP, Quarters = I / 64;
  const int wg = item.get_group_linear_id(), lid = item.get_local_linear_id();
  auto sg = item.get_sub_group();
  const int s = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
  for (int task = wg; task < Top * Quarters; task += Groups) {
    const int choice = task / Quarters, quarter = task % Quarters;
    const int expert = work.experts[choice];
    const auto* base = weights.w13 + size_t(expert) * H * 2 * I +
                       ((lane & 4) ? I : 0) + quarter * 64 + (lane & 3) * 16;
    sycl::uint4 registers[32];
#pragma unroll
    for (int t = 0; t < 32; ++t)
      registers[t] = *reinterpret_cast<const sycl::uint4*>(
          base + size_t(s * 64 + 2 * t + lane / 8) * 2 * I);
    float out[16]{};
#pragma unroll
    for (int t = 0; t < 32; ++t)
      reference_axpy16(
          out, registers[t], float(input[s * 64 + 2 * t + lane / 8]));
#pragma unroll
    for (int i = 0; i < 16; ++i)
      out[i] += sycl::permute_group_by_xor(sg, out[i], 8);
    if (lane < 8)
#pragma unroll
      for (int i = 0; i < 16; ++i)
        partial[s * 128 + lane * 16 + i] = out[i];
    reference_wg_bar(item.get_group());
    if (lid < 64) {
      float g = 0.f, up = 0.f;
#pragma unroll
      for (int k = 0; k < Subgroups; ++k) {
        g += partial[k * 128 + lid];
        up += partial[k * 128 + 64 + lid];
      }
      g = float(scalar_t(g * weights.w13_scale[expert]));
      up = float(scalar_t(up * weights.w13_scale[expert]));
      work.intermediate[choice * I + quarter * 64 + lid] =
          scalar_t(g / (1.f + sycl::exp(-g)) * up);
    }
    reference_wg_bar(item.get_group());
  }
}

template <int TP>
inline void reference_moe_down_m1(
    sycl::nd_item<1> item, Weights weights, Workspace work, float* partial) {
  constexpr int I = 512 / TP, Chunk = I / 4;
  const int wg = item.get_group_linear_id(), lid = item.get_local_linear_id();
  auto sg = item.get_sub_group();
  const int s = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
  for (int wave = 0; wave < Top / 8; ++wave) {
    const int choice = wave * 8 + s / 4, expert = work.experts[choice];
    const auto* base =
        weights.w2 + size_t(expert) * I * H + wg * 64 + (lane & 3) * 16;
    sycl::uint4 registers[Chunk / 4];
#pragma unroll
    for (int t = 0; t < Chunk / 4; ++t)
      registers[t] = *reinterpret_cast<const sycl::uint4*>(
          base + size_t(s % 4 * Chunk + 4 * t + lane / 4) * H);
    float out[16]{};
#pragma unroll
    for (int t = 0; t < Chunk / 4; ++t)
      reference_axpy16(
          out,
          registers[t],
          float(work.intermediate
                    [choice * I + s % 4 * Chunk + 4 * t + lane / 4]));
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      out[i] += sycl::permute_group_by_xor(sg, out[i], 4);
      out[i] += sycl::permute_group_by_xor(sg, out[i], 8);
    }
    if (lane < 4)
#pragma unroll
      for (int i = 0; i < 16; ++i)
        partial[s * 64 + lane * 16 + i] = out[i];
    reference_wg_bar(item.get_group());
    if (lid < 64)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float sum = 0.f;
#pragma unroll
        for (int k = 0; k < 4; ++k)
          sum += partial[(j * 4 + k) * 64 + lid];
        const int chosen = wave * 8 + j;
        work.expert_out[chosen * H + wg * 64 + lid] =
            scalar_t(sum * weights.w2_scale[work.experts[chosen]]);
      }
    reference_wg_bar(item.get_group());
  }
}
