// SPDX-License-Identifier: Apache-2.0
#pragma once

// Adapted from mono_layers_b.hpp input_norm_b, revision
// 922432bc4288f7a323cd550b2599fb318d4568a9. Keep ordinary BF16 RMS weights,
// residual rounding and a fixed SG-partial order; compute all rows together.
template <int Rows, bool RoundResidual, bool B1Workgroup = false>
inline void reference_norm(
    sycl::nd_item<1> item,
    const scalar_t* input,
    const scalar_t* residual,
    const scalar_t* weight,
    scalar_t* output,
    float* partial) {
  static_assert(H == Local * 4 && Rows <= 8);
  const int lid = item.get_local_linear_id();
  if constexpr (Rows == 1 && B1Workgroup) {
    // mono_layers.hpp input_norm B1 keeps four residual values in registers
    // and performs one workgroup reduction. Preserve ordinary MiniCPM RMS
    // weights and activation-dtype rounding instead of Gemma's 1+w weights.
    float value[4], sum = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k = lid * 4 + i;
      value[i] = float(input[k]);
      if constexpr (RoundResidual)
        value[i] =
            float(scalar_t(value[i] + (residual ? float(residual[k]) : 0.f)));
      sum += value[i] * value[i];
    }
    sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>());
    const float scale = sycl::rsqrt(sum / H + 1e-6f);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k = lid * 4 + i;
      output[k] = scalar_t(value[i] * scale * float(weight[k]));
    }
    return;
  }
  const auto sg = item.get_sub_group();
  const int s = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
#pragma unroll 1
  for (int r = 0; r < Rows; ++r) {
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int index = r * H + lid * 4 + j;
      float value = float(input[index]);
      if constexpr (RoundResidual)
        value =
            float(scalar_t(value + (residual ? float(residual[index]) : 0.f)));
      output[index] = scalar_t(value);
      sum += value * value;
    }
    sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
    if (lane == 0) partial[r * Subgroups + s] = sum;
  }
  reference_wg_bar(item.get_group());
  if (lid < Rows) {
    float sum = 0.f;
#pragma unroll 1
    for (int i = 0; i < Subgroups; ++i)
      sum += partial[lid * Subgroups + i];
    partial[Rows * Subgroups + lid] = sycl::rsqrt(sum / H + 1e-6f);
  }
  reference_wg_bar(item.get_group());
#pragma unroll 1
  for (int r = 0; r < Rows; ++r) {
    const float scale = partial[Rows * Subgroups + r];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      output[r * H + lid * 4 + j] = scalar_t(
          float(output[r * H + lid * 4 + j]) * scale *
          float(weight[lid * 4 + j]));
  }
}
