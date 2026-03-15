// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace vllm {
namespace fp4 {

// Maximum representable FP4 E2M1 value.
static constexpr float FP4_MAX = 6.0f;

// Convert a single float32 value to a 4-bit FP4 E2M1 code (stored in the
// lower 4 bits of a uint8_t) using round-to-nearest-even (RNE).
inline uint8_t float_to_fp4_e2m1(float x) {
  const float b[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};

  uint8_t sign = (x < 0.0f) ? 0x8u : 0x0u;
  float a = sycl::fabs(x);

  // midpoint rounding: branchless
  uint8_t code = (a > b[0]) + (a > b[1]) + (a > b[2]) + (a > b[3]) +
                 (a > b[4]) + (a > b[5]) + (a > b[6]);

  return code | sign;
}

// Quantise an input tensor [M, N] to MXFP4 block format.
// output_q  : packed FP4 nibbles, two elements per uint8 byte.
//             Byte at index i stores elements [2i, 2i+1]:
//             byte[i] = fp4[2i+1] << 4 | fp4[2i]
// output_s  : UE8M0-rounded scale (power-of-two) stored as float32.
//             One scale per group_size input elements.
template <typename scalar_t>
class per_token_group_quant_mxfp4_kernel {
 private:
  uint8_t* out;
  float* scale;
  scalar_t const* input;
  const int group_size;  // = 32 for MX format
  const int groups_per_block;
  float eps;
  const int scale_num_cols;  // N / group_size  (columns in scale matrix)
  const int scale_stride;    // stride(1) of scale tensor
  bool is_column_major;

 public:
  per_token_group_quant_mxfp4_kernel(
      uint8_t* out_,
      float* scale_,
      scalar_t const* input_,
      const int group_size_,
      const int groups_per_block_,
      float eps_,
      const int scale_num_cols_ = 0,
      const int scale_stride_ = 0,
      bool is_column_major_ = false)
      : out(out_),
        scale(scale_),
        input(input_),
        group_size(group_size_),
        groups_per_block(groups_per_block_),
        eps(eps_),
        scale_num_cols(scale_num_cols_),
        scale_stride(scale_stride_),
        is_column_major(is_column_major_) {}

  void operator()
      [[sycl::reqd_sub_group_size(32)]] (sycl::nd_item<1> item) const {
    constexpr int THREADS_PER_GROUP = 32;

    int local_id = item.get_local_id(0);
    int local_group_id = local_id / THREADS_PER_GROUP;
    int lane_id = local_id % THREADS_PER_GROUP;

    int block_group_id = item.get_group(0) * groups_per_block;
    int global_group_id = block_group_id + local_group_id;

    int64_t group_offset = static_cast<int64_t>(global_group_id) * group_size;
    scalar_t const* group_input = input + group_offset;
    // Two FP4 nibbles packed into one byte, so the output offset is halved.
    uint8_t* group_output = out + group_offset / 2;

    // Resolve the scale output pointer for this group.
    float* scale_output;
    if (is_column_major) {
      const int row_idx = global_group_id / scale_num_cols;
      const int col_idx = global_group_id % scale_num_cols;
      scale_output = scale + col_idx * scale_stride + row_idx;
    } else {
      scale_output = scale + global_group_id;
    }

    float local_absmax = eps;
    if (lane_id < group_size) {
      float val = static_cast<float>(group_input[lane_id]);
      local_absmax = sycl::max(local_absmax, sycl::fabs(val));
    }
    local_absmax = sycl::reduce_over_group(
        item.get_sub_group(), local_absmax, sycl::maximum<float>());

    float y_s = local_absmax / FP4_MAX;
    y_s = sycl::exp2(sycl::ceil(sycl::log2(sycl::fmax(sycl::fabs(y_s), eps))));

    if (lane_id == 0) {
      *scale_output = y_s;
    }
    sycl::group_barrier(item.get_group());

    const float inv_scale = 1.0f / y_s;

    if (lane_id < group_size) {
      float val = static_cast<float>(group_input[lane_id]);
      float scaled_val = val * inv_scale;
      // Clamp to the representable FP4 E2M1 range [-6, 6].
      scaled_val = sycl::fmax(-FP4_MAX, sycl::fmin(scaled_val, FP4_MAX));
      uint8_t fp4_val = float_to_fp4_e2m1(scaled_val);

      // Pack: even lane → low nibble, odd lane (lane+1) → high nibble.
      // Retrieve the adjacent (odd) thread's FP4 code via sub-group shuffle.
      uint8_t next_fp4 = static_cast<uint8_t>(sycl::shift_group_left(
          item.get_sub_group(), static_cast<uint32_t>(fp4_val), 1u));

      if (lane_id % 2 == 0) {
        // byte[k] = fp4[2k+1] << 4 | fp4[2k]  (matches pack_uint4 in Python)
        group_output[lane_id / 2] =
            ((next_fp4 & 0x0Fu) << 4) | (fp4_val & 0x0Fu);
      }
    }
  }
};

}  // namespace fp4
}  // namespace vllm
