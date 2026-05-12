// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/xpu/XPUContext.h>
#include <sycl/sycl.hpp>

#include "dispatch_utils.h"
#include "ops.h"
#include "utils.h"

#include "quantization/fp4/mxfp4_quant.h"

// per_token_group_quant_mxfp4
//
// Quantise input [M, N] to MXFP4 block format (MX specification).
//
// Arguments:
//   input    [M, N]       – float / half / bfloat16 input tensor
//   output_q [M, N/2]     – packed FP4 output (uint8, two nibbles per byte)
//   output_s [M, N/g]     – UE8M0-rounded power-of-two scale (float32)
//                           May be column-major (see below).
//   group_size            – block size; must be 32 for the MX format
//   eps                   – absolute minimum to avoid log2(0); default 1e-10
void per_token_group_quant_mxfp4(
    const torch::Tensor& input,
    torch::Tensor& output_q,
    torch::Tensor& output_s,
    int64_t group_size,
    double eps) {
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(output_q.is_contiguous(), "output_q must be contiguous");
  TORCH_CHECK(
      group_size == 32, "MXFP4 requires group_size == 32, got ", group_size);
  TORCH_CHECK(
      input.numel() % group_size == 0,
      "input numel must be divisible by group_size");
  TORCH_CHECK(output_s.dim() == 2, "output_s must be 2-D");
  TORCH_CHECK(
      output_q.scalar_type() == at::ScalarType::Byte, "output_q must be uint8");

  const int num_groups = static_cast<int>(input.numel() / group_size);
  if (num_groups == 0) {
    // No work to do for empty input; ensure outputs are empty as well.
    TORCH_CHECK(
        output_q.numel() == 0, "output_q must be empty when input is empty");
    TORCH_CHECK(
        output_s.numel() == 0, "output_s must be empty when input is empty");
    return;
  }

  const at::DeviceGuard device_guard(input.device());

  // Choose how many sub-groups to pack into one work-group.
  constexpr int THREADS_PER_GROUP = 32;
  int groups_per_block = 1;
  if (num_groups % 16 == 0)
    groups_per_block = 16;
  else if (num_groups % 8 == 0)
    groups_per_block = 8;
  else if (num_groups % 4 == 0)
    groups_per_block = 4;
  else if (num_groups % 2 == 0)
    groups_per_block = 2;

  const int num_blocks = num_groups / groups_per_block;
  const int num_threads = groups_per_block * THREADS_PER_GROUP;

  // Detect column-major scale layout.
  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  // scale_num_cols = number of groups per token row (= N / group_size).
  const int scale_num_cols = output_s.size(1);
  const int scale_stride = static_cast<int>(output_s.stride(1));

  sycl::range<1> grid(num_blocks);
  sycl::range<1> block(num_threads);
  auto& queue = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "per_token_group_quant_mxfp4_kernel", [&] {
        queue.submit([&](sycl::handler& cgh) {
          auto kernel =
              vllm::mxfp4::per_token_group_quant_mxfp4_kernel<scalar_t>(
                  output_q.data_ptr<uint8_t>(),
                  output_s.data_ptr<float>(),
                  input.data_ptr<scalar_t>(),
                  static_cast<int>(group_size),
                  groups_per_block,
                  static_cast<float>(eps),
                  scale_num_cols,
                  scale_stride,
                  is_column_major);
          cgh.parallel_for(sycl::nd_range<1>(grid * block, block), kernel);
        });
      });
}
