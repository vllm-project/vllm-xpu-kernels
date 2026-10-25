// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace gpu::cutlass_kernel::grouped_gemm {

enum class GreedyDtype { BF16, FP16, MXFP8, MXFP4, W4A8, FP8Tensor, FP8Block };

struct GreedyArguments {
  const void* a;
  const void* sa;
  const void* b;
  const void* sb;
  const float* bias;
  void* d;
  const int* counts;
  int n, k, groups;
  // Per-expert counts stay on the device, but the packed row total is known
  // on the host and bounds the largest possible uniform expert.
  int64_t total_rows;
};

template <GreedyDtype Dtype>
void launch_greedy(sycl::queue& stream, GreedyArguments args);

}  // namespace gpu::cutlass_kernel::grouped_gemm
