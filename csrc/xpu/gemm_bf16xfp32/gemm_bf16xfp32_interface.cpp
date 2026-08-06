/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#include "csrc/utils.h"
#include "gemm_bf16xfp32_interface.h"

#ifdef VLLM_XPU_ENABLE_XE2
  #include "xe_2/gemm_bf16xfp32_xe2.h"
#endif

at::Tensor gemm_bf16xfp32(
    at::Tensor A, at::Tensor B_high, at::Tensor B_low, double scale) {
#ifdef VLLM_XPU_ENABLE_XE2
  if (vllm::xpu::is_xe2_arch()) {
    // Pass by lvalue reference to xe2 impl
    return gemm_bf16xfp32_xe2(A, B_high, B_low, scale);
  }
#endif
  TORCH_CHECK(false, "gemm_bf16xfp32: requires XE2 architecture (BMG/LNL)");
  return at::Tensor();
}
