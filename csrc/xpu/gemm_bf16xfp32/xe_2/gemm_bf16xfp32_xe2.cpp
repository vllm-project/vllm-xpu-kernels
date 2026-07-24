/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#include <torch/all.h>
#include "gemm_bf16xfp32_xe2.h"
#include "gemm_bf16xfp32_xe2_impl.hpp"

at::Tensor gemm_bf16xfp32_xe2(
    at::Tensor& A, at::Tensor& B_high, at::Tensor& B_low, double scale) {
  return gemm_bf16xfp32::gemm_bf16xfp32_xe2_impl(A, B_high, B_low, scale);
}
