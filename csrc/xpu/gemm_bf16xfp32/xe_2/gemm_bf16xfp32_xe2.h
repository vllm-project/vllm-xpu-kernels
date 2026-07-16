#pragma once

#include <torch/all.h>

at::Tensor gemm_bf16xfp32_xe2(
    at::Tensor& A,       // [M, K] bf16
    at::Tensor& B_high,  // [K, N] bf16 (transposed weight)
    at::Tensor& B_low,   // [K, N] bf16 (transposed weight)
    double scale);
