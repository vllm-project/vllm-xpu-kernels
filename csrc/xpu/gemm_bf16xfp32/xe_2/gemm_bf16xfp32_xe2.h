#pragma once

#include <torch/all.h>

at::Tensor gemm_bf16xfp32_xe2(
    at::Tensor& A,       // [M, K] bf16
    at::Tensor& B_high,  // [N, K] bf16
    at::Tensor& B_low,   // [N, K] bf16
    double scale);
