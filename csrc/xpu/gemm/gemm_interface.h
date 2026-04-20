#pragma once

#include <torch/all.h>

torch::Tensor
cutlass_gemm_interface(const torch::Tensor& A, const torch::Tensor& B);
