#pragma once

#include <torch/all.h>
#include <sycl/sycl.hpp>

torch::Tensor cutlass_gemm_xe2(
    sycl::queue& queue, const torch::Tensor& A, const torch::Tensor& B);
