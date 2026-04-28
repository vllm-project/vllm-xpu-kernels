#pragma once

#include "gemm_xe2_impl.hpp"

template <typename Config>
torch::Tensor run_gemm(
    sycl::queue& queue,
    const torch::Tensor& A,
    const torch::Tensor& B,
    int M,
    int N,
    int K) {
  using Gemm = typename Config::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  auto D = torch::empty({M, N}, A.options());

  auto stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  auto stride_B =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  auto stride_C =
      cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  auto stride_D =
      cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  using ProblemShape = typename Gemm::GemmKernel::ProblemShape;

  typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      ProblemShape{M, N, K, 1},
      {reinterpret_cast<ElementInputA*>(A.data_ptr()),
       stride_A,
       reinterpret_cast<ElementInputB*>(B.data_ptr()),
       stride_B},
      {{1.0f, 0.0f},
       nullptr,
       stride_C,
       reinterpret_cast<ElementOutput*>(D.data_ptr()),
       stride_D},
      hw_info};

  Gemm gemm_op;

  TORCH_CHECK(
      gemm_op.can_implement(arguments) == cutlass::Status::kSuccess,
      "CUTLASS cannot implement this GEMM");

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  TORCH_CHECK(workspace_size == 0, "CUTLASS GEMM requires non-zero workspace");

  auto status = gemm_op.initialize(arguments, nullptr, &queue);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM init failed");

  status = gemm_op.run(&queue);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM run failed");

  return D;
}
