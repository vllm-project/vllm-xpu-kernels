#pragma once

#include <torch/all.h>

#include <optional>
#include <tuple>

// Small-M (decode) fp16 x fp8-e4m3 GEMV with a per-tensor fp32 scale, for
// Xe2 (BMG). Implemented in xe_2/fp8_gemv_xe2.cpp (gemv_kernels_xe_2).
namespace vllm::fp8_gemv {

// Whether the GEMV path handles fp8_gemm_w8a16(A, B = W^T [K, N] NT view,
// B_scale, bias): fp16 A with one row, e4m3 [N, K] row-major W, 1-element
// fp32 scale, no bias, BMG device.
bool fp8_gemv_w8a16_supported_nt(
    const torch::Tensor& a,
    const torch::Tensor& b_kn,
    const std::optional<torch::Tensor>& scale,
    const std::optional<torch::Tensor>& bias);

// Runs the GEMV when supported, otherwise returns nullopt (caller keeps its
// oneDNN path).
std::optional<torch::Tensor> try_fp8_gemv_w8a16(
    const torch::Tensor& a,
    const torch::Tensor& b_kn,
    const std::optional<torch::Tensor>& scale,
    const std::optional<torch::Tensor>& bias);

// out = x @ (w * scale)^T, w [N, K] e4m3 row-major.
torch::Tensor fp8_gemv_w8a16(
    const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& scale);

// Two weights sharing x in one launch.
std::tuple<torch::Tensor, torch::Tensor> fp8_gemv2_w8a16(
    const torch::Tensor& x,
    const torch::Tensor& w1,
    const torch::Tensor& scale1,
    const torch::Tensor& w2,
    const torch::Tensor& scale2);

// fp8_gemm_w8a16 for two weights sharing A (both [K, N] NT views): one GEMV
// launch when both are supported, otherwise two fp8_gemm_w8a16 calls.
std::tuple<torch::Tensor, torch::Tensor> fp8_gemm_w8a16_pair(
    const torch::Tensor& a,
    const torch::Tensor& b1_kn,
    const torch::Tensor& scale1,
    const torch::Tensor& b2_kn,
    const torch::Tensor& scale2);

}  // namespace vllm::fp8_gemv
