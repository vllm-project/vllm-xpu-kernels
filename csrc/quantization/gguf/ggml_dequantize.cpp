#include "ggml_dequantize.hpp"
#include "utils.h"
#include "xpu/ops.h"

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>

namespace ggml = vllm::ggml;

torch::Tensor ggml_dequantize(
    const torch::Tensor& W,
    int64_t type,
    int64_t m,
    int64_t n,
    std::optional<c10::ScalarType> out_dtype) {
  CHECK_DEVICE(W);
  CHECK_CONTIGUOUS(W);

  TORCH_CHECK(
      type == ggml::GGML_TYPE_Q4_0 || type == ggml::GGML_TYPE_Q5_0 ||
          type == ggml::GGML_TYPE_Q8_0,
      "XPU ggml_dequantize currently only supports Q4_0 (type=2), "
      "Q5_0 (type=6) and Q8_0 (type=8), got ", type);
  TORCH_CHECK(
      W.scalar_type() == at::ScalarType::Byte,
      "XPU ggml_dequantize expects uint8 weights, got ", W.scalar_type());
  TORCH_CHECK(m >= 0 && n >= 0, "m and n must be non-negative");

  const int64_t numel = m * n;
  const int64_t quant_block_size = ggml::get_quant_block_size(type);
  TORCH_CHECK(
      numel % quant_block_size == 0, ggml::ggml_type_name(type),
      " dequantize expects m * n to be divisible by ", quant_block_size,
      ", got ", numel);

  const int64_t expected_nbytes = ggml::get_expected_nbytes(type, numel);
  const int64_t weight_nbytes = W.numel() * W.element_size();
  TORCH_CHECK(
      weight_nbytes == expected_nbytes, ggml::ggml_type_name(type),
      " packed weight size mismatch: expected ", expected_nbytes,
      " bytes for shape (", m, ", ", n, "), got ", weight_nbytes, " bytes");

  const auto dtype = out_dtype.value_or(torch::kFloat16);
  TORCH_CHECK(
      dtype == torch::kFloat16 || dtype == torch::kBFloat16 ||
          dtype == torch::kFloat32,
      "XPU ggml_dequantize only supports fp16, bf16 or fp32 outputs, got ",
      dtype);

  auto options = torch::TensorOptions().dtype(dtype).device(W.device());
  auto output = torch::empty({m, n}, options);
  if (numel == 0) {
    return output;
  }

  at::DeviceGuard device_guard(W.device());
  auto& queue = vllm::xpu::vllmGetQueue(W.device().index());
  const auto* weight_ptr = W.data_ptr<uint8_t>();

  VLLM_DISPATCH_FLOATING_TYPES(output.scalar_type(), "ggml_dequantize", [&] {
    using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
    auto* out_ptr = reinterpret_cast<sycl_t*>(output.data_ptr<scalar_t>());

    switch (type) {
      case ggml::GGML_TYPE_Q4_0: {
        auto* blocks = reinterpret_cast<const ggml::block_q4_0*>(weight_ptr);
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::range<1>(static_cast<size_t>(numel)),
              ggml::ggml_dequantize_q4_0_kernel<scalar_t>(
                  blocks, out_ptr, numel));
        });
        break;
      }
      case ggml::GGML_TYPE_Q5_0: {
        auto* blocks = reinterpret_cast<const ggml::block_q5_0*>(weight_ptr);
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::range<1>(static_cast<size_t>(numel)),
              ggml::ggml_dequantize_q5_0_kernel<scalar_t>(
                  blocks, out_ptr, numel));
        });
        break;
      }
      case ggml::GGML_TYPE_Q8_0: {
        auto* blocks = reinterpret_cast<const ggml::block_q8_0*>(weight_ptr);
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::range<1>(static_cast<size_t>(numel)),
              ggml::ggml_dequantize_q8_0_kernel<scalar_t>(
                  blocks, out_ptr, numel));
        });
        break;
      }
      default:
        TORCH_CHECK(
            false, "Unsupported GGML type for XPU ggml_dequantize: ", type);
    }
  });

  return output;
}