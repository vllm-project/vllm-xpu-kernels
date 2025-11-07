#include <vector>
#include "fp8_gemm_w8a8.h"
#include "fp8_gemm_w8a16.h"
#include "int4_gemm_w4a16.h"

inline bool is_supported_fp8(at::ScalarType t) {
  return (t == at::ScalarType::Float8_e5m2) ||
         (t == at::ScalarType::Float8_e4m3fn);
}

torch::Tensor check_and_create_output_tensor(
    const torch::Tensor& A, const torch::Tensor& B,
    std::optional<c10::ScalarType> out_dtype) {
  TORCH_CHECK(A.dim() == 2 || A.dim() == 3,
              "OneDNN Matmul only support 2D and 3D inputs!\n");
  TORCH_CHECK(B.dim() == 2, "OneDNN Matmul only support 2D weights!\n");

  std::vector<int64_t> result_shape;
  if (A.dim() == 2) {
    result_shape = {A.size(0), B.size(1)};
    // src{m, k}, wei{k, n}, bias{n}, dst{m, n}
  } else {
    result_shape = {A.size(0), A.size(1), B.size(1)};
    // src{b, m, k}, wei{k, n}, bias{n}, dst{b, m, n}
  }

  // deal with input shape [m, b, k] stride [k, m * k, 1]
  auto k = A.size(A.dim() - 1);
  auto n = result_shape.back();
  auto res_stride = A.strides().vec();
  for (int i = 0; i < res_stride.size() - 1; i++) {
    res_stride[i] = res_stride[i] / k * n;
  }

  // If out_dtype is not given, use fp16 as default
  const auto out_dtype_ = out_dtype.value_or(torch::kHalf);
  auto options = A.options().dtype(out_dtype_);
  return at::empty_strided(result_shape, res_stride, options);
}

torch::Tensor fp8_gemm(const torch::Tensor& A,  // [b, m ,k]
                       const torch::Tensor& B,  // [k, n]
                       std::optional<c10::ScalarType> out_dtype,
                       const std::optional<torch::Tensor>& A_scale_,
                       const std::optional<torch::Tensor>& B_scale_,
                       const std::optional<torch::Tensor>& bias_) {
  const at::DeviceGuard device_guard(A.device());
  torch::Tensor result = check_and_create_output_tensor(A, B, out_dtype);
  auto a_st = A.scalar_type();
  auto b_st = B.scalar_type();
  TORCH_CHECK(is_supported_fp8(a_st) && is_supported_fp8(b_st) && a_st == b_st,
              "input and weight must be f8_e5m2 or f8_e4m3fn for fp8 matmul");
  TORCH_CHECK(result.scalar_type() == torch::kFloat16 ||
                  result.scalar_type() == torch::kBFloat16,
              "output must be float16 or bfloat16 for fp8 matmul");
  // check if nt format
  bool is_nt = B.strides()[B.dim() - 2] == 1;

  torch::Tensor A_scale = A_scale_.value_or(at::ones({1}, torch::kFloat));
  torch::Tensor B_scale = B_scale_.value_or(at::ones({1}, torch::kFloat));
  oneDNN::dnnl_matmul_w8a8_fp8(result, A, B, is_nt, bias_, A_scale, B_scale);
  return result;
}

torch::Tensor fp8_gemm_w8a16(const torch::Tensor& A, const torch::Tensor& B,
                             const std::optional<torch::Tensor>& B_scale_,
                             const std::optional<torch::Tensor>& bias_) {
  const at::DeviceGuard device_guard(A.device());
  torch::Tensor result = check_and_create_output_tensor(A, B, A.scalar_type());
  TORCH_CHECK(is_supported_fp8(B.scalar_type()),
              "weight must be f8_e5m2 or f8_e4m3fn for fp8 matmul");
  // check if nt format
  bool is_nt = B.strides()[B.dim() - 2] == 1;

  torch::Tensor B_scale = B_scale_.has_value()
                              ? B_scale_.value()
                              : at::ones({1}, B.options().dtype(A.dtype()));
  oneDNN::dnnl_matmul_w8a16_fp8(result, A, B, is_nt, bias_, B_scale);
  return result;
}

torch::Tensor int4_gemm_w4a16(
    const torch::Tensor& A_,  // src, [b, m, k]
    const torch::Tensor& B,   // quantized weight, [k, n]
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& B_scale,  // [k/group_size, n]
    const torch::Tensor& B_zp,     // [k/group_size, n/8]
    int64_t group_size, const std::optional<torch::Tensor>& g_idx) {
  const at::DeviceGuard device_guard(A_.device());

  // For GPTQ with desc_act=True scenario
  auto A = g_idx.has_value() ? A_.index_select(-1, g_idx.value()) : A_;
  torch::Tensor result = check_and_create_output_tensor(A, B, A.scalar_type());

  // check if nt format
  bool is_nt = B.strides()[B.dim() - 2] == 1;
  oneDNN::dnnl_matmul_w4a16_int4(result, A, B, is_nt, bias, B_scale, B_zp,
                                 group_size);
  return result;
}