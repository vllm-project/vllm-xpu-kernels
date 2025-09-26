#include <vector>
#include "fp8_gemm_w8a16.h"
#include "int4_gemm_w4a16.h"

torch::Tensor check_and_create_output_tensor(const torch::Tensor& A,
                                             const torch::Tensor& B,
                                             bool trans_B) {
  TORCH_CHECK(A.dim() == 2 || A.dim() == 3,
              "OneDNN Matmul only support 2D and 3D inputs!\n");
  TORCH_CHECK(B.dim() == 2, "OneDNN Matmul only support 2D weights!\n");

  std::vector<int64_t> result_shape;
  if (A.dim() == 2) {
    if (trans_B) {
      result_shape = {A.size(0), B.size(0)};
    } else {
      result_shape = {A.size(0), B.size(1)};
    }
    // src{m, k}, wei{k, n}, bias{n}, dst{m, n}
  } else {
    if (trans_B) {
      result_shape = {A.size(0), A.size(1), B.size(0)};
    } else {
      result_shape = {A.size(0), A.size(1), B.size(1)};
    }
    // src{b, m, k}, wei{k, n}, bias{n}, dst{b, m, n}
  }

  // deal with input shape [m, b, k] stride [k, m * k, 1]
  auto k = A.size(A.dim() - 1);
  auto n = result_shape.back();
  auto res_stride = A.strides().vec();
  for (int i = 0; i < res_stride.size() - 1; i++) {
    res_stride[i] = res_stride[i] / k * n;
  }

  return at::empty_strided(result_shape, res_stride, A.options());
}

torch::Tensor fp8_gemm_w8a16(const torch::Tensor& A, const torch::Tensor& B,
                             bool trans_B,
                             const std::optional<torch::Tensor>& B_scale_,
                             const std::optional<torch::Tensor>& bias_) {
  torch::Tensor result = check_and_create_output_tensor(A, B, trans_B);
  TORCH_CHECK(B.scalar_type() == at::ScalarType::Float8_e5m2 ||
                  B.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "weight must be f8_e5m2 or f8_e4m3fn for fp8 matmul");
  // check if nt format
  bool is_nt =
      trans_B ? B.strides()[B.dim() - 1] == 1 : B.strides()[B.dim() - 2] == 1;

  torch::Tensor B_scale = B_scale_.has_value()
                              ? B_scale_.value()
                              : at::ones({1}, B.options().dtype(A.dtype()));
  oneDNN::dnnl_matmul_w8a16_fp8(result, A, B, is_nt, bias_, B_scale);
  return result;
}

torch::Tensor int4_gemm_w4a16(
    const torch::Tensor& A_,  // src, [b, m, k]
    const torch::Tensor& B,   // quantized weight, [k, n] transpose
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& B_scale,  // [k/group_size, n]
    const torch::Tensor& B_zp,     // [k/group_size, n/8]
    int64_t group_size, bool trans_B,
    const std::optional<torch::Tensor>& g_idx) {
  // For GPTQ with desc_act=True scenario
  auto A = g_idx.has_value() ? A_.index_select(-1, g_idx.value()) : A_;
  torch::Tensor result = check_and_create_output_tensor(A, B, trans_B);

  // check if nt format
  bool is_nt =
      trans_B ? B.strides()[B.dim() - 1] == 1 : B.strides()[B.dim() - 2] == 1;
  oneDNN::dnnl_matmul_w4a16_int4(result, A, B, is_nt, bias, B_scale, B_zp,
                                 group_size);
  return result;
}