// Stub implementations for the oneDNN-backed ops. These are compiled instead
// of the real oneDNN sources when the build is configured with
// VLLM_XPU_ENABLE_ONEDNN=OFF. Each stub keeps the op registered so that
// torch.ops._xpu_C.<op> still resolves, but raises a clear runtime error
// telling the user how to enable the real kernels.
#include "../ops.h"

namespace {

[[noreturn]] void onednn_not_built(const char* op_name) {
  TORCH_CHECK(
      false,
      op_name,
      ": oneDNN kernels are not built. Rebuild with "
      "VLLM_XPU_ENABLE_ONEDNN=ON (or pass -DVLLM_XPU_ENABLE_ONEDNN=ON to "
      "CMake) to enable this op.");
}

}  // namespace

torch::Tensor fp8_gemm(
    const torch::Tensor& A,
    const torch::Tensor& B,
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& A_scale_,
    const std::optional<torch::Tensor>& B_scale_,
    const std::optional<torch::Tensor>& bias_) {
  onednn_not_built("fp8_gemm");
}

torch::Tensor fp8_gemm_out(
    torch::Tensor out,
    const torch::Tensor& A,
    const torch::Tensor& B,
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& A_scale_,
    const std::optional<torch::Tensor>& B_scale_,
    const std::optional<torch::Tensor>& bias_) {
  onednn_not_built("fp8_gemm_out");
}

torch::Tensor fp8_bmm(
    const torch::Tensor& A,
    const torch::Tensor& B,
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& A_scale_,
    const std::optional<torch::Tensor>& B_scale_,
    const std::optional<torch::Tensor>& bias_) {
  onednn_not_built("fp8_bmm");
}

torch::Tensor fp8_gemm_w8a16(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const std::optional<torch::Tensor>& B_scale_,
    const std::optional<torch::Tensor>& bias_) {
  onednn_not_built("fp8_gemm_w8a16");
}

torch::Tensor fp4_gemm(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& A_scale,
    const torch::Tensor& B_scale,
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& bias) {
  onednn_not_built("fp4_gemm");
}

torch::Tensor int4_gemm_w4a16(
    const torch::Tensor& A_,
    const torch::Tensor& B,
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& B_scale,
    const torch::Tensor& B_zp,
    int64_t group_size,
    const std::optional<torch::Tensor>& g_idx) {
  onednn_not_built("int4_gemm_w4a16");
}

torch::Tensor int4_gemm_w4a8(
    const torch::Tensor& A_,
    const torch::Tensor& A_scale,
    const torch::Tensor& A_zp,
    const torch::Tensor& B,
    const torch::Tensor& B_scale,
    const torch::Tensor& B_zp,
    int64_t group_size,
    const std::optional<torch::Tensor>& g_idx,
    const std::optional<torch::Tensor>& bias) {
  onednn_not_built("int4_gemm_w4a8");
}

std::string get_onednn_version() { onednn_not_built("get_onednn_version"); }
