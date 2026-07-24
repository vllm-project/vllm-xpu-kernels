#include "csrc/utils.h"
#include "fused_moe_up.h"
#include "fused_moe_up_dispatch.h"

namespace {

FusedMOE::FusedMOEUpDispatchFn get_fused_moe_up_dispatch(
    FusedMOE::FusedMOEWeightType weight_type, const std::string& activation) {
  using namespace FusedMOE;

#define SELECT_ACTIVATION(WeightType, suffix)    \
  if (activation == "silu") {                    \
    return fused_moe_up_##suffix##_silu;         \
  } else if (activation == "gelu") {             \
    return fused_moe_up_##suffix##_gelu;         \
  } else if (activation == "gelu_tanh") {        \
    return fused_moe_up_##suffix##_gelu_tanh;    \
  } else if (activation == "swigluoai") {        \
    return fused_moe_up_##suffix##_swigluoai;    \
  } else if (activation == "relu2_no_mul") {     \
    return fused_moe_up_##suffix##_relu2_no_mul; \
  } else if (activation == "swiglustep") {       \
    return fused_moe_up_##suffix##_swiglustep;   \
  }

  switch (weight_type) {
    case FusedMOEWeightType::W4A16:
      SELECT_ACTIVATION(W4A16, w4a16)
      break;
    case FusedMOEWeightType::W8A16:
      SELECT_ACTIVATION(W8A16, w8a16)
      break;
    case FusedMOEWeightType::W16A16:
      SELECT_ACTIVATION(W16A16, w16a16)
      break;
  }

#undef SELECT_ACTIVATION
  return nullptr;
}

}  // namespace

torch::Tensor fused_moe_up(
    torch::Tensor& ptr_A,
    const c10::optional<at::Tensor>& ptr_A_scale,
    torch::Tensor& ptr_B,
    const c10::optional<at::Tensor>& ptr_B_scale,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor& ptr_D,
    torch::Tensor& rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    std::string activation,
    double gemm1_clamp_limit) {
  auto& dpcpp_queue =
      at::xpu::getCurrentXPUStream(ptr_A.device().index()).queue();
  auto A_dtype = ptr_A.dtype();
  auto B_dtype = ptr_B.dtype();
  bool is_weight_fp8 =
      ((B_dtype == at::kFloat8_e4m3fn) || (B_dtype == at::kFloat8_e5m2));
  bool is_B_int4 = (B_dtype == at::kChar) && ptr_B_scale.has_value();
  bool is_B_mxfp4 =
      (B_dtype == at::kFloat4_e2m1fn_x2) && ptr_B_scale.has_value();

  TORCH_CHECK(ptr_A.dim() == 2, "ptr_A must be 2D [Total_M, K]");
  TORCH_CHECK(ptr_B.dim() == 3, "ptr_B must be 3D [num_experts, K, 2 * N]");
  TORCH_CHECK(ptr_D.dim() == 2, "ptr_D must be 2D [Total_M, N]");

  if (ptr_bias.has_value()) {
    TORCH_CHECK(
        ptr_bias->dim() == 2, "ptr_bias must be 2D [num_experts, 2 * N]");
  }

  TORCH_CHECK(ptr_A.is_contiguous(), "ptr_A must be contiguous");
  TORCH_CHECK(ptr_B.is_contiguous(), "ptr_B must be contiguous");
  TORCH_CHECK(ptr_D.is_contiguous(), "ptr_D must be contiguous");
  if (ptr_bias.has_value()) {
    TORCH_CHECK(ptr_bias->is_contiguous(), "ptr_bias must be contiguous");
  }

  int A_total_M = ptr_A.size(0);
  int A_K = ptr_A.size(1);
  int B_E = ptr_B.size(0);
  int B_K = ptr_B.size(1);
  int B_N = ptr_B.size(2);
  if (is_B_int4 || is_B_mxfp4) {
    B_K = ptr_B.size(2) * 2;
    B_N = ptr_B.size(1);
  }

  int D_total_M = ptr_D.size(0);
  int D_N = ptr_D.size(1);
  int group_size = -1;
  int A_avg_M = A_total_M / num_experts;

  TORCH_CHECK(B_E == num_experts, "ptr_B.size(0) must match num_experts");
  TORCH_CHECK(A_total_M == D_total_M, "ptr_A.size(0) must match ptr_D.size(0)");
  TORCH_CHECK(A_K == B_K && B_K == K, "ptr_A.size(1) must match ptr_B.size(1)");
  TORCH_CHECK(
      B_N == 2 * D_N && D_N == N,
      "ptr_B.size(2) must match double ptr_D.size(1)");

  if (ptr_bias.has_value()) {
    TORCH_CHECK(
        ptr_bias->size(0) == num_experts,
        "ptr_bias.size(0) must match num_experts");
    TORCH_CHECK(
        ptr_bias->size(1) == 2 * N, "ptr_bias.size(1) must match double N");
  }

  at::Tensor atomic_buffer =
      at::empty({static_cast<long>(1)}, ptr_A.options().dtype(at::kInt));
  const bool has_clamping = gemm1_clamp_limit > 0;

  if (is_B_int4 || is_B_mxfp4) {
    TORCH_CHECK(ptr_B_scale.has_value(), "w4a16 grouped gemm must have scales");
    TORCH_CHECK(ptr_B_scale->is_contiguous(), "ptr_B_scale must be contiguous");
    TORCH_CHECK(
        ptr_B_scale->dim() == 3,
        "ptr_B_scale of int4 must be 3D [num_experts, 2 * N, group_num]");
    TORCH_CHECK(
        ptr_B_scale->size(0) == num_experts,
        "ptr_B_scale.size(0) of int4 must match num_experts");
    TORCH_CHECK(
        K % ptr_B_scale->size(2) == 0,
        "K must be divisible by ptr_B_scale.size(2) (group_num) of int4");
    TORCH_CHECK(
        ptr_B_scale->size(1) == 2 * N,
        "ptr_B_scale.size(1) of int4 must match 2 * N");
    int group_num = ptr_B_scale->size(2);
    group_size = K / group_num;

    TORCH_CHECK(
        group_size == 32 || group_size == 64 || group_size == 128 ||
            group_size == 256,
        "group_size must be 32, 64, 128 or 256");
  } else if (is_weight_fp8) {
    TORCH_CHECK(ptr_B_scale.has_value(), "w8a16 grouped gemm must have scales");
    TORCH_CHECK(ptr_B_scale->is_contiguous(), "ptr_B_scale must be contiguous");
    TORCH_CHECK(
        ptr_B_scale->dim() == 1, "ptr_B_scale of fp8 must be 1D [num_experts]");
    TORCH_CHECK(
        ptr_B_scale->size(0) == num_experts,
        "ptr_B_scale.size(0) of fp8 must match num_experts");
    TORCH_CHECK(
        ptr_B_scale->dtype() == at::kFloat, "ptr_B_scale must be float");
  } else {
    TORCH_CHECK(
        !ptr_B_scale.has_value(), "w16a16 grouped gemm must not have scales");
  }

  if (A_dtype != at::kBFloat16 && A_dtype != at::kHalf) {
    return ptr_D;
  }

  FusedMOE::FusedMOEWeightType weight_type =
      is_B_int4 || is_B_mxfp4 ? FusedMOE::FusedMOEWeightType::W4A16
      : is_weight_fp8         ? FusedMOE::FusedMOEWeightType::W8A16
                              : FusedMOE::FusedMOEWeightType::W16A16;
  FusedMOE::FusedMOEUpLaunchParams params{
      dpcpp_queue,
      ptr_A.data_ptr(),
      ptr_B.data_ptr(),
      ptr_B_scale.has_value() ? ptr_B_scale->data_ptr() : nullptr,
      ptr_bias.has_value() ? ptr_bias->data_ptr() : nullptr,
      ptr_D.data_ptr(),
      reinterpret_cast<int*>(rows_per_expert.data_ptr()),
      static_cast<int>(N),
      static_cast<int>(K),
      static_cast<int>(num_experts),
      group_size,
      static_cast<int*>(atomic_buffer.data_ptr()),
      gemm1_clamp_limit,
      A_avg_M,
      B_N,
      has_clamping,
      A_dtype == at::kBFloat16,
      is_B_int4 || B_dtype == at::kFloat8_e4m3fn};
  auto dispatch = get_fused_moe_up_dispatch(weight_type, activation);
  TORCH_CHECK(dispatch != nullptr, "Unsupported activation: ", activation);
  dispatch(params);

  return ptr_D;
}
