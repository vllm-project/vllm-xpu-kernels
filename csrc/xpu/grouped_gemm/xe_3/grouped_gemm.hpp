#include <torch/all.h>
#include <limits>

#include "runner/greedy_dispatch.hpp"
#include "csrc/utils.h"

namespace gpu::cutlass_kernel {

/* gemm2(group_A, w2, output, offset) */

at::Tensor grouped_gemm_func(
    at::Tensor& ptr_A,
    const c10::optional<at::Tensor>& ptr_A_scale,
    at::Tensor& ptr_B,
    const c10::optional<at::Tensor>& ptr_B_scale,
    const c10::optional<at::Tensor>& ptr_bias,
    at::Tensor& ptr_D,
    at::Tensor& rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t groups) {
  auto& dpcpp_queue =
      at::xpu::getCurrentXPUStream(ptr_A.device().index()).queue();
  auto A_dtype = ptr_A.dtype();

  // Every stride is rebuilt from {N, K, groups} via make_cute_packed_stride()
  // and the tensors' own strides are never read, so a mismatched layout is
  // silently wrong instead of an error. LayoutA/C/D are RowMajor for all
  // policies; LayoutB is RowMajor (dense along N) except for the 4-bit weight
  // policies (mxfp4, w4a8) which are ColumnMajor (dense along K). Both scale
  // surfaces are MN-major.
  const bool a_is_fp4 = A_dtype == at::kFloat4_e2m1fn_x2;
  const bool b_is_fp4 = ptr_B.dtype() == at::kFloat4_e2m1fn_x2;
  const int64_t A_K = a_is_fp4 ? K / 2 : K;

  TORCH_CHECK(groups > 0 && N > 0 && K > 0, "N, K and num_experts must be > 0");
  TORCH_CHECK(!a_is_fp4 || K % 2 == 0, "K must be even for 4-bit ptr_A");

  // The XE 2D block copies need each row of B and D to start on a 128-bit
  // boundary, and cutlass reports a violation by aborting the process rather
  // than returning, so reject it here instead.
  const int64_t b_bits = ptr_B.dtype().itemsize() * 8 / (b_is_fp4 ? 2 : 1);
  const int64_t a_bits = ptr_A.dtype().itemsize() * 8 / (a_is_fp4 ? 2 : 1);
  const int64_t d_bits = ptr_D.dtype().itemsize() * 8;
  TORCH_CHECK(
      (N * b_bits) % 128 == 0 && (N * d_bits) % 128 == 0,
      "N must be a multiple of 128 bits of the weight and output element type, "
      "so N must be a multiple of 32 for 4-bit, 16 for 8-bit and 8 for 16-bit "
      "weights");
  TORCH_CHECK(
      K % (128 / a_bits) == 0 && (!b_is_fp4 || K % (128 / b_bits) == 0),
      "K must provide 128-bit aligned activation and packed-weight rows");

  TORCH_CHECK(
      ptr_A.dim() == 2 && ptr_A.size(1) == A_K && ptr_A.is_contiguous(),
      "ptr_A must be a contiguous 2D [total_M, K] tensor, with K halved when "
      "it is 4-bit packed");

  TORCH_CHECK(
      ptr_D.dim() == 2 && ptr_D.size(0) == ptr_A.size(0) &&
          ptr_D.size(1) == N && ptr_D.is_contiguous(),
      "ptr_D must be a contiguous 2D [total_M, N] tensor");

  TORCH_CHECK(
      ptr_B.dim() == 3 && ptr_B.size(0) == groups,
      "ptr_B must be 3D with one leading entry per expert");
  if (b_is_fp4) {
    TORCH_CHECK(
        ptr_B.size(1) == N && ptr_B.size(2) == K / 2 && ptr_B.is_contiguous(),
        "4-bit ptr_B must be a contiguous [num_experts, N, K/2] tensor");
  } else {
    // Element (e, n, k) must sit at e * N * K + n + k * N. Callers spell that
    // either as a contiguous (E, K, N) or as an (E, N, K) view of it.
    TORCH_CHECK(
        (ptr_B.size(1) == K && ptr_B.size(2) == N && ptr_B.is_contiguous()) ||
            (ptr_B.size(1) == N && ptr_B.size(2) == K &&
             ptr_B.stride(0) == N * K && (N == 1 || ptr_B.stride(1) == 1) &&
             (K == 1 || ptr_B.stride(2) == N)),
        "ptr_B must be dense along N: either a contiguous [num_experts, K, N] "
        "tensor or its [num_experts, N, K] transposed view");
  }

  TORCH_CHECK(
      rows_per_expert.dim() == 1 && rows_per_expert.numel() == groups &&
          rows_per_expert.scalar_type() == at::kInt &&
          rows_per_expert.is_contiguous(),
      "rows_per_expert must be a contiguous 1D int32 tensor with one entry "
      "per expert");

  // cutlass_grouped_gemm_xe3() has already expanded the caller's
  // [num_experts, N] bias into one fp32 row per token (ElementC).
  if (ptr_bias) {
    TORCH_CHECK(
        ptr_bias->dim() == 2 && ptr_bias->size(0) == ptr_A.size(0) &&
            ptr_bias->size(1) == N && ptr_bias->is_contiguous() &&
            ptr_bias->scalar_type() == at::kFloat,
        "ptr_bias must expand to a contiguous fp32 [total_M, N] tensor, so the "
        "caller's bias must be [num_experts, N]");
  }

  if (ptr_A_scale && ptr_A_scale->dtype() == at::kFloat8_e8m0fnu) {
    const int64_t scale_k = (K + 31) / 32;
    TORCH_CHECK(
        ptr_B_scale && ptr_B_scale->dtype() == at::kFloat8_e8m0fnu,
        "block-scaled grouped GEMM requires an e8m0 ptr_B_scale");
    // reorder_mxfp_scales() hands over a plain [padded_M, scale_k] buffer that
    // it filled MN-major, so only its extent is meaningful here. Each expert's
    // row count is rounded up to a multiple of 4.
    TORCH_CHECK(
        ptr_A_scale->dim() == 2 && ptr_A_scale->size(1) == scale_k &&
            ptr_A_scale->is_contiguous(),
        "ptr_A_scale must be a contiguous [padded_M, ceil(K/32)] tensor");
    TORCH_CHECK(
        ptr_A_scale->size(0) >= ptr_A.size(0) &&
            ptr_A_scale->size(0) <= ptr_A.size(0) + 3 * groups,
        "ptr_A_scale must have total_M rows with each expert padded up to a "
        "multiple of 4");
    // Element (e, n, k) must sit at e * N * scale_k + n + k * N.
    TORCH_CHECK(
        ptr_B_scale->dim() == 3 && ptr_B_scale->size(0) == groups &&
            ((ptr_B_scale->size(1) == scale_k && ptr_B_scale->size(2) == N &&
              ptr_B_scale->is_contiguous()) ||
             (ptr_B_scale->size(1) == N && ptr_B_scale->size(2) == scale_k &&
              ptr_B_scale->stride(0) == N * scale_k &&
              (N == 1 || ptr_B_scale->stride(1) == 1) &&
              (scale_k == 1 || ptr_B_scale->stride(2) == N))),
        "ptr_B_scale must be dense along N: either a contiguous "
        "[num_experts, ceil(K/32), N] tensor or its [num_experts, N, "
        "ceil(K/32)] transposed view");
  } else if (ptr_A_scale && ptr_A_scale->dtype() == at::kFloat) {
    TORCH_CHECK(
        ptr_B_scale && ptr_B_scale->dtype() == at::kFloat,
        "fp8 grouped GEMM requires a float32 ptr_B_scale");
    if (ptr_A_scale->numel() == 1) {
      TORCH_CHECK(
          ptr_B_scale->numel() == groups,
          "per-tensor fp8 ptr_B_scale must hold one scale per expert");
    } else {
      const int64_t scale_k = (K + 127) / 128;
      const int64_t scale_n = (N + 127) / 128;
      TORCH_CHECK(
          ptr_A_scale->numel() == ptr_A.size(0) * scale_k,
          "block fp8 ptr_A_scale must hold total_M * ceil(K/128) scales");
      TORCH_CHECK(
          ptr_B_scale->dim() == 3 && ptr_B_scale->size(0) == groups &&
              ptr_B_scale->size(1) == scale_n &&
              ptr_B_scale->size(2) == scale_k && ptr_B_scale->is_contiguous(),
          "block fp8 ptr_B_scale must be a contiguous [num_experts, "
          "ceil(N/128), ceil(K/128)] tensor");
    }
  }

  TORCH_CHECK(
      N <= std::numeric_limits<int>::max() &&
          K <= std::numeric_limits<int>::max() &&
          groups <= std::numeric_limits<int>::max(),
      "Xe3 greedy GEMM requires N, K and num_experts to fit in int32");

#define CALL_GREEDY(DTYPE)                                       \
  grouped_gemm::launch_greedy<grouped_gemm::GreedyDtype::DTYPE>( \
      dpcpp_queue,                                               \
      {ptr_A.data_ptr(),                                         \
       ptr_A_scale ? ptr_A_scale->data_ptr() : nullptr,          \
       ptr_B.data_ptr(),                                         \
       ptr_B_scale ? ptr_B_scale->data_ptr() : nullptr,          \
       ptr_bias ? ptr_bias->data_ptr<float>() : nullptr,         \
       ptr_D.data_ptr(),                                         \
       rows_per_expert.data_ptr<int>(),                          \
       int(N),                                                   \
       int(K),                                                   \
       int(groups),                                              \
       int64_t(ptr_A.size(0))})

  if (A_dtype == at::kBFloat16) {
    CALL_GREEDY(BF16);
  } else if (A_dtype == at::kHalf) {
    CALL_GREEDY(FP16);
  } else if (
      A_dtype == at::kFloat8_e4m3fn && ptr_A_scale &&
      ptr_A_scale->dtype() == at::kFloat8_e8m0fnu) {
    if (ptr_B.dtype() == at::kFloat4_e2m1fn_x2) {
      // W4A8: MXFP8 activation (A=e4m3) x MXFP4 weight (B=e2m1). Same e8m0
      // block scales as the symmetric recipes; only the weight is 4-bit.
      CALL_GREEDY(W4A8);
    } else {
      CALL_GREEDY(MXFP8);
    }
  } else if (
      A_dtype == at::kFloat4_e2m1fn_x2 && ptr_A_scale &&
      ptr_A_scale->dtype() == at::kFloat8_e8m0fnu) {
    CALL_GREEDY(MXFP4);
  } else if (
      A_dtype == at::kFloat8_e4m3fn && ptr_A_scale &&
      ptr_A_scale->dtype() == at::kFloat) {
    // Per-tensor FP8 carries a single scalar A scale [1] and one scalar per
    // expert [E]; anything else is the 128x128 block-scaled recipe.
    if (ptr_A_scale->numel() == 1) {
      CALL_GREEDY(FP8Tensor);
    } else {
      CALL_GREEDY(FP8Block);
    }
  } else {
    TORCH_CHECK(
        false,
        "grouped_gemm_func only supports BF16/FP16/MXFP8/MXFP4/FP8(block) "
        "dtypes, but got: ",
        A_dtype);
  }
#undef CALL_GREEDY
  return ptr_D;
}

}  // namespace gpu::cutlass_kernel
