

// #include <sycl/sycl.hpp>
// #include <cassert>
// #include <vector>

// #include <ATen/Tensor.h>
/* #include "pytorch_shim.h" */

#include <torch/all.h>

#include "collective/gemm/moe_dtype_policy.hpp"
#include "utils.h"

namespace gpu::cutlass_kernel {

namespace grouped_gemm {
template <class moe_policy>
void kernel_functor(sycl::queue& stream, void* ptr_A, void* ptr_B,
                    void* ptr_B_scale, void* ptr_bias, void* ptr_D,
                    void* expert_token_count, void* expert_first_token_offset,
                    int64_t N, int64_t K, int64_t groups);
extern template void kernel_functor<moe_bf16_policy>(
    sycl::queue& stream, void* ptr_A, void* ptr_B, void* ptr_B_scale,
    void* ptr_bias, void* ptr_D, void* expert_token_count,
    void* expert_first_token_offset, int64_t N, int64_t K, int64_t groups);
extern template void kernel_functor<moe_fp16_policy>(
    sycl::queue& stream, void* ptr_A, void* ptr_B, void* ptr_B_scale,
    void* ptr_bias, void* ptr_D, void* expert_token_count,
    void* expert_first_token_offset, int64_t N, int64_t K, int64_t groups);
extern template void kernel_functor<moe_e4m3fp16_policy>(
    sycl::queue& stream, void* ptr_A, void* ptr_B, void* ptr_B_scale,
    void* ptr_bias, void* ptr_D, void* expert_token_count,
    void* expert_first_token_offset, int64_t N, int64_t K, int64_t groups);
extern template void kernel_functor<moe_e5m2fp16_policy>(
    sycl::queue& stream, void* ptr_A, void* ptr_B, void* ptr_B_scale,
    void* ptr_bias, void* ptr_D, void* expert_token_count,
    void* expert_first_token_offset, int64_t N, int64_t K, int64_t groups);
extern template void kernel_functor<moe_e4m3bf16_policy>(
    sycl::queue& stream, void* ptr_A, void* ptr_B, void* ptr_B_scale,
    void* ptr_bias, void* ptr_D, void* expert_token_count,
    void* expert_first_token_offset, int64_t N, int64_t K, int64_t groups);
extern template void kernel_functor<moe_e5m2bf16_policy>(
    sycl::queue& stream, void* ptr_A, void* ptr_B, void* ptr_B_scale,
    void* ptr_bias, void* ptr_D, void* expert_token_count,
    void* expert_first_token_offset, int64_t N, int64_t K, int64_t groups);
}  // namespace grouped_gemm

/* gemm2(group_A, w2, output, offset) */

at::Tensor grouped_gemm_func(at::Tensor& ptr_A, at::Tensor& ptr_B,
                             const c10::optional<at::Tensor>& ptr_B_scale,
                             const c10::optional<at::Tensor>& ptr_bias,
                             at::Tensor& ptr_D, at::Tensor& expert_token_count,
                             at::Tensor& expert_first_token_offset, int64_t N,
                             int64_t K, int64_t groups) {
  auto& dpcpp_queue =
      at::xpu::getCurrentXPUStream(ptr_A.device().index()).queue();
  auto A_dtype = ptr_A.dtype();
  auto B_dtype = ptr_B.scalar_type();
  bool is_weight_fp8 =
      ((B_dtype == at::kFloat8_e4m3fn) || (B_dtype == at::kFloat8_e5m2));

  if (is_weight_fp8) {
    if (B_dtype == at::kFloat8_e4m3fn && A_dtype == at::kHalf) {
      using moe_policy = grouped_gemm::moe_e4m3fp16_policy;
      grouped_gemm::kernel_functor<moe_policy>(
          dpcpp_queue, ptr_A.data_ptr(), ptr_B.data_ptr(),
          ptr_B_scale.has_value() ? ptr_B_scale->data_ptr() : nullptr,
          ptr_bias.has_value() ? ptr_bias->data_ptr() : nullptr,
          ptr_D.data_ptr(), expert_token_count.data_ptr(),
          expert_first_token_offset.data_ptr(), N, K, groups);
    } else if (B_dtype == at::kFloat8_e5m2 && A_dtype == at::kHalf) {
      using moe_policy = grouped_gemm::moe_e5m2fp16_policy;
      grouped_gemm::kernel_functor<moe_policy>(
          dpcpp_queue, ptr_A.data_ptr(), ptr_B.data_ptr(),
          ptr_B_scale.has_value() ? ptr_B_scale->data_ptr() : nullptr,
          ptr_bias.has_value() ? ptr_bias->data_ptr() : nullptr,
          ptr_D.data_ptr(), expert_token_count.data_ptr(),
          expert_first_token_offset.data_ptr(), N, K, groups);
    } else if (B_dtype == at::kFloat8_e4m3fn && A_dtype == at::kBFloat16) {
      using moe_policy = grouped_gemm::moe_e4m3bf16_policy;
      grouped_gemm::kernel_functor<moe_policy>(
          dpcpp_queue, ptr_A.data_ptr(), ptr_B.data_ptr(),
          ptr_B_scale.has_value() ? ptr_B_scale->data_ptr() : nullptr,
          ptr_bias.has_value() ? ptr_bias->data_ptr() : nullptr,
          ptr_D.data_ptr(), expert_token_count.data_ptr(),
          expert_first_token_offset.data_ptr(), N, K, groups);
    } else if (B_dtype == at::kFloat8_e5m2 && A_dtype == at::kBFloat16) {
      using moe_policy = grouped_gemm::moe_e5m2bf16_policy;
      grouped_gemm::kernel_functor<moe_policy>(
          dpcpp_queue, ptr_A.data_ptr(), ptr_B.data_ptr(),
          ptr_B_scale.has_value() ? ptr_B_scale->data_ptr() : nullptr,
          ptr_bias.has_value() ? ptr_bias->data_ptr() : nullptr,
          ptr_D.data_ptr(), expert_token_count.data_ptr(),
          expert_first_token_offset.data_ptr(), N, K, groups);
    }
  } else if (A_dtype == at::kBFloat16) {
    using moe_policy = grouped_gemm::moe_bf16_policy;
    grouped_gemm::kernel_functor<moe_policy>(
        dpcpp_queue, ptr_A.data_ptr(), ptr_B.data_ptr(),
        ptr_B_scale.has_value() ? ptr_B_scale->data_ptr() : nullptr,
        ptr_bias.has_value() ? ptr_bias->data_ptr() : nullptr, ptr_D.data_ptr(),
        expert_token_count.data_ptr(), expert_first_token_offset.data_ptr(), N,
        K, groups);
  } else if (A_dtype == at::kHalf) {
    using moe_policy = grouped_gemm::moe_fp16_policy;
    grouped_gemm::kernel_functor<moe_policy>(
        dpcpp_queue, ptr_A.data_ptr(), ptr_B.data_ptr(),
        ptr_B_scale.has_value() ? ptr_B_scale->data_ptr() : nullptr,
        ptr_bias.has_value() ? ptr_bias->data_ptr() : nullptr, ptr_D.data_ptr(),
        expert_token_count.data_ptr(), expert_first_token_offset.data_ptr(), N,
        K, groups);
  } else {
    TORCH_CHECK(
        false,
        "grouped_gemm_func only supports BFloat16 and Half dtypes, but got: ",
        A_dtype);
  }
  return ptr_D;
}

}  // namespace gpu::cutlass_kernel
