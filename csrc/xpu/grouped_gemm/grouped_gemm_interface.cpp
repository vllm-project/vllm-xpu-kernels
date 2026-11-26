#include "csrc/utils.h"
#include "grouped_gemm_interface.h"
#include <stdio.h>
#include <c10/core/DeviceGuard.h>
#include <limits>

#ifdef VLLM_XPU_ENABLE_XE2
  #include "xe_2/grouped_gemm_xe2.h"
#endif
#ifdef VLLM_XPU_ENABLE_XE3P
  #include "xe_3/grouped_gemm_xe3.h"
#endif

torch::Tensor cutlass_grouped_gemm_interface(
    torch::Tensor ptr_A,
    const c10::optional<at::Tensor>& ptr_A_scale,
    torch::Tensor ptr_B,
    const c10::optional<at::Tensor>& ptr_B_scale,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t num_experts) {
  const bool force_xe3p = vllm::xpu::force_xe3p_kernel();
#ifdef VLLM_XPU_ENABLE_XE3P
  if (force_xe3p || vllm::xpu::is_xe3p_arch()) {
    // Xe3P consumes block-scaled activations (MXFP8 / MXFP4 / block-FP8)
    // natively, so ptr_A_scale is forwarded to the device kernel.
    return cutlass_grouped_gemm_xe3(
        ptr_A,
        ptr_A_scale,
        ptr_B,
        ptr_B_scale,
        ptr_bias,
        ptr_D,
        rows_per_expert,
        N,
        K,
        num_experts);
  }

#else
  TORCH_CHECK(
      !force_xe3p,
      "FORCE_XE3P_KERNEL is set, but XE3P grouped GEMM is not enabled in "
      "this build.");
#endif
#ifdef VLLM_XPU_ENABLE_XE2
  // BMG / PVC / LNL and the Xe3 client parts run the xe_2 kernel, which
  // consumes high-precision activations only.
  (void)ptr_A_scale;
  return cutlass_grouped_gemm_xe2(
      ptr_A,
      ptr_B,
      ptr_B_scale,
      ptr_bias,
      ptr_D,
      rows_per_expert,
      N,
      K,
      num_experts);
#else
  TORCH_CHECK(
      false, "No cutlass grouped GEMM kernel is enabled in this build.");
#endif
}

void cutlass_batched_gemm_interface(
    torch::Tensor input,
    torch::Tensor weight,
    const c10::optional<at::Tensor>& bias,
    torch::Tensor output,
    torch::Tensor rows_per_expert,
    int64_t tile_m) {
  TORCH_CHECK(input.is_xpu(), "input must be an XPU tensor");
  TORCH_CHECK(
      tile_m == 0 || tile_m == 8 || tile_m == 16 || tile_m == 32 ||
          tile_m == 64 || tile_m == 128 || tile_m == 256,
      "tile_m must be 0 (auto), 8, 16, 32, 64, 128 or 256");
  const c10::DeviceGuard device_guard(input.device());
  TORCH_CHECK(
      weight.device() == input.device() && output.device() == input.device() &&
          rows_per_expert.device() == input.device(),
      "all tensors must be on the same XPU device");
  TORCH_CHECK(
      input.dim() == 3 && weight.dim() == 3 && output.dim() == 3,
      "expected input [E,T,K], weight [E,K,N], output [E,T,N]");
  TORCH_CHECK(
      input.is_contiguous() && weight.is_contiguous() &&
          output.is_contiguous() && rows_per_expert.is_contiguous(),
      "batched GEMM tensors must be contiguous");
  TORCH_CHECK(
      (input.scalar_type() == at::kHalf ||
       input.scalar_type() == at::kBFloat16) &&
          weight.scalar_type() == input.scalar_type() &&
          output.scalar_type() == input.scalar_type(),
      "input, weight and output must have the same FP16 or BF16 dtype");
  const auto E = input.size(0);
  const auto T = input.size(1);
  const auto K = input.size(2);
  const auto N = weight.size(2);
  TORCH_CHECK(E > 0 && K > 0 && N > 0, "E, K and N must be positive");
  TORCH_CHECK(K % 8 == 0 && N % 8 == 0, "K and N must be divisible by 8");
  TORCH_CHECK(
      weight.size(0) == E && weight.size(1) == K && output.size(0) == E &&
          output.size(1) == T && output.size(2) == N,
      "incompatible batched GEMM shapes");
  TORCH_CHECK(
      rows_per_expert.dim() == 1 && rows_per_expert.size(0) == E &&
          rows_per_expert.scalar_type() == at::kInt,
      "rows_per_expert must be int32 [E]");
  TORCH_CHECK(
      E <= std::numeric_limits<int>::max() &&
          T <= std::numeric_limits<int>::max() / E &&
          K <= std::numeric_limits<int>::max() &&
          N <= std::numeric_limits<int>::max(),
      "batched GEMM dimensions exceed int32 indexing");
  if (bias.has_value()) {
    TORCH_CHECK(
        bias->device() == input.device() &&
            bias->scalar_type() == input.scalar_type() &&
            bias->is_contiguous() && bias->dim() == 2 && bias->size(0) == E &&
            bias->size(1) == N,
        "bias must be contiguous [E,N] with the input device and dtype");
  }
  TORCH_CHECK(
      !vllm::xpu::force_xe3p_kernel() &&
          vllm::xpu::is_xe2_arch(input.get_device()),
      "batched GEMM currently supports only Xe2 devices");
#ifdef VLLM_XPU_ENABLE_XE2
  cutlass_batched_gemm_xe2(
      input, weight, bias, output, rows_per_expert, tile_m);
#else
  TORCH_CHECK(false, "XE2 kernels are not enabled in this build.");
#endif
}
