#include "core/registration.h"
#include "xpu/ops.h"
#include "xpu/cutlass_kernels/grouped_gemm.hpp"
#include "xpu/lora/lora_ops.h"

#include <torch/library.h>
#include <torch/version.h>

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, xpu_ops) {
  at::Tag stride_tag = at::Tag::needs_fixed_stride_order;

  xpu_ops.def(
      "fp8_gemm_w8a16(Tensor! A, Tensor! B, bool trans_B, Tensor? B_scale_, "
      "Tensor? bias_) -> Tensor");
  xpu_ops.impl("fp8_gemm_w8a16", torch::kXPU, &fp8_gemm_w8a16);

  xpu_ops.def(
      "cutlass_grouped_gemm(Tensor ptr_A, Tensor ptr_B, Tensor ptr_D, Tensor "
      "ptr_alpha, Tensor ptr_beta, Tensor offset, int N, int K, int groups) -> "
      "Tensor");
  xpu_ops.impl("cutlass_grouped_gemm", torch::kXPU,
               gpu::cutlass_kernel::grouped_gemm_func);

  xpu_ops.def(
      "deepseek_scaling_rope(Tensor! positions, Tensor! query, Tensor! key, "
      "Tensor? offsets_opt, Tensor! cos_sin_cache, int rotary_dim, bool "
      "is_neox_style) "
      "-> (Tensor, Tensor)");
  xpu_ops.impl("deepseek_scaling_rope", torch::kXPU, &deepseek_scaling_rope);

  xpu_ops.def(
      "bgmv_shrink(Tensor! outputs, Tensor inputs, Tensor weights, Tensor "
      "indices, float scale) -> ()");
  xpu_ops.impl("bgmv_shrink", torch::kXPU, &bgmv_shrink);

  xpu_ops.def(
      "bgmv_expand(Tensor! outputs, Tensor inputs, Tensor weights, Tensor "
      "indices, bool add_to_output) -> ()");
  xpu_ops.impl("bgmv_expand", torch::kXPU, &bgmv_expand);

  xpu_ops.def(
      "bgmv_expand_slice(Tensor! outputs, Tensor inputs, Tensor weights, "
      "Tensor indices, int slice_offset,bool add_to_output) -> ()");
  xpu_ops.impl("bgmv_expand_slice", torch::kXPU, &bgmv_expand_slice);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
