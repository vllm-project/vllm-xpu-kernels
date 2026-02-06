#include "core/registration.h"
#include "xpu/ops.h"
#include "xpu/grouped_gemm/grouped_gemm_interface.h"
#include "xpu/lora/lora_ops.h"

#include <torch/library.h>
#include <torch/version.h>

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, xpu_ops) {
  at::Tag stride_tag = at::Tag::needs_fixed_stride_order;

  xpu_ops.def(
      "fp8_gemm(Tensor A, Tensor B, ScalarType? out_dtype, Tensor? A_scale_, "
      "Tensor? B_scale_, Tensor? bias_) -> Tensor");
  xpu_ops.impl("fp8_gemm", torch::kXPU, &fp8_gemm);

  xpu_ops.def(
      "fp8_gemm_w8a16(Tensor A, Tensor B, Tensor? B_scale_, "
      "Tensor? bias_) -> Tensor");
  xpu_ops.impl("fp8_gemm_w8a16", torch::kXPU, &fp8_gemm_w8a16);

  xpu_ops.def(
      "int4_gemm_w4a16(Tensor A, Tensor B, Tensor? bias, Tensor B_scale, "
      "Tensor B_zp, int group_size, Tensor? g_idx) -> Tensor");
  xpu_ops.impl("int4_gemm_w4a16", torch::kXPU, &int4_gemm_w4a16);

  xpu_ops.def(
      "int4_gemm_w4a8(Tensor A_, Tensor A_scale, Tensor A_zp, Tensor B, "
      "Tensor B_scale, Tensor B_zp, int group_size, Tensor? g_idx, Tensor? "
      "bias) -> Tensor");
  xpu_ops.impl("int4_gemm_w4a8", torch::kXPU, &int4_gemm_w4a8);

  xpu_ops.def(
      "cutlass_grouped_gemm_interface(Tensor ptr_A, Tensor ptr_B, Tensor? "
      "ptr_scales, "
      "Tensor? ptr_bias, "
      "Tensor "
      "ptr_D, Tensor "
      "expert_first_token_offset, int N, int K, int "
      "num_experts, bool is_B_int4, bool is_B_mxfp4) -> "
      "Tensor");
  xpu_ops.impl(
      "cutlass_grouped_gemm_interface",
      torch::kXPU,
      &cutlass_grouped_gemm_interface);

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
      "Tensor indices, int slice_offset, int slice_size, bool add_to_output) "
      "-> ()");
  xpu_ops.impl("bgmv_expand_slice", torch::kXPU, &bgmv_expand_slice);

  xpu_ops.def("get_onednn_version() -> str");
  xpu_ops.impl("get_onednn_version", &get_onednn_version);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
