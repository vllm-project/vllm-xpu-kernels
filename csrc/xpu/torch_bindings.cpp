#include "core/registration.h"
#include "xpu/ops.h"

#include <torch/library.h>
#include <torch/version.h>
// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  at::Tag stride_tag = at::Tag::needs_fixed_stride_order;

  // Layernorm
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.def(
      // FIXME: torch op check consider input & weight is mutable in some ut
      // cases. so we make it mutable here.
      "rms_norm(Tensor! result, Tensor! input, Tensor! weight, float epsilon) "
      "-> "
      "()");
  ops.impl("rms_norm", torch::kXPU, &rms_norm);

  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
      "float epsilon) -> ()");
  ops.impl("fused_add_rms_norm", torch::kXPU, &fused_add_rms_norm);

  ops.def("cutlass_sycl_demo(Tensor a) -> ()");
  ops.impl("cutlass_sycl_demo", torch::kXPU, &cutlass_sycl_demo);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
