#include "core/registration.h"
#include "ops.h"

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

  // activation ops
  ops.def("silu_and_mul(Tensor! out, Tensor! input) -> ()");
  ops.impl("silu_and_mul", torch::kXPU, &silu_and_mul);

  ops.def("mul_and_silu(Tensor! out, Tensor! input) -> ()");
  ops.impl("mul_and_silu", torch::kXPU, &mul_and_silu);

  ops.def("gelu_and_mul(Tensor! out, Tensor! input) -> ()");
  ops.impl("gelu_and_mul", torch::kXPU, &gelu_and_mul);

  ops.def("gelu_tanh_and_mul(Tensor! out, Tensor! input) -> ()");
  ops.impl("gelu_tanh_and_mul", torch::kXPU, &gelu_tanh_and_mul);

  ops.def("gelu_fast(Tensor! out, Tensor! input) -> ()");
  ops.impl("gelu_fast", torch::kXPU, &gelu_fast);

  ops.def("gelu_new(Tensor! out, Tensor! input) -> ()");
  ops.impl("gelu_new", torch::kXPU, &gelu_new);

  ops.def("gelu_quick(Tensor! out, Tensor! input) -> ()");
  ops.impl("gelu_quick", torch::kXPU, &gelu_quick);

  // pos_embedding
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor!? key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox) -> ()");
  ops.impl("rotary_embedding", torch::kXPU, &rotary_embedding);

  // Compute FP8 quantized tensor for given scaling factor.
  ops.def(
      "static_scaled_fp8_quant(Tensor! result, Tensor input, Tensor scale) -> "
      "()");
  ops.impl("static_scaled_fp8_quant", torch::kXPU, &static_scaled_fp8_quant);

  // Compute dynamic-per-tensor FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_scaled_fp8_quant(Tensor! result, Tensor input, Tensor! scale) "
      "-> "
      "()");
  ops.impl("dynamic_scaled_fp8_quant", torch::kXPU, &dynamic_scaled_fp8_quant);

  // Compute dynamic-per-token FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_per_token_scaled_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! scale, Tensor? scale_ub) -> "
      "()");
  ops.impl("dynamic_per_token_scaled_fp8_quant", torch::kXPU,
           &dynamic_per_token_scaled_fp8_quant);
  // swigluoai_and_mul
  ops.def(
      "swigluoai_and_mul(Tensor! out, Tensor input, float alpha=1.702, float "
      "limit=7.0) "
      "-> ()");
  ops.impl("swigluoai_and_mul", torch::kXPU, &swigluoai_and_mul);
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cache_ops), cache_ops) {
  // Reshape the key and value tensors and cache them.
  cache_ops.def(
      "reshape_and_cache(Tensor key, Tensor value,"
      "                  Tensor! key_cache, Tensor! value_cache,"
      "                  Tensor slot_mapping,"
      "                  str kv_cache_dtype,"
      "                  Tensor k_scale, Tensor v_scale) -> ()");
  cache_ops.impl("reshape_and_cache", torch::kXPU, &reshape_and_cache);

  // Reshape the key and value tensors and cache them.
  cache_ops.def(
      "reshape_and_cache_flash(Tensor key, Tensor value,"
      "                        Tensor! key_cache,"
      "                        Tensor! value_cache,"
      "                        Tensor slot_mapping,"
      "                        str kv_cache_dtype,"
      "                        Tensor k_scale, Tensor v_scale) -> ()");
  cache_ops.impl("reshape_and_cache_flash", torch::kXPU,
                 &reshape_and_cache_flash);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
