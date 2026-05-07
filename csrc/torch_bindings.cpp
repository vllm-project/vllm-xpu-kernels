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

  ops.def("weak_ref_tensor(Tensor input) -> Tensor");
  ops.impl("weak_ref_tensor", torch::kXPU, &weak_ref_tensor);

  // Layernorm
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  // FIXME: torch op check consider input & weight is mutable in some ut
  // cases. so we make it mutable here.
  ops.def(
      "rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) "
      "-> "
      "()");
  ops.impl("rms_norm", torch::kXPU, &rms_norm);

  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
      "float epsilon) -> ()");
  ops.impl("fused_add_rms_norm", torch::kXPU, &fused_add_rms_norm);

  // Fused RMSNorm + dynamic per-token quantization (FP8 or INT8).
  ops.def(
      "rms_norm_dynamic_per_token_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual) -> ()");
  ops.impl(
      "rms_norm_dynamic_per_token_quant",
      torch::kXPU,
      &rms_norm_dynamic_per_token_quant);

  // Fused RMSNorm + per-column-block quantization (FP8 or INT8).
  ops.def(
      "rms_norm_per_block_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual, int group_size, "
      "bool is_scale_transposed) -> ()");
  ops.impl("rms_norm_per_block_quant", torch::kXPU, &rms_norm_per_block_quant);

  // Fused RMSNorm + static FP8 quantization.
  ops.def(
      "rms_norm_static_fp8_quant(Tensor! result, Tensor input, Tensor weight, "
      "Tensor scale, float epsilon) -> ()");
  ops.impl(
      "rms_norm_static_fp8_quant", torch::kXPU, &rms_norm_static_fp8_quant);

  // In-place fused Add + RMSNorm + static FP8 quantization.
  ops.def(
      "fused_add_rms_norm_static_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! residual, Tensor weight, "
      "Tensor scale, float epsilon) -> ()");
  ops.impl(
      "fused_add_rms_norm_static_fp8_quant",
      torch::kXPU,
      &fused_add_rms_norm_static_fp8_quant);

  // activation ops
  ops.def("silu_and_mul(Tensor! result, Tensor input) -> ()");
  ops.impl("silu_and_mul", torch::kXPU, &silu_and_mul);

  // Fused SiLU + Mul + FP8 Quantization
  ops.def(
      "silu_and_mul_quant(Tensor! result, Tensor input, Tensor scale) -> ()");
  ops.impl("silu_and_mul_quant", torch::kXPU, &silu_and_mul_quant);

  ops.def("mul_and_silu(Tensor! out, Tensor input) -> ()");
  ops.impl("mul_and_silu", torch::kXPU, &mul_and_silu);

  ops.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_and_mul", torch::kXPU, &gelu_and_mul);

  ops.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_tanh_and_mul", torch::kXPU, &gelu_tanh_and_mul);

  ops.def("fatrelu_and_mul(Tensor! out, Tensor! input, float threshold) -> ()");
  ops.impl("fatrelu_and_mul", torch::kXPU, &fatrelu_and_mul);

  ops.def("gelu_fast(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_fast", torch::kXPU, &gelu_fast);

  ops.def("gelu_new(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_new", torch::kXPU, &gelu_new);

  ops.def("gelu_quick(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_quick", torch::kXPU, &gelu_quick);

  // pos_embedding
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor!? key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox) -> ()");
  ops.impl("rotary_embedding", torch::kXPU, &rotary_embedding);

  // Fused QK RMSNorm + RoPE
  ops.def(
      "fused_qk_norm_rope(Tensor! qkv, int num_heads_q, "
      "int num_heads_k, int num_heads_v, int head_dim, float eps, "
      "Tensor q_weight, Tensor k_weight, Tensor cos_sin_cache, "
      "bool is_neox, Tensor position_ids) -> ()");
  ops.impl("fused_qk_norm_rope", torch::kXPU, &fused_qk_norm_rope);

  // Compute FP8 quantized tensor for given scaling factor.
  ops.def(
      "static_scaled_fp8_quant(Tensor! result, Tensor input, Tensor scale, "
      "(int, int)? group_shape=None) -> ()");
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
  ops.impl(
      "dynamic_per_token_scaled_fp8_quant",
      torch::kXPU,
      &dynamic_per_token_scaled_fp8_quant);

  // Compute per-token-group FP8 quantized tensor and scaling factor.
  ops.def(
      "per_token_group_fp8_quant(Tensor input, Tensor! output_q, Tensor! "
      "output_s, "
      "int group_size, float eps, float fp8_min, float fp8_max, bool "
      "scale_ue8m0) -> ()");
  ops.impl(
      "per_token_group_fp8_quant", torch::kXPU, &per_token_group_quant_fp8);

  // Compute per-token-group MXFP4 quantized tensor and scaling factor.
  ops.def(
      "per_token_group_quant_mxfp4(Tensor input, Tensor! output_q, Tensor! "
      "output_s, int group_size, float eps) -> ()");
  ops.impl(
      "per_token_group_quant_mxfp4", torch::kXPU, &per_token_group_quant_mxfp4);

  // swigluoai_and_mul
  ops.def(
      "swigluoai_and_mul(Tensor! out, Tensor input, float alpha=1.702, float "
      "limit=7.0) "
      "-> ()");
  ops.impl("swigluoai_and_mul", torch::kXPU, &swigluoai_and_mul);

  // relu2_no_mul
  ops.def("relu2_no_mul(Tensor! out, Tensor! input) -> ()");
  ops.impl("relu2_no_mul", torch::kXPU, &relu2_no_mul);

  // swiglustep_and_mul
  ops.def(
      "swiglustep_and_mul(Tensor! out, Tensor input, float limit=7.0) "
      "-> ()");
  ops.impl("swiglustep_and_mul", torch::kXPU, &swiglustep_and_mul);

  ops.def(
      "get_xpu_view_from_cpu_tensor(Tensor cpu_tensor) -> "
      "Tensor");
  ops.impl(
      "get_xpu_view_from_cpu_tensor",
      torch::kCPU,
      &get_xpu_view_from_cpu_tensor);

  ops.def(
      "top_k_per_row_prefill(Tensor logits, Tensor rowStarts, Tensor rowEnds, "
      "Tensor! indices, int numRows, int stride0, "
      "int stride1, int topK) -> ()");
  ops.impl("top_k_per_row_prefill", torch::kXPU, &top_k_per_row_prefill);

  ops.def(
      "top_k_per_row_decode(Tensor logits, int next_n, "
      "Tensor seq_lens, Tensor! indices, "
      "int numRows, int stride0, int stride1, int topK) -> ()");
  ops.impl("top_k_per_row_decode", torch::kXPU, &top_k_per_row_decode);

  // Synchronous raw-pointer memcpy helper (0=H2D, 1=D2H, 2=D2D).
  // This is intentionally pointer-based to support allocator-managed buffers.
  ops.def(
      "xpu_memcpy_sync(int dst_ptr, int src_ptr, int n_bytes, int kind, "
      "int device=-1) -> ()");
  ops.impl("xpu_memcpy_sync", &xpu_memcpy_sync);

  // Merge attn states
  // Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
  // can be used to combine partial attention results (in the split-KV case)
  ops.def(
      "merge_attn_states("
      "    Tensor! output,"
      "    Tensor!? output_lse,"
      "    Tensor prefix_output,"
      "    Tensor prefix_lse,"
      "    Tensor suffix_output,"
      "    Tensor suffix_lse) -> ()");
  ops.impl("merge_attn_states", torch::kXPU, &merge_attn_states);
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
  cache_ops.impl(
      "reshape_and_cache_flash", torch::kXPU, &reshape_and_cache_flash);

  // Concat kv_c and k_pe and cache them.
  cache_ops.def(
      "concat_and_cache_mla(Tensor kv_c, Tensor k_pe,"
      "                     Tensor! kv_cache,"
      "                     Tensor slot_mapping,"
      "                     str kv_cache_dtype,"
      "                     Tensor scale) -> ()");
  cache_ops.impl("concat_and_cache_mla", torch::kXPU, &concat_and_cache_mla);

  // Gather cache blocks from src_cache to dst.
  cache_ops.def(
      "gather_cache(Tensor src_cache, Tensor! dst, Tensor block_table, "
      "Tensor cu_seq_lens, int batch_size, Tensor? seq_starts) -> ()");
  cache_ops.impl("gather_cache", torch::kXPU, &gather_cache);

  // Convert between FP8 and FP16/BF16/FP32 formats with scaling
  cache_ops.def(
      "convert_fp8(Tensor! dst, Tensor src, "
      "            float scale, str kv_cache_dtype) -> ()");
  cache_ops.impl("convert_fp8", torch::kXPU, &convert_fp8);

  // Cache ops
  // Swap in (out) the cache blocks from src to dst.
  cache_ops.def(
      "swap_blocks(Tensor src, Tensor! dst,"
      "            int block_size_in_bytes, Tensor block_mapping) -> ()");
  cache_ops.impl("swap_blocks", torch::kXPU, &swap_blocks);
  // Batch swap: copies N (src_ptr, dst_ptr, size) triples in one call.
  // The target XPU device is auto-inferred from the device pointer.
  cache_ops.def(
      "swap_blocks_batch(Tensor src_ptrs, Tensor dst_ptrs,"
      "                  Tensor sizes) -> ()");
  cache_ops.impl("swap_blocks_batch", torch::kCPU, &swap_blocks_batch);
  cache_ops.def(
      "indexer_k_quant_and_cache(Tensor k, Tensor! kv_cache,"
      "Tensor slot_mapping, int quant_block_size, str scale_fmt) -> ()");
  cache_ops.impl(
      "indexer_k_quant_and_cache", torch::kXPU, &indexer_k_quant_and_cache);
  cache_ops.def(
      "cp_gather_indexer_k_quant_cache(Tensor kv_cache, Tensor! dst_k, "
      "Tensor! dst_scale, Tensor block_table, Tensor cu_seq_lens) -> ()");
  cache_ops.impl(
      "cp_gather_indexer_k_quant_cache",
      torch::kXPU,
      &cp_gather_indexer_k_quant_cache);

  // Gather cache blocks with optional FP8 dequantization.
  cache_ops.def(
      "gather_and_maybe_dequant_cache(Tensor src_cache, Tensor! dst, "
      "Tensor block_table, Tensor cu_seq_lens, Tensor token_to_seq, "
      "int num_tokens, str kv_cache_dtype, Tensor scale, "
      "Tensor? seq_starts) -> ()");
  cache_ops.impl(
      "gather_and_maybe_dequant_cache",
      torch::kXPU,
      &gather_and_maybe_dequant_cache);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
