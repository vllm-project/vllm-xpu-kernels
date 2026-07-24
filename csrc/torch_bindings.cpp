#include "core/registration.h"
#include "ops.h"
// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

VLLM_TORCH_LIBRARY_FRAGMENT_EXPAND(VLLM_TORCH_OP_NAMESPACE, ops) {
  ops.def("weak_ref_tensor(Tensor input) -> Tensor");

  // Layernorm
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  // FIXME: torch op check consider input & weight is mutable in some ut
  // cases. so we make it mutable here.
  ops.def(
      "rms_norm(Tensor! result, Tensor input, Tensor? weight, float epsilon) "
      "-> "
      "()");

  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor? weight, "
      "float epsilon) -> ()");

  // Fused RMSNorm + dynamic per-token quantization (FP8 or INT8).
  ops.def(
      "rms_norm_dynamic_per_token_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual) -> ()");

  // Fused RMSNorm + per-column-block quantization (FP8 or INT8).
  ops.def(
      "rms_norm_per_block_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual, int group_size, "
      "bool is_scale_transposed, bool scale_ue8m0=False) -> ()");

  ops.def(
      "rms_norm_mxfp4_quant(Tensor! result, Tensor input, Tensor weight, "
      "Tensor! scale, float epsilon, Tensor!? residual, int group_size) "
      "-> ()");

  // Fused RMSNorm + static FP8 quantization.
  ops.def(
      "rms_norm_static_fp8_quant(Tensor! result, Tensor input, Tensor weight, "
      "Tensor scale, float epsilon) -> ()");

  // In-place fused Add + RMSNorm + static FP8 quantization.
  ops.def(
      "fused_add_rms_norm_static_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! residual, Tensor weight, "
      "Tensor scale, float epsilon) -> ()");

  // activation ops
  ops.def("silu_and_mul(Tensor! result, Tensor input) -> ()");

  // Fused SiLU + Mul + FP8 Quantization
  ops.def(
      "silu_and_mul_quant(Tensor! result, Tensor input, Tensor scale) -> ()");

  // Fused SiLU + Mul + per-block dynamic quantization (FP8 or INT8).
  ops.def(
      "silu_and_mul_per_block_quant("
      "Tensor! out, "
      "Tensor input, "
      "Tensor! scales, "
      "int group_size, "
      "Tensor? scale_ub=None, "
      "bool is_scale_transposed=False, "
      "bool scale_ue8m0=False) -> ()");

  ops.def(
      "silu_and_mul_mxfp4_quant("
      "Tensor! out, Tensor input, Tensor! scales, "
      "int group_size, float eps=1e-10) -> ()");

  ops.def("mul_and_silu(Tensor! out, Tensor input) -> ()");

  ops.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");

  ops.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");

  ops.def("fatrelu_and_mul(Tensor! out, Tensor! input, float threshold) -> ()");

  ops.def("gelu_fast(Tensor! out, Tensor input) -> ()");

  ops.def("gelu_new(Tensor! out, Tensor input) -> ()");

  ops.def("gelu_quick(Tensor! out, Tensor input) -> ()");

  // pos_embedding
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor!? key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox) -> ()");

  // Fused QK RMSNorm + RoPE
  ops.def(
      "fused_qk_norm_rope(Tensor! qkv, int num_heads_q, "
      "int num_heads_k, int num_heads_v, int head_dim, float eps, "
      "Tensor q_weight, Tensor k_weight, Tensor cos_sin_cache, "
      "bool is_neox, Tensor position_ids, "
      "int forced_token_heads_per_warp=-1) -> ()");

  // Compute FP8 quantized tensor for given scaling factor.
  ops.def(
      "static_scaled_fp8_quant(Tensor! result, Tensor input, Tensor scale, "
      "(int, int)? group_shape=None) -> ()");

  // Compute dynamic-per-tensor FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_scaled_fp8_quant(Tensor! result, Tensor input, Tensor! scale) "
      "-> "
      "()");

  // Compute dynamic-per-token FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_per_token_scaled_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! scale, Tensor? scale_ub) -> "
      "()");

  // Compute per-token-group FP8 quantized tensor and scaling factor.
  ops.def(
      "per_token_group_fp8_quant(Tensor input, Tensor! output_q, Tensor! "
      "output_s, "
      "int group_size, float eps, float fp8_min, float fp8_max, bool "
      "scale_ue8m0, bool dummy_is_scale_transposed, bool dummy_is_tma_aligned "
      ") -> ()");

  // Compute per-token-group MXFP4 quantized tensor and scaling factor.
  ops.def(
      "per_token_group_quant_mxfp4(Tensor input, Tensor! output_q, Tensor! "
      "output_s, int group_size, float eps) -> ()");

  // swigluoai_and_mul
  ops.def(
      "swigluoai_and_mul(Tensor! out, Tensor input, float alpha=1.702, float "
      "limit=7.0) "
      "-> ()");

  // relu2_no_mul
  ops.def("relu2_no_mul(Tensor! out, Tensor! input) -> ()");

  // swiglustep_and_mul
  ops.def(
      "swiglustep_and_mul(Tensor! out, Tensor input, float limit=7.0) "
      "-> ()");

  ops.def(
      "get_xpu_view_from_cpu_tensor(Tensor cpu_tensor) -> "
      "Tensor");

  ops.def(
      "top_k_per_row_prefill(Tensor logits, Tensor rowStarts, Tensor rowEnds, "
      "Tensor! indices, int numRows, int stride0, "
      "int stride1, int topK) -> ()");

  ops.def(
      "top_k_per_row_decode(Tensor logits, int next_n, "
      "Tensor seq_lens, Tensor! indices, "
      "int numRows, int stride0, int stride1, int topK) -> ()");

  // Synchronous raw-pointer memcpy helper (0=H2D, 1=D2H, 2=D2D).
  // This is intentionally pointer-based to support allocator-managed buffers.
  ops.def(
      "xpu_memcpy_sync(int dst_ptr, int src_ptr, int n_bytes, int kind, "
      "int device=-1) -> ()");

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
}

VLLM_TORCH_LIBRARY_IMPL_EXPAND(VLLM_TORCH_OP_NAMESPACE, XPU, ops) {
  VLLM_TORCH_IMPL(ops, "weak_ref_tensor", &weak_ref_tensor);
  VLLM_TORCH_IMPL(ops, "rms_norm", &rms_norm);
  VLLM_TORCH_IMPL(ops, "fused_add_rms_norm", &fused_add_rms_norm);
  VLLM_TORCH_IMPL(
      ops,
      "rms_norm_dynamic_per_token_quant",
      &rms_norm_dynamic_per_token_quant);
  VLLM_TORCH_IMPL(ops, "rms_norm_per_block_quant", &rms_norm_per_block_quant);
  VLLM_TORCH_IMPL(ops, "rms_norm_mxfp4_quant", &rms_norm_mxfp4_quant);
  VLLM_TORCH_IMPL(ops, "rms_norm_static_fp8_quant", &rms_norm_static_fp8_quant);
  VLLM_TORCH_IMPL(
      ops,
      "fused_add_rms_norm_static_fp8_quant",
      &fused_add_rms_norm_static_fp8_quant);
  VLLM_TORCH_IMPL(ops, "silu_and_mul", &silu_and_mul);
  VLLM_TORCH_IMPL(ops, "silu_and_mul_quant", &silu_and_mul_quant);
  VLLM_TORCH_IMPL(
      ops, "silu_and_mul_per_block_quant", &silu_and_mul_per_block_quant);
  VLLM_TORCH_IMPL(ops, "silu_and_mul_mxfp4_quant", &silu_and_mul_mxfp4_quant);
  VLLM_TORCH_IMPL(ops, "mul_and_silu", &mul_and_silu);
  VLLM_TORCH_IMPL(ops, "gelu_and_mul", &gelu_and_mul);
  VLLM_TORCH_IMPL(ops, "gelu_tanh_and_mul", &gelu_tanh_and_mul);
  VLLM_TORCH_IMPL(ops, "fatrelu_and_mul", &fatrelu_and_mul);
  VLLM_TORCH_IMPL(ops, "gelu_fast", &gelu_fast);
  VLLM_TORCH_IMPL(ops, "gelu_new", &gelu_new);
  VLLM_TORCH_IMPL(ops, "gelu_quick", &gelu_quick);
  VLLM_TORCH_IMPL(ops, "rotary_embedding", &rotary_embedding);
  VLLM_TORCH_IMPL(ops, "fused_qk_norm_rope", &fused_qk_norm_rope);
  VLLM_TORCH_IMPL(ops, "static_scaled_fp8_quant", &static_scaled_fp8_quant);
  VLLM_TORCH_IMPL(ops, "dynamic_scaled_fp8_quant", &dynamic_scaled_fp8_quant);
  VLLM_TORCH_IMPL(
      ops,
      "dynamic_per_token_scaled_fp8_quant",
      &dynamic_per_token_scaled_fp8_quant);
  VLLM_TORCH_IMPL(ops, "per_token_group_fp8_quant", &per_token_group_quant_fp8);
  VLLM_TORCH_IMPL(
      ops, "per_token_group_quant_mxfp4", &per_token_group_quant_mxfp4);
  VLLM_TORCH_IMPL(ops, "swigluoai_and_mul", &swigluoai_and_mul);
  VLLM_TORCH_IMPL(ops, "relu2_no_mul", &relu2_no_mul);
  VLLM_TORCH_IMPL(ops, "swiglustep_and_mul", &swiglustep_and_mul);
  VLLM_TORCH_IMPL(ops, "top_k_per_row_prefill", &top_k_per_row_prefill);
  VLLM_TORCH_IMPL(ops, "top_k_per_row_decode", &top_k_per_row_decode);
  VLLM_TORCH_IMPL(ops, "merge_attn_states", &merge_attn_states);
}

VLLM_TORCH_LIBRARY_IMPL_EXPAND(VLLM_TORCH_OP_NAMESPACE, CPU, ops) {
  VLLM_TORCH_IMPL(
      ops, "get_xpu_view_from_cpu_tensor", &get_xpu_view_from_cpu_tensor);
}

VLLM_TORCH_LIBRARY_IMPL_EXPAND(
    VLLM_TORCH_OP_NAMESPACE, CompositeExplicitAutograd, ops) {
  VLLM_TORCH_IMPL(ops, "xpu_memcpy_sync", &xpu_memcpy_sync);
}

VLLM_TORCH_LIBRARY_FRAGMENT_EXPAND(VLLM_TORCH_CACHE_OP_NAMESPACE, cache_ops) {
  // Reshape the key and value tensors and cache them.
  cache_ops.def(
      "reshape_and_cache(Tensor key, Tensor value,"
      "                  Tensor! key_cache, Tensor! value_cache,"
      "                  Tensor slot_mapping,"
      "                  str kv_cache_dtype,"
      "                  Tensor k_scale, Tensor v_scale) -> ()");

  // Reshape the key and value tensors and cache them.
  cache_ops.def(
      "reshape_and_cache_flash(Tensor key, Tensor value,"
      "                        Tensor! key_cache,"
      "                        Tensor! value_cache,"
      "                        Tensor slot_mapping,"
      "                        str kv_cache_dtype,"
      "                        Tensor k_scale, Tensor v_scale) -> ()");

  // Concat kv_c and k_pe and cache them.
  cache_ops.def(
      "concat_and_cache_mla(Tensor kv_c, Tensor k_pe,"
      "                     Tensor! kv_cache,"
      "                     Tensor slot_mapping,"
      "                     str kv_cache_dtype,"
      "                     Tensor scale) -> ()");

  // Gather cache blocks from src_cache to dst.
  cache_ops.def(
      "gather_cache(Tensor src_cache, Tensor! dst, Tensor block_table, "
      "Tensor cu_seq_lens, int batch_size, Tensor? seq_starts) -> ()");

  // Convert between FP8 and FP16/BF16/FP32 formats with scaling
  cache_ops.def(
      "convert_fp8(Tensor! dst, Tensor src, "
      "            float scale, str kv_cache_dtype) -> ()");

  // Cache ops
  // Swap in (out) the cache blocks from src to dst.
  cache_ops.def(
      "swap_blocks(Tensor src, Tensor! dst,"
      "            int block_size_in_bytes, Tensor block_mapping) -> ()");
  // Batch swap: copies N (src_ptr, dst_ptr, size) triples in one call.
  // The target XPU device is auto-inferred from the device pointer.
  cache_ops.def(
      "swap_blocks_batch(Tensor src_ptrs, Tensor dst_ptrs,"
      "                  Tensor sizes) -> ()");
  cache_ops.def(
      "indexer_k_quant_and_cache(Tensor k, Tensor! kv_cache,"
      "Tensor slot_mapping, int quant_block_size, str scale_fmt) -> ()");
  cache_ops.def(
      "cp_gather_indexer_k_quant_cache(Tensor kv_cache, Tensor! dst_k, "
      "Tensor! dst_scale, Tensor block_table, Tensor cu_seq_lens) -> ()");

  // Gather cache blocks with optional FP8 dequantization.
  cache_ops.def(
      "gather_and_maybe_dequant_cache(Tensor src_cache, Tensor! dst, "
      "Tensor block_table, Tensor cu_seq_lens, Tensor token_to_seq, "
      "int num_tokens, str kv_cache_dtype, Tensor scale, "
      "Tensor? seq_starts) -> ()");

  cache_ops.def("getMemoryInfo(int device_index) -> (int, int)");
}

VLLM_TORCH_LIBRARY_IMPL_EXPAND(VLLM_TORCH_CACHE_OP_NAMESPACE, XPU, cache_ops) {
  VLLM_TORCH_IMPL(cache_ops, "reshape_and_cache", &reshape_and_cache);
  VLLM_TORCH_IMPL(
      cache_ops, "reshape_and_cache_flash", &reshape_and_cache_flash);
  VLLM_TORCH_IMPL(cache_ops, "concat_and_cache_mla", &concat_and_cache_mla);
  VLLM_TORCH_IMPL(cache_ops, "gather_cache", &gather_cache);
  VLLM_TORCH_IMPL(cache_ops, "convert_fp8", &convert_fp8);
  VLLM_TORCH_IMPL(cache_ops, "swap_blocks", &swap_blocks);
  VLLM_TORCH_IMPL(
      cache_ops, "indexer_k_quant_and_cache", &indexer_k_quant_and_cache);
  VLLM_TORCH_IMPL(
      cache_ops,
      "cp_gather_indexer_k_quant_cache",
      &cp_gather_indexer_k_quant_cache);
  VLLM_TORCH_IMPL(
      cache_ops,
      "gather_and_maybe_dequant_cache",
      &gather_and_maybe_dequant_cache);
}

VLLM_TORCH_LIBRARY_IMPL_EXPAND(VLLM_TORCH_CACHE_OP_NAMESPACE, CPU, cache_ops) {
  VLLM_TORCH_IMPL(cache_ops, "swap_blocks_batch", &swap_blocks_batch);
}

VLLM_TORCH_LIBRARY_IMPL_EXPAND(
    VLLM_TORCH_CACHE_OP_NAMESPACE, CompositeExplicitAutograd, cache_ops) {
  VLLM_TORCH_IMPL(cache_ops, "getMemoryInfo", &getMemoryInfo);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
