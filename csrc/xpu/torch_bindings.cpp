#include "core/registration.h"
#include "xpu/ops.h"
#ifdef VLLM_MOE_ENABLED
  #include "xpu/grouped_gemm/grouped_gemm_interface.h"
#endif
#include "xpu/lora/lora_ops.h"

VLLM_TORCH_LIBRARY_FRAGMENT_EXPAND(VLLM_TORCH_OP_NAMESPACE, xpu_ops) {
  xpu_ops.def(
      "fp8_gemm(Tensor A, Tensor B, ScalarType? out_dtype, Tensor? A_scale_, "
      "Tensor? B_scale_, Tensor? bias_) -> Tensor");

  xpu_ops.def(
      "fp8_bmm(Tensor A, Tensor B, ScalarType? out_dtype, Tensor? A_scale_, "
      "Tensor? B_scale_, Tensor? bias_) -> Tensor");

  xpu_ops.def(
      "fp8_gemm_w8a16(Tensor A, Tensor B, Tensor? B_scale_, "
      "Tensor? bias_) -> Tensor");

  xpu_ops.def(
      "fp4_gemm(Tensor A, Tensor B, Tensor A_scale, Tensor B_scale, "
      "ScalarType? out_dtype, Tensor? bias_) -> Tensor");

  xpu_ops.def(
      "int4_gemm_w4a16(Tensor A, Tensor B, Tensor? bias, Tensor B_scale, "
      "Tensor B_zp, int group_size, Tensor? g_idx) -> Tensor");

  xpu_ops.def(
      "int4_gemm_w4a8(Tensor A_, Tensor A_scale, Tensor A_zp, Tensor B, "
      "Tensor B_scale, Tensor B_zp, int group_size, Tensor? g_idx, Tensor? "
      "bias) -> Tensor");

#ifdef VLLM_MOE_ENABLED
  xpu_ops.def(
      "cutlass_grouped_gemm_interface(Tensor ptr_A, Tensor? ptr_A_scale, "
      "Tensor ptr_B, Tensor? ptr_B_scale, Tensor? ptr_bias, Tensor ptr_D, "
      "Tensor rows_per_expert, int N, int K, int num_experts) -> Tensor");
#endif

  xpu_ops.def(
      "deepseek_scaling_rope(Tensor! positions, Tensor! query, Tensor! key, "
      "Tensor? offsets_opt, Tensor! cos_sin_cache, int rotary_dim, bool "
      "is_neox_style) "
      "-> (Tensor, Tensor)");

  // Multi-modal Rotary Embedding (M-RoPE) — used by e.g. Qwen2-VL.
  // positions has shape [num_mrope_sections, num_tokens]; mrope_section is
  // an int32 device tensor of length num_mrope_sections that partitions the
  // rotation dimensions across positional axes (e.g. time / height / width).
  xpu_ops.def(
      "multimodal_rotary_embedding(Tensor positions, Tensor! query,"
      "                            Tensor!? key, int head_size,"
      "                            Tensor cos_sin_cache, bool is_neox,"
      "                            int[] mrope_section) -> ()");

  // Apply rotary embedding taking cos/sin directly (flash_attn style).
  // For diffusion models where cos/sin are pre-computed externally.
  xpu_ops.def(
      "apply_rotary_emb(Tensor! output, Tensor input,"
      "                 Tensor cos, Tensor sin, bool is_neox) -> ()");

  // DeepSeek-V4 fused Q-RMSNorm + GPT-J RoPE + KV cache insert.
  // kv_cache_dtype: "bf16" or "fp8_ds_mla"
  xpu_ops.def(
      "deepseek_qnorm_rope_kv_insert(Tensor! q, Tensor kv,"
      "                              Tensor! cache,"
      "                              Tensor slot_mapping,"
      "                              Tensor position_ids,"
      "                              Tensor cos_sin_cache,"
      "                              float eps, int block_size,"
      "                              str kv_cache_dtype) -> ()");

  // DeepSeek-V4 inverse RoPE (bf16): fused inv GPT-J rotation + group-major
  // layout transform.
  xpu_ops.def(
      "deepseek_inv_rope_bf16(Tensor attn_output, Tensor positions,"
      "                       Tensor cos_sin_cache, int n_groups,"
      "                       int heads_per_group, int nope_dim,"
      "                       int rope_dim) -> Tensor");

  // DeepSeek-V4 fused inverse RoPE + FP8 quantization: inv GPT-J rotation +
  // per-128-block UE8M0 FP8 E4M3 quantization + group-major layout transform.
  xpu_ops.def(
      "deepseek_inv_rope_fp8_quant(Tensor attn_output, Tensor positions,"
      "                            Tensor cos_sin_cache, int n_groups,"
      "                            int heads_per_group, int nope_dim,"
      "                            int rope_dim) -> (Tensor, Tensor)");

#ifdef VLLM_MHC_ENABLED
  xpu_ops.def(
      "mhc_pre(Tensor residual, Tensor fn, Tensor hc_scale, Tensor hc_base,"
      "        float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,"
      "        float hc_post_mult_value, int sinkhorn_repeat)"
      "        -> (Tensor, Tensor, Tensor)");

  xpu_ops.def(
      "mhc_post(Tensor x, Tensor residual, Tensor post_layer_mix,"
      "         Tensor comb_res_mix) -> Tensor");

  xpu_ops.def(
      "hc_head_fused(Tensor hs_flat, Tensor fn, Tensor hc_scale,"
      "              Tensor hc_base, Tensor! out, float rms_eps,"
      "              float hc_eps) -> ()");

  xpu_ops.def(
      "mhc_fused_post_pre(Tensor x, Tensor residual, Tensor post_layer_mix,"
      "                   Tensor comb_res_mix, Tensor fn, Tensor hc_scale,"
      "                   Tensor hc_base, float rms_eps, float hc_pre_eps,"
      "                   float hc_sinkhorn_eps, float hc_post_mult_value,"
      "                   int sinkhorn_repeat)"
      "                   -> (Tensor, Tensor, Tensor, Tensor)");
#endif

  xpu_ops.def(
      "bgmv_shrink(Tensor! outputs, Tensor inputs, Tensor weights, Tensor "
      "indices, float scale) -> ()");

  xpu_ops.def(
      "bgmv_expand(Tensor! outputs, Tensor inputs, Tensor weights, Tensor "
      "indices, bool add_to_output) -> ()");

  xpu_ops.def(
      "bgmv_expand_slice(Tensor! outputs, Tensor inputs, Tensor weights, "
      "Tensor indices, int slice_offset, int slice_size, bool add_to_output) "
      "-> ()");

  xpu_ops.def(
      "lora_shrink(Tensor inputs, Tensor[] lora_a_weights, "
      "Tensor! output_tensor, Tensor lora_indices, float scaling) -> ()");

  xpu_ops.def(
      "lora_expand(Tensor inputs, Tensor[] lora_b_weights, "
      "Tensor! output_tensor, Tensor lora_indices, int offset_start, "
      "bool add_inputs) -> ()");

#ifdef VLLM_GDN_ENABLED
  xpu_ops.def(
      "gdn_attention(Tensor! core_attn_out, Tensor! z, Tensor "
      "projected_states_qkvz, Tensor projected_states_ba,"
      "int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,"
      "Tensor! conv_state, Tensor! ssm_state, Tensor conv_weights, Tensor? "
      "conv_bias, str activation, Tensor A_log, Tensor dt_bias,"
      "int num_prefills, int num_decodes, int num_spec_decodes, Tensor? "
      "has_initial_state, Tensor? "
      "non_spec_query_start_loc,  Tensor? non_spec_token_indx,"
      "Tensor? non_spec_state_indices_tensor, Tensor? spec_query_start_loc, "
      "Tensor? spec_token_indx, Tensor? spec_state_indices_tensor, Tensor? "
      "num_accepted_tokens, int num_actual_tokens, int "
      "tp_size, bool reorder_input) -> ()");
#endif

  // for empty tensor functions, we don't need dispatch key like torch::kXPU
  xpu_ops.def("is_bmg_g21(int device_index) -> bool");

  xpu_ops.def("is_bmg_g31(int device_index) -> bool");

  xpu_ops.def("is_bmg(int device_index) -> bool");

  xpu_ops.def("is_pvc(int device_index) -> bool");

  xpu_ops.def("is_xe2_arch(int device_index=-1) -> bool");

  xpu_ops.def("is_xe3_arch(int device_index=-1) -> bool");

  // test only, will not use in vllm
  xpu_ops.def(
      "exponential_2d_(Tensor! tensor, Tensor! seeds, float lambda) -> ()");

  xpu_ops.def(
      "topk_topp_sampler(Tensor! random_sampled, Tensor? logits_to_return,"
      "Tensor! logits, Tensor? k, Tensor? p, str logprobs_mode, Tensor! seeds, "
      "float lambda) -> ()");

#ifdef VLLM_MQA_LOGITS_ENABLED
  xpu_ops.def(
      "fp8_mqa_logits(Tensor q, Tensor kv, Tensor kv_scales, Tensor weights, "
      "Tensor cu_seqlen_ks, Tensor cu_seqlen_ke) -> Tensor");

  xpu_ops.def(
      "fp8_paged_mqa_logits(Tensor q_fp8, Tensor kv_cache_fp8, Tensor "
      "weights, Tensor context_lens, Tensor block_tables, Tensor? "
      "schedule_metadata, int max_model_len) -> Tensor");
#endif

  xpu_ops.def("get_onednn_version() -> str");

  xpu_ops.def(
      "deepseek_fused_indexer_q_rope_fp8(Tensor q, Tensor positions, "
      "Tensor cos_sin_cache, Tensor index_weights, float softmax_scale, "
      "float head_scale, Tensor! q_fp8, Tensor! weights_out) -> ()");

  xpu_ops.def(
      "deepseek_fused_indexer_q_rope_mxfp4(Tensor q, Tensor positions, "
      "Tensor cos_sin_cache, Tensor index_weights, float softmax_scale, "
      "float head_scale, Tensor! packed_out, Tensor! scales_out, "
      "Tensor! weights_out) -> ()");
}

VLLM_TORCH_LIBRARY_IMPL_EXPAND(VLLM_TORCH_OP_NAMESPACE, XPU, xpu_ops) {
  VLLM_TORCH_IMPL(xpu_ops, "fp8_gemm", &fp8_gemm);
  VLLM_TORCH_IMPL(xpu_ops, "fp8_bmm", &fp8_bmm);
  VLLM_TORCH_IMPL(xpu_ops, "fp8_gemm_w8a16", &fp8_gemm_w8a16);
  VLLM_TORCH_IMPL(xpu_ops, "fp4_gemm", &fp4_gemm);
  VLLM_TORCH_IMPL(xpu_ops, "int4_gemm_w4a16", &int4_gemm_w4a16);
  VLLM_TORCH_IMPL(xpu_ops, "int4_gemm_w4a8", &int4_gemm_w4a8);
#ifdef VLLM_MOE_ENABLED
  VLLM_TORCH_IMPL(
      xpu_ops,
      "cutlass_grouped_gemm_interface",
      &cutlass_grouped_gemm_interface);
#endif
  VLLM_TORCH_IMPL(xpu_ops, "deepseek_scaling_rope", &deepseek_scaling_rope);
  VLLM_TORCH_IMPL(
      xpu_ops, "multimodal_rotary_embedding", &multimodal_rotary_embedding);
  VLLM_TORCH_IMPL(xpu_ops, "apply_rotary_emb", &apply_rotary_emb);
  VLLM_TORCH_IMPL(
      xpu_ops, "deepseek_qnorm_rope_kv_insert", &deepseek_qnorm_rope_kv_insert);
  VLLM_TORCH_IMPL(xpu_ops, "deepseek_inv_rope_bf16", &deepseek_inv_rope_bf16);
  VLLM_TORCH_IMPL(
      xpu_ops, "deepseek_inv_rope_fp8_quant", &deepseek_inv_rope_fp8_quant);
#ifdef VLLM_MHC_ENABLED
  VLLM_TORCH_IMPL(xpu_ops, "mhc_pre", &mhc_pre);
  VLLM_TORCH_IMPL(xpu_ops, "mhc_post", &mhc_post);
  VLLM_TORCH_IMPL(xpu_ops, "hc_head_fused", &hc_head_fused);
  VLLM_TORCH_IMPL(xpu_ops, "mhc_fused_post_pre", &mhc_fused_post_pre);
#endif
  VLLM_TORCH_IMPL(xpu_ops, "bgmv_shrink", &bgmv_shrink);
  VLLM_TORCH_IMPL(xpu_ops, "bgmv_expand", &bgmv_expand);
  VLLM_TORCH_IMPL(xpu_ops, "bgmv_expand_slice", &bgmv_expand_slice);
  VLLM_TORCH_IMPL(xpu_ops, "lora_shrink", &lora_shrink);
  VLLM_TORCH_IMPL(xpu_ops, "lora_expand", &lora_expand);
#ifdef VLLM_GDN_ENABLED
  VLLM_TORCH_IMPL(xpu_ops, "gdn_attention", &gdn_attention);
#endif
  VLLM_TORCH_IMPL(xpu_ops, "exponential_2d_", &exponential_2d_);
  VLLM_TORCH_IMPL(xpu_ops, "topk_topp_sampler", &topk_topp_sampler);
#ifdef VLLM_MQA_LOGITS_ENABLED
  VLLM_TORCH_IMPL(xpu_ops, "fp8_mqa_logits", &fp8_mqa_logits);
  VLLM_TORCH_IMPL(xpu_ops, "fp8_paged_mqa_logits", &fp8_paged_mqa_logits);
#endif
  VLLM_TORCH_IMPL(
      xpu_ops,
      "deepseek_fused_indexer_q_rope_fp8",
      &deepseek_fused_indexer_q_rope_fp8);
  VLLM_TORCH_IMPL(
      xpu_ops,
      "deepseek_fused_indexer_q_rope_mxfp4",
      &deepseek_fused_indexer_q_rope_mxfp4);
}

VLLM_TORCH_LIBRARY_IMPL_EXPAND(
    VLLM_TORCH_OP_NAMESPACE, CompositeExplicitAutograd, xpu_ops) {
  VLLM_TORCH_IMPL(xpu_ops, "is_bmg_g21", &is_bmg_g21);
  VLLM_TORCH_IMPL(xpu_ops, "is_bmg_g31", &is_bmg_g31);
  VLLM_TORCH_IMPL(xpu_ops, "is_bmg", &is_bmg);
  VLLM_TORCH_IMPL(xpu_ops, "is_pvc", &is_pvc);
  VLLM_TORCH_IMPL(xpu_ops, "is_xe2_arch", &is_xe2_arch);
  VLLM_TORCH_IMPL(xpu_ops, "is_xe3_arch", &is_xe3_arch);
  VLLM_TORCH_IMPL(xpu_ops, "get_onednn_version", &get_onednn_version);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
