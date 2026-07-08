#include "core/registration.h"
#include "xpu/ops.h"
#ifdef VLLM_MOE_ENABLED
  #include "xpu/grouped_gemm/grouped_gemm_interface.h"
#endif
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
      "fp8_bmm(Tensor A, Tensor B, ScalarType? out_dtype, Tensor? A_scale_, "
      "Tensor? B_scale_, Tensor? bias_) -> Tensor");
  xpu_ops.impl("fp8_bmm", torch::kXPU, &fp8_bmm);

  xpu_ops.def(
      "fp8_gemm_w8a16(Tensor A, Tensor B, Tensor? B_scale_, "
      "Tensor? bias_) -> Tensor");
  xpu_ops.impl("fp8_gemm_w8a16", torch::kXPU, &fp8_gemm_w8a16);

  xpu_ops.def(
      "fp4_gemm(Tensor A, Tensor B, Tensor A_scale, Tensor B_scale, "
      "ScalarType? out_dtype, Tensor? bias_) -> Tensor");
  xpu_ops.impl("fp4_gemm", torch::kXPU, &fp4_gemm);

  xpu_ops.def(
      "int4_gemm_w4a16(Tensor A, Tensor B, Tensor? bias, Tensor B_scale, "
      "Tensor B_zp, int group_size, Tensor? g_idx) -> Tensor");
  xpu_ops.impl("int4_gemm_w4a16", torch::kXPU, &int4_gemm_w4a16);

  xpu_ops.def(
      "int4_gemm_w4a8(Tensor A_, Tensor A_scale, Tensor A_zp, Tensor B, "
      "Tensor B_scale, Tensor B_zp, int group_size, Tensor? g_idx, Tensor? "
      "bias) -> Tensor");
  xpu_ops.impl("int4_gemm_w4a8", torch::kXPU, &int4_gemm_w4a8);

#ifdef VLLM_MOE_ENABLED
  xpu_ops.def(
      "cutlass_grouped_gemm_interface(Tensor ptr_A, Tensor? ptr_A_scale, "
      "Tensor ptr_B, Tensor? ptr_B_scale, Tensor? ptr_bias, Tensor ptr_D, "
      "Tensor rows_per_expert, int N, int K, int num_experts) -> Tensor");
  xpu_ops.impl(
      "cutlass_grouped_gemm_interface",
      torch::kXPU,
      &cutlass_grouped_gemm_interface);
#endif

  xpu_ops.def(
      "deepseek_scaling_rope(Tensor! positions, Tensor! query, Tensor! key, "
      "Tensor? offsets_opt, Tensor! cos_sin_cache, int rotary_dim, bool "
      "is_neox_style) "
      "-> (Tensor, Tensor)");
  xpu_ops.impl("deepseek_scaling_rope", torch::kXPU, &deepseek_scaling_rope);

  // Multi-modal Rotary Embedding (M-RoPE) — used by e.g. Qwen2-VL.
  // positions has shape [num_mrope_sections, num_tokens]; mrope_section is
  // an int32 device tensor of length num_mrope_sections that partitions the
  // rotation dimensions across positional axes (e.g. time / height / width).
  xpu_ops.def(
      "multimodal_rotary_embedding(Tensor positions, Tensor! query,"
      "                            Tensor!? key, int head_size,"
      "                            Tensor cos_sin_cache, bool is_neox,"
      "                            int[] mrope_section) -> ()");
  xpu_ops.impl(
      "multimodal_rotary_embedding", torch::kXPU, &multimodal_rotary_embedding);

  // Apply rotary embedding taking cos/sin directly (flash_attn style).
  // For diffusion models where cos/sin are pre-computed externally.
  xpu_ops.def(
      "apply_rotary_emb(Tensor! output, Tensor input,"
      "                 Tensor cos, Tensor sin, bool is_neox) -> ()");
  xpu_ops.impl("apply_rotary_emb", torch::kXPU, &apply_rotary_emb);

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
  xpu_ops.impl(
      "deepseek_qnorm_rope_kv_insert",
      torch::kXPU,
      &deepseek_qnorm_rope_kv_insert);

  // DeepSeek-V4 inverse RoPE (bf16): fused inv GPT-J rotation + group-major
  // layout transform.
  xpu_ops.def(
      "deepseek_inv_rope_bf16(Tensor attn_output, Tensor positions,"
      "                       Tensor cos_sin_cache, int n_groups,"
      "                       int heads_per_group, int nope_dim,"
      "                       int rope_dim) -> Tensor");
  xpu_ops.impl("deepseek_inv_rope_bf16", torch::kXPU, &deepseek_inv_rope_bf16);

  // DeepSeek-V4 fused inverse RoPE + FP8 quantization: inv GPT-J rotation +
  // per-128-block UE8M0 FP8 E4M3 quantization + group-major layout transform.
  xpu_ops.def(
      "deepseek_inv_rope_fp8_quant(Tensor attn_output, Tensor positions,"
      "                            Tensor cos_sin_cache, int n_groups,"
      "                            int heads_per_group, int nope_dim,"
      "                            int rope_dim) -> (Tensor, Tensor)");
  xpu_ops.impl(
      "deepseek_inv_rope_fp8_quant", torch::kXPU, &deepseek_inv_rope_fp8_quant);

#ifdef VLLM_MHC_ENABLED
  xpu_ops.def(
      "mhc_pre(Tensor residual, Tensor fn, Tensor hc_scale, Tensor hc_base,"
      "        float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,"
      "        float hc_post_mult_value, int sinkhorn_repeat)"
      "        -> (Tensor, Tensor, Tensor)");
  xpu_ops.impl("mhc_pre", torch::kXPU, &mhc_pre);

  xpu_ops.def(
      "mhc_post(Tensor x, Tensor residual, Tensor post_layer_mix,"
      "         Tensor comb_res_mix) -> Tensor");
  xpu_ops.impl("mhc_post", torch::kXPU, &mhc_post);

  xpu_ops.def(
      "hc_head_fused(Tensor hs_flat, Tensor fn, Tensor hc_scale,"
      "              Tensor hc_base, Tensor! out, float rms_eps,"
      "              float hc_eps) -> ()");
  xpu_ops.impl("hc_head_fused", torch::kXPU, &hc_head_fused);

  xpu_ops.def(
      "mhc_fused_post_pre(Tensor x, Tensor residual, Tensor post_layer_mix,"
      "                   Tensor comb_res_mix, Tensor fn, Tensor hc_scale,"
      "                   Tensor hc_base, float rms_eps, float hc_pre_eps,"
      "                   float hc_sinkhorn_eps, float hc_post_mult_value,"
      "                   int sinkhorn_repeat)"
      "                   -> (Tensor, Tensor, Tensor, Tensor)");
  xpu_ops.impl("mhc_fused_post_pre", torch::kXPU, &mhc_fused_post_pre);
#endif

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
  xpu_ops.impl("gdn_attention", torch::kXPU, &gdn_attention);
#endif

  // for empty tensor functions, we don't need dispatch key like torch::kXPU
  xpu_ops.def("is_bmg_g21(int device_index) -> bool");
  xpu_ops.impl("is_bmg_g21", &is_bmg_g21);

  xpu_ops.def("is_bmg_g31(int device_index) -> bool");
  xpu_ops.impl("is_bmg_g31", &is_bmg_g31);

  xpu_ops.def("is_bmg(int device_index) -> bool");
  xpu_ops.impl("is_bmg", &is_bmg);

  xpu_ops.def("is_pvc(int device_index) -> bool");
  xpu_ops.impl("is_pvc", &is_pvc);

  xpu_ops.def("is_xe2_arch(int device_index=-1) -> bool");
  xpu_ops.impl("is_xe2_arch", &is_xe2_arch);

  xpu_ops.def("is_xe3_arch(int device_index=-1) -> bool");
  xpu_ops.impl("is_xe3_arch", &is_xe3_arch);

  // test only, will not use in vllm
  xpu_ops.def(
      "exponential_2d_(Tensor! tensor, Tensor! seeds, float lambda) -> ()");
  xpu_ops.impl("exponential_2d_", torch::kXPU, &exponential_2d_);

  xpu_ops.def(
      "topk_topp_sampler(Tensor! random_sampled, Tensor? logits_to_return,"
      "Tensor! logits, Tensor? k, Tensor? p, str logprobs_mode, Tensor! seeds, "
      "float lambda) -> ()");
  xpu_ops.impl("topk_topp_sampler", torch::kXPU, &topk_topp_sampler);

#ifdef VLLM_MQA_LOGITS_ENABLED
  xpu_ops.def(
      "fp8_mqa_logits(Tensor q, Tensor kv, Tensor kv_scales, Tensor weights, "
      "Tensor cu_seqlen_ks, Tensor cu_seqlen_ke) -> Tensor");
  xpu_ops.impl("fp8_mqa_logits", torch::kXPU, &fp8_mqa_logits);

  xpu_ops.def(
      "fp8_paged_mqa_logits(Tensor q_fp8, Tensor kv_cache_fp8, Tensor "
      "weights, Tensor context_lens, Tensor block_tables, Tensor? "
      "schedule_metadata, int max_model_len) -> Tensor");
  xpu_ops.impl("fp8_paged_mqa_logits", torch::kXPU, &fp8_paged_mqa_logits);
#endif

  xpu_ops.def("get_onednn_version() -> str");
  xpu_ops.impl("get_onednn_version", &get_onednn_version);

  xpu_ops.def(
      "deepseek_fused_indexer_q_rope_fp8(Tensor q, Tensor positions, "
      "Tensor cos_sin_cache, Tensor index_weights, float softmax_scale, "
      "float head_scale, Tensor! q_fp8, Tensor! weights_out) -> ()");
  xpu_ops.impl(
      "deepseek_fused_indexer_q_rope_fp8",
      torch::kXPU,
      &deepseek_fused_indexer_q_rope_fp8);

  xpu_ops.def(
      "deepseek_fused_indexer_q_rope_mxfp4(Tensor q, Tensor positions, "
      "Tensor cos_sin_cache, Tensor index_weights, float softmax_scale, "
      "float head_scale, Tensor! packed_out, Tensor! scales_out, "
      "Tensor! weights_out) -> ()");
  xpu_ops.impl(
      "deepseek_fused_indexer_q_rope_mxfp4",
      torch::kXPU,
      &deepseek_fused_indexer_q_rope_mxfp4);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
