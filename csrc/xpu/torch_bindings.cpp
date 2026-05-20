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
      "cutlass_grouped_gemm_interface(Tensor ptr_A, Tensor ptr_B, Tensor? "
      "ptr_scales, "
      "Tensor? ptr_bias, "
      "Tensor "
      "ptr_D, Tensor "
      "rows_per_expert, int N, int K, int "
      "num_experts, bool is_B_int4, bool is_B_mxfp4) -> "
      "Tensor");
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
      "int num_prefills, int num_decodes, Tensor? has_initial_state, Tensor "
      "non_spec_query_start_loc,"
      "Tensor non_spec_state_indices_tensor, int num_actual_tokens, int "
      "tp_size, bool reorder_input) -> ()");
  xpu_ops.impl("gdn_attention", torch::kXPU, &gdn_attention);
#endif

  // for empty tensor functions, we don't need dispatch key like torch::kXPU
  xpu_ops.def("is_bmg(int device_index) -> bool");
  xpu_ops.impl("is_bmg", &is_bmg);

  xpu_ops.def("is_pvc(int device_index) -> bool");
  xpu_ops.impl("is_pvc", &is_pvc);

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
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
