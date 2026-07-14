#pragma once

#include <torch/all.h>

/**
 * Make sure the shape of A and B is correctly setting before calling below gemm
 * method implemented with OneDNN.
 *  A should be one of [b, m, k] and [m, k] or [m, k//2] in fp4 precision
 *  B should be [k, n] or [k//8, n] in int4 precision, where [k//8, n] indicates
 * a packed representation with 8 int4 values packed into one byte along the k
 * dimension.
 */
torch::Tensor fp8_gemm(
    const torch::Tensor& A,
    const torch::Tensor& B,
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& A_scale_,
    const std::optional<torch::Tensor>& B_scale_,
    const std::optional<torch::Tensor>& bias_);

torch::Tensor fp8_bmm(
    const torch::Tensor& A,
    const torch::Tensor& B,
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& A_scale_,
    const std::optional<torch::Tensor>& B_scale_,
    const std::optional<torch::Tensor>& bias_);

torch::Tensor fp8_gemm_w8a16(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const std::optional<torch::Tensor>& B_scale_,
    const std::optional<torch::Tensor>& bias_);

torch::Tensor fp4_gemm(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& A_scale,
    const torch::Tensor& B_scale,
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& bias);

torch::Tensor int4_gemm_w4a16(
    const torch::Tensor& A_,
    const torch::Tensor& B,
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& B_scale,
    const torch::Tensor& B_zp,
    int64_t group_size,
    const std::optional<torch::Tensor>& g_idx);

torch::Tensor int4_gemm_w4a8(
    const torch::Tensor& A_,
    const torch::Tensor& A_scale,
    const torch::Tensor& A_zp,
    const torch::Tensor& B,
    const torch::Tensor& B_scale,
    const torch::Tensor& B_zp,
    int64_t group_size,
    const std::optional<torch::Tensor>& g_idx,
    const std::optional<torch::Tensor>& bias);

#ifdef VLLM_MOE_ENABLED
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
    int64_t num_experts);
#endif

std::tuple<at::Tensor, at::Tensor> deepseek_scaling_rope(
    const at::Tensor& positions,
    const at::Tensor& query,
    const at::Tensor& key,
    const c10::optional<at::Tensor>& offsets_opt,
    const at::Tensor& cos_sin_cache,
    int64_t rotary_dim,
    bool is_neox);

void multimodal_rotary_embedding(
    torch::Tensor& positions,  // [num_mrope_sections, num_tokens]
    torch::Tensor& query,
    std::optional<torch::Tensor> key,
    int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox,
    std::vector<int64_t> mrope_section);  // host int list [num_mrope_sections]

void apply_rotary_emb(
    torch::Tensor& output,  // [num_tokens, num_heads, head_size]
    torch::Tensor& input,   // [num_tokens, num_heads, head_size]
    torch::Tensor& cos,     // [num_tokens, rot_dim/2]
    torch::Tensor& sin,     // [num_tokens, rot_dim/2]
    bool is_neox);

void deepseek_qnorm_rope_kv_insert(
    torch::Tensor& q,
    const torch::Tensor& kv,
    torch::Tensor& cache,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& position_ids,
    const torch::Tensor& cos_sin_cache,
    double eps,
    int64_t block_size,
    const std::string& kv_cache_dtype);

torch::Tensor deepseek_inv_rope_bf16(
    const torch::Tensor& attn_output,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin_cache,
    int64_t n_groups,
    int64_t heads_per_group,
    int64_t nope_dim,
    int64_t rope_dim);

std::tuple<torch::Tensor, torch::Tensor> deepseek_inv_rope_fp8_quant(
    const torch::Tensor& attn_output,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin_cache,
    int64_t n_groups,
    int64_t heads_per_group,
    int64_t nope_dim,
    int64_t rope_dim);

#ifdef VLLM_MHC_ENABLED
std::tuple<at::Tensor, at::Tensor, at::Tensor> mhc_pre(
    const at::Tensor& residual,
    const at::Tensor& fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    double rms_eps,
    double hc_pre_eps,
    double hc_sinkhorn_eps,
    double hc_post_mult_value,
    int64_t sinkhorn_repeat);

at::Tensor mhc_post(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post_layer_mix,
    const at::Tensor& comb_res_mix);

void hc_head_fused(
    const at::Tensor& hs_flat,
    const at::Tensor& fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    at::Tensor& out,
    double rms_eps,
    double hc_eps);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mhc_fused_post_pre(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post_layer_mix,
    const at::Tensor& comb_res_mix,
    const at::Tensor& fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    double rms_eps,
    double hc_pre_eps,
    double hc_sinkhorn_eps,
    double hc_post_mult_value,
    int64_t sinkhorn_repeat);
#endif

#ifdef VLLM_GDN_ENABLED
void gdn_attention(
    torch::Tensor& core_attn_out,
    torch::Tensor& z,
    const torch::Tensor& projected_states_qkvz,
    const torch::Tensor& projected_states_ba,
    const int64_t num_k_heads,
    const int64_t num_v_heads,
    const int64_t head_k_dim,
    const int64_t head_v_dim,
    torch::Tensor& conv_state,
    torch::Tensor& ssm_state,
    const torch::Tensor& conv_weights,
    const std::optional<torch::Tensor>& conv_bias,
    const std::string& activation,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    const int64_t num_prefills,
    const int64_t num_decodes,
    const int64_t num_spec_decodes,
    const std::optional<torch::Tensor>& has_initial_state,
    const std::optional<torch::Tensor>& non_spec_query_start_loc,
    const std::optional<torch::Tensor>& non_spec_token_indx,
    const std::optional<torch::Tensor>& non_spec_state_indices_tensor,
    const std::optional<torch::Tensor>& spec_query_start_loc,
    const std::optional<torch::Tensor>& spec_token_indx,
    const std::optional<torch::Tensor>& spec_state_indices_tensor,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    const int64_t num_actual_tokens,
    const int64_t tp_size,
    const bool reorder_input);
#endif

bool is_bmg_g21(int64_t device_index);

bool is_bmg_g31(int64_t device_index);

bool is_bmg(int64_t device_index);

bool is_pvc(int64_t device_index);

bool is_xe2_arch(int64_t device_index);

bool is_xe3_arch(int64_t device_index);

void exponential_2d_(
    torch::Tensor& tensor,
    torch::Tensor& seeds,  // should on CPU
    const double lambda);

void topk_topp_sampler(
    torch::Tensor& random_sampled,
    const std::optional<torch::Tensor>& logits_to_return,
    torch::Tensor& logits,
    const std::optional<torch::Tensor>& k,
    const std::optional<torch::Tensor>& p,
    const std::string& logprobs_mode,
    torch::Tensor& seeds,  // should on CPU
    const double lambda);

#ifdef VLLM_MQA_LOGITS_ENABLED
torch::Tensor fp8_mqa_logits(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& kv_scales,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seqlen_ks,
    const torch::Tensor& cu_seqlen_ke);

torch::Tensor fp8_paged_mqa_logits(
    const torch::Tensor& q_fp8,
    const torch::Tensor& kv_cache_fp8,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_tables,
    const c10::optional<at::Tensor>& schedule_metadata,
    int64_t max_model_len);
#endif

std::string get_onednn_version();

void deepseek_fused_indexer_q_rope_fp8(
    const torch::Tensor& q,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin_cache,
    const torch::Tensor& index_weights,
    double softmax_scale,
    double head_scale,
    torch::Tensor& q_fp8,
    torch::Tensor& weights_out);

void deepseek_fused_indexer_q_rope_mxfp4(
    const torch::Tensor& q,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin_cache,
    const torch::Tensor& index_weights,
    double softmax_scale,
    double head_scale,
    torch::Tensor& packed_out,
    torch::Tensor& scales_out,
    torch::Tensor& weights_out);
