#pragma once

#include <torch/all.h>

void moe_sum(torch::Tensor& input, torch::Tensor& output);

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad);

void batched_moe_align_block_size(int64_t max_tokens_per_batch,
                                  int64_t block_size,
                                  torch::Tensor const& expert_num_tokens,
                                  torch::Tensor sorted_ids,
                                  torch::Tensor expert_ids,
                                  torch::Tensor num_tokens_post_pad);

std::tuple<torch::Tensor, torch::Tensor> grouped_topk(
    torch::Tensor const& scores, torch::Tensor const& scores_with_bias,
    int64_t n_group, int64_t topk_group, int64_t topk, bool renormalize,
    double routed_scaling_factor);

std::tuple<torch::Tensor, torch::Tensor> fused_grouped_topk(
    const torch::Tensor& hidden_states, const torch::Tensor& gating_output,
    const int64_t n_topk, const bool renormalize, const int64_t n_expert_group,
    const int64_t n_topk_group, const c10::string_view scoring_func,
    const double routed_scaling_factor,
    const c10::optional<torch::Tensor>& bias);

void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output, const bool renormalize);