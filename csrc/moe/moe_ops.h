#pragma once

#include <torch/all.h>

void moe_sum(torch::Tensor& input, torch::Tensor& output);

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
