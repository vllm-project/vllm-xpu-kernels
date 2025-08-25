#pragma once

#include <torch/all.h>

std::tuple<torch::Tensor, torch::Tensor> grouped_topk(
    const torch::Tensor& hidden_states, const torch::Tensor& gating_output,
    const int64_t n_topk, const bool renormalize, const int64_t n_expert_group,
    const int64_t n_topk_group, const c10::string_view scoring_func,
    const c10::optional<torch::Tensor>& bias);


void moe_sum(torch::Tensor& input, torch::Tensor& output);
