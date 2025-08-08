#pragma once

#include <torch/all.h>

void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
              double epsilon);

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        torch::Tensor& weight, double epsilon);

std::tuple<torch::Tensor, torch::Tensor> grouped_topk(
    const torch::Tensor& hidden_states, const torch::Tensor& gating_output,
    const int64_t n_topk, const bool renormalize, const int64_t n_expert_group,
    const int64_t n_topk_group, const c10::string_view scoring_func,
    const c10::optional<torch::Tensor>& bias);
