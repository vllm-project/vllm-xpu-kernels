#pragma once

#include <torch/all.h>

void moe_sum(torch::Tensor& input, torch::Tensor& output);

std::tuple<torch::Tensor, torch::Tensor> grouped_topk(
    torch::Tensor const& scores, torch::Tensor const& scores_with_bias,
    int64_t n_group, int64_t topk_group, int64_t topk, bool renormalize,
    double routed_scaling_factor);
