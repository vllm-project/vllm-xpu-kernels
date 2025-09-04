#pragma once

#include <torch/all.h>

void moe_sum(torch::Tensor& input, torch::Tensor& output);
