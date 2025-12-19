#include <torch/all.h>
#include "utils.h"
#include <iostream>

typedef at::BFloat16 bfloat16;

void fused_moe_prologue(
    torch::Tensor input,
    torch::Tensor token_selected_experts,
    torch::Tensor token_final_scales,
    torch::Tensor workspace,
    int64_t hidden_size,
    int64_t inter_size,
    int64_t num_experts_on_rank);