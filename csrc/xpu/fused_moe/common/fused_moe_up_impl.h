#pragma once
#include <torch/all.h>

#include "fused_moe_arch.h"

FUSED_MOE_NS_BEGIN

torch::Tensor fused_moe_up_impl(
    torch::Tensor& ptr_A,
    const c10::optional<at::Tensor>& ptr_A_scale,
    torch::Tensor& ptr_B,
    const c10::optional<at::Tensor>& ptr_B_scale,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor& ptr_D,
    torch::Tensor& rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    std::string activation,
    double gemm1_clamp_limit);

FUSED_MOE_NS_END
