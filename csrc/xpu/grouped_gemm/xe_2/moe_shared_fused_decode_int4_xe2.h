#pragma once

#include <torch/all.h>

// GEMM M-tile of the int4 fused routed + shared-expert decode kernel.
constexpr int64_t kMoeSharedFusedInt4TileM8 = 8;
constexpr int64_t kMoeSharedFusedInt4TileM16 = 16;

// See moe_shared_fused_decode_int4_xe2.cpp for the recipe and weight layouts.
void moe_shared_fused_decode_int4_xe2(
    torch::Tensor& output,
    const torch::Tensor& x,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    const torch::Tensor& w13,
    const torch::Tensor& w13_scale,
    const torch::Tensor& w2,
    const torch::Tensor& w2_scale,
    const torch::Tensor& shared_w13,
    const torch::Tensor& shared_w13_scale,
    const torch::Tensor& shared_w2,
    const torch::Tensor& shared_w2_scale,
    const torch::Tensor& shared_gate,
    torch::Tensor& ws,
    int64_t tile_m);
