// MHC (Manifold-Constrained Hyper-Connections) interface declarations.
//
// These functions are implemented in mhc_pre.cpp, mhc_post.cpp,
// hc_head_fused.cpp, and mhc_fused_post_pre.cpp.

#pragma once

#include <ATen/ATen.h>
#include <tuple>

// mhc_pre: RMS-norm projection → sigmoid/Sinkhorn gating → weighted reduction
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

// mhc_post: einsum-based residual mixing + post-layer scaling
at::Tensor mhc_post(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post_layer_mix,
    const at::Tensor& comb_res_mix);

// hc_head_fused: fused projection + RMS-norm + sigmoid + weighted reduction
void hc_head_fused(
    const at::Tensor& hs_flat,
    const at::Tensor& fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    at::Tensor& out,
    double rms_eps,
    double hc_eps);

// mhc_fused_post_pre: composed post→pre in one call
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
