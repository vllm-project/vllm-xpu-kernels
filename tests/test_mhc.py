# SPDX-License-Identifier: Apache-2.0
"""Tests for the MHC layers.

Run `pytest tests/test_mhc.py`.
"""

import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [1, 33],
        "hidden_size": [1024],
        "hc_mult": [2, 4],
    },
}

def mhc_pre_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (post_mix [..., hc, 1] fp32, comb_mix [..., hc, hc] fp32,
    layer_input [..., H] bf16)."""
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32

    hc = residual.shape[-2]
    H = residual.shape[-1]
    hc3 = hc * 2 + hc * hc
    hcH = hc * H
    assert fn.shape == (hc3, hcH)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc3,)

    outer_shape = residual.shape[:-2]
    N = int(torch.tensor(outer_shape).prod().item()) if outer_shape else 1

    res_fp32 = residual.reshape(N, hc, H).to(torch.float32)
    res_2d = res_fp32.reshape(N, hcH)

    gemm_out = res_2d @ fn.t()
    sqrsum = (res_2d * res_2d).sum(dim=-1)
    rms = torch.rsqrt(sqrsum / hcH + rms_eps)
    mixes = gemm_out * rms.unsqueeze(-1)

    pre_raw = mixes[:, :hc]
    post_raw = mixes[:, hc : 2 * hc]
    cm_raw = mixes[:, 2 * hc :].reshape(N, hc, hc)

    pre_mix = torch.sigmoid(pre_raw * hc_scale[0] + hc_base[:hc]) + hc_pre_eps
    post_mix = (
        torch.sigmoid(post_raw * hc_scale[1] + hc_base[hc : 2 * hc])
        * hc_post_mult_value
    )
    cm = cm_raw * hc_scale[2] + hc_base[2 * hc :].reshape(hc, hc)

    cm = torch.softmax(cm, dim=-1) + hc_sinkhorn_eps
    cm = cm / (cm.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        cm = cm / (cm.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        cm = cm / (cm.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = torch.einsum("nj,njh->nh", pre_mix, res_fp32).to(torch.bfloat16)

    return (
        post_mix.reshape(*outer_shape, hc, 1),
        cm.reshape(*outer_shape, hc, hc),
        layer_input.reshape(*outer_shape, H),
    )

def mhc_post_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    mixed_residual = torch.einsum(
        "...ij,...ih->...jh",
        comb_res_mix.to(torch.float32),
        residual.to(torch.float32),
    )
    post_term = post_layer_mix.to(torch.float32) * x.unsqueeze(-2).to(torch.float32)
    return (mixed_residual + post_term).to(residual.dtype)

def hc_head_fused_reference(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    num_tokens = hs_flat.shape[0]
    if num_tokens == 0:
        return
    x = hs_flat.reshape(num_tokens, hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    rsqrt = torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)
    pre_mix = torch.sigmoid(mixes * rsqrt * hc_scale[0] + hc_base) + hc_eps
    result = torch.sum(pre_mix.unsqueeze(-1) * hs_flat.to(torch.float32), dim=1).to(
        out.dtype
    )
    out.copy_(result)

@pytest.fixture
def test_data(request):
    device = torch.device('xpu')
    dtype = torch.bfloat16
    return device, dtype

@pytest.mark.parametrize("num_tokens", [1, 33, 64, 4096])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_pre(num_tokens: int, hidden_size: int, hc_mult: int, test_data):
    device, dtype = test_data
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult

    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=dtype, device=device)
    fn = torch.randn((hc_mult3, hc_mult * hidden_size), dtype=torch.float32, device=device)
    hc_scale = torch.randn((3,), dtype=torch.float32, device=device)
    hc_base = torch.randn((hc_mult3,), dtype=torch.float32, device=device)

    rms_eps = 1e-6
    hc_pre_eps = 1e-6
    hc_sinkhorn_eps = 1e-6
    hc_post_mult_value = 1.0
    sinkhorn_repeat = 3

    post_mix, comb_mix, layer_input = torch.ops._xpu_C.mhc_pre(
        residual, fn, hc_scale, hc_base,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
    )

    post_mix_ref, comb_mix_ref, layer_input_ref = mhc_pre_reference(
        residual, fn, hc_scale, hc_base,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
    )

    torch.testing.assert_close(post_mix, post_mix_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(comb_mix, comb_mix_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(layer_input, layer_input_ref, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("num_tokens", [1, 33, 64, 4096])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_post(num_tokens: int, hidden_size: int, hc_mult: int, test_data):
    device, dtype = test_data
    
    x = torch.randn((num_tokens, hidden_size), dtype=dtype, device=device)
    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=dtype, device=device)
    post_mix = torch.randn((num_tokens, hc_mult, 1), dtype=torch.float32, device=device)
    comb_mix = torch.randn((num_tokens, hc_mult, hc_mult), dtype=torch.float32, device=device)

    xpu_out = torch.ops._xpu_C.mhc_post(x, residual, post_mix, comb_mix)
    torch_out = mhc_post_reference(x, residual, post_mix, comb_mix)

    torch.testing.assert_close(xpu_out, torch_out, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("num_tokens", [1, 33, 64, 4096])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("hc_mult", [4])
def test_hc_head_fused(num_tokens: int, hidden_size: int, hc_mult: int, test_data):
    device, dtype = test_data
    
    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=dtype, device=device)
    fn = torch.randn((hc_mult, hc_mult * hidden_size), dtype=torch.float32, device=device)
    hc_scale = torch.randn((1,), dtype=torch.float32, device=device)
    hc_base = torch.randn((hc_mult,), dtype=torch.float32, device=device)
    
    out_head = torch.empty((num_tokens, hidden_size), dtype=dtype, device=device)
    out_torch = torch.empty((num_tokens, hidden_size), dtype=dtype, device=device)
    
    rms_eps = 1e-6
    hc_eps = 1e-6

    torch.ops._xpu_C.hc_head_fused(
        residual, fn, hc_scale, hc_base,
        out_head, hidden_size, rms_eps, hc_eps, hc_mult
    )
    
    hc_head_fused_reference(
        residual, fn, hc_scale, hc_base,
        out_torch, hidden_size, rms_eps, hc_eps, hc_mult
    )

    torch.testing.assert_close(out_head, out_torch, atol=1e-2, rtol=1e-2)