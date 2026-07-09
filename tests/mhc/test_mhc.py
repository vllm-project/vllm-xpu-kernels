# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

# Override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens,hidden_size": [
            (1, 4096),
            (256, 4096),
            (1, 7168),
            (256, 7168),
        ],
    },
}

MHC_PRE_CASES = [
    (1, 4096),
    (33, 4096),
    (256, 4096),
    (1024, 4096),
    (2048, 4096),
    (1, 7168),
    (33, 7168),
    (256, 7168),
    (1024, 7168),
    (2048, 7168),
]

MHC_POST_CASES = [
    (1, 4096),
    (33, 4096),
    (256, 4096),
    (1024, 4096),
    (1, 7168),
    (33, 7168),
    (256, 7168),
    (1024, 7168),
]

HC_HEAD_CASES = [
    (1, 4096),
    (33, 4096),
    (256, 4096),
    (1024, 4096),
    (1, 7168),
    (33, 7168),
    (256, 7168),
    (1024, 7168),
]

FUSED_POST_PRE_CASES = [
    (1, 4096),
    (33, 4096),
    (256, 4096),
    (1024, 4096),
    (2048, 4096),
    (1, 7168),
    (33, 7168),
    (256, 7168),
    (1024, 7168),
    (2048, 7168),
]

HC = 4


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
):
    sinkhorn_repeat = int(sinkhorn_repeat)
    hidden_size = residual.shape[-1]
    hc3 = HC * 2 + HC * HC

    x = residual.view(-1, HC, hidden_size)
    num_tokens = x.shape[0]
    x2d = x.view(num_tokens, HC * hidden_size).to(torch.float32)

    mixes = torch.matmul(x2d, fn.t())
    sqrsum = x2d.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (HC * hidden_size) + rms_eps)

    pre_logits = mixes[:, :HC] * hc_scale[0] + hc_base[:HC]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

    post_logits = mixes[:, HC:2 * HC] * hc_scale[1] + hc_base[HC:2 * HC]
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = (
        mixes[:, 2 * HC:].view(num_tokens, HC, HC) * hc_scale[2]
        + hc_base[2 * HC:hc3].view(1, HC, HC)
    )
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (
            comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps
        )
        comb_mix = comb_mix / (
            comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps
        )

    layer_input = torch.sum(
        pre_mix.unsqueeze(-1) * x.to(torch.float32), dim=1).to(torch.bfloat16)

    return post_mix.view(num_tokens, HC, 1), comb_mix, layer_input


def mhc_post_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
):
    mixed_residual = torch.einsum(
        "...ij,...ih->...jh",
        comb_res_mix.to(torch.float32),
        residual.to(torch.float32),
    )
    post_term = post_layer_mix.to(torch.float32) * x.unsqueeze(-2).to(
        torch.float32
    )
    return (mixed_residual + post_term).to(torch.bfloat16)


def hc_head_fused_reference(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
):
    num_tokens = hs_flat.shape[0]
    hidden_size = hs_flat.shape[-1]
    x = hs_flat.view(num_tokens, HC * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    rsqrt = torch.rsqrt(sqrsum / (HC * hidden_size) + rms_eps)
    pre_mix = torch.sigmoid(mixes * rsqrt * hc_scale[0] + hc_base) + hc_eps
    return torch.sum(
        pre_mix.unsqueeze(-1) * hs_flat.to(torch.float32),
        dim=1,
    ).to(torch.bfloat16)


def mhc_fused_post_pre_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
):
    """Composed reference: mhc_post then mhc_pre."""
    residual_cur = mhc_post_reference(x, residual, post_layer_mix, comb_res_mix)
    post_mix_cur, comb_mix_cur, layer_input_cur = mhc_pre_reference(
        residual_cur,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur


@pytest.mark.parametrize("num_tokens,hidden_size", MHC_PRE_CASES)
def test_mhc_pre(num_tokens: int, hidden_size: int):
    torch.manual_seed(0)
    device = "xpu"
    hc3 = HC * 2 + HC * HC

    residual = torch.randn(
        (num_tokens, HC, hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    fn = torch.randn(
        (hc3, HC * hidden_size),
        dtype=torch.float32,
        device=device,
    )
    hc_scale = torch.randn((3,), dtype=torch.float32, device=device)
    hc_base = torch.randn((hc3,), dtype=torch.float32, device=device)

    rms_eps = 1e-6
    hc_pre_eps = 1e-3
    hc_sinkhorn_eps = 1e-3
    hc_post_mult_value = 1.0
    sinkhorn_repeat = 20

    post_mix, comb_mix, layer_input = torch.ops._xpu_C.mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )

    ref_post_mix, ref_comb_mix, ref_layer_input = mhc_pre_reference(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )

    if num_tokens < 128:
        torch.testing.assert_close(post_mix, ref_post_mix, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(comb_mix, ref_comb_mix, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            layer_input,
            ref_layer_input,
            atol=1e-2,
            rtol=1e-2,
        )
    else:
        # For token numbers >= 128, we use a looser tolerance because
        # the kernel uses tf32 internally.
        torch.testing.assert_close(post_mix, ref_post_mix, atol=6e-2, rtol=6e-2)
        torch.testing.assert_close(comb_mix, ref_comb_mix, atol=6e-2, rtol=6e-2)
        cos_sim = torch.cosine_similarity(
            layer_input.flatten(),
            ref_layer_input.flatten(),
            dim=0,
        )
        assert cos_sim > 0.99, f"Cosine similarity too low: {cos_sim.item()}"


@pytest.mark.parametrize("num_tokens,hidden_size", MHC_POST_CASES)
def test_mhc_post(num_tokens: int, hidden_size: int):
    torch.manual_seed(0)
    device = "xpu"

    x = torch.randn(
        (num_tokens, hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    residual = torch.randn(
        (num_tokens, HC, hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    post_mix = torch.randn(
        (num_tokens, HC, 1),
        dtype=torch.float32,
        device=device,
    )
    comb_mix = torch.randn(
        (num_tokens, HC, HC),
        dtype=torch.float32,
        device=device,
    )

    out = torch.ops._xpu_C.mhc_post(x, residual, post_mix, comb_mix)
    ref = mhc_post_reference(x, residual, post_mix, comb_mix)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("num_tokens,hidden_size", HC_HEAD_CASES)
def test_hc_head_fused(num_tokens: int, hidden_size: int):
    torch.manual_seed(0)
    device = "xpu"

    hs_flat = torch.randn(
        (num_tokens, HC, hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    fn = torch.randn((HC, HC * hidden_size), dtype=torch.float32, device=device)
    hc_scale = torch.randn((1,), dtype=torch.float32, device=device)
    hc_base = torch.randn((HC,), dtype=torch.float32, device=device)
    out = torch.empty(
        (num_tokens, hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )

    rms_eps = 1e-6
    hc_eps = 1e-6

    torch.ops._xpu_C.hc_head_fused(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        out,
        rms_eps,
        hc_eps,
    )
    ref = hc_head_fused_reference(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
    )

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("num_tokens,hidden_size", FUSED_POST_PRE_CASES)
def test_mhc_fused_post_pre(num_tokens: int, hidden_size: int):
    torch.manual_seed(0)
    device = "xpu"
    hc3 = HC * 2 + HC * HC

    # Inputs for the post stage (from previous layer's mhc_pre)
    x = torch.randn(
        (num_tokens, hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    residual = torch.randn(
        (num_tokens, HC, hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    post_layer_mix = torch.randn(
        (num_tokens, HC, 1),
        dtype=torch.float32,
        device=device,
    )
    comb_res_mix = torch.randn(
        (num_tokens, HC, HC),
        dtype=torch.float32,
        device=device,
    )

    # Weights for the pre stage (current layer)
    fn = torch.randn(
        (hc3, HC * hidden_size),
        dtype=torch.float32,
        device=device,
    )
    hc_scale = torch.randn((3,), dtype=torch.float32, device=device)
    hc_base = torch.randn((hc3,), dtype=torch.float32, device=device)

    rms_eps = 1e-6
    hc_pre_eps = 1e-3
    hc_sinkhorn_eps = 1e-3
    hc_post_mult_value = 1.0
    sinkhorn_repeat = 20

    residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur = (
        torch.ops._xpu_C.mhc_fused_post_pre(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    )

    ref_residual, ref_post_mix, ref_comb_mix, ref_layer_input = (
        mhc_fused_post_pre_reference(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    )

    torch.testing.assert_close(residual_cur, ref_residual, atol=1e-2, rtol=1e-2)
    if num_tokens < 128:
        torch.testing.assert_close(
            post_mix_cur,
            ref_post_mix,
            atol=5e-3,
            rtol=5e-3,
        )
        torch.testing.assert_close(
            comb_mix_cur,
            ref_comb_mix,
            atol=5e-3,
            rtol=5e-3,
        )
        torch.testing.assert_close(
            layer_input_cur,
            ref_layer_input,
            atol=2e-2,
            rtol=2e-2,
        )
    else:
        # For token numbers >= 128, mhc_pre uses tf32 DPAS path,
        # so we use a looser tolerance.
        torch.testing.assert_close(
            post_mix_cur,
            ref_post_mix,
            atol=6e-2,
            rtol=6e-2,
        )
        torch.testing.assert_close(
            comb_mix_cur,
            ref_comb_mix,
            atol=6e-2,
            rtol=6e-2,
        )
        cos_sim = torch.cosine_similarity(
            layer_input_cur.flatten(),
            ref_layer_input.flatten(),
            dim=0,
        )
        assert cos_sim > 0.99, f"Cosine similarity too low: {cos_sim.item()}"
