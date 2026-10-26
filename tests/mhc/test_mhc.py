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

# Hidden sizes exercising the different compile-time ITERS buckets of the
# fused epilogue RMSNorm (ITERS = ceil(H / (256 * 8))):
#   2048 -> 1, 4096 -> 2, 6144 -> 3, 7168 -> 4, 12288 -> 6, 16384 -> 8
# 24576 exceeds the register-resident limit and exercises the ITERS == 0
# recompute fallback.
FUSE_NORM_SHAPE_CASES = [
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

NORM_EPS = 1e-6


def rmsnorm_bf16(x: torch.Tensor, weight: torch.Tensor, eps: float):
    """Reference RMSNorm matching the fused epilogue.

    The kernel normalizes the value that would otherwise have been stored to
    `layer_input`, i.e. an already-bf16-rounded tensor. Taking a bf16 input
    here keeps the reference numerically aligned with the fused kernel.
    """
    xf = x.to(torch.float32)
    inv_rms = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return (xf * inv_rms * weight.to(torch.float32)).to(torch.bfloat16)


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
    norm_weight: torch.Tensor = None,
    norm_eps: float = NORM_EPS,
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

    # Optional fused epilogue RMSNorm.
    if norm_weight is not None:
        layer_input = rmsnorm_bf16(layer_input, norm_weight, norm_eps)

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
    norm_weight: torch.Tensor = None,
    norm_eps: float = NORM_EPS,
):
    """Composed reference: mhc_post then mhc_pre (+ optional fused norm)."""
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
        norm_weight,
        norm_eps,
    )
    return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur


def make_norm_weight(hidden_size: int, device: str, fuse_norm: bool):
    """Norm weight tensor, or None when the fused epilogue is disabled."""
    if not fuse_norm:
        return None
    # Center around 1.0 so the normalized output stays in a sane bf16 range.
    return (
        1.0
        + 0.1
        * torch.randn((hidden_size,), dtype=torch.float32, device=device)
    ).to(torch.bfloat16)


@pytest.mark.parametrize("fuse_norm", [False, True])
@pytest.mark.parametrize("num_tokens,hidden_size", MHC_PRE_CASES)
def test_mhc_pre(num_tokens: int, hidden_size: int, fuse_norm: bool):
    if fuse_norm and hidden_size > 8192:
        pytest.skip(
            "fused norm requires hidden_size <= 8192 (WG_THREADS*VEC*MAX_TILES)"
        )
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
    norm_weight = make_norm_weight(hidden_size, device, fuse_norm)

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
        norm_weight,
        NORM_EPS,
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
        norm_weight,
        NORM_EPS,
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
            layer_input.flatten().to(torch.float32),
            ref_layer_input.flatten().to(torch.float32),
            dim=0,
        )
        assert cos_sim > 0.99, f"Cosine similarity too low: {cos_sim.item()}"


@pytest.mark.parametrize("num_tokens,hidden_size", FUSE_NORM_SHAPE_CASES)
def test_mhc_pre_fused_norm_shapes(num_tokens: int, hidden_size: int):
    """Cover every compile-time ITERS bucket of the fused RMSNorm epilogue.

    Also asserts that the fused result matches running the unfused kernel
    followed by a separate RMSNorm, which is the property the fusion must
    preserve.
    """
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
    norm_weight = make_norm_weight(hidden_size, device, True)

    rms_eps = 1e-6
    hc_pre_eps = 1e-3
    hc_sinkhorn_eps = 1e-3
    hc_post_mult_value = 1.0
    sinkhorn_repeat = 20

    common = (
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

    fused_post, fused_comb, fused_layer = torch.ops._xpu_C.mhc_pre(
        *common, norm_weight, NORM_EPS
    )
    plain_post, plain_comb, plain_layer = torch.ops._xpu_C.mhc_pre(
        *common, None, NORM_EPS
    )

    # The gating outputs must be bit-identical between the two paths.
    torch.testing.assert_close(fused_post, plain_post, atol=0.0, rtol=0.0)
    torch.testing.assert_close(fused_comb, plain_comb, atol=0.0, rtol=0.0)

    # The fused output must match unfused-kernel + separate RMSNorm.
    expected = rmsnorm_bf16(plain_layer, norm_weight, NORM_EPS)
    torch.testing.assert_close(fused_layer, expected, atol=1e-2, rtol=1e-2)


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


@pytest.mark.parametrize("fuse_norm", [False, True])
@pytest.mark.parametrize("num_tokens,hidden_size", FUSED_POST_PRE_CASES)
def test_mhc_fused_post_pre(
    num_tokens: int,
    hidden_size: int,
    fuse_norm: bool,
):
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
    norm_weight = make_norm_weight(hidden_size, device, fuse_norm)

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
            norm_weight,
            NORM_EPS,
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
            norm_weight,
            NORM_EPS,
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
            layer_input_cur.flatten().to(torch.float32),
            ref_layer_input.flatten().to(torch.float32),
            dim=0,
        )
        assert cos_sim > 0.99, f"Cosine similarity too low: {cos_sim.item()}"


@pytest.mark.parametrize("num_tokens,hidden_size", FUSE_NORM_SHAPE_CASES)
def test_mhc_fused_post_pre_norm_matches_unfused(
    num_tokens: int,
    hidden_size: int,
):
    """Fused-norm post_pre must equal unfused post_pre + separate RMSNorm."""
    torch.manual_seed(0)
    device = "xpu"
    hc3 = HC * 2 + HC * HC

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
    fn = torch.randn(
        (hc3, HC * hidden_size),
        dtype=torch.float32,
        device=device,
    )
    hc_scale = torch.randn((3,), dtype=torch.float32, device=device)
    hc_base = torch.randn((hc3,), dtype=torch.float32, device=device)
    norm_weight = make_norm_weight(hidden_size, device, True)

    common = (
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        1e-6,   # rms_eps
        1e-3,   # hc_pre_eps
        1e-3,   # hc_sinkhorn_eps
        1.0,    # hc_post_mult_value
        20,     # sinkhorn_repeat
    )

    _, f_post, f_comb, f_layer = torch.ops._xpu_C.mhc_fused_post_pre(
        *common, norm_weight, NORM_EPS
    )
    _, p_post, p_comb, p_layer = torch.ops._xpu_C.mhc_fused_post_pre(
        *common, None, NORM_EPS
    )

    torch.testing.assert_close(f_post, p_post, atol=0.0, rtol=0.0)
    torch.testing.assert_close(f_comb, p_comb, atol=0.0, rtol=0.0)

    expected = rmsnorm_bf16(p_layer, norm_weight, NORM_EPS)
    torch.testing.assert_close(f_layer, expected, atol=1e-2, rtol=1e-2)