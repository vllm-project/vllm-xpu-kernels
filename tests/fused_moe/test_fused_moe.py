# SPDX-License-Identifier: Apache-2.0
import gc

import pytest
import torch

from tests.ops.fp8_quant_op import scaled_fp8_quant
from tests.utils import format_tc, seed_everything
from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

DEVICE = "xpu"

# shape for Llama-4-scout
FUSED_MOE_MNK_FACTORS = [
    (1, 5120, 8192),
    (4, 5120, 8192),
    (16, 5120, 8192),
    (8192, 5120, 8192),
]
NUM_EXPERTS = [16]
TOP_KS = [1]
EP_RANK = [0, 1, 2, 3]
EP_SIZE = [4]

# override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "m,n,k": [(1, 256, 128)],
        "e": [2],
        "topk": [1],
        "dtype": [torch.bfloat16],
        "has_bias": [True],
    },
}


def dequantize_uint4(qweight, scales, group_size):
    import numpy as np
    k = qweight.shape[1] * 2
    n = qweight.shape[0]
    unpack_idx = np.array([0, 1])
    data = qweight[:, [i // 2 for i in range(k)]]
    shift = (torch.tensor(unpack_idx[[i % 2 for i in range(k)]],
                          dtype=torch.int32,
                          device="xpu")[None, :].expand([n, -1]) * 4)
    dst_data = (data >> shift) & 0xF
    expand_scales = scales[:, [i // group_size for i in range(k)]]
    weight_16 = (dst_data - 8) * expand_scales

    return weight_16.to(scales.dtype)


def dequantize_mxfp4(qweight, scales, group_size, dtype):
    import numpy as np
    k = qweight.shape[1] * 2
    n = qweight.shape[0]
    unpack_idx = np.array([0, 1])
    data = qweight[:, [i // 2 for i in range(k)]]
    shift = (torch.tensor(unpack_idx[[i % 2 for i in range(k)]],
                          dtype=torch.int32,
                          device="xpu")[None, :].expand([n, -1]) * 4)
    dst_data = (data >> shift) & 0xF

    table = torch.tensor([
        +0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
                         dtype=dtype,
                         device="xpu")
    dst_data = table[dst_data]
    expand_scales = scales[:, [i // group_size for i in range(k)]]
    dst_scale = (expand_scales.to(torch.int32) << 7).to(torch.uint16).view(
        torch.bfloat16).to(dtype)
    weight_16 = dst_data * dst_scale

    return weight_16.to(dtype)


def ref_fused_moe(x,
                  w13,
                  w13_bias,
                  w2,
                  w2_bias,
                  flat_expert_weights,
                  flat_expert_indices,
                  num_per_tok,
                  activation,
                  num_experts,
                  ep_rank=0,
                  ep_size=1,
                  gemm1_clamp_limit=None):
    expert_start_id = num_experts * ep_rank
    expert_end_id = expert_start_id + num_experts
    expert_cache = torch.zeros_like(x)
    idxs = flat_expert_indices.argsort()
    counts = flat_expert_indices.bincount().cpu().numpy()
    tokens_per_expert = counts.cumsum()
    token_idxs = idxs // num_per_tok
    for expert_id, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        if (start_idx == end_idx) or (expert_id
                                      < expert_start_id) or (expert_id
                                                             >= expert_end_id):
            continue
        exp_token_idxs = token_idxs[start_idx:end_idx]
        expert_tokens = x[exp_token_idxs]

        expert_w13 = w13[expert_id, :, :]
        w1, w3 = torch.split(expert_w13,
                             int(list(expert_w13.shape)[0] / 2),
                             dim=0)
        if w13_bias is not None:
            w1_bias, w3_bias = w13_bias[expert_id, :].chunk(2)
        act_fn = torch.nn.SiLU()
        gemm1 = (expert_tokens.to(torch.float32) @ w1.T.to(torch.float32))
        if w13_bias is not None:
            gemm1 += w1_bias.to(torch.float32)
        up = (expert_tokens.to(torch.float32) @ w3.T.to(torch.float32))
        if w13_bias is not None:
            up += w3_bias.to(torch.float32)
        if gemm1_clamp_limit is not None and gemm1_clamp_limit > 0:
            gemm1.clamp_(max=gemm1_clamp_limit)
            up.clamp_(min=-gemm1_clamp_limit, max=gemm1_clamp_limit)
        gate = act_fn(gemm1)
        expert_out = ((gate * up) @ w2[expert_id, :, :].T.to(torch.float32))
        if w2_bias is not None:
            expert_out += w2_bias[expert_id, :].to(torch.float32)
        expert_out = expert_out.to(x.dtype)
        expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
        expert_cache.scatter_reduce_(0,
                                     exp_token_idxs.view(-1, 1).repeat(
                                         1, x.shape[-1]),
                                     expert_out,
                                     reduce='sum')

    return expert_cache


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16],
                         ids=format_tc)
@pytest.mark.parametrize("w_dtype",
                         [torch.float8_e5m2, torch.float8_e4m3fn, None],
                         ids=format_tc)
@pytest.mark.parametrize("has_bias", [True, False])
def test_fused_moe(m, n, k, e, topk, dtype, w_dtype, has_bias):
    seed_everything(7)

    input_len = m
    hidden_size = n
    intermediate_size = k
    num_experts = e

    a = torch.randn((input_len, hidden_size), device=DEVICE, dtype=dtype) / 16
    w13 = torch.randn((num_experts, 2 * intermediate_size, hidden_size),
                      device=DEVICE,
                      dtype=dtype) / 16
    w2 = torch.randn((num_experts, hidden_size, intermediate_size),
                     device=DEVICE,
                     dtype=dtype) / 16
    ref_a = a.clone()

    if has_bias:
        w13_bias = torch.randn(
            (num_experts, 2 * intermediate_size), device=DEVICE,
            dtype=dtype) / 16
        w2_bias = torch.randn(
            (num_experts, hidden_size), device=DEVICE, dtype=dtype) / 16
    else:
        w13_bias = None
        w2_bias = None
    # moe gate
    scores = torch.randn((input_len, num_experts),
                         device=DEVICE,
                         dtype=torch.float32)
    expert_scores, expert_indices = torch.topk(scores,
                                               k=topk,
                                               dim=-1,
                                               sorted=False)

    flat_expert_indices = expert_indices.view(-1)
    flat_expert_weights = expert_scores.view(-1, 1)

    if w_dtype is not None:
        w13_fp8 = torch.empty_like(w13, dtype=w_dtype)
        w2_fp8 = torch.empty_like(w2, dtype=w_dtype)

        # scale
        random_exponents = torch.randint(-3, 4, (num_experts, ), device=DEVICE)
        w13_scales = torch.pow(2.0, random_exponents.float())
        random_exponents = torch.randint(-3, 4, (num_experts, ), device=DEVICE)
        w2_scales = torch.pow(2.0, random_exponents.float())

        for i in range(num_experts):
            w13_fp8[i], _ = scaled_fp8_quant(w13[i],
                                             w13_scales[i].to(torch.float32),
                                             False,
                                             False,
                                             fp8_dtype=w_dtype)
            w2_fp8[i], _ = scaled_fp8_quant(w2[i],
                                            w2_scales[i].to(torch.float32),
                                            False,
                                            False,
                                            fp8_dtype=w_dtype)
        w13 = w13_fp8
        w2 = w2_fp8

        ref_w13 = torch.empty_like(w13_fp8, dtype=dtype)
        ref_w2 = torch.empty_like(w2_fp8, dtype=dtype)
        for i in range(num_experts):
            ref_w13[i] = w13_fp8[i].to(dtype) * w13_scales[i]
            ref_w2[i] = w2_fp8[i].to(dtype) * w2_scales[i]
    else:
        w13_scales = None
        w2_scales = None
        ref_w13 = w13
        ref_w2 = w2

    ref_out = ref_fused_moe(ref_a, ref_w13, w13_bias, ref_w2, w2_bias,
                            flat_expert_weights, flat_expert_indices, topk,
                            "silu", e)

    w13.data = w13.transpose(-1, -2).contiguous()
    w2.data = w2.transpose(-1, -2).contiguous()

    fused_moe_impl = XpuFusedMoe(
                w13=w13,
                w13_scales=w13_scales,
                w13_bias=w13_bias,
                w2=w2,
                w2_scales=w2_scales,
                w2_bias=w2_bias,
                n_experts_per_token=topk,
                activation="silu",
                num_experts=e,
            )

    output = torch.empty_like(ref_out)
    fused_moe_impl.apply(
        output=output,
        hidden_states=a,
        topk_weights=expert_scores,
        topk_ids=expert_indices,
    )

    if dtype == torch.float16:
        rtol = 1e-2
        atol = 1e-2
    else:
        rtol = 2e-2
        atol = 2e-2
    torch.testing.assert_close(output, ref_out, rtol=rtol, atol=atol)
    del fused_moe_impl


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("has_bias", [True, False])
def test_fused_moe_int4(m, n, k, e, topk, dtype, has_bias):
    seed_everything(7)
    torch.xpu.empty_cache()
    gc.collect()

    input_len = m
    hidden_size = n
    intermediate_size = k
    num_experts = e
    group_size = 128

    a = torch.randn((input_len, hidden_size), device=DEVICE, dtype=dtype) / 16
    w13 = (torch.randint(
        0,
        0xff, [num_experts, 2 * intermediate_size, hidden_size // 2],
        device=DEVICE)).to(torch.uint8)
    w2 = (torch.randint(0,
                        0xff,
                        [num_experts, hidden_size, intermediate_size // 2],
                        device=DEVICE)).to(torch.uint8)
    ref_a = a.clone()

    # scale
    group_num_13 = hidden_size // group_size
    group_num_2 = intermediate_size // group_size
    assert hidden_size % group_size == 0, \
        "hidden_size must be divisible by group_size"
    assert intermediate_size % group_size == 0, \
        "intermediate_size must be divisible by group_size"
    random_exponents = torch.randint(
        -5,
        -4, (num_experts, 2 * intermediate_size, group_num_13),
        device=DEVICE)
    w13_scales = torch.pow(2.0, random_exponents.float()).to(dtype)
    random_exponents = torch.randint(-5,
                                     -4,
                                     (num_experts, hidden_size, group_num_2),
                                     device=DEVICE)
    w2_scales = torch.pow(2.0, random_exponents.float()).to(dtype)

    if has_bias:
        w13_bias = torch.randn(
            (num_experts, 2 * intermediate_size), device=DEVICE,
            dtype=dtype) / 16
        w2_bias = torch.randn(
            (num_experts, hidden_size), device=DEVICE, dtype=dtype) / 16
    else:
        w13_bias = None
        w2_bias = None
    # moe gate
    scores = torch.randn((input_len, num_experts),
                         device=DEVICE,
                         dtype=torch.float32)
    expert_scores, expert_indices = torch.topk(scores,
                                               k=topk,
                                               dim=-1,
                                               sorted=False)

    flat_expert_indices = expert_indices.view(-1)
    flat_expert_weights = expert_scores.view(-1, 1)

    ref_13 = torch.empty(num_experts,
                         2 * intermediate_size,
                         hidden_size,
                         dtype=dtype,
                         device=DEVICE)
    ref_2 = torch.empty(num_experts,
                        hidden_size,
                        intermediate_size,
                        dtype=dtype,
                        device=DEVICE)

    for i in range(num_experts):
        ref_13[i] = dequantize_uint4(w13[i], w13_scales[i], group_size)
        ref_2[i] = dequantize_uint4(w2[i], w2_scales[i], group_size)

    ref_out = ref_fused_moe(ref_a, ref_13, w13_bias, ref_2, w2_bias,
                            flat_expert_weights, flat_expert_indices, topk,
                            "silu", e)
    
    fused_moe_impl = XpuFusedMoe(
                w13=w13,
                w13_scales=w13_scales,
                w13_bias=w13_bias,
                w2=w2,
                w2_scales=w2_scales,
                w2_bias=w2_bias,
                n_experts_per_token=topk,
                activation="silu",
                num_experts=e,
            )

    output = torch.empty_like(ref_out)
    fused_moe_impl.apply(
        output=output,
        hidden_states=a,
        topk_weights=expert_scores,
        topk_ids=expert_indices,
    )

    if dtype == torch.float16:
        rtol = 2e-2
        atol = 2e-2
    else:
        rtol = 2e-1
        atol = 2e-1
    torch.testing.assert_close(output, ref_out, rtol=rtol, atol=atol)


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16],
                         ids=format_tc)
@pytest.mark.parametrize("has_bias", [True, False])
def test_fused_moe_mxfp4(m, n, k, e, topk, dtype, has_bias):
    seed_everything(7)

    torch.xpu.empty_cache()
    gc.collect()

    input_len = m
    hidden_size = n
    intermediate_size = k
    num_experts = e
    group_size = 32

    a = torch.randn((input_len, hidden_size), device=DEVICE, dtype=dtype) / 16
    w13 = (torch.randint(
        0,
        0xff, [num_experts, 2 * intermediate_size, hidden_size // 2],
        device=DEVICE)).to(torch.uint8)
    w2 = (torch.randint(0,
                        0xff,
                        [num_experts, hidden_size, intermediate_size // 2],
                        device=DEVICE)).to(torch.uint8)
    ref_a = a.clone()

    # scale
    group_num_13 = hidden_size // group_size
    group_num_2 = intermediate_size // group_size
    w13_scales = torch.randint(
        0,
        0x6f, (num_experts, 2 * intermediate_size, group_num_13),
        dtype=torch.uint8,
        device=DEVICE)
    w2_scales = torch.randint(0,
                              0x6f, (num_experts, hidden_size, group_num_2),
                              dtype=torch.uint8,
                              device=DEVICE)

    if has_bias:
        w13_bias = torch.randn(
            (num_experts, 2 * intermediate_size), device=DEVICE,
            dtype=dtype) / 16
        w2_bias = torch.randn(
            (num_experts, hidden_size), device=DEVICE, dtype=dtype) / 16
    else:
        w13_bias = None
        w2_bias = None
    # moe gate
    scores = torch.randn((input_len, num_experts),
                         device=DEVICE,
                         dtype=torch.float32)
    expert_scores, expert_indices = torch.topk(scores,
                                               k=topk,
                                               dim=-1,
                                               sorted=False)

    flat_expert_indices = expert_indices.view(-1)
    flat_expert_weights = expert_scores.view(-1, 1)

    ref_13 = torch.empty(num_experts,
                         2 * intermediate_size,
                         hidden_size,
                         dtype=dtype,
                         device=DEVICE)
    ref_2 = torch.empty(num_experts,
                        hidden_size,
                        intermediate_size,
                        dtype=dtype,
                        device=DEVICE)

    for i in range(num_experts):
        ref_13[i] = dequantize_mxfp4(w13[i], w13_scales[i], group_size, dtype)
        ref_2[i] = dequantize_mxfp4(w2[i], w2_scales[i], group_size, dtype)

    ref_out = ref_fused_moe(ref_a, ref_13, w13_bias, ref_2, w2_bias,
                            flat_expert_weights, flat_expert_indices, topk,
                            "silu", e)

    fused_moe_impl = XpuFusedMoe(
                w13=w13.view(torch.float4_e2m1fn_x2),
                w13_scales=w13_scales,
                w13_bias=w13_bias,
                w2=w2.view(torch.float4_e2m1fn_x2),
                w2_scales=w2_scales,
                w2_bias=w2_bias,
                n_experts_per_token=topk,
                activation="silu",
                num_experts=e,
            )

    output = torch.empty_like(ref_out)
    fused_moe_impl.apply(
        output=output,
        hidden_states=a,
        topk_weights=expert_scores,
        topk_ids=expert_indices,
    )

    if dtype == torch.float16:
        rtol = 1e-2
        atol = 1e-2
    else:
        rtol = 2e-2
        atol = 2e-2
    torch.testing.assert_close(output, ref_out, rtol=rtol, atol=atol)



@pytest.mark.parametrize("m,n,k", [(1, 256, 128), (4, 256, 128)])
@pytest.mark.parametrize("e", [4])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16],
                         ids=format_tc)
@pytest.mark.parametrize("has_bias", [True, False])
def test_fused_moe_mxfp8(m, n, k, e, topk, dtype, has_bias):
    """Native MXFP8 fused MoE vs dequant-weight reference."""
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    seed_everything(7)
    torch.xpu.empty_cache()
    gc.collect()

    input_len = m
    hidden_size = n
    intermediate_size = k
    num_experts = e
    group_size = 32
    assert hidden_size % group_size == 0
    assert intermediate_size % group_size == 0

    a = torch.randn((input_len, hidden_size), device=DEVICE, dtype=dtype) / 16
    from tests.ops.mx_utils import to_mxfp

    w13 = torch.empty(num_experts,
                      hidden_size,
                      2 * intermediate_size,
                      dtype=torch.float8_e4m3fn,
                      device=DEVICE)
    w2 = torch.empty(num_experts,
                     intermediate_size,
                     hidden_size,
                     dtype=torch.float8_e4m3fn,
                     device=DEVICE)
    w13_scales = torch.empty(num_experts,
                             hidden_size // group_size,
                             2 * intermediate_size,
                             dtype=torch.uint8,
                             device=DEVICE)
    w2_scales = torch.empty(num_experts,
                            intermediate_size // group_size,
                            hidden_size,
                            dtype=torch.uint8,
                            device=DEVICE)
    ref_13 = torch.empty(num_experts,
                         hidden_size,
                         2 * intermediate_size,
                         dtype=dtype,
                         device=DEVICE)
    ref_2 = torch.empty(num_experts,
                        intermediate_size,
                        hidden_size,
                        dtype=dtype,
                        device=DEVICE)

    for i in range(num_experts):
        w13_hp = torch.randn(hidden_size,
                             2 * intermediate_size,
                             dtype=torch.float32) / 16
        w2_hp = torch.randn(intermediate_size,
                            hidden_size,
                            dtype=torch.float32) / 16
        # to_mxfp blocks last dim; E8M0 along K (dim 0) → quantize [N, K].
        sc13, lp13 = to_mxfp(w13_hp.transpose(0, 1).contiguous(),
                             format="mxfp8")
        sc2, lp2 = to_mxfp(w2_hp.transpose(0, 1).contiguous(), format="mxfp8")
        lp13 = lp13.transpose(0, 1).contiguous()
        lp2 = lp2.transpose(0, 1).contiguous()
        sc13 = sc13.transpose(0, 1).contiguous()
        sc2 = sc2.transpose(0, 1).contiguous()
        w13[i] = lp13.to(DEVICE)
        w2[i] = lp2.to(DEVICE)
        w13_scales[i] = sc13.view(torch.uint8).to(DEVICE)
        w2_scales[i] = sc2.view(torch.uint8).to(DEVICE)
        # Dequant gold (same as moe_utils.dequant_mxfp8_wei)
        ref_13[i] = (
            lp13.to(torch.float32) *
            sc13.view(torch.float8_e8m0fnu).to(torch.float32).repeat_interleave(
                group_size, dim=0)).to(dtype).to(DEVICE)
        ref_2[i] = (
            lp2.to(torch.float32) *
            sc2.view(torch.float8_e8m0fnu).to(torch.float32).repeat_interleave(
                group_size, dim=0)).to(dtype).to(DEVICE)

    if has_bias:
        w13_bias = torch.randn(
            (num_experts, 2 * intermediate_size), device=DEVICE,
            dtype=dtype) / 16
        w2_bias = torch.randn(
            (num_experts, hidden_size), device=DEVICE, dtype=dtype) / 16
    else:
        w13_bias = None
        w2_bias = None

    scores = torch.randn((input_len, num_experts),
                         device=DEVICE,
                         dtype=torch.float32)
    expert_scores, expert_indices = torch.topk(scores,
                                               k=topk,
                                               dim=-1,
                                               sorted=False)
    flat_expert_indices = expert_indices.view(-1)
    flat_expert_weights = expert_scores.view(-1, 1)

    # Local ref_fused_moe expects [E, N, K] and uses A @ W.T.
    ref_13_nk = ref_13.transpose(-1, -2).contiguous()
    ref_2_nk = ref_2.transpose(-1, -2).contiguous()
    ref_out = ref_fused_moe(a.clone(), ref_13_nk, w13_bias, ref_2_nk, w2_bias,
                            flat_expert_weights, flat_expert_indices, topk,
                            "silu", e)

    fused_moe_impl = XpuFusedMoe(
        w13=w13,
        w13_scales=w13_scales,
        w13_bias=w13_bias,
        w2=w2,
        w2_scales=w2_scales,
        w2_bias=w2_bias,
        n_experts_per_token=topk,
        activation="silu",
        num_experts=e,
    )
    assert fused_moe_impl.is_mxfp8
    assert fused_moe_impl._use_ref is False

    output = torch.empty_like(ref_out)
    fused_moe_impl.apply(
        output=output,
        hidden_states=a,
        topk_weights=expert_scores,
        topk_ids=expert_indices,
    )

    # Native path is W8A16 (bf16/fp16 A, MXFP8 B); gold is dequant-weight bf16.
    # Act-side qdq from moe_utils.ref_fused_moe is intentionally not applied.
    torch.testing.assert_close(output, ref_out, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("m,n,k", [(4, 256, 256)])
@pytest.mark.parametrize("e", [4])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16],
                         ids=format_tc)
@pytest.mark.parametrize("has_bias", [True, False])
def test_fused_moe_fp8block(m, n, k, e, topk, dtype, has_bias):
    """Native block-FP8 fused MoE (in-kernel scales) vs dequant-weight ref."""
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    seed_everything(7)
    torch.xpu.empty_cache()
    gc.collect()

    from vllm_xpu_kernels.moe_utils import dequant_fp8_block_wei

    input_len = m
    hidden_size = n
    intermediate_size = k
    num_experts = e
    assert hidden_size % 128 == 0
    assert intermediate_size % 128 == 0

    a = torch.randn((input_len, hidden_size), device=DEVICE, dtype=dtype) / 16
    w13 = torch.empty(num_experts,
                      hidden_size,
                      2 * intermediate_size,
                      dtype=torch.float8_e4m3fn,
                      device=DEVICE)
    w2 = torch.empty(num_experts,
                     intermediate_size,
                     hidden_size,
                     dtype=torch.float8_e4m3fn,
                     device=DEVICE)
    w13_scales = (torch.randn(num_experts,
                              hidden_size // 128,
                              (2 * intermediate_size) // 128,
                              device=DEVICE,
                              dtype=torch.float32).abs() + 0.01)
    w2_scales = (torch.randn(num_experts,
                             intermediate_size // 128,
                             hidden_size // 128,
                             device=DEVICE,
                             dtype=torch.float32).abs() + 0.01)
    ref_13 = torch.empty(num_experts,
                         hidden_size,
                         2 * intermediate_size,
                         dtype=dtype,
                         device=DEVICE)
    ref_2 = torch.empty(num_experts,
                        intermediate_size,
                        hidden_size,
                        dtype=dtype,
                        device=DEVICE)

    for i in range(num_experts):
        w13_hp = torch.randn(hidden_size,
                             2 * intermediate_size,
                             device=DEVICE,
                             dtype=torch.float32) / 16
        w2_hp = torch.randn(intermediate_size,
                            hidden_size,
                            device=DEVICE,
                            dtype=torch.float32) / 16
        w13[i] = w13_hp.to(torch.float8_e4m3fn)
        w2[i] = w2_hp.to(torch.float8_e4m3fn)
        ref_13[i] = dequant_fp8_block_wei(w13[i], w13_scales[i]).to(dtype)
        ref_2[i] = dequant_fp8_block_wei(w2[i], w2_scales[i]).to(dtype)

    if has_bias:
        w13_bias = torch.randn(
            (num_experts, 2 * intermediate_size), device=DEVICE,
            dtype=dtype) / 16
        w2_bias = torch.randn(
            (num_experts, hidden_size), device=DEVICE, dtype=dtype) / 16
    else:
        w13_bias = None
        w2_bias = None

    scores = torch.randn((input_len, num_experts),
                         device=DEVICE,
                         dtype=torch.float32)
    expert_scores, expert_indices = torch.topk(scores,
                                               k=topk,
                                               dim=-1,
                                               sorted=False)
    flat_expert_indices = expert_indices.view(-1)
    flat_expert_weights = expert_scores.view(-1, 1)

    ref_13_nk = ref_13.transpose(-1, -2).contiguous()
    ref_2_nk = ref_2.transpose(-1, -2).contiguous()
    ref_out = ref_fused_moe(a.clone(), ref_13_nk, w13_bias, ref_2_nk, w2_bias,
                            flat_expert_weights, flat_expert_indices, topk,
                            "silu", e)

    fused_moe_impl = XpuFusedMoe(
        w13=w13,
        w13_scales=w13_scales,
        w13_bias=w13_bias,
        w2=w2,
        w2_scales=w2_scales,
        w2_bias=w2_bias,
        n_experts_per_token=topk,
        activation="silu",
        num_experts=e,
    )
    assert fused_moe_impl.is_block_fp8
    assert not fused_moe_impl._block_fp8_promoted
    assert fused_moe_impl.gemm1_wei_scales is not None
    assert fused_moe_impl.w13.dtype == torch.float8_e4m3fn
    assert fused_moe_impl._use_ref is False

    output = torch.empty_like(ref_out)
    fused_moe_impl.apply(
        output=output,
        hidden_states=a,
        topk_weights=expert_scores,
        topk_ids=expert_indices,
    )
    torch.testing.assert_close(output, ref_out, rtol=5e-2, atol=5e-2)


FUSED_MOE_MNK_FACTORS = [
    (1, 1024, 1024),
    (4, 1024, 1024),
    (16, 1024, 1024),
    (8192, 1024, 1024),
]


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("ep_rank", EP_RANK)
@pytest.mark.parametrize("ep_size", EP_SIZE)
@pytest.mark.parametrize("dtype", [torch.float16], ids=format_tc)
@pytest.mark.parametrize("w_dtype", [torch.float8_e5m2, None], ids=format_tc)
@pytest.mark.parametrize("has_bias", [True, False])
def test_fused_moe_ep(m, n, k, e, topk, ep_rank, ep_size, dtype, w_dtype,
                      has_bias):
    seed_everything(7)

    input_len = m
    hidden_size = n
    intermediate_size = k
    num_experts = e

    a = torch.randn((input_len, hidden_size), device=DEVICE, dtype=dtype) / 16
    w13 = torch.randn((num_experts, 2 * intermediate_size, hidden_size),
                      device=DEVICE,
                      dtype=dtype) / 16
    w2 = torch.randn((num_experts, hidden_size, intermediate_size),
                     device=DEVICE,
                     dtype=dtype) / 16
    ref_a = a.clone()

    if has_bias:
        w13_bias = torch.randn(
            (num_experts, 2 * intermediate_size), device=DEVICE,
            dtype=dtype) / 16
        w2_bias = torch.randn(
            (num_experts, hidden_size), device=DEVICE, dtype=dtype) / 16
    else:
        w13_bias = None
        w2_bias = None
    # moe gate
    scores = torch.randn((input_len, num_experts),
                         device=DEVICE,
                         dtype=torch.float32)
    expert_scores, expert_indices = torch.topk(scores,
                                               k=topk,
                                               dim=-1,
                                               sorted=False)

    flat_expert_indices = expert_indices.view(-1)
    flat_expert_weights = expert_scores.view(-1, 1)

    if w_dtype is not None:
        w13_fp8 = torch.empty_like(w13, dtype=w_dtype)
        w2_fp8 = torch.empty_like(w2, dtype=w_dtype)

        # scale
        random_exponents = torch.randint(-3, 4, (num_experts, ), device=DEVICE)
        w13_scales = torch.pow(2.0, random_exponents.float())
        random_exponents = torch.randint(-3, 4, (num_experts, ), device=DEVICE)
        w2_scales = torch.pow(2.0, random_exponents.float())

        for i in range(num_experts):
            w13_fp8[i], _ = scaled_fp8_quant(w13[i],
                                             w13_scales[i].to(torch.float32),
                                             False,
                                             False,
                                             fp8_dtype=w_dtype)
            w2_fp8[i], _ = scaled_fp8_quant(w2[i],
                                            w2_scales[i].to(torch.float32),
                                            False,
                                            False,
                                            fp8_dtype=w_dtype)
        w13 = w13_fp8
        w2 = w2_fp8

        ref_w13 = torch.empty_like(w13_fp8, dtype=dtype)
        ref_w2 = torch.empty_like(w2_fp8, dtype=dtype)
        for i in range(num_experts):
            ref_w13[i] = w13_fp8[i].to(dtype) * w13_scales[i]
            ref_w2[i] = w2_fp8[i].to(dtype) * w2_scales[i]
    else:
        w13_scales = None
        w2_scales = None
        ref_w13 = w13
        ref_w2 = w2

    e //= ep_size

    ref_out = ref_fused_moe(ref_a, ref_w13, w13_bias, ref_w2, w2_bias,
                            flat_expert_weights, flat_expert_indices, topk,
                            "silu", e, ep_rank, ep_size)

    expert_start_id = e * ep_rank
    expert_end_id = expert_start_id + e

    w13.data = w13.transpose(-1, -2).contiguous()
    w2.data = w2.transpose(-1, -2).contiguous()

    fused_moe_impl = XpuFusedMoe(
                w13=w13[expert_start_id:expert_end_id],
                w13_scales=w13_scales[expert_start_id:expert_end_id]
                if w13_scales is not None else None,
                w13_bias=w13_bias[expert_start_id:expert_end_id]
                if w13_bias is not None else None,
                w2=w2[expert_start_id:expert_end_id],
                w2_scales=w2_scales[expert_start_id:expert_end_id]
                if w2_scales is not None else None,
                w2_bias=w2_bias[expert_start_id:expert_end_id]
                if w2_bias is not None else None,
                n_experts_per_token=topk,
                activation="silu",
                num_experts=e,
                ep_rank=ep_rank,
                ep_size=ep_size,
            )

    output = torch.empty_like(ref_out)
    fused_moe_impl.apply(
        output=output,
        hidden_states=a,
        topk_weights=expert_scores,
        topk_ids=expert_indices,
    )

    if dtype == torch.float16:
        rtol = 1e-2
        atol = 1e-2
    else:
        rtol = 2e-2
        atol = 2e-2
    torch.testing.assert_close(output, ref_out, rtol=rtol, atol=atol)


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("ep_rank", EP_RANK)
@pytest.mark.parametrize("ep_size", EP_SIZE)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("has_bias", [True, False])
def test_fused_moe_int4_ep(m, n, k, e, topk, ep_rank, ep_size, dtype,
                           has_bias):
    seed_everything(7)
    torch.xpu.empty_cache()
    gc.collect()

    input_len = m
    hidden_size = n
    intermediate_size = k
    num_experts = e
    group_size = 128

    a = torch.randn((input_len, hidden_size), device=DEVICE, dtype=dtype) / 16
    w13 = (torch.randint(
        0,
        0xff, [num_experts, 2 * intermediate_size, hidden_size // 2],
        device=DEVICE)).to(torch.uint8)
    w2 = (torch.randint(0,
                        0xff,
                        [num_experts, hidden_size, intermediate_size // 2],
                        device=DEVICE)).to(torch.uint8)
    ref_a = a.clone()

    # scale
    group_num_13 = hidden_size // group_size
    group_num_2 = intermediate_size // group_size
    assert hidden_size % group_size == 0, \
        "hidden_size must be divisible by group_size"
    assert intermediate_size % group_size == 0, \
        "intermediate_size must be divisible by group_size"
    random_exponents = torch.randint(
        -5,
        -4, (num_experts, 2 * intermediate_size, group_num_13),
        device=DEVICE)
    w13_scales = torch.pow(2.0, random_exponents.float()).to(dtype)
    random_exponents = torch.randint(-5,
                                     -4,
                                     (num_experts, hidden_size, group_num_2),
                                     device=DEVICE)
    w2_scales = torch.pow(2.0, random_exponents.float()).to(dtype)

    if has_bias:
        w13_bias = torch.randn(
            (num_experts, 2 * intermediate_size), device=DEVICE,
            dtype=dtype) / 16
        w2_bias = torch.randn(
            (num_experts, hidden_size), device=DEVICE, dtype=dtype) / 16
    else:
        w13_bias = None
        w2_bias = None
    # moe gate
    scores = torch.randn((input_len, num_experts),
                         device=DEVICE,
                         dtype=torch.float32)
    expert_scores, expert_indices = torch.topk(scores,
                                               k=topk,
                                               dim=-1,
                                               sorted=False)

    flat_expert_indices = expert_indices.view(-1)
    flat_expert_weights = expert_scores.view(-1, 1)

    ref_13 = torch.empty(num_experts,
                         2 * intermediate_size,
                         hidden_size,
                         dtype=dtype,
                         device=DEVICE)
    ref_2 = torch.empty(num_experts,
                        hidden_size,
                        intermediate_size,
                        dtype=dtype,
                        device=DEVICE)

    for i in range(num_experts):
        ref_13[i] = dequantize_uint4(w13[i], w13_scales[i], group_size)
        ref_2[i] = dequantize_uint4(w2[i], w2_scales[i], group_size)

    e //= ep_size

    ref_out = ref_fused_moe(ref_a, ref_13, w13_bias, ref_2, w2_bias,
                            flat_expert_weights, flat_expert_indices, topk,
                            "silu", e, ep_rank, ep_size)

    expert_start_id = e * ep_rank
    expert_end_id = expert_start_id + e

    fused_moe_impl = XpuFusedMoe(
                w13=w13[expert_start_id:expert_end_id],
                w13_scales=w13_scales[expert_start_id:expert_end_id]
                if w13_scales is not None else None,
                w13_bias=w13_bias[expert_start_id:expert_end_id]
                if w13_bias is not None else None,
                w2=w2[expert_start_id:expert_end_id],
                w2_scales=w2_scales[expert_start_id:expert_end_id]
                if w2_scales is not None else None,
                w2_bias=w2_bias[expert_start_id:expert_end_id]
                if w2_bias is not None else None,
                n_experts_per_token=topk,
                activation="silu",
                num_experts=e,
                ep_rank=ep_rank,
                ep_size=ep_size,
            )

    output = torch.empty_like(ref_out)
    fused_moe_impl.apply(
        output=output,
        hidden_states=a,
        topk_weights=expert_scores,
        topk_ids=expert_indices,
    )

    if dtype == torch.float16:
        rtol = 2e-2
        atol = 2e-2
    else:
        rtol = 2e-1
        atol = 2e-1
    torch.testing.assert_close(output, ref_out, rtol=rtol, atol=atol)


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("ep_rank", EP_RANK)
@pytest.mark.parametrize("ep_size", EP_SIZE)
@pytest.mark.parametrize("dtype", [torch.float16], ids=format_tc)
@pytest.mark.parametrize("has_bias", [True, False])
def test_fused_moe_mxfp4_ep(m, n, k, e, topk, ep_rank, ep_size, dtype,
                            has_bias):
    seed_everything(7)

    torch.xpu.empty_cache()
    gc.collect()

    input_len = m
    hidden_size = n
    intermediate_size = k
    num_experts = e
    group_size = 32

    a = torch.randn((input_len, hidden_size), device=DEVICE, dtype=dtype) / 16
    w13 = (torch.randint(
        0,
        0xff, [num_experts, 2 * intermediate_size, hidden_size // 2],
        device=DEVICE)).to(torch.uint8)
    w2 = (torch.randint(0,
                        0xff,
                        [num_experts, hidden_size, intermediate_size // 2],
                        device=DEVICE)).to(torch.uint8)
    ref_a = a.clone()

    # scale
    group_num_13 = hidden_size // group_size
    group_num_2 = intermediate_size // group_size
    w13_scales = torch.randint(
        0,
        0x6f, (num_experts, 2 * intermediate_size, group_num_13),
        dtype=torch.uint8,
        device=DEVICE)
    w2_scales = torch.randint(0,
                              0x6f, (num_experts, hidden_size, group_num_2),
                              dtype=torch.uint8,
                              device=DEVICE)

    if has_bias:
        w13_bias = torch.randn(
            (num_experts, 2 * intermediate_size), device=DEVICE,
            dtype=dtype) / 16
        w2_bias = torch.randn(
            (num_experts, hidden_size), device=DEVICE, dtype=dtype) / 16
    else:
        w13_bias = None
        w2_bias = None
    # moe gate
    scores = torch.randn((input_len, num_experts),
                         device=DEVICE,
                         dtype=torch.float32)
    expert_scores, expert_indices = torch.topk(scores,
                                               k=topk,
                                               dim=-1,
                                               sorted=False)

    flat_expert_indices = expert_indices.view(-1)
    flat_expert_weights = expert_scores.view(-1, 1)

    ref_13 = torch.empty(num_experts,
                         2 * intermediate_size,
                         hidden_size,
                         dtype=dtype,
                         device=DEVICE)
    ref_2 = torch.empty(num_experts,
                        hidden_size,
                        intermediate_size,
                        dtype=dtype,
                        device=DEVICE)

    for i in range(num_experts):
        ref_13[i] = dequantize_mxfp4(w13[i], w13_scales[i], group_size, dtype)
        ref_2[i] = dequantize_mxfp4(w2[i], w2_scales[i], group_size, dtype)

    e //= ep_size

    ref_out = ref_fused_moe(ref_a, ref_13, w13_bias, ref_2, w2_bias,
                            flat_expert_weights, flat_expert_indices, topk,
                            "silu", e, ep_rank, ep_size)

    expert_start_id = e * ep_rank
    expert_end_id = expert_start_id + e

    fused_moe_impl = XpuFusedMoe(
                w13=w13[expert_start_id:expert_end_id].view(torch.float4_e2m1fn_x2),
                w13_scales=w13_scales[expert_start_id:expert_end_id]
                if w13_scales is not None else None,
                w13_bias=w13_bias[expert_start_id:expert_end_id]
                if w13_bias is not None else None,
                w2=w2[expert_start_id:expert_end_id].view(torch.float4_e2m1fn_x2),
                w2_scales=w2_scales[expert_start_id:expert_end_id]
                if w2_scales is not None else None,
                w2_bias=w2_bias[expert_start_id:expert_end_id]
                if w2_bias is not None else None,
                n_experts_per_token=topk,
                activation="silu",
                num_experts=e,
                ep_rank=ep_rank,
                ep_size=ep_size,
            )

    output = torch.empty_like(ref_out)
    fused_moe_impl.apply(
        output=output,
        hidden_states=a,
        topk_weights=expert_scores,
        topk_ids=expert_indices,
    )

    if dtype == torch.float16:
        rtol = 1e-2
        atol = 1e-2
    else:
        rtol = 2e-2
        atol = 2e-2
    torch.testing.assert_close(output, ref_out, rtol=rtol, atol=atol)


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS) 
@pytest.mark.parametrize("e", [16])
@pytest.mark.parametrize("topk", [1])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("gemm1_clamp_limit", [5.0, 10.0])
def test_fused_moe_clamp_limit(m, n, k, e, topk, dtype, gemm1_clamp_limit):
    """Test that gemm1_clamp_limit correctly clamps gate/up after GEMM1."""
    seed_everything(7)

    input_len = m
    hidden_size = n
    intermediate_size = k
    num_experts = e

    # Use /2 scaling so GEMM1 std ≈ 0.25 * sqrt(k) ≈ 8 for k=1024,
    # ensuring values exceed clamp_limit=5.0 (~53%) and 10.0 (~21%).
    a = torch.randn((input_len, hidden_size), device=DEVICE, dtype=dtype) / 2
    w13 = torch.randn((num_experts, 2 * intermediate_size, hidden_size),
                      device=DEVICE,
                      dtype=dtype) / 2
    w2 = torch.randn((num_experts, hidden_size, intermediate_size),
                     device=DEVICE,
                     dtype=dtype) / 16
    ref_a = a.clone()

    # moe gate
    scores = torch.randn((input_len, num_experts),
                         device=DEVICE,
                         dtype=torch.float32)
    expert_scores, expert_indices = torch.topk(scores,
                                               k=topk,
                                               dim=-1,
                                               sorted=False)

    flat_expert_indices = expert_indices.view(-1)
    flat_expert_weights = expert_scores.view(-1, 1)

    ref_out = ref_fused_moe(ref_a, w13.clone(), None, w2.clone(), None,
                            flat_expert_weights, flat_expert_indices, topk,
                            "silu", e,
                            gemm1_clamp_limit=gemm1_clamp_limit)

    w13.data = w13.transpose(-1, -2).contiguous()
    w2.data = w2.transpose(-1, -2).contiguous()

    fused_moe_impl = XpuFusedMoe(
                w13=w13,
                w13_scales=None,
                w13_bias=None,
                w2=w2,
                w2_scales=None,
                w2_bias=None,
                n_experts_per_token=topk,
                activation="silu",
                num_experts=e,
                gemm1_clamp_limit=gemm1_clamp_limit,
            )

    output = torch.empty_like(ref_out)
    fused_moe_impl.apply(
        output=output,
        hidden_states=a,
        topk_weights=expert_scores,
        topk_ids=expert_indices,
    )

    # Use relative tolerance proportional to output magnitude since bf16
    # accumulation error scales with intermediate values after clamp.
    output_scale = ref_out.abs().max().item()
    rtol = 2e-2
    atol = max(output_scale * 0.02, 0.5)
    torch.testing.assert_close(output, ref_out, rtol=rtol, atol=atol)

    # Verify clamp actually took effect: without clamp the result should differ.
    ref_out_no_clamp = ref_fused_moe(ref_a, w13.transpose(-1, -2).clone(),
                                     None,
                                     w2.transpose(-1, -2).clone(), None,
                                     flat_expert_weights,
                                     flat_expert_indices, topk, "silu", e,
                                     gemm1_clamp_limit=None)
    assert not torch.allclose(output, ref_out_no_clamp, rtol=rtol,
                              atol=atol), \
        f"clamp_limit={gemm1_clamp_limit} should produce different results"
