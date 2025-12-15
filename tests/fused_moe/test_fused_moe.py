# SPDX-License-Identifier: Apache-2.0
import gc

import pytest
import torch

from tests.ops.fp8_quant_op import scaled_fp8_quant
from tests.utils import seed_everything
from vllm_xpu_kernels.fused_moe_interface import xpu_fused_moe

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

MINI_PYTEST_PARAMS = {
    "default": {
        "m,n,k": [(1, 256, 128)],
        "e": [2],
        "topk": [1],
        "dtype": [torch.bfloat16],
        "has_bias": [True]
    }
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


def ref_fused_moe(x, w13, w13_bias, w2, w2_bias, flat_expert_weights,
                  flat_expert_indices, num_per_tok, activation, num_experts):
    expert_cache = torch.zeros_like(x)
    idxs = flat_expert_indices.argsort()
    counts = flat_expert_indices.bincount().cpu().numpy()
    tokens_per_expert = counts.cumsum()
    token_idxs = idxs // num_per_tok
    for expert_id, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        if start_idx == end_idx:
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
        gate = act_fn(gemm1)
        up = (expert_tokens.to(torch.float32) @ w3.T.to(torch.float32))
        if w13_bias is not None:
            up += w3_bias.to(torch.float32)
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
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("w_dtype",
                         [torch.float8_e5m2, torch.float8_e4m3fn, None])
@pytest.mark.parametrize("has_bias", [True, False])
def test_fused_moe(m, n, k, e, topk, dtype, w_dtype, has_bias):
    seed_everything(7)

    input_len = m
    hidden_size = k
    imidiate_size = n
    num_experts = e

    a = torch.randn((input_len, hidden_size), device=DEVICE, dtype=dtype) / 16
    w13 = torch.randn((num_experts, 2 * imidiate_size, hidden_size),
                      device=DEVICE,
                      dtype=dtype) / 16
    w2 = torch.randn(
        (num_experts, hidden_size, imidiate_size), device=DEVICE,
        dtype=dtype) / 16
    ref_a = a.clone()

    if has_bias:
        w13_bias = torch.randn(
            (num_experts, 2 * imidiate_size), device=DEVICE, dtype=dtype) / 16
        w2_bias = torch.randn(
            (num_experts, hidden_size), device=DEVICE, dtype=dtype) / 16
    else:
        w13_bias = None
        w2_bias = None
    # moe gate
    scores = torch.randn((input_len, num_experts), device=DEVICE, dtype=dtype)
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
        w13_scales = torch.pow(2.0, random_exponents.float()).to(dtype)
        random_exponents = torch.randint(-3, 4, (num_experts, ), device=DEVICE)
        w2_scales = torch.pow(2.0, random_exponents.float()).to(dtype)

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

    output = xpu_fused_moe(hidden_states=a,
                           w13=w13,
                           w13_scales=w13_scales,
                           w13_bias=w13_bias,
                           w2=w2,
                           w2_scales=w2_scales,
                           w2_bias=w2_bias,
                           topk_weights=expert_scores,
                           topk_ids=expert_indices,
                           n_experts_per_token=topk,
                           activation="silu",
                           num_experts=e,
                           is_fp8=(w_dtype is not None))

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
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("has_bias", [True, False])
def test_fused_moe_int4(m, n, k, e, topk, dtype, has_bias):
    seed_everything(7)
    torch.xpu.empty_cache()
    gc.collect()

    input_len = m
    hidden_size = n
    imidiate_size = k
    num_experts = e
    group_size = 128

    a = torch.randn((input_len, hidden_size), device=DEVICE, dtype=dtype) / 16
    w13 = (torch.randint(0,
                         0xff,
                         [num_experts, 2 * imidiate_size, hidden_size // 2],
                         device=DEVICE)).to(torch.uint8)
    w2 = (torch.randint(0,
                        0xff, [num_experts, hidden_size, imidiate_size // 2],
                        device=DEVICE)).to(torch.uint8)
    ref_a = a.clone()

    # scale
    group_num_13 = hidden_size // group_size
    group_num_2 = imidiate_size // group_size
    random_exponents = torch.randint(
        -5, -4, (num_experts, 2 * imidiate_size, group_num_13), device=DEVICE)
    w13_scales = torch.pow(2.0, random_exponents.float()).to(dtype)
    random_exponents = torch.randint(-5,
                                     -4,
                                     (num_experts, hidden_size, group_num_2),
                                     device=DEVICE)
    w2_scales = torch.pow(2.0, random_exponents.float()).to(dtype)

    if has_bias:
        w13_bias = torch.randn(
            (num_experts, 2 * imidiate_size), device=DEVICE, dtype=dtype) / 16
        w2_bias = torch.randn(
            (num_experts, hidden_size), device=DEVICE, dtype=dtype) / 16
    else:
        w13_bias = None
        w2_bias = None
    # moe gate
    scores = torch.randn((input_len, num_experts), device=DEVICE, dtype=dtype)
    expert_scores, expert_indices = torch.topk(scores,
                                               k=topk,
                                               dim=-1,
                                               sorted=False)

    flat_expert_indices = expert_indices.view(-1)
    flat_expert_weights = expert_scores.view(-1, 1)

    ref_13 = torch.empty(num_experts,
                         2 * imidiate_size,
                         hidden_size,
                         dtype=dtype,
                         device=DEVICE)
    ref_2 = torch.empty(num_experts,
                        hidden_size,
                        imidiate_size,
                        dtype=dtype,
                        device=DEVICE)

    for i in range(num_experts):
        ref_13[i] = dequantize_uint4(w13[i], w13_scales[i], group_size)
        ref_2[i] = dequantize_uint4(w2[i], w2_scales[i], group_size)

    ref_out = ref_fused_moe(ref_a, ref_13, w13_bias, ref_2, w2_bias,
                            flat_expert_weights, flat_expert_indices, topk,
                            "silu", e)

    output = xpu_fused_moe(hidden_states=a,
                           w13=w13,
                           w13_scales=w13_scales,
                           w13_bias=w13_bias,
                           w2=w2,
                           w2_scales=w2_scales,
                           w2_bias=w2_bias,
                           topk_weights=expert_scores,
                           topk_ids=expert_indices,
                           n_experts_per_token=topk,
                           activation="silu",
                           num_experts=e,
                           is_int4=True)

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
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_bias", [True, False])
def test_fused_moe_mxfp4(m, n, k, e, topk, dtype, has_bias):
    seed_everything(7)

    torch.xpu.empty_cache()
    gc.collect()

    input_len = m
    hidden_size = n
    imidiate_size = k
    num_experts = e
    group_size = 32

    a = torch.randn((input_len, hidden_size), device=DEVICE, dtype=dtype) / 16
    w13 = (torch.randint(0,
                         0xff,
                         [num_experts, 2 * imidiate_size, hidden_size // 2],
                         device=DEVICE)).to(torch.uint8)
    w2 = (torch.randint(0,
                        0xff, [num_experts, hidden_size, imidiate_size // 2],
                        device=DEVICE)).to(torch.uint8)
    ref_a = a.clone()

    # scale
    group_num_13 = hidden_size // group_size
    group_num_2 = imidiate_size // group_size
    w13_scales = torch.randint(0,
                               0x6f,
                               (num_experts, 2 * imidiate_size, group_num_13),
                               dtype=torch.uint8,
                               device=DEVICE)
    w2_scales = torch.randint(0,
                              0x6f, (num_experts, hidden_size, group_num_2),
                              dtype=torch.uint8,
                              device=DEVICE)

    if has_bias:
        w13_bias = torch.randn(
            (num_experts, 2 * imidiate_size), device=DEVICE, dtype=dtype) / 16
        w2_bias = torch.randn(
            (num_experts, hidden_size), device=DEVICE, dtype=dtype) / 16
    else:
        w13_bias = None
        w2_bias = None
    # moe gate
    scores = torch.randn((input_len, num_experts), device=DEVICE, dtype=dtype)
    expert_scores, expert_indices = torch.topk(scores,
                                               k=topk,
                                               dim=-1,
                                               sorted=False)

    flat_expert_indices = expert_indices.view(-1)
    flat_expert_weights = expert_scores.view(-1, 1)

    ref_13 = torch.empty(num_experts,
                         2 * imidiate_size,
                         hidden_size,
                         dtype=dtype,
                         device=DEVICE)
    ref_2 = torch.empty(num_experts,
                        hidden_size,
                        imidiate_size,
                        dtype=dtype,
                        device=DEVICE)

    for i in range(num_experts):
        ref_13[i] = dequantize_mxfp4(w13[i], w13_scales[i], group_size, dtype)
        ref_2[i] = dequantize_mxfp4(w2[i], w2_scales[i], group_size, dtype)

    ref_out = ref_fused_moe(ref_a, ref_13, w13_bias, ref_2, w2_bias,
                            flat_expert_weights, flat_expert_indices, topk,
                            "silu", e)

    output = xpu_fused_moe(hidden_states=a,
                           w13=w13,
                           w13_scales=w13_scales,
                           w13_bias=w13_bias,
                           w2=w2,
                           w2_scales=w2_scales,
                           w2_bias=w2_bias,
                           topk_weights=expert_scores,
                           topk_ids=expert_indices,
                           n_experts_per_token=topk,
                           activation="silu",
                           num_experts=e,
                           is_mxfp4=True)

    if dtype == torch.float16:
        rtol = 1e-2
        atol = 1e-2
    else:
        rtol = 2e-2
        atol = 2e-2
    torch.testing.assert_close(output, ref_out, rtol=rtol, atol=atol)
