# SPDX-License-Identifier: Apache-2.0
import random

import pytest
import torch

from tests.ops.fp8_quant_op import scaled_fp8_quant
from tests.utils import seed_everything
from vllm_xpu_kernels.fused_moe_interface import (cutlass_grouped_gemm,
                                                  cutlass_xe_grouped_gemm,
                                                  xpu_fused_moe)

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


def random_partition(size_a: int, target: int):
    cuts = sorted(random.sample(range(target + size_a - 1), size_a - 1))
    cuts = [-1] + cuts + [target + size_a - 1]
    result = [cuts[i + 1] - cuts[i] - 1 for i in range(size_a)]
    return result


MINI_PYTEST_PARAMS = {
    "default": {
        "m,n,k": [(1, 256, 128)],
        "e": [2],
        "topk": [1],
        "dtype": [torch.bfloat16],
        "has_bias": [True]
    }
}


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_bias", [True, False])
def test_grouped_gemm(m, n, k, e, topk, dtype, has_bias):
    seed_everything(7)
    num_experts = e
    token_per_group = random_partition(e, m * topk)
    assert (len(token_per_group) == e)
    # input
    input_A = torch.randn((sum(token_per_group), k),
                          dtype=dtype,
                          device=DEVICE).contiguous()
    ref_A = input_A
    # weight
    input_B = torch.randn((num_experts, n, k), dtype=dtype, device=DEVICE)
    input_B = input_B.transpose(-1, -2).contiguous().transpose(-1, -2)
    if has_bias:
        bias = torch.randn((num_experts, n), dtype=dtype, device=DEVICE)
    else:
        bias = None

    # output offset
    output = torch.empty((sum(token_per_group), n), dtype=dtype, device=DEVICE)
    cutlass_grouped_gemm(input_A, input_B, bias, output, token_per_group, n, k,
                         num_experts)
    # ref gg
    ref = []
    pre_token_sum = 0
    for i in range(num_experts):
        cur_token_num = token_per_group[i]
        if cur_token_num == 0:
            continue
        input = ref_A[pre_token_sum:pre_token_sum + cur_token_num, :]
        weight = input_B[i, :, :]
        expert_output = input @ weight.T
        if has_bias:
            expert_output += bias[i]
        ref.append(expert_output)
        pre_token_sum += cur_token_num
    ref = torch.cat(ref, dim=0)

    torch.testing.assert_close(output, ref, rtol=2e-2, atol=1e-2)


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_bias", [True, False])
def test_xe_grouped_gemm(m, n, k, e, topk, dtype, has_bias):
    seed_everything(7)
    num_experts = e
    token_per_group = random_partition(e, m * topk)
    assert (len(token_per_group) == e)
    # input
    input_A = torch.randn((sum(token_per_group), k),
                          dtype=dtype,
                          device=DEVICE).contiguous()
    ref_A = input_A
    # weight
    input_B = torch.randn((num_experts, k, n), dtype=dtype, device=DEVICE)
    if has_bias:
        bias = torch.randn((num_experts, n), dtype=dtype, device=DEVICE)
    else:
        bias = None

    # output offset
    num_rows_per_expert = torch.tensor(token_per_group,
                                       dtype=torch.int32,
                                       device=input_A.device)
    output = torch.empty((sum(token_per_group), n), dtype=dtype, device=DEVICE)
    cutlass_xe_grouped_gemm(input_A, input_B, None, bias, output,
                            num_rows_per_expert, n, k, num_experts, False)
    # ref gg
    ref = []
    pre_token_sum = 0
    for i in range(num_experts):
        cur_token_num = token_per_group[i]
        if cur_token_num == 0:
            continue
        input = ref_A[pre_token_sum:pre_token_sum + cur_token_num, :].to(
            torch.float32)
        weight = input_B[i, :, :].to(torch.float32)
        expert_output_fp32 = input @ weight
        if has_bias:
            expert_output_fp32 += bias[i]
        ref.append(expert_output_fp32.to(dtype))
        pre_token_sum += cur_token_num
    ref = torch.cat(ref, dim=0)

    torch.testing.assert_close(output, ref, rtol=2e-2, atol=1e-2)


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e5m2, torch.float8_e4m3fn])
@pytest.mark.parametrize("has_bias", [False, True])
def test_xe_grouped_gemm_fp8(m, n, k, e, topk, dtype, fp8_dtype, has_bias):
    seed_everything(7)
    num_experts = e
    token_per_group = random_partition(e, m * topk)
    assert (len(token_per_group) == e)
    # input
    input_A = torch.randn((sum(token_per_group), k),
                          dtype=dtype,
                          device=DEVICE).contiguous()
    ref_A = input_A
    # weight
    input_B = torch.randn((num_experts, k, n), dtype=dtype, device=DEVICE)
    # scale
    random_exponents = torch.randint(-3, 4, (num_experts, ), device=DEVICE)
    scale_B = torch.pow(2.0, random_exponents.float()).to(dtype)
    if has_bias:
        bias = torch.randn((num_experts, n), dtype=dtype, device=DEVICE) * 100
    else:
        bias = None

    # quantize weight
    input_B_fp8 = torch.empty_like(input_B, dtype=fp8_dtype)
    for i in range(num_experts):
        input_B_fp8[i], _ = scaled_fp8_quant(input_B[i],
                                             scale_B[i].to(torch.float32),
                                             False,
                                             False,
                                             fp8_dtype=fp8_dtype)
    input_B_dequatize = torch.empty_like(input_B, dtype=dtype)
    for i in range(num_experts):
        input_B_dequatize[i] = input_B_fp8[i].to(dtype) * scale_B[i]

    # output offset
    num_rows_per_expert = torch.tensor(token_per_group,
                                       dtype=torch.int32,
                                       device=input_A.device)
    output = torch.empty((sum(token_per_group), n), dtype=dtype, device=DEVICE)
    cutlass_xe_grouped_gemm(input_A, input_B_fp8, scale_B, bias, output,
                            num_rows_per_expert, n, k, num_experts, False)
    # ref gg
    ref = []
    pre_token_sum = 0
    for i in range(num_experts):
        cur_token_num = token_per_group[i]
        if cur_token_num == 0:
            continue
        # mma uses fp32 as calculate dtype
        # so here use fp32 to avoid accuracy error
        input = ref_A[pre_token_sum:pre_token_sum + cur_token_num, :].to(
            torch.float32)
        weight = input_B_dequatize[i, :, :].to(torch.float32)
        expert_output_fp32 = input @ weight
        if has_bias:
            expert_output_fp32 += bias[i]
        ref.append(expert_output_fp32.to(dtype))
        pre_token_sum += cur_token_num
    ref = torch.cat(ref, dim=0)

    try:
        torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
        print("a and b close enough")
    except AssertionError as e:
        print("a and b diffs")
        raise (e)

def dequantize_int4(qweight, scales, group_size):
    import numpy as np
    k = qweight.shape[1] * 2
    n = qweight.shape[0]
    unpack_idx = np.array([0, 1])
    data = qweight[:, [i // 2 for i in range(k)]]
    shift = (
        torch.tensor(
            unpack_idx[[i % 2 for i in range(k)]], dtype=torch.int32, device="xpu"
        )[None, :].expand([n, -1])
        * 4
    )
    dst_data = (data >> shift) & 0xF
    expand_scales = scales[:, [i // group_size for i in range(k)]]
    weight_16 = (dst_data - 8) * expand_scales

    return weight_16.to(scales.dtype)


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_bias", [False, True])
def test_xe_grouped_gemm_int4(m, n, k, e, topk, dtype, has_bias):
    seed_everything(7)
    num_experts = e
    group_size = 128
    group_num = k // group_size
    token_per_group = random_partition(e, m * topk)
    assert (len(token_per_group) == e)
    # input
    input_A = torch.randn((sum(token_per_group), k),
                          dtype=dtype,
                          device=DEVICE).contiguous()
    ref_A = input_A
    # weight
    input_B_int4 = (
        torch.randint(0, 0xff, [num_experts, n, k // 2], device=DEVICE)
    ).to(torch.uint8)
    # scale
    random_exponents = torch.randint(-3, 4, (num_experts, n, group_num), device=DEVICE)
    scale_B = torch.pow(2.0, random_exponents.float()).to(dtype)

    if has_bias:
        bias = torch.randn((num_experts, n), dtype=dtype, device=DEVICE) * 100
    else:
        bias = None


    input_B_16 = torch.empty(
        num_experts, n, k, dtype=dtype, device=DEVICE
    )
    for i in range(num_experts):
        input_B_16[i] = dequantize_int4(
            input_B_int4[i], scale_B[i], group_size
        )

    # output offset
    num_rows_per_expert = torch.tensor(token_per_group,
                                       dtype=torch.int32,
                                       device=input_A.device)
    output = torch.empty((sum(token_per_group), n), dtype=dtype, device=DEVICE)
    cutlass_xe_grouped_gemm(input_A, input_B_int4, scale_B, bias, output,
                            num_rows_per_expert, n, k, num_experts, True)
    # ref gg
    ref = []
    pre_token_sum = 0
    for i in range(num_experts):
        cur_token_num = token_per_group[i]
        if cur_token_num == 0:
            continue
        # mma uses fp32 as calculate dtype
        # so here use fp32 to avoid accuracy error
        input = ref_A[pre_token_sum:pre_token_sum + cur_token_num, :].to(
            torch.float32)
        weight = input_B_16[i, :, :].to(torch.float32)
        expert_output_fp32 = input @ weight.T
        if has_bias:
            expert_output_fp32 += bias[i]
        ref.append(expert_output_fp32.to(dtype))
        pre_token_sum += cur_token_num
    ref = torch.cat(ref, dim=0)

    try:
        torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
        print("a and b close enough")
    except AssertionError as e:
        print("a and b diffs")
        raise (e)

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
        gemm1 = expert_tokens @ w1.T
        if w13_bias is not None:
            gemm1 += w1_bias
        gate = act_fn(gemm1)
        up = expert_tokens @ w3.T
        if w13_bias is not None:
            up += w3_bias
        expert_out = (gate * up) @ w2[expert_id, :, :].T
        if w2_bias is not None:
            expert_out += w2_bias[expert_id, :]
        expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
        expert_cache.scatter_reduce_(0,
                                     exp_token_idxs.view(-1, 1).repeat(
                                         1, x.shape[-1]),
                                     expert_out,
                                     reduce='sum')

    return expert_cache


def check_fused_moe(
    m: int,  # num of tokens
    n: int,  # intermediate_size
    k: int,  # hidden_size
    e: int,
    topk: int,
    has_bias: bool,
    dtype: torch.dtype,
):
    seed_everything(7)
    # Setup test data
    a = torch.randn((m, k), device=DEVICE, dtype=dtype) / 10
    w13 = torch.randn((e, 2 * n, k), device=DEVICE, dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device=DEVICE, dtype=dtype) / 10
    ref_a = a.clone()
    if has_bias:
        w13_bias = torch.randn(
            (e, 2 * n), device=DEVICE, dtype=torch.float) / 10
        w2_bias = torch.randn((e, k), device=DEVICE, dtype=torch.float) / 10
    else:
        w13_bias = None
        w2_bias = None
    # moe gate
    scores = torch.randn((m, e), device=DEVICE, dtype=dtype)
    expert_scores, expert_indices = torch.topk(scores,
                                               k=topk,
                                               dim=-1,
                                               sorted=False)

    flat_expert_indices = expert_indices.view(-1)
    flat_expert_weights = expert_scores.view(-1, 1)

    output = xpu_fused_moe(hidden_states=a,
                           w13=w13,
                           w13_bias=w13_bias,
                           w2=w2,
                           w2_bias=w2_bias,
                           topk_weights=expert_scores,
                           topk_ids=expert_indices,
                           n_experts_per_token=topk,
                           activation="silu",
                           num_experts=e)

    ref_out = ref_fused_moe(ref_a, w13, w13_bias, w2, w2_bias,
                            flat_expert_weights, flat_expert_indices, topk,
                            "silu", e)

    print("ref result", ref_out, ref_out.shape)
    print("kernel result", output, output.shape)
    try:
        torch.testing.assert_close(output, ref_out, rtol=1e-2, atol=1e-2)
        print("a and b close enough")
    except AssertionError as e:
        print("a and b diffs")
        print(e)
