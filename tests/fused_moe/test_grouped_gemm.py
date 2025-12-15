# SPDX-License-Identifier: Apache-2.0
import random

import pytest
import torch

from tests.ops.fp8_quant_op import scaled_fp8_quant
from tests.utils import seed_everything
from vllm_xpu_kernels.fused_moe_interface import (cutlass_grouped_gemm,
                                                  cutlass_grouped_gemm_XE2)

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
    input_B = input_B.transpose(-1, -2).contiguous()
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
        expert_output = input @ weight
        if has_bias:
            expert_output += bias[i]
        ref.append(expert_output)
        pre_token_sum += cur_token_num
    ref = torch.cat(ref, dim=0)

    torch.testing.assert_close(output, ref, rtol=2e-2, atol=1e-2)


def init_rows_for_experts(tokens, topk, num_rows_per_expert):
    if num_rows_per_expert.shape[0] == 1:
        num_rows_per_expert[0] = tokens * topk
        return
    n_experts = num_rows_per_expert.numel()
    rand = torch.rand(tokens, n_experts, device=num_rows_per_expert.device)
    topk_idx = torch.topk(rand, topk, dim=1).indices  # [tokens, topk]
    flat_idx = topk_idx.flatten()
    num_rows_per_expert += torch.bincount(flat_idx, minlength=n_experts)


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_bias", [True, False])
def test_xe_grouped_gemm(m, n, k, e, topk, dtype, has_bias):
    seed_everything(7)
    num_experts = e
    total_m = m * topk
    # input
    input_A = torch.randn((total_m, k), dtype=dtype,
                          device=DEVICE).contiguous()
    ref_A = input_A
    # weight
    input_B = torch.randn((num_experts, k, n), dtype=dtype, device=DEVICE)
    if has_bias:
        bias = torch.randn((num_experts, n), dtype=dtype, device=DEVICE)
    else:
        bias = None

    # output offset
    num_rows_per_expert = torch.zeros(num_experts,
                                      device=DEVICE,
                                      dtype=torch.int32)
    init_rows_for_experts(m, topk, num_rows_per_expert)
    output = torch.empty((total_m, n), dtype=dtype, device=DEVICE)

    cutlass_grouped_gemm_XE2(input_A, input_B, None, bias, output,
                             num_rows_per_expert, n, k, num_experts, False,
                             False)

    # ref gg
    ref = []
    pre_token_sum = 0
    for i in range(num_experts):
        cur_token_num = num_rows_per_expert[i]
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
    total_m = m * topk
    # input
    input_A = torch.randn((total_m, k), dtype=dtype,
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
    num_rows_per_expert = torch.zeros(num_experts,
                                      device=DEVICE,
                                      dtype=torch.int32)
    init_rows_for_experts(m, topk, num_rows_per_expert)
    output = torch.empty((total_m, n), dtype=dtype, device=DEVICE)

    cutlass_grouped_gemm_XE2(input_A, input_B_fp8, scale_B, bias, output,
                             num_rows_per_expert, n, k, num_experts, False,
                             False)
    # ref gg
    ref = []
    pre_token_sum = 0
    for i in range(num_experts):
        cur_token_num = num_rows_per_expert[i]
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

    torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)


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


def implement_zp(qweight, zp=None):
    assert qweight.dtype == torch.uint8, "Input tensor must be uint8"

    high_u4 = (qweight >> 4) & 0x0F
    low_u4 = qweight & 0x0F

    high_s8 = high_u4.to(torch.int8)
    low_s8 = low_u4.to(torch.int8)

    high_s8 = high_s8 - 8
    low_s8 = low_s8 - 8

    def pack_compact(a, b):

        def process_number(x):
            sign = (x < 0).to(torch.uint8)
            abs_low3 = (x.view(torch.uint8) & 0x7).to(torch.uint8)
            return (sign << 3) | abs_low3

        packed_a = process_number(a)
        packed_b = process_number(b)

        return (packed_a << 4) | packed_b

    result = pack_compact(high_s8, low_s8)

    return result


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
    total_m = m * topk
    # input
    input_A = torch.randn((total_m, k), dtype=dtype,
                          device=DEVICE).contiguous()
    ref_A = input_A
    # weight
    input_B_uint4 = (torch.randint(0,
                                   0xff, [num_experts, n, k // 2],
                                   device=DEVICE)).to(torch.uint8)
    # scale
    random_exponents = torch.randint(-3,
                                     4, (num_experts, n, group_num),
                                     device=DEVICE)
    scale_B = torch.pow(2.0, random_exponents.float()).to(dtype)

    if has_bias:
        bias = torch.randn((num_experts, n), dtype=dtype, device=DEVICE) * 100
    else:
        bias = None

    input_B_16 = torch.empty(num_experts, n, k, dtype=dtype, device=DEVICE)
    input_B_int4 = torch.empty_like(input_B_uint4)
    for i in range(num_experts):
        # default zp=8
        input_B_16[i] = dequantize_uint4(input_B_uint4[i], scale_B[i],
                                         group_size)
        input_B_int4[i] = implement_zp(input_B_uint4[i], None)

    # output offset
    num_rows_per_expert = torch.zeros(num_experts,
                                      device=DEVICE,
                                      dtype=torch.int32)
    init_rows_for_experts(m, topk, num_rows_per_expert)

    output = torch.empty((total_m, n), dtype=dtype, device=DEVICE)
    cutlass_grouped_gemm_XE2(input_A, input_B_int4, scale_B, bias, output,
                             num_rows_per_expert, n, k, num_experts, True,
                             False)
    # ref gg
    ref = []
    pre_token_sum = 0
    for i in range(num_experts):
        cur_token_num = num_rows_per_expert[i]
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

    torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)


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
    # weight_16 = dst_data

    return weight_16.to(dtype)


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_bias", [False, True])
def test_xe_grouped_gemm_mxfp4(m, n, k, e, topk, dtype, has_bias):
    seed_everything(7)
    num_experts = e
    group_size = 32
    group_num = k // group_size
    total_m = m * topk
    # input
    input_A = torch.randn((total_m, k), dtype=dtype,
                          device=DEVICE).contiguous()
    ref_A = input_A
    # weight
    input_B_int4 = (torch.randint(0,
                                  0xff, [num_experts, n, k // 2],
                                  device=DEVICE)).to(torch.uint8)
    # scale
    scale_B = torch.randint(0,
                            0x7f, (num_experts, n, group_num),
                            dtype=torch.uint8,
                            device="xpu")

    if has_bias:
        bias = torch.randn((num_experts, n), dtype=dtype, device=DEVICE) * 100
    else:
        bias = None

    input_B_16 = torch.empty(num_experts, n, k, dtype=dtype, device=DEVICE)
    for i in range(num_experts):
        input_B_16[i] = dequantize_mxfp4(input_B_int4[i], scale_B[i],
                                         group_size, dtype)

    # output offset
    num_rows_per_expert = torch.zeros(num_experts,
                                      device=DEVICE,
                                      dtype=torch.int32)
    init_rows_for_experts(m, topk, num_rows_per_expert)

    output = torch.empty((total_m, n), dtype=dtype, device=DEVICE)
    cutlass_grouped_gemm_XE2(input_A, input_B_int4, scale_B, bias, output,
                             num_rows_per_expert, n, k, num_experts, False,
                             True)
    # ref gg
    ref = []
    pre_token_sum = 0
    for i in range(num_experts):
        cur_token_num = num_rows_per_expert[i]
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

    torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
