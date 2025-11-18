# SPDX-License-Identifier: Apache-2.0
from enum import Enum

import pytest
import torch

from tests.register_ops import int4_gemm_w4a16
from vllm_xpu_kernels.quantization._quantize_convert import (GPTQUtils,
                                                             dequantize)

BATCHES = [1]
MNK_FACTORS = [
    (8, 4096, 4096),
    (1, 4096, 11008),
    (32, 4096, 4096),
]

MINI_MNK_FACTORS = [
    (1, 16, 32),
    (4, 32, 64),
    (8, 32, 32),
]

MINI_PYTEST_PARAMS = {
    "default": {
        "mnk_factors": MINI_MNK_FACTORS,
    },
}


def rand_int4(size, dtype=torch.int32, device="xpu"):
    rand = torch.randint(-128, 128, [size // 2], device=device).to(torch.int8)
    return rand.view(dtype=dtype)


class QuantMode(Enum):
    SYM = 1
    ASYM = 2


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
@pytest.mark.parametrize("qmode", [QuantMode.ASYM, QuantMode.SYM])
def test_int4_gemm(dtype, act_order, mnk_factors, qmode: QuantMode):
    seed = 1234
    torch.manual_seed(seed)

    m, n, k = mnk_factors
    input = torch.rand([m, k], device="xpu", dtype=dtype)
    input_torch = input.cpu()
    weight = rand_int4(k * n, torch.int32, "xpu").reshape(k // 8, n)

    group_size = min(128, k)
    group_num = int(k / group_size)

    scales = torch.rand([group_num, n], device="xpu", dtype=dtype)
    if qmode == QuantMode.SYM:
        zero_points = None
    elif qmode == QuantMode.ASYM:
        zero_points = rand_int4(group_num * n, torch.int32,
                                "xpu").reshape(group_num, n // 8)
    gptq_utils = GPTQUtils(bits=4, blocksize=group_size)
    if act_order:
        g_idx = torch.randperm(k, dtype=torch.int32) // group_size
        shuffled_weight, g_idx4kernel = gptq_utils.shuffle(weight, g_idx)
    else:
        g_idx = None
        g_idx4kernel = None
        shuffled_weight = weight

    # check fp16 gemm
    weight_fp = dequantize(weight, scales, zero_points, group_size,
                           g_idx).cpu()
    out_torch = torch.matmul(input_torch, weight_fp)

    # onednn int4 gemm
    weight_ba = shuffled_weight.transpose(0, 1).contiguous().transpose(0, 1)
    if qmode == QuantMode.SYM:
        zero_points = torch.Tensor([8]).to(torch.int8).to("xpu")
    output_int4 = int4_gemm_w4a16(
        input,
        weight_ba,
        torch.Tensor(),
        scales,
        zero_points,
        group_size,
        g_idx4kernel,
    )

    torch.testing.assert_close(
        output_int4.cpu().float(),
        out_torch.cpu().float(),
        atol=1e-2,
        rtol=1e-2,
    )
