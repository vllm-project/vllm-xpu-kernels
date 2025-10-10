# SPDX-License-Identifier: Apache-2.0
from enum import Enum

import pytest
import torch

from tests.quantization._quantize_convert import GPTQShuffle
from tests.register_ops import int4_gemm_w4a16

BATCHES = [1]
MNK_FACTORS = [
    (8, 4096, 4096),
    (1, 4096, 11008),
    (32, 4096, 4096),
]


def rand_int4(size, dtype=torch.int32, device="xpu"):
    rand = torch.randint(-128, 128, [size // 2], device=device).to(torch.int8)
    return rand.view(dtype=dtype)


def unpack_weight(qweight, scales, qzeros, q_config):
    bits = q_config["bits"]
    s32_bits = 32

    assert bits == 4
    # Int32 can store 8 * 4bits data. This is the offset for each data.
    wf = (torch.tensor(list(range(0, s32_bits, bits)),
                       dtype=torch.int32).unsqueeze(0).to("xpu"))
    zeros = qzeros
    if qzeros is not None:
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits),
            wf.unsqueeze(0)).to(torch.int16 if bits == 8 else torch.int8)
        torch.bitwise_and(zeros, (2**bits) - 1, out=zeros)

        zeros = zeros.reshape(scales.shape)

    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1),
        wf.unsqueeze(-1)).to(torch.int16 if bits == 8 else torch.int8)
    torch.bitwise_and(weight, (2**bits) - 1, out=weight)

    return weight, scales, zeros


def dequantize(qweight, scales, qzeros, group_size, g_idx=None):
    q_config = {"group_size": group_size, "bits": 4}
    weight, gptq_scales, gptq_zeros = unpack_weight(qweight, scales, qzeros,
                                                    q_config)
    if len(weight.shape) > 2:
        weight = weight.reshape(-1, weight.shape[-1])
    infeatures = weight.shape[0]
    if g_idx is None:
        g_idx = torch.tensor(
            [i // q_config["group_size"] for i in range(infeatures)],
            dtype=torch.int32,
        )
    if gptq_zeros is None:
        return (weight - 8) * gptq_scales[g_idx]
    else:
        return (weight - gptq_zeros[g_idx]) * gptq_scales[g_idx]


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
    if act_order:
        g_idx = torch.randperm(k, dtype=torch.int32) // group_size
        shuf_weight = GPTQShuffle(bits=4, blocksize=group_size)
        shuffled_weight, g_idx4kernel = shuf_weight(weight, g_idx)
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
        False,
        g_idx4kernel,
    )

    torch.testing.assert_close(
        output_int4.cpu().float(),
        out_torch.cpu().float(),
        atol=1e-2,
        rtol=1e-2,
    )
