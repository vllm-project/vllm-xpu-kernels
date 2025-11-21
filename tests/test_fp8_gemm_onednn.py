# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.ops.fp8_quant_op import scaled_fp8_quant
from tests.register_ops import fp8_gemm, fp8_gemm_w8a16

BATCHES = [1]
MNK_FACTORS = [
    (1, 4096, 1),
    (1, 32, 1024),
    (4, 16, 1024),
    (8, 32, 1024),
    (8, 512, 1024),
]

MINI_MNK_FACTORS = [
    (1, 4, 8),
    (2, 4, 8),
    (4, 32, 16),
]

#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "test_fp8_gemm_w8a16": {
        "mnk_factors": MINI_MNK_FACTORS[:1],
    },
    "test_fp8_gemm_per_tensor": {
        "mnk_factors": MINI_MNK_FACTORS,
    },
    "test_fp8_gemm_per_channel": {
        "mnk_factors": MINI_MNK_FACTORS,
    },
}


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("trans_wei", [True, False])
@pytest.mark.parametrize("is_mbk", [True, False])
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_fp8_gemm_w8a16(fp8_dtype, dtype, trans_wei, is_mbk, batch,
                        mnk_factors):
    seed = 1234
    torch.manual_seed(seed)

    m, n, k = mnk_factors

    input = torch.randn([batch, m, k], dtype=dtype,
                        device=torch.device("xpu")) / 10.0
    if trans_wei:
        weight = torch.ones([n, k], dtype=dtype).xpu()
    else:
        weight = torch.ones([k, n], dtype=dtype).xpu()
    scale_wei = (torch.ones(batch) * 4).xpu()
    scale_shape = None

    weight_fp8, _ = scaled_fp8_quant(weight, scale_wei, False, False,
                                     fp8_dtype, scale_shape)

    # reference fp16 gemm
    if trans_wei:
        output_ref = torch.matmul(input, weight.t())
    else:
        output_ref = torch.matmul(input, weight)

    # onednn fp8 gemm
    if is_mbk:
        input = input.transpose(0, 1)
    output_fp8 = fp8_gemm_w8a16(
        input,
        weight_fp8.transpose(0, 1) if trans_wei else weight_fp8,
        scale_wei,
        torch.Tensor(),
    )
    output_fp8 = output_fp8.transpose(0, 1) if is_mbk else output_fp8

    torch.testing.assert_close(output_fp8, output_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_nt", [True, False])
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_fp8_gemm_per_tensor(fp8_dtype, dtype, is_nt, batch, mnk_factors):
    seed = 1234
    torch.manual_seed(seed)

    m, n, k = mnk_factors

    input = torch.randn(
        [batch * m, k], dtype=dtype, device=torch.device("xpu")) / 10.0
    weight = torch.randn([n, k], dtype=dtype).xpu() / 10.0

    scale_src = (torch.ones(batch) * 4).xpu()
    scale_wei = (torch.ones(batch) * 4).xpu()

    input_fp8, _ = scaled_fp8_quant(input.reshape(-1, k),
                                    scale_src,
                                    False,
                                    False,
                                    fp8_dtype=fp8_dtype)

    weight_fp8, _ = scaled_fp8_quant(weight,
                                     scale_wei,
                                     False,
                                     False,
                                     fp8_dtype=fp8_dtype)

    # reference fp16 gemm
    output_ref = torch.matmul(input, weight.t())

    weight_fp8 = weight_fp8.transpose(0, 1)
    if is_nt:
        weight_fp8 = weight_fp8.contiguous()

    output_fp8 = fp8_gemm(
        input_fp8,
        weight_fp8,
        dtype,
        scale_src,
        scale_wei,
        torch.Tensor(),
    )

    torch.testing.assert_close(output_fp8, output_ref, atol=6e-2, rtol=6e-2)


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_nt", [True, False])
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_fp8_gemm_per_channel(fp8_dtype, dtype, is_nt, batch, mnk_factors):
    seed = 1234
    torch.manual_seed(seed)

    m, n, k = mnk_factors

    input = torch.randn(
        [batch * m, k], dtype=dtype, device=torch.device("xpu")) / 10.0
    weight = torch.randn([n, k], dtype=dtype).xpu() / 10.0

    input_fp8, scale_src_fp8 = scaled_fp8_quant(input.reshape(-1, k),
                                                use_per_token_if_dynamic=True,
                                                fp8_dtype=fp8_dtype)

    weight_fp8, scale_wei_fp8 = scaled_fp8_quant(weight,
                                                 use_per_token_if_dynamic=True,
                                                 fp8_dtype=fp8_dtype)

    # reference fp16 gemm
    output_ref = torch.matmul(input, weight.t())

    weight_fp8 = weight_fp8.transpose(0, 1)
    if is_nt:
        weight_fp8 = weight_fp8.contiguous()

    output_fp8 = fp8_gemm(
        input_fp8,
        weight_fp8,
        dtype,
        scale_src_fp8,
        scale_wei_fp8,
        torch.Tensor(),
    )

    torch.testing.assert_close(output_fp8, output_ref, atol=6e-2, rtol=6e-2)
