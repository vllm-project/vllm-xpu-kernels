# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.ops.fp8_quant_op import scaled_fp8_quant
from tests.register_ops import fp8_gemm
from tests.register_ops import fp8_gemm_w8a16

BATCHES = [1]
MNK_FACTORS = [
    (1, 4, 8),
    (2, 4, 8),
    (32, 64, 16),
    # (32, 64, 128),  @TODO: enable this test after fixing w8a8 accuracy issue
    # (64, 128, 256),
]


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_nt", [True, False])
@pytest.mark.parametrize("is_mbk", [True, False])
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_fp8_gemm_w8a16(fp8_dtype, dtype, is_nt, is_mbk, batch, mnk_factors):
    seed = 1234
    torch.manual_seed(seed)

    m, n, k = mnk_factors

    input = torch.randn([batch, m, k], dtype=dtype, device=torch.device("xpu")) / 10.0
    weight = torch.rand([n, k], dtype=dtype).xpu() / 10.0
    if not is_nt:
        weight = weight.transpose(0, 1).contiguous()
    scale_wei = (torch.ones(batch) * 4).xpu()
    scale_shape = None

    weight_fp8, _ = scaled_fp8_quant(
        weight, scale_wei, False, False, fp8_dtype, scale_shape
    )

    # reference fp16 gemm
    if not is_nt:
        weight = weight.transpose(0, 1).contiguous()
    output_ref = torch.matmul(input, weight.t())

    # onednn fp8 gemm
    if is_mbk:
        input = input.transpose(0, 1)
    output_fp8 = fp8_gemm_w8a16(
        input,
        weight_fp8,
        is_nt,
        scale_wei,
        torch.Tensor(),
    )
    output_fp8 = output_fp8.transpose(0, 1) if is_mbk else output_fp8

    torch.testing.assert_close(output_fp8, output_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_nt", [True, False])
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
@pytest.mark.parametrize("per_channel", [False])
def test_fp8_gemm(fp8_dtype, dtype, is_nt, batch, mnk_factors, per_channel):
    seed = 1234
    torch.manual_seed(seed)

    m, n, k = mnk_factors

    input = torch.randn([batch * m, k], dtype=dtype, device=torch.device("xpu")) / 10.0
    weight = torch.rand([n, k], dtype=dtype).xpu() / 10.0 + 1e-6  # avoid nans
    if not is_nt:
        weight = weight.transpose(0, 1).contiguous()
    scale_src = (torch.ones(batch) * 4).xpu()
    scale_wei = (torch.ones(batch) * 4).xpu()

    input_fp8, _ = scaled_fp8_quant(
        input.reshape(-1, k), scale_src, False, False, fp8_dtype=fp8_dtype
    )

    if per_channel:
        weight_fp8, scale_fp8 = scaled_fp8_quant(
            weight, use_per_token_if_dynamic=True, fp8_dtype=fp8_dtype
        )
    else:
        weight_fp8, _ = scaled_fp8_quant(
            weight, scale_wei, False, False, fp8_dtype=fp8_dtype
        )

    # reference fp16 gemm
    if not is_nt:
        weight = weight.transpose(0, 1).contiguous()
    output_ref = torch.matmul(input, weight.t())

    # onednn fp8 gemm
    output_fp8 = fp8_gemm(
        input_fp8,
        weight_fp8,
        dtype,
        is_nt,
        scale_src,
        scale_fp8 if per_channel else scale_wei,
        torch.Tensor(),
    )

    torch.testing.assert_close(output_fp8, output_ref, atol=1e-1, rtol=1e-1)
