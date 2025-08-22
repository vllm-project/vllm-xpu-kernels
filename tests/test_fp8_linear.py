# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.ops.fp8_quant_op import scaled_fp8_quant
from vllm_xpu_kernels.layers.quantization.fp8_linear import (
    QuantDtype, WeightOnlyQuantizedLinear)


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_bias", [True, False])
@pytest.mark.parametrize("is_nt", [True, False])
@pytest.mark.parametrize("is_mbk", [True, False])
def test_fp8_linear_w8a16(fp8_dtype, dtype, is_bias, is_nt, is_mbk):
    seed = 1234
    torch.manual_seed(seed)

    input = torch.randn([1, 8, 2], dtype=dtype,
                        device=torch.device("xpu")) / 10.0
    weight = torch.rand([3, 2], dtype=dtype).xpu() / 10.0

    gemm_ref = torch.nn.Linear(2, 3, bias=is_bias).xpu().to(dtype)

    scale_wei = (torch.ones(1) * 4).xpu()
    scale_wei_inv = torch.tensor([0.25]).xpu()
    scale_shape = None

    weight_fp8, _ = scaled_fp8_quant(weight, scale_wei, False, False,
                                     fp8_dtype, scale_shape)

    gemm_ref.weight.data = weight
    output_ref = gemm_ref(input)

    if not is_nt:
        weight_fp8 = weight_fp8.transpose(0, 1).contiguous().transpose(0, 1)

    fp8_linear = WeightOnlyQuantizedLinear.from_weight(
        weight_fp8,
        scale_wei_inv,
        torch.Tensor(),
        in_feature=weight.shape[1],
        out_feature=weight.shape[0],
        dtype=(QuantDtype.FP8_E5M2
               if fp8_dtype == torch.float8_e5m2 else QuantDtype.FP8_E4M3FN),
    )

    if is_mbk:
        input = input.transpose(0, 1)

    output_fp8 = fp8_linear(input,
                            gemm_ref.bias.data.clone() if is_bias else None)
    output_fp8 = output_fp8.transpose(0, 1) if is_mbk else output_fp8

    torch.testing.assert_close(output_fp8, output_ref, atol=1e-2, rtol=1e-2)