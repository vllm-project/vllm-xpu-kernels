# SPDX-License-Identifier: Apache-2.0
from enum import Enum

import pytest
import torch

from tests.register_ops import int8_gemm_w8a8
from vllm_xpu_kernels.quantization._quantize_convert import (
    dynamic_per_token_quant_ref)

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


class QuantMode(Enum):
    SYM = 1
    ASYM = 2


def _make_int8_weight(n: int, k: int, device: str = "xpu") -> torch.Tensor:
    """Create a random int8 weight tensor of shape [n, k]."""
    return torch.randint(-128, 127, (n, k), dtype=torch.int8, device=device)


def _per_channel_scale(n: int, dtype: torch.dtype, device: str = "xpu"):
    """Return per-channel weight scale of shape [n, 1]."""
    return torch.rand(n, 1, dtype=dtype, device=device).clamp(min=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
@pytest.mark.parametrize("act_qmode", [QuantMode.SYM, QuantMode.ASYM])
def test_int8_gemm_w8a8(dtype, mnk_factors, act_qmode: QuantMode):
    """W8A8: per-token activation scales, per-channel weight scales."""
    seed = 1234
    torch.manual_seed(seed)

    m, n, k = mnk_factors
    use_sym_quant = (act_qmode == QuantMode.SYM)

    # Floating-point input
    input_fp = torch.randn(m, k, device="xpu", dtype=dtype)

    # Per-token quantization of activations (symmetric or asymmetric)
    input_quant, act_scales, act_zp = dynamic_per_token_quant_ref(
        input_fp, use_sym_quant=use_sym_quant, bits=8)

    # Per-channel weight quantization (symmetric only)
    weight_quant = _make_int8_weight(n, k)  # [n, k]
    weight_scale = _per_channel_scale(n, dtype)  # [n, 1]
    weight_zp = torch.zeros((n, 1), dtype=torch.int32, device="xpu")
    # Use int matmul + dequant to match kernel computation order
    if use_sym_quant:
        acc_ref = torch.matmul(input_quant.cpu().to(torch.int32),
                               weight_quant.cpu().to(
                                   torch.int32).t())  # [m, n]
        out_ref = acc_ref.float() * act_scales.cpu().float() * weight_scale.t(
        ).cpu().float()
    else:
        acc_ref = torch.matmul(
            input_quant.cpu().to(torch.int32) - act_zp.cpu(),
            weight_quant.cpu().to(torch.int32).t())  # [m, n]
        out_ref = acc_ref.float() * act_scales.cpu().float() * weight_scale.t(
        ).cpu().float()

    # oneDNN int8 gemm
    weight_scale_t = weight_scale.t().contiguous()  # [1, n]
    weight_zp_t = weight_zp.t().contiguous()  # [1, n]

    out_int8 = int8_gemm_w8a8(
        input_quant,
        act_scales,
        act_zp,
        weight_quant.t(),  # [k, n] NT-format view
        weight_scale_t,  # [1, n]
        weight_zp_t,  # [1, n]
        -1,  # group_size=-1 → per-channel
        None,  # no bias
    )

    torch.testing.assert_close(
        out_int8.cpu().float(),
        out_ref.cpu().float(),
        atol=5e-1,
        rtol=5e-1,
    )


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
@pytest.mark.parametrize("act_qmode", [QuantMode.SYM, QuantMode.ASYM])
def test_int8_gemm_w8a8_with_bias(dtype, mnk_factors, act_qmode: QuantMode):
    """W8A8 per-channel with an additive bias vector."""
    seed = 42
    torch.manual_seed(seed)

    m, n, k = mnk_factors
    use_sym_quant = (act_qmode == QuantMode.SYM)

    # Floating-point input
    input_fp = torch.randn(m, k, device="xpu", dtype=dtype)
    bias = torch.randn(n, dtype=dtype, device="xpu")

    # Per-token quantization of activations
    input_quant, act_scales, act_zp = dynamic_per_token_quant_ref(
        input_fp, use_sym_quant=use_sym_quant, bits=8)

    # Per-channel weight quantization (symmetric only)
    weight_quant = _make_int8_weight(n, k)  # [n, k]
    weight_scale = _per_channel_scale(n, dtype)  # [n, 1]
    weight_zp = torch.zeros((n, 1), dtype=torch.int32, device="xpu")

    # Use int matmul + dequant to match kernel computation order
    if use_sym_quant:
        acc_ref = torch.matmul(input_quant.cpu().to(torch.int32),
                               weight_quant.cpu().to(
                                   torch.int32).t())  # [m, n]
        out_ref = (acc_ref.float() * act_scales.cpu().float() *
                   weight_scale.t().cpu().float() + bias.float().cpu())
    else:
        acc_ref = torch.matmul(
            input_quant.cpu().to(torch.int32) - act_zp.cpu(),
            weight_quant.cpu().to(torch.int32).t())  # [m, n]
        out_ref = (acc_ref.float() * act_scales.cpu().float() *
                   weight_scale.t().cpu().float() + bias.float().cpu())

    # oneDNN int8 gemm
    weight_scale_t = weight_scale.t().contiguous()
    weight_zp_t = weight_zp.t().contiguous()

    out_int8 = int8_gemm_w8a8(
        input_quant,
        act_scales,
        act_zp,
        weight_quant.t(),
        weight_scale_t,
        weight_zp_t,
        -1,
        bias,
    )

    torch.testing.assert_close(
        out_int8.cpu().float(),
        out_ref.cpu().float(),
        atol=5e-1,
        rtol=5e-1,
    )
