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


def _make_uint8_weight_and_zp(
        n: int,
        k: int,
        device: str = "xpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Create a random uint8 weight [n, k] with per-channel non-zero int32
    zero_points [n, 1].  Small ZPs keep int32 accumulator within range."""
    weight = torch.randint(0, 255, (n, k), dtype=torch.uint8, device=device)
    zp = torch.randint(64, 192, (n, 1), dtype=torch.int32, device=device)
    return weight, zp


def _per_channel_scale(n: int, dtype: torch.dtype, device: str = "xpu"):
    """Return per-channel weight scale of shape [n, 1]."""
    return torch.rand(n, 1, dtype=dtype, device=device).clamp(min=1e-3)


def _w8a8_ref(input_quant: torch.Tensor, act_scales: torch.Tensor,
              act_zp: torch.Tensor, weight_quant: torch.Tensor,
              weight_scale: torch.Tensor,
              weight_zp: torch.Tensor) -> torch.Tensor:
    """CPU reference for W8A8 with optional zero_points on both sides.

    out[m,n] = act_scale[m] * wei_scale[n]
               * sum_k( (act[m,k] - act_zp[m]) * (wei[n,k] - wei_zp[n]) )
    """
    act_adj = input_quant.cpu().to(torch.int32) - act_zp.cpu()  # [m, k]
    wei_adj = weight_quant.cpu().to(torch.int32) - weight_zp.cpu()  # [n, k]
    acc = torch.matmul(act_adj, wei_adj.t()).float()  # [m, n]
    return acc * act_scales.cpu().float() * weight_scale.t().cpu().float()


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("wei_qmode", [QuantMode.SYM, QuantMode.ASYM])
@pytest.mark.parametrize("act_qmode", [QuantMode.SYM, QuantMode.ASYM])
def test_int8_gemm_w8a8(dtype, mnk_factors, act_qmode: QuantMode,
                        wei_qmode: QuantMode, with_bias: bool):
    """W8A8: per-token activation scales, per-channel weight scales.

    act_qmode=SYM  → int8 activations with zero_point=0
    act_qmode=ASYM → uint8 activations with non-zero zero_points
    wei_qmode=SYM  → int8 weights with zero_point=0
    wei_qmode=ASYM → uint8 weights with non-zero zero_points
    with_bias      → add a per-output-channel bias after dequantization
    """
    seed = 1234
    torch.manual_seed(seed)

    m, n, k = mnk_factors
    use_sym_act = (act_qmode == QuantMode.SYM)
    use_sym_wei = (wei_qmode == QuantMode.SYM)

    input_fp = torch.randn(m, k, device="xpu", dtype=dtype)
    bias = torch.randn(n, dtype=dtype, device="xpu") if with_bias else None

    # Per-token quantization of activations (symmetric or asymmetric)
    input_quant, act_scales, act_zp = dynamic_per_token_quant_ref(
        input_fp, use_sym_quant=use_sym_act, bits=8)

    # Per-channel weight quantization
    if use_sym_wei:
        weight_quant = _make_int8_weight(n, k)  # [n, k] int8
        weight_zp = torch.zeros((n, 1), dtype=torch.int32, device="xpu")
    else:
        weight_quant, weight_zp = _make_uint8_weight_and_zp(n, k)  # uint8
    weight_scale = _per_channel_scale(n, dtype)  # [n, 1]

    out_ref = _w8a8_ref(input_quant, act_scales, act_zp, weight_quant,
                        weight_scale, weight_zp)
    if bias is not None:
        out_ref = out_ref + bias.float().cpu()

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
        bias,
    )

    torch.testing.assert_close(
        out_int8.cpu().float(),
        out_ref.cpu().float(),
        atol=5e-1,
        rtol=5e-1,
    )
