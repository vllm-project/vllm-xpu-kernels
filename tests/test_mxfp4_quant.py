# SPDX-License-Identifier: Apache-2.0

import random

import numpy as np
import pytest
import torch

from tests.ops.mx_utils import (FP4_EBITS, FP4_MBITS, _floatx_unpacked_to_f32,
                                to_mxfp, unpack_uint4)
from tests.ops.mxfp4_quant_op import per_token_group_quant_mxfp4

DTYPES = [torch.float, torch.bfloat16, torch.half]

NUM_TOKENS = [1, 2, 4, 8]
HIDDEN_SIZES = [32, 64, 128, 256, 512]

# MXFP4 block size is fixed at 32 by the MX specification.
MXFP4_GROUP_SIZE = 32
COLUMN_MAJOR_SCALE = [True, False]
SEEDS = [0]

MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [1],
        "hidden_size": [64],
    },
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dequantize_mxfp4(
    packed_q: torch.Tensor,  # [M, N/2] float4_e2m1fn_x2 or uint8
    scales: torch.Tensor,  # [M, N/32] float32 – UE8M0-rounded scales
    group_size: int = 32,
) -> torch.Tensor:
    M = scales.shape[0]
    packed_u8 = packed_q.view(torch.uint8)
    N = packed_u8.shape[-1] * 2

    unpacked = unpack_uint4(packed_u8)  # [M, N] uint8

    fp4_float = _floatx_unpacked_to_f32(unpacked.cpu(),
                                        FP4_EBITS, FP4_MBITS).to(
                                            scales.device)  # [M, N] float32

    fp4_blocked = fp4_float.reshape(M, N // group_size, group_size)
    scales_flat = scales.reshape(M, N // group_size)
    dequant = (fp4_blocked * scales_flat.unsqueeze(-1)).reshape(M, N)
    return dequant


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("column_major_scale", COLUMN_MAJOR_SCALE)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_per_block_mxfp4_quant(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    column_major_scale: bool,
    seed: int,
) -> None:
    seed_everything(seed)

    x = (torch.rand(num_tokens, hidden_size, dtype=dtype, device="xpu") + 1e-6
         )  # avoid all-zero groups which can produce degenerate scales

    # Reference: to_mxfp operates on float32 or bfloat16.
    x_ref = x.to(torch.float32) if dtype == torch.half else x
    ref_scales, ref_out = to_mxfp(x_ref.cpu(),
                                  block_size=MXFP4_GROUP_SIZE,
                                  format="mxfp4")

    # Kernel output – both return float4_e2m1fn_x2 + float8_e8m0fnu.
    ops_out, ops_scales = per_token_group_quant_mxfp4(
        x,
        group_size=MXFP4_GROUP_SIZE,
        column_major_scales=column_major_scale,
    )

    # 1. Compare UE8M0-rounded scales.
    torch.testing.assert_close(
        ref_scales.float(),
        ops_scales.cpu().float(),
        atol=1e-5,
        rtol=1e-5,
        msg="MXFP4 UE8M0 scales do not match the reference",
    )

    # 2. Compare dequantised outputs.
    ref_dequant = dequantize_mxfp4(ref_out, ref_scales.float(),
                                   MXFP4_GROUP_SIZE)
    ops_dequant = dequantize_mxfp4(ops_out.cpu(),
                                   ops_scales.cpu().float(), MXFP4_GROUP_SIZE)
    torch.testing.assert_close(
        ref_dequant,
        ops_dequant,
        atol=0.2,
        rtol=0.2,
        msg="MXFP4 dequantised values do not match the reference",
    )


# Sanity check: known single-value conversions
@torch.inference_mode()
def test_mxfp4_known_values() -> None:
    """
    Verify FP4 E2M1 quantisation of known input values against the reference
    implementation.  Uses a [1, 32] tensor with a single non-zero element so
    that the scale is easily predictable.
    """
    # FP4 E2M1 positive values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
    fp4_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

    for val in fp4_values:
        # Build a [1, 32] tensor: first element = val, rest near 0.
        x = torch.zeros(1, 32, dtype=torch.float32, device="xpu")
        x[0, 0] = val if val > 0 else 0.5  # avoid all-zero → use 0.5 as probe

        ref_scales, ref_out = to_mxfp(x.cpu(), block_size=32, format="mxfp4")

        ops_out, ops_scales = per_token_group_quant_mxfp4(x, group_size=32)

        torch.testing.assert_close(ref_scales.float(),
                                   ops_scales.cpu().float(),
                                   atol=1e-5,
                                   rtol=1e-5)

        # Both are float4_e2m1fn_x2; compare via uint8 view.
        torch.testing.assert_close(ref_out.view(torch.uint8),
                                   ops_out.cpu().view(torch.uint8),
                                   atol=0.2,
                                   rtol=0.2)
