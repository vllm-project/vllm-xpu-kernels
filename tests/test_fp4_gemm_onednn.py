# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.ops.mx_utils import (FP4_EBITS, FP4_MBITS, _floatx_unpacked_to_f32,
                                from_blocked_format, to_mxfp, unpack_uint4)
from tests.register_ops import fp4_gemm

OUT_DTYPES = [torch.float16, torch.bfloat16]
MNK_FACTORS = [
    (1, 32, 1024),
    (1, 32, 2048),
    (1, 32, 5120),
    (8, 512, 2048),
    (512, 1024, 2048),
    (1024, 2048, 2048),
]

MINI_MX_MNK_FACTORS = [
    (1, 32, 32),
    (1, 64, 32),
    (32, 32, 32),
    (32, 64, 32),
]

# override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "mnk_factors": MINI_MX_MNK_FACTORS,
    }
}


def _convert_to_mxfp4_with_hp_ref(t):
    # Convert a tensor to mxfp4, returning:
    #   t_hp : reconstructed bf16 version of t_lp
    #   t_lp : fp4_e2m1x2 tensor
    #   t_scale: fp8_e8m0 block-wise scaling factors (non-swizzled)
    t_scale, t_lp = to_mxfp(t, format="mxfp4")
    t_hp = from_blocked_format(
        _floatx_unpacked_to_f32(unpack_uint4(t_lp), FP4_EBITS, FP4_MBITS),
        t_scale,
        blocksize=32,
    )

    return t_hp, t_lp, t_scale


@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
def test_mxfp4_gemm(mnk_factors, out_dtype):
    m, n, k = mnk_factors

    inputs = torch.randn((m, k), dtype=out_dtype).xpu() * 0.01
    weights = torch.randn((n, k), dtype=out_dtype).xpu() * 0.01

    # Reference: to_mxfp operates on float32 or bfloat16.
    if out_dtype is torch.half:
        inputs = inputs.to(torch.float)
        weights = weights.to(torch.float)
    inputs_hp, inputs_lp, inputs_scale = _convert_to_mxfp4_with_hp_ref(inputs)
    weights_hp, weights_lp, weights_scale = _convert_to_mxfp4_with_hp_ref(
        weights)

    output = fp4_gemm(
        inputs_lp,
        weights_lp.transpose(0, 1),
        inputs_scale,
        weights_scale,
        out_dtype,
        torch.Tensor(),
    )

    output_ref = torch.matmul(inputs_hp.to(out_dtype),
                              weights_hp.to(out_dtype).t())
    torch.testing.assert_close(output.to(torch.float),
                               output_ref.to(torch.float),
                               atol=5e-2,
                               rtol=5e-2)
