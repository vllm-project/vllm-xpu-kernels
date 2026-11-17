# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.ops.mx_utils import (FP4_EBITS, FP4_MBITS, _floatx_unpacked_to_f32,
                                from_blocked_format, to_mxfp, unpack_uint4)
from tests.register_ops import mxfp4_gemm_w4a16, onednn_available

pytestmark = pytest.mark.skipif(
    not onednn_available(),
    reason="oneDNN kernels are not built "
    "(rebuild with VLLM_XPU_ENABLE_ONEDNN=ON)")

DTYPES = [torch.float16, torch.bfloat16]
MNK_FACTORS = [
    (1, 32, 1024),
    (16, 2048, 8192),
    (64, 8768, 8192),
    (300, 512, 2048),
    (2048, 1024, 2048),
]

MINI_PYTEST_PARAMS = {
    "default": {
        "mnk_factors": [(1, 32, 32), (32, 64, 64)],
    }
}


@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("use_bias", [False, True])
def test_mxfp4_gemm_w4a16(mnk_factors, dtype, use_bias):
    """bf16/fp16 activations x MXFP4 weights (weight-only, no activation
    quantization) match a matmul against the dequantized weights."""
    m, n, k = mnk_factors
    inputs = torch.randn((m, k), dtype=dtype).xpu()
    weights = torch.randn((n, k), dtype=torch.float).xpu() * 0.01
    w_scale, w_lp = to_mxfp(weights, format="mxfp4")  # [n, k/32], [n, k/2]
    w_hp = from_blocked_format(
        _floatx_unpacked_to_f32(unpack_uint4(w_lp), FP4_EBITS, FP4_MBITS),
        w_scale,
        blocksize=32,
    )
    bias = torch.randn(n, dtype=dtype).xpu() if use_bias else None

    output = mxfp4_gemm_w4a16(inputs, w_lp.t(), w_scale.t().contiguous(), bias)

    ref = inputs.float() @ w_hp.float().t()
    if bias is not None:
        ref = ref + bias.float()
    assert output.dtype == dtype and output.shape == (m, n)
    torch.testing.assert_close(output.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("bad", ["a_k_strided", "k_not_32", "scale_shape"])
def test_mxfp4_gemm_w4a16_rejects_bad_layout(bad):
    """Layouts the kernel would misread raise instead of returning garbage."""
    m, n, k = 4, 64, 128
    if bad == "k_not_32":
        k = 48
    a = torch.randn((m, k), dtype=torch.bfloat16).xpu()
    if bad == "a_k_strided":
        a = torch.randn((k, m), dtype=torch.bfloat16).xpu().t()
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8).xpu()
    s = torch.full((max(k // 32, 1) + (bad == "scale_shape"), n), 127,
                   dtype=torch.uint8).xpu()
    with pytest.raises(RuntimeError):
        mxfp4_gemm_w4a16(a, w.view(torch.float4_e2m1fn_x2).t(),
                         s.view(torch.float8_e8m0fnu), None)
