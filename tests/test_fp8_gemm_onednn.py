# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.ops.fp8_quant_op import scaled_fp8_quant
from tests.ops.mx_utils import from_blocked_format, to_mxfp
from tests.register_ops import fp8_gemm, fp8_gemm_w8a16

BATCHES = [1, 2, 8]
OUT_DTYPES = [torch.float16, torch.bfloat16]
MNK_FACTORS = [
    (1, 32, 1024),
    (1, 32, 2048),
    (1, 32, 5120),
    (8, 512, 2048),
    (512, 1024, 2048),
    (1024, 2048, 2048),
]

MINI_MNK_FACTORS = [
    (1, 4, 8),
    (2, 4, 8),
    (4, 32, 16),
]

MINI_MX_MNK_FACTORS = [
    (1, 32, 32),
    (1, 64, 32),
    (32, 32, 32),
    (32, 64, 32),
]

# override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "test_fp8_gemm_w8a16": {
        "batch": [1],
        "mnk_factors": MINI_MNK_FACTORS[:1],
    },
    "test_fp8_gemm_per_tensor": {
        "mnk_factors": MINI_MNK_FACTORS,
    },
    "test_fp8_gemm_per_channel": {
        "mnk_factors": MINI_MNK_FACTORS,
    },
    "test_fp8_gemm_w8a16_per_channel": {
        "mnk_factors": MINI_MNK_FACTORS[:1],
    },
    "test_mxfp8_gemm": {
        "mnk_factors": MINI_MX_MNK_FACTORS,
    },
}


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
@pytest.mark.parametrize("trans_wei", [True, False])
@pytest.mark.parametrize("is_mbk", [True, False])
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_fp8_gemm_w8a16(fp8_dtype, out_dtype, trans_wei, is_mbk, batch,
                        mnk_factors):
    seed = 1234
    torch.manual_seed(seed)

    m, n, k = mnk_factors

    input = torch.randn(
        [batch, m, k], dtype=out_dtype, device=torch.device("xpu")) / 10.0
    if trans_wei:
        weight = torch.ones([n, k], dtype=out_dtype).xpu()
    else:
        weight = torch.ones([k, n], dtype=out_dtype).xpu()
    scale_wei = torch.tensor(4.0).xpu()

    weight_fp8, _ = scaled_fp8_quant(weight,
                                     scale_wei,
                                     fp8_dtype=fp8_dtype,
                                     group_shape=(-1, 1))
    weight_fp8_hp = weight_fp8.to(out_dtype) * scale_wei.to(out_dtype)

    # reference fp16 gemm
    if trans_wei:
        output_ref = torch.matmul(input, weight_fp8_hp.t())
    else:
        output_ref = torch.matmul(input, weight_fp8_hp)

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
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
@pytest.mark.parametrize("is_nt", [True, False])
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_fp8_gemm_per_tensor(fp8_dtype, out_dtype, is_nt, batch, mnk_factors):
    seed = 1234
    torch.manual_seed(seed)

    m, n, k = mnk_factors

    input = torch.randn(
        [batch, m, k], dtype=out_dtype, device=torch.device("xpu")) / 10.0
    weight = torch.randn([n, k], dtype=out_dtype).xpu() / 10.0

    scale_src = torch.tensor(4.0).xpu()
    scale_wei = torch.tensor(4.0).xpu()

    input_fp8, _ = scaled_fp8_quant(input.reshape(-1, k),
                                    scale_src,
                                    fp8_dtype=fp8_dtype)
    weight_fp8, _ = scaled_fp8_quant(weight, scale_wei, fp8_dtype=fp8_dtype)

    input_fp8_hp = input_fp8.to(out_dtype) * scale_src.to(out_dtype)
    weight_fp8_hp = weight_fp8.to(out_dtype) * scale_wei.to(out_dtype)

    # reference fp16 gemm
    output_ref = torch.matmul(input_fp8_hp, weight_fp8_hp.t())

    weight_fp8 = weight_fp8.transpose(0, 1)
    if is_nt:
        weight_fp8 = weight_fp8.contiguous()

    output_fp8 = fp8_gemm(
        input_fp8,
        weight_fp8,
        out_dtype,
        scale_src,
        scale_wei,
        torch.Tensor(),
    )

    torch.testing.assert_close(output_fp8, output_ref, atol=6e-2, rtol=6e-2)


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
@pytest.mark.parametrize("is_nt", [True, False])
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_fp8_gemm_per_channel(fp8_dtype, out_dtype, is_nt, batch, mnk_factors):
    seed = 1234
    torch.manual_seed(seed)

    m, n, k = mnk_factors

    input = torch.randn(
        [batch, m, k], dtype=out_dtype, device=torch.device("xpu")) / 10.0
    weight = torch.randn([n, k], dtype=out_dtype).xpu() / 10.0

    input_fp8, scale_src_fp8 = scaled_fp8_quant(input.reshape(-1, k),
                                                use_per_token_if_dynamic=True,
                                                fp8_dtype=fp8_dtype)
    weight_fp8, scale_wei_fp8 = scaled_fp8_quant(weight,
                                                 use_per_token_if_dynamic=True,
                                                 fp8_dtype=fp8_dtype)

    # reference fp16 gemm
    input_fp8_hp = input_fp8.to(out_dtype) * scale_src_fp8.to(out_dtype)
    weight_fp8_hp = weight_fp8.to(out_dtype) * scale_wei_fp8.to(out_dtype)
    output_ref = torch.matmul(input_fp8_hp, weight_fp8_hp.t())

    weight_fp8 = weight_fp8.transpose(0, 1)
    if is_nt:
        weight_fp8 = weight_fp8.contiguous()

    output_fp8 = fp8_gemm(
        input_fp8,
        weight_fp8,
        out_dtype,
        scale_src_fp8,
        scale_wei_fp8,
        torch.Tensor(),
    )

    torch.testing.assert_close(output_fp8, output_ref, atol=6e-2, rtol=6e-2)


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
@pytest.mark.parametrize("is_nt", [True, False])
@pytest.mark.parametrize("is_mbk", [True, False])
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_fp8_gemm_w8a16_per_channel(fp8_dtype, out_dtype, is_nt, is_mbk, batch,
                                    mnk_factors):
    seed = 1234
    torch.manual_seed(seed)

    m, n, k = mnk_factors

    input = torch.randn(
        [batch, m, k], dtype=out_dtype, device=torch.device("xpu")) / 10.0
    weight = torch.randn([n, k], dtype=out_dtype).xpu() / 10.0

    weight_fp8, scale_wei_fp8 = scaled_fp8_quant(weight,
                                                 use_per_token_if_dynamic=True,
                                                 fp8_dtype=fp8_dtype)
    # scale_wei_fp8 is [n, 1], flatten to [n] for per-channel scale
    scale_wei_flat = scale_wei_fp8.flatten()

    # reference: dequantize weight then fp16/bf16 matmul
    weight_dequant = weight_fp8.to(out_dtype) * scale_wei_fp8.to(out_dtype)
    output_ref = torch.matmul(input, weight_dequant.t())

    weight_fp8_t = weight_fp8.transpose(0, 1)
    if is_nt:
        weight_fp8_t = weight_fp8_t.contiguous()

    if is_mbk:
        input = input.transpose(0, 1)
    output_fp8 = fp8_gemm_w8a16(
        input,
        weight_fp8_t,
        scale_wei_flat,
        torch.Tensor(),
    )
    output_fp8 = output_fp8.transpose(0, 1) if is_mbk else output_fp8

    torch.testing.assert_close(output_fp8, output_ref, atol=5e-2, rtol=5e-2)


def _convert_to_mxfp8_with_hp_ref(t):
    # Convert a tensor to mxfp8, returning:
    #   t_hp : reconstructed bf16 version of t_lp
    #   t_lp : fp8_e4m3 tensor
    #   t_scale: fp8_e8m0 block-wise scaling factors (non-swizzled)
    t_scale, t_lp = to_mxfp(t, format="mxfp8")
    t_hp = from_blocked_format(t_lp, t_scale, blocksize=32)

    return t_hp, t_lp, t_scale


@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
def test_mxfp8_gemm(mnk_factors, out_dtype):
    m, n, k = mnk_factors
    inputs = torch.randn((m, k), dtype=out_dtype).xpu() * 0.01
    weights = torch.randn((n, k), dtype=out_dtype).xpu() * 0.01

    # Reference: to_mxfp operates on float32 or bfloat16.
    if out_dtype == torch.half:
        inputs = inputs.to(torch.float32)
        weights = weights.to(torch.float32)

    inputs_hp, inputs_lp, inputs_scale = _convert_to_mxfp8_with_hp_ref(inputs)
    weights_hp, weights_lp, weights_scale = _convert_to_mxfp8_with_hp_ref(
        weights)

    output = fp8_gemm(
        inputs_lp,
        weights_lp.transpose(0, 1),
        out_dtype,
        inputs_scale,
        weights_scale,
        torch.Tensor(),
    )

    output_ref = torch.matmul(inputs_hp.to(out_dtype),
                              weights_hp.to(out_dtype).t())
    torch.testing.assert_close(output, output_ref, atol=5e-2, rtol=5e-2)
