# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

import tests.register_ops as ops
from tests.utils import seed_everything

DTYPES = [torch.float16, torch.bfloat16]
QUANT_DTYPES = [torch.float8_e4m3fn, torch.int8]
SEEDS = [0]
XPU_DEVICES = [f"xpu:{i}" for i in range(min(torch.xpu.device_count(), 2))] or [
    "xpu:0"
]  # fallback so parametrize list is non-empty; tests will skip on no-device

# --- per-block-quant params ---
VEC_HIDDEN_SIZES = [1024, 1025, 1027, 1029]
NUM_TOKENS_HIDDEN_SIZES = [
    *[(1, i) for i in [64, *VEC_HIDDEN_SIZES, 2048, 5120]],
    *[(16, i) for i in [64, *VEC_HIDDEN_SIZES, 5120]],
    *[(128, i) for i in [64, *VEC_HIDDEN_SIZES]],
    *[(512, i) for i in [64, 5120]],
]
GROUP_SIZES = [64, 128]
IS_SCALE_TRANSPOSED = [False, True]

MINI_PYTEST_PARAMS = {
    "test_silu_and_mul_per_block_quant": {
        "num_tokens, hidden_size": [(1, 128)],
        "group_size": [128],
        "dtype": [torch.float16],
        "quant_dtype": [torch.float8_e4m3fn],
        "seed": [0],
        "device": [XPU_DEVICES[0]],
    },
    "test_silu_and_mul_per_block_quant_shapes": {
        "device": [XPU_DEVICES[0]],
    },
    "test_silu_and_mul_per_block_quant_swiglu_limit": {
        "num_tokens, hidden_size": [(83, 1024)],
        "quant_dtype, group_size, scale_ue8m0": [
            (torch.float8_e4m3fn, 32, True),
            (torch.float8_e4m3fn, 128, False),
        ],
        "is_scale_transposed": [False],
        "dtype": [torch.bfloat16],
        "swiglu_limit": [3.3],
        "device": [XPU_DEVICES[0]],
    },
    "test_silu_and_mul_per_block_quant_edge_cases": {
        "hidden_size": [1024],
        "batch_size": [1],
        "device": [XPU_DEVICES[0]],
    },
}


def _per_token_group_quant(
    x: torch.Tensor,
    group_size: int,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token-group quantization reference (pure PyTorch, fp8 or int8)."""
    x_f32 = x.float()
    num_groups = x.shape[-1] // group_size
    grouped = x_f32.view(*x.shape[:-1], num_groups, group_size)
    group_max = grouped.abs().amax(dim=-1)

    if quant_dtype == torch.float8_e4m3fn:
        fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
        scale = group_max.clamp(min=torch.finfo(torch.float32).tiny) / fp8_max
        x_q = (grouped / scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max)
    else:  # int8
        scale = group_max.clamp(min=1e-10) / 127.0
        x_q = (grouped / scale.unsqueeze(-1)).round().clamp(-128, 127)

    return x_q.view_as(x).to(quant_dtype), scale.to(torch.float32)


def ref_silu_and_mul_per_block_quant(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = x.shape[-1] // 2
    gate, up = x.split(hidden, dim=-1)
    silu_out = (F.silu(gate) * up).contiguous()
    return _per_token_group_quant(silu_out, group_size, quant_dtype)


@pytest.mark.parametrize("num_tokens, hidden_size", NUM_TOKENS_HIDDEN_SIZES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("is_scale_transposed", IS_SCALE_TRANSPOSED)
@pytest.mark.parametrize("quant_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_per_block_quant(
    num_tokens: int,
    hidden_size: int,
    group_size: int,
    is_scale_transposed: bool,
    quant_dtype: torch.dtype,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    if hidden_size % group_size != 0:
        return

    seed_everything(seed)
    torch.set_default_device(device)
    scale = 1 / hidden_size
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype) * scale

    ref_out, ref_scales = ref_silu_and_mul_per_block_quant(
        x, quant_dtype, group_size)
    ops_out, ops_scales = ops.silu_and_mul_per_block_quant(
        x, group_size, quant_dtype, None, is_scale_transposed)
    ops_scales_row = ops_scales.contiguous()

    assert ref_out.dtype == quant_dtype
    assert ops_out.dtype == quant_dtype
    assert ops_scales.dtype == torch.float32
    assert not torch.isnan(ops_out.float()).any(), "output contains NaN"
    assert not torch.isinf(ops_out.float()).any(), "output contains Inf"
    assert not torch.isnan(ops_scales_row).any(), "scales contain NaN"
    assert not torch.isinf(ops_scales_row).any(), "scales contain Inf"

    torch.testing.assert_close(ref_scales,
                               ops_scales_row,
                               rtol=1e-5,
                               atol=1e-5)

    ref_scales_expanded = ref_scales.repeat_interleave(group_size, dim=1)
    ops_scales_expanded = ops_scales_row.repeat_interleave(group_size, dim=1)
    ref_deq = ref_out.float() * ref_scales_expanded
    ops_deq = ops_out.float() * ops_scales_expanded
    torch.testing.assert_close(ref_deq, ops_deq, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("num_tokens", [128])
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_per_block_quant_shapes(
    dtype: torch.dtype,
    hidden_size: int,
    num_tokens: int,
    group_size: int,
    device: str,
) -> None:
    torch.set_default_device(device)
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype)
    num_groups = hidden_size // group_size

    out, scales = ops.silu_and_mul_per_block_quant(x,
                                                   group_size,
                                                   torch.float8_e4m3fn,
                                                   is_scale_transposed=False)
    assert out.shape == (num_tokens, hidden_size)
    assert scales.shape == (num_tokens, num_groups)

    out, scales_t = ops.silu_and_mul_per_block_quant(x,
                                                     group_size,
                                                     torch.float8_e4m3fn,
                                                     is_scale_transposed=True)
    assert out.shape == (num_tokens, hidden_size)
    assert scales_t.shape == (num_tokens, num_groups)


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("batch_size", [1, 16, 256])
@pytest.mark.parametrize("hidden_size", [1024, 5120, 14336])
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_per_block_quant_edge_cases(
    dtype: torch.dtype,
    batch_size: int,
    hidden_size: int,
    device: str,
) -> None:
    torch.set_default_device(device)
    x = torch.randn(batch_size, hidden_size * 2, dtype=dtype)

    out, scales = ops.silu_and_mul_per_block_quant(x,
                                                   128,
                                                   torch.float8_e4m3fn,
                                                   is_scale_transposed=False)

    assert out.shape == (batch_size, hidden_size)
    assert out.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.float32
    assert not torch.isnan(out.float()).any()
    assert not torch.isnan(scales).any()
    assert not torch.isinf(scales).any()


def ref_silu_and_mul_mxfp8_quant(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 reference in fp32, on the CPU for an exact fp8 cast."""
    hidden = x.shape[-1] // 2
    xf = x.float().cpu()
    gate, up = xf[:, :hidden], xf[:, hidden:]
    res = gate / (1.0 + torch.exp(-gate)) * up
    grouped = res.view(res.shape[0], -1, 32)
    fp8_max = torch.finfo(quant_dtype).max
    min_scale = 1.0 / (fp8_max * 512.0)
    y = (grouped.abs().amax(dim=-1) / fp8_max).clamp(min=min_scale)
    scale = torch.exp2(torch.ceil(torch.log2(y.double()))).float()
    q = (grouped / scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max)
    return q.view_as(res).to(quant_dtype), scale


@pytest.mark.parametrize("num_tokens, hidden_size", [(1, 32), (1, 5120),
                                                     (7, 96), (64, 2048),
                                                     (300, 7168)])
@pytest.mark.parametrize("is_scale_transposed", IS_SCALE_TRANSPOSED)
@pytest.mark.parametrize("quant_dtype",
                         [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_mxfp8_quant(
    num_tokens: int,
    hidden_size: int,
    is_scale_transposed: bool,
    quant_dtype: torch.dtype,
    dtype: torch.dtype,
    device: str,
) -> None:
    """group_size 32 with scale_ue8m0: the MXFP8 path."""
    seed_everything(0)
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype, device=device)
    x[:, ::97] *= 64  # outliers, so that groups get different scales

    num_groups = hidden_size // 32
    out = torch.empty(num_tokens,
                      hidden_size,
                      dtype=quant_dtype,
                      device=device)
    if is_scale_transposed:
        scales = torch.empty(num_groups, num_tokens, device=device).t()
    else:
        scales = torch.empty(num_tokens, num_groups, device=device)
    torch.ops._C.silu_and_mul_per_block_quant(out, x, scales, 32, None,
                                              is_scale_transposed, True)

    ref_out, ref_scales = ref_silu_and_mul_mxfp8_quant(x, quant_dtype)
    # exp differs from the CPU's in the last bit, which can flip an fp8
    # rounding but not a scale.
    torch.testing.assert_close(scales.cpu(), ref_scales, rtol=0, atol=0)
    torch.testing.assert_close(out.cpu().float(),
                               ref_out.float(),
                               rtol=0.13,
                               atol=0)
    mismatch = (out.cpu().view(torch.uint8)
                != ref_out.view(torch.uint8)).float().mean().item()
    assert mismatch < 1e-3, f"{mismatch:.2%} of fp8 values differ"


def _eager_swiglu_limit(x: torch.Tensor, limit: float) -> torch.Tensor:
    x = x.clone()
    hidden = x.shape[-1] // 2
    x[:, :hidden].clamp_(max=limit)
    x[:, hidden:].clamp_(min=-limit, max=limit)
    return x


def _run_per_block_quant(x, group_size, quant_dtype, scale_ue8m0,
                         is_scale_transposed, swiglu_limit):
    num_tokens, hidden = x.shape[0], x.shape[-1] // 2
    out = torch.empty((num_tokens, hidden), dtype=quant_dtype)
    num_groups = hidden // group_size
    if is_scale_transposed:
        scales = torch.empty((num_groups, num_tokens), dtype=torch.float32).t()
    else:
        scales = torch.empty((num_tokens, num_groups), dtype=torch.float32)
    torch.ops._C.silu_and_mul_per_block_quant(out, x, scales, group_size, None,
                                              is_scale_transposed,
                                              scale_ue8m0, None, swiglu_limit)
    return out, scales


# group_size 32 + ue8m0 + fp8 takes the dedicated MXFP8 kernel, the rest the
# generic per-block kernel. 0.1 and 3.3 are not representable in fp16/bf16,
# and 1e-8 rounds to 0 in fp16, which must still clamp (to 0).
@pytest.mark.parametrize("num_tokens, hidden_size", [(1, 256), (83, 1024),
                                                     (128, 5120)])
@pytest.mark.parametrize("quant_dtype, group_size, scale_ue8m0", [
    (torch.float8_e4m3fn, 32, True),
    (torch.float8_e4m3fn, 128, False),
    (torch.float8_e4m3fn, 128, True),
    (torch.int8, 128, False),
])
@pytest.mark.parametrize("is_scale_transposed", IS_SCALE_TRANSPOSED)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("swiglu_limit", [1e-8, 0.1, 1.0, 3.3, 7.0, 10.0])
@pytest.mark.parametrize("device", XPU_DEVICES[:1])
@torch.inference_mode()
def test_silu_and_mul_per_block_quant_swiglu_limit(
    num_tokens: int,
    hidden_size: int,
    quant_dtype: torch.dtype,
    group_size: int,
    scale_ue8m0: bool,
    is_scale_transposed: bool,
    dtype: torch.dtype,
    swiglu_limit: float,
    device: str,
) -> None:
    """The in-kernel SwiGLU limit is bitwise identical to the eager clamp."""
    seed_everything(0)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype) * 8.0
    ref_out, ref_scales = _run_per_block_quant(
        _eager_swiglu_limit(x, swiglu_limit), group_size, quant_dtype,
        scale_ue8m0, is_scale_transposed, None)
    out, scales = _run_per_block_quant(x, group_size, quant_dtype,
                                       scale_ue8m0, is_scale_transposed,
                                       swiglu_limit)
    torch.testing.assert_close(scales, ref_scales, atol=0, rtol=0)
    torch.testing.assert_close(out.view(torch.uint8),
                               ref_out.view(torch.uint8),
                               atol=0,
                               rtol=0)


@pytest.mark.parametrize("swiglu_limit", [None, 0.0])
@torch.inference_mode()
def test_silu_and_mul_per_block_quant_swiglu_limit_disabled(swiglu_limit):
    seed_everything(0)
    torch.set_default_device(XPU_DEVICES[0])
    x = torch.randn(16, 512, dtype=torch.bfloat16) * 8.0
    ref_out, ref_scales = _run_per_block_quant(x, 128, torch.float8_e4m3fn,
                                               False, False, None)
    out, scales = _run_per_block_quant(x, 128, torch.float8_e4m3fn, False,
                                       False, swiglu_limit)
    torch.testing.assert_close(scales, ref_scales, atol=0, rtol=0)
    torch.testing.assert_close(out.view(torch.uint8),
                               ref_out.view(torch.uint8),
                               atol=0,
                               rtol=0)


@pytest.mark.parametrize("swiglu_limit", [-1.0, float("nan")])
@pytest.mark.parametrize("num_tokens", [0, 4])
@torch.inference_mode()
def test_silu_and_mul_per_block_quant_swiglu_limit_invalid(
        swiglu_limit, num_tokens):
    torch.set_default_device(XPU_DEVICES[0])
    x = torch.randn(num_tokens, 256, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="swiglu_limit"):
        _run_per_block_quant(x, 128, torch.float8_e4m3fn, False, False,
                             swiglu_limit)
