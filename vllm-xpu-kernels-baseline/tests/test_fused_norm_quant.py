# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# UT for fused RMSNorm + Quantization kernels.
# Adapted from tests/kernels/core/test_fused_quant_layernorm.py (CUDA).

import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401 – registers torch.ops._C.*
from tests.ops.layernorm_op import RMSNorm

DTYPES = [torch.half, torch.bfloat16]
# XPU FP8 default is e4m3fn; INT8 is always tested
QUANT_DTYPES = [torch.float8_e4m3fn, torch.int8]

NUM_TOKENS = [1, 7, 83, 2048]
HIDDEN_SIZES = [64, 128, 1024, 5120]
ADD_RESIDUAL = [False, True]
GROUP_SIZES = [64, 128]
SEEDS = [0]
EPS = 1e-6

XPU_DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [4],
        "HIDDEN_SIZES": [64],
        "GROUP_SIZES": [64],
    },
}


def _ref_rms_norm(
    layer: RMSNorm,
    x: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    x_f32 = x.float()
    if residual is not None:
        residual = residual.clone()
        # Kernel: z = input + residual, stored as orig_dtype, then read back
        z_half = (x_f32 + residual.float()).to(x.dtype)
        residual = z_half.clone()
        x_f32 = z_half.float()
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + layer.variance_epsilon)
    normed = x_f32 * inv_rms * layer.weight.float()
    return normed, residual


def _ref_per_token_quant(
    normed: torch.Tensor,
    quant_dtype: torch.dtype,
    scale_ub: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference dynamic per-token quantization."""
    num_tokens = normed.numel() // normed.shape[-1]
    normed_2d = normed.view(num_tokens, -1).float()

    if quant_dtype == torch.int8:
        absmax = normed_2d.abs().amax(dim=1, keepdim=True)  # [T, 1]
        scale = torch.where(absmax > 0, absmax / 127.0,
                            torch.ones_like(absmax))
        q = (normed_2d / scale).round().clamp(-128, 127).to(torch.int8)
        return q, scale.squeeze(1)
    else:
        # FP8 e4m3fn
        fp8_max = torch.finfo(quant_dtype).max
        absmax = normed_2d.abs().amax(dim=1, keepdim=True)  # [T, 1]
        if scale_ub is not None:
            absmax = torch.min(absmax, scale_ub.float())
        min_sf = 1.0 / (fp8_max * 512.0)
        scale = torch.clamp(absmax / fp8_max, min=min_sf)  # [T, 1]
        inv_scale = 1.0 / scale
        q = (normed_2d * inv_scale).clamp(-fp8_max, fp8_max).to(quant_dtype)
        return q, scale.squeeze(1)


def _ref_per_group_quant(
    normed: torch.Tensor,
    group_size: int,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference dynamic per-column-group quantization."""
    num_tokens = normed.numel() // normed.shape[-1]
    hidden = normed.shape[-1]
    num_groups = hidden // group_size
    normed_2d = normed.view(num_tokens, hidden).float()

    q_out = torch.empty_like(normed_2d, dtype=quant_dtype)
    scales = torch.empty(num_tokens,
                         num_groups,
                         dtype=torch.float32,
                         device=normed.device)

    fp8_max = torch.finfo(
        quant_dtype).max if quant_dtype != torch.int8 else 127.0
    min_sf = 1.0 / (fp8_max * 512.0) if quant_dtype != torch.int8 else 0.0

    for g in range(num_groups):
        chunk = normed_2d[:, g * group_size:(g + 1) * group_size]  # [T, G]
        absmax = chunk.abs().amax(dim=1, keepdim=True)  # [T, 1]

        if quant_dtype == torch.int8:
            scale = torch.where(absmax > 0, absmax / 127.0,
                                torch.ones_like(absmax))
            q = (chunk / scale).round().clamp(-128, 127).to(torch.int8)
        else:
            scale = torch.clamp(absmax / fp8_max, min=min_sf)
            inv_scale = 1.0 / scale
            q = (chunk * inv_scale).clamp(-fp8_max, fp8_max).to(quant_dtype)

        q_out[:, g * group_size:(g + 1) * group_size] = q
        scales[:, g] = scale.squeeze(1)

    return q_out, scales


def _ops_per_token_quant(
    weight: torch.Tensor,
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    residual: torch.Tensor | None,
    scale_ub: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    x = x.contiguous()
    out = torch.empty_like(x, dtype=quant_dtype)
    num_tokens = x.numel() // x.shape[-1]
    scales = torch.empty(num_tokens, dtype=torch.float32, device=x.device)
    if residual is not None:
        residual = residual.clone().contiguous()
    torch.ops._C.rms_norm_dynamic_per_token_quant(out, x, weight, scales, EPS,
                                                  scale_ub, residual)
    return out, scales, residual


def _ops_per_group_quant(
    weight: torch.Tensor,
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    group_size: int,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    x = x.contiguous()
    out = torch.empty_like(x, dtype=quant_dtype)
    num_tokens = x.numel() // x.shape[-1]
    num_groups = x.shape[-1] // group_size
    scales = torch.empty(num_tokens,
                         num_groups,
                         dtype=torch.float32,
                         device=x.device)
    if residual is not None:
        residual = residual.clone().contiguous()
    torch.ops._C.rms_norm_per_block_quant(
        out,
        x,
        weight,
        scales,
        EPS,
        None,  # scale_ub (not used for per-block)
        residual,
        group_size,
        False,  # is_scale_transposed
    )
    return out, scales, residual


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_rms_norm_dynamic_per_token_quant(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    quant_dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    layer = RMSNorm(hidden_size, eps=EPS).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)

    scale = 1.0 / hidden_size
    x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
    residual = torch.randn_like(x) * scale if add_residual else None

    # Reference: native RMSNorm + per-token quant
    ref_normed, ref_residual = _ref_rms_norm(layer, x, residual)
    ref_q, ref_scales = _ref_per_token_quant(ref_normed, quant_dtype)

    # Kernel
    ops_q, ops_scales, ops_residual = _ops_per_token_quant(
        layer.weight.data, x, quant_dtype, residual, None)

    assert ops_q.dtype == quant_dtype
    assert ops_scales.dtype == torch.float32

    if quant_dtype == torch.int8:
        torch.testing.assert_close(ref_scales,
                                   ops_scales,
                                   atol=1e-6,
                                   rtol=1e-6)
        torch.testing.assert_close(ref_q, ops_q, atol=1, rtol=0)
    else:
        # FP8: scales should be close; compare quantized values or dequantized
        torch.testing.assert_close(ref_scales,
                                   ops_scales,
                                   atol=1e-5,
                                   rtol=1e-5)
        ref_qf = ref_q.float()
        ops_qf = ops_q.float()
        if not torch.allclose(ref_qf, ops_qf, atol=1e-6):
            # Fallback: dequantize both with their own scales and compare.
            # NOTE: It is possible that some future test cases trigger this
            # max diff due to precision issues. If such an error is
            # encountered, it's recommended to inspect the differences between
            # all corresponding elements from each tensor (e.g. by looping over
            # them) and checking how many the max diff error shows up on (just
            # a few bad elements should still be considered acceptable).
            ref_deq = ref_qf * ref_scales.view(-1, 1)
            ops_deq = ops_qf * ops_scales.view(-1, 1)
            torch.testing.assert_close(ref_deq, ops_deq, atol=0.2, rtol=0.15)

    if add_residual:
        torch.testing.assert_close(ref_residual,
                                   ops_residual,
                                   atol=1e-2,
                                   rtol=1e-2)


@pytest.mark.parametrize("num_tokens", [1, 7, 83, 2048])
@pytest.mark.parametrize("hidden_size", [128, 1024, 5120])
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_dtype", [torch.float8_e4m3fn, torch.int8])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_rms_norm_per_block_quant(
    num_tokens: int,
    hidden_size: int,
    group_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    quant_dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    if hidden_size % group_size != 0:
        pytest.skip(f"hidden_size {hidden_size} not divisible by \
            group_size {group_size}")

    torch.manual_seed(seed)
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    layer = RMSNorm(hidden_size, eps=EPS).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)

    scale = 1.0 / hidden_size
    x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
    residual = torch.randn_like(x) * scale if add_residual else None

    # Reference: native RMSNorm + per-group quant
    ref_normed, ref_residual = _ref_rms_norm(layer, x, residual)
    ref_q, ref_scales = _ref_per_group_quant(ref_normed, group_size,
                                             quant_dtype)

    # Kernel
    ops_q, ops_scales, ops_residual = _ops_per_group_quant(
        layer.weight.data, x, quant_dtype, group_size, residual)

    assert ops_q.dtype == quant_dtype
    assert ops_scales.dtype == torch.float32
    assert ops_scales.shape == (num_tokens, hidden_size // group_size)

    torch.testing.assert_close(ref_scales, ops_scales, atol=1e-4, rtol=1e-4)

    if quant_dtype == torch.int8:
        torch.testing.assert_close(ref_q, ops_q, atol=1, rtol=0)
    else:
        num_groups = hidden_size // group_size
        ref_qf = ref_q.float().view(num_tokens, num_groups, group_size)
        ops_qf = ops_q.float().view(num_tokens, num_groups, group_size)
        if not torch.allclose(ref_qf, ops_qf, atol=1e-6):
            ref_scales_e = ref_scales.unsqueeze(-1)
            ops_scales_e = ops_scales.unsqueeze(-1)
            ref_deq = ref_qf * ref_scales_e
            ops_deq = ops_qf * ops_scales_e
            torch.testing.assert_close(ref_deq, ops_deq, atol=0.2, rtol=0.15)

    if add_residual:
        torch.testing.assert_close(ref_residual,
                                   ops_residual,
                                   atol=1e-2,
                                   rtol=1e-2)


def _ops_rms_norm_static_fp8_quant(
    weight: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    torch.ops._C.rms_norm_static_fp8_quant(out, x, weight, scale, EPS)
    return out


def _ops_fused_add_rms_norm_static_fp8_quant(
    weight: torch.Tensor,
    x: torch.Tensor,
    residual: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(x.shape[-1], dtype=torch.float8_e4m3fn,
                      device=x.device).expand_as(x).contiguous()
    residual = residual.clone().contiguous()
    torch.ops._C.fused_add_rms_norm_static_fp8_quant(out, x, residual, weight,
                                                     scale, EPS)
    return out, residual


def _ref_static_fp8_quant(
    normed: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Reference static FP8 quantization with pre-computed scale."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    inv_scale = 1.0 / scale.float()
    q = (normed.float() * inv_scale).clamp(-fp8_max, fp8_max)
    return q.to(torch.float8_e4m3fn)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
@torch.inference_mode()
def test_rms_norm_static_fp8_quant(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    strided_input: bool,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    layer = RMSNorm(hidden_size, eps=EPS).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)

    scale_val = 1.0
    quant_scale = torch.tensor(scale_val, dtype=torch.float32)
    input_scale = 1.0 / hidden_size
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x = torch.randn(num_tokens, last_dim, dtype=dtype) * input_scale
    x = x[..., :hidden_size]
    if num_tokens > 1:
        assert x.is_contiguous() != strided_input

    # Reference: native RMSNorm then static FP8 quant
    ref_normed, _ = _ref_rms_norm(layer, x, None)
    ref_q = _ref_static_fp8_quant(ref_normed, quant_scale)

    # Kernel
    ops_q = _ops_rms_norm_static_fp8_quant(layer.weight.data, x, quant_scale)

    assert ops_q.dtype == torch.float8_e4m3fn
    # FP8 has coarse quantization; boundary elements may differ by up to 1 LSB.
    ref_qf = ref_q.float()
    ops_qf = ops_q.float()
    if not torch.allclose(ref_qf, ops_qf, atol=1e-6):
        ref_deq = ref_qf * quant_scale
        ops_deq = ops_qf * quant_scale
        torch.testing.assert_close(ref_deq, ops_deq, atol=0.2, rtol=0.15)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
@torch.inference_mode()
def test_fused_add_rms_norm_static_fp8_quant(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    strided_input: bool,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    layer = RMSNorm(hidden_size, eps=EPS).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)

    scale_val = 1.0
    quant_scale = torch.tensor(scale_val, dtype=torch.float32)
    input_scale = 1.0 / hidden_size
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x = torch.randn(num_tokens, last_dim, dtype=dtype) * input_scale
    x = x[..., :hidden_size]
    if num_tokens > 1:
        assert x.is_contiguous() != strided_input
    residual = torch.randn(num_tokens, hidden_size, dtype=dtype) * input_scale

    # Reference: native RMSNorm (with residual) then static FP8 quant
    ref_normed, ref_residual = _ref_rms_norm(layer, x, residual)
    ref_q = _ref_static_fp8_quant(ref_normed, quant_scale)

    # Kernel
    ops_q, ops_residual = _ops_fused_add_rms_norm_static_fp8_quant(
        layer.weight.data, x, residual, quant_scale)

    assert ops_q.dtype == torch.float8_e4m3fn
    # Compare quantized values first, then fall back to dequantized comparison.
    ref_qf = ref_q.float()
    ops_qf = ops_q.float()
    if not torch.allclose(ref_qf, ops_qf, atol=1e-6):
        ref_deq = ref_qf * quant_scale
        ops_deq = ops_qf * quant_scale
        torch.testing.assert_close(ref_deq, ops_deq, atol=0.2, rtol=0.15)
    torch.testing.assert_close(ops_residual,
                               ref_residual,
                               atol=1e-2,
                               rtol=1e-2)
