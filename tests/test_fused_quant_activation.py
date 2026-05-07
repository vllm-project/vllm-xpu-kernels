# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

import tests.register_ops as ops
from tests.utils import seed_everything

DTYPES = [torch.half, torch.bfloat16]
FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
NUM_TOKENS = [1, 7, 83, 512]
HIDDEN_SIZES = [16, 128, 512, 4096]
SEEDS = [0]
XPU_DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [1],
        "HIDDEN_SIZES": [16],
    },
}


def ref_silu_and_mul_quant(
    x: torch.Tensor,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference implementation: SiLU+Mul then static FP8 quant."""
    d = x.shape[-1] // 2
    silu_mul_out = F.silu(x[..., :d]) * x[..., d:]

    fp8_max = torch.finfo(fp8_dtype).max
    inv_scale = 1.0 / scale.item()
    result = (silu_mul_out.float() * inv_scale).clamp(-fp8_max, fp8_max)
    return result.to(fp8_dtype)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_quant(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    fp8_dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    seed_everything(seed)
    torch.set_default_device(device)

    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype)
    scale = torch.tensor([0.5], dtype=torch.float32, device=device)

    # Reference
    ref_out = ref_silu_and_mul_quant(x, scale, fp8_dtype)

    # Fused kernel
    d = x.shape[-1] // 2
    out = torch.empty(num_tokens, d, dtype=fp8_dtype, device=device)
    ops.silu_and_mul_quant(out, x, scale)

    assert out.dtype == fp8_dtype
    assert out.shape == ref_out.shape
    torch.testing.assert_close(
        out.to(dtype=torch.float32),
        ref_out.to(dtype=torch.float32),
    )


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_quant_vs_separate(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    fp8_dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    seed_everything(seed)
    torch.set_default_device(device)

    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype)
    scale = torch.tensor([0.5], dtype=torch.float32, device=device)

    # Separate ops
    d = x.shape[-1] // 2
    silu_mul_out = torch.empty(num_tokens, d, dtype=dtype, device=device)
    ops.silu_and_mul(silu_mul_out, x)

    separate_out = torch.empty(num_tokens, d, dtype=fp8_dtype, device=device)
    ops.static_scaled_fp8_quant(separate_out, silu_mul_out, scale)

    # Fused kernel
    fused_out = torch.empty(num_tokens, d, dtype=fp8_dtype, device=device)
    ops.silu_and_mul_quant(fused_out, x, scale)

    assert fused_out.dtype == fp8_dtype
    assert fused_out.shape == separate_out.shape
    torch.testing.assert_close(
        fused_out.to(dtype=torch.float32),
        separate_out.to(dtype=torch.float32),
    )
