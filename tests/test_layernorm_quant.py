# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the fused RMS-norm + static FP8 quantization SYCL kernels:
  - rms_norm_static_fp8_quant
  - fused_add_rms_norm_static_fp8_quant
"""

import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401

DTYPES = [torch.half, torch.bfloat16]
FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
NUM_TOKENS = [1, 7, 83, 256]
HIDDEN_SIZES = [8, 64, 768, 5120, 8192]
SEEDS = [0]
EPSILONS = [1e-5, 1e-6]
XPU_DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

# override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [7],
        "hidden_size": [64],
    },
}

# ---------------------------------------------------------------------------
# Reference implementations (pure PyTorch on CPU/GPU, float32 arithmetic)
# ---------------------------------------------------------------------------


def ref_rms_norm_static_fp8_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    fp8_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Reference: RMS-norm → multiply by weight → static FP8 quant.
    """
    x = input.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + epsilon)
    x_normed = x_normed.to(input.dtype) * weight  # match kernel cast order

    # static quantization: out = clamp(x_normed / scale, -fp8_max, fp8_max)
    fp8_max = torch.finfo(fp8_dtype).max
    inv_scale = 1.0 / scale.float()
    out_f = x_normed.float() * inv_scale
    out_f = out_f.clamp(-fp8_max, fp8_max)
    return out_f.to(fp8_dtype)


def ref_fused_add_rms_norm_static_fp8_quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    fp8_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference: residual += input → RMS-norm(residual) → weight → FP8 quant.
    Returns (fp8_out, updated_residual).
    """
    residual = residual + input.to(residual.dtype)
    x = residual.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + epsilon)
    x_normed = x_normed.to(input.dtype) * weight

    fp8_max = torch.finfo(fp8_dtype).max
    inv_scale = 1.0 / scale.float()
    out_f = x_normed.float() * inv_scale
    out_f = out_f.clamp(-fp8_max, fp8_max)
    return out_f.to(fp8_dtype), residual


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_rms_norm_static_fp8_quant(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    fp8_dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    # Create inputs
    scale_val = 0.1 + torch.rand(1, device=device, dtype=torch.float32)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    weight.data.normal_(mean=1.0, std=0.1)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    epsilon = 1e-5

    # Reference
    ref_out = ref_rms_norm_static_fp8_quant(x, weight, scale_val, epsilon,
                                            fp8_dtype)

    # Kernel under test
    out = torch.empty(num_tokens, hidden_size, dtype=fp8_dtype, device=device)
    torch.ops._C.rms_norm_static_fp8_quant(out, x, weight, scale_val, epsilon)

    # Compare: fp8 has limited precision, so we compare in float
    torch.testing.assert_close(
        out.to(torch.float32),
        ref_out.to(torch.float32),
        atol=1.0,  # fp8 quantization can have up to 1 ULP difference
        rtol=0.1,
    )

    # Additional: check the values are valid fp8 (not NaN)
    assert not torch.isnan(out.to(torch.float32)).any(), \
        "Output contains NaN values"


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_fused_add_rms_norm_static_fp8_quant(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    fp8_dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    # Create inputs
    scale_val = 0.1 + torch.rand(1, device=device, dtype=torch.float32)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    weight.data.normal_(mean=1.0, std=0.1)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    epsilon = 1e-5

    # Clone residual for reference (kernel modifies it in-place)
    residual_ref = residual.clone()
    residual_kern = residual.clone()

    # Reference
    ref_out, ref_residual = ref_fused_add_rms_norm_static_fp8_quant(
        x, residual_ref, weight, scale_val, epsilon, fp8_dtype)

    # Kernel under test
    out = torch.empty(num_tokens, hidden_size, dtype=fp8_dtype, device=device)
    torch.ops._C.fused_add_rms_norm_static_fp8_quant(out, x, residual_kern,
                                                     weight, scale_val,
                                                     epsilon)

    # Check quantized output
    torch.testing.assert_close(
        out.to(torch.float32),
        ref_out.to(torch.float32),
        atol=1.0,
        rtol=0.1,
    )

    # Check that residual was updated correctly (residual = input + residual)
    torch.testing.assert_close(
        residual_kern,
        ref_residual,
        atol=1e-3,
        rtol=1e-3,
    )

    # No NaN in output
    assert not torch.isnan(out.to(torch.float32)).any(), \
        "Output contains NaN values"


@pytest.mark.parametrize("num_tokens", [7, 32, 4096])
@pytest.mark.parametrize("hidden_size", [64, 768])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_rms_norm_static_fp8_quant_strided_input(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: str,
) -> None:
    """Test with non-contiguous (strided) input tensors."""
    torch.manual_seed(0)
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    fp8_dtype = torch.float8_e4m3fn
    scale_val = torch.tensor([0.5], device=device, dtype=torch.float32)
    weight = torch.ones(hidden_size, dtype=dtype, device=device)
    epsilon = 1e-5

    # Create strided input: allocate wider tensor, take a slice
    last_dim = 2 * hidden_size
    x_wide = torch.randn(num_tokens, last_dim, dtype=dtype, device=device)
    x = x_wide[..., :hidden_size]
    assert not x.is_contiguous(), "Expected non-contiguous input"

    # Reference uses contiguous version
    ref_out = ref_rms_norm_static_fp8_quant(x.contiguous(), weight, scale_val,
                                            epsilon, fp8_dtype)

    out = torch.empty(num_tokens, hidden_size, dtype=fp8_dtype, device=device)
    torch.ops._C.rms_norm_static_fp8_quant(out, x, weight, scale_val, epsilon)

    torch.testing.assert_close(
        out.to(torch.float32),
        ref_out.to(torch.float32),
        atol=1.0,
        rtol=0.1,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_rms_norm_static_fp8_quant_opcheck(
    dtype: torch.dtype,
    fp8_dtype: torch.dtype,
    device: str,
) -> None:
    """opcheck: schema / autograd / faketensor compliance."""
    from tests.utils import opcheck

    torch.manual_seed(0)
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    hidden_size = 64
    num_tokens = 8
    weight = torch.ones(hidden_size, dtype=dtype, device=device)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    scale_val = torch.tensor([1.0], device=device, dtype=torch.float32)
    out = torch.empty(num_tokens, hidden_size, dtype=fp8_dtype, device=device)
    epsilon = 1e-5

    opcheck(
        torch.ops._C.rms_norm_static_fp8_quant,
        (out, x, weight, scale_val, epsilon),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_fused_add_rms_norm_static_fp8_quant_opcheck(
    dtype: torch.dtype,
    fp8_dtype: torch.dtype,
    device: str,
) -> None:
    """opcheck: schema / autograd / faketensor compliance."""
    from tests.utils import opcheck

    torch.manual_seed(0)
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    hidden_size = 64
    num_tokens = 8
    weight = torch.ones(hidden_size, dtype=dtype, device=device)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    scale_val = torch.tensor([1.0], device=device, dtype=torch.float32)
    out = torch.empty(num_tokens, hidden_size, dtype=fp8_dtype, device=device)
    epsilon = 1e-5

    opcheck(
        torch.ops._C.fused_add_rms_norm_static_fp8_quant,
        (out, x, residual, weight, scale_val, epsilon),
    )
