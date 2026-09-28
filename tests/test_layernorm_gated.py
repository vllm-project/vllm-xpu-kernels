# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.ops.layernorm_op import RMSNormGated
from tests.utils import opcheck

DTYPES = [torch.half, torch.bfloat16]
NUM_TOKENS = [1, 7, 83, 4096]
# 128 is the KDA/GDN head_dim (hits the multi-row path); the odd sizes exercise
# the scalar fallback and the single-row vectorized path.
HIDDEN_SIZES = [8, 128, 769, 5120, 8199]
ACTIVATIONS = ["sigmoid", "swish"]
HAS_WEIGHT = [False, True]
SEEDS = [0]
XPU_DEVICES = [
    f"xpu:{i}" for i in range(min(2, torch.xpu.device_count()))
] if torch.xpu.is_available() else []

# override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [7],
        "hidden_size": [128],
    },
}


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("has_weight", HAS_WEIGHT)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_rms_norm_gated(
    num_tokens: int,
    hidden_size: int,
    activation: str,
    has_weight: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.manual_seed(seed)

    layer = RMSNormGated(hidden_size,
                         activation=activation,
                         has_weight=has_weight).to(device=device, dtype=dtype)
    if has_weight:
        layer.weight.data.normal_(mean=1.0, std=0.1)

    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device) * scale
    gate = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    ref_out = layer.forward_native(x, gate)
    out = layer(x, gate)
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

    weight = layer.weight.data if has_weight else None
    opcheck(
        torch.ops._C.fused_rms_norm_gated,
        (out, x, gate, weight, layer.variance_epsilon, activation),
    )


@pytest.mark.parametrize("num_tokens", [1, 5, 32])
@pytest.mark.parametrize("num_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_rms_norm_gated_per_head(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    activation: str,
    dtype: torch.dtype,
    device: str,
) -> None:
    """Per-(token, head) normalization with a head_dim weight, i.e. how the
    linear-attention output norm is applied."""
    torch.manual_seed(0)
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    layer = RMSNormGated(head_dim, activation=activation).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)

    x = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype)
    gate = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype)

    ref_out = layer.forward_native(x, gate)
    out = layer(x, gate)
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_rms_norm_gated_strided(activation: str, dtype: torch.dtype) -> None:
    """The gate typically arrives as a slice of a fused projection, so both
    input and gate may be non-contiguous views."""
    torch.manual_seed(0)
    torch.set_default_device("xpu")

    num_tokens, num_heads, head_dim = 17, 8, 128
    layer = RMSNormGated(head_dim, activation=activation).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)

    fused = torch.randn(num_tokens, 2 * num_heads * head_dim, dtype=dtype)
    x, gate = fused.split([num_heads * head_dim] * 2, dim=-1)
    x = x.view(num_tokens, num_heads, head_dim)
    gate = gate.view(num_tokens, num_heads, head_dim)
    assert not x.is_contiguous()

    ref_out = layer.forward_native(x, gate)
    out = layer(x, gate)
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)


@torch.inference_mode()
def test_rms_norm_gated_rejects_unknown_activation() -> None:
    torch.set_default_device("xpu")
    x = torch.randn(4, 128, dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    out = torch.empty_like(x)
    with pytest.raises(RuntimeError, match="unsupported activation"):
        torch.ops._C.fused_rms_norm_gated(out, x, gate, None, 1e-5, "relu")


@pytest.mark.parametrize("shape", [(), (1, 0)])
@pytest.mark.parametrize("has_weight", HAS_WEIGHT)
@torch.inference_mode()
def test_rms_norm_gated_rejects_invalid_dimensions(shape, has_weight):
    x = torch.empty(shape, device="xpu", dtype=torch.bfloat16)
    gate = torch.empty_like(x)
    out = torch.empty_like(x)
    weight = torch.empty(0, device="xpu", dtype=x.dtype) if has_weight else None
    message = "at least one dimension" if not shape else "hidden_size"
    with pytest.raises(RuntimeError, match=message):
        torch.ops._C.fused_rms_norm_gated(out, x, gate, weight, 1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("has_weight", HAS_WEIGHT)
@torch.inference_mode()
def test_rms_norm_gated_empty_rows(dtype, activation, has_weight):
    x = torch.empty(0, 128, device="xpu", dtype=dtype)
    gate = torch.empty_like(x)
    out = torch.empty_like(x)
    weight = torch.ones(128, device="xpu", dtype=dtype) if has_weight else None
    torch.ops._C.fused_rms_norm_gated(out, x, gate, weight, 1e-5, activation)
    torch.xpu.synchronize()
    assert out.shape == x.shape
    assert out.numel() == 0


@pytest.mark.parametrize("operand", ["input", "gate", "out", "weight"])
@pytest.mark.parametrize("other_device", ["cpu", "xpu:1"])
@torch.inference_mode()
def test_rms_norm_gated_rejects_mixed_devices(operand, other_device):
    if other_device == "xpu:1" and torch.xpu.device_count() < 2:
        pytest.skip("requires two XPU devices")
    tensors = {
        name: torch.ones(2, 128, device="xpu:0", dtype=torch.bfloat16)
        for name in ("input", "gate", "out")
    }
    tensors["weight"] = torch.ones(128, device="xpu:0", dtype=torch.bfloat16)
    tensors[operand] = tensors[operand].to(other_device)
    message = ("input must be on XPU" if operand == "input"
               and other_device == "cpu" else "same device")
    with pytest.raises(RuntimeError, match=message):
        torch.ops._C.fused_rms_norm_gated(
            tensors["out"], tensors["input"], tensors["gate"],
            tensors["weight"], 1e-5)
