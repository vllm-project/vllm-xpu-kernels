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
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

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
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    layer = RMSNormGated(hidden_size,
                         activation=activation,
                         has_weight=has_weight).to(dtype=dtype)
    if has_weight:
        layer.weight.data.normal_(mean=1.0, std=0.1)

    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
    gate = torch.randn(num_tokens, hidden_size, dtype=dtype)

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
