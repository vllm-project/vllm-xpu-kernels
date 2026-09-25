# SPDX-License-Identifier: Apache-2.0
import random

import pytest
import torch

from tests.allclose_default import get_default_atol, get_default_rtol
from tests.ops.activation_op import (FastGELU, FatreluAndMul, GeluAndMul,
                                     MulAndSilu, NewGELU, QuickGELU)
from tests.ops.activation_op import Relu2NoMul
from tests.ops.activation_op import Relu2NoMul as Relu2
from tests.ops.activation_op import SiluAndMul
from tests.utils import opcheck, seed_everything

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048]  # Arbitrary values for testing
D = [512, 1800, 3000, 13824]  # Cover scalar and vectorized kernels
SEEDS = [0]
XPU_DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [1],
        "d": [128],
    },
}


@pytest.mark.parametrize(
    "activation",
    ["silu_and_mul", "mul_and_silu", "gelu", "gelu_tanh", "fatrelu"])
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_act_and_mul(
    activation: str,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    seed_everything(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, 2 * d, dtype=dtype)
    if activation == "silu_and_mul":
        layer = SiluAndMul()
        fn = torch.ops._C.silu_and_mul
    elif activation == "mul_and_silu":
        layer = MulAndSilu()
        fn = torch.ops._C.mul_and_silu
    elif activation == "gelu":
        layer = GeluAndMul(approximate="none")
        fn = torch.ops._C.gelu_and_mul
    elif activation == "gelu_tanh":
        layer = GeluAndMul(approximate="tanh")
        fn = torch.ops._C.gelu_tanh_and_mul
    elif activation == "fatrelu":
        threshold = random.uniform(0, 1)
        layer = FatreluAndMul(threshold)
        fn = torch.ops._C.fatrelu_and_mul
    out = layer(x)
    ref_out = layer.forward_native(x)

    if activation == "fatrelu":
        torch.testing.assert_close(out, ref_out, atol=0.0, rtol=0.0)
    else:
        torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=1e-3)

    d = x.shape[-1] // 2
    output_shape = (x.shape[:-1] + (d, ))
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    if activation == "fatrelu":
        opcheck(fn, (out, x, threshold))
    else:
        opcheck(fn, (out, x))


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("linear_beta", [-1.0, 2.0])
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_situ_and_mul(
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    linear_beta: float,
    device: str,
) -> None:
    seed_everything(0)
    beta = 1.7
    x = torch.randn(num_tokens, 2 * d, dtype=dtype, device=device)
    gate, up = x.float().chunk(2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta > 0:
        up = linear_beta * torch.tanh(up / linear_beta)
    ref_out = (gate * up).to(dtype)
    out = torch.empty_like(ref_out)

    torch.ops._C.situ_and_mul(out, x, beta, linear_beta)

    torch.testing.assert_close(
        out,
        ref_out,
        atol=get_default_atol(out),
        rtol=get_default_rtol(out),
    )
    opcheck(torch.ops._C.situ_and_mul, (out, x, beta, linear_beta))


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("odd_width", "last dimension must be positive and even"),
        ("noncontiguous_input", "input must be contiguous"),
        ("noncontiguous_output", "output must be contiguous"),
        ("wrong_output_shape", "output last dimension must be half"),
        ("wrong_output_dtype", "input and output must have the same dtype"),
    ],
)
@pytest.mark.parametrize("device", XPU_DEVICES)
def test_situ_and_mul_rejects_invalid_tensor_contract(
    case: str,
    error: str,
    device: str,
) -> None:
    x = torch.randn(2, 8, device=device)
    out = torch.empty(2, 4, device=device)
    if case == "odd_width":
        x = torch.randn(2, 7, device=device)
        out = torch.empty(2, 3, device=device)
    elif case == "noncontiguous_input":
        x = torch.randn(2, 16, device=device)[:, ::2]
    elif case == "noncontiguous_output":
        out = torch.empty(2, 8, device=device)[:, ::2]
    elif case == "wrong_output_shape":
        out = torch.empty(2, 3, device=device)
    elif case == "wrong_output_dtype":
        out = out.to(torch.bfloat16)

    with pytest.raises(RuntimeError, match=error):
        torch.ops._C.situ_and_mul(out, x, 1.7, 2.0)


@pytest.mark.parametrize("beta", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("device", XPU_DEVICES)
def test_situ_and_mul_rejects_invalid_beta(
    beta: float,
    device: str,
) -> None:
    x = torch.randn(2, 8, device=device)
    out = torch.empty(2, 4, device=device)

    with pytest.raises(RuntimeError, match="finite and greater than zero"):
        torch.ops._C.situ_and_mul(out, x, beta, 2.0)


@pytest.mark.parametrize(
    "linear_beta", [float("nan"), float("inf"), float("-inf")]
)
@pytest.mark.parametrize("device", XPU_DEVICES)
def test_situ_and_mul_rejects_invalid_linear_beta(
    linear_beta: float,
    device: str,
) -> None:
    x = torch.randn(2, 8, device=device)
    out = torch.empty(2, 4, device=device)

    with pytest.raises(RuntimeError, match="linear_beta must be finite"):
        torch.ops._C.situ_and_mul(out, x, 1.7, linear_beta)


@pytest.mark.parametrize("activation",
                         [(FastGELU, torch.ops._C.gelu_fast),
                          (NewGELU, torch.ops._C.gelu_new),
                          (QuickGELU, torch.ops._C.gelu_quick),
                          (Relu2NoMul, torch.ops._C.relu2_no_mul),
                          (Relu2, torch.ops._C.relu2)])
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_activation(
    activation: type[torch.nn.Module],
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    seed_everything(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, d, dtype=dtype)
    layer = activation[0]()
    fn = activation[1]
    out = layer(x)
    ref_out = layer.forward_native(x)
    torch.testing.assert_close(out,
                               ref_out,
                               atol=get_default_atol(out),
                               rtol=get_default_rtol(out))

    out = torch.empty_like(x)
    opcheck(fn, (out, x))
