# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.allclose_default import get_default_atol, get_default_rtol
from tests.ops.activation_op import (FastGELU, GeluAndMul, MulAndSilu, NewGELU,
                                     QuickGELU, SiluAndMul)
from tests.utils import opcheck, seed_everything

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048]  # Arbitrary values for testing
D = [512, 13824]  # Arbitrary values for testing
SEEDS = [0]
XPU_DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

#override pytest parameters when 'USE_SIMULATOR' eq 1
SIMULATOR_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [1],
        "d": [128],
    },
}


@pytest.mark.parametrize("activation",
                         ["silu_and_mul", "mul_and_silu", "gelu", "gelu_tanh"])
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
    out = layer(x)
    ref_out = layer.forward_native(x)

    torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=1e-3)

    d = x.shape[-1] // 2
    output_shape = (x.shape[:-1] + (d, ))
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    opcheck(fn, (out, x))


@pytest.mark.parametrize("activation", [(FastGELU, torch.ops._C.gelu_fast),
                                        (NewGELU, torch.ops._C.gelu_new),
                                        (QuickGELU, torch.ops._C.gelu_quick)])
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
