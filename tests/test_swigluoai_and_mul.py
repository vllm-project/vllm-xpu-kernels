# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.ops.swigluoai_and_mul_op import SwigluOAIAndMul
from tests.utils import opcheck, seed_everything

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048]  # Arbitrary values for testing
D = [512, 13824]  # Arbitrary values for testing
SEEDS = [0]
XPU_DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float: 1e-5}
default_rtol = {
    torch.float16: 1e-3,
    torch.bfloat16: 1.6e-2,
    torch.float: 1.3e-6
}


def get_default_atol(output) -> float:
    return default_atol[output.dtype]


def get_default_rtol(output) -> float:
    return default_rtol[output.dtype]


@pytest.mark.parametrize(
    "activation",
    [
        "swigluoai_and_mul",
    ],
)
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

    layer = SwigluOAIAndMul()

    out = layer(x)
    ref_out = layer.forward_native(x)

    rtol = {
        # For fp16, change the relative tolerance from 1e-3 to 2e-3
        torch.float16: 2e-3,
        torch.bfloat16: 2e-2,
        torch.float: 1.3e-6
    }

    def _get_rtol(output) -> float:
        return rtol[output.dtype]

    torch.testing.assert_close(out,
                               ref_out,
                               atol=get_default_atol(out),
                               rtol=_get_rtol(out))

    d = x.shape[-1] // 2
    output_shape = (x.shape[:-1] + (d, ))
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    fn = torch.ops._C.swigluoai_and_mul
    opcheck(fn, (out, x, layer.alpha, layer.limit))
