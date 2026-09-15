# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.ops.silu_and_mul_with_clamp_op import SiluAndMulWithClamp
from tests.utils import opcheck, seed_everything

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048]  # Arbitrary values for testing
D = [512, 13824, 17]  # 17 exercises the scalar fallback (d % vec_size != 0)
LIMITS = [7.0]
PARAMS = [(1.0, 0.0), (1.702, 1.0)]  # (alpha, beta)
SEEDS = [0]
XPU_DEVICES = [f"xpu:{i}" for i in range(min(torch.xpu.device_count(), 2))]

#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [1, 7],
        "d": [32, 64],
    },
}

default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float: 1e-5}
default_rtol = {
    torch.float16: 2e-3,
    torch.bfloat16: 2e-2,
    torch.float: 1.3e-6
}


def get_default_atol(output) -> float:
    return default_atol[output.dtype]


def get_default_rtol(output) -> float:
    return default_rtol[output.dtype]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("limit", LIMITS)
@pytest.mark.parametrize("params", PARAMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_with_clamp(
    num_tokens: int,
    d: int,
    limit: float,
    params: tuple,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    seed_everything(seed)
    # Note: torch.set_default_device("xpu:1") not works.
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)
    alpha, beta = params
    x = torch.randn(num_tokens, 2 * d, dtype=dtype)

    layer = SiluAndMulWithClamp(limit=limit, alpha=alpha, beta=beta)

    out = layer(x)
    ref_out = layer.forward_native(x)

    torch.testing.assert_close(out,
                               ref_out,
                               atol=get_default_atol(out),
                               rtol=get_default_rtol(out))

    d = x.shape[-1] // 2
    output_shape = (x.shape[:-1] + (d, ))
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    fn = torch.ops._C.silu_and_mul_with_clamp
    opcheck(fn, (out, x, limit, alpha, beta))


@pytest.mark.parametrize("shape", [(0, 512), (4, 0)])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_with_clamp_empty_input(
    shape: tuple,
    dtype: torch.dtype,
    device: str,
) -> None:
    num_tokens, d = shape
    x = torch.randn(num_tokens, 2 * d, dtype=dtype, device=device)
    out = torch.empty(num_tokens, d, dtype=dtype, device=device)

    torch.ops._C.silu_and_mul_with_clamp(out, x, 7.0, 1.0, 0.0)

    assert out.numel() == 0
