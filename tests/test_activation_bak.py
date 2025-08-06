import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from typing import Type, Optional, Union, Tuple
from tests.ops.activation_op import SiluAndMul

DTYPES = [torch.half, torch.bfloat16, torch.float]

NUM_TOKENS = [7, 2046]  # Arbitrary values for testing

D = [512, 5120]  # Arbitrary values for testing

SEEDS = [0]
DEVICES = ["xpu"]

@pytest.mark.parametrize("activation", ["silu"])
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_act_and_mul(
    activation: str,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, 2 * d, dtype=dtype)
    if activation == "silu":
        op = SiluAndMul()
    out = op(x)
    ref_out = op.naive_forward(x)

    # The SiLU and GELU implementations are equivalent to the native PyTorch
    # implementations, so we can do exact comparison.
    assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)


