# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.ops.softmax_op import Softmax
from tests.utils import opcheck, seed_everything


DEVICE = "cpu"
KERNEL_DEVICE = "xpu"


def _to_kernel(x):
    return None if x is None else x.to(KERNEL_DEVICE)


DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048, 4096]  # Arbitrary values for testing (batch size)
D = [3, 512, 2048, 4096, 13824, 65536]
SEEDS = [0]
XPU_DEVICES = ["xpu:0"]

# override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [16],
        "d": [128],
    },
}


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_softmax(
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    seed_everything(seed)
    torch.xpu.set_device(device)
    x = torch.randn(num_tokens, d, dtype=dtype)
    layer = Softmax()
    fn = torch.ops._xpu_C.softmax
    ref_out = layer.forward_native(x)
    layer_k = layer.to(KERNEL_DEVICE)
    out = layer_k(_to_kernel(x))
    out = out.cpu()

    torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=1e-3)

    out_k = torch.empty_like(x).to(KERNEL_DEVICE)
    opcheck(fn, (out_k, _to_kernel(x)))
