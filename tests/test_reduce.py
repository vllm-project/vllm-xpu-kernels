# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.ops.reduce_op import ReduceMax, ReduceMin
from tests.utils import opcheck, seed_everything

DEVICE = "cpu"
KERNEL_DEVICE = "xpu"


def _to_kernel(x):
    return None if x is None else x.to(KERNEL_DEVICE)


DTYPES = [torch.float, torch.half, torch.bfloat16]
NUM_TOKENS = [1, 7, 83, 4096]  # Arbitrary values for testing (batch size)
D = [1, 8, 128, 512, 1024, 2048, 4096, 16384]  # Arbitrary values for testing (reduce dim)
SEEDS = [0]
XPU_DEVICES = ["xpu:0"]

# override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [1],
        "d": [128],
    },
}


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_reduce_max(
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    seed_everything(seed)
    torch.xpu.set_device(device)
    x = torch.randn(num_tokens, d, dtype=dtype)
    layer = ReduceMax()
    fn = torch.ops._xpu_C.reduce_max_row_wise
    ref_values, ref_indices = layer.forward_native(x)
    layer_k = layer.to(KERNEL_DEVICE)
    values, indices = layer_k(_to_kernel(x))
    values = values.cpu()
    indices = indices.cpu()

    torch.testing.assert_close(values, ref_values)
    torch.testing.assert_close(indices, ref_indices)

    values_k = torch.empty_like(values).to(KERNEL_DEVICE)
    indices_k = torch.empty_like(indices).to(KERNEL_DEVICE)
    opcheck(fn, (values_k, indices_k, _to_kernel(x)))


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_reduce_min(
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    seed_everything(seed)
    torch.xpu.set_device(device)
    x = torch.randn(num_tokens, d, dtype=dtype)
    layer = ReduceMin()
    fn = torch.ops._xpu_C.reduce_min_row_wise
    ref_values, ref_indices = layer.forward_native(x)
    layer_k = layer.to(KERNEL_DEVICE)
    values, indices = layer_k(_to_kernel(x))
    values = values.cpu()
    indices = indices.cpu()

    torch.testing.assert_close(values, ref_values)
    torch.testing.assert_close(indices, ref_indices)

    values_k = torch.empty_like(values).to(KERNEL_DEVICE)
    indices_k = torch.empty_like(indices).to(KERNEL_DEVICE)
    opcheck(fn, (values_k, indices_k, _to_kernel(x)))
