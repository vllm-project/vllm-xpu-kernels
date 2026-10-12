# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.ops.reduce_op import ReduceMax, ReduceMin
from tests.utils import REF_DEVICE, opcheck, seed_everything

DEVICE = REF_DEVICE
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
    x = torch.randn(num_tokens, d, dtype=dtype, device=DEVICE)
    layer = ReduceMax()
    fn = torch.ops._xpu_C.reduce_max_row_wise
    ref_values, ref_indices = layer.forward_native(x)
    layer_k = layer.to(KERNEL_DEVICE)
    values, indices = layer_k(_to_kernel(x))
    values = values.to(DEVICE)
    indices = indices.to(DEVICE)

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
    x = torch.randn(num_tokens, d, dtype=dtype, device=DEVICE)
    layer = ReduceMin()
    fn = torch.ops._xpu_C.reduce_min_row_wise
    ref_values, ref_indices = layer.forward_native(x)
    layer_k = layer.to(KERNEL_DEVICE)
    values, indices = layer_k(_to_kernel(x))
    values = values.to(DEVICE)
    indices = indices.to(DEVICE)

    torch.testing.assert_close(values, ref_values)
    torch.testing.assert_close(indices, ref_indices)

    values_k = torch.empty_like(values).to(KERNEL_DEVICE)
    indices_k = torch.empty_like(indices).to(KERNEL_DEVICE)
    opcheck(fn, (values_k, indices_k, _to_kernel(x)))


@pytest.mark.parametrize("batch,d, dtype", [
    (7, 1025, torch.half),  # Unaligned rows fall back to the baseline kernel.
    (2049, 4096, torch.float),
    (2049, 4096, torch.bfloat16),
])
@pytest.mark.parametrize("op,ref", [
    ("reduce_max_row_wise", torch.max),
    ("reduce_min_row_wise", torch.min),
])
@torch.inference_mode()
def test_reduce_vectorized_multirow(batch, d, dtype, op, ref):
    seed_everything(0)
    x = torch.randn(batch, d, dtype=dtype, device=DEVICE)
    # Duplicate extrema in each row to exercise the lowest-index tie rule.
    extreme = 10 if op == "reduce_max_row_wise" else -10
    x[:, 0] = extreme
    x[:, -1] = extreme
    ref_values, ref_indices = ref(x, dim=-1, keepdim=True)

    x_xpu = x.to(KERNEL_DEVICE)
    values = torch.empty((batch, 1), device=KERNEL_DEVICE, dtype=dtype)
    indices = torch.empty((batch, 1), device=KERNEL_DEVICE, dtype=torch.int32)
    getattr(torch.ops._xpu_C, op)(values, indices, x_xpu)

    torch.testing.assert_close(values.to(DEVICE), ref_values)
    torch.testing.assert_close(indices.to(DEVICE), ref_indices.to(torch.int32))
