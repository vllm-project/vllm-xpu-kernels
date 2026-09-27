# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.ops.scatter_op import scatter_row, scatter_row_native
from tests.utils import opcheck, seed_everything

DEVICE = "cpu"
KERNEL_DEVICE = "xpu"


def _to_kernel(x):
    return None if x is None else x.to(KERNEL_DEVICE)


DTYPES = [torch.float, torch.half, torch.bfloat16]
NUM_TOKENS = [1, 7, 83, 4096]  # Arbitrary values for testing (batch size)
# dim_size * element_size must be a multiple of 16 bytes for every dtype
# above (half/bfloat16: 2 bytes, float: 4 bytes) -> multiples of 16 elements.
D = [16, 32, 128, 1024, 2048, 65536]
SEEDS = [0]
XPU_DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

# override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [7],
        "d": [16],
    },
}


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_scatter_row(
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    seed_everything(seed)
    torch.xpu.set_device(device)
    src = torch.randn(num_tokens, d, dtype=dtype)
    # A permutation guarantees every dst row is written exactly once, so the
    # result is well-defined regardless of scatter order.
    index = torch.randperm(num_tokens, dtype=torch.int64)
    dst = torch.zeros(num_tokens, d, dtype=dtype)

    ref_out = scatter_row_native(dst, src, index)

    dst_k = _to_kernel(dst.clone())
    src_k = _to_kernel(src)
    index_k = _to_kernel(index)
    out = scatter_row(dst_k, src_k, index_k)
    out = out.cpu()

    torch.testing.assert_close(out, ref_out)

    fn = torch.ops._xpu_C.scatter_row
    opcheck(fn, (dst_k, src_k, index_k))
