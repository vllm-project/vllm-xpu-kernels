# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.ops.topk_op import stable_topk
from tests.utils import REF_DEVICE, format_tc, seed_everything

DEVICE = REF_DEVICE
KERNEL_DEVICE = "xpu"

# override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [7],
        "dim_size": [1024],
        "k": [8],
        "largest": [True],
        "dtype": [torch.bfloat16],
    },
}


# torch.ops._xpu_C.topk_row_wise requires the reduced dim to be a multiple
# of 32 (sub-group width) and >= 32, and k in [1, 16].
@pytest.mark.parametrize("num_tokens", [1, 7, 83])
@pytest.mark.parametrize("dim_size", [32, 64, 128, 1024, 4096, 8192])
@pytest.mark.parametrize("k", [1, 3, 8, 16])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32],
                         ids=format_tc)
def test_topk_row_wise(
    num_tokens: int,
    dim_size: int,
    k: int,
    largest: bool,
    dtype: torch.dtype,
):
    seed_everything(0)
    x = torch.randn(num_tokens, dim_size, dtype=dtype, device=DEVICE)

    # torch.sort(descending=True) on XPU is wrong for bfloat16 with a large
    # last dim, so the reference is computed on CPU.
    # topk_row_wise is not based on stable sort. Skip the check for indices.
    ref_values, _ = stable_topk(x.cpu(), k, dim=-1, largest=largest)
    values, indices = torch.ops._xpu_C.topk_row_wise(x.to(KERNEL_DEVICE), k,
                                                     largest)

    torch.testing.assert_close(values.cpu(), ref_values)
    # The returned indices must point at the returned values.
    torch.testing.assert_close(torch.gather(x.cpu(), -1, indices.cpu()),
                               values.cpu())
