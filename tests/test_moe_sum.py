# SPDX-License-Identifier: Apache-2.0
"""Tests for the MOE layers.

Run `pytest tests/test_moe_sum.py`.
"""

import pytest
import torch

from tests.register_ops import moe_sum
from tests.utils import opcheck

TOP_KS = [2, 6]

#override pytest parameters when 'USE_SIMULATOR' eq 1
SIMULATOR_PYTEST_PARAMS = {
    "default": {
        "m": [1, 33],
        "k": [128, 256],
    },
}


@pytest.mark.parametrize("m", [1, 33, 64, 222])
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("k", [128, 511, 1024])
@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float16, torch.bfloat16])
def test_moe_sum(m: int, topk: int, k: int, dtype: torch.dtype):
    input = torch.randn((m, topk, k), device="xpu", dtype=dtype)
    actual = torch.empty((m, k), device="xpu", dtype=dtype)

    expected = input.sum(dim=1)
    moe_sum(input, actual)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=0)

    opcheck(torch.ops._moe_C.moe_sum, (input, actual))
