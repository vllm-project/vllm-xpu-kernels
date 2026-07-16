# SPDX-License-Identifier: Apache-2.0
"""Tests for the MOE layers.

Run `pytest tests/test_moe_sum.py`.
"""

import pytest
import torch

from tests.register_ops import moe_sum
from tests.utils import format_tc, opcheck

TOP_KS = [2, 6]

#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "m": [1, 33],
        "k": [128, 256],
    },
}


# topk=8 covers topk > 4; k=511 covers the non-vectorized scalar path. The
# layouts exercise contiguous input plus the two non-contiguous cases: a
# transpose (strided hidden -> scalar gather) and a topk-slice (hidden still
# contiguous -> vectorized).
@pytest.mark.parametrize("m", [1, 33, 222])
@pytest.mark.parametrize("topk", [*TOP_KS, 8])
@pytest.mark.parametrize("k", [128, 511, 1024])
@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.bfloat16],
                         ids=format_tc)
@pytest.mark.parametrize("layout", ["contig", "transpose", "slice"])
def test_moe_sum(m: int, topk: int, k: int, dtype: torch.dtype, layout: str):
    if layout == "transpose":
        input = torch.randn((m, k, topk), device="xpu",
                            dtype=dtype).transpose(1, 2)
    elif layout == "slice":
        input = torch.randn((m, 2 * topk, k), device="xpu",
                            dtype=dtype)[:, ::2, :]
    else:
        input = torch.randn((m, topk, k), device="xpu", dtype=dtype)
    assert input.is_contiguous() == (layout == "contig")
    actual = torch.empty((m, k), device="xpu", dtype=dtype)

    # Reduction accumulates in fp32.
    expected = input.float().sum(dim=1).to(dtype)
    moe_sum(input, actual)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=0)

    opcheck(torch.ops._moe_C.moe_sum, (input, actual))


@pytest.mark.parametrize("topk", [2, 8])
@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.bfloat16],
                         ids=format_tc)
@pytest.mark.parametrize("topk_ids_dtype", [torch.int32, torch.int64])
def test_moe_sum_pad_aware(topk: int, dtype: torch.dtype,
                          topk_ids_dtype: torch.dtype):
    m, k = 17, 128
    input = torch.randn((m, topk, k), device="xpu", dtype=dtype)

    topk_ids = torch.randint(0, 4, (m, topk), device="xpu",
                             dtype=topk_ids_dtype)
    # Force some slots unrouted, independent of expert parallelism.
    topk_ids[0, :] = -1
    topk_ids[1, 0] = -1

    expert_map = torch.tensor([0, -1, 1, -1], device="xpu", dtype=torch.int32)

    actual = torch.empty((m, k), device="xpu", dtype=dtype)
    moe_sum(input, actual, topk_ids, expert_map)

    routed = expert_map[topk_ids.clamp(min=0).long()] >= 0
    routed &= topk_ids >= 0
    expected = (input.float() * routed.unsqueeze(-1)).sum(dim=1).to(dtype)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=0)

    opcheck(torch.ops._moe_C.moe_sum, (input, actual, topk_ids, expert_map))
