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


@pytest.mark.parametrize("m", [1, 33, 64, 222])
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("k", [128, 511, 1024])
@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float16, torch.bfloat16],
                         ids=format_tc)
def test_moe_sum(m: int, topk: int, k: int, dtype: torch.dtype):
    input = torch.randn((m, topk, k), device="xpu", dtype=dtype)
    actual = torch.empty((m, k), device="xpu", dtype=dtype)

    expected = input.sum(dim=1)
    moe_sum(input, actual)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=0)

    opcheck(torch.ops._moe_C.moe_sum, (input, actual, None, None))


@pytest.mark.parametrize("m", [1, 33, 64])
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("k", [128, 1024])
@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float16, torch.bfloat16],
                         ids=format_tc)
def test_moe_sum_expert_map(m: int, topk: int, k: int, dtype: torch.dtype):
    num_experts = 8
    input = torch.randn((m, topk, k), device="xpu", dtype=dtype)
    actual = torch.empty((m, k), device="xpu", dtype=dtype)

    topk_ids = torch.randint(0,
                             num_experts, (m, topk),
                             device="xpu",
                             dtype=torch.int32)
    # Mark half of the experts as non-local (mapped to -1).
    expert_map = torch.arange(num_experts, device="xpu", dtype=torch.int32)
    expert_map[num_experts // 2:] = -1

    # Reference: zero out contributions from non-local experts.
    local_mask = (expert_map[topk_ids.long()] >= 0)  # [m, topk]
    expected = (input *
                local_mask.unsqueeze(-1).to(input.dtype)).sum(dim=1)

    moe_sum(input, actual, topk_ids, expert_map)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=0)

    opcheck(torch.ops._moe_C.moe_sum, (input, actual, topk_ids, expert_map))


@pytest.mark.parametrize("m", [1, 33, 64])
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("k", [128, 1024])
@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float16, torch.bfloat16],
                         ids=format_tc)
def test_moe_sum_padded_topk_ids(m: int, topk: int, k: int,
                                 dtype: torch.dtype):
    # topk_ids may contain -1 padding slots (no expert_map). Those slots must
    # be skipped without indexing expert_map out of bounds.
    num_experts = 8
    input = torch.randn((m, topk, k), device="xpu", dtype=dtype)
    actual = torch.empty((m, k), device="xpu", dtype=dtype)

    topk_ids = torch.randint(0,
                             num_experts, (m, topk),
                             device="xpu",
                             dtype=torch.int32)
    # Mark the last slot of every token as padding.
    topk_ids[:, -1] = -1

    valid = (topk_ids >= 0)  # [m, topk]
    expected = (input * valid.unsqueeze(-1).to(input.dtype)).sum(dim=1)

    moe_sum(input, actual, topk_ids, None)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=0)

    opcheck(torch.ops._moe_C.moe_sum, (input, actual, topk_ids, None))


def test_moe_sum_input_validation():
    m, topk, k = 32, 4, 128
    input = torch.randn((m, topk, k), device="xpu", dtype=torch.float32)
    output = torch.empty((m, k), device="xpu", dtype=torch.float32)
    topk_ids = torch.zeros((m, topk), device="xpu", dtype=torch.int32)
    expert_map = torch.arange(8, device="xpu", dtype=torch.int32)

    # expert_map without topk_ids must be rejected rather than silently ignored.
    with pytest.raises(RuntimeError, match="expert_map requires topk_ids"):
        moe_sum(input, output, None, expert_map)

    # dtype mismatch between input and output.
    bad_out = torch.empty((m, k), device="xpu", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="same dtype"):
        moe_sum(input, bad_out)

    # non-contiguous input (a strided slice along hidden).
    wide = torch.randn((m, topk, k * 2), device="xpu", dtype=torch.float32)
    noncontig = wide[:, :, ::2]  # shape [m, topk, k], non-contiguous
    assert not noncontig.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        moe_sum(noncontig, output)

    # topk_ids with a wrong shape.
    with pytest.raises(RuntimeError, match="num_tokens, topk"):
        moe_sum(input, output, topk_ids[:, :topk - 1].contiguous(), None)

    # expert_map with a non-int32 dtype.
    with pytest.raises(RuntimeError, match="expert_map must be int32"):
        moe_sum(input, output, topk_ids, expert_map.to(torch.int64))
