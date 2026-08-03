# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import vllm_xpu_kernels._moe_C  # noqa: F401
from tests.utils import seed_everything

KERNEL_DEVICE = "xpu"

# scale_k = hidden_size // block_k (block_k = 32 for mxfp).
SCALE_K = [4, 8, 40]
# Per-expert row counts covering: zeros, non-multiples of 4, exact multiples.
ROWS_PER_EXPERT = [
    [0, 1, 2, 3, 4, 5, 7, 8],
    [16, 0, 33, 100],
    [1],
    [0, 0, 0],
    [128, 256, 200, 137],
]


def round_up_4(r):
    return (r + 3) & ~3


def ref_reorder(A_scales, rows_list, scale_k):
    padded_rows = [round_up_4(r) for r in rows_list]
    total_padded = sum(padded_rows)
    out = torch.zeros((total_padded, scale_k),
                      dtype=A_scales.dtype,
                      device=A_scales.device)
    src_off = 0
    dst_off = 0
    for r, pr in zip(rows_list, padded_rows):
        if r != 0:
            dst_block = out[dst_off:dst_off + pr, :]
            view = dst_block.view(scale_k, pr)
            src = A_scales[src_off:src_off + r, :].transpose(-1,
                                                             -2).contiguous()
            view[:, :r] = src
        src_off += r
        dst_off += pr
    return out, total_padded


@pytest.mark.parametrize("rows_list", ROWS_PER_EXPERT)
@pytest.mark.parametrize("scale_k", SCALE_K)
def test_reorder_mxfp_scales(rows_list, scale_k):
    seed_everything(0)
    total_rows = sum(rows_list)
    num_experts = len(rows_list)

    # Build random e8m0 scales via uint8 bit patterns (e8m0 is a raw 8-bit
    # exponent code; any byte is a valid value for a copy test).
    raw = torch.randint(0, 255, (max(total_rows, 1), scale_k),
                        dtype=torch.uint8)[:total_rows]
    A_scales = raw.view(torch.float8_e8m0fnu).to(KERNEL_DEVICE)
    rows_per_expert = torch.tensor(rows_list,
                                   dtype=torch.int32,
                                   device=KERNEL_DEVICE)

    ref, total_padded = ref_reorder(A_scales, rows_list, scale_k)

    upper = total_rows + 3 * num_experts
    out = torch.ops._moe_C.reorder_mxfp_scales(A_scales, rows_per_expert,
                                               upper)

    assert out.shape[0] == upper
    assert out.shape[1] == scale_k
    # The meaningful region matches the reference exactly.
    assert torch.equal(
        out[:total_padded].view(torch.uint8),
        ref.view(torch.uint8))
    # Extra tail rows are zero-initialized.
    assert torch.equal(
        out[total_padded:].view(torch.uint8),
        torch.zeros_like(out[total_padded:].view(torch.uint8)))
