# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for compute_slot_mapping XPU kernel.

Run `pytest tests/test_compute_slot_mapping.py`.
"""

from __future__ import annotations

import pytest
import torch

from tests.register_ops import compute_slot_mapping
from tests.utils import seed_everything

PAD_SLOT_ID = -1
DEVICE = "xpu"

seed_everything(0)


# ---------------------------------------------------------------------------
# Reference implementation (mirrors the Triton/SYCL kernel)
# ---------------------------------------------------------------------------
def ref_compute_slot_mapping(
    num_tokens: int,
    max_num_tokens: int,
    query_start_loc: torch.Tensor,  # [num_reqs+1], int32, on CPU
    positions: torch.Tensor,        # [num_tokens],   int64, on CPU
    block_table: torch.Tensor,      # [num_reqs, max_blocks_per_req], int32, CPU
    block_size: int,
    total_cp_world_size: int,
    total_cp_rank: int,
    cp_kv_cache_interleave_size: int,
    pad_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    out = torch.full((max_num_tokens,), pad_id, dtype=torch.int64)
    virtual_block_size = block_size * total_cp_world_size
    interleave = cp_kv_cache_interleave_size

    num_reqs = block_table.shape[0]
    qsl = query_start_loc.tolist()
    pos = positions.tolist()

    for req_idx in range(num_reqs):
        start, end = qsl[req_idx], qsl[req_idx + 1]
        for tok in range(start, end):
            p = pos[tok]
            block_index = p // virtual_block_size
            block_number = int(block_table[req_idx, block_index].item())
            vbo = p - block_index * virtual_block_size

            is_local = (vbo // interleave) % total_cp_world_size == total_cp_rank
            local_off = (vbo // (total_cp_world_size * interleave)) * interleave \
                        + (vbo % interleave)
            slot = block_number * block_size + local_off
            out[tok] = slot if is_local else pad_id

    # Tokens in [num_tokens, max_num_tokens) are padding.
    out[num_tokens:max_num_tokens] = pad_id
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_inputs(
    num_reqs: int,
    avg_tokens_per_req: int,
    block_size: int,
    total_cp_world_size: int,
    max_blocks_per_req: int,
    pad_extra: int,
    dtype_pos=torch.int64,
):
    """Generate a random but consistent test case."""
    # Variable per-request token counts.
    g = torch.Generator().manual_seed(num_reqs * 131 + avg_tokens_per_req)
    lens = torch.randint(
        low=1,
        high=max(2, 2 * avg_tokens_per_req),
        size=(num_reqs,),
        generator=g,
    )
    num_tokens = int(lens.sum().item())
    max_num_tokens = num_tokens + pad_extra

    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.cumsum(lens, dim=0).to(torch.int32)

    virtual_block_size = block_size * total_cp_world_size

    # Build per-request positions; bounded by max_blocks_per_req * virtual_block_size.
    positions = torch.empty(num_tokens, dtype=dtype_pos)
    max_pos = max_blocks_per_req * virtual_block_size - 1
    for r in range(num_reqs):
        s, e = int(query_start_loc[r]), int(query_start_loc[r + 1])
        # Random monotone-ish positions within a request (not required by kernel,
        # but mimics real workloads).
        base = torch.randint(0, max_pos // 2 + 1, (1,), generator=g).item()
        offs = torch.randint(0, max_pos // 2 + 1, (e - s,), generator=g)
        offs, _ = torch.sort(offs)
        positions[s:e] = (base + offs).clamp_max(max_pos).to(dtype_pos)

    # Block table with non-trivial block numbers.
    block_table = torch.randint(
        low=1,
        high=100_000,
        size=(num_reqs, max_blocks_per_req),
        dtype=torch.int32,
        generator=g,
    )
    return num_tokens, max_num_tokens, query_start_loc, positions, block_table


# ---------------------------------------------------------------------------
# Parameterization
# ---------------------------------------------------------------------------
NUM_REQS = [1, 4, 17, 64]
AVG_TOKENS = [1, 8, 64, 257]
BLOCK_SIZES = [16, 32, 64]
PAD_EXTRA = [0, 33, 1024]

# (total_cp_world_size, total_cp_rank, cp_kv_cache_interleave_size)
CP_CONFIGS = [
    (1, 0, 1),     # no CP — fast path
    (2, 0, 1),
    (2, 1, 1),
    (4, 2, 16),
    (8, 5, 64),
]


@pytest.mark.parametrize("num_reqs", NUM_REQS)
@pytest.mark.parametrize("avg_tokens", AVG_TOKENS)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("pad_extra", PAD_EXTRA)
@pytest.mark.parametrize("cp_cfg", CP_CONFIGS)
@torch.inference_mode()
def test_compute_slot_mapping(
    num_reqs: int,
    avg_tokens: int,
    block_size: int,
    pad_extra: int,
    cp_cfg: tuple[int, int, int],
):
    total_cp_world_size, total_cp_rank, interleave = cp_cfg
    max_blocks_per_req = 32

    (
        num_tokens,
        max_num_tokens,
        qsl_cpu,
        pos_cpu,
        bt_cpu,
    ) = _make_inputs(
        num_reqs=num_reqs,
        avg_tokens_per_req=avg_tokens,
        block_size=block_size,
        total_cp_world_size=total_cp_world_size,
        max_blocks_per_req=max_blocks_per_req,
        pad_extra=pad_extra,
    )

    # ---- Reference (CPU) ----
    ref = ref_compute_slot_mapping(
        num_tokens=num_tokens,
        max_num_tokens=max_num_tokens,
        query_start_loc=qsl_cpu,
        positions=pos_cpu,
        block_table=bt_cpu,
        block_size=block_size,
        total_cp_world_size=total_cp_world_size,
        total_cp_rank=total_cp_rank,
        cp_kv_cache_interleave_size=interleave,
        pad_id=PAD_SLOT_ID,
    )

    # ---- Kernel (XPU) ----
    qsl = qsl_cpu.to(DEVICE)
    pos = pos_cpu.to(DEVICE)
    bt = bt_cpu.to(DEVICE)
    # Pre-fill slot_mapping with a sentinel != PAD_ID to make sure the kernel
    # actually writes both regular and padding positions.
    slot_mapping = torch.full((max_num_tokens,), -999, dtype=torch.int64, device=DEVICE)

    compute_slot_mapping(
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        max_num_tokens=max_num_tokens,
        query_start_loc=qsl,
        positions=pos,
        block_table=bt,
        block_table_stride=max_blocks_per_req,
        block_size=block_size,
        slot_mapping=slot_mapping,
        total_cp_world_size=total_cp_world_size,
        total_cp_rank=total_cp_rank,
        cp_kv_cache_interleave_size=interleave,
        pad_id=PAD_SLOT_ID,
    )

    torch.xpu.synchronize()
    out = slot_mapping.cpu()

    # All sentinel values must be overwritten.
    assert (out != -999).all(), "kernel left some positions unwritten"

    # Padding region must be PAD_ID.
    if pad_extra > 0:
        assert (out[num_tokens:] == PAD_SLOT_ID).all(), \
            "padding region not filled with PAD_SLOT_ID"

    # Full comparison.
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@torch.inference_mode()
def test_compute_slot_mapping_all_remote():
    """When total_cp_world_size > 1 and every token maps to a non-local rank,
    all valid slots must become PAD_ID."""
    block_size = 16
    total_cp_world_size = 2
    total_cp_rank = 0  # we'll craft positions that always belong to rank 1
    interleave = 1
    num_reqs = 4
    max_blocks_per_req = 8

    lens = torch.tensor([3, 5, 1, 7], dtype=torch.int32)
    num_tokens = int(lens.sum().item())
    max_num_tokens = num_tokens
    qsl_cpu = torch.zeros(num_reqs + 1, dtype=torch.int32)
    qsl_cpu[1:] = torch.cumsum(lens, dim=0).to(torch.int32)

    virtual_block_size = block_size * total_cp_world_size
    # Choose positions so that (pos // interleave) % world_size == 1 (= other rank).
    # With interleave=1, that means pos % 2 == 1.
    positions = (torch.arange(num_tokens, dtype=torch.int64) * 2 + 1) \
        .clamp_max(max_blocks_per_req * virtual_block_size - 1)
    block_table = torch.randint(1, 1000, (num_reqs, max_blocks_per_req), dtype=torch.int32)

    slot_mapping = torch.full((max_num_tokens,), 0, dtype=torch.int64, device=DEVICE)

    compute_slot_mapping(
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        max_num_tokens=max_num_tokens,
        query_start_loc=qsl_cpu.to(DEVICE),
        positions=positions.to(DEVICE),
        block_table=block_table.to(DEVICE),
        block_table_stride=max_blocks_per_req,
        block_size=block_size,
        slot_mapping=slot_mapping,
        total_cp_world_size=total_cp_world_size,
        total_cp_rank=total_cp_rank,
        cp_kv_cache_interleave_size=interleave,
        pad_id=PAD_SLOT_ID,
    )
    torch.xpu.synchronize()
    assert (slot_mapping.cpu() == PAD_SLOT_ID).all()
