# SPDX-License-Identifier: Apache-2.0
"""Single-card check of the vocabulary-parallel candidate exchange
(mono_shard_topk, top-k and greedy) through a loopback mailbox: both "ranks"
hold the same shard."""

import pytest
import torch

from tests.mono_decode.mono_utils import addr, requires_mono

DEVICE = "xpu:0"
VS = 248320 // 2  # one rank's vocabulary shard (Qwen3.5 / 3.6)


def _mbox():
    n = torch.ops._xpu_C.mono_lmhead_mbox_bytes()
    return torch.zeros(n, dtype=torch.uint8, device=DEVICE)


def _shard(rows, gen, dtype):
    return (torch.randn((rows, VS), generator=gen) * 3).to(dtype).to(DEVICE)


@requires_mono
@pytest.mark.parametrize("rows", [1, 2, 8])
@pytest.mark.parametrize("k", [1, 20, 64])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_shard_topk_loopback(rows, k, rank, dtype):
    gen = torch.Generator().manual_seed(rows * 100 + k)
    lg = _shard(rows, gen, dtype)
    mb = _mbox()
    width = max(2 * k, 128)
    for _ in range(3):  # repeated calls: the mailbox tags must stay in step
        vals = torch.empty((rows, width), dtype=torch.float32, device=DEVICE)
        idx = torch.empty((rows, width), dtype=torch.int32, device=DEVICE)
        torch.ops._xpu_C.mono_shard_topk(vals, idx, lg, k, rank,
                                         addr(mb), addr(mb))
        torch.xpu.synchronize()
        ref = lg.float().cpu()
        for b in range(rows):
            # the k largest (ties: the smaller id), in ascending id order
            order = torch.sort(-ref[b], stable=True).indices
            top = order[:k].sort().values
            exp_v = ref[b, top]
            exp_i = top.to(torch.int32) + rank * VS
            got_v, got_i = vals[b].cpu(), idx[b].cpu()
            for part in (0, 1):  # own part, the "peer" part (loopback)
                assert torch.equal(got_v[part * k:(part + 1) * k], exp_v)
                assert torch.equal(got_i[part * k:(part + 1) * k], exp_i)
            assert torch.isneginf(got_v[2 * k:]).all()
    # no exchange wait gave up
    assert torch.ops._xpu_C.mono_health(0)[:2].tolist() == [0, 0]

