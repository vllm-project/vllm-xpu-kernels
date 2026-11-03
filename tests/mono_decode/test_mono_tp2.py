# SPDX-License-Identifier: Apache-2.0
"""Two-card (TP2) checks of the mono decode kernels, one process per card with
the mailboxes shared through Level Zero IPC as in vLLM:

* decoder layers: the two ranks' outputs are bitwise equal, and swapping the
  ranks' weight shards (and their caches) gives bitwise the same outputs (the
  all-reduces add the two partial sums; a rank reading its own partial instead
  of the peer's would not be symmetric);
* MTP prologue: the gathered fc output against a torch reference;
* vocabulary-parallel candidates (top-k and greedy) against the full logits.
"""

import io
import os
import tempfile
import time

import pytest
import torch
import torch.multiprocessing as mp

from tests.mono_decode.mono_utils import (HID, exchange_fd, layer_tensors,
                                          requires_two_cards, rows)

VS = 248320 // 2
def _worker(rank, path, conn):
    import vllm_xpu_kernels._xpu_C  # noqa: F401
    ops = torch.ops._xpu_C
    torch.xpu.set_device(rank)
    dev = f"xpu:{rank}"
    nl = 16
    h = ops.mono_ipc_create(nl * ops.mono_mbox_bytes() +
                            ops.mono_lmhead_mbox_bytes())
    info = ops.mono_ipc_export(h)
    conn.send(info)
    peer_info = conn.recv()
    peer_fd = exchange_fd(rank, path, int(info[1]))
    ops.mono_ipc_open(h, peer_info, peer_fd)
    own, peer = ops.mono_ipc_ptr(h, False), ops.mono_ipc_ptr(h, True)
    mb = ops.mono_mbox_bytes()
    slot = iter(range(nl))
    out = {}

    def layer_case(name, attn, nb, tokens=1, swap=False):
        i = next(slot)
        seed = (2 if rank == 0 else 1) if swap else (1 if rank == 0 else 2)
        bt = torch.arange(nb * 4, dtype=torch.int32,
                          device=dev).view(nb, 4)
        t = layer_tensors(attn, seed, dev, nb, tokens, nb * 4)
        ops.mono_register_layer(100 + i, attn, own + i * mb, peer + i * mb, t)
        gen = torch.Generator().manual_seed(5)
        hs, rs = rows(nb * tokens, gen, dev)
        ho, ro = torch.empty_like(hs), torch.empty_like(rs)
        n = nb * tokens
        if attn:
            ctx = 150
            pos = torch.tensor([ctx + j % tokens for j in range(n)],
                               device=dev)
            slots = torch.tensor([b * 4 * 64 + ctx + j for b in range(nb)
                                  for j in range(tokens)], device=dev)
            seq = torch.full((nb, ), ctx + tokens, dtype=torch.int32,
                             device=dev)
            if tokens == 1:
                ops.mono_decode_layer(100 + i, hs, rs, ho, ro, pos, bt, seq,
                                      slots)
            else:
                assert ops.mono_decode_layer_spec(100 + i, tokens, hs, rs, ho,
                                                  ro, pos, bt, seq, slots)
        else:
            zero = torch.zeros(1, dtype=torch.int32, device=dev)
            if tokens == 1:
                ops.mono_decode_layer(
                    100 + i, hs, rs, ho, ro, zero,
                    torch.arange(nb, dtype=torch.int32, device=dev), None,
                    None)
            else:
                idx = torch.arange(nb * tokens, dtype=torch.int32,
                                   device=dev).view(nb, tokens)
                acc = torch.ones(nb, dtype=torch.int32, device=dev)
                assert ops.mono_decode_layer_spec(100 + i, tokens, hs, rs, ho,
                                                  ro, zero, idx, acc, None)
        torch.xpu.synchronize()
        out[name] = (ho.cpu(), ro.cpu())

    for swap in (False, True):
        s = "swap" if swap else "base"
        layer_case(f"gdn1_{s}", False, 1, swap=swap)
        layer_case(f"gdn8_{s}", False, 8, swap=swap)
        layer_case(f"gdn_spec_{s}", False, 2, tokens=4, swap=swap)
        layer_case(f"attn1_{s}", True, 1, swap=swap)
        layer_case(f"attn4_{s}", True, 4, swap=swap)
        layer_case(f"attn_spec_{s}", True, 1, tokens=8, swap=swap)

    # MTP prologue: rank r holds fc rows r * 1024 ..
    gen = torch.Generator().manual_seed(9)
    ne = (torch.randn(HID, generator=gen) * 0.1).half()
    nh = (torch.randn(HID, generator=gen) * 0.1).half()
    fc = (torch.randn((HID, 2 * HID), generator=gen)).clamp(-240, 240).to(
        torch.float8_e4m3fn)
    fsc = torch.tensor([0.002])
    i = next(slot)
    ops.mono_register_mtp(
        200, own + i * mb, peer + i * mb,
        [ne.to(dev), nh.to(dev),
         fc[rank * 1024:(rank + 1) * 1024].to(dev).t(), fsc.to(dev)], rank)
    x = (torch.randn((3, 2 * HID), generator=gen) * 2).half()
    y = torch.empty((3, HID), dtype=torch.float16, device=dev)
    assert ops.mono_mtp_prologue(200, x.to(dev), y)
    torch.xpu.synchronize()
    out["mtp"] = (y.cpu(), ne, nh, fc, fsc, x)

    # candidates / argmax of distinct shards
    lm = own + nl * mb
    lmp = peer + nl * mb
    gen = torch.Generator().manual_seed(11 + rank)
    lg = (torch.randn((4, VS), generator=gen) * 3).half().to(dev)
    vals = torch.empty((4, 128), dtype=torch.float32, device=dev)
    idx = torch.empty((4, 128), dtype=torch.int32, device=dev)
    ops.mono_shard_topk(vals, idx, lg, 20, rank, lm, lmp)
    v1 = torch.empty((4, 2), dtype=torch.float32, device=dev)
    i1 = torch.empty((4, 2), dtype=torch.int32, device=dev)
    ops.mono_shard_topk(v1, i1, lg, 1, rank, lm, lmp)  # greedy (k = 1)
    torch.xpu.synchronize()
    tok = i1.gather(1, v1.argmax(dim=1, keepdim=True)).squeeze(1).long()
    out["topk"] = (vals.cpu(), idx.cpu(), tok.cpu(), lg.cpu())

    # fail-stop: no wait gave up so far; then rank 1's tags skip ahead, so
    # neither rank ever sees the words it waits for: both give up (about 12 s),
    # record it, and a later call exits at once (sticky)
    health = ops.mono_health(rank)
    out["health_ok"] = health.cpu().clone()
    i = next(slot)
    t = layer_tensors(False, 1 if rank == 0 else 2, dev, 1, 1, 4)
    ops.mono_register_layer(100 + i, False, own + i * mb, peer + i * mb, t)
    gen = torch.Generator().manual_seed(5)
    hs, rs = rows(1, gen, dev)
    ho, ro = torch.empty_like(hs), torch.empty_like(rs)
    zero = torch.zeros(1, dtype=torch.int32, device=dev)
    if rank == 1:
        ops.mono_fault_inject(100 + i, 7)
    times = []
    for _ in range(2):
        t0 = time.monotonic()
        ops.mono_decode_layer(100 + i, hs, rs, ho, ro, zero, zero, None, None)
        torch.xpu.synchronize()
        times.append(time.monotonic() - t0)
    out["fault"] = (health.cpu().clone(), times)
    buf = io.BytesIO()  # bytes: the process may exit before the tensors are
    torch.save(out, buf)  # read back from shared memory
    conn.send(buf.getvalue())
    conn.close()


def _run_pair():
    ctx = mp.get_context("spawn")
    path = os.path.join(tempfile.mkdtemp(), "mono_tp2.sock")
    c0, c1 = ctx.Pipe(), ctx.Pipe()
    procs = [
        ctx.Process(target=_worker, args=(0, path, c0[1])),
        ctx.Process(target=_worker, args=(1, path, c1[1])),
    ]
    for p in procs:
        p.start()
    i0, i1 = c0[0].recv(), c1[0].recv()
    c0[0].send(i1)
    c1[0].send(i0)
    r0, r1 = (torch.load(io.BytesIO(c[0].recv())) for c in (c0, c1))
    for p in procs:
        p.join(timeout=300)
        assert p.exitcode == 0
    return r0, r1


@pytest.fixture(scope="module")
def pair():
    return _run_pair()


CASES = ["gdn1", "gdn8", "gdn_spec", "attn1", "attn4", "attn_spec"]


@requires_two_cards
@pytest.mark.parametrize("case", CASES)
def test_layer_ranks_bitwise_equal(pair, case):
    r0, r1 = pair
    for s in ("base", "swap"):
        ho0, ro0 = r0[f"{case}_{s}"]
        ho1, ro1 = r1[f"{case}_{s}"]
        assert torch.isfinite(ho0.float()).all()
        assert torch.equal(ho0, ho1)


@requires_two_cards
@pytest.mark.parametrize("case", CASES)
def test_layer_all_reduce_symmetric(pair, case):
    r0, _ = pair
    assert torch.equal(r0[f"{case}_base"][0], r0[f"{case}_swap"][0])


@requires_two_cards
def test_mtp_prologue_gather(pair):
    r0, r1 = pair
    y0, ne, nh, fc, fsc, x = r0["mtp"]
    assert torch.equal(y0, r1["mtp"][0])

    def rms(v, w):
        v = v.float()
        return (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6) *
                (1 + w.float())).half().float()

    z = torch.cat([rms(x[:, :HID], ne), rms(x[:, HID:], nh)], dim=-1)
    ref = (z @ fc.float().t() * fsc).half()
    c = torch.nn.functional.cosine_similarity(y0.float().flatten(),
                                              ref.float().flatten(), dim=0)
    assert c > 0.99999


@requires_two_cards
def test_candidates_and_greedy(pair):
    r0, r1 = pair
    v0, i0, t0, lg0 = r0["topk"]
    v1, i1, t1, lg1 = r1["topk"]
    assert torch.equal(v0, v1) and torch.equal(i0, i1) and torch.equal(t0, t1)
    full = torch.cat([lg0, lg1], dim=1).float()
    assert torch.equal(t0, full.argmax(dim=-1))
    for b in range(4):
        exp = torch.sort(-full[b], stable=True).indices[:20]
        assert set(exp.tolist()) <= set(i0[b, :40].tolist())
        assert torch.equal(v0[b, :40], full[b, i0[b, :40].long()])
        assert torch.isneginf(v0[b, 40:]).all()


@requires_two_cards
def test_fail_stop(pair):
    """A stalled peer fails both ranks instead of hanging them: every wait is
    bounded, the failure is recorded in the device's health words (sticky), and
    healthy calls never give up a wait."""
    for r in pair:
        assert int(r["health_ok"][0]) == 0 and int(r["health_ok"][1]) == 0
        health, (first, second) = r["fault"]
        assert int(health[0]) == 1 and int(health[1]) >= 1
        assert first < 60, f"the first call gave up after {first:.1f} s"
        assert second < 5, f"a call after the failure took {second:.1f} s"
