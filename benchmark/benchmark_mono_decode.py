# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""Mono decode layers at TP2: device time per decoder layer.

Two processes, one per card (cards 0 and 1, or ZE_AFFINITY_MASK), share their
mailboxes through Level Zero IPC as vLLM does. Each case registers one
GatedDeltaNet and one full-attention layer (Qwen3.5 / 3.6 35B-A3B shapes, one
TP2 rank's shards, random weights) and times back-to-back launches with XPU
events on rank 0: decode steps of 1, 2, 4 and 8 rows and verify steps (rows x
tokens), at the given attention context.

Usage: python benchmark/benchmark_mono_decode.py [--iters N] [--context T]
"""

# isort: off
import argparse
import os
import tempfile

import torch
import torch.multiprocessing as mp

from utils import bootstrap_benchmark_env

bootstrap_benchmark_env(__file__)
from tests.mono_decode.mono_utils import (exchange_fd, layer_tensors,
                                          mono_available, rows)
# isort: on

BLOCK = 64
# (requests, tokens a request): decode steps (1 token), then every verify
# kernel (rows x tokens: 2x2, 4x2, 8x2, 3x3, 6x3, 4x4, 8x4, 8x8)
CASES = [(1, 1), (2, 1), (4, 1), (8, 1), (1, 2), (2, 2), (4, 2), (1, 3),
         (2, 3), (1, 4), (2, 4), (1, 8)]


def _launch(ops, layer, tokens, bufs, meta):
    if tokens == 1:
        ops.mono_decode_layer(layer, *bufs, *meta)
    else:
        assert ops.mono_decode_layer_spec(layer, tokens, *bufs, *meta)


def _worker(rank, path, conn, args):
    import vllm_xpu_kernels._xpu_C  # noqa: F401
    ops = torch.ops._xpu_C
    torch.xpu.set_device(rank)
    dev = f"xpu:{rank}"
    cases = CASES
    nl = 2 * len(cases)
    h = ops.mono_ipc_create(nl * ops.mono_mbox_bytes())
    info = ops.mono_ipc_export(h)
    conn.send(info)
    ops.mono_ipc_open(h, conn.recv(), exchange_fd(rank, path, int(info[1])))
    own, peer = ops.mono_ipc_ptr(h, False), ops.mono_ipc_ptr(h, True)
    mb = ops.mono_mbox_bytes()
    nblk = (args.context + 64 + BLOCK - 1) // BLOCK
    results = []
    lid = 0
    for nb, tokens in cases:
        n = nb * tokens
        for attn in (False, True):
            t = layer_tensors(attn, 1 + rank, dev, nb, tokens, nb * nblk)
            ops.mono_register_layer(lid, attn, own + lid * mb,
                                    peer + lid * mb, t)
            gen = torch.Generator().manual_seed(5)
            hs, rs = rows(n, gen, dev)
            ho, ro = torch.empty_like(hs), torch.empty_like(rs)
            ctx = args.context
            if attn:
                pos = torch.tensor([ctx + j % tokens for j in range(n)],
                                   device=dev)
                bt = torch.arange(nb * nblk, dtype=torch.int32,
                                  device=dev).view(nb, nblk)
                seq = torch.full((nb, ), ctx + tokens, dtype=torch.int32,
                                 device=dev)
                slots = torch.tensor([b * nblk * BLOCK + ctx + j
                                      for b in range(nb)
                                      for j in range(tokens)], device=dev)
                meta = (pos, bt, seq, slots)
            elif tokens == 1:
                meta = (torch.zeros(1, dtype=torch.int32, device=dev),
                        torch.arange(nb, dtype=torch.int32, device=dev), None,
                        None)
            else:
                meta = (torch.zeros(1, dtype=torch.int32, device=dev),
                        torch.arange(n, dtype=torch.int32,
                                     device=dev).view(nb, tokens),
                        torch.ones(nb, dtype=torch.int32, device=dev), None)

            for _ in range(20):
                _launch(ops, lid, tokens, (hs, rs, ho, ro), meta)
            torch.xpu.synchronize()
            start = torch.xpu.Event(enable_timing=True)
            end = torch.xpu.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                _launch(ops, lid, tokens, (hs, rs, ho, ro), meta)
            end.record()
            torch.xpu.synchronize()
            us = start.elapsed_time(end) * 1e3 / args.iters
            results.append((nb, tokens, "attention" if attn else "gdn", us))
            lid += 1
    health = ops.mono_health(-1)[:2].tolist()
    conn.send((results, health))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--context", type=int, default=1024,
                        help="cached tokens of each request (attention)")
    args = parser.parse_args()
    if args.iters <= 0 or args.context < 0:
        parser.error("--iters must be positive, --context non-negative")
    if not (mono_available(0) and mono_available(1)):
        raise SystemExit("needs the mono decode kernels and two Xe2 GPUs")
    ctx = mp.get_context("spawn")
    path = os.path.join(tempfile.mkdtemp(), "mono_bench.sock")
    pipes = [ctx.Pipe() for _ in range(2)]
    procs = [
        ctx.Process(target=_worker, args=(r, path, pipes[r][1], args))
        for r in range(2)
    ]
    for p in procs:
        p.start()
    a, b = pipes[0][0], pipes[1][0]
    info0, info1 = a.recv(), b.recv()
    a.send(info1)
    b.send(info0)
    (res, health0), (_, health1) = a.recv(), b.recv()
    for p in procs:
        p.join()
    assert health0 == [0, 0] and health1 == [0, 0], (health0, health1)
    print(f"context {args.context}, {args.iters} launches per case "
          "(device time per layer, rank 0)")
    print(f"{'requests':>8} {'tokens':>6} {'layer':>10} {'us':>8}")
    for nb, tokens, kind, us in res:
        print(f"{nb:>8} {tokens:>6} {kind:>10} {us:>8.1f}")


if __name__ == "__main__":
    main()
