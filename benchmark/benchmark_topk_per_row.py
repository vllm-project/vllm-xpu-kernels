# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the DSA indexer top-k (``_C::top_k_per_row_prefill/decode``).

The lightning indexer of DeepSeek Sparse Attention scores every query row
against every cached key, and this op selects the ``top_k`` positions that
sparse MLA then gathers. It runs once per indexer layer, on every rank, over
the full sequence.

The presets are the shapes vLLM produces for the models that use the op:

* DeepSeek-V3.2 / V3.2-Exp, GLM-5 / 5.1 / 5.2 / 5.3, Hy4: ``index_topk=2048``
* DeepSeek-V4-Pro: 1024, DeepSeek-V4-Flash / V4.1-Flash: 512, over KV that is
  compressed 4x, so a 128K context is a 32K row.

Prefill: vLLM splits a request's rows so that rows * KV width stays within
``VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`` (512 MB of fp32, 134217728 elements):
8192 rows at 16K of KV, 2048 at 64K, 1024 at 128K, 128 at 1M. Rows are causal,
the last ``s_q`` query tokens of the request. Several short requests share one
call with ragged row bounds (``row_starts`` != 0).

Decode: ``batch * next_n`` rows over a logits buffer as wide as
``max_model_len``, whatever the context length actually is.

Three logit layouts control where the winners sit inside each row: spread
over it (``uniform``), packed into its first (``front``) or last (``tail``)
``--concentration`` of it.

Examples
--------
    python benchmark_topk_per_row.py                      # every preset
    python benchmark_topk_per_row.py --filter dsv32       # one model family
    python benchmark_topk_per_row.py --list
    python benchmark_topk_per_row.py --check --csv out.csv
    python benchmark_topk_per_row.py --custom prefill:2048:49152:2048
"""

import argparse
import csv
import dataclasses
import time
from typing import Optional

import torch

DISTS = ("uniform", "front", "tail")

# Standard-normal logits stay well under this, so boosted positions win.
BOOST = 20.0


@dataclasses.dataclass
class Case:
    name: str
    model: str
    mode: str  # "prefill" or "decode"
    top_k: int
    # prefill: the query length of each request in the call and its KV length
    # (the request's rows are its last q_len tokens, causal).
    requests: tuple[tuple[int, int], ...] = ()
    # decode
    batch: int = 0
    next_n: int = 1
    seq_len: int = 0
    width: int = 0  # logits row stride, max_model_len

    def shape(self) -> str:
        if self.mode == "decode":
            return (f"B={self.batch} next_n={self.next_n} "
                    f"kv={self.seq_len} width={self.width}")
        if len(self.requests) == 1:
            q, kv = self.requests[0]
            return f"s_q={q} s_kv={kv}"
        return "reqs=" + ",".join(f"{q}/{kv}" for q, kv in self.requests)


def _prefill(name, model, top_k, *requests):
    return Case(name, model, "prefill", top_k, requests=tuple(requests))


def _decode(name, model, top_k, batch, seq_len, width, next_n=1):
    return Case(name, model, "decode", top_k, batch=batch, next_n=next_n,
                seq_len=seq_len, width=width)


PRESETS: list[Case] = [
    # DeepSeek-V3.2 / GLM-5.x, top_k 2048. One request per call.
    _prefill("dsv32-p-2k", "dsv32/glm5", 2048, (2048, 2048)),
    _prefill("dsv32-p-4k", "dsv32/glm5", 2048, (4096, 4096)),
    _prefill("dsv32-p-8k", "dsv32/glm5", 2048, (8192, 8192)),
    _prefill("dsv32-p-16k", "dsv32/glm5", 2048, (8192, 16384)),
    _prefill("dsv32-p-32k", "dsv32/glm5", 2048, (4096, 32768)),
    # Also the AE prefill shape.
    _prefill("dsv32-p-48k", "dsv32/glm5", 2048, (2048, 49152)),
    _prefill("dsv32-p-64k", "dsv32/glm5", 2048, (2048, 65536)),
    _prefill("dsv32-p-128k", "dsv32/glm5", 2048, (1024, 131072)),
    _prefill("dsv32-p-160k", "dsv32", 2048, (819, 163840)),
    _prefill("glm5-p-198k", "glm5", 2048, (661, 202752)),
    _prefill("glm52-p-1m", "glm5.2/hy4", 2048, (128, 1048576)),
    # Several requests packed into one call, ragged row bounds.
    _prefill("dsv32-p-2x4k", "dsv32/glm5", 2048, (4096, 4096), (4096, 4096)),
    _prefill("dsv32-p-4x2k", "dsv32/glm5", 2048, *[(2048, 2048)] * 4),
    _prefill("dsv32-p-11x1k", "dsv32/glm5", 2048, *[(1024, 1024)] * 11),
    _prefill("dsv32-p-mixed", "dsv32/glm5", 2048, (3000, 3000),
             (1500, 1500), (2500, 2500), (1192, 1192)),
    _prefill("dsv32-p-chunk2", "dsv32/glm5", 2048, (2048, 6144),
             (2048, 2048)),
    # DeepSeek-V4-Pro, top_k 1024, KV compressed 4x.
    _prefill("dsv4pro-p-32k", "dsv4-pro", 1024, (8192, 8192)),
    _prefill("dsv4pro-p-128k", "dsv4-pro", 1024, (4096, 32768)),
    _prefill("dsv4pro-p-512k", "dsv4-pro", 1024, (1024, 131072)),
    _prefill("dsv4pro-p-1m", "dsv4-pro", 1024, (512, 262144)),
    # DeepSeek-V4-Flash / V4.1-Flash, top_k 512.
    _prefill("dsv4flash-p-32k", "dsv4-flash", 512, (8192, 8192)),
    _prefill("dsv4flash-p-128k", "dsv4-flash", 512, (4096, 32768)),
    _prefill("dsv4flash-p-512k", "dsv4-flash", 512, (1024, 131072)),
    _prefill("dsv4flash-p-1m", "dsv4-flash", 512, (512, 262144)),
    _prefill("dsv41flash-p-1m-full", "dsv4.1-flash", 512, (128, 1048576)),
    _prefill("dsv4flash-p-4x2k", "dsv4-flash", 512, *[(2048, 2048)] * 4),
    # Decode. Width is max_model_len: 163840 for V3.2, 1M/4 for V4.
    _decode("dsv32-d-b1-9k", "dsv32", 2048, 1, 9216, 163840),
    _decode("dsv32-d-b32-9k", "dsv32", 2048, 32, 9216, 163840),
    _decode("dsv32-d-b128-9k", "dsv32", 2048, 128, 9216, 163840),
    _decode("dsv32-d-b512-9k", "dsv32", 2048, 512, 9216, 163840),
    _decode("dsv32-d-b32-mtp2-9k", "dsv32", 2048, 32, 9216, 163840, 2),
    _decode("dsv32-d-b32-64k", "dsv32", 2048, 32, 65536, 163840),
    _decode("dsv32-d-b16-128k", "dsv32", 2048, 16, 131072, 163840),
    _decode("glm52-d-b8-512k", "glm5.2", 2048, 8, 524288, 1048576),
    _decode("dsv4pro-d-b128-8k", "dsv4-pro", 1024, 128, 2048, 262144),
    _decode("dsv4pro-d-b128-128k", "dsv4-pro", 1024, 128, 32768, 262144),
    _decode("dsv4pro-d-b512-32k", "dsv4-pro", 1024, 512, 8192, 262144),
    _decode("dsv4flash-d-b128-8k", "dsv4-flash", 512, 128, 2048, 262144),
    _decode("dsv4flash-d-b512-32k", "dsv4-flash", 512, 512, 8192, 262144),
    # AE decode: rows as long as the logits buffer.
    _decode("ae-d-b1-50k", "ae", 2048, 1, 50000, 50000),
    _decode("ae-d-b8-50k", "ae", 2048, 8, 50000, 50000),
    _decode("ae-d-b4-mtp2-50k", "ae", 2048, 4, 50000, 50000, 2),
    # The edges of the split decode path: just below and at its smallest
    # width, long rows, its most rows, and just past them.
    _decode("split-d-b1-12k", "split", 2048, 1, 12288, 12288),
    _decode("split-d-b1-16k", "split", 2048, 1, 16384, 16384),
    _decode("split-d-b1-128k", "split", 2048, 1, 131072, 131072),
    _decode("split-d-b1-300k", "split", 2048, 1, 300000, 300000),
    _decode("split-d-b16-128k", "split", 2048, 16, 131072, 131072),
    _decode("split-d-b32-50k", "split", 2048, 32, 50000, 50000),
]


def resolve_device(name: str) -> torch.device:
    """Load the backend kernels; both register the same ``_C`` schemas."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "xpu"
    device = torch.device(name)
    if device.type == "cuda":
        from vllm import _custom_ops  # noqa: F401
    elif device.type == "xpu":
        import vllm_xpu_kernels._C  # noqa: F401
    else:
        raise SystemExit(f"unsupported device: {name}")
    return device


def sync(device: torch.device) -> None:
    getattr(torch, device.type).synchronize()


def row_bounds(case: Case) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Per-row [start, end) inside the logits row, and the logits width."""
    if case.mode == "decode":
        rows = torch.arange(case.batch * case.next_n, dtype=torch.int64)
        ends = case.seq_len - case.next_n + rows % case.next_n + 1
        return torch.zeros_like(ends), ends, case.width
    starts, ends, base = [], [], 0
    for q_len, kv_len in case.requests:
        # Query token i of the request sees keys [0, kv_len - q_len + i].
        first = kv_len - q_len + 1
        ends.append(base + torch.arange(first, kv_len + 1, dtype=torch.int64))
        starts.append(torch.full((q_len, ), base, dtype=torch.int64))
        base += kv_len
    return torch.cat(starts), torch.cat(ends), base


def build_logits(case: Case, dist: str, starts: torch.Tensor,
                 ends: torch.Tensor, width: int, concentration: float,
                 device: torch.device, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    rows = starts.numel()
    logits = torch.randn(rows, width, dtype=torch.float32, device=device,
                         generator=gen)
    s = starts.to(device).unsqueeze(1)
    e = ends.to(device).unsqueeze(1)
    n = e - s
    hot = (n.to(torch.float32) * concentration).to(torch.int64)
    hot = torch.minimum(torch.clamp(hot, min=case.top_k), n)
    pos = torch.arange(width, dtype=torch.int64, device=device).unsqueeze(0)
    if dist == "front":
        mask = (pos >= s) & (pos < s + hot)
    elif dist == "tail":
        mask = (pos >= e - hot) & (pos < e)
    else:
        prob = hot.to(torch.float32) / n.clamp(min=1).to(torch.float32)
        draw = torch.rand(rows, width, dtype=torch.float32, device=device,
                          generator=gen)
        mask = (draw < prob) & (pos >= s) & (pos < e)
        del draw
    logits.add_(mask.to(logits.dtype), alpha=BOOST)
    del mask
    return logits


def make_runner(case: Case, dist: str, args, device: torch.device):
    starts, ends, width = row_bounds(case)
    logits = build_logits(case, dist, starts, ends, width, args.concentration,
                          device, args.seed)
    rows = logits.shape[0]
    indices = torch.empty((rows, case.top_k), dtype=torch.int32,
                          device=device)
    if case.mode == "prefill":
        row_starts = starts.to(torch.int32).to(device)
        row_ends = ends.to(torch.int32).to(device)

        def run() -> None:
            torch.ops._C.top_k_per_row_prefill(logits, row_starts, row_ends,
                                               indices, rows,
                                               logits.stride(0),
                                               logits.stride(1), case.top_k)
    else:
        seq_lens = torch.full((case.batch, ), case.seq_len,
                              dtype=torch.int32, device=device)

        def run() -> None:
            torch.ops._C.top_k_per_row_decode(logits, case.next_n, seq_lens,
                                              indices, rows,
                                              logits.stride(0),
                                              logits.stride(1), case.top_k)

    return run, logits, indices, starts, ends


def check(case: Case, logits: torch.Tensor, indices: torch.Tensor,
          starts: torch.Tensor, ends: torch.Tensor, max_rows: int) -> int:
    """Rows whose selected values differ from a CPU top-k. Tie tolerant.

    The reference is computed on the CPU: torch.topk on some XPU stacks
    returns a wrong set.
    """
    rows = logits.shape[0]
    pick = torch.linspace(0, rows - 1, min(rows, max_rows)).long().unique()
    cpu_logits = logits[pick].cpu()
    cpu_idx = indices[pick].cpu().long()
    bad = 0
    for j, r in enumerate(pick.tolist()):
        s, e = int(starts[r]), int(ends[r])
        n = e - s
        row = cpu_logits[j, s:e]
        k = min(case.top_k, n)
        want = torch.topk(row, k).values.sort().values
        # The prefill kernel writes positions relative to the row start.
        got_idx = cpu_idx[j, :k]
        if (got_idx < 0).any() or (got_idx >= n).any():
            bad += 1
            continue
        got = row[got_idx].sort().values
        if got_idx.unique().numel() != k or not torch.equal(got, want):
            bad += 1
    return bad


def bench(run, device: torch.device, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        run()
    sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        run()
    sync(device)
    return (time.perf_counter() - start) * 1e6 / iters


def parse_custom(spec: str) -> Case:
    """prefill:S_Q:S_KV:TOP_K or decode:BATCH:NEXT_N:S_KV:WIDTH:TOP_K"""
    parts = spec.split(":")
    if parts[0] == "prefill" and len(parts) == 4:
        q, kv, k = map(int, parts[1:])
        return _prefill(f"custom-p-{q}x{kv}-k{k}", "custom", k, (q, kv))
    if parts[0] == "decode" and len(parts) == 6:
        b, nn, kv, w, k = map(int, parts[1:])
        return _decode(f"custom-d-b{b}x{nn}-{kv}-k{k}", "custom", k, b, kv, w,
                       nn)
    raise SystemExit(f"bad --custom spec: {spec}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filter", action="append", default=[],
                    help="run presets whose name or model contains this "
                    "substring; repeatable")
    ap.add_argument("--mode", choices=("prefill", "decode", "all"),
                    default="all")
    ap.add_argument("--custom", action="append", default=[],
                    help="prefill:S_Q:S_KV:TOP_K or "
                    "decode:BATCH:NEXT_N:S_KV:WIDTH:TOP_K; repeatable, "
                    "replaces the presets")
    ap.add_argument("--dist", choices=DISTS + ("all", ), default="all")
    ap.add_argument("--concentration", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup-iters", type=int, default=10)
    ap.add_argument("--num-iters", type=int, default=100)
    ap.add_argument("--check", action="store_true",
                    help="also verify sampled rows against a CPU top-k")
    ap.add_argument("--check-rows", type=int, default=64)
    ap.add_argument("--csv", default=None, help="also write results here")
    ap.add_argument("--list", action="store_true",
                    help="print the presets and exit")
    ap.add_argument("--device", default="auto",
                    choices=("auto", "xpu", "cuda"))
    return ap.parse_args()


def select(args) -> list[Case]:
    if args.custom:
        return [parse_custom(s) for s in args.custom]
    cases = PRESETS
    if args.mode != "all":
        cases = [c for c in cases if c.mode == args.mode]
    if args.filter:
        cases = [
            c for c in cases
            if any(f in c.name or f in c.model for f in args.filter)
        ]
    return cases


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    cases = select(args)
    if args.list:
        for c in cases:
            print(f"{c.name:<24}{c.model:<14}{c.mode:<9}k={c.top_k:<6}"
                  f"{c.shape()}")
        return
    device = resolve_device(args.device)
    dists = DISTS if args.dist == "all" else (args.dist, )

    header = (f"{'case':<24}{'top_k':>6}  {'shape':<34}{'dist':<9}"
              f"{'us':>10}{'GB/s':>9}")
    if args.check:
        header += f"{'bad rows':>10}"
    print(f"device={device.type} concentration={args.concentration} "
          f"warmup={args.warmup_iters} iters={args.num_iters}")
    print(header)

    writer: Optional[csv.writer] = None
    fh = None
    if args.csv:
        fh = open(args.csv, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["case", "model", "mode", "top_k", "shape", "dist",
                         "us", "GBps", "bad_rows"])
    try:
        for case in cases:
            for dist in dists:
                run, logits, indices, starts, ends = make_runner(
                    case, dist, args, device)
                us = bench(run, device, args.warmup_iters, args.num_iters)
                scanned = float((ends - starts).clamp(min=0).sum())
                gbps = scanned * 4 / us / 1e3
                line = (f"{case.name:<24}{case.top_k:>6}  "
                        f"{case.shape():<34}{dist:<9}{us:>10.1f}{gbps:>9.1f}")
                bad = ""
                if args.check:
                    bad = check(case, logits, indices, starts, ends,
                                args.check_rows)
                    line += f"{bad:>10}"
                print(line, flush=True)
                if writer:
                    writer.writerow([case.name, case.model, case.mode,
                                     case.top_k, case.shape(), dist,
                                     f"{us:.1f}", f"{gbps:.1f}", bad])
                del run, logits, indices
                getattr(torch, device.type).empty_cache()
    finally:
        if fh:
            fh.close()


if __name__ == "__main__":
    main()
