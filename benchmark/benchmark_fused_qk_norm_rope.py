# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""
Benchmark `torch.ops._C.fused_qk_norm_rope` and gate its heads_per_sg choice.

Times the automatic choice against every forced heads_per_sg value and exits 1
if auto/best exceeds --max-gap (and --min-gap-us) on any model shape.
"""
# isort: off
import argparse
import csv
import math
import random
import statistics
import sys
from dataclasses import dataclass

import torch

from utils import bootstrap_benchmark_env, ensure_save_path_exists

bootstrap_benchmark_env(__file__)

import vllm_xpu_kernels._C  # noqa: F401
# isort: on

DEVICE = "xpu"
H_VALUES = (1, 2, 4, 8)
AUTO = -1
MAX_POSITION = 8192
EPS = 1e-6
TARGET_LOOP_MS = 10.0
MIN_ITERS, MAX_ITERS = 20, 400
WARMUP = 10


@dataclass(frozen=True)
class Shape:
    name: str
    num_heads_q: int
    num_heads_kv: int
    head_dim: int


# Per-rank head counts.
MODEL_SHAPES = [
    Shape("Qwen3-0.6B", 16, 8, 128),
    Shape("Qwen3-4B", 32, 8, 128),
    Shape("Qwen3-14B", 40, 8, 128),
    Shape("Qwen3-32B", 64, 8, 128),
    Shape("Qwen3-32B-tp4", 16, 2, 128),
    Shape("Qwen3-30B-A3B", 32, 4, 128),
    Shape("Qwen3-30B-A3B-tp2", 16, 2, 128),
    Shape("Qwen3-235B-A22B-tp4", 16, 1, 128),
    Shape("GLM-4.5-Air-tp2", 48, 4, 128),
    Shape("Gemma-3-4B", 8, 4, 256),
    Shape("Gemma-3-27B", 32, 16, 128),
]
MODEL_TOKENS = [1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

SYNTHETIC_HEADS = [(4, 2), (8, 2), (8, 4), (12, 4), (16, 4), (16, 8),
                   (24, 8), (32, 8), (40, 8), (48, 8), (64, 8), (64, 16),
                   (96, 8)]
SYNTHETIC_SHAPES = [
    Shape(f"syn-{q}q{kv}kv-d{d}", q, kv, d) for q, kv in SYNTHETIC_HEADS
    for d in (64, 128, 256)
]
SYNTHETIC_TOKENS = [
    1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
]


def make_inputs(shape: Shape, num_tokens: int, dtype: torch.dtype):
    hd = shape.head_dim
    total = (shape.num_heads_q + 2 * shape.num_heads_kv) * hd
    qkv = torch.randn(num_tokens, total, dtype=dtype, device=DEVICE)
    positions = torch.randint(0,
                              MAX_POSITION, (num_tokens, ),
                              dtype=torch.long,
                              device=DEVICE)
    q_weight = torch.empty(hd, dtype=dtype, device=DEVICE).normal_(1.0, 0.1)
    k_weight = torch.empty(hd, dtype=dtype, device=DEVICE).normal_(1.0, 0.1)
    inv_freq = 1.0 / (10000.0**(
        torch.arange(0, hd // 2, dtype=torch.float32, device=DEVICE) /
        (hd // 2)))
    freqs = torch.outer(
        torch.arange(MAX_POSITION, dtype=torch.float32, device=DEVICE),
        inv_freq)
    cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return qkv, positions, q_weight, k_weight, cos_sin_cache


def time_us(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(MIN_ITERS):
        fn()
    end.record()
    torch.xpu.synchronize()
    per_call_ms = start.elapsed_time(end) / MIN_ITERS
    iters = int(min(MAX_ITERS, max(MIN_ITERS,
                                   TARGET_LOOP_MS / max(per_call_ms, 1e-6))))
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.xpu.synchronize()
    return 1000.0 * start.elapsed_time(end) / iters


def bench_case(shape: Shape, num_tokens: int, is_neox: bool,
               dtype: torch.dtype, rounds: int) -> dict[int, float]:
    base, positions, q_weight, k_weight, cache = make_inputs(
        shape, num_tokens, dtype)

    def run(qkv, h):
        torch.ops._C.fused_qk_norm_rope(qkv, shape.num_heads_q,
                                        shape.num_heads_kv,
                                        shape.num_heads_kv, shape.head_dim,
                                        EPS, q_weight, k_weight, cache,
                                        is_neox, positions, h)

    # Forced values above the head count are clamped by the kernel, so they
    # would only re-time a smaller h under the wrong label.
    total_qk_heads = shape.num_heads_q + shape.num_heads_kv
    variants = [AUTO, *(h for h in H_VALUES if h <= total_qk_heads)]
    samples: dict[int, list[float]] = {h: [] for h in variants}
    for _ in range(rounds):
        order = variants[:]
        random.shuffle(order)
        for h in order:
            # The op is in-place: start every variant from the same input.
            qkv = base.clone()
            samples[h].append(time_us(lambda h=h, qkv=qkv: run(qkv, h)))
    return {h: statistics.median(v) for h, v in samples.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--save-path", default="./fused_qk_norm_rope/")
    parser.add_argument("--grid",
                        choices=["models", "synthetic", "all"],
                        default="models")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"],
                        default="bfloat16")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-gap", type=float, default=1.30,
                        help="fail if auto/best exceeds this ratio")
    parser.add_argument("--min-gap-us", type=float, default=1.0,
                        help="... and auto exceeds best by this many us")
    parser.add_argument("--no-gate", action="store_true",
                        help="report only, always exit 0")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.xpu.set_device("xpu:0")
    dtype = getattr(torch, args.dtype)

    grids = []
    if args.grid in ("models", "all"):
        grids.append(("models", MODEL_SHAPES, MODEL_TOKENS))
    if args.grid in ("synthetic", "all"):
        grids.append(("synthetic", SYNTHETIC_SHAPES, SYNTHETIC_TOKENS))

    save_path = ensure_save_path_exists(args.save_path)
    violations = []
    for grid_name, shapes, tokens in grids:
        rows = []
        for shape in shapes:
            for is_neox in (True, False):
                for num_tokens in tokens:
                    t = bench_case(shape, num_tokens, is_neox, dtype,
                                   args.rounds)
                    best_us, best_h = min(
                        (t[h], h) for h in H_VALUES if h in t)
                    ratio = t[AUTO] / best_us
                    rows.append({
                        "shape": shape.name,
                        "num_heads_q": shape.num_heads_q,
                        "num_heads_kv": shape.num_heads_kv,
                        "head_dim": shape.head_dim,
                        "is_neox": int(is_neox),
                        "num_tokens": num_tokens,
                        "dtype": args.dtype,
                        "auto_us": round(t[AUTO], 3),
                        "best_us": round(best_us, 3),
                        **{
                            f"forced_h{h}_us":
                            round(t[h], 3) if h in t else ""
                            for h in H_VALUES
                        },
                        # "_time" keeps it a metric in benchmark_ci.py.
                        "auto_over_best_time_ratio": round(ratio, 4),
                    })
                    gated = (grid_name == "models" and ratio > args.max_gap
                             and t[AUTO] - best_us > args.min_gap_us)
                    if gated:
                        violations.append(rows[-1] | {"best_h": best_h})
                    print(f"{shape.name:22s} neox={int(is_neox)} "
                          f"tokens={num_tokens:6d} auto={t[AUTO]:9.2f}us "
                          f"best=h{best_h} {best_us:9.2f}us "
                          f"ratio={ratio:5.2f}{'  GATE' if gated else ''}",
                          flush=True)

        csv_file = (f"{save_path}/fused_qk_norm_rope_{grid_name}_"
                    f"{args.dtype}.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        ratios = [r["auto_over_best_time_ratio"] for r in rows]
        decode = [
            r["auto_over_best_time_ratio"] for r in rows
            if r["num_tokens"] <= 256
        ]
        print(f"[{grid_name}] {len(rows)} cases, auto/best geomean "
              f"{math.exp(statistics.fmean(map(math.log, ratios))):.3f} "
              f"(decode<=256: "
              f"{math.exp(statistics.fmean(map(math.log, decode))):.3f}), "
              f"worst {max(ratios):.2f} -> {csv_file}")

    if violations:
        print(f"\nheads_per_sg gate: {len(violations)} model case(s) with "
              f"auto/best > {args.max_gap} and > {args.min_gap_us} us excess:",
              file=sys.stderr)
        for v in violations:
            print(f"  {v['shape']} neox={v['is_neox']} "
                  f"tokens={v['num_tokens']}: auto {v['auto_us']} us vs "
                  f"h{v['best_h']} {v['best_us']} us "
                  f"({v['auto_over_best_time_ratio']}x)",
                  file=sys.stderr)
        if not args.no_gate:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
