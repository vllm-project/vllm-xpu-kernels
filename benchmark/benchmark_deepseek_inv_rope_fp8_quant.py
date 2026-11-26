# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""
Benchmark for deepseek_inv_rope_fp8_quant kernel (DeepSeek-V4 config).

Measures XPU device time via the torch profiler and rotates through distinct
input buffers so calls miss the LLC, as they would mid-model.

Measures the fused SYCL kernel's device latency and memory bandwidth.

DeepSeek-V4 config: head_dim=512 (nope=448, rope=64), 64 heads / 8 groups.
"""

import argparse
import os
import subprocess
import sys

# Kineto logs a profiler_start/stop line per timed shape at its USDT severity,
# which is level 5 here. It reads this when torch is imported, so it has to be
# set first.
os.environ.setdefault("KINETO_LOG_LEVEL", "6")

import torch  # noqa: E402
from utils import bootstrap_benchmark_env  # noqa: E402

bootstrap_benchmark_env(__file__)

import vllm_xpu_kernels._xpu_C  # noqa: F401,E402 - registers torch.ops._xpu_C

# DeepSeek-V4 default dimensions
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
QUANT_GROUP_SIZE = 128
CHUNKS_PER_HEAD = HEAD_DIM // QUANT_GROUP_SIZE  # 4
MXFP8_GROUP_SIZE = 32
MX_BLOCKS_PER_HEAD = HEAD_DIM // MXFP8_GROUP_SIZE  # 16
FP8_MAX = 448.0
MAX_POS = 1_048_576

# DeepSeek-V4 has num_attention_heads=64, head_dim=512, qk_rope_head_dim=64
# (-> nope=448) and o_groups=8. Tensor parallelism shards heads and o_groups
# together, so a TP degree fully determines the per-rank shape.
NUM_HEADS = 64
NUM_GROUPS = 8
TP_DEGREES = [1, 4, 8]

# Decode (1..128) through chunked prefill (max_num_batched_tokens is 2048 in
# the bench recipes; larger values cover long-context batches).
TOKEN_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
                8192, 16384, 32768]


def tp_shape(tp: int):
    """Per-rank (num_heads, n_groups) for a tensor-parallel degree."""
    if NUM_HEADS % tp or NUM_GROUPS % tp:
        raise SystemExit(f"TP {tp} does not evenly shard {NUM_HEADS} heads / "
                         f"{NUM_GROUPS} groups")
    return NUM_HEADS // tp, NUM_GROUPS // tp

# Set from --rotate-mb in __main__.
_ROTATE_BYTES = 256 * 1024 * 1024


def rotation_count(bytes_per_call: int) -> int:
    if _ROTATE_BYTES <= 0 or bytes_per_call <= 0:
        return 1
    return max(1, -(-_ROTATE_BYTES // bytes_per_call))


def make_pool(n: int, shape, dtype, fill=torch.randn):
    pool = fill((n, *shape), dtype=dtype, device="xpu")
    return [pool[i] for i in range(n)]


def benchmark_op(fn, warmup: int, iters: int, rotate: int = 1) -> float:
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    ring = [None] * rotate

    for i in range(warmup):
        ring[i % rotate] = fn(i)
    torch.xpu.synchronize()

    with profile(activities=[ProfilerActivity.XPU]) as prof:
        for i in range(iters):
            ring[i % rotate] = fn(i)
        torch.xpu.synchronize()

    total_us = sum(e.self_device_time_total for e in prof.key_averages()
                   if e.device_type == DeviceType.XPU)
    if total_us <= 0.0:
        raise RuntimeError("XPU profiler returned no kernel time")
    return total_us / iters


def run_inv_rope_fp8_quant(num_tokens: int, num_heads: int, n_groups: int,
                           warmup: int, iters: int, use_mxfp8: bool = False):
    heads_per_group = num_heads // n_groups
    elems = num_tokens * num_heads * HEAD_DIM
    blocks_per_head = MX_BLOCKS_PER_HEAD if use_mxfp8 else CHUNKS_PER_HEAD
    scale_bytes = 1 if use_mxfp8 else 4  # e8m0 vs fp32
    scales = num_tokens * num_heads * blocks_per_head

    bytes_moved = (
        elems * 2  # o (bf16)
        + num_tokens * 8  # positions (int64)
        + num_tokens * ROPE_DIM * 4  # cos_sin_cache gather (fp32)
        + elems * 1  # fp8 out
        + scales * scale_bytes  # scale out
    )

    n = rotation_count(bytes_moved)
    inputs = make_pool(n, (num_tokens, num_heads, HEAD_DIM), torch.bfloat16)
    positions = torch.randint(0, MAX_POS, (num_tokens, ),
                              dtype=torch.int64, device="xpu")
    cos_sin_cache = torch.randn((MAX_POS, ROPE_DIM),
                                dtype=torch.float32, device="xpu")

    def run(i):
        return torch.ops._xpu_C.deepseek_inv_rope_fp8_quant(
            inputs[i % n], positions, cos_sin_cache, n_groups,
            heads_per_group, NOPE_DIM, ROPE_DIM, use_mxfp8)

    avg_us = benchmark_op(run, warmup, iters, n)
    return avg_us, bytes_moved


def compute_metrics(latency_us: float, bytes_moved: float):
    latency_s = latency_us / 1e6
    bandwidth = bytes_moved / latency_s / 1e9
    return latency_us, bandwidth


def collect_rows(bench_cases, warmup, iters, use_mxfp8=False):
    rows = []
    for tp, num_tokens in bench_cases:
        num_heads, n_groups = tp_shape(tp)
        avg_us, bytes_moved = run_inv_rope_fp8_quant(
            num_tokens, num_heads, n_groups, warmup, iters, use_mxfp8)
        latency_us, bandwidth = compute_metrics(avg_us, bytes_moved)
        rows.append((tp, num_tokens, num_heads, n_groups, latency_us,
                     bandwidth))
    return rows


def print_md_table(op_name: str, rows):
    """Print benchmark results as a Markdown table."""
    print(f"\n## {op_name}\n")
    print("| TP | num_tokens | num_heads | n_groups | Latency (us) "
          "| BW (GB/s) |")
    print("|----|------------|-----------|----------|--------------"
          "|-----------|")
    for (tp, num_tokens, num_heads, n_groups, latency_us,
         bandwidth) in rows:
        print(f"| {tp:>2d} | {num_tokens:>10d} | {num_heads:>9d} "
              f"| {n_groups:>8d} | {latency_us:>12.3f} "
              f"| {bandwidth:>9.3f} |")


def run_isolated(args, formats):
    """Run each (TP, format) combination in a fresh process, sequentially."""
    script = os.path.abspath(__file__)
    for tp in args.tp:
        for fmt in formats:
            print(f"\n# Starting isolated TP={tp} format={fmt}", flush=True)
            command = [
                sys.executable,
                script,
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
                "--tp",
                str(tp),
                "--tokens",
                *(str(token_count) for token_count in args.tokens),
                "--rotate-mb",
                str(args.rotate_mb),
                "--format",
                fmt,
            ]
            subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark deepseek_inv_rope_fp8_quant on XPU")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--tp", type=int, nargs="+", default=TP_DEGREES,
                        help=f"tensor-parallel degrees to bench; each shards "
                             f"{NUM_HEADS} heads / {NUM_GROUPS} groups "
                            f"(default: {' '.join(map(str, TP_DEGREES))}); "
                            "multiple TP degrees run in isolated processes")
    parser.add_argument("--tokens", type=int, nargs="+", default=TOKEN_COUNTS,
                        help="token counts to bench (default: full sweep)")
    parser.add_argument(
        "--rotate-mb", type=int, default=256,
        help="cycle enough distinct inputs to span this much traffic, so "
             "calls miss the LLC as they would mid-model; 0 reuses one set")
    parser.add_argument(
        "--format", choices=["fp32", "mxfp8", "both"], default="both",
        help="scale output format: fp32 = per-128 float32 scales, "
             "mxfp8 = per-32 e8m0 scales (default: both)")
    args = parser.parse_args()

    formats = ["fp32", "mxfp8"] if args.format == "both" else [args.format]

    # Each shape/format allocates its own multi-GB rotation pool, and the XPU
    # caching allocator never gives it back, so keep one per process.
    if len(args.tp) > 1 or len(formats) > 1:
        run_isolated(args, formats)
        raise SystemExit(0)

    use_mxfp8 = formats[0] == "mxfp8"

    _ROTATE_BYTES = args.rotate_mb * 1024 * 1024

    bench_cases = [(tp, t) for tp in args.tp for t in args.tokens]

    torch.manual_seed(0)
    torch.set_default_device("xpu")

    block = MXFP8_GROUP_SIZE if use_mxfp8 else QUANT_GROUP_SIZE
    scale_kind = "e8m0" if use_mxfp8 else "float32"
    print(f"# deepseek_inv_rope_fp8_quant — DeepSeek-V4 "
          f"(head_dim={HEAD_DIM}, nope={NOPE_DIM}, rope={ROPE_DIM}, "
          f"FP8 E4M3, per-{block} {scale_kind} scales)")
    print(f"# device: {torch.xpu.get_device_name(0)}")

    print_md_table(
        f"deepseek_inv_rope_fp8_quant — per-{block} {scale_kind} scales",
        collect_rows(bench_cases, args.warmup, args.iters, use_mxfp8))

