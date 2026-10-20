# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DeepseekV4 dequantize_and_gather_k_cache on XPU."""

from __future__ import annotations

import argparse
import gc
import math
from dataclasses import dataclass
from typing import Any

import torch
import triton
from tabulate import tabulate

import vllm_xpu_kernels._C  # noqa: F401

HEAD_DIM = 512
FP8_DIM = 448
BF16_DIM = 64
SCALE_DIM = 8
TOKEN_DATA_BYTES = FP8_DIM + BF16_DIM * 2
NUM_QUANT_BLOCKS = 7

# A UE8M0 scale byte is an IEEE-754 biased exponent, so scale = 2^(byte-127).
# The producer computes exponent = ceil(log2(max(amax, 1e-4) / 448)), so the
# amax floor pins the smallest realistic exponent at -22. Sample 2^-22 .. 2^8,
# the production range, which cannot overflow the bf16 output.
SCALE_BYTE_MIN = 105
SCALE_BYTE_MAX = 136

BF16_BYTES = torch.tensor([], dtype=torch.bfloat16).element_size()

QUANTILES = [0.5, 0.2, 0.8]

@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    seq_len: int
    gather_len: int | None  # None gathers the whole sequence
    block_size: int
    offset: int = 0

    @property
    def copied_tokens(self) -> int:
        return self.seq_len if self.gather_len is None else self.gather_len


# SWA layers keep the sequence uncompressed and gather an explicit window.
# Compressed layers cache seq_len = logical_len / ratio tokens and gather all
# of them; the case name records the logical length before compression.
BUILTIN_CASES: list[BenchmarkCase] = [
    BenchmarkCase("swa_128", 128, 128, 64),
    BenchmarkCase("swa_512", 512, 512, 64),
    BenchmarkCase("swa_1k", 1024, 1024, 64),
    BenchmarkCase("swa_8k", 8192, 8192, 64),
    BenchmarkCase("swa_16k", 16384, 16384, 64),
    BenchmarkCase("swa_32k", 32768, 32768, 64),
    BenchmarkCase("swa_64k", 65536, 65536, 64),
    BenchmarkCase("swa_100k", 102400, 102400, 64),
    BenchmarkCase("c4_128", 32, None, 64),
    BenchmarkCase("c4_2k", 512, None, 64),
    BenchmarkCase("c4_4k", 1024, None, 64),
    BenchmarkCase("c4_8k", 2048, None, 64),
    BenchmarkCase("c4_16k", 4096, None, 64),
    BenchmarkCase("c4_32k", 8192, None, 64),
    BenchmarkCase("c4_64k", 16384, None, 64),
    BenchmarkCase("c4_100k", 25600, None, 64),
    BenchmarkCase("c128_8k", 64, None, 2),
    BenchmarkCase("c128_16k", 128, None, 2),
    BenchmarkCase("c128_32k", 256, None, 2),
    BenchmarkCase("c128_64k", 512, None, 2),
    BenchmarkCase("c128_100k", 800, None, 2),
]


def clear_xpu_cache() -> None:
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    gc.collect()


def build_inputs(
    case: BenchmarkCase, batch_size: int, device: str
) -> tuple[Any, ...]:
    """Build the argument tuple for dequantize_and_gather_k_cache."""
    block_size = case.block_size
    blocks_per_req = math.ceil(case.seq_len / block_size)
    total_blocks = batch_size * blocks_per_req

    # Byte values do not affect the timing of this memory-bound kernel, so
    # fill the cache with noise and only constrain the scale exponents.
    block_bytes = block_size * (TOKEN_DATA_BYTES + SCALE_DIM)
    k_cache = torch.randint(
        0, 256, (total_blocks, block_bytes), dtype=torch.uint8, device=device
    )
    k_cache[:, block_size * TOKEN_DATA_BYTES:].random_(
        SCALE_BYTE_MIN, SCALE_BYTE_MAX
    )

    out = torch.empty(
        batch_size,
        case.offset + case.copied_tokens,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    seq_lens = torch.full(
        (batch_size,), case.seq_len, dtype=torch.int32, device=device
    )
    gather_lens = (
        None
        if case.gather_len is None
        else torch.full(
            (batch_size,), case.gather_len, dtype=torch.int32, device=device
        )
    )
    block_table = torch.arange(
        total_blocks, dtype=torch.int32, device=device
    ).view(batch_size, blocks_per_req)

    return (
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        case.offset,
    )


def compute_bandwidth_gbps(
    case: BenchmarkCase, batch_size: int, latency_ms: float
) -> float:
    """Compute effective bandwidth in GB/s.

    Only per-token traffic is counted; the block table and the length
    tensors are orders of magnitude smaller.
    """
    copied_tokens = case.copied_tokens * batch_size
    read_bytes = copied_tokens * (TOKEN_DATA_BYTES + NUM_QUANT_BLOCKS)
    write_bytes = copied_tokens * HEAD_DIM * BF16_BYTES
    return (read_bytes + write_bytes) / (latency_ms * 1e-3) / 1e9


def run_benchmark(
    cases: list[BenchmarkCase],
    batch_size: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        op_args = build_inputs(case, batch_size, "xpu")

        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda a=op_args: torch.ops._C_cache_ops.
            dequantize_and_gather_k_cache(*a),
            quantiles=QUANTILES,
        )
        bw = compute_bandwidth_gbps(case, batch_size, ms)

        results.append(
            {
                "case": case.name,
                "seq_len": case.seq_len,
                "gather": (
                    case.gather_len if case.gather_len is not None else "full"
                ),
                "blk": case.block_size,
                "batch": batch_size,
                "us": f"{ms * 1e3:.1f}",
                "us_p20": f"{min_ms * 1e3:.1f}",
                "us_p80": f"{max_ms * 1e3:.1f}",
                "bw_gbps": f"{bw:.1f}",
            }
        )
        del op_args
        clear_xpu_cache()
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark the dequantize_and_gather_k_cache kernel."
    )
    p.add_argument("--batch-size", type=int, default=1)
    return p


@torch.inference_mode()
def main(args) -> None:
    print(
        f"Benchmarking {len(BUILTIN_CASES)} cases, "
        f"batch_size={args.batch_size}"
    )

    results = run_benchmark(BUILTIN_CASES, args.batch_size)

    print(tabulate(results, headers="keys", tablefmt="github",
                   stralign="right"))


if __name__ == "__main__":
    main(build_parser().parse_args())
