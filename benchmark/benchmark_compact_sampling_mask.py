#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""
Benchmark comparing Sycl vs PyTorch implementations of compact_sampling_mask.

Compares:
- compact_sampling_mask_xpu (Sycl kernel)
- compact_sampling_mask_torch (pure-PyTorch reference)
- compact_sampling_mask_triton (vllm's Triton kernel), only when
  USE_TRITON_COMPACT_SAMPLING_MASK=1 and vllm (with Triton) is importable.

Scenarios (per batch/vocab size):
- dense: (almost) all logits finite
- sparse: only a small fraction of logits finite (post top_k/top_p filtering)
- half_inactive: half the rows have no sampled token (num_sampled_tokens == 0)
"""

import argparse
import gc
from dataclasses import dataclass

import torch
from utils import bootstrap_benchmark_env

bootstrap_benchmark_env(__file__)

from tests.ops.compact_sampling_mask_op import (TRITON_AVAILABLE,
                                                compact_sampling_mask_torch,
                                                compact_sampling_mask_triton,
                                                compact_sampling_mask_xpu)

MAX_NUM_KEPT = 64


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str
    batch_size: int
    vocab_size: int
    max_num_kept: int
    finite_fraction: float
    inactive_fraction: float
    description: str


def create_logits(
    batch_size: int,
    vocab_size: int,
    finite_fraction: float,
    device: str = "xpu",
) -> torch.Tensor:
    """Create random logits with a controllable fraction of finite entries,
    mimicking rows that already went through top_k/top_p filtering."""
    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32,
                         device=device)
    if finite_fraction < 1.0:
        drop_mask = torch.rand(batch_size, vocab_size,
                               device=device) >= finite_fraction
        logits = logits.masked_fill(drop_mask, float("-inf"))
    return logits


def create_num_sampled_tokens(
    batch_size: int,
    inactive_fraction: float,
    device: str = "xpu",
) -> torch.Tensor:
    num_sampled_tokens = torch.ones(batch_size, dtype=torch.int32,
                                    device=device)
    num_inactive = int(round(batch_size * inactive_fraction))
    if num_inactive > 0:
        num_sampled_tokens[:num_inactive] = 0
    return num_sampled_tokens


def measure_memory() -> tuple[int, int]:
    """Return (allocated, reserved) memory in bytes."""
    torch.xpu.synchronize()
    return torch.xpu.memory_allocated(), torch.xpu.max_memory_allocated()


def reset_memory_stats():
    """Reset peak memory statistics."""
    torch.xpu.reset_peak_memory_stats()
    torch.xpu.empty_cache()
    gc.collect()


def benchmark_function(
    func: str,
    logits: torch.Tensor,
    num_sampled_tokens: torch.Tensor,
    max_num_kept: int,
    warmup_iters: int = 5,
    benchmark_iters: int = 20,
) -> tuple[float, int]:
    """
    Benchmark a function and return (avg_time_ms, peak_memory_bytes).
    """
    # Warmup
    for _ in range(warmup_iters):
        if func == "sycl":
            compact_sampling_mask_xpu(logits, num_sampled_tokens,
                                      max_num_kept)
        elif func == "pytorch":
            compact_sampling_mask_torch(logits, num_sampled_tokens,
                                        max_num_kept)
        elif func == "triton":
            compact_sampling_mask_triton(logits, num_sampled_tokens,
                                         max_num_kept)
    torch.xpu.synchronize()

    # Reset memory stats before benchmark
    reset_memory_stats()

    # Benchmark
    start_events = [
        torch.xpu.Event(enable_timing=True) for _ in range(benchmark_iters)
    ]
    end_events = [
        torch.xpu.Event(enable_timing=True) for _ in range(benchmark_iters)
    ]

    for i in range(benchmark_iters):
        start_events[i].record()
        if func == "sycl":
            compact_sampling_mask_xpu(logits, num_sampled_tokens,
                                      max_num_kept)
        elif func == "pytorch":
            compact_sampling_mask_torch(logits, num_sampled_tokens,
                                        max_num_kept)
        elif func == "triton":
            compact_sampling_mask_triton(logits, num_sampled_tokens,
                                         max_num_kept)
        end_events[i].record()

    torch.xpu.synchronize()

    # Calculate timing
    times = [
        start_events[i].elapsed_time(end_events[i])
        for i in range(benchmark_iters)
    ]
    avg_time = sum(times) / len(times)

    # Get peak memory
    _, peak_memory = measure_memory()

    return avg_time, peak_memory


def create_benchmark_configs(
    batch_sizes: list[int],
    vocab_sizes: list[int],
) -> list[BenchmarkConfig]:
    """Create all benchmark configurations."""
    configs = []

    for vocab_size in vocab_sizes:
        for batch_size in batch_sizes:
            configs.append(
                BenchmarkConfig(
                    name=f"dense_b{batch_size}_v{vocab_size // 1000}k",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    max_num_kept=MAX_NUM_KEPT,
                    finite_fraction=1.0,
                    inactive_fraction=0.0,
                    description=f"Dense (all finite), "
                    f"batch={batch_size}, vocab={vocab_size}",
                ))

            configs.append(
                BenchmarkConfig(
                    name=f"sparse_b{batch_size}_v{vocab_size // 1000}k",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    max_num_kept=MAX_NUM_KEPT,
                    finite_fraction=0.05,
                    inactive_fraction=0.0,
                    description=
                    f"Sparse (5% finite, post top_k/top_p filtering), "
                    f"batch={batch_size}, vocab={vocab_size}",
                ))

            configs.append(
                BenchmarkConfig(
                    name=f"half_inactive_b{batch_size}_v{vocab_size // 1000}k",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    max_num_kept=MAX_NUM_KEPT,
                    finite_fraction=1.0,
                    inactive_fraction=0.5,
                    description=f"Half inactive (no sampled token), "
                    f"batch={batch_size}, vocab={vocab_size}",
                ))

    return configs


def format_memory(bytes_val: int) -> str:
    """Format memory in human-readable form."""
    if bytes_val >= 1024**3:
        return f"{bytes_val / (1024**3):.2f} GB"
    elif bytes_val >= 1024**2:
        return f"{bytes_val / (1024**2):.2f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.2f} KB"
    return f"{bytes_val} B"


def run_benchmark(
    configs: list[BenchmarkConfig],
    warmup_iters: int = 5,
    benchmark_iters: int = 20,
    verbose: bool = True,
):
    """Run all benchmarks and print results."""
    results = []

    print("=" * 100)
    label = "compact_sampling_mask Benchmark: Sycl vs PyTorch"
    if TRITON_AVAILABLE:
        label += " vs Triton"
    print(label)
    print("=" * 100)
    print()

    for config in configs:
        if verbose:
            print(f"Running: {config.description}")

        logits = create_logits(config.batch_size, config.vocab_size,
                               config.finite_fraction)
        num_sampled_tokens = create_num_sampled_tokens(
            config.batch_size, config.inactive_fraction)

        # Benchmark Sycl
        reset_memory_stats()
        sycl_time, sycl_mem = benchmark_function(
            "sycl",
            logits,
            num_sampled_tokens,
            config.max_num_kept,
            warmup_iters,
            benchmark_iters,
        )

        # Benchmark PyTorch
        reset_memory_stats()
        pytorch_time, pytorch_mem = benchmark_function(
            "pytorch",
            logits,
            num_sampled_tokens,
            config.max_num_kept,
            warmup_iters,
            benchmark_iters,
        )

        speedup = pytorch_time / sycl_time if sycl_time > 0 else float("inf")
        mem_ratio = pytorch_mem / sycl_mem if sycl_mem > 0 else float("inf")

        result = {
            "config": config,
            "sycl_time_ms": sycl_time,
            "pytorch_time_ms": pytorch_time,
            "sycl_mem": sycl_mem,
            "pytorch_mem": pytorch_mem,
            "speedup": speedup,
            "mem_ratio": mem_ratio,
        }

        # Optional: benchmark vllm's own Triton kernel too.
        if TRITON_AVAILABLE:
            reset_memory_stats()
            triton_time, triton_mem = benchmark_function(
                "triton",
                logits,
                num_sampled_tokens,
                config.max_num_kept,
                warmup_iters,
                benchmark_iters,
            )
            result["triton_time_ms"] = triton_time
            result["triton_mem"] = triton_mem
            result["triton_speedup"] = (
                triton_time / sycl_time if sycl_time > 0 else float("inf"))

        results.append(result)

        if verbose:
            print(f"  Sycl:  {sycl_time:.3f} ms, {format_memory(sycl_mem)}")
            print(
                f"  PyT: {pytorch_time:.3f} ms, {format_memory(pytorch_mem)}")
            print(f"  Speedup: {speedup:.2f}x, Memory ratio: {mem_ratio:.2f}x")
            if TRITON_AVAILABLE:
                print(f"  Triton: {result['triton_time_ms']:.3f} ms, "
                      f"{format_memory(result['triton_mem'])} "
                      f"(Sycl speedup: {result['triton_speedup']:.2f}x)")
            print()

        # Clean up
        del logits, num_sampled_tokens
        reset_memory_stats()

    return results


def print_summary_table(results: list[dict]):
    """Print a summary table of results."""
    print()
    print("=" * 130)
    print("SUMMARY TABLE")
    print("=" * 130)
    print()

    has_triton = bool(results) and "triton_time_ms" in results[0]

    # Header
    header = (f"{'Scenario':<40} {'Batch':>6} {'Vocab':>7} "
              f"{'Sycl (ms)':>12} {'PyTorch (ms)':>13} {'Speedup':>8} "
              f"{'Sycl Mem':>10} {'Pyt Mem':>10}")
    if has_triton:
        header += f" {'Triton (ms)':>12} {'SyclVsTr':>9}"
    print(header)
    print("-" * 130)

    # Group by scenario type
    current_vocab = None
    for result in results:
        config = result["config"]

        # Add separator between vocab sizes
        if current_vocab != config.vocab_size:
            if current_vocab is not None:
                print("-" * 130)
            current_vocab = config.vocab_size

        scenario = config.name.split("_b")[0]  # Extract scenario name
        row = (f"{scenario:<40} {config.batch_size:>6} {config.vocab_size:>7} "
              f"{result['sycl_time_ms']:>12.3f} "
              f"{result['pytorch_time_ms']:>13.3f} "
              f"{result['speedup']:>7.2f}x "
              f"{format_memory(result['sycl_mem']):>10} "
              f"{format_memory(result['pytorch_mem']):>10}")
        if has_triton:
            row += (f" {result['triton_time_ms']:>12.3f} "
                    f"{result['triton_speedup']:>8.2f}x")
        print(row)

    print("=" * 130)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Sycl vs PyTorch compact_sampling_mask")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64, 128, 512, 1024, 2048],
        help="Batch sizes to test (default: 1 4 16 64 128 512 1024 2048)",
    )
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=[32768, 131072],  # 32k, 128k
        help="Vocabulary sizes to test (default: 32768 131072)",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=20,
        help="Number of benchmark iterations (default: 20)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary table",
    )

    args = parser.parse_args()

    # Print configuration
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Vocab sizes: {args.vocab_sizes}")
    print(f"Warmup iterations: {args.warmup_iters}")
    print(f"Benchmark iterations: {args.benchmark_iters}")
    if TRITON_AVAILABLE:
        print("Triton comparison: enabled (USE_TRITON_COMPACT_SAMPLING_MASK)")
    else:
        print("Triton comparison: disabled (set "
              "USE_TRITON_COMPACT_SAMPLING_MASK=1 to enable, requires vllm "
              "with Triton)")
    print()

    if not torch.xpu.is_available():
        print("ERROR: XPU is not available. This benchmark requires a GPU.")
        return

    device_name = torch.xpu.get_device_name(0)
    print(f"GPU: {device_name}")
    print()

    # Create configs
    configs = create_benchmark_configs(
        args.batch_sizes,
        args.vocab_sizes,
    )

    # Run benchmarks
    results = run_benchmark(
        configs,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
        verbose=not args.quiet,
    )

    # Print summary
    print_summary_table(results)


if __name__ == "__main__":
    main()
