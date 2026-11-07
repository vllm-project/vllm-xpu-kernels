# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

import itertools
from argparse import ArgumentParser

import torch
from utils import bootstrap_benchmark_env

bootstrap_benchmark_env(__file__)

from tests.ops.moe_align_block_size_ops import (batched_moe_align_block_size,
                                                moe_align_block_size)

# isort: on

WARMUP = 10
ITERS = 100


def bench_us(fn) -> float:
    """Time a callable with torch.xpu.Event and return avg microseconds."""
    for _ in range(WARMUP):
        fn()
    torch.xpu.synchronize()

    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)

    start_event.record()
    for _ in range(ITERS):
        fn()
    end_event.record()
    torch.xpu.synchronize()

    # elapsed_time returns milliseconds; convert to per-iter microseconds.
    return start_event.elapsed_time(end_event) * 1000.0 / ITERS


def get_topk_ids(num_tokens: int, num_experts: int, topk: int) -> torch.Tensor:
    return torch.stack([
        torch.randperm(num_experts, dtype=torch.int32, device="xpu")[:topk]
        for _ in range(num_tokens)
    ])


# moe_align_block_size configurations
num_tokens_range = [1, 16, 256, 4096]
num_experts_range = [16, 64, 224, 256, 280, 512]
topk_range = [1, 2, 8]
ep_size_range = [1, 8]
configs = list(
    itertools.product(num_tokens_range, num_experts_range, topk_range,
                      ep_size_range))


def run_moe_align(num_tokens, num_experts, topk, ep_size) -> float:
    block_size = 256
    topk_ids = get_topk_ids(num_tokens, num_experts, topk)

    e_map = None
    if ep_size != 1:
        local_e = num_experts // ep_size
        e_ids = torch.randperm(num_experts, device="xpu",
                               dtype=torch.int32)[:local_e]
        e_map = torch.full((num_experts, ),
                           -1,
                           device="xpu",
                           dtype=torch.int32)
        e_map[e_ids] = torch.arange(local_e, device="xpu", dtype=torch.int32)

    return bench_us(lambda: moe_align_block_size(topk_ids,
                                                 block_size,
                                                 num_experts,
                                                 e_map,
                                                 ignore_invalid_experts=True))


def benchmark_moe_align():
    print(f"{'num_tokens':>10} {'num_experts':>12} {'topk':>5} "
          f"{'ep_size':>8} {'time(us)':>10}")
    for num_tokens, num_experts, topk, ep_size in configs:
        us = run_moe_align(num_tokens, num_experts, topk, ep_size)
        print(f"{num_tokens:>10} {num_experts:>12} {topk:>5} "
              f"{ep_size:>8} {us:>10.3f}")


# batched_moe_align_block_size configurations
batched_configs = list(
    itertools.product(
        [8, 32, 64, 256],
        [16, 256, 512, 2048],
        [8, 32, 48, 64],
        ["uniform", "skewed", "mostly_empty"],
    ))


def run_batched_moe_align(num_batches, max_tokens_per_batch, block_size,
                          distribution) -> float:
    if distribution == "uniform":
        expert_num_tokens = torch.full((num_batches, ),
                                       max_tokens_per_batch,
                                       dtype=torch.int32,
                                       device="xpu")
    elif distribution == "skewed":
        expert_num_tokens = torch.ones((num_batches, ),
                                       dtype=torch.int32,
                                       device="xpu")
        expert_num_tokens[0] = max_tokens_per_batch
    else:
        expert_num_tokens = torch.zeros((num_batches, ),
                                        dtype=torch.int32,
                                        device="xpu")
        expert_num_tokens[::8] = max_tokens_per_batch

    return bench_us(lambda: batched_moe_align_block_size(
        max_tokens_per_batch, block_size, expert_num_tokens))


def benchmark_batched_moe_align():
    print(f"{'num_batches':>12} {'max_tokens':>11} {'block_size':>11} "
          f"{'distribution':>14} {'time(us)':>10}")
    for num_batches, max_tokens_per_batch, block_size, dist in batched_configs:
        us = run_batched_moe_align(num_batches, max_tokens_per_batch,
                                   block_size, dist)
        print(f"{num_batches:>12} {max_tokens_per_batch:>11} "
              f"{block_size:>11} {dist:>14} {us:>10.3f}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Benchmark the moe_align_block_size kernel.")
    parser.add_argument(
        "--batched",
        action="store_true",
        help="Benchmark batched_moe_align_block_size instead.",
    )
    args = parser.parse_args()

    if args.batched:
        benchmark_batched_moe_align()
    else:
        benchmark_moe_align()
