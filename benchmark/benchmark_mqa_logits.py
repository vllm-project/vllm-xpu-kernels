# SPDX-License-Identifier: Apache-2.0
import random
import time
from argparse import ArgumentParser

import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def per_custom_dims_cast_to_fp8(
    x: torch.Tensor,
    dims: tuple[int, ...],
    use_ue8m0: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    excluded_dims = tuple(i for i in range(x.dim()) if i not in set(dims))
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled, sf.squeeze()


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)

    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp8[:, :block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim).view(dtype=torch.uint8)
    x_fp8[:, block_size * head_dim:] = sf.view(
        num_blocks,
        block_size,
    ).view(dtype=torch.uint8)
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def benchmark_fp8_mqa_logits(
    seq_len: int,
    seq_len_kv: int,
    num_heads: int,
    head_dim: int,
    full_kv_range: bool,
    num_warmup_iters: int,
    num_iters: int,
) -> None:
    q = torch.randn(seq_len,
                    num_heads,
                    head_dim,
                    device="xpu",
                    dtype=torch.bfloat16)
    kv = torch.randn(seq_len_kv,
                     head_dim,
                     device="xpu",
                     dtype=torch.bfloat16)
    weights = torch.randn(seq_len,
                          num_heads,
                          device="xpu",
                          dtype=torch.float32)

    if full_kv_range:
        ks = torch.zeros(seq_len, dtype=torch.int32, device="xpu")
        ke = torch.full(
            (seq_len, ),
            seq_len_kv,
            dtype=torch.int32,
            device="xpu",
        )
    else:
        ks = torch.zeros(seq_len, dtype=torch.int32, device="xpu")
        ke = torch.arange(seq_len, dtype=torch.int32, device="xpu") + (
            seq_len_kv - seq_len)

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8, kv_scales = per_custom_dims_cast_to_fp8(kv, (0, ), False)

    def run_once():
        return torch.ops._xpu_C.fp8_mqa_logits(
            q_fp8,
            kv_fp8,
            kv_scales,
            weights,
            ks,
            ke,
        )

    for _ in range(num_warmup_iters):
        run_once()
    torch.xpu.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        run_once()
    torch.xpu.synchronize()
    end = time.perf_counter()

    avg_s = (end - start) / num_iters
    avg_us = avg_s * 1e6

    flops = 2.0 * seq_len * seq_len_kv * num_heads * head_dim
    tflops = flops / avg_s / 1e12

    bytes = (
        seq_len * num_heads * head_dim +  # q
        seq_len_kv * head_dim +  # kv
        seq_len_kv * 4 +  # kv scales
        seq_len * num_heads * 4 +  # weights
        seq_len * 4 +  # ks
        seq_len * 4 +  # ke
        seq_len * seq_len_kv * 4  # output logits
    )
    bandwidth = bytes / avg_s / 1e9

    print("=== fp8_mqa_logits benchmark ===")
    print(
        f"shape: q=({seq_len},{num_heads},{head_dim}), "
        f"kv=({seq_len_kv},{head_dim}), "
        f"full_kv_range={full_kv_range}")
    print(f"avg latency: {avg_us:.3f} us")
    print(f"approx throughput: {tflops:.3f} TFLOPS")
    print(f"approx bandwidth: {bandwidth:.3f} GB/s")


def benchmark_fp8_paged_mqa_logits(
    batch_size: int,
    next_n: int,
    context_len: int,
    heads: int,
    index_dim: int,
    block_size: int,
    max_model_len: int,
    num_warmup_iters: int,
    num_iters: int,
) -> None:
    if block_size != 64:
        raise ValueError("current kernel only supports block_size == 64")

    num_blocks = max_model_len * 2

    q = torch.randn((batch_size, next_n, heads, index_dim),
                    device="xpu",
                    dtype=torch.bfloat16)
    kv_cache = torch.randn((num_blocks, block_size, 1, index_dim),
                           device="xpu",
                           dtype=torch.bfloat16)
    weights = torch.randn((batch_size * next_n, heads),
                          device="xpu",
                          dtype=torch.float32)

    context_lens = torch.full((batch_size, ),
                              context_len,
                              device="xpu",
                              dtype=torch.int32)
    max_blocks = (context_len + block_size - 1) // block_size
    block_tables = torch.zeros((batch_size, max_blocks),
                               device="xpu",
                               dtype=torch.int32)

    block_idx_pool = list(range(num_blocks))
    random.shuffle(block_idx_pool)
    cursor = 0
    for i in range(batch_size):
        for j in range(max_blocks):
            block_tables[i, j] = block_idx_pool[cursor]
            cursor += 1

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache)

    def run_once():
        return torch.ops._xpu_C.fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            context_lens,
            block_tables,
            None,
            max_model_len,
        )

    for _ in range(num_warmup_iters):
        run_once()
    torch.xpu.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        run_once()
    torch.xpu.synchronize()
    end = time.perf_counter()

    avg_s = (end - start) / num_iters
    avg_us = avg_s * 1e6

    effective_kv = context_len - (next_n - 1) / 2.0
    flops = 2.0 * batch_size * next_n * effective_kv * heads * index_dim
    tflops = flops / avg_s / 1e12

    bytes = (
        batch_size * next_n * heads * index_dim +  # q
        batch_size * max_blocks * block_size * (index_dim + 4) +  # kv cache
        batch_size * next_n * heads * 4 +  # weights
        batch_size * 4 +  # context_lens
        batch_size * max_blocks * 4 +  # block_tables
        batch_size * next_n * context_len * 4  # output logits
    )
    bandwidth = bytes / avg_s / 1e9

    print("=== fp8_paged_mqa_logits benchmark ===")
    print(
        f"shape: q=({batch_size},{next_n},{heads},{index_dim}), "
        f"context_len={context_len}, "
        f"block_size={block_size}, max_model_len={max_model_len}")
    print(f"avg latency: {avg_us:.3f} us")
    print(f"approx throughput: {tflops:.3f} TFLOPS")
    print(f"approx bandwidth: {bandwidth:.3f} GB/s")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Benchmark MQA logits kernels on XPU.")
    parser.add_argument("--mode",
                        type=str,
                        choices=["non-paged", "paged"],
                        default="non-paged")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=100)

    # non-paged args
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seq-len-kv", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--full-kv-range", action="store_true")

    # paged args
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--next-n", type=int, default=8)
    parser.add_argument("--context-len", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--index-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=8192)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_device("xpu")

    if args.mode == "non-paged":
        benchmark_fp8_mqa_logits(
            seq_len=args.seq_len,
            seq_len_kv=args.seq_len_kv,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            full_kv_range=args.full_kv_range,
            num_warmup_iters=args.num_warmup_iters,
            num_iters=args.num_iters,
        )
    else:
        benchmark_fp8_paged_mqa_logits(
            batch_size=args.batch_size,
            next_n=args.next_n,
            context_len=args.context_len,
            heads=args.heads,
            index_dim=args.index_dim,
            block_size=args.block_size,
            max_model_len=args.max_model_len,
            num_warmup_iters=args.num_warmup_iters,
            num_iters=args.num_iters,
        )
