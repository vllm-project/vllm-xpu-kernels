# SPDX-License-Identifier: Apache-2.0
# Benchmark paged_decode attention with fp8 KV cache vs higher-precision KV
# cache for DeepSeek-R1-Distill-Qwen-32B with TP=4.
#
# Per-shard shapes (Qwen2.5-32B base):
#   num_query_heads = 40 / TP4 = 10
#   num_kv_heads    = 8  / TP4 = 2
#   head_dim        = 128
# ruff: noqa: E402

# isort: off
import argparse
import gc
import itertools

import torch

from utils import bootstrap_benchmark_env

bootstrap_benchmark_env(__file__)

from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
# isort: on

DEVICE = "xpu"


def clear_xpu_cache():
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    gc.collect()


def make_inputs(num_seqs, kv_len, num_query_heads, num_kv_heads, head_dim,
                block_size, num_blocks, out_dtype, kv_cache_dtype):
    query_lens = [1] * num_seqs
    kv_lens = [kv_len] * num_seqs
    max_kv_len = max(kv_lens)
    max_query_len = max(query_lens)
    scale = head_dim**-0.5

    query = torch.randn(sum(query_lens),
                        num_query_heads,
                        head_dim,
                        dtype=out_dtype,
                        device=DEVICE)
    key_cache = torch.randn(num_blocks,
                            block_size,
                            num_kv_heads,
                            head_dim,
                            dtype=out_dtype,
                            device=DEVICE)
    value_cache = torch.randn_like(key_cache)

    cu_query_lens = torch.tensor([0] + query_lens,
                                 dtype=torch.int32,
                                 device=DEVICE).cumsum(dim=0,
                                                       dtype=torch.int32)
    seq_k = torch.tensor(kv_lens, dtype=torch.int32, device=DEVICE)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 num_blocks,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32,
                                 device=DEVICE)

    k_descale = None
    v_descale = None
    if kv_cache_dtype is not None:
        scale_shape = (num_seqs, num_kv_heads)
        k_descale_scalar = (torch.abs(key_cache).max() / 200).to(torch.float32)
        v_descale_scalar = (torch.abs(value_cache).max() / 200).to(
            torch.float32)
        key_cache = (key_cache / k_descale_scalar).to(kv_cache_dtype)
        value_cache = (value_cache / v_descale_scalar).to(kv_cache_dtype)
        k_descale = k_descale_scalar.expand(scale_shape)
        v_descale = v_descale_scalar.expand(scale_shape)

    return (query, key_cache, value_cache, max_query_len, cu_query_lens,
            max_kv_len, seq_k, scale, block_tables, k_descale, v_descale)


def kv_cache_bytes(num_seqs, kv_len, num_kv_heads, head_dim, dtype_bytes):
    return 2 * num_seqs * kv_len * num_kv_heads * head_dim * dtype_bytes


def benchmark_one(num_seqs, kv_len, num_query_heads, num_kv_heads, head_dim,
                  block_size, num_blocks, out_dtype, kv_cache_dtype,
                  iterations):
    (query, key_cache, value_cache, max_query_len, cu_query_lens, max_kv_len,
     seq_k, scale, block_tables, k_descale,
     v_descale) = make_inputs(num_seqs, kv_len, num_query_heads, num_kv_heads,
                              head_dim, block_size, num_blocks, out_dtype,
                              kv_cache_dtype)

    # Warmup
    for _ in range(20):
        flash_attn_varlen_func(query,
                               key_cache,
                               value_cache,
                               max_query_len,
                               cu_query_lens,
                               max_kv_len,
                               seqused_k=seq_k,
                               softmax_scale=scale,
                               causal=False,
                               block_table=block_tables,
                               window_size=(-1, -1),
                               k_descale=k_descale,
                               v_descale=v_descale)
    torch.xpu.synchronize()

    start_events = [
        torch.xpu.Event(enable_timing=True) for _ in range(iterations)
    ]
    end_events = [
        torch.xpu.Event(enable_timing=True) for _ in range(iterations)
    ]
    max_num_blocks_per_seq = block_tables.shape[1]
    num_seqs = block_tables.shape[0]
    for i in range(iterations):
        # random block_tables for each iteration to avoid any caching effects
        block_tables = torch.randint(0,
                                     num_blocks,
                                     (num_seqs, max_num_blocks_per_seq),
                                     dtype=torch.int32,
                                     device=DEVICE)
        start_events[i].record()
        flash_attn_varlen_func(query,
                               key_cache,
                               value_cache,
                               max_query_len,
                               cu_query_lens,
                               max_kv_len,
                               seqused_k=seq_k,
                               softmax_scale=scale,
                               causal=False,
                               block_table=block_tables,
                               window_size=(-1, -1),
                               k_descale=k_descale,
                               v_descale=v_descale)
        end_events[i].record()
    torch.xpu.synchronize()
    times_us = sorted(start_events[i].elapsed_time(end_events[i]) * 1000.0
                      for i in range(iterations))
    # Trim the highest 20% to suppress outliers from one-off allocator stalls;
    # report median of remaining samples.
    keep = times_us[: max(1, int(iterations * 0.8))]
    avg_us = keep[len(keep) // 2]
    avg_ms = avg_us / 1000.0

    dtype_bytes = (torch.tensor([], dtype=kv_cache_dtype).element_size()
                   if kv_cache_dtype is not None
                   else torch.tensor([], dtype=out_dtype).element_size())
    kv_bytes = kv_cache_bytes(num_seqs, kv_len, num_kv_heads, head_dim,
                              dtype_bytes)
    bw_gbs = kv_bytes / (avg_ms / 1000.0) / (1024**3)

    clear_xpu_cache()
    return avg_us, bw_gbs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    torch.manual_seed(1234)
    torch.set_default_device(DEVICE)
    torch.xpu.set_device("xpu:0")

    # DeepSeek-R1-Distill-Qwen-32B TP=4 per-shard shapes
    num_query_heads = 10
    num_kv_heads = 2
    head_dim = 128

    out_dtype = torch.bfloat16
    block_size = 64
    num_blocks = 32768

    batch_sizes = [32, 48, 64]
    kv_lens = [1024, 4096, 8192, 16384]
    kv_cache_dtypes = [None, torch.float8_e5m2, torch.float8_e4m3fn]

    print(f"Device: {torch.xpu.get_device_name()}")
    print(f"Model: DeepSeek-R1-Distill-Qwen-32B (TP=4)")
    print(f"num_q_heads={num_query_heads}, num_kv_heads={num_kv_heads}, "
          f"head_dim={head_dim}, block_size={block_size}, "
          f"out_dtype={out_dtype}")
    print()

    header = (f"{'batch':>5} {'kv_len':>7} {'kv_dtype':>14} "
              f"{'latency(us)':>12} {'bw(GB/s)':>10} {'speedup':>9}")
    print(header)
    print("-" * len(header))

    rows = []
    for bs, kv_len in itertools.product(batch_sizes, kv_lens):
        baseline_us = None
        for kv_dtype in kv_cache_dtypes:
            try:
                avg_us, bw = benchmark_one(num_seqs=bs,
                                           kv_len=kv_len,
                                           num_query_heads=num_query_heads,
                                           num_kv_heads=num_kv_heads,
                                           head_dim=head_dim,
                                           block_size=block_size,
                                           num_blocks=num_blocks,
                                           out_dtype=out_dtype,
                                           kv_cache_dtype=kv_dtype,
                                           iterations=args.iterations)
            except Exception as e:
                print(f"  failed bs={bs} kv_len={kv_len} dtype={kv_dtype}: "
                      f"{e}")
                continue
            if kv_dtype is None:
                baseline_us = avg_us
                speedup_str = "1.00x"
            else:
                speedup_str = (f"{baseline_us / avg_us:.2f}x"
                               if baseline_us is not None else "-")
            name = "bf16" if kv_dtype is None else str(kv_dtype).replace(
                "torch.", "")
            row = (bs, kv_len, name, avg_us, bw, speedup_str)
            rows.append(row)
            print(f"{bs:>5} {kv_len:>7} {name:>14} "
                  f"{avg_us:>12.2f} {bw:>10.2f} {speedup_str:>9}")
        print()


if __name__ == "__main__":
    main()
