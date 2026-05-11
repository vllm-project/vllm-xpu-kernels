"""
Decode attention benchmark for batch > 1 configs.
Tests correctness and performance of per-seq adaptive split-K.

Usage:
  python benchmark/bench_decode_batch.py
  python benchmark/bench_decode_batch.py --iterations 500
  python benchmark/bench_decode_batch.py --only-perf
"""
import argparse
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from utils import bootstrap_benchmark_env
bootstrap_benchmark_env(__file__)

from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
from tests.flash_attn.test_flash_attn_varlen_func import ref_paged_attn
from tests.utils import seed_everything

DEVICE = "xpu"
DTYPE = torch.bfloat16

# ============================================================
# Test configurations: (seq_lens_str, num_heads, head_size, block_size)
# seq_lens_str format: "num_seqs,query_lens,kv_lens"
# ============================================================
CONFIGS = [
    # Uniform KV, batch=32, short seq (per-seq: all splits=1, skip reduce)
    ("32," + "+".join(["1"]*32) + "," + "+".join(["512"]*32),
     (32, 2), 128, 64, "32×512 uniform"),

    # Uniform KV, batch=32, medium seq
    ("32," + "+".join(["1"]*32) + "," + "+".join(["4096"]*32),
     (32, 2), 128, 64, "32×4096 uniform"),

    # Mixed KV, batch=8 (the key optimization target)
    ("8,1+1+1+1+1+1+1+1,128+256+512+1024+2048+4096+8192+16384",
     (32, 2), 128, 64, "8×mixed [128..16384]"),

    # Mixed KV, batch=8, different GQA
    ("8,1+1+1+1+1+1+1+1,128+256+512+1024+2048+4096+8192+16384",
     (32, 4), 128, 64, "8×mixed (32,4)"),

    # Skewed: mostly short, one long
    ("8,1+1+1+1+1+1+1+1,256+256+256+256+256+256+256+16384",
     (32, 2), 128, 64, "8×skewed [256..16384]"),

    # All short (per-seq: all splits=1)
    ("8,1+1+1+1+1+1+1+1,128+128+256+256+512+512+1024+1024",
     (32, 2), 128, 64, "8×short [128..1024]"),

    # Realistic: vLLM-like mixed decode batch
    ("16," + "+".join(["1"]*16) + "," +
     "+".join(["256","512","1024","1024","2048","2048","4096","4096",
               "4096","8192","8192","8192","16384","16384","16384","16384"]),
     (32, 2), 128, 64, "16×realistic mixed"),

    # block_size=128 variant
    ("8,1+1+1+1+1+1+1+1,128+256+512+1024+2048+4096+8192+16384",
     (32, 2), 128, 128, "8×mixed bs=128"),

    # GQA=4 (40,8)
    ("8,1+1+1+1+1+1+1+1,128+256+512+1024+2048+4096+8192+16384",
     (40, 8), 128, 64, "8×mixed (40,8)"),
]


def make_inputs(seq_lens_str, num_heads, head_size, block_size):
    """Create input tensors for decode benchmark."""
    parts = seq_lens_str.split(",")
    num_seqs = int(parts[0])
    query_lens = list(map(int, parts[1].split("+")))
    kv_lens = list(map(int, parts[2].split("+")))

    num_query_heads, num_kv_heads = num_heads
    num_blocks = 2048
    max_kv_len = max(kv_lens)
    max_query_len = max(query_lens)
    scale = head_size ** -0.5
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=DTYPE)
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=DTYPE)
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    seq_k = torch.tensor(kv_lens, dtype=torch.int32)
    block_tables = torch.randint(0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32)

    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "max_query_len": max_query_len,
        "cu_query_lens": cu_query_lens,
        "max_kv_len": max_kv_len,
        "seq_k": seq_k,
        "scale": scale,
        "block_tables": block_tables,
        "query_lens": query_lens,
        "kv_lens": kv_lens,
        "num_seqs": num_seqs,
        "num_blocks": num_blocks,
        "max_num_blocks_per_seq": max_num_blocks_per_seq,
    }


def check_correctness(inputs, num_splits_kv=None):
    """Compare kernel output against reference implementation."""
    query = inputs["query"]
    key_cache = inputs["key_cache"]
    value_cache = inputs["value_cache"]

    # Reference: splits=1 (single-pass, no reduce, most numerically accurate)
    ref_output = flash_attn_varlen_func(
        query, key_cache, value_cache,
        inputs["max_query_len"], inputs["cu_query_lens"], inputs["max_kv_len"],
        seqused_k=inputs["seq_k"], softmax_scale=inputs["scale"],
        causal=False, block_table=inputs["block_tables"],
        window_size=(-1, -1),
        num_splits_kv=1)
    torch.xpu.synchronize()

    # Test output with requested splits (exercises per-seq path)
    test_output = flash_attn_varlen_func(
        query, key_cache, value_cache,
        inputs["max_query_len"], inputs["cu_query_lens"], inputs["max_kv_len"],
        seqused_k=inputs["seq_k"], softmax_scale=inputs["scale"],
        causal=False, block_table=inputs["block_tables"],
        window_size=(-1, -1),
        num_splits_kv=num_splits_kv)
    torch.xpu.synchronize()

    out = test_output.float() if isinstance(test_output, torch.Tensor) else test_output[0].float()
    ref = ref_output.float() if isinstance(ref_output, torch.Tensor) else ref_output[0].float()

    max_diff = (out - ref).abs().max().item()
    mean_diff = (out - ref).abs().mean().item()
    return max_diff, mean_diff


def bench_perf(inputs, num_splits_kv, iterations, warmup=10):
    """Measure kernel performance using batch GPU event timing."""
    query = inputs["query"]
    key_cache = inputs["key_cache"]
    value_cache = inputs["value_cache"]
    num_seqs = inputs["num_seqs"]
    num_blocks = inputs["num_blocks"]
    max_num_blocks_per_seq = inputs["max_num_blocks_per_seq"]

    # Pre-generate queries and block_tables
    queries = [torch.rand_like(query) for _ in range(iterations + warmup)]
    block_tables_list = [
        torch.randint(0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32)
        for _ in range(iterations + warmup)
    ]

    # Warmup
    for i in range(warmup):
        flash_attn_varlen_func(
            queries[i], key_cache, value_cache,
            inputs["max_query_len"], inputs["cu_query_lens"], inputs["max_kv_len"],
            seqused_k=inputs["seq_k"], softmax_scale=inputs["scale"],
            causal=False, block_table=block_tables_list[i],
            window_size=(-1, -1), num_splits_kv=num_splits_kv)
    torch.xpu.synchronize()

    # Timed
    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)

    start_event.record()
    for i in range(warmup, warmup + iterations):
        flash_attn_varlen_func(
            queries[i], key_cache, value_cache,
            inputs["max_query_len"], inputs["cu_query_lens"], inputs["max_kv_len"],
            seqused_k=inputs["seq_k"], softmax_scale=inputs["scale"],
            causal=False, block_table=block_tables_list[i],
            window_size=(-1, -1), num_splits_kv=num_splits_kv)
    end_event.record()
    torch.xpu.synchronize()

    gpu_time_ms = start_event.elapsed_time(end_event)
    avg_us = gpu_time_ms * 1000.0 / iterations
    return avg_us


def main():
    parser = argparse.ArgumentParser(description="Batch decode benchmark")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--only-perf", action="store_true", help="Skip correctness")
    parser.add_argument("--only-correctness", action="store_true", help="Skip perf")
    args = parser.parse_args()

    seed_everything(42)
    torch.set_default_device(DEVICE)
    torch.xpu.set_device("xpu:0")

    print("=" * 90)
    print("Batch Decode Benchmark (per-seq adaptive split-K)")
    print(f"iterations={args.iterations}, dtype={DTYPE}")
    print("=" * 90)

    # ---- Correctness ----
    if not args.only_perf:
        print("\n--- Correctness ---")
        print(f"{'config':<30} | {'splits':>6} | {'max_diff':>9} {'mean_diff':>10} | {'status':>6}")
        print("-" * 80)

        all_pass = True
        for seq_str, heads, head_size, block_size, name in CONFIGS:
            inputs = make_inputs(seq_str, heads, head_size, block_size)
            try:
                max_d, mean_d = check_correctness(inputs, num_splits_kv=None)
                status = "PASS" if max_d < 0.1 else "WARN" if max_d < 0.5 else "FAIL"
                if status == "FAIL":
                    all_pass = False
                print(f"{name:<30} | {'auto':>6} | {max_d:>9.4f} {mean_d:>10.6f} | {status:>6}")
            except Exception as e:
                all_pass = False
                print(f"{name:<30} | {'auto':>6} | {'ERROR':>9} {str(e)[:30]:>30}")
            torch.xpu.empty_cache()

        print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
        print()

    # ---- Performance ----
    if not args.only_correctness:
        print("\n--- Performance ---")
        print(f"{'config':<30} | {'batch':>5} {'kv_sum':>7} | {'time(us)':>9} {'BW(GB/s)':>9}")
        print("-" * 80)

        for seq_str, heads, head_size, block_size, name in CONFIGS:
            inputs = make_inputs(seq_str, heads, head_size, block_size)
            num_query_heads, num_kv_heads = heads
            kv_lens = inputs["kv_lens"]
            kv_sum = sum(kv_lens)

            try:
                avg_us = bench_perf(inputs, num_splits_kv=None,
                                    iterations=args.iterations)
                # Bandwidth: KV bytes read
                kv_bytes = kv_sum * num_kv_heads * head_size * 2 * 2  # K+V, bf16
                bw_gbs = (kv_bytes / 1e9) / (avg_us / 1e6)
                print(f"{name:<30} | {inputs['num_seqs']:>5} {kv_sum:>7} | {avg_us:>9.1f} {bw_gbs:>9.1f}")
            except Exception as e:
                print(f"{name:<30} | {inputs['num_seqs']:>5} {kv_sum:>7} | {'ERROR':>9} {str(e)[:20]}")
            torch.xpu.empty_cache()

        print()
    print("=" * 90)


if __name__ == "__main__":
    main()
