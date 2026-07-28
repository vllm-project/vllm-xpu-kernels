# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark for fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn kernel.

Measures throughput (GB/s) of the fused DeepSeek-V4 indexer MXFP4 path:
  Online Softmax Compression → RMSNorm → RoPE → MXFP4 quant → KV cache insert.

Usage:
  python benchmark/benchmark_compress_insert_mxfp4.py
  python benchmark/benchmark_compress_insert_mxfp4.py --compress-ratio 4
  python benchmark/benchmark_compress_insert_mxfp4.py --check-correctness
"""

import argparse

import torch

try:
    from tests import register_ops as ops  # noqa: F401
except (ImportError, ModuleNotFoundError):
    import vllm_xpu_kernels._xpu_C  # noqa: F401

HEAD_DIM = 128
ROPE_DIM = 64
BLOCK_SIZE = 16
RMS_EPS = 1e-6
TOKEN_STRIDE = HEAD_DIM // 2  # 64
SCALE_DIM = HEAD_DIM // 32  # 4
QUANT_BLOCK = 32

WARMUP = 10
REPEAT = 50


def setup_inputs(num_tokens, kv_block_size, compress_ratio, overlap, device):
    """Create input tensors for the kernel."""
    coff = 1 + overlap
    state_width = coff * HEAD_DIM

    num_pages = (compress_ratio * num_tokens - 1) // BLOCK_SIZE + 2
    state_cache = torch.randn(
        num_pages, BLOCK_SIZE, 2 * state_width,
        dtype=torch.bfloat16, device=device,
    )
    block_table = torch.arange(
        num_pages, dtype=torch.int32, device=device
    ).unsqueeze(0)
    token_to_req = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    positions = torch.arange(
        compress_ratio - 1, compress_ratio * num_tokens, compress_ratio,
        dtype=torch.int64, device=device,
    )
    rms_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)
    cos_sin_cache = torch.randn(
        compress_ratio * num_tokens, ROPE_DIM, device=device
    )

    kv_n_blocks = (num_tokens + kv_block_size - 1) // kv_block_size + 1
    kv_cache = torch.zeros(
        kv_n_blocks, kv_block_size * (TOKEN_STRIDE + SCALE_DIM),
        dtype=torch.uint8, device=device,
    )

    return (state_cache, token_to_req, positions, slot_mapping, block_table,
            BLOCK_SIZE, state_width, rms_weight, RMS_EPS, cos_sin_cache,
            kv_cache, slot_mapping, kv_block_size, HEAD_DIM, ROPE_DIM,
            compress_ratio, overlap, QUANT_BLOCK)


def run_kernel(args):
    """Run the kernel with given args tuple."""
    torch.ops._xpu_C.fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn(
        *args)


def _bytes_per_input_set(num_tokens, kv_block_size, compress_ratio, overlap):
    """Approximate device bytes for one input set (dominated by state_cache)."""
    coff = 1 + overlap
    state_width = coff * HEAD_DIM
    num_pages = (compress_ratio * num_tokens - 1) // BLOCK_SIZE + 2
    state_bytes = num_pages * BLOCK_SIZE * (2 * state_width) * 2  # bf16
    kv_n_blocks = (num_tokens + kv_block_size - 1) // kv_block_size + 1
    kv_bytes = kv_n_blocks * kv_block_size * (TOKEN_STRIDE + SCALE_DIM)
    return state_bytes + kv_bytes


# Rotate enough distinct input sets to exceed on-chip (L2) cache; otherwise
# state_cache stays L2-resident across REPEAT and BW reflects L2, not HBM.
L2_BYPASS_BYTES = 512 * 1024 * 1024
MAX_INPUT_SETS = 64


def _num_input_sets(num_tokens, kv_block_size, compress_ratio, overlap):
    per = _bytes_per_input_set(
        num_tokens, kv_block_size, compress_ratio, overlap)
    n = (L2_BYPASS_BYTES + per - 1) // per
    return max(1, min(MAX_INPUT_SETS, n))


def benchmark_config(num_tokens, kv_block_size, compress_ratio, device):
    """Benchmark a single configuration, return (time_us, gbps)."""
    overlap = 1 if compress_ratio == 4 else 0
    n_gather = (1 + overlap) * compress_ratio

    n_sets = _num_input_sets(num_tokens, kv_block_size, compress_ratio, overlap)
    arg_sets = [
        setup_inputs(num_tokens, kv_block_size, compress_ratio, overlap, device)
        for _ in range(n_sets)
    ]

    # Warmup
    for _ in range(WARMUP):
        run_kernel(arg_sets[0])
    torch.xpu.synchronize()

    # Rotate input sets to defeat L2 caching; time with device events.
    start_evt = torch.xpu.Event(enable_timing=True)
    end_evt = torch.xpu.Event(enable_timing=True)
    start_evt.record()
    for i in range(REPEAT):
        run_kernel(arg_sets[i % n_sets])
    end_evt.record()
    torch.xpu.synchronize()
    elapsed = start_evt.elapsed_time(end_evt) / 1e3 / REPEAT  # ms -> s

    # Compute effective bandwidth
    # Read: num_tokens * n_gather * 2 * HEAD_DIM * 2 bytes (kv+score, bf16)
    # Write: num_tokens * (TOKEN_STRIDE + SCALE_DIM) bytes
    read_bytes = num_tokens * n_gather * 2 * HEAD_DIM * 2
    write_bytes = num_tokens * (TOKEN_STRIDE + SCALE_DIM)
    total_bytes = read_bytes + write_bytes
    gbps = total_bytes / elapsed / 1e9

    return elapsed * 1e6, gbps


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark fused compress+norm+rope+mxfp4 kernel")
    parser.add_argument("--compress-ratio", type=int, default=None,
                        help="Test only this compress ratio")
    parser.add_argument("--check-correctness", action="store_true",
                        help="Run correctness check instead of benchmark")
    parser.add_argument("--device", type=str, default="xpu")
    args = parser.parse_args()

    if args.check_correctness:
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "pytest",
             "tests/test_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn.py",
             "-v"],
            cwd=str(__import__("pathlib").Path(__file__).parent.parent),
        )
        raise SystemExit(result.returncode)

    compress_ratios = [args.compress_ratio] if args.compress_ratio else [4]
    token_counts = [1, 4, 16, 64, 128, 256, 1024, 2048, 4096]
    kv_block_size = 32

    print(
        "Rotating input sets to bypass L2; GB/s reflects HBM traffic."
    )
    print(f"{'CR':>4} | {'Tokens':>6} | {'Time (us)':>10} | {'GB/s':>8}")
    print("-" * 40)

    for cr in compress_ratios:
        for nt in token_counts:
            try:
                time_us, gbps = benchmark_config(nt, kv_block_size, cr,
                                                 args.device)
                print(f"{cr:>4} | {nt:>6} | {time_us:>10.1f} | {gbps:>8.1f}")
            except Exception as e:
                print(f"{cr:>4} | {nt:>6} | ERROR: {e}")
        print()


if __name__ == "__main__":
    main()
