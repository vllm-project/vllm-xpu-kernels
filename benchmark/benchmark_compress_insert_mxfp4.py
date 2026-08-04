# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark for fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn kernel.

Measures throughput (GB/s) of the fused DeepSeek-V4 indexer MXFP4 path:
  Online Softmax Compression → RMSNorm → RoPE → MXFP4 quant → KV cache insert.

Usage:
  python benchmark/benchmark_compress_insert_mxfp4.py

For accuracy, run the matching test instead:
  pytest tests/test_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn.py
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

# The kernel only supports the DeepSeek-V4 C4 indexer configuration: vLLM
# creates the indexer layer solely for compress_ratio == 4, and derives
# overlap as (compress_ratio == 4). The SYCL kernel TORCH_CHECKs both.
COMPRESS_RATIO = 4
OVERLAP = 1
STATE_WIDTH = (1 + OVERLAP) * HEAD_DIM

WARMUP = 10
REPEAT = 50


def setup_inputs(num_tokens, kv_block_size, device):
    """Create input tensors for the kernel."""
    num_pages = (COMPRESS_RATIO * num_tokens - 1) // BLOCK_SIZE + 2
    state_cache = torch.randn(
        num_pages, BLOCK_SIZE, 2 * STATE_WIDTH,
        dtype=torch.bfloat16, device=device,
    )
    block_table = torch.arange(
        num_pages, dtype=torch.int32, device=device
    ).unsqueeze(0)
    token_to_req = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    positions = torch.arange(
        COMPRESS_RATIO - 1, COMPRESS_RATIO * num_tokens, COMPRESS_RATIO,
        dtype=torch.int64, device=device,
    )
    rms_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)
    cos_sin_cache = torch.randn(
        COMPRESS_RATIO * num_tokens, ROPE_DIM, device=device
    )

    kv_n_blocks = (num_tokens + kv_block_size - 1) // kv_block_size + 1
    kv_cache = torch.zeros(
        kv_n_blocks, kv_block_size * (TOKEN_STRIDE + SCALE_DIM),
        dtype=torch.uint8, device=device,
    )

    return (state_cache, token_to_req, positions, slot_mapping, block_table,
            BLOCK_SIZE, STATE_WIDTH, rms_weight, RMS_EPS, cos_sin_cache,
            kv_cache, slot_mapping, kv_block_size, HEAD_DIM, ROPE_DIM,
            COMPRESS_RATIO, OVERLAP, QUANT_BLOCK)


def run_kernel(args):
    """Run the kernel with given args tuple."""
    torch.ops._xpu_C.fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn(
        *args)


def _bytes_per_input_set(num_tokens, kv_block_size):
    """Approximate device bytes for one input set (dominated by state_cache)."""
    num_pages = (COMPRESS_RATIO * num_tokens - 1) // BLOCK_SIZE + 2
    state_bytes = num_pages * BLOCK_SIZE * (2 * STATE_WIDTH) * 2  # bf16
    kv_n_blocks = (num_tokens + kv_block_size - 1) // kv_block_size + 1
    kv_bytes = kv_n_blocks * kv_block_size * (TOKEN_STRIDE + SCALE_DIM)
    return state_bytes + kv_bytes


# Rotate enough distinct input sets to exceed on-chip (L2) cache; otherwise
# state_cache stays L2-resident across REPEAT and BW reflects L2, not HBM.
L2_BYPASS_BYTES = 512 * 1024 * 1024
# Never build more sets than the timed loop consumes.
MAX_INPUT_SETS = REPEAT


def _num_input_sets(num_tokens, kv_block_size):
    per = _bytes_per_input_set(num_tokens, kv_block_size)
    n = (L2_BYPASS_BYTES + per - 1) // per
    return max(1, min(MAX_INPUT_SETS, n))


def benchmark_config(num_tokens, kv_block_size, device):
    """Benchmark a single configuration, return (time_us, gbps)."""
    n_sets = _num_input_sets(num_tokens, kv_block_size)
    arg_sets = [
        setup_inputs(num_tokens, kv_block_size, device)
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

    del arg_sets
    torch.xpu.empty_cache()

    # Effective bandwidth. Each row is gathered twice, but the two visits use
    # different head-block offsets, so together they read the row exactly once.
    row_bytes = 2 * STATE_WIDTH * 2  # kv + scores, bf16
    read_bytes = num_tokens * COMPRESS_RATIO * row_bytes
    write_bytes = num_tokens * (TOKEN_STRIDE + SCALE_DIM)
    gbps = (read_bytes + write_bytes) / elapsed / 1e9

    return elapsed * 1e6, gbps


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark fused compress+norm+rope+mxfp4 kernel")
    parser.add_argument("--device", type=str, default="xpu")
    args = parser.parse_args()

    token_counts = [1, 4, 16, 64, 128, 256, 1024, 2048, 4096]
    kv_block_size = 32

    print("Rotating input sets to bypass L2; small token counts still fit in "
          "cache, so their GB/s is not HBM bandwidth.")
    print(f"{'Tokens':>6} | {'Time (us)':>10} | {'GB/s':>8}")
    print("-" * 34)

    for nt in token_counts:
        try:
            time_us, gbps = benchmark_config(nt, kv_block_size, args.device)
            print(f"{nt:>6} | {time_us:>10.1f} | {gbps:>8.1f}")
        except Exception as e:
            print(f"{nt:>6} | ERROR: {e}")
    print()


if __name__ == "__main__":
    main()
