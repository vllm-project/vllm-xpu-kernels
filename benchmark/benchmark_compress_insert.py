# SPDX-License-Identifier: Apache-2.0
"""Benchmark for the fused_kv_compress_norm_rope_insert_sparse_attn kernel.

Mirrors real input shapes from DeepSeek-V4:
  Layer A: compress_ratio=4,   overlap=1, kv_blk=64
  Layer B: compress_ratio=128, overlap=0, kv_blk=2

``num_tokens`` counts the number of compression boundaries (real work items).
They are embedded in a contiguous prefill of length num_tokens * cr, so the
boundary density is 1/cr and the non-boundary tokens early-exit exactly like
real inference.

Usage (BENCH=benchmark_compress_insert.py):
    ZE_AFFINITY_MASK=0 python $BENCH
    ZE_AFFINITY_MASK=0 python $BENCH --layer B --num-tokens 8192
    ZE_AFFINITY_MASK=0 python $BENCH --iters 100
"""
import argparse
import gc

import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401 - registers torch.ops._xpu_C ops

HEAD_SIZE = 512
ROPE_HEAD_DIM = 64
RMS_EPS = 1e-6
TOKEN_STRIDE = 576
SCALE_DIM = 8

LAYER_CONFIGS = {
    "A": dict(compress_ratio=4,   overlap=1, kv_cache_block_size=64,
              state_blk_sz=4, label="A_cr4"),
    "B": dict(compress_ratio=128, overlap=0, kv_cache_block_size=2,
              state_blk_sz=8, label="B_cr128"),
}

NUM_TOKENS_DEFAULT = [1, 4, 16, 64, 128, 256, 1024, 2048, 4096]


def make_inputs(cfg, num_tokens, seed=0, num_blocks=152522):
    device = torch.device("xpu")
    cr = cfg["compress_ratio"]
    overlap = cfg["overlap"]
    kv_blk = cfg["kv_cache_block_size"]
    state_blk_sz = cfg["state_blk_sz"]
    state_width = (1 + overlap) * HEAD_SIZE
    gen = torch.Generator(device="cpu").manual_seed(seed)

    # ``num_tokens`` counts the compression boundaries (the real work items).
    # The kernel launches one program per token and early-exits every token
    # whose ``(position + 1) % compress_ratio != 0``, so embedding the
    # boundaries in a contiguous prefill of ``num_tokens * cr`` tokens gives
    # the 1/cr density a genuine prefill chunk sees.
    positions = torch.arange(num_tokens * cr).to(device)
    n_launch = int(positions.numel())

    state_cache = (torch.randn(num_blocks, state_blk_sz, 2 * state_width,
                               generator=gen) * 0.1).to(device)
    k_cache_fp8mix = torch.zeros(
        num_blocks,
        kv_blk,
        TOKEN_STRIDE + SCALE_DIM,
        dtype=torch.uint8,
        device=device,
    )
    # block_table must cover the largest state block index touched by the
    # gather window, i.e. (n_launch - 1) // state_blk_sz.
    bt_cols = (n_launch // state_blk_sz) + 8
    block_table = torch.randint(0, num_blocks, (1, bt_cols),
                                dtype=torch.int32, generator=gen).to(device)

    token_to_req = torch.zeros(n_launch, dtype=torch.int32, device=device)
    slot_mapping = torch.arange(n_launch, dtype=torch.int64, device=device)
    # Only boundary tokens ever reach the write; their compressed cache slot is
    # position // cr. Non-boundary tokens early-exit before this is used.
    kv_slot_mapping = (positions // cr).to(torch.int64)
    # Match production: DeepSeek-V4 compressor RMSNorm weight follows the model
    # dtype (bf16), so use bf16 here to exercise the in-kernel upcast path.
    rms_norm_weight = torch.ones(HEAD_SIZE, dtype=torch.bfloat16, device=device)

    # RoPE indexes the cache at position + 1 - cr, so n_launch rows suffice.
    half = ROPE_HEAD_DIM // 2
    freq = torch.arange(half, dtype=torch.float32) / half
    base = torch.arange(n_launch, dtype=torch.float32).unsqueeze(1)
    cos_sin_cache = torch.cat([
        torch.cos(base * (10000 ** (-freq))),
        torch.sin(base * (10000 ** (-freq))),
    ], dim=1).to(device)

    return (
        state_cache, token_to_req, positions, slot_mapping, block_table,
        rms_norm_weight, RMS_EPS, cos_sin_cache, k_cache_fp8mix,
        kv_slot_mapping, kv_blk, cr, overlap, ROPE_HEAD_DIM,
        TOKEN_STRIDE, SCALE_DIM, int(k_cache_fp8mix.stride(0)),
    )


def estimate_bytes(cfg, num_tokens):
    n_gather = (1 + cfg["overlap"]) * cfg["compress_ratio"]
    read_bytes = num_tokens * n_gather * HEAD_SIZE * 4 * 2
    write_bytes = num_tokens * (TOKEN_STRIDE + SCALE_DIM)
    return read_bytes + write_bytes


def benchmark_path(cfg, num_tokens, fp8mix_inputs, warmup=5, iters=50):
    (
        state_cache,
        token_to_req,
        positions,
        slot_mapping,
        block_table,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        k_cache_fp8mix,
        kv_slot_mapping,
        kv_blk,
        compress_ratio,
        overlap,
        rope_head_dim,
        token_stride,
        scale_dim,
        kv_block_stride,
    ) = fp8mix_inputs

    def run():
        torch.ops._xpu_C.fused_kv_compress_norm_rope_insert_sparse_attn(
            state_cache,
            token_to_req,
            positions,
            slot_mapping,
            block_table,
            rms_norm_weight,
            rms_norm_eps,
            cos_sin_cache,
            k_cache_fp8mix,
            kv_slot_mapping,
            kv_blk,
            compress_ratio,
            overlap,
            rope_head_dim,
            token_stride,
            scale_dim,
            kv_block_stride,
        )

    for _ in range(warmup):
        run()
    torch.xpu.synchronize()

    # Device events, so host-side launch overhead is not counted.
    start_evt = torch.xpu.Event(enable_timing=True)
    end_evt = torch.xpu.Event(enable_timing=True)
    start_evt.record()
    for _ in range(iters):
        run()
    end_evt.record()
    torch.xpu.synchronize()

    lat_s = start_evt.elapsed_time(end_evt) / 1e3 / iters  # ms -> s
    nbytes = estimate_bytes(cfg, num_tokens)
    gbps = nbytes / lat_s / 1e9
    return lat_s * 1e6, gbps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", choices=["A", "B", "both"], default="both")
    ap.add_argument("--num-blocks", type=int, default=8192,
                    help="number of state/kv blocks for synthetic input")
    ap.add_argument("--num-tokens", type=int, default=None,
                    help="single num_tokens value; default: sweep")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    layers = (["A", "B"] if args.layer == "both"
              else [args.layer])
    nt_list = ([args.num_tokens] if args.num_tokens
               else NUM_TOKENS_DEFAULT)

    print("# num_tok = # compressions; prefill_len = num_tok x cr")
    print(
        f"{'layer':<10} {'num_tok':>8} "
        f"{'lat_us':>10} {'GB/s':>10} {'us/tok':>10}"
    )
    print("-" * 52)

    for lname in layers:
        cfg = LAYER_CONFIGS[lname]
        for nt in nt_list:
            inputs = make_inputs(
                cfg,
                nt,
                num_blocks=args.num_blocks,
            )
            lat_us, gbps = benchmark_path(
                cfg, nt, inputs,
                warmup=args.warmup, iters=args.iters,
            )
            print(
                f"{cfg['label']:<10} "
                f"{nt:>8} {lat_us:>10.2f} "
                f"{gbps:>10.1f} {lat_us/nt:>10.2f}"
            )
            # Avoid long sweeps accumulating allocator pressure and
            # triggering DEVICE_LOST.
            del inputs
            gc.collect()
            torch.xpu.empty_cache()
        print()


if __name__ == "__main__":
    main()
