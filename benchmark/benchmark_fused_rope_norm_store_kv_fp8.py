# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""
Benchmark for fused_rope_norm_store_kv_fp8 kernel (Hy3-FP8 config).

Compares:
  - Fused kernel: single fused_rope_norm_store_kv_fp8 call
  - Unfused kernels: rotary_embedding + rms_norm × 2 + reshape_and_cache
                     + dynamic_per_token_scaled_fp8_quant

Hy3-FP8 config: 64 Q heads, 8 KV heads, GQA=8, head_dim=128,
                qk_norm_policy=ROPE_THEN_NORM, FP8 E4M3.
"""

import time

import torch
import triton
from utils import bootstrap_benchmark_env

bootstrap_benchmark_env(__file__)

import vllm_xpu_kernels._C  # noqa: F401 - registers torch.ops._C ops (unfused kernels)
import vllm_xpu_kernels._xpu_C  # noqa: F401 - registers torch.ops._xpu_C ops

DEVICE = "xpu"
HEAD_DIM = 128
MAX_POS = 8192
BLOCK_SIZE = 16

# Hy3-FP8: 295B/21B MoE, 64 Q heads, 8 KV heads, GQA=8, ROPE_THEN_NORM
NUM_HEADS = 64
NUM_KV_HEADS = 8
QK_NORM_POLICY = 1  # ROPE_THEN_NORM
EPS = 1e-6

TOKEN_COUNTS = [1, 4, 8, 16, 32, 64, 128, 256, 512]


# ---------------------------------------------------------------------------
# Shared input allocation
# ---------------------------------------------------------------------------

def _build_cos_sin_cache(max_pos, head_dim, device):
    half_dim = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (
        torch.arange(0, half_dim, dtype=torch.float32, device=device) /
        half_dim))
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def make_inputs(num_tokens, device=DEVICE):
    """Create all inputs on device, ready for benchmarking."""
    q_size = NUM_HEADS * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM
    total_dim = q_size + 2 * kv_size

    qkv = torch.randn(num_tokens, total_dim, dtype=torch.bfloat16,
                       device=device)
    positions = torch.randint(0, MAX_POS, (num_tokens,), dtype=torch.long,
                              device=device)
    cos_sin_cache = _build_cos_sin_cache(MAX_POS, HEAD_DIM, device=device)
    cos_sin_cache_bf16 = cos_sin_cache.bfloat16()  # rotary_embedding needs bf16

    # Slot mapping: sequential
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=device)

    # Scales
    k_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    v_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

    # Norm weights (bf16 for unfused rms_norm, fp32 for fused kernel)
    q_norm_weight_fp32 = torch.empty(HEAD_DIM, dtype=torch.float32,
                                     device=device).normal_(mean=1.0, std=0.1)
    k_norm_weight_fp32 = torch.empty(HEAD_DIM, dtype=torch.float32,
                                     device=device).normal_(mean=1.0, std=0.1)
    q_norm_weight_bf16 = q_norm_weight_fp32.bfloat16()
    k_norm_weight_bf16 = k_norm_weight_fp32.bfloat16()

    # --- Fused kernel buffers ---
    num_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
    key_cache_fused = torch.zeros(
        num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
        dtype=torch.uint8, device=device)
    value_cache_fused = torch.zeros_like(key_cache_fused)
    out_q_fused = torch.zeros(num_tokens, NUM_HEADS, HEAD_DIM,
                              dtype=torch.uint8, device=device)

    # --- Unfused kernel buffers ---
    # reshape_and_cache uses separate K/V caches:
    #   key_cache:   (num_blocks, num_heads, head_size // x, block_size, x)
    #   value_cache: (num_blocks, num_heads, head_size, block_size)
    x = 16  # fp8 = 1 byte, x = 16 // 1 = 16
    key_cache_unfused = torch.zeros(
        num_blocks, NUM_KV_HEADS, HEAD_DIM // x, BLOCK_SIZE, x,
        dtype=torch.uint8, device=device)
    value_cache_unfused = torch.zeros(
        num_blocks, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE,
        dtype=torch.uint8, device=device)
    # Q FP8 output + scale for unfused path
    out_q_unfused = torch.zeros(num_tokens * NUM_HEADS, HEAD_DIM,
                                dtype=torch.float8_e4m3fn, device=device)
    q_scale_unfused = torch.zeros(num_tokens * NUM_HEADS,
                                  dtype=torch.float32, device=device)
    # rms_norm output buffers
    q_norm_buf = torch.empty(num_tokens * NUM_HEADS, HEAD_DIM,
                             dtype=torch.bfloat16, device=device)
    k_norm_buf = torch.empty(num_tokens * NUM_KV_HEADS, HEAD_DIM,
                             dtype=torch.bfloat16, device=device)

    return {
        "qkv": qkv,
        "positions": positions,
        "cos_sin_cache": cos_sin_cache,
        "cos_sin_cache_bf16": cos_sin_cache_bf16,
        "slot_mapping": slot_mapping,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "q_norm_fp32": q_norm_weight_fp32,
        "k_norm_fp32": k_norm_weight_fp32,
        "q_norm_bf16": q_norm_weight_bf16,
        "k_norm_bf16": k_norm_weight_bf16,
        # Fused buffers
        "key_cache_fused": key_cache_fused,
        "value_cache_fused": value_cache_fused,
        "out_q_fused": out_q_fused,
        # Unfused buffers
        "key_cache_unfused": key_cache_unfused,
        "value_cache_unfused": value_cache_unfused,
        "out_q_unfused": out_q_unfused,
        "q_scale_unfused": q_scale_unfused,
        "q_norm_buf": q_norm_buf,
        "k_norm_buf": k_norm_buf,
    }


# ---------------------------------------------------------------------------
# Fused kernel path (1 kernel launch)
# ---------------------------------------------------------------------------

def run_fused(inp):
    """Single fused kernel: RoPE + QK-Norm + KV cache FP8 + Q FP8 quant."""
    torch.ops._xpu_C.fused_rope_norm_store_kv_fp8(
        inp["qkv"], inp["cos_sin_cache"], inp["positions"],
        inp["key_cache_fused"], inp["value_cache_fused"],
        inp["slot_mapping"], inp["out_q_fused"],
        NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        inp["k_scale"], inp["v_scale"],
        inp["q_norm_fp32"], inp["k_norm_fp32"],
        QK_NORM_POLICY, True,
    )


# ---------------------------------------------------------------------------
# Unfused kernel path (5 kernel launches)
# ---------------------------------------------------------------------------

def run_unfused(inp, num_tokens):
    """Unfused pipeline using existing SYCL kernels.

    For Hy3-FP8 (ROPE_THEN_NORM):
      1. rotary_embedding                     — RoPE on Q & K (in-place)
      2. rms_norm                             — QK-Norm on Q
      3. rms_norm                             — QK-Norm on K
      4. reshape_and_cache                    — KV cache write (FP8)
      5. dynamic_per_token_scaled_fp8_quant   — Q FP8 quant
    """
    qkv = inp["qkv"]
    q_size = NUM_HEADS * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM

    # Split QKV (views, no copy)
    q = qkv[:, :q_size].view(num_tokens, NUM_HEADS, HEAD_DIM)
    k = qkv[:, q_size:q_size + kv_size].view(
        num_tokens, NUM_KV_HEADS, HEAD_DIM)
    v = qkv[:, q_size + kv_size:].view(
        num_tokens, NUM_KV_HEADS, HEAD_DIM)

    # 1. RoPE (in-place on q, k)
    q_flat = q.reshape(num_tokens, NUM_HEADS * HEAD_DIM)
    k_flat = k.reshape(num_tokens, NUM_KV_HEADS * HEAD_DIM)
    torch.ops._C.rotary_embedding(
        inp["positions"], q_flat, k_flat,
        HEAD_DIM, inp["cos_sin_cache_bf16"], True,
    )

    # 2. RMSNorm Q: reshape to (T*Hq, D) so per-row = per-head
    q_2d = q.reshape(num_tokens * NUM_HEADS, HEAD_DIM)
    torch.ops._C.rms_norm(
        inp["q_norm_buf"], q_2d, inp["q_norm_bf16"], EPS,
    )

    # 3. RMSNorm K: reshape to (T*Hkv, D)
    k_2d = k.reshape(num_tokens * NUM_KV_HEADS, HEAD_DIM)
    torch.ops._C.rms_norm(
        inp["k_norm_buf"], k_2d, inp["k_norm_bf16"], EPS,
    )

    # 4. KV cache write (reshape_and_cache handles FP8 quant internally)
    k_normed = inp["k_norm_buf"].view(num_tokens, NUM_KV_HEADS, HEAD_DIM)
    torch.ops._C_cache_ops.reshape_and_cache(
        k_normed, v,
        inp["key_cache_unfused"], inp["value_cache_unfused"],
        inp["slot_mapping"], "fp8",
        inp["k_scale"], inp["v_scale"],
    )

    # 5. Q FP8 quant: per-row on (T*Hq, D) = per-token-per-head
    torch.ops._C.dynamic_per_token_scaled_fp8_quant(
        inp["out_q_unfused"], inp["q_norm_buf"],
        inp["q_scale_unfused"], None,
    )


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def measure_latency(fn, warmup=100, repeat=200):
    """Measure kernel latency with sync, return trimmed stats in us."""
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    times = []
    for _ in range(repeat):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    times.sort()
    trim = len(times) // 10
    trimmed = times[trim:-trim] if trim > 0 else times
    avg = sum(trimmed) / len(trimmed)
    p50 = trimmed[len(trimmed) // 2]
    p10 = trimmed[len(trimmed) // 10]
    p90 = trimmed[int(len(trimmed) * 0.9)]
    return avg, p50, p10, p90


def get_benchmark():
    """Create triton perf_report benchmark."""

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens"],
            x_vals=TOKEN_COUNTS,
            line_arg="provider",
            line_vals=["fused_kernel", "unfused_kernels"],
            line_names=[
                "Fused (1 kernel)",
                "Unfused (rope + norm*2 + cache + quant)",
            ],
            styles=[("green", "-"), ("blue", "--")],
            ylabel="us",
            plot_name="hy3_fp8_fused_vs_unfused",
            args={},
        ))
    def benchmark(num_tokens, provider):
        inp = make_inputs(num_tokens)
        quantiles = [0.5, 0.2, 0.8]

        if provider == "fused_kernel":
            def fn():
                inp_c = {**inp}
                inp_c["qkv"] = inp["qkv"].clone()
                inp_c["key_cache_fused"] = inp["key_cache_fused"].clone()
                inp_c["value_cache_fused"] = inp["value_cache_fused"].clone()
                inp_c["out_q_fused"] = inp["out_q_fused"].clone()
                run_fused(inp_c)

            ms, min_ms, max_ms = triton.testing.do_bench(
                fn, quantiles=quantiles)
        else:
            def fn():
                inp_c = {**inp}
                inp_c["qkv"] = inp["qkv"].clone()
                inp_c["key_cache_unfused"] = \
                    inp["key_cache_unfused"].clone()
                inp_c["value_cache_unfused"] = \
                    inp["value_cache_unfused"].clone()
                run_unfused(inp_c, num_tokens)

            ms, min_ms, max_ms = triton.testing.do_bench(
                fn, quantiles=quantiles)

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark fused_rope_norm_store_kv_fp8 (Hy3-FP8)")
    parser.add_argument("--tokens", type=int, nargs="+", default=None,
                        help="Token counts (default: sweep)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate triton perf plot")
    parser.add_argument("--save-path", type=str, default="./benchmark_results",
                        help="Path to save plots")
    args = parser.parse_args()

    print("=" * 80)
    print("Benchmark: fused_rope_norm_store_kv_fp8 — Hy3-FP8")
    print("=" * 80)
    print("  Model       = Hy3-FP8 (295B/21B MoE)")
    print(f"  Q heads     = {NUM_HEADS}, KV heads = {NUM_KV_HEADS}, "
          f"GQA = {NUM_HEADS // NUM_KV_HEADS}")
    print(f"  HEAD_DIM    = {HEAD_DIM}, BLOCK_SIZE = {BLOCK_SIZE}")
    print("  Norm policy = ROPE_THEN_NORM")
    print(f"  Device      = {torch.xpu.get_device_name(0)}")
    print("=" * 80)
    print("\n  Fused:   1 kernel launch")
    print("  Unfused: rotary_embedding + rms_norm x2 "
          "+ reshape_and_cache + dynamic_fp8_quant")

    token_counts = args.tokens or [1, 4, 16, 32, 64, 128, 256, 512]

    # --- Latency comparison table ---
    hdr = (f"\n{'Tokens':<8} {'Fused(us)':<12} {'Unfused(us)':<13} "
           f"{'Speedup':<10} {'F-P50(us)':<12} {'U-P50(us)':<12}")
    print(hdr)
    print("-" * 80)

    for num_tokens in token_counts:
        inp = make_inputs(num_tokens)

        # Fused
        def run_f(inp=inp):
            inp_c = {**inp}
            inp_c["qkv"] = inp["qkv"].clone()
            inp_c["key_cache_fused"] = inp["key_cache_fused"].clone()
            inp_c["value_cache_fused"] = inp["value_cache_fused"].clone()
            inp_c["out_q_fused"] = inp["out_q_fused"].clone()
            run_fused(inp_c)

        f_avg, f_p50, _, _ = measure_latency(run_f)

        # Unfused
        def run_u(inp=inp, nt=num_tokens):
            inp_c = {**inp}
            inp_c["qkv"] = inp["qkv"].clone()
            inp_c["key_cache_unfused"] = inp["key_cache_unfused"].clone()
            inp_c["value_cache_unfused"] = \
                inp["value_cache_unfused"].clone()
            run_unfused(inp_c, nt)

        u_avg, u_p50, _, _ = measure_latency(run_u)

        speedup = u_avg / f_avg if f_avg > 0 else float("inf")
        print(f"{num_tokens:<8} {f_avg:<12.2f} {u_avg:<13.2f} "
              f"{speedup:<10.2f}x {f_p50:<12.2f} {u_p50:<12.2f}")

    # --- Triton perf plot (optional) ---
    if args.plot:
        from pathlib import Path
        Path(args.save_path).mkdir(parents=True, exist_ok=True)
        print("\nGenerating plot...")
        bench_fn = get_benchmark()
        bench_fn.run(print_data=True, save_path=args.save_path)

    print("\nDone.")