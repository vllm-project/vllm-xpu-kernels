# SPDX-License-Identifier: Apache-2.0
"""Benchmark the BF16xFP32 emulated-FP32 router GEMM on XPU.

The kernel emulates an FP32-precision GEMM (BF16 activation x FP32 weight) with
two BF16 DPAS ops (DualGemm), targeting the Hunyuan-V3 MoE router shape
(N = num_experts, K = hidden_size, M = num_tokens). It is compared against the
correctness-equivalent FP32 ``F.linear`` baseline.

  python benchmark/benchmark_gemm_bf16xfp32.py
  python benchmark/benchmark_gemm_bf16xfp32.py --iters 200 --warmup 50 --check
  python benchmark/benchmark_gemm_bf16xfp32.py --n 192 --k 4096 --m 2,64,1024
"""

import os
from argparse import ArgumentParser

# Silence libkineto INFO spam; must be set before importing torch.
os.environ.setdefault("KINETO_LOG_LEVEL", "5")

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

import vllm_xpu_kernels._xpu_C  # noqa: F401
from vllm_xpu_kernels.gemm_bf16xfp32 import gemm_bf16xfp32, split_fp32_weight

SCALE = 1.0 / 256.0

M_LIST = [2, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
N_LIST = [128, 192, 256]
K_LIST = [3072, 4096, 6144]


def _make_inputs(m, n, k, seed=1234):
    torch.manual_seed(seed)
    x_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="xpu") / 8.0
    w_fp32 = torch.randn(n, k, dtype=torch.float32, device="xpu") / 8.0
    return x_bf16, w_fp32


def _bench_wall(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.xpu.synchronize()
    return start.elapsed_time(end) * 1e3 / iters  # ms -> us


def _bench_device(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    with profile(activities=[ProfilerActivity.XPU]) as prof:
        for _ in range(iters):
            fn()
        torch.xpu.synchronize()

    def _self_us(evt):
        for attr in ("self_device_time_total", "self_xpu_time_total"):
            val = getattr(evt, attr, None)
            if val:
                return val
        return 0.0

    return sum(_self_us(e) for e in prof.key_averages()) / iters


def benchmark_gemm_bf16xfp32(shapes, peaks, warmup, iters, check):
    peak_bf16, peak_fp32, peak_bw = peaks

    print(f"torch={torch.__version__}, scale={SCALE}, "
          f"iters/warmup={iters}/{warmup}")
    print(f"peaks: bf16 XMX {peak_bf16} TFLOP/s | fp32 vector {peak_fp32} "
          f"TFLOP/s | BW {peak_bw} GB/s")
    print()

    hdr = (f"{'M':>6}{'N':>5}{'K':>6} | "
           f"{'ker dev':>8} {'ker wall':>9} | "
           f"{'TFLOP/s':>9} {'MFU':>7} {'GB/s':>9} {'MBU':>7} | "
           f"{'fp32 dev':>9} {'fp32 wall':>10} | "
           f"{'dev speedup':>11} {'wall speedup':>12}")
    print(hdr)
    print("-" * len(hdr))

    prev_nk = None
    for (m, n, k) in shapes:
        if prev_nk is not None and (n, k) != prev_nk:
            print()
        prev_nk = (n, k)

        x_bf16, w_fp32 = _make_inputs(m, n, k)
        w_high, w_low = split_fp32_weight(w_fp32, SCALE)  # -> [K, N] bf16

        def run_kernel(x=x_bf16, wh=w_high, wl=w_low):
            return gemm_bf16xfp32(x, wh, wl, SCALE)

        def run_fp32(x=x_bf16, w=w_fp32):
            return F.linear(x.float(), w)

        if check:
            out = run_kernel().float()
            ref = run_fp32()
            torch.xpu.synchronize()
            denom = ref.abs().max().item() + 1e-6
            rel = (out - ref).abs().max().item() / denom
            assert rel < 2e-2, (
                f"correctness failed at {(m, n, k)}: rel={rel:.3e}")

        dev_k = _bench_device(run_kernel, warmup, iters)
        dev_f = _bench_device(run_fp32, warmup, iters)
        wall_k = _bench_wall(run_kernel, warmup, iters)
        wall_f = _bench_wall(run_fp32, warmup, iters)

        tflops = (4.0 * m * n * k) / (dev_k * 1e-6) / 1e12
        gbps = (m * k * 2 + 2 * (k * n * 2) + m * n * 4) / (dev_k * 1e-6) / 1e9

        print(f"{m:>6}{n:>5}{k:>6} | "
              f"{dev_k:>8.2f} {wall_k:>9.2f} | "
              f"{tflops:>9.2f} {tflops / peak_bf16 * 100:>6.1f}% "
              f"{gbps:>9.1f} {gbps / peak_bw * 100:>6.1f}% | "
              f"{dev_f:>9.2f} {wall_f:>10.2f} | "
              f"{dev_f / dev_k:>10.2f}x {wall_f / wall_k:>11.2f}x")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Benchmark the BF16xFP32 router GEMM kernel on XPU.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--check", action="store_true",
                        help="verify kernel output vs FP32 F.linear per shape")
    parser.add_argument("--m", type=str, default=None,
                        help="comma-separated M values (overrides the sweep)")
    parser.add_argument("--n", type=str, default=None,
                        help="comma-separated N values (overrides the sweep)")
    parser.add_argument("--k", type=str, default=None,
                        help="comma-separated K values (overrides the sweep)")

    # Hardware peaks for MFU/MBU roofline (defaults: Intel Arc Pro B60 / BMG).
    parser.add_argument("--peak-bf16-tflops", type=float, default=98.5)
    parser.add_argument("--peak-fp32-tflops", type=float, default=12.28)
    parser.add_argument("--peak-bw-gbps", type=float, default=456.0)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_default_device("xpu")

    def _parse(arg, default):
        return [int(x) for x in arg.split(",")] if arg else default

    m_list = _parse(args.m, M_LIST)
    n_list = _parse(args.n, N_LIST)
    k_list = _parse(args.k, K_LIST)
    shapes = [(m, n, k) for n in n_list for k in k_list for m in m_list]

    benchmark_gemm_bf16xfp32(
        shapes,
        (args.peak_bf16_tflops, args.peak_fp32_tflops, args.peak_bw_gbps),
        args.warmup,
        args.iters,
        args.check,
    )
