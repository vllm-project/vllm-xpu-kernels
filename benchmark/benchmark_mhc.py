# SPDX-License-Identifier: Apache-2.0

import argparse

import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

HC = 4
BENCH_CASES = [
    (1, 4096),
    (33, 4096),
    (128, 4096),
    (256, 4096),
    (1024, 4096),
    (2048, 4096),
    (4096, 4096),
    (8192, 4096),
    (16384, 4096),
    (1, 7168),
    (33, 7168),
    (128, 7168),
    (256, 7168),
    (1024, 7168),
    (2048, 7168),
    (4096, 7168),
    (8192, 7168),
    (16384, 7168),
]


def benchmark_op(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    end_event.synchronize()

    return start_event.elapsed_time(end_event) * 1e3 / iters


def run_mhc_pre(num_tokens: int, hidden_size: int, warmup: int, iters: int):
    hc3 = HC * 2 + HC * HC
    residual = torch.randn(
        (num_tokens, HC, hidden_size),
        dtype=torch.bfloat16,
        device="xpu",
    )
    fn = torch.randn((hc3, HC * hidden_size), dtype=torch.float32, device="xpu")
    hc_scale = torch.randn((3,), dtype=torch.float32, device="xpu")
    hc_base = torch.randn((hc3,), dtype=torch.float32, device="xpu")

    rms_eps = 1e-6
    hc_pre_eps = 1e-3
    hc_sinkhorn_eps = 1e-3
    hc_post_mult_value = 1.0
    sinkhorn_repeat = 20

    def run():
        torch.ops._xpu_C.mhc_pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )

    avg_s = benchmark_op(run, warmup, iters)

    # FLOPs model for `mhc_pre`
    # - Stage 1 dominant cost is one dense matmul: [num_tokens, HC *
    #     hidden_size] x [HC * hidden_size, hc3]
    # - We count 2 FLOPs per MAC.
    flops = 2.0 * num_tokens * hc3 * (HC * hidden_size)

    # Bytes moved model for `mhc_pre` (ideal / theoretical minimum)
    bytes_moved = (
        # inputs
        num_tokens * HC * hidden_size * 2        # residual
        + hc3 * HC * hidden_size * 4             # fn
        + 3 * 4                                  # hc_scale
        + hc3 * 4                                # hc_base
        # outputs
        + num_tokens * HC * 4                    # post_mix
        + num_tokens * HC * HC * 4               # comb_mix
        + num_tokens * hidden_size * 2           # layer_input
    )
    return avg_s, flops, bytes_moved


def run_mhc_post(num_tokens: int, hidden_size: int, warmup: int, iters: int):
    x = torch.randn(
        (num_tokens, hidden_size),
        dtype=torch.bfloat16,
        device="xpu",
    )
    residual = torch.randn(
        (num_tokens, HC, hidden_size),
        dtype=torch.bfloat16,
        device="xpu",
    )
    post_mix = torch.randn(
        (num_tokens, HC, 1),
        dtype=torch.float32,
        device="xpu",
    )
    comb_mix = torch.randn(
        (num_tokens, HC, HC),
        dtype=torch.float32,
        device="xpu",
    )

    def run():
        torch.ops._xpu_C.mhc_post(x, residual, post_mix, comb_mix)

    avg_s = benchmark_op(run, warmup, iters)

    # FLOPs model for `mhc_post`
    flops = num_tokens * HC * hidden_size * (2.0 * HC + 1.0)

    # Bytes moved model for `mhc_post`
    bytes_moved = (
        num_tokens * hidden_size * 2
        + num_tokens * HC * hidden_size * 2
        + num_tokens * HC * 4
        + num_tokens * HC * HC * 4
        + num_tokens * HC * hidden_size * 2
    )
    return avg_s, flops, bytes_moved


def run_hc_head_fused(
    num_tokens: int,
    hidden_size: int,
    warmup: int,
    iters: int,
):
    hs_flat = torch.randn(
        (num_tokens, HC, hidden_size),
        dtype=torch.bfloat16,
        device="xpu",
    )
    fn = torch.randn((HC, HC * hidden_size), dtype=torch.float32, device="xpu")
    hc_scale = torch.randn((1,), dtype=torch.float32, device="xpu")
    hc_base = torch.randn((HC,), dtype=torch.float32, device="xpu")
    out = torch.empty(
        (num_tokens, hidden_size),
        dtype=torch.bfloat16,
        device="xpu",
    )

    rms_eps = 1e-6
    hc_eps = 1e-6

    def run():
        torch.ops._xpu_C.hc_head_fused(
            hs_flat,
            fn,
            hc_scale,
            hc_base,
            out,
            rms_eps,
            hc_eps,
        )

    avg_s = benchmark_op(run, warmup, iters)

    # FLOPs model for `hc_head_fused`
    flops = 2.0 * num_tokens * HC * (HC * hidden_size)

    # Bytes moved model for `hc_head_fused`
    bytes_moved = (
        num_tokens * HC * hidden_size * 2
        + HC * HC * hidden_size * 4
        + 4
        + HC * 4
        + num_tokens * hidden_size * 2
    )
    return avg_s, flops, bytes_moved


def run_mhc_fused_post_pre(
    num_tokens: int,
    hidden_size: int,
    warmup: int,
    iters: int,
):
    hc3 = HC * 2 + HC * HC

    x = torch.randn(
        (num_tokens, hidden_size),
        dtype=torch.bfloat16,
        device="xpu",
    )
    residual = torch.randn(
        (num_tokens, HC, hidden_size),
        dtype=torch.bfloat16,
        device="xpu",
    )
    post_mix = torch.randn(
        (num_tokens, HC, 1),
        dtype=torch.float32,
        device="xpu",
    )
    comb_mix = torch.randn(
        (num_tokens, HC, HC),
        dtype=torch.float32,
        device="xpu",
    )
    fn = torch.randn((hc3, HC * hidden_size), dtype=torch.float32, device="xpu")
    hc_scale = torch.randn((3,), dtype=torch.float32, device="xpu")
    hc_base = torch.randn((hc3,), dtype=torch.float32, device="xpu")

    rms_eps = 1e-6
    hc_pre_eps = 1e-3
    hc_sinkhorn_eps = 1e-3
    hc_post_mult_value = 1.0
    sinkhorn_repeat = 20

    def run():
        torch.ops._xpu_C.mhc_fused_post_pre(
            x, residual, post_mix, comb_mix,
            fn, hc_scale, hc_base,
            rms_eps, hc_pre_eps, hc_sinkhorn_eps,
            hc_post_mult_value, sinkhorn_repeat,
        )

    avg_s = benchmark_op(run, warmup, iters)

    # FLOPs model for `mhc_fused_post_pre` (Phase 0: composed)
    flops_post = num_tokens * HC * hidden_size * (2.0 * HC + 1.0)
    flops_pre = 2.0 * num_tokens * hc3 * (HC * hidden_size)
    flops = flops_post + flops_pre

    # Bytes moved model for `mhc_fused_post_pre` (ideal / theoretical minimum)
    bytes_moved = (
        # inputs
        num_tokens * hidden_size * 2             # x
        + num_tokens * HC * hidden_size * 2      # residual
        + num_tokens * HC * 4                    # post_mix
        + num_tokens * HC * HC * 4               # comb_mix
        + hc3 * HC * hidden_size * 4             # fn
        + 3 * 4                                  # hc_scale
        + hc3 * 4                                # hc_base
        # outputs
        + num_tokens * HC * hidden_size * 2      # residual_cur
        + num_tokens * HC * 4                    # post_mix_cur
        + num_tokens * HC * HC * 4               # comb_mix_cur
        + num_tokens * hidden_size * 2           # layer_input
    )
    return avg_s, flops, bytes_moved


def compute_metrics(latency_us: float, flops: float, bytes_moved: float):
    latency_s = latency_us / 1e6
    tflops = flops / latency_s / 1e12
    bandwidth = bytes_moved / latency_s / 1e9
    intensity = flops / bytes_moved if bytes_moved else 0.0
    return latency_us, tflops, bandwidth, intensity


def print_md_table(op_name: str, rows):
    """Print benchmark results as a Markdown table."""
    print(f"\n## {op_name}\n")
    print(
        "| num_tokens | hidden_size | Latency (us) | TFLOPS "
        "| BW (GB/s) | Arith. Intensity |"
    )
    print(
        "|------------|-------------|--------------|--------"
        "|-----------|------------------|"
    )
    for (
        num_tokens,
        hidden_size,
        latency_us,
        tflops,
        bandwidth,
        intensity,
    ) in rows:
        print(
            f"| {num_tokens:>10d} | {hidden_size:>11d} "
            f"| {latency_us:>12.3f} | {tflops:>6.3f} "
            f"| {bandwidth:>9.3f} | {intensity:>16.3f} |"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark mHC kernels on XPU (_xpu_C)"
    )
    parser.add_argument(
        "--op",
        choices=[
            "mhc_pre",
            "mhc_post",
            "hc_head_fused",
            "mhc_fused_post_pre",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=300)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_default_device("xpu")

    if args.op in ("mhc_pre", "all"):
        rows = []
        for num_tokens, hidden_size in BENCH_CASES:
            avg_s, flops, bytes_moved = run_mhc_pre(
                num_tokens,
                hidden_size,
                args.warmup,
                args.iters,
            )
            latency_us, tflops, bandwidth, intensity = compute_metrics(
                avg_s,
                flops,
                bytes_moved,
            )
            rows.append(
                (
                    num_tokens,
                    hidden_size,
                    latency_us,
                    tflops,
                    bandwidth,
                    intensity,
                )
            )
        print_md_table("mhc_pre", rows)

    if args.op in ("mhc_post", "all"):
        rows = []
        for num_tokens, hidden_size in BENCH_CASES:
            avg_s, flops, bytes_moved = run_mhc_post(
                num_tokens,
                hidden_size,
                args.warmup,
                args.iters,
            )
            latency_us, tflops, bandwidth, intensity = compute_metrics(
                avg_s,
                flops,
                bytes_moved,
            )
            rows.append(
                (
                    num_tokens,
                    hidden_size,
                    latency_us,
                    tflops,
                    bandwidth,
                    intensity,
                )
            )
        print_md_table("mhc_post", rows)

    if args.op in ("hc_head_fused", "all"):
        rows = []
        for num_tokens, hidden_size in BENCH_CASES:
            avg_s, flops, bytes_moved = run_hc_head_fused(
                num_tokens,
                hidden_size,
                args.warmup,
                args.iters,
            )
            latency_us, tflops, bandwidth, intensity = compute_metrics(
                avg_s,
                flops,
                bytes_moved,
            )
            rows.append(
                (
                    num_tokens,
                    hidden_size,
                    latency_us,
                    tflops,
                    bandwidth,
                    intensity,
                )
            )
        print_md_table("hc_head_fused", rows)

    if args.op in ("mhc_fused_post_pre", "all"):
        rows = []
        for num_tokens, hidden_size in BENCH_CASES:
            avg_s, flops, bytes_moved = run_mhc_fused_post_pre(
                num_tokens,
                hidden_size,
                args.warmup,
                args.iters,
            )
            latency_us, tflops, bandwidth, intensity = compute_metrics(
                avg_s,
                flops,
                bytes_moved,
            )
            rows.append(
                (
                    num_tokens,
                    hidden_size,
                    latency_us,
                    tflops,
                    bandwidth,
                    intensity,
                )
            )
        print_md_table("mhc_fused_post_pre", rows)
