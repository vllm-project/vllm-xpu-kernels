# SPDX-License-Identifier: Apache-2.0

import argparse

import torch

import vllm_xpu_kernels._C  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401

HC = 4
NORM_EPS = 1e-6

BENCH_CASES = [
    (1, 4096),
    (4, 4096),
    (33, 4096),
    (128, 4096),
    (256, 4096),
    (1024, 4096),
    (2048, 4096),
    (4096, 4096),
    (8192, 4096),
    (16384, 4096),
    (1, 7168),
    (4, 7168),
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


def make_norm_weight(hidden_size: int, fuse_norm: bool):
    """Norm weight for the fused epilogue, or None to disable it."""
    if not fuse_norm:
        return None
    return torch.ones((hidden_size,), dtype=torch.bfloat16, device="xpu")


def vllm_rmsnorm(
    out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float
):
    """Unfused baseline norm.

    Uses the same `rms_norm` kernel vLLM dispatches to at runtime, so the
    fused-vs-unfused delta reflects what serving actually gains.
    """
    torch.ops._C.rms_norm(out, x, weight, eps)
    return out


def run_mhc_pre(
    num_tokens: int,
    hidden_size: int,
    warmup: int,
    iters: int,
    fuse_norm: bool = False,
):
    hc3 = HC * 2 + HC * HC
    residual = torch.randn(
        (num_tokens, HC, hidden_size),
        dtype=torch.bfloat16,
        device="xpu",
    )
    fn = torch.randn((hc3, HC * hidden_size), dtype=torch.float32, device="xpu")
    hc_scale = torch.randn((3,), dtype=torch.float32, device="xpu")
    hc_base = torch.randn((hc3,), dtype=torch.float32, device="xpu")
    norm_weight = make_norm_weight(hidden_size, fuse_norm)

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
            norm_weight,
            NORM_EPS,
        )

    avg_s = benchmark_op(run, warmup, iters)

    # FLOPs model for `mhc_pre`
    # - Stage 1 dominant cost is one dense matmul: [num_tokens, HC *
    #     hidden_size] x [HC * hidden_size, hc3]
    # - We count 2 FLOPs per MAC.
    flops = 2.0 * num_tokens * hc3 * (HC * hidden_size)
    if fuse_norm:
        # RMSNorm epilogue: square + add for the sum of squares, then
        # multiply by inv_rms and by the weight => ~4 FLOPs per element.
        flops += 4.0 * num_tokens * hidden_size

    # Bytes moved model for `mhc_pre` (ideal / theoretical minimum).
    # Note: the fused RMSNorm adds only the norm weight read; `layer_input`
    # is still written exactly once, which is the whole point of the fusion.
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
    if fuse_norm:
        bytes_moved += hidden_size * 2           # norm_weight
    return avg_s, flops, bytes_moved


def run_mhc_pre_unfused(
    num_tokens: int,
    hidden_size: int,
    warmup: int,
    iters: int,
):
    """Baseline: mhc_pre without fused norm + a separate RMSNorm pass.

    Compare against `run_mhc_pre(..., fuse_norm=True)` to quantify the
    benefit of fusing the trailing norm into the pre-stage epilogue.
    """
    hc3 = HC * 2 + HC * HC
    residual = torch.randn(
        (num_tokens, HC, hidden_size),
        dtype=torch.bfloat16,
        device="xpu",
    )
    fn = torch.randn((hc3, HC * hidden_size), dtype=torch.float32, device="xpu")
    hc_scale = torch.randn((3,), dtype=torch.float32, device="xpu")
    hc_base = torch.randn((hc3,), dtype=torch.float32, device="xpu")
    norm_weight = torch.ones(
        (hidden_size,), dtype=torch.bfloat16, device="xpu"
    )
    norm_out = torch.empty(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device="xpu"
    )

    rms_eps = 1e-6
    hc_pre_eps = 1e-3
    hc_sinkhorn_eps = 1e-3
    hc_post_mult_value = 1.0
    sinkhorn_repeat = 20

    def run():
        _, _, layer_input = torch.ops._xpu_C.mhc_pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            None,
            NORM_EPS,
        )
        vllm_rmsnorm(norm_out, layer_input, norm_weight, NORM_EPS)

    avg_s = benchmark_op(run, warmup, iters)

    flops = 2.0 * num_tokens * hc3 * (HC * hidden_size)
    flops += 4.0 * num_tokens * hidden_size

    bytes_moved = (
        num_tokens * HC * hidden_size * 2        # residual
        + hc3 * HC * hidden_size * 4             # fn
        + 3 * 4                                  # hc_scale
        + hc3 * 4                                # hc_base
        + num_tokens * HC * 4                    # post_mix
        + num_tokens * HC * HC * 4               # comb_mix
        + num_tokens * hidden_size * 2           # layer_input (write)
        # separate norm pass: read layer_input, read weight, write output
        + num_tokens * hidden_size * 2           # layer_input (read)
        + hidden_size * 2                        # norm_weight
        + num_tokens * hidden_size * 2           # normed output (write)
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
    fuse_norm: bool = False,
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
    norm_weight = make_norm_weight(hidden_size, fuse_norm)

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
            norm_weight, NORM_EPS,
        )

    avg_s = benchmark_op(run, warmup, iters)

    # FLOPs model for `mhc_fused_post_pre` (Phase 0: composed)
    flops_post = num_tokens * HC * hidden_size * (2.0 * HC + 1.0)
    flops_pre = 2.0 * num_tokens * hc3 * (HC * hidden_size)
    flops = flops_post + flops_pre
    if fuse_norm:
        flops += 4.0 * num_tokens * hidden_size

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
    if fuse_norm:
        bytes_moved += hidden_size * 2           # norm_weight
    return avg_s, flops, bytes_moved


def run_mhc_fused_post_pre_unfused(
    num_tokens: int,
    hidden_size: int,
    warmup: int,
    iters: int,
):
    """Baseline: post_pre without fused norm + a separate RMSNorm pass."""
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
    norm_weight = torch.ones(
        (hidden_size,), dtype=torch.bfloat16, device="xpu"
    )
    norm_out = torch.empty(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device="xpu"
    )

    rms_eps = 1e-6
    hc_pre_eps = 1e-3
    hc_sinkhorn_eps = 1e-3
    hc_post_mult_value = 1.0
    sinkhorn_repeat = 20

    def run():
        _, _, _, layer_input = torch.ops._xpu_C.mhc_fused_post_pre(
            x, residual, post_mix, comb_mix,
            fn, hc_scale, hc_base,
            rms_eps, hc_pre_eps, hc_sinkhorn_eps,
            hc_post_mult_value, sinkhorn_repeat,
            None, NORM_EPS,
        )
        vllm_rmsnorm(norm_out, layer_input, norm_weight, NORM_EPS)

    avg_s = benchmark_op(run, warmup, iters)

    flops_post = num_tokens * HC * hidden_size * (2.0 * HC + 1.0)
    flops_pre = 2.0 * num_tokens * hc3 * (HC * hidden_size)
    flops = flops_post + flops_pre + 4.0 * num_tokens * hidden_size

    bytes_moved = (
        num_tokens * hidden_size * 2             # x
        + num_tokens * HC * hidden_size * 2      # residual
        + num_tokens * HC * 4                    # post_mix
        + num_tokens * HC * HC * 4               # comb_mix
        + hc3 * HC * hidden_size * 4             # fn
        + 3 * 4                                  # hc_scale
        + hc3 * 4                                # hc_base
        + num_tokens * HC * hidden_size * 2      # residual_cur
        + num_tokens * HC * 4                    # post_mix_cur
        + num_tokens * HC * HC * 4               # comb_mix_cur
        + num_tokens * hidden_size * 2           # layer_input (write)
        # separate norm pass
        + num_tokens * hidden_size * 2           # layer_input (read)
        + hidden_size * 2                        # norm_weight
        + num_tokens * hidden_size * 2           # normed output (write)
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


def print_speedup_table(op_name: str, rows):
    """Print fused-vs-unfused latency comparison."""
    print(f"\n## {op_name}\n")
    print(
        "| num_tokens | hidden_size | Unfused (us) | Fused (us) | Speedup |"
    )
    print(
        "|------------|-------------|--------------|------------|---------|"
    )
    for num_tokens, hidden_size, unfused_us, fused_us in rows:
        speedup = unfused_us / fused_us if fused_us else 0.0
        print(
            f"| {num_tokens:>10d} | {hidden_size:>11d} "
            f"| {unfused_us:>12.3f} | {fused_us:>10.3f} "
            f"| {speedup:>6.2f}x |"
        )


def bench_table(op_name: str, runner, cases, warmup, iters, **kwargs):
    rows = []
    for num_tokens, hidden_size in cases:
        avg_s, flops, bytes_moved = runner(
            num_tokens,
            hidden_size,
            warmup,
            iters,
            **kwargs,
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
    print_md_table(op_name, rows)


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
    parser.add_argument(
        "--fuse-norm",
        action="store_true",
        help="Enable the fused epilogue RMSNorm in mhc_pre / "
             "mhc_fused_post_pre.",
    )
    parser.add_argument(
        "--compare-norm",
        action="store_true",
        help="Also run the unfused (kernel + separate RMSNorm) baseline and "
             "report the speedup from fusing the norm. Implies --fuse-norm.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=300)
    args = parser.parse_args()

    fuse_norm = args.fuse_norm or args.compare_norm
    norm_tag = " (fused norm)" if fuse_norm else ""

    torch.manual_seed(0)
    torch.set_default_device("xpu")

    if args.op in ("mhc_pre", "all"):
        bench_table(
            f"mhc_pre{norm_tag}",
            run_mhc_pre,
            BENCH_CASES,
            args.warmup,
            args.iters,
            fuse_norm=fuse_norm,
        )

        if args.compare_norm:
            rows = []
            for num_tokens, hidden_size in BENCH_CASES:
                unfused_s, _, _ = run_mhc_pre_unfused(
                    num_tokens, hidden_size, args.warmup, args.iters
                )
                fused_s, _, _ = run_mhc_pre(
                    num_tokens,
                    hidden_size,
                    args.warmup,
                    args.iters,
                    fuse_norm=True,
                )
                rows.append((num_tokens, hidden_size, unfused_s, fused_s))
            print_speedup_table("mhc_pre: norm fusion speedup", rows)

    if args.op in ("mhc_post", "all"):
        bench_table(
            "mhc_post",
            run_mhc_post,
            BENCH_CASES,
            args.warmup,
            args.iters,
        )

    if args.op in ("hc_head_fused", "all"):
        bench_table(
            "hc_head_fused",
            run_hc_head_fused,
            BENCH_CASES,
            args.warmup,
            args.iters,
        )

    if args.op in ("mhc_fused_post_pre", "all"):
        bench_table(
            f"mhc_fused_post_pre{norm_tag}",
            run_mhc_fused_post_pre,
            BENCH_CASES,
            args.warmup,
            args.iters,
            fuse_norm=fuse_norm,
        )

        if args.compare_norm:
            rows = []
            for num_tokens, hidden_size in BENCH_CASES:
                unfused_s, _, _ = run_mhc_fused_post_pre_unfused(
                    num_tokens, hidden_size, args.warmup, args.iters
                )
                fused_s, _, _ = run_mhc_fused_post_pre(
                    num_tokens,
                    hidden_size,
                    args.warmup,
                    args.iters,
                    fuse_norm=True,
                )
                rows.append((num_tokens, hidden_size, unfused_s, fused_s))
            print_speedup_table(
                "mhc_fused_post_pre: norm fusion speedup", rows
            )
