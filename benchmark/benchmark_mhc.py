#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark comparing Sycl kernel vs Triton implementations for MHC layers.

Compares:
- _xpu_C.mhc_pre vs mhc_pre_xpu_triton
- _xpu_C.mhc_post vs mhc_post_xpu_triton

Scenarios:
- Assorted combinations of num_tokens, hidden_size, and hc_mult.
"""

import argparse
import gc
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    op_name: str
    num_tokens: int
    hidden_size: int
    hc_mult: int
    description: str


# ==============================================================================
# Triton Kernel Implementations
# ==============================================================================


@triton.jit
def _mhc_pre_fused_kernel(
    # Inputs
    res_ptr,         # [N, hc, H] fp32
    gemm_out_ptr,    # [N, hc3] fp32
    hc_scale_ptr,    # [3] fp32
    hc_base_ptr,     # [hc3] fp32
    # Outputs
    post_mix_ptr,    # [N, hc] fp32
    comb_mix_ptr,    # [N, hc*hc] fp32
    # Intermediate buffer for pre_mix scalars (for einsum kernel)
    pre_mix_buf_ptr, # [N, hc] fp32
    # Strides
    stride_res_n,    # = hc * H
    stride_gemm_n,   # = hc3
    stride_post_n,   # = hc
    stride_comb_n,   # = hc * hc
    # Scalar params
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    # Constexpr
    HC: tl.constexpr,           # 4
    HC3: tl.constexpr,          # 24
    HCH: tl.constexpr,         # hc * H = 16384
    SINKHORN_REPEAT: tl.constexpr,  # 20
    BLOCK_H: tl.constexpr,     # tile size for sqrsum reduction
):
    pid_n = tl.program_id(0)

    # --- Load constants ---
    scale0 = tl.load(hc_scale_ptr).to(tl.float32)
    scale1 = tl.load(hc_scale_ptr + 1).to(tl.float32)
    scale2 = tl.load(hc_scale_ptr + 2).to(tl.float32)

    # --- Pass 1: compute sqrsum = sum(res_2d^2) ---
    sqrsum = tl.zeros([], dtype=tl.float32)
    res_base = pid_n * stride_res_n
    for off in range(0, HCH, BLOCK_H):
        h_offs = tl.arange(0, BLOCK_H) + off
        mask = h_offs < HCH
        vals = tl.load(res_ptr + res_base + h_offs, mask=mask, other=0.0)
        sqrsum += tl.sum(vals * vals)

    # --- Compute RMS ---
    rms = tl.math.rsqrt(sqrsum / HCH + rms_eps)

    # --- Load gemm_out sections and apply rms ---
    pre_offs = tl.arange(0, 4)
    post_offs = tl.arange(0, 4) + HC
    cm_offs = tl.arange(0, 16) + 2 * HC

    pre_raw = tl.load(gemm_out_ptr + pid_n * stride_gemm_n + pre_offs) * rms
    post_raw = tl.load(gemm_out_ptr + pid_n * stride_gemm_n + post_offs) * rms
    cm_raw = tl.load(gemm_out_ptr + pid_n * stride_gemm_n + cm_offs) * rms

    # --- Sigmoid + scaling ---
    pre_base = tl.load(hc_base_ptr + pre_offs)
    pre_mix = tl.sigmoid(pre_raw * scale0 + pre_base) + hc_pre_eps  # [4]

    post_base = tl.load(hc_base_ptr + post_offs)
    post_mix = tl.sigmoid(post_raw * scale1 + post_base) * hc_post_mult_value  # [4]

    cm_base = tl.load(hc_base_ptr + cm_offs)
    cm = cm_raw * scale2 + cm_base  # [16]

    # --- Softmax along rows (dim=-1 in [4,4]) ---
    cm_2d = tl.reshape(cm, (HC, HC))
    row_max = tl.max(cm_2d, axis=1)  # [4]
    cm_2d = cm_2d - row_max[:, None]
    cm_2d = tl.exp(cm_2d)
    row_sum = tl.sum(cm_2d, axis=1)  # [4]
    cm_2d = cm_2d / row_sum[:, None]
    cm_2d = cm_2d + hc_sinkhorn_eps

    # --- Sinkhorn normalization ---
    col_sum = tl.sum(cm_2d, axis=0)  # [4]
    cm_2d = cm_2d / (col_sum[None, :] + hc_sinkhorn_eps)

    for _ in range(SINKHORN_REPEAT - 1):
        r_sum = tl.sum(cm_2d, axis=1)
        cm_2d = cm_2d / (r_sum[:, None] + hc_sinkhorn_eps)
        c_sum = tl.sum(cm_2d, axis=0)
        cm_2d = cm_2d / (c_sum[None, :] + hc_sinkhorn_eps)

    # --- Store post_mix and comb_mix ---
    tl.store(post_mix_ptr + pid_n * stride_post_n + tl.arange(0, 4), post_mix)
    comb_flat = tl.reshape(cm_2d, (HC * HC,))
    tl.store(comb_mix_ptr + pid_n * stride_comb_n + tl.arange(0, 16), comb_flat)

    # --- Store pre_mix to buffer for Pass 2 ---
    tl.store(pre_mix_buf_ptr + pid_n * HC + tl.arange(0, 4), pre_mix)


@triton.jit
def _mhc_pre_einsum_kernel(
    # Inputs
    res_ptr,         # [N, hc, H] fp32
    pre_mix_ptr,     # [N, hc] fp32
    # Output
    layer_input_ptr, # [N, H] bf16
    # Dims
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    # Load pre_mix scalars for this token
    pm0 = tl.load(pre_mix_ptr + pid_n * HC + 0).to(tl.float32)
    pm1 = tl.load(pre_mix_ptr + pid_n * HC + 1).to(tl.float32)
    pm2 = tl.load(pre_mix_ptr + pid_n * HC + 2).to(tl.float32)
    pm3 = tl.load(pre_mix_ptr + pid_n * HC + 3).to(tl.float32)

    res_base = pid_n * HC * H

    # Accumulate: layer_input[h] = pm0*res[0,h] + pm1*res[1,h] + ...
    r0 = tl.load(res_ptr + res_base + 0 * H + h_offs, mask=h_mask, other=0.0)
    r1 = tl.load(res_ptr + res_base + 1 * H + h_offs, mask=h_mask, other=0.0)
    r2 = tl.load(res_ptr + res_base + 2 * H + h_offs, mask=h_mask, other=0.0)
    r3 = tl.load(res_ptr + res_base + 3 * H + h_offs, mask=h_mask, other=0.0)

    acc = pm0 * r0 + pm1 * r1 + pm2 * r2 + pm3 * r3

    tl.store(layer_input_ptr + pid_n * H + h_offs, acc.to(tl.bfloat16), mask=h_mask)


@triton.jit
def _mhc_post_fused_kernel(
    # Inputs
    x_ptr,           # [N, H] bf16 - layer output
    res_ptr,         # [N, hc, H] bf16 - residual
    post_mix_ptr,    # [N, hc] fp32
    comb_mix_ptr,    # [N, hc*hc] fp32
    # Output
    out_ptr,         # [N, hc, H] bf16
    # Strides
    stride_x_n,      # = H
    stride_res_n,    # = hc * H
    stride_post_n,   # = hc
    stride_comb_n,   # = hc * hc
    stride_out_n,    # = hc * H
    # Constexpr
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    # Load post_mix scalars
    post_base = pid_n * stride_post_n
    p0 = tl.load(post_mix_ptr + post_base + 0).to(tl.float32)
    p1 = tl.load(post_mix_ptr + post_base + 1).to(tl.float32)
    p2 = tl.load(post_mix_ptr + post_base + 2).to(tl.float32)
    p3 = tl.load(post_mix_ptr + post_base + 3).to(tl.float32)

    # Load comb_mix scalars
    comb_base = pid_n * stride_comb_n
    c00 = tl.load(comb_mix_ptr + comb_base + 0).to(tl.float32)
    c01 = tl.load(comb_mix_ptr + comb_base + 1).to(tl.float32)
    c02 = tl.load(comb_mix_ptr + comb_base + 2).to(tl.float32)
    c03 = tl.load(comb_mix_ptr + comb_base + 3).to(tl.float32)
    c10 = tl.load(comb_mix_ptr + comb_base + 4).to(tl.float32)
    c11 = tl.load(comb_mix_ptr + comb_base + 5).to(tl.float32)
    c12 = tl.load(comb_mix_ptr + comb_base + 6).to(tl.float32)
    c13 = tl.load(comb_mix_ptr + comb_base + 7).to(tl.float32)
    c20 = tl.load(comb_mix_ptr + comb_base + 8).to(tl.float32)
    c21 = tl.load(comb_mix_ptr + comb_base + 9).to(tl.float32)
    c22 = tl.load(comb_mix_ptr + comb_base + 10).to(tl.float32)
    c23 = tl.load(comb_mix_ptr + comb_base + 11).to(tl.float32)
    c30 = tl.load(comb_mix_ptr + comb_base + 12).to(tl.float32)
    c31 = tl.load(comb_mix_ptr + comb_base + 13).to(tl.float32)
    c32 = tl.load(comb_mix_ptr + comb_base + 14).to(tl.float32)
    c33 = tl.load(comb_mix_ptr + comb_base + 15).to(tl.float32)

    # Load x[pid_n, h_offs]
    x_vals = tl.load(x_ptr + pid_n * stride_x_n + h_offs, mask=h_mask, other=0.0)
    x_f32 = x_vals.to(tl.float32)

    # Load residual[pid_n, j, h_offs] for j=0..3
    res_base = pid_n * stride_res_n
    r0 = tl.load(res_ptr + res_base + 0 * H + h_offs, mask=h_mask, other=0.0).to(tl.float32)
    r1 = tl.load(res_ptr + res_base + 1 * H + h_offs, mask=h_mask, other=0.0).to(tl.float32)
    r2 = tl.load(res_ptr + res_base + 2 * H + h_offs, mask=h_mask, other=0.0).to(tl.float32)
    r3 = tl.load(res_ptr + res_base + 3 * H + h_offs, mask=h_mask, other=0.0).to(tl.float32)

    # out[hco=0..3]
    out0 = p0 * x_f32 + c00 * r0 + c10 * r1 + c20 * r2 + c30 * r3
    out1 = p1 * x_f32 + c01 * r0 + c11 * r1 + c21 * r2 + c31 * r3
    out2 = p2 * x_f32 + c02 * r0 + c12 * r1 + c22 * r2 + c32 * r3
    out3 = p3 * x_f32 + c03 * r0 + c13 * r1 + c23 * r2 + c33 * r3

    # Store as bf16
    out_base = pid_n * stride_out_n
    tl.store(out_ptr + out_base + 0 * H + h_offs, out0.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 1 * H + h_offs, out1.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 2 * H + h_offs, out2.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 3 * H + h_offs, out3.to(tl.bfloat16), mask=h_mask)


# --- Triton Python wrappers ---


def mhc_pre_xpu_triton(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32

    hc = residual.shape[-2]
    H = residual.shape[-1]
    hc3 = hc * 2 + hc * hc
    hcH = hc * H

    outer_shape = residual.shape[:-2]
    N = 1
    for s in outer_shape:
        N *= s

    res_fp32 = residual.reshape(N, hc, H).to(torch.float32)
    res_2d = res_fp32.reshape(N, hcH)

    # Launch 1: GEMM
    gemm_out = res_2d @ fn.t()

    # Allocate outputs
    post_mix = torch.empty(N, hc, dtype=torch.float32, device=residual.device)
    comb_mix = torch.empty(N, hc * hc, dtype=torch.float32, device=residual.device)
    layer_input = torch.empty(N, H, dtype=torch.bfloat16, device=residual.device)
    pre_mix_buf = torch.empty(N, hc, dtype=torch.float32, device=residual.device)

    # Launch 2: fused sinkhorn kernel
    BLOCK_H = 1024
    grid_sinkhorn = (N,)

    _mhc_pre_fused_kernel[grid_sinkhorn](
        res_fp32, gemm_out, hc_scale, hc_base,
        post_mix, comb_mix, pre_mix_buf,
        hc * H, hc3, hc, hc * hc,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        HC=hc, HC3=hc3, HCH=hcH,
        SINKHORN_REPEAT=sinkhorn_repeat,
        BLOCK_H=BLOCK_H,
    )

    # Launch 3: einsum kernel
    BLOCK_H_EINSUM = 1024
    grid_einsum = (N, triton.cdiv(H, BLOCK_H_EINSUM))

    _mhc_pre_einsum_kernel[grid_einsum](
        res_fp32, pre_mix_buf, layer_input,
        H=H, HC=hc, BLOCK_H=BLOCK_H_EINSUM,
    )

    return (
        post_mix.reshape(*outer_shape, hc, 1),
        comb_mix.reshape(*outer_shape, hc, hc),
        layer_input.reshape(*outer_shape, H),
    )


def mhc_post_xpu_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    assert residual.dtype == torch.bfloat16

    hc = residual.shape[-2]
    H = residual.shape[-1]
    N_shape = residual.shape[:-2]
    N = 1
    for s in N_shape:
        N *= s

    x_flat = x.reshape(N, H)
    res_flat = residual.reshape(N, hc, H)
    post_flat = post_layer_mix.reshape(N, hc).to(torch.float32)
    comb_flat = comb_res_mix.reshape(N, hc * hc).to(torch.float32)

    out = torch.empty(N, hc, H, dtype=torch.bfloat16, device=residual.device)

    BLOCK_H = 1024
    grid = (N, triton.cdiv(H, BLOCK_H))

    _mhc_post_fused_kernel[grid](
        x_flat, res_flat, post_flat, comb_flat, out,
        H, hc * H, hc, hc * hc, hc * H,
        H=H, HC=hc, BLOCK_H=BLOCK_H,
    )

    return out.reshape(*N_shape, hc, H)


# ==============================================================================
# Benchmarking Engine
# ==============================================================================
def measure_memory() -> tuple[int, int]:
    """Return (allocated, reserved) memory in bytes."""
    torch.xpu.synchronize()
    return torch.xpu.memory_allocated(), torch.xpu.max_memory_allocated()


def reset_memory_stats():
    """Reset peak memory statistics."""
    torch.xpu.reset_peak_memory_stats()
    torch.xpu.empty_cache()
    gc.collect()


def format_memory(bytes_val: int) -> str:
    """Format memory in human-readable form."""
    if bytes_val >= 1024**3:
        return f"{bytes_val / (1024**3):.2f} GB"
    elif bytes_val >= 1024**2:
        return f"{bytes_val / (1024**2):.2f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.2f} KB"
    return f"{bytes_val} B"


def benchmark_function(
    func_callable,
    args: tuple,
    warmup_iters: int = 5,
    benchmark_iters: int = 20,
) -> tuple[float, int]:
    """
    Benchmark a function and return (avg_time_ms, peak_memory_bytes).
    """
    # Warmup
    for _ in range(warmup_iters):
        func_callable(*args)
    torch.xpu.synchronize()

    # Reset memory stats before benchmark
    reset_memory_stats()

    # Benchmark
    start_events = [torch.xpu.Event(enable_timing=True) for _ in range(benchmark_iters)]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(benchmark_iters)]

    for i in range(benchmark_iters):
        start_events[i].record()
        func_callable(*args)
        end_events[i].record()

    torch.xpu.synchronize()

    # Calculate timing
    times = [
        start_events[i].elapsed_time(end_events[i])
        for i in range(benchmark_iters)
    ]
    avg_time = sum(times) / len(times)

    # Get peak memory
    _, peak_memory = measure_memory()

    return avg_time, peak_memory


def create_benchmark_configs(
    token_sizes: list[int],
    hidden_sizes: list[int],
    hc_mults: list[int]
) -> list[BenchmarkConfig]:
    """Create all benchmark configurations."""
    configs = []
    
    # ops = ["mhc_pre", "mhc_post"]
    ops = ["mhc_pre","mhc_post"]

    for op in ops:
        for hidden in hidden_sizes:
            for hc in hc_mults:
                for tokens in token_sizes:
                    configs.append(
                        BenchmarkConfig(
                            name=f"{op}_t{tokens}_h{hidden}_hc{hc}",
                            op_name=op,
                            num_tokens=tokens,
                            hidden_size=hidden,
                            hc_mult=hc,
                            description=f"{op} | tokens={tokens}, hidden={hidden}, hc_mult={hc}"
                        )
                    )
    return configs


def run_op_benchmark(config: BenchmarkConfig, device: torch.device, dtype: torch.dtype, warmup_iters: int, benchmark_iters: int):
    # Setup inputs based on op
    n = config.num_tokens
    h = config.hidden_size
    hc = config.hc_mult
    rms_eps, hc_eps, hc_sinkhorn_eps = 1e-6, 1e-3, 1e-3
    
    sycl_func = None
    triton_func = None
    args = ()
    
    if config.op_name == "mhc_pre":
        hc3 = hc * 2 + hc * hc
        residual = torch.randn((n, hc, h), dtype=dtype, device=device)
        fn = torch.randn((hc3, hc * h), dtype=torch.float32, device=device)
        hc_scale = torch.randn((3,), dtype=torch.float32, device=device)
        hc_base = torch.randn((hc3,), dtype=torch.float32, device=device)
        
        args = (residual, fn, hc_scale, hc_base, rms_eps, hc_eps, hc_sinkhorn_eps, 1.0, 3)
        sycl_func = torch.ops._xpu_C.mhc_pre
        triton_func = mhc_pre_xpu_triton
        
    elif config.op_name == "mhc_post":
        x = torch.randn((n, h), dtype=dtype, device=device)
        residual = torch.randn((n, hc, h), dtype=dtype, device=device)
        post_mix = torch.randn((n, hc, 1), dtype=torch.float32, device=device)
        comb_mix = torch.randn((n, hc, hc), dtype=torch.float32, device=device)
        
        args = (x, residual, post_mix, comb_mix)
        sycl_func = torch.ops._xpu_C.mhc_post
        triton_func = mhc_post_xpu_triton

    # Sycl Bench
    sycl_time, sycl_mem = benchmark_function(sycl_func, args, warmup_iters, benchmark_iters)
    
    # Triton Bench
    triton_time, triton_mem = benchmark_function(triton_func, args, warmup_iters, benchmark_iters)

    speedup = triton_time / sycl_time if sycl_time > 0 else float("inf")
    mem_ratio = triton_mem / sycl_mem if sycl_mem > 0 else float("inf")
    
    return sycl_time, triton_time, sycl_mem, triton_mem, speedup, mem_ratio


def run_benchmark(
    configs: list[BenchmarkConfig],
    warmup_iters: int = 5,
    benchmark_iters: int = 30,
    verbose: bool = True,
):
    """Run all benchmarks and print results."""
    results = []

    print("=" * 120)
    print("MHC Benchmark: Sycl Kernel vs Triton")
    print("=" * 120)
    print()

    device = torch.device("xpu")
    dtype = torch.bfloat16

    for config in configs:
        if verbose:
            print(f"Running: {config.description}")

        sycl_time, triton_time, sycl_mem, triton_mem, speedup, mem_ratio = run_op_benchmark(
            config, device, dtype, warmup_iters, benchmark_iters
        )

        result = {
            "config": config,
            "sycl_time_ms": sycl_time,
            "triton_time_ms": triton_time,
            "sycl_mem": sycl_mem,
            "triton_mem": triton_mem,
            "speedup": speedup,
            "mem_ratio": mem_ratio,
        }
        results.append(result)

        if verbose:
            print(f"  Sycl:    {sycl_time:.3f} ms, {format_memory(sycl_mem)}")
            print(f"  Triton:  {triton_time:.3f} ms, {format_memory(triton_mem)}")
            print(f"  Speedup: {speedup:.2f}x, Memory ratio: {mem_ratio:.2f}x")
            print()

        reset_memory_stats()

    return results


def print_summary_table(results: list[dict]):
    """Print a summary table of results."""
    print()
    print("=" * 105)
    print("SUMMARY TABLE")
    print("=" * 105)
    print()

    header = (f"{'Operation':<15} | {'Tokens':>6} | {'Hidden':>6} | {'HC':>2} | "
              f"{'Sycl (ms)':>10} | {'Triton (ms)':>12} | {'Speedup':>8}")
    print(header)
    print("-" * 105)

    current_op = None
    for result in results:
        config = result["config"]

        if current_op != config.op_name:
            if current_op is not None:
                print("-" * 105)
            current_op = config.op_name

        print(f"{config.op_name:<15} | {config.num_tokens:>6} | {config.hidden_size:>6} | {config.hc_mult:>2} | "
              f"{result['sycl_time_ms']:>10.3f} | {result['triton_time_ms']:>12.3f} | {result['speedup']:>7.2f}x")

    print("=" * 105)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Sycl kernel vs Triton for MHC ops")
    parser.add_argument("--token-sizes", type=int, nargs="+", default=[1,33,128,256,1024,4096],
                        help="Token shapes / batch sizes to test.")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[4096,7168], ## 4096 for flash 7168 for pro
                        help="Hidden sizes to test.")
    parser.add_argument("--hc-mults", type=int, nargs="+", default=[4],
                        help="hc_mult configurations.")
    parser.add_argument("--warmup-iters", type=int, default=5,
                        help="Number of warmup iterations.")
    parser.add_argument("--benchmark-iters", type=int, default=20,
                        help="Number of benchmark iterations.")
    parser.add_argument("--quiet", action="store_true", help="Only print summary table")

    args = parser.parse_args()

    if not torch.xpu.is_available():
        print("ERROR: XPU is not available. This benchmark requires an XPU device.")
        return

    configs = create_benchmark_configs(args.token_sizes, args.hidden_sizes, args.hc_mults)

    results = run_benchmark(
        configs,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
        verbose=not args.quiet,
    )

    print_summary_table(results)

if __name__ == "__main__":
    main()