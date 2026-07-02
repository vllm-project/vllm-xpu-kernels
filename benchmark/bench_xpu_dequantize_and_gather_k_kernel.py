# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DeepseekV4 dequantize_and_gather_k_cache on XPU.

Compares Triton and SYCL implementations side-by-side with speedup calculation.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import sys
import tempfile
import time
from glob import glob
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _peek_cli_value(argv: list[str], flag: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


def _argv_requests_sycl_backend(argv: list[str]) -> bool:
    backend = _peek_cli_value(argv, "--backend")
    return backend in {"sycl", "both"} or "--validate-sycl" in argv


def _bootstrap_sycl_kernels_path_from_argv(argv: list[str]) -> None:
    if not _argv_requests_sycl_backend(argv):
        return
    sycl_root_arg = _peek_cli_value(argv, "--sycl-kernels-root")
    candidate_root = (
        REPO_ROOT.parent / "applications.ai.gpu.vllm-xpu-kernels"
        if sycl_root_arg is None
        else Path(sycl_root_arg).resolve()
    )
    if candidate_root.exists() and str(candidate_root) not in sys.path:
        sys.path.insert(0, str(candidate_root))


_bootstrap_sycl_kernels_path_from_argv(sys.argv[1:])

import torch

try:
    from tabulate import tabulate
except ModuleNotFoundError:
    tabulate = None

from vllm.triton_utils import HAS_TRITON
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.attention.ops.deepseek_v4_ops.cache_utils import (
    _dequantize_and_gather_k_kernel,
    dequantize_and_gather_k_cache,
)

HEAD_DIM = 512
FP8_DIM = 448
BF16_DIM = 64
SCALE_DIM = 8
TOKEN_DATA_BYTES = FP8_DIM + BF16_DIM * 2
NUM_QUANT_BLOCKS = 7
FP8_MAX = 448.0

BF16_BYTES = torch.tensor([], dtype=torch.bfloat16).element_size()
INT32_BYTES = torch.tensor([], dtype=torch.int32).element_size()
UINT8_BYTES = torch.tensor([], dtype=torch.uint8).element_size()

KERNEL_NAME_SUBSTRING = "dequantize_and_gather"
B60_PEAK_BANDWIDTH_GBPS = 456.0
B60_CACHE_SIZE_MB = 512

GatherOp = Callable[..., None]


# ============================================================================
# Hardcoded benchmark cases
# ============================================================================
@dataclass(frozen=True)
class BenchmarkCase:
    scenario: str
    compress_ratio: int
    logical_seq_len: int
    raw_gather_len: int | None
    kernel_block_size: int
    offset: int

    @property
    def api_seq_len(self) -> int:
        return self.logical_seq_len if self.compress_ratio <= 1 else self.logical_seq_len // self.compress_ratio

    @property
    def api_gather_len(self) -> int | None:
        return None if self.compress_ratio > 1 else self.raw_gather_len

    @property
    def copied_tokens(self) -> int:
        return self.api_seq_len if self.api_gather_len is None else self.api_gather_len

    @property
    def output_tokens(self) -> int:
        return self.offset + self.copied_tokens


BUILTIN_CASES: list[BenchmarkCase] = [
    # SWA-only layer 0/1
    BenchmarkCase("SWA-only", 1, 128, 128, 64, 0),
    BenchmarkCase("SWA-only", 1, 512, 512, 64, 0),
    BenchmarkCase("SWA-only", 1, 1024, 1024, 64, 0),
    BenchmarkCase("SWA-only", 1, 8192, 8192, 64, 0),
    BenchmarkCase("SWA-only", 1, 16384, 16384, 64, 0),
    BenchmarkCase("SWA-only", 1, 32768, 32768, 64, 0),
    BenchmarkCase("SWA-only", 1, 65536, 65536, 64, 0),
    BenchmarkCase("SWA-only", 1, 102400, 102400, 64, 0),
    # C4 compressed full gather
    BenchmarkCase("C4-compressed", 4, 128, None, 64, 0),
    BenchmarkCase("C4-compressed", 4, 2048, None, 64, 0),
    BenchmarkCase("C4-compressed", 4, 4096, None, 64, 0),
    BenchmarkCase("C4-compressed", 4, 8192, None, 64, 0),
    BenchmarkCase("C4-compressed", 4, 16384, None, 64, 0),
    BenchmarkCase("C4-compressed", 4, 32768, None, 64, 0),
    BenchmarkCase("C4-compressed", 4, 65536, None, 64, 0),
    BenchmarkCase("C4-compressed", 4, 102400, None, 64, 0),
    # C128 compressed full gather
    BenchmarkCase("C128-compressed", 128, 8192, None, 2, 0),
    BenchmarkCase("C128-compressed", 128, 16384, None, 2, 0),
    BenchmarkCase("C128-compressed", 128, 32768, None, 2, 0),
    BenchmarkCase("C128-compressed", 128, 65536, None, 2, 0),
    BenchmarkCase("C128-compressed", 128, 102400, None, 2, 0),
]


# ============================================================================
# Runtime setup helpers
# ============================================================================
def _default_sycl_kernels_root() -> Path | None:
    p = REPO_ROOT.parent / "applications.ai.gpu.vllm-xpu-kernels"
    return p if p.exists() else None


def _require_xpu() -> None:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("This benchmark requires an available XPU device.")


def _require_triton_runtime() -> None:
    if not HAS_TRITON or not hasattr(_dequantize_and_gather_k_kernel, "__getitem__"):
        raise RuntimeError("Triton runtime is unavailable for dequantize_and_gather_k_cache.")


def _has_sycl_op() -> bool:
    cache_ops = getattr(torch.ops, "_C_cache_ops", None)
    return cache_ops is not None and hasattr(cache_ops, "dequantize_and_gather_k_cache")


def _require_sycl_runtime(sycl_kernels_root: Path | None, sycl_library: Path | None) -> None:
    if _has_sycl_op():
        return
    errors: list[str] = []
    if sycl_kernels_root is not None and str(sycl_kernels_root) not in sys.path:
        sys.path.insert(0, str(sycl_kernels_root))
    try:
        import vllm_xpu_kernels._C  # noqa: F401
    except Exception as exc:
        errors.append(f"import vllm_xpu_kernels._C failed: {exc}")
    if _has_sycl_op():
        return

    candidates: list[Path] = []
    if sycl_library is not None:
        candidates.append(sycl_library)
    if sycl_kernels_root is not None:
        for pattern in ("vllm_xpu_kernels/_C*.so", "build/temp/_C*.so"):
            candidates.extend(Path(p) for p in glob(str(sycl_kernels_root / pattern)))

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            torch.ops.load_library(str(candidate))
        except OSError as exc:
            if "Only a single TORCH_LIBRARY" not in str(exc):
                errors.append(f"load_library({candidate}) failed: {exc}")
        if _has_sycl_op():
            return

    if not _has_sycl_op():
        details = " | ".join(errors) if errors else "no loader details"
        raise RuntimeError(
            f"SYCL op dequantize_and_gather_k_cache unavailable. Details: {details}"
        )


# ============================================================================
# Backend op wrappers
# ============================================================================
def dequantize_and_gather_k_cache_triton(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    num_reqs = seq_lens.shape[0]
    num_workers = 128
    _dequantize_and_gather_k_kernel[(num_reqs, num_workers)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        offset,
        gather_lens,
        max_blocks_per_seq=block_table.shape[-1],
        fp8_dim=FP8_DIM,
        bf16_dim=BF16_DIM,
        scale_dim=SCALE_DIM,
        quant_block=64,
        cache_block_size=block_size,
        token_data_size=TOKEN_DATA_BYTES,
        block_stride=k_cache.stride(0),
        output_dim=HEAD_DIM,
        fp8_max=FP8_MAX,
        n_quant_blocks=NUM_QUANT_BLOCKS,
    )


def dequantize_and_gather_k_cache_sycl(out, k_cache, seq_lens, gather_lens, block_table, block_size, offset):
    torch.ops._C_cache_ops.dequantize_and_gather_k_cache(out, k_cache, seq_lens, gather_lens, block_table, block_size, offset)


# ============================================================================
# Benchmark infrastructure
# ============================================================================
def _synchronize() -> None:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()


def _release_device_memory() -> None:
    gc.collect()
    if hasattr(torch, "xpu") and hasattr(torch.xpu, "empty_cache"):
        torch.xpu.empty_cache()


def _profiler_activities(device: str) -> list[Any]:
    acts = [torch.profiler.ProfilerActivity.CPU]
    if device.startswith("xpu") and hasattr(torch.profiler.ProfilerActivity, "XPU"):
        acts.append(torch.profiler.ProfilerActivity.XPU)
    return acts


def _kernel_latency_samples_us(trace_path: Path) -> list[float]:
    profile = json.loads(trace_path.read_text(encoding="utf-8"))
    samples: list[float] = []
    for event in profile.get("traceEvents", []):
        if event.get("cat") == "kernel" and KERNEL_NAME_SUBSTRING in str(event.get("name", "")):
            dur = event.get("dur")
            if isinstance(dur, (int, float)):
                samples.append(float(dur))
    if not samples:
        raise RuntimeError("Profiler trace missing dequantize_and_gather kernel event.")
    return samples


def _make_valid_fp8_bytes(shape: tuple[int, ...], device: str) -> torch.Tensor:
    vals = torch.randn(*shape, dtype=torch.float32, device=device).clamp(-FP8_MAX, FP8_MAX)
    return vals.to(torch.float8_e4m3fn).view(torch.uint8)


def build_inputs(case: BenchmarkCase, batch_size: int, device: str) -> dict[str, torch.Tensor | int | None]:
    seq_len = case.api_seq_len
    block_size = case.kernel_block_size
    blocks_per_req = math.ceil(seq_len / block_size)
    total_blocks = batch_size * blocks_per_req

    block_bytes = block_size * TOKEN_DATA_BYTES + block_size * SCALE_DIM
    k_cache = torch.empty(total_blocks, block_bytes, dtype=torch.uint8, device=device)
    token_data = k_cache[:, :block_size * TOKEN_DATA_BYTES].view(total_blocks, block_size, TOKEN_DATA_BYTES)
    token_data[:, :, :FP8_DIM] = _make_valid_fp8_bytes((total_blocks, block_size, FP8_DIM), device)
    bf16_tail = torch.randn(total_blocks, block_size, BF16_DIM, dtype=torch.bfloat16, device=device)
    token_data[:, :, FP8_DIM:TOKEN_DATA_BYTES] = bf16_tail.view(torch.uint8).view(total_blocks, block_size, BF16_DIM * 2)
    scales = k_cache[:, block_size * TOKEN_DATA_BYTES:].view(total_blocks, block_size, SCALE_DIM)
    scales[:, :, :NUM_QUANT_BLOCKS] = torch.randint(120, 136, (total_blocks, block_size, NUM_QUANT_BLOCKS), dtype=torch.uint8, device=device)
    scales[:, :, NUM_QUANT_BLOCKS] = 0

    out = torch.empty(batch_size, case.output_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    gather_lens = None if case.api_gather_len is None else torch.full((batch_size,), case.api_gather_len, dtype=torch.int32, device=device)
    block_table = torch.empty(batch_size, blocks_per_req, dtype=torch.int32, device=device)
    for req_idx in range(batch_size):
        start = req_idx * blocks_per_req
        block_table[req_idx] = torch.arange(start, start + blocks_per_req, dtype=torch.int32, device=device)

    return {"out": out, "k_cache": k_cache, "seq_lens": seq_lens, "gather_lens": gather_lens, "block_table": block_table, "block_size": block_size, "offset": case.offset}


def _run_backend_op(backend_op: GatherOp, tensors: dict[str, torch.Tensor | int | None], out: torch.Tensor | None = None) -> None:
    target_out = tensors["out"] if out is None else out
    backend_op(target_out, tensors["k_cache"], tensors["seq_lens"], tensors["gather_lens"], tensors["block_table"], tensors["block_size"], tensors["offset"])


def benchmark_e2e_us(fn, warmup_iters: int, measure_iters: int) -> float:
    for _ in range(warmup_iters):
        fn()
        _synchronize()
    t0 = time.perf_counter()
    for _ in range(measure_iters):
        fn()
    _synchronize()
    return (time.perf_counter() - t0) * 1e6 / measure_iters


def benchmark_kernel_us(fn, device: str, warmup_iters: int, measure_iters: int) -> float:
    with tempfile.TemporaryDirectory(prefix="bench_xpu_dequantize_and_gather_") as td:
        trace_path = Path(td) / "trace.pt.trace.json"
        schedule = torch.profiler.schedule(wait=0, warmup=warmup_iters, active=measure_iters, repeat=1)
        with torch.profiler.profile(activities=_profiler_activities(device), schedule=schedule, on_trace_ready=lambda p: p.export_chrome_trace(str(trace_path)), record_shapes=False, with_stack=False) as prof:
            for _ in range(warmup_iters + measure_iters):
                fn()
                _synchronize()
                prof.step()
        return sum(_kernel_latency_samples_us(trace_path)) / measure_iters


def compute_bandwidth_gbps(case: BenchmarkCase, batch_size: int, kernel_us: float) -> float:
    """Compute effective bandwidth in GB/s."""
    copied_tokens = case.copied_tokens * batch_size
    read_bytes = copied_tokens * (FP8_DIM + BF16_DIM * 2 + NUM_QUANT_BLOCKS * UINT8_BYTES) + copied_tokens * INT32_BYTES
    write_bytes = copied_tokens * HEAD_DIM * BF16_BYTES
    total_bytes = read_bytes + write_bytes
    if kernel_us <= 0:
        return 0.0
    return total_bytes / (kernel_us * 1e-6) / 1e9


# ============================================================================
# Format helpers
# ============================================================================
def format_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<empty>"
    if tabulate is not None:
        return tabulate(rows, headers="keys", tablefmt="github", stralign="right")
    headers = list(rows[0].keys())
    widths = {h: max(len(str(h)), *(len(str(r[h])) for r in rows)) for h in headers}
    hdr = " | ".join(h.rjust(widths[h]) for h in headers)
    div = "-+-".join("-" * widths[h] for h in headers)
    body = "\n".join(" | ".join(str(r[h]).rjust(widths[h]) for h in headers) for r in rows)
    return f"{hdr}\n{div}\n{body}"


def format_speedup(triton_val: float, sycl_val: float) -> str:
    if triton_val <= 0 or sycl_val <= 0:
        return "N/A"
    return f"{triton_val / sycl_val:.2f}x"


# ============================================================================
# Main benchmark logic
# ============================================================================
def run_comparison(
    cases: list[BenchmarkCase],
    batch_size: int,
    device: str,
    warmup_iters: int,
    measure_iters: int,
    peak_bw_gbps: float,
) -> list[dict[str, Any]]:
    """Run both backends for each case, return side-by-side comparison rows."""
    results: list[dict[str, Any]] = []

    for case in cases:
        tensors = build_inputs(case, batch_size, device)
        triton_fn = lambda t=tensors: _run_backend_op(dequantize_and_gather_k_cache_triton, t)
        sycl_fn = lambda t=tensors: _run_backend_op(dequantize_and_gather_k_cache_sycl, t)

        # Triton timing
        triton_e2e = benchmark_e2e_us(triton_fn, warmup_iters, measure_iters)
        triton_kernel = benchmark_kernel_us(triton_fn, device, max(1, warmup_iters // 2), measure_iters)
        triton_bw = compute_bandwidth_gbps(case, batch_size, triton_kernel)

        # SYCL timing
        sycl_e2e = benchmark_e2e_us(sycl_fn, warmup_iters, measure_iters)
        sycl_kernel = benchmark_kernel_us(sycl_fn, device, max(1, warmup_iters // 2), measure_iters)
        sycl_bw = compute_bandwidth_gbps(case, batch_size, sycl_kernel)

        results.append({
            "scenario": case.scenario,
            "ratio": case.compress_ratio,
            "api_seq": case.api_seq_len,
            "gather": case.api_gather_len if case.api_gather_len is not None else "full",
            "blk": case.kernel_block_size,
            "batch": batch_size,
            "triton_e2e_us": f"{triton_e2e:.1f}",
            "sycl_e2e_us": f"{sycl_e2e:.1f}",
            "e2e_speedup": format_speedup(triton_e2e, sycl_e2e),
            "triton_kern_us": f"{triton_kernel:.1f}",
            "sycl_kern_us": f"{sycl_kernel:.1f}",
            "kern_speedup": format_speedup(triton_kernel, sycl_kernel),
            "triton_bw": f"{triton_bw:.1f}",
            "sycl_bw": f"{sycl_bw:.1f}",
            "sycl_mbu%": f"{sycl_bw / peak_bw_gbps * 100:.1f}",
        })

        _release_device_memory()

    return results


def run_single_backend(
    cases: list[BenchmarkCase],
    batch_size: int,
    device: str,
    backend_name: str,
    backend_op: GatherOp,
    warmup_iters: int,
    measure_iters: int,
    peak_bw_gbps: float,
) -> list[dict[str, Any]]:
    """Run a single backend for each case."""
    results: list[dict[str, Any]] = []
    for case in cases:
        tensors = build_inputs(case, batch_size, device)
        fn = lambda t=tensors: _run_backend_op(backend_op, t)

        e2e = benchmark_e2e_us(fn, warmup_iters, measure_iters)
        kernel = benchmark_kernel_us(fn, device, max(1, warmup_iters // 2), measure_iters)
        bw = compute_bandwidth_gbps(case, batch_size, kernel)

        results.append({
            "scenario": case.scenario,
            "ratio": case.compress_ratio,
            "api_seq": case.api_seq_len,
            "gather": case.api_gather_len if case.api_gather_len is not None else "full",
            "blk": case.kernel_block_size,
            "batch": batch_size,
            "backend": backend_name,
            "e2e_us": f"{e2e:.1f}",
            "kernel_us": f"{kernel:.1f}",
            "bw_gbps": f"{bw:.1f}",
            "mbu%": f"{bw / peak_bw_gbps * 100:.1f}",
        })
        _release_device_memory()
    return results


def maybe_write_csv(results: list[dict[str, Any]], csv_path: Path | None) -> None:
    if csv_path is None or not results:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved to: {csv_path}")


def build_parser() -> FlexibleArgumentParser:
    p = FlexibleArgumentParser(description="Benchmark dequantize_and_gather_k_cache (Triton vs SYCL)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup-iters", type=int, default=10)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--device", type=str, default="xpu")
    p.add_argument("--backend", type=str, choices=("triton", "sycl", "both"), default="both")
    p.add_argument("--validate-sycl", action="store_true", help="Validate SYCL correctness vs Triton before benchmarking")
    p.add_argument("--sycl-kernels-root", type=Path, default=_default_sycl_kernels_root())
    p.add_argument("--sycl-library", type=Path, default=None)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--csv", type=Path, default=None, help="Output CSV file path")
    p.add_argument("--peak-bw-gbps", type=float, default=B60_PEAK_BANDWIDTH_GBPS,
                   help=f"Peak memory bandwidth in GB/s (default: {B60_PEAK_BANDWIDTH_GBPS})")
    return p


@torch.inference_mode()
def main(args) -> None:
    _require_xpu()
    cases = BUILTIN_CASES[:]
    if args.max_cases is not None:
        cases = cases[:args.max_cases]

    print(f"Benchmarking {len(cases)} cases, batch_size={args.batch_size}, "
          f"warmup={args.warmup_iters}, iters={args.iters}, backend={args.backend}")

    # Setup backends
    if args.backend in {"triton", "both"}:
        _require_triton_runtime()
    if args.backend in {"sycl", "both"}:
        _require_sycl_runtime(args.sycl_kernels_root, args.sycl_library)

    # Correctness validation
    if args.validate_sycl:
        print("\n=== Correctness Validation (SYCL vs Triton) ===")
        _require_triton_runtime()
        _require_sycl_runtime(args.sycl_kernels_root, args.sycl_library)
        for case in cases:
            tensors = build_inputs(case, args.batch_size, args.device)
            triton_out = torch.zeros_like(tensors["out"])
            sycl_out = torch.zeros_like(tensors["out"])
            _run_backend_op(dequantize_and_gather_k_cache_triton, tensors, triton_out)
            _run_backend_op(dequantize_and_gather_k_cache_sycl, tensors, sycl_out)
            _synchronize()
            max_diff = (sycl_out.float() - triton_out.float()).abs().max().item()
            torch.testing.assert_close(sycl_out, triton_out, atol=2e-2, rtol=1e-2)
            print(f"  PASS: {case.scenario} seq={case.api_seq_len} max_diff={max_diff:.6f}")
            _release_device_memory()
        print("  All cases passed!\n")

    # Benchmark
    if args.backend == "both":
        print("\n=== Performance Comparison (Triton vs SYCL) ===")
        results = run_comparison(
            cases, args.batch_size, args.device,
            args.warmup_iters, args.iters, args.peak_bw_gbps,
        )
    elif args.backend == "triton":
        print("\n=== Triton Performance ===")
        results = run_single_backend(
            cases, args.batch_size, args.device, "triton",
            dequantize_and_gather_k_cache_triton,
            args.warmup_iters, args.iters, args.peak_bw_gbps,
        )
    else:
        print("\n=== SYCL Performance ===")
        results = run_single_backend(
            cases, args.batch_size, args.device, "sycl",
            dequantize_and_gather_k_cache_sycl,
            args.warmup_iters, args.iters, args.peak_bw_gbps,
        )

    print(format_table(results))
    maybe_write_csv(results, args.csv)


if __name__ == "__main__":
    main(build_parser().parse_args())