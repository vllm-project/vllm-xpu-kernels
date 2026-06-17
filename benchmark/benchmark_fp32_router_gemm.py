# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""Benchmark fp32_router_gemm (SYCL) vs torch.matmul on XPU.

The MoE router computes ``out[m, n] = sum_k mat_a[m, k] * mat_b[n, k]`` in fp32
where ``mat_a`` (activation) is bf16 or fp32 and ``mat_b`` (weight) is fp32.
The reference is ``torch.matmul(mat_a.float(), mat_b.t())`` (fp32 accumulate),
which is what the kernel replaces. Reports per-call latency and speedup for the
MiniMax-M3 router shape (num_tokens <= 32, N=256 experts, K=3072 hidden).

Run: python benchmark/benchmark_fp32_router_gemm.py
"""

import itertools

import torch
import triton
from utils import bootstrap_benchmark_env

bootstrap_benchmark_env(__file__)

from tests import register_ops as vllm_ops

DEVICE = torch.device("xpu")


def torch_router_gemm(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    # fp32-accumulate reference the SYCL kernel replaces.
    return torch.matmul(mat_a.float(), mat_b.t())


def _bench(fn) -> float:
    ms, _, _ = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    return ms * 1e3  # microseconds


def run(num_tokens: int, num_experts: int, hidden: int, act_dtype: torch.dtype):
    torch.manual_seed(0)
    mat_a = torch.randn(num_tokens, hidden, device=DEVICE, dtype=act_dtype)
    mat_b = torch.randn(num_experts, hidden, device=DEVICE, dtype=torch.float32)

    out = vllm_ops.fp32_router_gemm(mat_a, mat_b)
    ref = torch_router_gemm(mat_a, mat_b)
    max_abs = (out - ref).abs().max().item()

    sycl_us = _bench(lambda: vllm_ops.fp32_router_gemm(mat_a, mat_b))
    torch_us = _bench(lambda: torch_router_gemm(mat_a, mat_b))
    return sycl_us, torch_us, max_abs


def main():
    num_experts, hidden = 256, 3072
    token_grid = [1, 4, 8, 16, 32]
    dtypes = [torch.bfloat16, torch.float32]

    # speedup = torch.matmul / sycl wall time; > 1 means the SYCL kernel is
    # faster, < 1 means torch.matmul is faster.
    header = (
        f"{'act':>7} {'M':>4} {'N':>5} {'K':>6} "
        f"{'torch.matmul(us)':>17} {'sycl(us)':>10} {'speedup':>8} {'max_abs':>9}"
    )
    print(header)
    print("-" * len(header))
    for act_dtype, m in itertools.product(dtypes, token_grid):
        sycl_us, torch_us, max_abs = run(m, num_experts, hidden, act_dtype)
        name = "bf16" if act_dtype == torch.bfloat16 else "fp32"
        print(
            f"{name:>7} {m:>4} {num_experts:>5} {hidden:>6} "
            f"{torch_us:>17.2f} {sycl_us:>10.2f} {torch_us / sycl_us:>7.2f}x "
            f"{max_abs:>9.2e}"
        )


if __name__ == "__main__":
    main()
