#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Temporary PR before/after perf driver for the local-window (sliding window)
# right-boundary fix in chunk_prefill_kernel.hpp.
#
# Reuses the OFFICIAL measurement function `benchmark_varlen_with_paged_kv`
# (XPU-event kernel timing + FLOPS) from benchmark_cutlass_flash_attn_varlen.py,
# but drives it for local-window configs that the official `filter_configs`
# otherwise skips. Numbers are directly comparable across builds.
#
# Run inside container, from the benchmark/ directory:
#   ZE_AFFINITY_MASK=2 python3 benchmark_cutlass_flash_attn_window.py
import torch

# Importing the official module sets up sys.path (bootstrap_benchmark_env) and
# pulls in the kernel-time benchmark machinery.
from benchmark_cutlass_flash_attn_varlen import benchmark_varlen_with_paged_kv

DTYPE = torch.bfloat16

# Representative long-sequence prefill shape (q == k), GQA 32/8, head_size 128.
# window_size is given as the raw (left, right) flash-attn convention; the host
# normalizes local+causal into right=0 local-only internally.
BASE = dict(
    num_seqs=1,
    query_lens="4096",
    kv_lens="4096",
    num_heads=(32, 8),
    head_size=128,
    block_size=64,
    output_dtype=DTYPE,
    soft_cap=None,
    num_blocks=2048,
    fa_versions=2,
    q_dtype=None,
    is_sink=False,
    is_causal=True,
    is_paged=True,
    kv_dtype=None,
)

WINDOWS = [(64, 0), (128, 0), (256, 0), (512, 0), (1024, 0), (2048, 0),(-1, -1)]
ITERS = 20


def main():
    torch.set_default_device("xpu")
    torch.xpu.set_device("xpu:0")
    print(f"device={torch.xpu.get_device_name()}")
    print(f"shape: q=k={BASE['query_lens']} heads={BASE['num_heads']} "
          f"head_size={BASE['head_size']} causal={BASE['is_causal']} "
          f"paged={BASE['is_paged']} dtype={DTYPE}")
    print()
    print(f"{'window':>10} | {'kernel_time(us)':>16} | {'TFLOPS':>10}")
    print("-" * 44)
    for w in WINDOWS:
        cfg = dict(BASE, window_size=w)
        kt = benchmark_varlen_with_paged_kv(provider="flash_kernel_time",
                                            iterations=ITERS, **cfg)
        tf = benchmark_varlen_with_paged_kv(provider="flash_kernel_TFLOPS",
                                            iterations=ITERS, **cfg)
        print(f"{str(list(w)):>10} | {kt:>16.3f} | {tf:>10.3f}")


if __name__ == "__main__":
    main()
