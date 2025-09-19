# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from typing import Optional

import torch
import triton
import random
from torch import Tensor

from tests import register_ops as vllm_ops
from tests.utils import (
    check_ipex_availability,
    create_kv_caches_with_random,
    parse_args,
)

HAS_IPEX = check_ipex_availability()

if HAS_IPEX:
    import intel_extension_for_pytorch as ipex


def reshape_and_cache_vllm(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> None:
    """vLLM's fused kernel for reshaping and caching K/V tensors."""
    vllm_ops.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping,
                               kv_cache_dtype, k_scale, v_scale)


def reshape_and_cache_ipex(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> None:
    """IPEX native implementation using ipex.llm.modules.PagedAttention."""
    if not HAS_IPEX:
        raise RuntimeError("IPEX is not available")
    assert kv_cache_dtype == "auto", "IPEX reshape_and_cache uses 'auto' mode"

    ipex.llm.modules.PagedAttention.reshape_and_cache(
        key, value, key_cache, value_cache, slot_mapping
    )


def get_benchmark(
    dtype: torch.dtype,
    device: str = "xpu",
):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens", "num_heads", "head_size", "block_size", "num_blocks"],
            x_vals=configs,
            line_arg="provider",
            line_vals=["vllm", "ipex"] if HAS_IPEX else ["vllm"],
            line_names=["vLLM", "IPEX"] if HAS_IPEX else ["vLLM"],
            styles=[("blue", "-"), ("red", "-")] if HAS_IPEX else [("blue", "-")],
            ylabel="latency (us)",
            plot_name="reshape_and_cache-benchmark",
            args={},
        )
    )
    @torch.inference_mode()
    def benchmark(num_tokens, num_heads, head_size, block_size, num_blocks, provider, kv_cache_dtype="auto"):
        
        if kv_cache_dtype == "fp8" and head_size % 16:
            raise ValueError(
                "fp8 kv-cache requires head_size to be a multiple of 16.")

        torch.manual_seed(42)
        torch.set_default_device(device)

        key = torch.randn(num_tokens,
                      num_heads,
                      head_size,
                      dtype=dtype,
                      device=device)
        value = torch.randn_like(key)
        num_slots = block_size * num_blocks
        if num_tokens > num_slots:
            raise ValueError(
                "num_tokens cannot exceed the total number of cache slots")
        slot_mapping_lst = random.sample(range(num_slots), num_tokens)
        slot_mapping = torch.tensor(slot_mapping_lst,
                                dtype=torch.long,
                                device=device)

        num_layers = 1  # for simplicity, we use a single layer
        key_caches, value_caches = create_kv_caches_with_random(
            num_blocks,
            block_size,
            num_layers,
            num_heads,
            head_size,
            kv_cache_dtype,
            dtype,
            device=device,
        )
        key_cache, value_cache = key_caches[0], value_caches[0]

        # compute per-kernel scaling factors for fp8 conversion (if used).
        k_scale = (key.amax() / 64.0).to(torch.float32)
        v_scale = (value.amax() / 64.0).to(torch.float32)

        torch.xpu.synchronize()
        # Warm up
        for _ in range(5):
            if provider == "vllm":
                reshape_and_cache_vllm(
                    key, 
                    value, 
                    key_cache, 
                    value_cache, 
                    slot_mapping,
                    kv_cache_dtype, 
                    k_scale, 
                    v_scale,
                )
            elif provider == "ipex" and HAS_IPEX:
                reshape_and_cache_ipex(
                    key, 
                    value, 
                    key_cache, 
                    value_cache, 
                    slot_mapping,
                    kv_cache_dtype, 
                    k_scale, 
                    v_scale,
                )

        # Benchmark
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: {
                "vllm": reshape_and_cache_vllm,
                "ipex": reshape_and_cache_ipex
            }[provider](
                key, value, key_cache, value_cache, slot_mapping,
                kv_cache_dtype, k_scale, v_scale,
            ),
            quantiles=quantiles,
        )
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    args = parse_args()

    device = "xpu"

    print("Benchmark Configuration:")
    print(f"  Num Heads: {args.head_num_range}")
    print(f"  Head Size: {args.head_size}")
    print(f"  Block Size: {args.block_size}")
    print(f"  Num Blocks: {args.num_blocks}")
    print(f"  Data Type: {args.dtype}")
    print(f"  KV Cache Dtype: auto (IPEX & vLLM)")
    print(f"  Device: {device}")
    if HAS_IPEX:
        print(f"✅ IPEX {ipex.__version__} is available.")
    else:
        print("⚠️ IPEX not available. Only benchmarking vLLM.")

    num_token_range = [2**i for i in range(1, 12)]
    head_num_range = args.head_num_range
    head_size_range = [args.head_size]
    block_size_range = [args.block_size]
    num_blocks_range = [args.num_blocks]
    configs = list(
        itertools.product(num_token_range, head_num_range, head_size_range, block_size_range, num_blocks_range))
    
    benchmark = get_benchmark(
        dtype=args.dtype,
        device=device,
    )
    benchmark.run(print_data=True, save_path=None)