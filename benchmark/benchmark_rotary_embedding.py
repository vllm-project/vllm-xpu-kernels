# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import torch
import triton

from tests.utils import parse_args
from tests.ops.rotary_embedding_op import RotaryEmbedding
from tests.test_rotary_embedding import rotary_embedding_opcheck


def check_correctness(config):
    max_position, is_neox_style, rotary_dim, head_size, seq_len, use_key, head_stride_is_contiguous = config
    batch_size = 1
    base = 10000
    num_heads = 7
    device = torch.device("xpu")
    rot_native = RotaryEmbedding(head_size, rotary_dim, max_position, base,
                          is_neox_style, torch.float32)
    rot_vllm = RotaryEmbedding(head_size, rotary_dim, max_position, base,
                          is_neox_style, torch.float32)

    positions = torch.randint(0,
                              max_position, (batch_size, seq_len),
                              device=device)
    head_stride = head_size + (64 if head_stride_is_contiguous else 0)

    query = torch.randn(batch_size,
                        seq_len,
                        num_heads,
                        head_stride,
                        dtype=torch.float32,
                        device=device)
    key = torch.randn_like(query) if use_key else None
    query = query[..., :head_size]
    key = key[..., :head_size] if use_key else None

    rotary_embedding_opcheck(rot_native, positions.clone(), query.clone(), key.clone() if use_key else None)
    rotary_embedding_opcheck(rot_vllm, positions.clone(), query.clone(), key.clone() if use_key else None)

    # if we have a contiguous head stride, test the alternate
    # [..., num_heads * head_dim] shape/layout
    if head_stride_is_contiguous:
        rotary_embedding_opcheck(
            rot_native, positions.clone(), query.flatten(start_dim=-2).clone(),
            key.flatten(start_dim=-2).clone() if use_key else None)
        rotary_embedding_opcheck(
            rot_vllm, positions.clone(), query.flatten(start_dim=-2).clone(),
            key.flatten(start_dim=-2).clone() if use_key else None)
    print(f"Correctness test passed for config: {config}")


def get_benchmark():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["max_position", "is_neox_style", "rotary_dim",
                     "head_size", "seq_len", "use_key", "head_stride_is_contiguous"],
            x_vals=[tuple(i) for i in configs],
            line_arg="provider",
            line_vals=["native", "vllm"],
            line_names=["Native", "vLLM"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="us",
            plot_name="rotary-embedding-perf",
            args={},
        )
    )
    def benchmark(max_position, is_neox_style, rotary_dim, head_size, seq_len, use_key, head_stride_is_contiguous, provider):
        quantiles = [0.5, 0.2, 0.8]
        batch_size = 1
        base = 10000
        num_heads = 7

        # batched_rotary_embedding is not implemented yet.
        offsets = None

        device = torch.device("xpu")
        rot_native = RotaryEmbedding(head_size, rotary_dim, max_position, base,
                            is_neox_style, torch.float32)
        rot_vllm = RotaryEmbedding(head_size, rotary_dim, max_position, base,
                            is_neox_style, torch.float32)
        
        positions = torch.randint(0,
                              max_position, (batch_size, seq_len),
                              device=device)
        head_stride = head_size + (64 if head_stride_is_contiguous else 0)

        query = torch.randn(batch_size,
                            seq_len,
                            num_heads,
                            head_stride,
                            dtype=torch.float32,
                            device=device)
        key = torch.randn_like(query) if use_key else None
        query = query[..., :head_size]
        key = key[..., :head_size] if use_key else None
        print(f"Running config: {(max_position, is_neox_style, rotary_dim, head_size, seq_len, use_key, head_stride_is_contiguous)}, Provider: {provider}", flush=True)

        if provider == "native":
            positions_clone = positions.clone()
            query_clone = query.clone()
            key_clone = key.clone() if use_key else None
            rot_native.cos_sin_cache = rot_native.cos_sin_cache.to(device=query.device, dtype=query.dtype)
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rot_native.forward_native(
                    positions_clone, 
                    query_clone, 
                    key_clone, 
                    offsets
                ),
                quantiles=quantiles,
            )
        else:
            positions_clone = positions.clone()
            query_clone = query.clone()
            key_clone = key.clone() if use_key else None
            rot_vllm.cos_sin_cache = rot_vllm.cos_sin_cache.to(device=query.device, dtype=query.dtype)
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rot_vllm.forward_xpu(
                    positions_clone, 
                    query_clone, 
                    key_clone, 
                    offsets
                ),
                quantiles=quantiles,
            )
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":

    args = parse_args()
    seed = 1234
    torch.manual_seed(seed)

    max_position = [11, 4096, 32768]
    is_neox_style = [True, False]
    rotary_dim = [32]
    head_size = [32, 108]
    seq_len = [11, 1024]
    use_key = [True, False]
    head_stride_is_contiguous = [True, False]
    print("Final configuration:")
    print(f"max_position={max_position}")
    print(f"is_neox_style={is_neox_style}")
    print(f"rotary_dim={rotary_dim}")
    print(f"head_size={head_size}")
    print(f"seq_len={seq_len}")
    print(f"use_key={use_key}")
    print(f"head_stride_is_contiguous={head_stride_is_contiguous}")

    configs = list(
        itertools.product(max_position, is_neox_style, rotary_dim, head_size, seq_len, use_key, head_stride_is_contiguous))
    # Run correctness test
    for config in configs:
        check_correctness(config)

    benchmark = get_benchmark()
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)
