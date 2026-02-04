# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import torch
import triton

from tests.ops.moe_align_block_size_ops import (batched_moe_align_block_size,
                                                moe_align_block_size)
from tests.utils import parse_args, opcheck, round_up, seed_everything
from tests.test_moe_align_block_size import torch_moe_align_block_size, _verify_expert_level_sorting

seed_everything(0)
DEVICE = "xpu"

def moe_align_block_size_native(topk_ids, num_experts, block_size,
                              pad_sorted_ids):
    sorted_ids, expert_ids, num_tokens = torch_moe_align_block_size(
        topk_ids=topk_ids,
        block_size=block_size,
        num_experts=num_experts,
        pad_sorted_ids=pad_sorted_ids,
    )
    return sorted_ids, expert_ids, num_tokens


def moe_align_block_size_vllm(topk_ids, num_experts, block_size,
                              pad_sorted_ids):
    sorted_ids, expert_ids, num_tokens = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=block_size,
        num_experts=num_experts,
        pad_sorted_ids=pad_sorted_ids,
    )
    return sorted_ids, expert_ids, num_tokens


def moe_align_block_size_with_expert_map_native(topk_ids, num_experts, block_size, expert_map):
    sorted_ids, expert_ids, num_tokens = torch_moe_align_block_size(
        topk_ids=topk_ids,
        block_size=block_size,
        num_experts=num_experts,
        expert_map=expert_map
    )
    return sorted_ids, expert_ids, num_tokens


def moe_align_block_size_with_expert_map_vllm(topk_ids, num_experts, block_size, expert_map):
    sorted_ids, expert_ids, num_tokens = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=block_size,
        num_experts=num_experts,
        expert_map=expert_map,
    )
    return sorted_ids, expert_ids, num_tokens


def calculate_diff(config):
    m, topk, num_experts, block_size, pad_sorted_ids = config
    topk_ids = torch.zeros((m, topk), device=DEVICE, dtype=torch.int32)
    for i in range(m):
        experts = torch.randperm(num_experts, device=DEVICE)[:topk]
        topk_ids[i] = experts

    sorted_ids_native, expert_ids_native, num_tokens_native = \
        moe_align_block_size_native(topk_ids.clone(), num_experts,
                                  block_size, pad_sorted_ids)
    sorted_ids_vllm, expert_ids_vllm, num_tokens_vllm = \
        moe_align_block_size_vllm(topk_ids.clone(), num_experts,
                                 block_size, pad_sorted_ids)

    if torch.testing.assert_close(num_tokens_native, num_tokens_vllm, atol=0, rtol=0) and \
        torch.testing.assert_close(expert_ids_native, expert_ids_vllm, atol=0, rtol=0):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")

    _verify_expert_level_sorting(
        sorted_ids_vllm,
        sorted_ids_native,
        expert_ids_vllm,
        block_size,
        num_tokens_vllm.item(),
        m * topk,
    )

    total_tokens = m * topk
    assert num_tokens_vllm.item() % block_size == 0, (
        "num_tokens_post_pad should be divisible by block_size")
    assert num_tokens_vllm.item() >= total_tokens, (
        "num_tokens_post_pad should be at least total_tokens")
    valid_tokens = sorted_ids_vllm[sorted_ids_vllm < total_tokens]
    assert len(valid_tokens) == total_tokens, (
        f"Should have exactly {total_tokens} valid tokens,"
        f" got {len(valid_tokens)}")
    assert (expert_ids_vllm
            >= 0).all() and (expert_ids_vllm < num_experts).all(), (
                "expert_ids should contain valid expert indices")


def calculate_diff_with_expert_map(config):
    m, topk, num_experts, block_size = config
    topk_ids = torch.zeros((m, topk), device=DEVICE, dtype=torch.int32)
    for i in range(m):
        experts = torch.randperm(num_experts, device=DEVICE)[:topk]
        topk_ids[i] = experts

    expert_map = torch.full((num_experts, ),
                            -1,
                            device=DEVICE,
                            dtype=torch.int32)
    local_experts = list(range(0, num_experts, 2))
    for i, expert_id in enumerate(local_experts):
        expert_map[expert_id] = i

    sorted_ids_native, expert_ids_native, num_tokens_native = \
        moe_align_block_size_with_expert_map_native(topk_ids.clone(), num_experts,
                                  block_size, expert_map.clone())
    sorted_ids_vllm, expert_ids_vllm, num_tokens_vllm = \
        moe_align_block_size_with_expert_map_vllm(topk_ids.clone(), num_experts,
                                 block_size, expert_map.clone())
    if torch.testing.assert_close(num_tokens_native, num_tokens_vllm, atol=0, rtol=0) and \
        torch.testing.assert_close(expert_ids_native, expert_ids_vllm, atol=0, rtol=0):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")

    _verify_expert_level_sorting(
        sorted_ids_vllm,
        sorted_ids_native,
        expert_ids_vllm,
        block_size,
        num_tokens_vllm.item(),
        m * topk,
    )


def moe_align_block_size_deterministic():
    m = 128
    topk = 2
    num_experts = 32
    block_size = 64
    torch.manual_seed(42)
    topk_ids = torch.randint(0,
                             num_experts, (m, topk),
                             device=DEVICE,
                             dtype=torch.int32)

    # expect the results to be reproducible
    results = []
    for _ in range(5):
        sorted_ids, expert_ids, num_tokens = moe_align_block_size(
            topk_ids=topk_ids, block_size=block_size, num_experts=num_experts)
        results.append(
            (sorted_ids.clone(), expert_ids.clone(), num_tokens.clone()))

    for i in range(1, len(results)):
        assert torch.equal(
            results[0][0],
            results[i][0]), ("sorted_ids should be deterministic")
        assert torch.equal(
            results[0][1],
            results[i][1]), ("expert_ids should be deterministic")
        assert torch.equal(
            results[0][2],
            results[i][2]), ("num_tokens should be deterministic")


def test_batched_moe_align_block_size(config):
    max_tokens_per_batch, num_experts, block_size, simulate_empty_batches = config
    def ref_outputs(
        expert_num_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        E = expert_num_tokens.size(0)

        # Round up so each batch can be split to blocks evenly.
        Msum = round_up(max_tokens_per_batch, block_size) * E
        ref_sorted_ids = torch.empty((Msum, ), dtype=torch.int32)
        ref_expert_ids = torch.empty((Msum // block_size, ), dtype=torch.int32)
        ref_num_tokens_post_pad = torch.empty((1, ), dtype=torch.int32)

        # Initialize
        sentinel = E * max_tokens_per_batch
        ref_sorted_ids.fill_(sentinel)
        ref_expert_ids.fill_(-1)

        # Fill ref_sorted_ids
        i = 0
        for expert_id, expert_nt in enumerate(expert_num_tokens):
            token_offset = expert_id * max_tokens_per_batch
            for j in range(expert_nt):
                ref_sorted_ids[i] = token_offset + j
                i += 1
            # round up i to the next block_size
            i = round_up(i, block_size)

        ref_num_tokens_post_pad[0] = i

        # Fill expert_ids
        nt_ceil_sum = 0
        for expert_id, expert_nt in enumerate(expert_num_tokens):
            expert_ids_offset = nt_ceil_sum // block_size
            ceil_expert_nt = round_up(int(expert_nt.item()), block_size)
            num_blocks = ceil_expert_nt // block_size
            for x in range(num_blocks):
                ref_expert_ids[expert_ids_offset + x] = expert_id
            nt_ceil_sum += ceil_expert_nt

        return (
            ref_sorted_ids.to(DEVICE),
            ref_expert_ids.to(DEVICE),
            ref_num_tokens_post_pad.to(DEVICE),
        )

    # Compute expert_num_tokens
    expert_num_tokens = torch.randint(
        low=0,
        high=max_tokens_per_batch,
        size=(num_experts, ),
        device="cpu",
        dtype=torch.int32,
    )
    if simulate_empty_batches:
        # mark half the batches to have 0 tokens
        zero_batches = torch.randperm(num_experts)[:num_experts // 2]
        expert_num_tokens[zero_batches] = 0

    # ref outputs
    ref_sorted_ids, ref_expert_ids, ref_num_tokens_post_pad = ref_outputs(
        expert_num_tokens.clone()
    )

    # outputs
    sorted_ids, expert_ids, num_tokens_post_pad = batched_moe_align_block_size(
        max_tokens_per_batch, block_size, expert_num_tokens.clone().to(DEVICE))

    assert ref_sorted_ids.size() == sorted_ids.size(), (
        f"{ref_sorted_ids.size()} vs {sorted_ids.size()}")
    assert ref_expert_ids.size() == expert_ids.size(), (
        f"{ref_expert_ids.size()} vs {expert_ids.size()}")
    assert ref_num_tokens_post_pad.size() == num_tokens_post_pad.size(), (
        f"{ref_num_tokens_post_pad.size()} vs {num_tokens_post_pad.size()}")
    
    if torch.testing.assert_close(ref_sorted_ids, sorted_ids, atol=0, rtol=0) and \
            torch.testing.assert_close(ref_expert_ids, expert_ids, atol=0, rtol=0) and \
            torch.testing.assert_close(ref_num_tokens_post_pad, num_tokens_post_pad, atol=0, rtol=0):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


def moe_align_block_size_opcheck():
    num_experts = 4
    block_size = 4
    topk_ids = torch.randint(0,
                             num_experts, (3, 4),
                             dtype=torch.int32,
                             device=DEVICE)

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)

    opcheck(
        torch.ops._moe_C.moe_align_block_size,
        (
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
        ),
    )


def batched_moe_align_block_size_opcheck():
    max_tokens_per_batch = 512
    num_experts = 4
    block_size = 16

    expert_num_tokens = torch.randint(
        low=0,
        high=max_tokens_per_batch,
        size=(num_experts, ),
        dtype=torch.int32,
        device=DEVICE,
    )

    max_num_tokens_padded = num_experts * max(max_tokens_per_batch, block_size)
    sorted_ids = torch.empty((max_num_tokens_padded, ),
                             dtype=torch.int32,
                             device=DEVICE)

    assert max_num_tokens_padded % block_size == 0
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks, ),
                             dtype=torch.int32,
                             device=DEVICE)

    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=DEVICE)

    opcheck(
        torch.ops._moe_C.batched_moe_align_block_size,
        (
            max_tokens_per_batch,
            block_size,
            expert_num_tokens,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
        ),
    )


def get_benchmark(configs):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "topk", "num_experts", "block_size", "pad_sorted_ids"],
            x_vals=[tuple(i) for i in configs],
            line_arg="provider",
            line_vals=["native", "vllm"],
            line_names=["Native", "vLLM"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="us",
            plot_name="moe-align-block-size-perf",
            args={},
        )
    )
    def benchmark(m, topk, num_experts, block_size, pad_sorted_ids, provider):
        topk_ids = torch.zeros((m, topk), device=DEVICE, dtype=torch.int32)
        for i in range(m):
            experts = torch.randperm(num_experts, device=DEVICE)[:topk]
            topk_ids[i] = experts

        quantiles = [0.5, 0.2, 0.8]

        if provider == "native":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch_moe_align_block_size(
                    topk_ids=topk_ids.clone(),
                    block_size=block_size,
                    num_experts=num_experts,
                    pad_sorted_ids=pad_sorted_ids,
                ),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: moe_align_block_size(
                    topk_ids=topk_ids.clone(),
                    block_size=block_size,
                    num_experts=num_experts,
                    pad_sorted_ids=pad_sorted_ids,
                ),
                quantiles=quantiles,
            )
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


def get_benchmark_with_expert_map(configs):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "topk", "num_experts", "block_size"],
            x_vals=[tuple(i) for i in configs],
            line_arg="provider",
            line_vals=["native", "vllm"],
            line_names=["Native", "vLLM"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="us",
            plot_name="moe-align-block-size_with-expert-map-perf",
            args={},
        )
    )
    def benchmark(m, topk, num_experts, block_size, provider):
        topk_ids = torch.zeros((m, topk), device=DEVICE, dtype=torch.int32)
        for i in range(m):
            experts = torch.randperm(num_experts, device=DEVICE)[:topk]
            topk_ids[i] = experts

        expert_map = torch.full((num_experts, ),
                                -1,
                                device=DEVICE,
                                dtype=torch.int32)
        local_experts = list(range(0, num_experts, 2))
        for i, expert_id in enumerate(local_experts):
            expert_map[expert_id] = i

        quantiles = [0.5, 0.2, 0.8]

        if provider == "native":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch_moe_align_block_size(
                    topk_ids=topk_ids,
                    block_size=block_size,
                    num_experts=num_experts,
                    expert_map=expert_map
                ),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: moe_align_block_size(
                    topk_ids=topk_ids,
                    block_size=block_size,
                    num_experts=num_experts,
                    expert_map=expert_map
                ),
                quantiles=quantiles,
            )
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


def get_benchmark_batched_moe_align_block_size(configs):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["max_tokens_per_batch", "num_experts", "block_size", "simulate_empty_batches"],
            x_vals=[tuple(i) for i in configs],
            line_arg="provider",
            line_vals=["native", "vllm"],
            line_names=["Native", "vLLM"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="us",
            plot_name="batched_moe-align-block-size-perf",
            args={},
        )
    )
    def benchmark(max_tokens_per_batch, num_experts, block_size, simulate_empty_batches, provider):
        def ref_outputs(
            expert_num_tokens: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            E = expert_num_tokens.size(0)

            # Round up so each batch can be split to blocks evenly.
            Msum = round_up(max_tokens_per_batch, block_size) * E
            ref_sorted_ids = torch.empty((Msum, ), dtype=torch.int32)
            ref_expert_ids = torch.empty((Msum // block_size, ), dtype=torch.int32)
            ref_num_tokens_post_pad = torch.empty((1, ), dtype=torch.int32)

            # Initialize
            sentinel = E * max_tokens_per_batch
            ref_sorted_ids.fill_(sentinel)
            ref_expert_ids.fill_(-1)

            # Fill ref_sorted_ids
            i = 0
            for expert_id, expert_nt in enumerate(expert_num_tokens):
                token_offset = expert_id * max_tokens_per_batch
                for j in range(expert_nt):
                    ref_sorted_ids[i] = token_offset + j
                    i += 1
                # round up i to the next block_size
                i = round_up(i, block_size)

            ref_num_tokens_post_pad[0] = i

            # Fill expert_ids
            nt_ceil_sum = 0
            for expert_id, expert_nt in enumerate(expert_num_tokens):
                expert_ids_offset = nt_ceil_sum // block_size
                ceil_expert_nt = round_up(int(expert_nt.item()), block_size)
                num_blocks = ceil_expert_nt // block_size
                for x in range(num_blocks):
                    ref_expert_ids[expert_ids_offset + x] = expert_id
                nt_ceil_sum += ceil_expert_nt

            return (
                ref_sorted_ids.to(DEVICE),
                ref_expert_ids.to(DEVICE),
                ref_num_tokens_post_pad.to(DEVICE),
            )

        # Compute expert_num_tokens
        expert_num_tokens = torch.randint(
            low=0,
            high=max_tokens_per_batch,
            size=(num_experts, ),
            device="cpu",
            dtype=torch.int32,
        )
        if simulate_empty_batches:
            # mark half the batches to have 0 tokens
            zero_batches = torch.randperm(num_experts)[:num_experts // 2]
            expert_num_tokens[zero_batches] = 0

        quantiles = [0.5, 0.2, 0.8]

        if provider == "native":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: ref_outputs(
                    expert_num_tokens
                ),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: batched_moe_align_block_size(
                    max_tokens_per_batch, block_size, expert_num_tokens.to(DEVICE)
                ),
                quantiles=quantiles,
            )
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


def sanity_check():
    m = [1, 3, 256, 2256, 4096]
    topk = [1, 2, 16, 32]
    num_experts = [32, 160, 256, 257]
    block_size = [32, 128]
    pad_sorted_ids = [False, True]
    print("Final configuration for sanity check:")
    print(f"  m: {m}")
    print(f"  topk: {topk}")
    print(f"  num_experts: {num_experts}")
    print(f"  block_size: {block_size}")
    print(f"  pad_sorted_ids: {pad_sorted_ids}")

    configs = list(
        itertools.product(m, topk, num_experts, block_size, pad_sorted_ids))
    # fp8_numerical_sanity_check
    for config in configs:
        calculate_diff(config)

    benchmark = get_benchmark(configs)
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)


def sanity_check_with_expert_map():
    m = [16, 32]
    topk = [2, 4]
    num_experts = [8]
    block_size = [64]
    print("Final configuration for expert_map sanity check:")
    print(f"  m: {m}")
    print(f"  topk: {topk}")
    print(f"  num_experts: {num_experts}")
    print(f"  block_size: {block_size}")
    configs = list(
        itertools.product(m, topk, num_experts, block_size))
    for config in configs:
        calculate_diff_with_expert_map(config)

    benchmark = get_benchmark_with_expert_map(configs)
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)


def sanity_check_batched_moe_align_block_size():
    max_tokens_per_batch = [13, 16, 512]
    num_experts = [8, 16, 32, 64]
    block_size = [8, 16, 32, 64]
    simulate_empty_batches = [False, True]
    print("Final configuration for batched_moe_align_block_size sanity check:")
    print(f"  max_tokens_per_batch: {max_tokens_per_batch}")
    print(f"  num_experts: {num_experts}")
    print(f"  block_size: {block_size}")
    print(f"  simulate_empty_batches: {simulate_empty_batches}")

    configs = list(
        itertools.product(max_tokens_per_batch,
                          num_experts,
                          block_size,
                          simulate_empty_batches))
    # batched_moe_align_block_size_sanity_check
    for config in configs:
        test_batched_moe_align_block_size(config)

    benchmark = get_benchmark_batched_moe_align_block_size(configs)
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)


if __name__ == "__main__":

    args = parse_args()
    seed = 1234
    torch.manual_seed(seed)

    sanity_check()
    sanity_check_with_expert_map()
    moe_align_block_size_deterministic()
    sanity_check_batched_moe_align_block_size()
    moe_align_block_size_opcheck()
    batched_moe_align_block_size_opcheck()
