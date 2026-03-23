# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
import time
from argparse import ArgumentParser

import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
}


def simple_random_distribute(total_tokens: int, batch_size: int) -> torch.Tensor:
    distribution = torch.ones([batch_size], dtype=torch.int32)
    for _ in range(total_tokens - batch_size):
        selected_idx = random.randint(0, batch_size - 1)
        distribution[selected_idx] += 1
    return distribution


@torch.inference_mode()
def main(
    num_actual_tokens: int,
    batch_size: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    width: int,
    tp_size: int,
    has_bias: bool,
    activation: str,
    reorder_input: bool,
    mode: str,
    dtype: torch.dtype,
    cache_batch_size: int,
    seed: int,
    num_warmup_iters: int,
    num_iters: int,
) -> None:
    torch.set_default_device("xpu")
    random.seed(seed)
    torch.manual_seed(seed)

    if batch_size > num_actual_tokens:
        batch_size = num_actual_tokens

    if mode == "prefill":
        num_prefills = batch_size
    elif mode == "decode":
        if batch_size < num_actual_tokens:
            raise ValueError(
                "decode mode requires batch_size >= num_actual_tokens")
        num_prefills = 0
    else:
        num_prefills = random.randint(1, batch_size - 1) if batch_size > 1 else 1
    num_decodes = batch_size - num_prefills

    mixed_qkvz_size = num_k_heads // tp_size * (
        2 * head_k_dim + 2 * head_v_dim * num_v_heads // num_k_heads)
    mixed_ba_size = num_k_heads // tp_size * (2 * num_v_heads // num_k_heads)

    projected_states_qkvz = torch.randn((num_actual_tokens, mixed_qkvz_size),
                                        dtype=dtype)
    projected_states_ba = torch.randn((num_actual_tokens, mixed_ba_size),
                                      dtype=dtype)

    mixed_qkv_size = num_k_heads // tp_size * (
        2 * head_k_dim + head_v_dim * num_v_heads // num_k_heads)
    conv_state = torch.randn((cache_batch_size, width - 1, mixed_qkv_size),
                             dtype=dtype)
    ssm_state = torch.randn(
        (cache_batch_size, num_v_heads // tp_size, head_v_dim, head_k_dim),
        dtype=dtype,
    )

    conv_weights = torch.randn((mixed_qkv_size, width), dtype=dtype)
    conv_bias = torch.randn((mixed_qkv_size), dtype=dtype) if has_bias else None

    A_log = torch.randn((num_v_heads // tp_size), dtype=dtype)
    dt_bias = torch.randn((num_v_heads // tp_size), dtype=dtype)

    prefill_batches = simple_random_distribute(num_actual_tokens - num_decodes,
                                               batch_size - num_decodes)
    token_batches = torch.cat([torch.ones([num_decodes], dtype=torch.int32),
                               prefill_batches]).to("xpu")
    perm = torch.randperm(token_batches.size(0), device="xpu")
    shuffled_tensor = token_batches[perm]
    non_spec_query_start_loc = torch.cat([
        torch.zeros([1], device="xpu", dtype=torch.int32),
        torch.cumsum(shuffled_tensor, dim=0)
    ])
    has_initial_state = perm >= num_decodes
    non_spec_state_indices_tensor = torch.tensor(
        random.sample(range(cache_batch_size), batch_size),
        device="xpu",
        dtype=torch.int32,
    )

    core_attn_out = torch.zeros(
        (num_actual_tokens, num_v_heads // tp_size, head_v_dim),
        dtype=dtype,
        device="xpu",
    )
    z = torch.empty_like(core_attn_out)

    def run_kernel() -> None:
        torch.ops._xpu_C.gdn_attention(
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_state=conv_state,
            ssm_state=ssm_state,
            conv_weights=conv_weights,
            conv_bias=conv_bias,
            activation=activation,
            A_log=A_log,
            dt_bias=dt_bias,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            has_initial_state=has_initial_state,
            non_spec_query_start_loc=non_spec_query_start_loc,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            num_actual_tokens=num_actual_tokens,
            tp_size=tp_size,
            reorder_input=reorder_input,
        )

    print("Warming up...")
    for _ in range(num_warmup_iters):
        run_kernel()
    torch.xpu.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iters):
        run_kernel()
    torch.xpu.synchronize()
    end_time = time.perf_counter()

    avg_latency_s = (end_time - start_time) / num_iters
    print(f"Latency: {avg_latency_s * 1e6:.3f} us")
    print(f"Tokens/s: {num_actual_tokens / avg_latency_s:.2f}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark gdn attention kernel.")
    parser.add_argument("--num-actual-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-k-heads", type=int, default=16)
    parser.add_argument("--head-k-dim", type=int, default=128)
    parser.add_argument("--num-v-heads", type=int, default=32)
    parser.add_argument("--head-v-dim", type=int, default=128)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--has-bias", action="store_true")
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument("--reorder-input", action="store_true")
    parser.add_argument("--mode",
                        choices=["prefill", "decode", "mix_mode"],
                        default="prefill")
    parser.add_argument("--dtype",
                        type=str,
                        choices=["half", "bfloat16"],
                        default="half")
    parser.add_argument("--cache-batch-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=100)

    args = parser.parse_args()
    print(args)

    main(
        num_actual_tokens=args.num_actual_tokens,
        batch_size=args.batch_size,
        num_k_heads=args.num_k_heads,
        head_k_dim=args.head_k_dim,
        num_v_heads=args.num_v_heads,
        head_v_dim=args.head_v_dim,
        width=args.width,
        tp_size=args.tp_size,
        has_bias=args.has_bias,
        activation=args.activation,
        reorder_input=args.reorder_input,
        mode=args.mode,
        dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
        cache_batch_size=args.cache_batch_size,
        seed=args.seed,
        num_warmup_iters=args.num_warmup_iters,
        num_iters=args.num_iters,
    )
