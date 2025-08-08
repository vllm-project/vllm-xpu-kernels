# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from argparse import ArgumentParser

import torch

from tests.ops.grouped_topk import grouped_topk

dpcpp_device = torch.device("xpu")


@torch.inference_mode()
def main(
    dtype: torch.dtype,
    num_tokens: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str = "softmax",
    has_bias: bool = False,
    seed: int = 0,
    num_warmup_iters: int = 5,
    num_iters: int = 100,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device("xpu")

    gating_output = torch.randn(num_tokens, num_experts,
                                device=dpcpp_device).to(dtype)
    hidden_states = torch.zeros(num_tokens, num_experts,
                                device=dpcpp_device).to(dtype)
    bias = None
    if has_bias:
        if has_bias and scoring_func == "sigmoid" \
            and dtype is not torch.float32:
            # using a bias of bigger number to avoid Low-precision
            bias = torch.arange(1, num_experts + 1).to(dpcpp_device).to(dtype)
        else:
            bias = torch.randn(num_experts, device=dpcpp_device).to(dtype)

    def run_xpu_benchmark(num_iters: int) -> float:
        torch.xpu.synchronize()

        start_time = time.perf_counter()

        for _ in range(num_iters):
            topk_weights, topk_indices = grouped_topk(
                hidden_states,
                gating_output,
                topk,
                renormalize,
                num_expert_group,
                topk_group,
                scoring_func=scoring_func,
                e_score_correction_bias=bias,
            )
        torch.xpu.synchronize()

        end_time = time.perf_counter()

        return (end_time - start_time) / num_iters

    # Warmup.
    print("Warming up...")
    run_benchmark = run_xpu_benchmark
    run_benchmark(num_iters=num_warmup_iters)

    # Benchmark.
    latency = run_benchmark(num_iters=num_iters)
    print(f"Kernel running time: {latency * 1000000:.3f} us")


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the layernorm kernel.")
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--renormalize", action="store_true")
    parser.add_argument("--num-expert-group", type=int, default=8)
    parser.add_argument("--topk-group", type=int, default=8)
    parser.add_argument("--scoring-func",
                        type=str,
                        choices=["sigmoid", "softmax"],
                        default="softmax")
    parser.add_argument("--has-bias", action="store_true")
    parser.add_argument("--dtype",
                        type=str,
                        choices=["half", "bfloat16", "float"],
                        default="half")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-warmup-iters", type=int, default=5)
    parser.add_argument("--num-iters",
                        type=int,
                        default=100,
                        help="Number of benchmark iterations. ")

    args = parser.parse_args()
    print(args)

    main(
        dtype=args.dtype,
        num_tokens=args.num_tokens,
        num_experts=args.num_experts,
        topk=args.topk,
        renormalize=args.renormalize,
        num_expert_group=args.num_expert_group,
        topk_group=args.topk_group,
        scoring_func=args.scoring_func,
        has_bias=args.has_bias,
        seed=args.seed,
        num_warmup_iters=args.num_warmup_iters,
        num_iters=args.num_iters,
    )
