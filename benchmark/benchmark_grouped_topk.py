# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from argparse import ArgumentParser

import torch

from tests.ops.grouped_topk import grouped_topk, grouped_topk_native

dpcpp_device = torch.device("xpu")

@torch.compile
def grouped_topk_compile(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert hidden_states.size(0) == gating_output.size(0), (
        "Number of tokens mismatch")
    if scoring_func == "softmax":
        gating_output = gating_output.to(torch.float32)
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.shape[0]
    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        e_score_correction_bias = e_score_correction_bias.to(torch.float32)
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (scores.view(num_token, num_expert_group,
                                    -1).topk(2, dim=-1)[0].sum(dim=-1))
    else:
        group_scores = (scores.view(num_token, num_expert_group,
                                    -1).max(dim=-1).values)  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=True)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.shape[-1] // num_expert_group).reshape(num_token, -1))  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(),
                                    float("-inf"))  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=True)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores,
                                            k=topk,
                                            dim=-1,
                                            sorted=True)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

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
    provider: str = "vllm",
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

        if provider == "vllm":
            run_op = grouped_topk
        elif provider == "native":
            run_op = grouped_topk_native
        elif provider == "compile":
            run_op = grouped_topk_compile
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        for _ in range(num_iters):
            topk_weights, topk_indices = run_op(
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
    parser = ArgumentParser(description="Benchmark the grouped topk kernel.")
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
    parser.add_argument("--provider",
                        type=str,
                        choices=["vllm", "native", "compile"],
                        default="vllm")

    args = parser.parse_args()
    print(args)

    # Convert dtype string to torch.dtype
    dtype = getattr(torch, args.dtype)
    main(
        dtype=dtype,
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
        provider=args.provider,
    )
