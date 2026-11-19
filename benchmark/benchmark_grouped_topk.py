# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

import itertools
from argparse import ArgumentParser
from typing import Callable, Optional

import torch
from utils import bootstrap_benchmark_env

bootstrap_benchmark_env(__file__)

from tests.ops.grouped_topk_op import fused_grouped_topk, grouped_topk

# isort: on


@torch.compile
def grouped_topk_compile(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    assert hidden_states.size(0) == gating_output.size(0), (
        "Number of tokens mismatch")
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")
    num_token = scores.size(0)
    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (scores.view(num_token, num_expert_group,
                                    -1).topk(2, dim=-1)[0].sum(dim=-1))
    else:
        group_scores = scores.view(num_token, num_expert_group,
                                   -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.size(-1) // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(),
                                    float("-inf"))  # [n, e]
    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores,
                                            k=topk,
                                            dim=-1,
                                            sorted=False)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


n_token_range = [1, 64, 256]
# Each entry: (n_expert, num_expert_group, topk_group, topk). The three groups
# below deliberately target the three kernel dispatch paths:
#   * single-group  (n_group==1, topk_group==1) -> single_group_topk fast path
#   * multi-group    experts_per_group<=32       -> small_expert_count kernel
#   * fallback       experts_per_group>32         -> WarpSelect kernel
shape_range = [
    # single_group_topk tiers (path 1)
    (128, 1, 1, 8),
    (256, 1, 1, 8),
    (288, 1, 1, 8),
    (512, 1, 1, 8),
    # grouped small-expert-count (path 2, DeepSeek-style)
    (64, 8, 4, 4),
    (256, 8, 4, 8),
    # WarpSelect fallback (path 3, experts_per_group == 64)
    (512, 8, 2, 8),
]
scoring_func_range = ["sigmoid", "softmax"]
dtype_range = [torch.float16, torch.bfloat16, torch.float32]
configs = list(
    itertools.product(
        n_token_range,
        shape_range,
        scoring_func_range,
        dtype_range,
    ))
# Flatten the (n_expert, num_expert_group, topk_group, topk) tuple into the
# config so it can be consumed by triton's perf_report x_vals.
configs = [(n_token, *shape, scoring_func, dtype)
           for (n_token, shape, scoring_func, dtype) in configs]



DEVICE = "xpu"


# --------------------------------------------------------------------------
# Timing and cache state (copied from benchmark_moe.py)
# --------------------------------------------------------------------------
def llc_bytes() -> int:
    props = torch.xpu.get_device_properties(0)
    for attr in ("last_level_cache_size", "global_mem_cache_size",
                 "l2_cache_size"):
        size = getattr(props, attr, 0) or 0
        if size:
            return int(size)
    # Falling back to 0 collapses the scratch buffer and the flush silently
    # stops evicting anything at all, which looks like a 2x speedup.
    return 32 << 20


def make_flush(mode: str, scratch_bytes: int, dirty_bytes: int):
    """Put the LLC into the state a real forward would leave it in.

    Eviction and dirtiness are independent, and conflating them is worth about
    2x on this part. A pure write pass evicts everything *and* leaves 100% of
    the victims dirty, so the timed kernel's read has to share DRAM with an
    LLC-sized write-back nobody asked for. A decode step instead arrives
    evicted-but-clean, while prefill arrives fully dirty.

    Hence the modes: read evicts clean, write evicts 100% dirty, and mix
    evicts clean then dirties `dirty_bytes` of it.
    """
    if mode == "none":
        return None
    scratch = torch.empty(scratch_bytes, dtype=torch.uint8, device=DEVICE)

    if mode == "write":

        def flush(i):
            scratch.fill_(i & 0xFF)
    elif mode == "read":

        def flush(i):
            scratch.sum()
    else:

        def flush(i):
            scratch.sum()
            # Written last, so these are the lines still resident and dirty
            # when the kernel starts; the rest of the LLC holds the clean
            # lines the read pass just brought in.
            scratch[:dirty_bytes].fill_(i & 0xFF)

    return flush


def time_op(fn, warmup: int, iters: int, flush) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    starts = [torch.xpu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.xpu.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if flush is not None:
            # Untimed, but on the same queue, so it is ordered before the
            # kernel and has evicted the weights by the time the start event
            # fires.
            flush(i)
        starts[i].record()
        fn()
        ends[i].record()
    torch.xpu.synchronize()
    # elapsed_time is in ms; return microseconds.
    return sorted(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, \
            max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def make_fn(
    provider: str,
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str,
    routed_scaling_factor: float,
    e_score_correction_bias: torch.Tensor,
) -> Callable[[], object]:
    impl = {
        "vllm": fused_grouped_topk,
        "native": grouped_topk,
        "compile": grouped_topk_compile,
    }[provider]

    def fn():
        return impl(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
        )

    return fn


def main():
    parser = ArgumentParser(description="Benchmark the grouped topk kernel.")
    parser.add_argument(
        "--flush",
        default="mix",
        choices=("mix", "read", "write", "none"),
        help="LLC state the timed kernel starts from. mix: evict clean, then "
        "leave --dirty-mb dirty (decode model, default). read: evict clean. "
        "write: evict 100%% dirty (prefill model). none: leave resident.")
    parser.add_argument(
        "--dirty-mb",
        type=float,
        default=0.2,
        help="MB of the LLC that --flush mix leaves dirty.")
    parser.add_argument(
        "--flush-mb",
        type=int,
        default=0,
        help="scratch buffer size; 0 uses 2x the device cache")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["vllm", "native", "compile"],
        choices=["vllm", "native", "compile"],
    )
    args = parser.parse_args()

    cache = llc_bytes()
    scratch_bytes = (args.flush_mb << 20) if args.flush_mb else 2 * cache
    dirty_bytes = min(int(args.dirty_mb * (1 << 20)), cache, scratch_bytes)
    flush = make_flush(args.flush, scratch_bytes, dirty_bytes)

    print(f"device {torch.xpu.get_device_name(0)}  LLC {cache >> 20} MiB  "
          f"flush={args.flush} scratch {scratch_bytes >> 20} MiB "
          f"dirty {dirty_bytes >> 10} KiB  "
          f"warmup {args.warmup} iters {args.iters}")
    header = (f"{'n_tok':>5} {'n_exp':>6} {'grp':>3} {'tkg':>3} {'tk':>3} "
              f"{'score':>7} {'dtype':>9} {'prov':>7} "
              f"{'p50_us':>9} {'p10_us':>9} {'p90_us':>9}")
    print(header)
    print("-" * len(header))

    n_hidden = 1024
    routed_scaling_factor = 1.0
    renormalize = True

    for (n_token, n_expert, num_expert_group, topk_group, topk, scoring_func,
         dtype) in configs:
        hidden_states = torch.randn((n_token, n_hidden),
                                    dtype=dtype,
                                    device=DEVICE)
        gating_output = torch.randn((n_token, n_expert),
                                    dtype=dtype,
                                    device=DEVICE)
        e_score_correction_bias = torch.randn((n_expert, ),
                                              dtype=torch.float32,
                                              device=DEVICE)

        for provider in args.providers:
            fn = make_fn(
                provider,
                hidden_states,
                gating_output,
                topk,
                renormalize,
                num_expert_group,
                topk_group,
                scoring_func,
                routed_scaling_factor,
                e_score_correction_bias,
            )
            lat = time_op(fn, args.warmup, args.iters, flush)
            p50 = percentile(lat, 0.5)
            p10 = percentile(lat, 0.1)
            p90 = percentile(lat, 0.9)
            print(f"{n_token:>5} {n_expert:>6} {num_expert_group:>3} "
                  f"{topk_group:>3} {topk:>3} {scoring_func:>7} "
                  f"{str(dtype).replace('torch.', ''):>9} {provider:>7} "
                  f"{p50:>9.3f} {p10:>9.3f} {p90:>9.3f}")


if __name__ == "__main__":
    main()


