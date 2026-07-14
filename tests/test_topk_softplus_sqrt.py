# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import pathlib
import shutil
import time
from argparse import ArgumentParser
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

import tests.register_ops as ops
from tests.utils import format_tc, seed_everything


def _torch_topk_softplus_sqrt(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    e_score_correction_bias: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    hash_indices_table: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = F.softplus(gating_output.float()).sqrt()
    original_scores = scores

    if hash_indices_table is not None:
        assert input_ids is not None
        topk_ids = hash_indices_table[input_ids.long()]
    else:
        if e_score_correction_bias is not None:
            scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
        else:
            scores_for_choice = scores
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=True)[1]

    topk_weights = original_scores.gather(1, topk_ids.long())
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def fused_topk_softplus_sqrt(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    correction_bias: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    hash_indices_table: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = gating_output.size(0)

    topk_weights = torch.empty(
        num_tokens,
        topk,
        dtype=torch.float32,
        device=gating_output.device,
    )
    topk_ids = torch.empty(
        num_tokens,
        topk,
        dtype=torch.int32,
        device=gating_output.device,
    )
    token_expert_indices = torch.empty(
        num_tokens,
        topk,
        dtype=torch.int32,
        device=gating_output.device,
    )

    ops.topk_softplus_sqrt(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        routed_scaling_factor,
        correction_bias,
        input_ids,
        hash_indices_table,
    )

    return topk_weights, topk_ids


def benchmark_fused_topk_softplus_sqrt(
    num_tokens: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    dtype: torch.dtype,
    use_bias: bool = True,
    use_hash: bool = False,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
) -> float:
    seed_everything(0)
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=dtype, device="xpu"
    )
    correction_bias = None
    input_ids = None
    hash_indices_table = None

    if use_bias:
        correction_bias = torch.randn(
            (num_experts,), dtype=torch.float32, device="xpu"
        )

    if use_hash:
        vocab_size = 1024
        hash_indices_table = torch.stack(
            [torch.randperm(num_experts)[:topk] for _ in range(vocab_size)]
        ).to(device="xpu", dtype=torch.int32)
        input_ids = torch.randint(
            0,
            vocab_size,
            (num_tokens,),
            dtype=torch.int32,
            device="xpu",
        )
        correction_bias = None

    for _ in range(warmup_iters):
        fused_topk_softplus_sqrt(
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            correction_bias=correction_bias,
            input_ids=input_ids,
            hash_indices_table=hash_indices_table,
        )
    torch.xpu.synchronize()

    start_time = time.perf_counter()
    for _ in range(benchmark_iters):
        fused_topk_softplus_sqrt(
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            correction_bias=correction_bias,
            input_ids=input_ids,
            hash_indices_table=hash_indices_table,
        )
    torch.xpu.synchronize()
    end_time = time.perf_counter()

    return (end_time - start_time) * 1e6 / benchmark_iters


def profile_fused_topk_softplus_sqrt(
    num_tokens: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    dtype: torch.dtype,
    use_bias: bool = True,
    use_hash: bool = False,
    warmup_iters: int = 10,
    active_iters: int = 20,
    profiling_dir: str = "profiling_topk_softplus_sqrt",
) -> dict[str, list[float]]:
    seed_everything(0)
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=dtype, device="xpu"
    )
    correction_bias = None
    input_ids = None
    hash_indices_table = None

    if use_bias:
        correction_bias = torch.randn(
            (num_experts,), dtype=torch.float32, device="xpu"
        )

    if use_hash:
        vocab_size = 1024
        hash_indices_table = torch.stack(
            [torch.randperm(num_experts)[:topk] for _ in range(vocab_size)]
        ).to(device="xpu", dtype=torch.int32)
        input_ids = torch.randint(
            0,
            vocab_size,
            (num_tokens,),
            dtype=torch.int32,
            device="xpu",
        )
        correction_bias = None

    trace_dir = pathlib.Path(profiling_dir)
    trace_file = trace_dir / "trace.json"
    if trace_dir.exists():
        shutil.rmtree(trace_dir)
    trace_dir.mkdir(parents=True)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.XPU],
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=warmup_iters,
            active=active_iters,
            repeat=1,
        ),
        on_trace_ready=lambda prof: prof.export_chrome_trace(str(trace_file)),
    ) as prof:
        for _ in range(warmup_iters + active_iters):
            fused_topk_softplus_sqrt(
                gating_output=gating_output,
                topk=topk,
                renormalize=renormalize,
                routed_scaling_factor=routed_scaling_factor,
                correction_bias=correction_bias,
                input_ids=input_ids,
                hash_indices_table=hash_indices_table,
            )
            torch.xpu.synchronize()
            prof.step()

    profiling_data = json.loads(trace_file.read_text())
    kernel_latency: dict[str, list[float]] = {}
    for event in profiling_data["traceEvents"]:
        if event.get("cat") == "kernel":
            kernel_latency.setdefault(event["name"], []).append(event["dur"])
    return kernel_latency


@pytest.mark.parametrize("num_tokens", [1, 33, 128])
@pytest.mark.parametrize("num_experts", [128, 256, 384, 512])
@pytest.mark.parametrize("topk", [6, 8, 16])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 1.5])
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float16, torch.float32],
    ids=format_tc,
)
def test_fused_topk_softplus_sqrt(
    num_tokens: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    dtype: torch.dtype,
):
    seed_everything(0)
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=dtype, device="xpu"
    )
    e_score_correction_bias = torch.randn(
        (num_experts,), dtype=torch.float32, device="xpu"
    )

    topk_weights_ref, topk_ids_ref = _torch_topk_softplus_sqrt(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
    )

    topk_weights, topk_ids = fused_topk_softplus_sqrt(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        correction_bias=e_score_correction_bias,
    )

    sorted_ref_ids, idx_ref = topk_ids_ref.sort(dim=-1)
    sorted_ids, idx_ops = topk_ids.sort(dim=-1)
    torch.testing.assert_close(sorted_ref_ids, sorted_ids, atol=0, rtol=0)

    sorted_w_ref = topk_weights_ref.gather(1, idx_ref)
    sorted_w = topk_weights.gather(1, idx_ops)
    torch.testing.assert_close(sorted_w_ref, sorted_w, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("num_tokens", [1, 33, 128,1024])
@pytest.mark.parametrize("num_experts", [256, 384, 512])
@pytest.mark.parametrize("topk", [6, 8, 16])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float16, torch.float32],
    ids=format_tc,
)
def test_fused_topk_softplus_sqrt_hash(
    num_tokens: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    dtype: torch.dtype,
):
    seed_everything(0)
    vocab_size = 1024
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=dtype, device="xpu"
    )
    hash_indices_table = torch.stack(
        [torch.randperm(num_experts)[:topk] for _ in range(vocab_size)]
    ).to(device="xpu", dtype=torch.int32)
    input_ids = torch.randint(
        0,
        vocab_size,
        (num_tokens,),
        dtype=torch.int32,
        device="xpu",
    )

    topk_weights_ref, topk_ids_ref = _torch_topk_softplus_sqrt(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        input_ids=input_ids,
        hash_indices_table=hash_indices_table,
    )

    topk_weights, topk_ids = fused_topk_softplus_sqrt(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        input_ids=input_ids,
        hash_indices_table=hash_indices_table,
    )

    sorted_ref_ids, idx_ref = topk_ids_ref.sort(dim=-1)
    sorted_ids, idx_ops = topk_ids.sort(dim=-1)
    torch.testing.assert_close(sorted_ref_ids, sorted_ids, atol=0, rtol=0)

    sorted_w_ref = topk_weights_ref.gather(1, idx_ref)
    sorted_w = topk_weights.gather(1, idx_ops)
    torch.testing.assert_close(sorted_w_ref, sorted_w, atol=2e-2, rtol=1e-2)


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark fused_topk_softplus_sqrt.")
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--dtype",
                        choices=["float16", "bfloat16", "float32"],
                        default="float32")
    parser.add_argument("--renormalize", action="store_true")
    parser.add_argument("--routed-scaling-factor", type=float, default=2.5)
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--hash", action="store_true")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--benchmark-iters", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-active-iters", type=int, default=20)
    parser.add_argument("--profiling-dir",
                        type=str,
                        default="profiling_topk_softplus_sqrt")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    latency_us = benchmark_fused_topk_softplus_sqrt(
        num_tokens=args.num_tokens,
        num_experts=args.num_experts,
        topk=args.topk,
        renormalize=args.renormalize,
        routed_scaling_factor=args.routed_scaling_factor,
        dtype=dtype,
        use_bias=not args.no_bias,
        use_hash=args.hash,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
    )
    print(f"fused_topk_softplus_sqrt average latency: {latency_us:.3f} us")

    if args.profile:
        kernel_latency = profile_fused_topk_softplus_sqrt(
            num_tokens=args.num_tokens,
            num_experts=args.num_experts,
            topk=args.topk,
            renormalize=args.renormalize,
            routed_scaling_factor=args.routed_scaling_factor,
            dtype=dtype,
            use_bias=not args.no_bias,
            use_hash=args.hash,
            warmup_iters=args.warmup_iters,
            active_iters=args.profile_active_iters,
            profiling_dir=args.profiling_dir,
        )
        for kernel_name, latency_list in sorted(kernel_latency.items()):
            avg_latency = sum(latency_list) / len(latency_list)
            print(
                f"{kernel_name}: avg={avg_latency:.3f} us, "
                f"count={len(latency_list)}"
            )