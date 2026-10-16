# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
import itertools
from argparse import ArgumentParser
from typing import Optional

import torch
import torch.nn.functional as F
import triton
from utils import bootstrap_benchmark_env

bootstrap_benchmark_env(__file__)

from tests.register_ops import topk_softplus_sqrt  # noqa: E402


@torch.compile
def topk_softplus_sqrt_compile(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype] = None,
    routed_scaling_factor: float = 1.0,
    input_ids: Optional[torch.Tensor] = None,
    tid2eid: Optional[torch.Tensor] = None,
    bias_vl: Optional[torch.Tensor] = None,
    image_sentinel_lo: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    del hidden_states, indices_type

    routing_weights = torch.sqrt(
        F.softplus(gating_output.to(torch.float32), beta=1.0, threshold=20.0)
    )

    if tid2eid is not None:
        assert input_ids is not None
        expert_ids = tid2eid[input_ids.to(torch.long)].to(torch.long)
        topk_ids = expert_ids
        if bias_vl is not None and image_sentinel_lo > 0:
            image_mask = (input_ids.to(torch.long) >= image_sentinel_lo) & (
                input_ids.to(torch.long) < image_sentinel_lo + 5
            )
            vl_ids = torch.topk(
                routing_weights + bias_vl, topk, dim=-1
            ).indices
            topk_ids = torch.where(image_mask.unsqueeze(-1), vl_ids, topk_ids)
        topk_weights = torch.gather(routing_weights, dim=-1, index=topk_ids)
    else:
        scores_for_choice = routing_weights
        if bias_vl is not None and image_sentinel_lo > 0:
            assert input_ids is not None
            image_mask = (input_ids.to(torch.long) >= image_sentinel_lo) & (
                input_ids.to(torch.long) < image_sentinel_lo + 5
            )
            scores_for_choice = routing_weights + (
                image_mask.unsqueeze(-1) * bias_vl.unsqueeze(0)
            )
        topk_ids = torch.topk(scores_for_choice, topk, dim=-1).indices
        topk_weights = torch.gather(routing_weights, dim=-1, index=topk_ids)

    if renormalize:
        denom = topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights / denom

    topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def topk_softplus_sqrt_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype] = None,
    routed_scaling_factor: float = 1.0,
    input_ids: Optional[torch.Tensor] = None,
    tid2eid: Optional[torch.Tensor] = None,
    bias_vl: Optional[torch.Tensor] = None,
    image_sentinel_lo: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    del hidden_states, indices_type

    routing_weights = torch.sqrt(
        F.softplus(gating_output.to(torch.float32), beta=1.0, threshold=20.0)
    )

    if tid2eid is not None:
        assert input_ids is not None
        expert_ids = tid2eid[input_ids.to(torch.long)].to(torch.long)
        topk_ids = expert_ids
        if bias_vl is not None and image_sentinel_lo > 0:
            image_mask = (input_ids.to(torch.long) >= image_sentinel_lo) & (
                input_ids.to(torch.long) < image_sentinel_lo + 5
            )
            vl_ids = torch.topk(
                routing_weights + bias_vl, topk, dim=-1
            ).indices
            topk_ids = torch.where(image_mask.unsqueeze(-1), vl_ids, topk_ids)
        topk_weights = torch.gather(routing_weights, dim=-1, index=topk_ids)
    else:
        scores_for_choice = routing_weights
        if bias_vl is not None and image_sentinel_lo > 0:
            assert input_ids is not None
            image_mask = (input_ids.to(torch.long) >= image_sentinel_lo) & (
                input_ids.to(torch.long) < image_sentinel_lo + 5
            )
            scores_for_choice = routing_weights + (
                image_mask.unsqueeze(-1) * bias_vl.unsqueeze(0)
            )
        topk_ids = torch.topk(scores_for_choice, topk, dim=-1).indices
        topk_weights = torch.gather(routing_weights, dim=-1, index=topk_ids)

    if renormalize:
        denom = topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights / denom

    topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def fused_topk_softplus_sqrt(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype] = None,
    routed_scaling_factor: float = 1.0,
    input_ids: Optional[torch.Tensor] = None,
    tid2eid: Optional[torch.Tensor] = None,
    bias_vl: Optional[torch.Tensor] = None,
    image_sentinel_lo: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Python adapter for the in-place C++ op signature."""
    del hidden_states

    if indices_type is None:
        indices_type = torch.int32

    num_tokens = gating_output.numel() // gating_output.shape[-1]
    topk_weights = torch.empty(
        (num_tokens, topk), dtype=torch.float32, device=gating_output.device
    )
    topk_indices = torch.empty(
        (num_tokens, topk), dtype=indices_type, device=gating_output.device
    )
    token_expert_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=gating_output.device
    )

    topk_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        float(routed_scaling_factor),
        None,
        input_ids,
        tid2eid,
        None,
        bias_vl,
        image_sentinel_lo,
    )
    return topk_weights, topk_indices.to(torch.int32)


def make_hash_inputs(
    n_token: int,
    n_expert: int,
    topk: int,
    dtype: torch.dtype,
    device: str = "xpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create input_ids and tid2eid for the C++ USE_HASH path.

    The kernel expects:
      input_ids[token_row] -> token_id
      tid2eid[token_id * topk + k_idx] -> selected expert id

    Keep input_ids/tid2eid dtype equal to topk_indices dtype used by benchmark
    here, i.e. int32.
    """
    del dtype
    input_ids = torch.arange(n_token, dtype=torch.int32, device=device)

    # Use deterministic random top-k expert ids, unique within each token.
    # This is generated once per benchmark case and reused inside do_bench.
    scores = torch.randn((n_token, n_expert), 
                         dtype=torch.float32, device=device)
    tid2eid = torch.topk(scores, topk, dim=-1).indices.to(torch.int32)
    return input_ids, tid2eid.contiguous()


def make_vl_input_ids(
    n_token: int,
    image_sentinel_lo: int,
    device: str = "xpu",
) -> torch.Tensor:
    input_ids = torch.randint(
        0, image_sentinel_lo, (n_token,), dtype=torch.int32, device=device
    )
    positions = torch.arange(n_token, device=device)
    image_rows = positions % 3 == 0
    input_ids[image_rows] = (
        image_sentinel_lo + (positions[image_rows] // 3) % 5
    ).to(torch.int32)
    input_ids[positions % 6 == 3] = image_sentinel_lo + 5
    return input_ids


n_token_range = [1, 64, 256]
# Keep only expert counts supported by the current C++ dispatch switch.
n_expert_range = [16, 192, 512]
topk_range = [2, 4, 8]
renormalize_range = [True, False]
with_hash_range = [False, True]
with_vl_range = [False, True]
dtype_range = [torch.float16, torch.bfloat16, torch.float32]
configs = list(
    itertools.product(
        n_token_range,
        n_expert_range,
        topk_range,
        renormalize_range,
        with_hash_range,
        with_vl_range,
        dtype_range,
    )
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=[
            "n_token",
            "n_expert",
            "topk",
            "renormalize",
            "with_hash",
            "with_vl",
            "dtype",
        ],
        x_vals=[tuple(_) for _ in configs],
        line_arg="provider",
        line_vals=["vllm", "native", "compile"],
        line_names=["vllm", "native", "compile"],
        styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
        ylabel="us",
        plot_name="topk_softplus_sqrt-perf",
        args={},
    )
)
def benchmark(
    n_token: int,
    n_expert: int,
    topk: int,
    renormalize: bool,
    with_hash: bool,
    with_vl: bool,
    dtype: torch.dtype,
    provider: str = "vllm",
):
    n_hidden = 1024
    hidden_states = torch.randn(
        (n_token, n_hidden), dtype=dtype, device="xpu"
    )
    gating_output = torch.randn(
        (n_token, n_expert), dtype=dtype, device="xpu"
    )

    if with_hash:
        input_ids, tid2eid = make_hash_inputs(n_token, n_expert, topk, dtype)
    else:
        input_ids, tid2eid = None, None

    bias_vl = None
    image_sentinel_lo = 0
    if with_vl:
        image_sentinel_lo = max(n_token, 16)
        input_ids = make_vl_input_ids(n_token, image_sentinel_lo)
        if with_hash:
            vocab_size = image_sentinel_lo + 8
            tid2eid = torch.stack(
                [torch.randperm(n_expert)[:topk] for _ in range(vocab_size)]
            ).to(device="xpu", dtype=torch.int32)
        bias_vl = torch.randn(n_expert, dtype=torch.float32, device="xpu")

    quantiles = [0.5, 0.2, 0.8]

    common_kwargs = dict(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        indices_type=torch.int32,
        routed_scaling_factor=1.0,
        input_ids=input_ids,
        tid2eid=tid2eid,
        bias_vl=bias_vl,
        image_sentinel_lo=image_sentinel_lo,
    )

    if provider == "vllm":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fused_topk_softplus_sqrt(**common_kwargs),
            quantiles=quantiles,
        )
    elif provider == "native":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: topk_softplus_sqrt_native(**common_kwargs),
            quantiles=quantiles,
        )
    elif provider == "compile":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: topk_softplus_sqrt_compile(**common_kwargs),
            quantiles=quantiles,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Benchmark the topk softplus-sqrt kernel."
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/topk-softplus-sqrt/",
        help="Path to save topk softplus-sqrt benchmark results",
    )

    args = parser.parse_args()
    save_path = f"{args.save_path.rstrip('/')}/softplus_sqrt"
    benchmark.run(print_data=True, save_path=save_path)
