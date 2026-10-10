# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""Fused routed + shared-expert decode op vs XpuFusedMoe + separate shared.

Shape: Qwen3.5/3.6-A3B per rank at TP2 (H=2048, I=256, E=256, top_k=8),
FP16/BF16 activations, fp8-e4m3 per-tensor weights. Reports mean wall time per
call over back-to-back calls (one sync at the end), rotating over several
weight copies and input buffers so the working set exceeds the L2 cache.

Usage: python benchmark/benchmark_moe_shared_fused_decode.py [--iters N]
       [--dtype float16|bfloat16]

The fused op targets decode (M <= 8); tile 8 vs 16 differ by <= ~1% there
and auto picks tile 8 up to 64 routed rows.
"""

# isort: off
import argparse
import time

import torch

from utils import bootstrap_benchmark_env

bootstrap_benchmark_env(__file__)
import vllm_xpu_kernels._C  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401
from tests.fused_moe.test_moe_shared_fused_decode import (HIDDEN, NUM_EXPERTS,
                                                           TOP_K, make_op,
                                                           make_weights, route)
from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe
# isort: on

DEVICE = "xpu"
M_LIST = [1, 2, 3, 4, 6, 8]
NUM_WEIGHT_COPIES = 2
NUM_INPUTS = 8


def make_baseline(weights):
    """XpuFusedMoe for routed experts + shared expert from existing ops."""
    (w13, s13, _), (w2, s2, _), (sw13, ss13, _), (sw2, ss2, _), gate = weights
    routed = XpuFusedMoe(w13=w13, w13_scales=s13, w13_bias=None, w2=w2,
                         w2_scales=s2, w2_bias=None,
                         n_experts_per_token=TOP_K, activation="silu",
                         num_experts=NUM_EXPERTS)
    inter = w2.shape[1]

    def run(x, topk_weights, topk_ids):
        out = torch.empty_like(x)
        routed.apply(output=out, hidden_states=x, topk_weights=topk_weights,
                     topk_ids=topk_ids)
        gu = torch.ops._xpu_C.fp8_gemm_w8a16(x, sw13, ss13, None)
        act = torch.empty(x.shape[0], inter, dtype=x.dtype, device=DEVICE)
        torch.ops._C.silu_and_mul(act, gu)
        sh = torch.ops._xpu_C.fp8_gemm_w8a16(act, sw2, ss2, None)
        return out + torch.sigmoid(torch.nn.functional.linear(x, gate)) * sh

    return run


def make_fused(weights, tile_m):
    op = make_op(weights, tile_m)

    def run(x, topk_weights, topk_ids):
        return op.forward(torch.empty_like(x), x, topk_weights, topk_ids)

    return run


def bench(fns, inputs, iters):
    n = len(fns) * len(inputs)
    for i in range(20):
        fns[i % len(fns)](*inputs[i % len(inputs)])
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for i in range(iters):
        fns[i % len(fns)](*inputs[(i // len(fns)) % len(inputs)])
    torch.xpu.synchronize()
    assert n > 0
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--inter", type=int, default=256)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"],
                        default="float16")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)

    torch.manual_seed(0)
    weight_sets = [make_weights(args.inter, dtype=dtype)
                   for _ in range(NUM_WEIGHT_COPIES)]
    variants = {
        "baseline": [make_baseline(w) for w in weight_sets],
        "fused_tile8": [make_fused(w, 8) for w in weight_sets],
        "fused_tile16": [make_fused(w, 16) for w in weight_sets],
        "fused_auto": [make_fused(w, 0) for w in weight_sets],
    }
    print(f"dtype={args.dtype} H={HIDDEN} I={args.inter} "
          f"E={NUM_EXPERTS} top_k={TOP_K} "
          f"(us/call, mean of {args.iters})")
    print("M".rjust(4) + "".join(k.rjust(14) for k in variants))
    for m in M_LIST:
        inputs = []
        for _ in range(NUM_INPUTS):
            x = (torch.randn(m, HIDDEN, device=DEVICE) / 4).to(dtype)
            inputs.append((x, *route(m)))
        row = [bench(fns, inputs, args.iters) for fns in variants.values()]
        print(f"{m:4d}" + "".join(f"{t:14.1f}" for t in row))


if __name__ == "__main__":
    main()
