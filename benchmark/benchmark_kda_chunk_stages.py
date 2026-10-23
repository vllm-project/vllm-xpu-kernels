# SPDX-License-Identifier: Apache-2.0
"""Focused benchmark for the Xe2 chunked KDA prefill pipeline.

Runs only shapes that reach the `chunk` backend so that unitrace's per-kernel
breakdown attributes time to the seven pipeline stages rather than to the
sequential fallback. Used as the optimization loop's benchmark harness.
"""
# ruff: noqa: E402
import argparse
import os
import sys

os.environ.setdefault("VLLM_XPU_KDA_RECURRENT_MODE", "chunk")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vllm_xpu_kernels._xpu_C  # noqa: F401

DEVICE = torch.device("xpu")

# (name, batch, seqlen) -- prefill only, which is what the chunk path serves.
CASES = [
    ("b1_1k", 1, 1024),
    ("b1_4k", 1, 4096),
    ("b1_8k", 1, 8192),
    ("b4_2k", 4, 2048),
    ("b8_1k", 8, 1024),
    ("b16_512", 16, 512),
    ("b32_256", 32, 256),
]


def make_inputs(batch, seqlen, num_heads, head_dim, dtype, lower_bound):
    torch.manual_seed(1234)
    n_tok = batch * seqlen
    hidden = num_heads * head_dim
    g = torch.Generator(device=DEVICE).manual_seed(1234)

    def rnd(*shape, d=dtype):
        return torch.randn(*shape, generator=g, device=DEVICE, dtype=d) * 0.2

    q = rnd(n_tok, hidden)
    k = rnd(n_tok, hidden)
    v = rnd(n_tok, hidden)
    raw_gate = rnd(1, n_tok, num_heads, head_dim)
    raw_beta = rnd(1, n_tok, num_heads, d=torch.float32)
    core_attn_out = torch.empty(
        1, n_tok, num_heads, head_dim, device=DEVICE, dtype=dtype
    )
    recurrent_state = torch.zeros(
        batch + 1, num_heads, head_dim, head_dim,
        device=DEVICE, dtype=torch.float32,
    )
    a_log = torch.rand(num_heads, generator=g, device=DEVICE) * 2.0 - 1.0
    dt_bias = torch.rand(hidden, generator=g, device=DEVICE) * 0.5
    qsl = torch.arange(
        0, n_tok + 1, seqlen, dtype=torch.int32, device=DEVICE
    )
    state_indices = torch.arange(batch, dtype=torch.int32, device=DEVICE)
    has_initial_state = torch.ones(batch, dtype=torch.bool, device=DEVICE)
    return dict(
        core_attn_out=core_attn_out, q=q, k=k, v=v, raw_gate=raw_gate,
        raw_beta=raw_beta, recurrent_state=recurrent_state, a_log=a_log,
        dt_bias=dt_bias, num_prefills=batch, num_decodes=0,
        num_spec_decodes=0, has_initial_state=has_initial_state,
        non_spec_query_start_loc=qsl, non_spec_token_indx=None,
        non_spec_state_indices=state_indices, spec_query_start_loc=None,
        spec_token_indx=None, spec_state_indices=None,
        num_accepted_tokens=None, num_actual_tokens=n_tok,
        gate_lower_bound=lower_bound,
    )


def run(kw):
    torch.ops._xpu_C.kda_gated_delta_rule(
        kw["core_attn_out"], kw["q"], kw["k"], kw["v"], kw["raw_gate"],
        kw["raw_beta"], kw["recurrent_state"], kw["a_log"], kw["dt_bias"],
        kw["num_prefills"], kw["num_decodes"], kw["num_spec_decodes"],
        kw["has_initial_state"], kw["non_spec_query_start_loc"],
        kw["non_spec_token_indx"], kw["non_spec_state_indices"],
        kw["spec_query_start_loc"], kw["spec_token_indx"],
        kw["spec_state_indices"], kw["num_accepted_tokens"],
        kw["num_actual_tokens"], kw["gate_lower_bound"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--lower-bound", type=float, default=None)
    ap.add_argument("--case", default=None, help="run a single case by name")
    ap.add_argument("--profile", action="store_true",
                    help="single case, few iters, for unitrace")
    args = ap.parse_args()

    cases = CASES
    if args.case:
        cases = [c for c in CASES if c[0] == args.case]
        assert cases, f"unknown case {args.case}"

    dtype = torch.bfloat16
    results = []
    for name, batch, seqlen in cases:
        kw = make_inputs(batch, seqlen, args.heads, args.head_dim, dtype,
                         args.lower_bound)
        for _ in range(args.warmup):
            run(kw)
        torch.xpu.synchronize()
        if args.profile:
            for _ in range(3):
                run(kw)
            torch.xpu.synchronize()
            continue
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            run(kw)
        end.record()
        torch.xpu.synchronize()
        us = start.elapsed_time(end) * 1000.0 / args.iterations
        n_tok = batch * seqlen
        # 7 head_dim^2 MACs per token per head dominates the pipeline.
        flops = n_tok * args.heads * 7 * args.head_dim * args.head_dim * 2
        results.append((name, us, flops / (us * 1e-6) / 1e12))

    if args.profile:
        return
    total = 0.0
    print(f"{'case':<10} {'us':>10} {'TFLOP/s':>9}")
    for name, us, tf in results:
        print(f"{name:<10} {us:10.2f} {tf:9.2f}")
        total += us
    print(f"{'TOTAL':<10} {total:10.2f}")


if __name__ == "__main__":
    main()
