# SPDX-License-Identifier: Apache-2.0
"""
Minimal script to profile per-kernel timing of GDN attention prefill path.
Run with unitrace or standalone (uses xpu events for end-to-end timing).

Usage:
  # End-to-end timing only:
  python benchmark/_gdn_kernel_profile.py

  # Per-kernel profiling with unitrace:
  unitrace --chrome-kernel-logging -o gdn_trace.json -- \
    python benchmark/_gdn_kernel_profile.py
"""
import gc
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
from utils import bootstrap_benchmark_env
bootstrap_benchmark_env(__file__)

import vllm_xpu_kernels._xpu_C  # noqa: F401

DEVICE = "xpu"
DTYPE = torch.bfloat16
ITERATIONS = 50
WARMUP = 10


def make_prefill_inputs(batch_size, seqlen, tp_size=1):
    """Build inputs for prefill workload (Qwen3-Next-80B shape)."""
    num_k_heads = 16
    num_v_heads = 32
    head_k_dim = 128
    head_v_dim = 128
    width = 4

    num_prefills = batch_size
    num_decodes = 0
    num_actual_tokens = batch_size * seqlen
    cache_batch_size = max(256, batch_size * 2)

    mixed_qkvz_size = (num_k_heads // tp_size *
                       (2 * head_k_dim +
                        2 * head_v_dim * num_v_heads // num_k_heads))
    mixed_ba_size = num_k_heads // tp_size * (2 * num_v_heads // num_k_heads)
    mixed_qkv_size = (num_k_heads // tp_size *
                      (2 * head_k_dim +
                       head_v_dim * num_v_heads // num_k_heads))

    projected_states_qkvz = torch.randn(
        (num_actual_tokens, mixed_qkvz_size), dtype=DTYPE, device=DEVICE)
    projected_states_ba = torch.randn(
        (num_actual_tokens, mixed_ba_size), dtype=DTYPE, device=DEVICE)
    conv_state = torch.randn(
        (cache_batch_size, width - 1, mixed_qkv_size), dtype=DTYPE, device=DEVICE)
    ssm_state = torch.randn(
        (cache_batch_size, num_v_heads // tp_size, head_v_dim, head_k_dim),
        dtype=DTYPE, device=DEVICE)
    conv_weights = torch.randn(
        (mixed_qkv_size, width), dtype=DTYPE, device=DEVICE)
    conv_bias = torch.randn((mixed_qkv_size,), dtype=DTYPE, device=DEVICE)
    A_log = torch.randn(
        (num_v_heads // tp_size,), dtype=torch.float32, device=DEVICE)
    dt_bias = torch.randn(
        (num_v_heads // tp_size,), dtype=DTYPE, device=DEVICE)

    per_seq = torch.full((batch_size,), seqlen, dtype=torch.int64)
    non_spec_query_start_loc = torch.cat([
        torch.zeros(1, dtype=torch.int64),
        torch.cumsum(per_seq, dim=0)
    ]).to(torch.int32).to(DEVICE)

    has_initial_state = torch.ones(batch_size, dtype=torch.bool, device=DEVICE)
    non_spec_state_indices_tensor = torch.arange(
        batch_size, device=DEVICE, dtype=torch.int32)

    core_attn_out = torch.zeros(
        (num_actual_tokens, num_v_heads // tp_size, head_v_dim),
        dtype=DTYPE, device=DEVICE)
    z = torch.empty_like(core_attn_out)

    return dict(
        core_attn_out=core_attn_out,
        z=z,
        projected_states_qkvz=projected_states_qkvz,
        projected_states_ba=projected_states_ba,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_state=conv_state,
        ssm_state=ssm_state,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        activation="silu",
        A_log=A_log,
        dt_bias=dt_bias,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        num_spec_decodes=0,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        spec_query_start_loc=None,
        spec_token_indx=None,
        spec_state_indices_tensor=None,
        num_accepted_tokens=None,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size,
        reorder_input=False,
    )


def run_profile(batch_size, seqlen, tp_size=1):
    label = f"prefill_b{batch_size}_{seqlen}_tp{tp_size}"
    print(f"\n{'='*60}")
    print(f"  Profiling: {label}")
    print(f"  Total tokens: {batch_size * seqlen}")
    print(f"{'='*60}")

    kwargs = make_prefill_inputs(batch_size, seqlen, tp_size)

    # Warmup
    for _ in range(WARMUP):
        torch.ops._xpu_C.gdn_attention(**kwargs)
    torch.xpu.synchronize()

    # Timed run
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)

    start.record()
    for _ in range(ITERATIONS):
        torch.ops._xpu_C.gdn_attention(**kwargs)
    end.record()
    torch.xpu.synchronize()

    total_ms = start.elapsed_time(end)
    avg_us = total_ms * 1000.0 / ITERATIONS
    print(f"  Average latency: {avg_us:.1f} us")
    print(f"  (Run with unitrace for per-kernel breakdown)")

    # Cleanup
    del kwargs
    torch.xpu.empty_cache()
    gc.collect()

    return avg_us


if __name__ == "__main__":
    print("GDN Attention Kernel Profiling")
    print(f"Device: {torch.xpu.get_device_name()}")
    print(f"Iterations: {ITERATIONS}, Warmup: {WARMUP}")

    configs = [
        # (batch, seqlen, tp)
        (1, 4096, 1),
        (8, 1024, 1),
        (16, 512, 1),
        (32, 256, 1),
    ]

    results = []
    for batch, seqlen, tp in configs:
        us = run_profile(batch, seqlen, tp)
        results.append((f"b{batch}_s{seqlen}_tp{tp}", us))

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"  {'Config':<25} {'Latency (us)':>12}")
    print(f"  {'-'*25} {'-'*12}")
    for name, us in results:
        print(f"  {name:<25} {us:>12.1f}")
