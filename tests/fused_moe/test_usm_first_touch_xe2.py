# SPDX-License-Identifier: Apache-2.0
"""Test that offloaded weights of various sizes can be accessed from the GPU.

When MoE weights are offloaded from the discrete GPU onto DRAM, they sit
in host memory with no device-side mapping, and the xe2 Cutlass MoE
kernel's first touch (an LSC block-2D access) cannot establish one, so
the first launch dies with -ENOENT and an engine reset. The launcher now
runs a one-shot demand read before the first submit.

A warmup maps the host views first, so a fault isolates to the launcher:
unfixed dies at production scale, fixed serves at both.
"""
import pytest
import torch

vllm = pytest.importorskip("vllm")

import vllm_xpu_kernels._xpu_C  # noqa: F401
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or
    (not torch.ops._xpu_C.is_bmg(0) and not torch.ops._xpu_C.is_pvc(0)),
    reason="XE2 CUTLASS tests only run on BMG or PVC.")

# Production-scale shapes: 256 experts, K=2048 (MiniCPM-V-4.7 MoE).
TOPK, M, K, INTER, GROUP = 8, 8, 2048, 512, 128
N = 2 * INTER


def view_of(t):
    return get_accelerator_view_from_cpu_tensor(t.pin_memory())


def run_case(n_exp, name, w13, s13, w2, s2):
    hidden = torch.ones(M, K, dtype=torch.bfloat16, device="xpu")
    topk_ids = torch.randint(0, n_exp, (M, TOPK), dtype=torch.int64,
                             device="xpu")
    topk_w = torch.full((M, TOPK), 1.0 / TOPK, dtype=torch.float32,
                        device="xpu")
    out = torch.empty(M, K, dtype=torch.bfloat16, device="xpu")
    moe = XpuFusedMoe(w13, s13, None, w2, s2, None,
                      n_experts_per_token=TOPK, activation="silu",
                      num_experts=n_exp)
    moe.apply(out, hidden, topk_w, topk_ids)
    torch.xpu.synchronize()
    assert torch.isfinite(out).all(), f"{name}: non-finite output"
    print(f"PASS {name}", flush=True)


def warmup(views):
    for v in views:
        v.float().sum()


def test_bf16_host_views():
    e = 256
    b13 = view_of(torch.zeros(e, K, N, dtype=torch.bfloat16))
    b2 = view_of(torch.zeros(e, INTER, K, dtype=torch.bfloat16))
    warmup((b13, b2))
    run_case(e, "bf16-host", b13, None, b2, None)


def test_bf16_fresh_device():
    e = 256
    d13 = torch.zeros(e, K, N, dtype=torch.bfloat16, device="xpu")
    d2 = torch.zeros(e, INTER, K, dtype=torch.bfloat16, device="xpu")
    run_case(e, "bf16-device", d13, None, d2, None)


@pytest.mark.parametrize("n_exp", [16, 256])
def test_int4_repack(n_exp):
    w13 = view_of(torch.zeros(n_exp, N, K // 2, dtype=torch.uint8))
    s13 = view_of(torch.ones(n_exp, N, K // GROUP, dtype=torch.float16))
    w2 = view_of(torch.zeros(n_exp, K, INTER // 2, dtype=torch.uint8))
    s2 = view_of(torch.ones(n_exp, K, INTER // GROUP, dtype=torch.float16))
    warmup((w13, s13, w2, s2))
    run_case(n_exp, f"int4-repack-{n_exp}", w13, s13, w2, s2)
