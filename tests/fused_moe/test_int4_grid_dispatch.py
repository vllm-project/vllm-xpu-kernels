# SPDX-License-Identifier: Apache-2.0
"""Branch-selection + parity tests for the kernels-side int4 grid dispatch.

The dispatch lives in XpuFusedMoe (fused_moe_interface.py): eligible
decode calls run the fused int4 grid, everything else keeps the
grouped-GEMM path. CPU legs cover every device-independent
fall-through; XPU legs prove the taken leg and numeric parity against
the fallback on the same weights.
"""
import pytest
import torch

import vllm_xpu_kernels.moe_shared_fused_int4_interface as grid_iface
from tests.fused_moe.test_moe_shared_fused_decode_int4 import (
    quant_int4_per_group,
)
from tests.utils import seed_everything
from vllm_xpu_kernels.fused_moe_interface import (
    INT4_GRID_ENV,
    INT4_GRID_MAX_M,
    XpuFusedMoe,
)

H = 256
INTER = 128
NUM_EXPERTS = 4
TOP_K = 2
GROUP_SIZE = 128
DTYPE = torch.float16

requires_moe_kernel = pytest.mark.skipif(
    not grid_iface.is_available(),
    reason="moe_shared_fused_decode_int4 op not built")
requires_xpu = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="no XPU device")


def pack_u4_pair(lo, hi):
    """Pre-image byte for implement_zp: u4 nibbles holding value+8."""
    return (((hi + 8) << 4) | (lo + 8)).to(torch.uint8)


def pack_s4_pair(lo, hi):
    """Post-rewrite byte: plain 4-bit two's complement nibbles."""
    lo4 = lo.to(torch.int64) & 0xF
    hi4 = hi.to(torch.int64) & 0xF
    return ((hi4 << 4) | lo4).to(torch.uint8)


def make_u4_weights(e, n, k, device):
    """Random s4 values; returns (pre-zp uint8, post-zp int8)."""
    assert k % 2 == 0
    vals = torch.randint(-8, 8, (e, n, k), device="cpu", dtype=torch.int64)
    lo, hi = vals[..., 0::2], vals[..., 1::2]
    pre = pack_u4_pair(lo, hi).to(device)
    post = pack_s4_pair(lo, hi).to(torch.int8).to(device)
    return pre.contiguous(), post.contiguous()


def make_scales(e, n, k, group, dtype, device):
    ng = k // group
    assert k % group == 0
    # Realistic per-group scales for small normalized weights
    # (matches a randn/32 base, amax/7); keeps fp16 activations finite.
    return (torch.rand(e, n, ng, device=device, dtype=torch.float32)
            * 0.04 + 0.01).to(dtype).contiguous()


def make_moe(device, m=2, top_k=TOP_K, activation="silu", w13_bias=None,
             w2_bias=None, clamp=None, dtype=DTYPE, group=GROUP_SIZE,
             h=H, inter=INTER, e=NUM_EXPERTS, ep_size=1, expert_map=None,
             seed=0):
    seed_everything(seed)
    dev = torch.device(device)
    w13_pre, w13_post = make_u4_weights(e, 2 * inter, h, device)
    w2_pre, w2_post = make_u4_weights(e, h, inter, device)
    s13 = make_scales(e, 2 * inter, h, group, dtype, device)
    s2 = make_scales(e, h, inter, group, dtype, device)
    bias13 = (torch.zeros(2 * inter, device=dev, dtype=dtype)
              if w13_bias else None)
    bias2 = torch.zeros(h, device=dev, dtype=dtype) if w2_bias else None
    moe = XpuFusedMoe(w13_pre, s13, bias13, w2_pre, s2, bias2, top_k,
                      activation, e, ep_size=ep_size, expert_map=expert_map,
                      gemm1_clamp_limit=clamp)
    # The in-place rewrite must reproduce the expected s4 packing.
    assert moe.w13.dtype == torch.int8
    assert torch.equal(moe.w13.cpu().to(torch.uint8),
                       w13_post.cpu().to(torch.uint8))
    assert torch.equal(moe.w2.cpu().to(torch.uint8),
                       w2_post.cpu().to(torch.uint8))
    return moe


def make_io(moe_device, m=2, top_k=TOP_K, dtype=DTYPE, h=H):
    dev = torch.device(moe_device)
    x = torch.randn(m, h, device=dev, dtype=dtype).contiguous()
    out = torch.empty(m, h, device=dev, dtype=dtype).contiguous()
    tw = (torch.rand(m, top_k, device=dev, dtype=torch.float32) + 0.1)
    tw = (tw / tw.sum(dim=-1, keepdim=True)).contiguous()
    ti = torch.randint(0, NUM_EXPERTS, (m, top_k), device=dev,
                       dtype=torch.int32).contiguous()
    return out, x, tw, ti


def eligible_args(moe, m=2, top_k=TOP_K, dtype=DTYPE, h=H):
    out, x, tw, ti = make_io(str(moe.w13.device), m, top_k, dtype, h)
    assert moe._int4_grid_eligible(out, x, tw, ti)
    return out, x, tw, ti


# CPU legs: each S4 fall-through alone forces the grouped-GEMM path. No
# device or built grid op is needed for these (they return before any
# device query), so they also run where the op is absent.
def test_grid_env_kill_switch_falls_through(monkeypatch):
    moe = make_moe("cpu")
    out, x, tw, ti = make_io("cpu")
    assert moe._int4_grid_eligible(out, x, tw, ti) is False  # cpu device
    monkeypatch.setenv(INT4_GRID_ENV, "0")
    assert moe._try_int4_grid(out, x, tw, ti) is False
    assert moe._int4_grid is None


def test_grid_non_silu_falls_through():
    moe = make_moe("cpu", activation="gelu")
    out, x, tw, ti = make_io("cpu")
    assert moe._int4_grid_eligible(out, x, tw, ti) is False


@pytest.mark.parametrize("which", ["w13", "w2"])
def test_grid_bias_falls_through(which):
    kw = {"w13_bias": True} if which == "w13" else {"w2_bias": True}
    moe = make_moe("cpu", **kw)
    out, x, tw, ti = make_io("cpu")
    assert moe._int4_grid_eligible(out, x, tw, ti) is False


def test_grid_clamp_falls_through():
    moe = make_moe("cpu", clamp=5.0)
    out, x, tw, ti = make_io("cpu")
    assert moe._int4_grid_eligible(out, x, tw, ti) is False


def test_grid_act_quant_falls_through():
    moe = make_moe("cpu")
    out, x, tw, ti = make_io("cpu")
    a1q = torch.ones(x.shape[0] * TOP_K, dtype=torch.float32)
    assert moe._int4_grid_eligible(out, x, tw, ti, a1q_scale=a1q) is False


def test_grid_expert_map_arg_falls_through():
    moe = make_moe("cpu")
    out, x, tw, ti = make_io("cpu")
    emap = torch.arange(NUM_EXPERTS, dtype=torch.int32)
    assert moe._int4_grid_eligible(out, x, tw, ti, expert_map=emap) is False


def test_grid_too_many_tokens_falls_through():
    moe = make_moe("cpu")
    out, x, tw, ti = make_io("cpu", m=INT4_GRID_MAX_M + 1)
    assert moe._int4_grid_eligible(out, x, tw, ti) is False


def test_grid_too_many_rows_falls_through():
    # R = M * top_k exceeds the C++ routing-row bound with top_k legal.
    e, top_k, m = 256, 128, 6
    assert m * top_k > 512 and top_k <= e
    moe = make_moe("cpu", m=m, top_k=top_k, e=e)
    out, x, tw, ti = make_io("cpu", m=m, top_k=top_k)
    ti = (ti % e).to(torch.int32)
    assert moe._int4_grid_eligible(out, x, tw, ti) is False


def test_grid_topk_mismatch_falls_through():
    moe = make_moe("cpu")
    out, x, tw, ti = make_io("cpu", top_k=TOP_K + 1)
    assert moe._int4_grid_eligible(out, x, tw, ti) is False


def test_grid_wrong_activation_dtype_falls_through():
    moe = make_moe("cpu")
    out, x, tw, ti = make_io("cpu", dtype=torch.float32)
    assert moe._int4_grid_eligible(out, x, tw, ti) is False


def test_grid_unrewritten_weights_fall_through():
    moe = make_moe("cpu")
    moe.w13.xpu_fused_moe = False
    out, x, tw, ti = make_io("cpu")
    assert moe._int4_grid_eligible(out, x, tw, ti) is False


def test_grid_non_int4_weights_fall_through():
    dev = torch.device("cpu")
    w13 = torch.randn(NUM_EXPERTS, H, INTER, device=dev,
                      dtype=torch.float16)
    w2 = torch.randn(NUM_EXPERTS, INTER, H // 2, device=dev,
                     dtype=torch.float16)
    moe = XpuFusedMoe(w13, None, None, w2, None, None, TOP_K, "silu",
                      NUM_EXPERTS)
    out, x, tw, ti = make_io("cpu")
    assert moe._int4_grid_eligible(out, x, tw, ti) is False


def test_grid_bad_group_size_falls_through():
    moe = make_moe("cpu", group=16)
    out, x, tw, ti = make_io("cpu")
    assert moe._int4_grid_eligible(out, x, tw, ti) is False


def test_grid_op_absent_falls_through(monkeypatch):
    # Against released kernels without the grid op the branch must stay
    # quiet: no import error, no launch, grouped GEMMs cover the call.
    monkeypatch.setattr(grid_iface, "is_available", lambda: False)
    moe = make_moe("cpu")
    out, x, tw, ti = make_io("cpu")
    assert moe._int4_grid_eligible(out, x, tw, ti) is False
    assert moe._int4_grid is None


# XPU legs: the taken leg, the remaining device-bound fall-throughs,
# and numeric parity against the grouped-GEMM fallback.
@requires_xpu
@requires_moe_kernel
def test_grid_taken_and_sticky():
    moe = make_moe("xpu")
    out, x, tw, ti = eligible_args(moe)
    assert moe._try_int4_grid(out, x, tw, ti) is True
    assert moe._int4_grid is not None
    grid = moe._int4_grid
    assert grid.has_shared is False
    ws0 = grid.workspace(x.shape[0])
    out2, x2, tw2, ti2 = make_io("xpu")
    assert moe._try_int4_grid(out2, x2, tw2, ti2) is True
    assert moe._int4_grid is grid
    assert grid.workspace(x2.shape[0]).data_ptr() == ws0.data_ptr()
    assert torch.isfinite(out).all() and torch.isfinite(out2).all()


@requires_xpu
@requires_moe_kernel
def test_grid_ep_size_falls_through():
    moe = make_moe("xpu", ep_size=2)
    out, x, tw, ti = make_io("xpu")
    assert moe._int4_grid_eligible(out, x, tw, ti) is False


@requires_xpu
@requires_moe_kernel
def test_grid_apply_takes_grid_path(monkeypatch):
    moe = make_moe("xpu")
    out, x, tw, ti = make_io("xpu")
    calls = []
    orig = XpuFusedMoe._apply_int4_grid

    def spy(self, o, h, w, i):
        calls.append(True)
        return orig(self, o, h, w, i)

    monkeypatch.setattr(XpuFusedMoe, "_apply_int4_grid", spy)
    moe.apply(out, x, tw, ti)
    assert len(calls) == 1
    assert torch.isfinite(out).all()


@requires_xpu
@requires_moe_kernel
def test_grid_apply_env_off_uses_fallback(monkeypatch):
    moe = make_moe("xpu")
    monkeypatch.setenv(INT4_GRID_ENV, "0")
    out, x, tw, ti = make_io("xpu")
    calls = []
    orig = XpuFusedMoe._apply_int4_grid

    def spy(self, o, h, w, i):
        calls.append(True)
        return orig(self, o, h, w, i)

    monkeypatch.setattr(XpuFusedMoe, "_apply_int4_grid", spy)
    moe.apply(out, x, tw, ti)
    assert calls == []
    assert torch.isfinite(out).all()


# Bar the int4 suite already accepts for grid-vs-fp32-reference
# (out.float() vs fp32 CPU ref): rtol=atol=1e-2. Grid-vs-fallback spans
# a smaller gap (same weights, same routing, kernel math only).
GRID_PARITY_TOL = {"rtol": 1e-2, "atol": 1e-2}


def to_pre_zp(post):
    """Invert the packed s4 bytes to the u4 pre-image implement_zp takes.

    Post nibbles hold the value mod 16; pre nibbles hold value+8 mod 16.
    """
    u = post.to(torch.uint8)
    return (((((u >> 4) + 8) & 0xF) << 4) | ((((u & 0xF) + 8)) & 0xF))


@requires_xpu
@requires_moe_kernel
@pytest.mark.parametrize("m", [1, 2, 6])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_grid_parity_vs_fallback(m, dtype, monkeypatch):
    # Same rewritten weights, same routing: grid output must match the
    # grouped-GEMM fallback within kernel-math tolerance.
    seed_everything(300 + m)
    e, h, inter, top_k, group = 8, 2048, 512, 8, 128
    dev = torch.device("xpu")
    w13_post, s13 = quant_int4_per_group(
        torch.randn(e, 2 * inter, h) / 32, group, dtype)
    w2_post, s2 = quant_int4_per_group(
        torch.randn(e, h, inter) / 32, group, dtype)
    w13_pre = to_pre_zp(w13_post).to(dev)
    w2_pre = to_pre_zp(w2_post).to(dev)
    moe = XpuFusedMoe(w13_pre, s13.to(dev), None, w2_pre, s2.to(dev),
                      None, top_k, "silu", e)
    # The in-place rewrite must reproduce the suite's packed bytes.
    assert torch.equal(moe.w13.cpu().to(torch.uint8),
                       w13_post.to(torch.uint8))
    assert torch.equal(moe.w2.cpu().to(torch.uint8), w2_post.to(torch.uint8))
    assert moe._int4_grid is None
    x = (torch.randn(m, h, device=dev) / 4).to(dtype).contiguous()
    torch.xpu.synchronize()
    logits = torch.randn(m, e, device=dev)
    tw, ti = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    tw = (tw / tw.sum(dim=-1, keepdim=True)).to(torch.float32).contiguous()
    ti = ti.to(torch.int32).contiguous()
    out_grid = torch.empty(m, h, device=dev, dtype=dtype)
    assert moe._try_int4_grid(out_grid, x, tw, ti) is True
    torch.xpu.synchronize()
    monkeypatch.setenv(INT4_GRID_ENV, "0")
    out_ref = torch.empty(m, h, device=dev, dtype=dtype)
    moe.apply(out_ref, x, tw, ti)
    torch.xpu.synchronize()
    assert torch.isfinite(out_grid).all()
    assert torch.isfinite(out_ref).all()
    torch.testing.assert_close(out_grid.float(), out_ref.float(),
                               rtol=GRID_PARITY_TOL["rtol"],
                               atol=GRID_PARITY_TOL["atol"])
    maxdiff = (out_grid.float() - out_ref.float()).abs().max().item()
    meandiff = (out_grid.float() - out_ref.float()).abs().mean().item()
    print(f"parity m={m} {dtype} maxdiff={maxdiff:.3e} "
          f"meandiff={meandiff:.3e}")
