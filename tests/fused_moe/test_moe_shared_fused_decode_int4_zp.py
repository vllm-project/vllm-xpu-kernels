# SPDX-License-Identifier: Apache-2.0
"""Zero-point battery for the fused int4 grid and its fallbacks.

Covers repack_awq_moe_for_grid on CPU (round-trip against a direct
canonical build, plus malformed-input rejections), the op-level
no-drift proof (zp-8 planes are bitwise identical to no zps), random
asymmetric planes vs an fp32 reference, malformed-plane rejections,
and the fallback lockdown: a zp layer that misses the grid (too many
rows, grid env off) still honors its zero-points through the grouped
GEMM, and the int4 GEMV stays out of the zp path.
"""
import json
import os
import struct

import pytest
import torch

import vllm_xpu_kernels.moe_shared_fused_int4_interface as interface
from tests.fused_moe.test_moe_shared_fused_decode_int4 import (
    dequant,
)
from tests.utils import seed_everything
from vllm_xpu_kernels.fused_moe_interface import (
    INT4_GRID_ENV,
    INT4_GRID_MAX_M,
    XpuFusedMoe,
    implement_zp,
)
from vllm_xpu_kernels.moe_shared_fused_int4_interface import (
    XpuMoESharedFusedDecodeInt4 as GridOp,
)
from vllm_xpu_kernels.moe_shared_fused_int4_interface import (
    is_available,
    repack_awq_moe_for_grid,
)

DEVICE = "xpu"
H = 512
INTER = 128
NUM_EXPERTS = 4
TOP_K = 2
GROUP_SIZE = 128

requires_moe_kernel = pytest.mark.skipif(
    not is_available(), reason="moe_shared_fused_decode_int4 op not built")
requires_xpu = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="no XPU device")


def quant_asym(w, group_size, dtype):
    """Asymmetric per-group int4 quant; w is [E, N, K] fp.

    Returns raw u4 pairs [E, N, K/2] uint8 (q in 0..15, element 2j in
    the low nibble), scales [E, N, K/group_size] and raw uint8 zps
    [E, N, K/group_size]. GridOp tests fold the pairs with
    fold_for_grid; XpuFusedMoe folds them itself at init.
    """
    e, n, k = w.shape
    assert k % group_size == 0
    ng = k // group_size
    wf = w.float().reshape(e, n, ng, group_size)
    wmin = wf.amin(dim=-1, keepdim=True)
    wmax = wf.amax(dim=-1, keepdim=True)
    span = (wmax - wmin).clamp_min(1e-12)
    scale = (span / 15).to(dtype)
    zp = (-wmin / span * 15).round().clamp(0, 15).to(torch.uint8)
    q = ((wf / span * 15).round() + zp.float()).clamp(0,
                                                      15).to(torch.uint8)
    pre = (q[..., 0::2] | (q[..., 1::2] << 4)).reshape(
        e, n, ng * group_size // 2).to(torch.uint8).contiguous()
    return (pre, scale.reshape(e, n, ng).contiguous(),
            zp.reshape(e, n, ng).contiguous())


def fold_for_grid(pre):
    """Fold raw u4 pairs to the signed-nibble int8 bytes the grid reads."""
    return implement_zp(pre.contiguous()).view(torch.int8)


def dequant_asym_pre(pre, s, zps, dtype):
    """Zp-aware dequant straight from raw u4 pairs: (q - zp) * s."""
    e, n, kh = pre.shape
    ng = s.shape[2]
    gs = kh * 2 // ng
    u = pre.to(torch.uint8)
    q = torch.stack([u & 0xF, (u >> 4) & 0xF], dim=-1).float()
    q = q.reshape(e, n, kh * 2)
    zp = zps.repeat_interleave(gs, dim=2).float()
    return ((q - zp) * s.repeat_interleave(gs, dim=2)).to(dtype)


def ref_routed(x, deq, topk_weights, topk_ids, inter):
    """Fp32 CPU reference for routed-only int4 MoE (any top-k)."""
    w13, w2 = deq[0].float(), deq[1].float()
    xf = x.float().cpu()
    tw = topk_weights.float().cpu()
    out = torch.zeros_like(xf)
    for m in range(x.shape[0]):
        for k in range(topk_ids.shape[1]):
            e = int(topk_ids[m, k])
            if e < 0:
                continue
            h = w13[e] @ xf[m]
            act = torch.nn.functional.silu(h[:inter]) * h[inter:]
            out[m] += tw[m, k] * (w2[e] @ act)
    return out.to(x.device)


def make_weights(inter=INTER, dtype=torch.float16, device=DEVICE):
    dev = torch.device(device)
    w13f = torch.randn(NUM_EXPERTS, 2 * inter, H, device=dev) / 32
    w2f = torch.randn(NUM_EXPERTS, H, inter, device=dev) / 32
    (pre13, s13, z13) = quant_asym(w13f, GROUP_SIZE, dtype)
    (pre2, s2, z2) = quant_asym(w2f, GROUP_SIZE, dtype)
    deq = [dequant_asym_pre(pre13, s13, z13, dtype).cpu(),
           dequant_asym_pre(pre2, s2, z2, dtype).cpu(), None, None]
    return (pre13, s13, z13, pre2, s2, z2, deq)


def route(m, e=NUM_EXPERTS, top_k=TOP_K, device=DEVICE):
    logits = torch.randn(m, e, device=device)
    probs = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(probs, top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.contiguous(), topk_ids.to(torch.int32).contiguous()


def make_io(m, top_k=TOP_K, dtype=torch.float16):
    dev = torch.device(DEVICE)
    x = (torch.randn(m, H, device=dev, dtype=dtype) / 4).contiguous()
    out = torch.empty(m, H, device=dev, dtype=dtype).contiguous()
    tw = (torch.rand(m, top_k, device=dev,
                     dtype=torch.float32) + 0.1).contiguous()
    tw = (tw / tw.sum(dim=-1, keepdim=True)).contiguous()
    ti = torch.randint(0, NUM_EXPERTS, (m, top_k), device=dev,
                       dtype=torch.int32).contiguous()
    return out, x, tw, ti


def awq_pack_logical(q):
    """Build true AWQ-order int32 from logical [K, N] u4 (CPU)."""
    from vllm_xpu_kernels.quantization._quantize_convert import AWQUtils
    perm = AWQUtils.apply_order(q, direction="column",
                                order=AWQUtils.AWQ_PACK_ORDER)
    return AWQUtils.pack(perm, direction="column")


# CPU legs: repack round-trips need no device or built op.
@pytest.mark.parametrize("group_size", [32, 128])
def test_repack_awq_roundtrip_cpu(group_size):
    seed_everything(11)
    e, k, n = 2, 256, 192
    ng = k // group_size
    q = torch.randint(0, 16, (e, k, n), dtype=torch.int8)
    zp = torch.randint(0, 16, (e, ng, n), dtype=torch.int8)
    sc = (torch.rand(e, ng, n, dtype=torch.float32) * 0.04 + 0.01)
    qw = torch.stack([awq_pack_logical(q[i]) for i in range(e)])
    qz = torch.stack([awq_pack_logical(zp[i]) for i in range(e)])
    (w, s, z) = repack_awq_moe_for_grid(qw, qz, sc, group_size)
    assert w.shape == (e, n, k // 2) and w.dtype == torch.int8
    assert s.shape == (e, n, ng) and s.dtype == sc.dtype
    assert z.shape == (e, n, ng) and z.dtype == torch.uint8
    assert w.is_contiguous() and s.is_contiguous() and z.is_contiguous()
    for i in range(e):
        qt = q[i].t().contiguous().to(torch.uint8)
        packed = qt[..., 0::2] | (qt[..., 1::2] << 4)
        assert torch.equal(w[i].cpu(),
                           implement_zp(packed.contiguous()).to(torch.int8))
        assert torch.equal(s[i].cpu(), sc[i].t().contiguous())
        assert torch.equal(z[i].cpu(), zp[i].t().contiguous().to(torch.uint8))


def test_repack_awq_rejects_malformed_cpu():
    seed_everything(12)
    e, k, n, ng = 1, 128, 64, 1
    qw = torch.zeros(e, k, n // 8, dtype=torch.int32)
    qz = torch.zeros(e, ng, n // 8, dtype=torch.int32)
    sc = torch.zeros(e, ng, n, dtype=torch.float16)
    with pytest.raises(ValueError):
        repack_awq_moe_for_grid(qw, qz, sc, 96)
    with pytest.raises(ValueError):
        repack_awq_moe_for_grid(qw, qz[:, :, :1], sc, k // ng)
    with pytest.raises(ValueError):
        repack_awq_moe_for_grid(qw[:, :, :3], qz, sc, k // ng)


def test_repack_awq_header_shapes_cpu():
    """Real-header shapes pin the synthetic geometry (opt-in, read-only).

    Set VLLM_XPU_TEST_AWQ_HEADER to a safetensors file; the test reads
    only its JSON header, rebuilds one expert linear synthetically at
    those K/N/G shapes, and checks the repack accepts them.
    """
    path = os.environ.get("VLLM_XPU_TEST_AWQ_HEADER")
    if not path:
        pytest.skip("VLLM_XPU_TEST_AWQ_HEADER unset")
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(size))
    triple = None
    for key in hdr:
        if key.endswith(".qweight") and ".experts." in key:
            stem = key[: -len(".qweight")]
            if stem + ".qzeros" in hdr and stem + ".scales" in hdr:
                triple = stem
                break
    assert triple is not None, "no AWQ MoE expert triple in header"
    k, np8 = hdr[triple + ".qweight"]["shape"]
    ng, _ = hdr[triple + ".qzeros"]["shape"]
    assert hdr[triple + ".scales"]["shape"] == [ng, np8 * 8]
    group_size = k // ng
    qw = torch.zeros(1, k, np8, dtype=torch.int32)
    qz = torch.zeros(1, ng, np8, dtype=torch.int32)
    sc = torch.zeros(1, ng, np8 * 8, dtype=torch.float16)
    (w, s, z) = repack_awq_moe_for_grid(qw, qz, sc, group_size)
    assert w.shape == (1, np8 * 8, k // 2)
    assert s.shape == (1, np8 * 8, ng) and z.shape == (1, np8 * 8, ng)


def test_check_zps_rejects_mixed_cpu():
    seed_everything(13)
    e, n, ng = 2, 32, 2
    ok = torch.zeros(e, n, ng, dtype=torch.uint8, device="cpu")
    sc = torch.zeros(e, n, ng, dtype=torch.float16, device="cpu")
    with pytest.raises(ValueError):
        interface._check_zps(False, ok, None, None, None, sc, sc, None,
                             None)
    with pytest.raises(ValueError):
        interface._check_zps(False, None, None, ok, None, sc, sc, None,
                             None)
    bad = torch.zeros(e, n, ng, dtype=torch.int8, device="cpu")
    with pytest.raises(ValueError):
        interface._check_zps(False, bad, bad, None, None, sc, sc, None,
                             None)
    big = torch.full((e, n, ng), 16, dtype=torch.uint8, device="cpu")
    with pytest.raises(ValueError):
        interface._check_zps(False, big, big, None, None, sc, sc, None,
                             None)
    interface._check_zps(False, None, None, None, None, sc, sc, None,
                         None)
    interface._check_zps(False, ok, ok, None, None, sc, sc, None, None)


# XPU legs: need the built op.
@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("tile_m", [0, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@requires_moe_kernel
@requires_xpu
def test_zp8_bitwise_identical_to_no_zp(m, tile_m, dtype):
    """The no-drift proof: zp-8 planes reproduce today's bytes exactly."""
    seed_everything(7)
    weights = make_weights(dtype=dtype)
    (w13, s13, _, w2, s2, _, _) = weights
    f13, f2 = fold_for_grid(w13), fold_for_grid(w2)
    z8_13 = torch.empty_like(s13, dtype=torch.uint8).fill_(8)
    z8_2 = torch.empty_like(s2, dtype=torch.uint8).fill_(8)
    op_zp = GridOp(f13, s13, f2, s2, None, None, None, None, None,
                   TOP_K, tile_m=tile_m, w13_zps=z8_13, w2_zps=z8_2)
    op_plain = GridOp(f13, s13, f2, s2, None, None, None, None, None,
                      TOP_K, tile_m=tile_m)
    x = (torch.randn(m, H, device=DEVICE) / 4).to(dtype)
    topk_weights, topk_ids = route(m)
    out_zp = op_zp.forward(torch.empty_like(x), x, topk_weights, topk_ids)
    out_plain = op_plain.forward(torch.empty_like(x), x, topk_weights,
                                 topk_ids)
    assert torch.equal(out_zp, out_plain)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@requires_moe_kernel
@requires_xpu
def test_zp_random_vs_fp32_ref(dtype):
    seed_everything(21)
    weights = make_weights(dtype=dtype)
    (w13, s13, z13, w2, s2, z2, deq) = weights
    assert not bool((z13 == 8).all() and (z2 == 8).all())
    op = GridOp(fold_for_grid(w13), s13, fold_for_grid(w2), s2, None,
                None, None, None, None, TOP_K, w13_zps=z13, w2_zps=z2)
    x = (torch.randn(2, H, device=DEVICE) / 4).to(dtype)
    topk_weights, topk_ids = route(2)
    out = op.forward(torch.empty_like(x), x, topk_weights, topk_ids)
    ref = ref_routed(x, deq, topk_weights, topk_ids, INTER)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


@requires_moe_kernel
@requires_xpu
def test_zp_grid_taken_matches_grouped(monkeypatch):
    """Grid and grouped GEMM agree on the same asymmetric weights."""
    seed_everything(22)
    weights = make_weights()
    (w13, s13, z13, w2, s2, z2, deq) = weights
    op = GridOp(fold_for_grid(w13), s13, fold_for_grid(w2), s2, None,
                None, None, None, None, TOP_K, w13_zps=z13, w2_zps=z2)
    out, x, tw, ti = make_io(2)
    out_grid = op.forward(torch.empty_like(x), x, tw, ti)
    ref = ref_routed(x, deq, tw, ti, INTER)
    torch.testing.assert_close(out_grid.float(), ref, rtol=1e-2,
                               atol=1e-2)
    monkeypatch.setenv(INT4_GRID_ENV, "0")
    moe = XpuFusedMoe(w13, s13, None, w2, s2, None, TOP_K, "silu",
                      NUM_EXPERTS, w13_zps=z13, w2_zps=z2)
    out_fallback = torch.empty_like(x)
    moe.apply(out_fallback, x, tw, ti)
    torch.testing.assert_close(out_grid.float(), out_fallback.float(),
                               rtol=1e-2, atol=1e-2)


@requires_moe_kernel
@requires_xpu
def test_zp_mixed_shared_rejected():
    """Mixed shared setups fail with ValueError, not an AttributeError."""
    seed_everything(30)
    weights = make_weights()
    (w13, s13, z13, w2, s2, z2, _) = weights
    n1, k1, n2, k2 = w13.shape[1], w13.shape[2], w2.shape[1], w2.shape[2]
    ng = s13.shape[2]
    sw13 = torch.zeros(n1, k1, dtype=torch.int8, device=DEVICE)
    ss13 = torch.zeros(1, n1, ng, dtype=torch.float16, device=DEVICE)
    sw2 = torch.zeros(n2, k2, dtype=torch.int8, device=DEVICE)
    ss2 = torch.zeros(1, n2, ng, dtype=torch.float16, device=DEVICE)
    gate = torch.zeros(1, H, dtype=torch.float16, device=DEVICE)
    with pytest.raises(ValueError):
        GridOp(fold_for_grid(w13), s13, fold_for_grid(w2), s2, sw13,
               ss13, sw2, ss2, gate, TOP_K, w13_zps=z13, w2_zps=z2)
    with pytest.raises(ValueError):
        GridOp(fold_for_grid(w13), s13, fold_for_grid(w2), s2, sw13,
               ss13, sw2, ss2, gate, TOP_K,
               shared_w13_zps=torch.zeros_like(ss13, dtype=torch.uint8),
               shared_w2_zps=torch.zeros_like(ss2, dtype=torch.uint8))


@requires_xpu
def test_zp_fallback_honors_zps_rows():
    """Too many rows miss the grid; the fallback still applies zps."""
    seed_everything(24)
    weights = make_weights()
    (w13, s13, z13, w2, s2, z2, deq) = weights
    assert not bool((z13 == 8).all() and (z2 == 8).all())
    # Fold before XpuFusedMoe rewrites the pairs in place below.
    blind = [dequant(fold_for_grid(w13), s13,
                     torch.float16).cpu(),
             dequant(fold_for_grid(w2), s2, torch.float16).cpu()]
    moe = XpuFusedMoe(w13, s13, None, w2, s2, None, TOP_K, "silu",
                      NUM_EXPERTS, w13_zps=z13, w2_zps=z2)
    out, x, tw, ti = make_io(INT4_GRID_MAX_M + 2)
    assert not moe._int4_grid_eligible(out, x, tw, ti)
    moe.apply(out, x, tw, ti)
    ref = ref_routed(x, deq, tw, ti, INTER)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
    ref_blind = ref_routed(x, blind, tw, ti, INTER)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out.float(), ref_blind)


@requires_xpu
def test_zp_fallback_honors_zps_env_off(monkeypatch):
    """Grid env off forces the fallback; zps must still apply."""
    seed_everything(25)
    monkeypatch.setenv(INT4_GRID_ENV, "0")
    weights = make_weights()
    (w13, s13, z13, w2, s2, z2, deq) = weights
    moe = XpuFusedMoe(w13, s13, None, w2, s2, None, TOP_K, "silu",
                      NUM_EXPERTS, w13_zps=z13, w2_zps=z2)
    out, x, tw, ti = make_io(2)
    assert not moe._int4_grid_eligible(out, x, tw, ti)
    moe.apply(out, x, tw, ti)
    ref = ref_routed(x, deq, tw, ti, INTER)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


def test_zp_grid_eligible_rejects_mixed_cpu():
    """Mixed zp presence never takes the grid (device-independent)."""
    seed_everything(26)
    dev = torch.device("cpu")
    e, h, inter, ng = 2, 256, 64, 2
    w13 = torch.zeros(e, 2 * inter, h // 2, dtype=torch.uint8, device=dev)
    s13 = torch.zeros(e, 2 * inter, ng, dtype=torch.float16, device=dev)
    w2 = torch.zeros(e, h, inter // 2, dtype=torch.uint8, device=dev)
    s2 = torch.zeros(e, h, ng, dtype=torch.float16, device=dev)
    z13 = torch.zeros(e, 2 * inter, ng, dtype=torch.uint8, device=dev)
    z2 = torch.zeros(e, h, ng, dtype=torch.uint8, device=dev)
    moe = XpuFusedMoe(w13, s13, None, w2, s2, None, TOP_K, "silu", e,
                      w13_zps=z13, w2_zps=None)
    x = torch.zeros(2, h, dtype=torch.float16, device=dev)
    out = torch.zeros(2, h, dtype=torch.float16, device=dev)
    tw = torch.ones(2, TOP_K, dtype=torch.float32, device=dev) / TOP_K
    ti = torch.zeros(2, TOP_K, dtype=torch.int32, device=dev)
    assert not moe._int4_grid_eligible(out, x, tw, ti)
    moe_ok = XpuFusedMoe(w13, s13, None, w2, s2, None, TOP_K, "silu", e,
                         w13_zps=z13, w2_zps=z2)
    bad = torch.zeros(e, h, ng + 1, dtype=torch.uint8, device=dev)
    moe_ok.w2_zps = bad
    assert not moe_ok._int4_grid_eligible(out, x, tw, ti)
