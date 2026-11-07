# SPDX-License-Identifier: Apache-2.0
"""Ornith AutoRound -> grid repack adapter tests.

CPU-runnable (no XPU build needed): synthetic symmetric-AutoRound tensors
at Ornith N/K/G slices plus a small shape, proving adapter dequant equals
the stock-loader dequant exactly, including the g_idx-permuted path and
all rejection paths. The XPU wiring test at the end builds a real routed
instance via from_autoround at small dims.
"""
import pytest
import torch

from vllm_xpu_kernels.moe_shared_fused_int4_interface import (
    autoround_to_grid, is_available)
from tests.utils import seed_everything

requires_moe_kernel = pytest.mark.skipif(
    not is_available(), reason="moe_shared_fused_decode_int4 op not built")


def make_autoround(n, k, g=128, permute_groups=False):
    """Synthetic symmetric-AutoRound linear: q in 0..15, scales, 7-zeros."""
    assert k % g == 0
    ng = k // g
    qw = torch.randint(0, 2 ** 31, (k // 8, n), dtype=torch.int32)
    scales = (torch.randn(ng, n) / 16).to(torch.float16)
    qzeros = torch.full((ng, n // 8), 0x77777777, dtype=torch.int32)
    if permute_groups:
        # swap the first two groups: a genuine non-identity permutation
        g2 = (torch.arange(k) // g).reshape(ng, g).clone()
        g2[[0, 1]] = g2[[1, 0]]
        g_idx = g2.reshape(k).to(torch.int32)
    else:
        g_idx = (torch.arange(k) // g).to(torch.int32)
    return qw, scales, qzeros, g_idx


def stock_dequant(qw, scales, g_idx, g=128):
    """Group-aware stock dequant: (u - 8) * s(g_idx[k])."""
    k = qw.shape[0] * 8
    n = qw.shape[1]
    u8 = qw.T.contiguous().view(torch.uint8).reshape(n, k // 2)
    lo = (u8 & 0xF).to(torch.float32)
    hi = ((u8 >> 4) & 0xF).to(torch.float32)
    u = torch.stack([lo, hi], dim=-1).reshape(n, k)
    s = scales.float()[g_idx.long()].T  # [N, K]
    return (u - 8.0) * s


def grid_dequant(w, s, g=128):
    lead = w.shape[:-2]
    w2 = w.reshape(-1, w.shape[-2], w.shape[-1])
    s2 = s.reshape(-1, s.shape[-2], s.shape[-1])
    u = w2.to(torch.uint8)
    lo = (u & 0xF).to(torch.int32)
    hi = ((u >> 4) & 0xF).to(torch.int32)
    q = torch.stack([((v & 0x7) - ((v >> 3) & 0x1) * 8).to(torch.float32)
                     for v in (lo, hi)], dim=-1).reshape(w2.shape[0],
                                                        w2.shape[1], -1)
    return (q * s2.float().repeat_interleave(g, dim=2)).reshape(
        *lead, w2.shape[1], w2.shape[2] * 2)


@pytest.mark.parametrize("n,k", [(2048, 512), (512, 2048), (64, 256)])
def test_autoround_adapter_round_trip(n, k):
    seed_everything(11)
    qw, sc, qz, gi = make_autoround(n, k)
    w, s = autoround_to_grid(qw, sc, qz, gi)
    assert w.shape == (n, k // 2) and w.dtype == torch.int8
    assert s.shape == (n, k // 128) and s.dtype == torch.float16
    a = grid_dequant(w, s)
    b = stock_dequant(qw, sc, gi)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_autoround_adapter_permuted_g_idx():
    seed_everything(12)
    n, k = 256, 512
    qw, sc, qz, gi = make_autoround(n, k, permute_groups=True)
    assert not torch.equal(gi, torch.arange(k) // 128)
    w, s = autoround_to_grid(qw, sc, qz, gi)
    order = torch.argsort(gi, stable=True)
    # stock on the K-sorted weight equals the adapter output
    u8 = qw.T.contiguous().view(torch.uint8).reshape(n, k // 2)
    lo = (u8 & 0xF).to(torch.float32)
    hi = ((u8 >> 4) & 0xF).to(torch.float32)
    u = torch.stack([lo, hi], dim=-1).reshape(n, k)[:, order]
    s_sorted = sc.float()[gi[order].long()].T
    torch.testing.assert_close(grid_dequant(w, s), (u - 8.0) * s_sorted,
                               rtol=0, atol=0)


def test_autoround_adapter_rejects():
    seed_everything(13)
    qw, sc, qz, gi = make_autoround(256, 512)
    with pytest.raises(ValueError, match="symmetric"):
        autoround_to_grid(qw, sc, torch.zeros_like(qz), gi)
    with pytest.raises(ValueError, match="int32"):
        autoround_to_grid(qw.float(), sc, qz, gi)
    with pytest.raises(ValueError, match="fp16 or bf16"):
        autoround_to_grid(qw, sc.float(), qz, gi)
    with pytest.raises(ValueError, match="group size"):
        autoround_to_grid(qw[:7], sc[:1], qz[:1], gi[:56])
    with pytest.raises(ValueError, match="g_idx"):
        autoround_to_grid(qw, sc, qz, gi[:100])
    with pytest.raises(ValueError, match="scales"):
        autoround_to_grid(qw, sc[:, :100], qz, gi)


def test_from_autoround_rejects_non_xpu_device():
    from vllm_xpu_kernels.moe_shared_fused_int4_interface import (
        XpuMoESharedFusedDecodeInt4)
    seed_everything(14)
    qw, sc, qz, gi = make_autoround(64, 256)
    e1 = lambda t: t.unsqueeze(0)
    with pytest.raises(ValueError, match="XPU"):
        XpuMoESharedFusedDecodeInt4.from_autoround(
            e1(qw), e1(sc), e1(qz), gi, e1(qw), e1(sc), e1(qz), gi,
            e1(qw), e1(sc), e1(qz), gi, top_k=1)
    with pytest.raises(ValueError, match="all-or-none"):
        XpuMoESharedFusedDecodeInt4.from_autoround(
            e1(qw), e1(sc), e1(qz), gi, e1(qw), e1(sc), e1(qz), gi,
            e1(qw), e1(sc), e1(qz), gi,
            shared_gate_qweight=qw, top_k=1)


@requires_moe_kernel
def test_from_autoround_routed_forward_small():
    from vllm_xpu_kernels.moe_shared_fused_int4_interface import (
        XpuMoESharedFusedDecodeInt4, supports)
    assert supports(torch.float16, 2, 256, 64, 32) is True
    assert supports(torch.float16, 8, 2048, 512, 128) is True  # Ornith dims
    seed_everything(15)
    dev = torch.device("xpu")
    e, h, i, g, topk = 8, 256, 64, 32, 2
    grp = lambda n, k: make_autoround(n, k, g)
    gqw, gsc, gqz, ggi = grp(i, h)
    uqw, usc, uqz, ugi = grp(i, h)
    dqw, dsc, dqz, dgi = grp(h, i)
    stack = lambda t: t.unsqueeze(0).expand(e, -1, -1).contiguous()
    op = XpuMoESharedFusedDecodeInt4.from_autoround(
        stack(gqw), stack(gsc), gqz, ggi,
        stack(uqw), stack(usc), uqz, ugi,
        stack(dqw), stack(dsc), dqz, dgi, top_k=topk, device=dev)
    assert op.w13.shape == (e, 2 * i, h // 2)
    assert op.w2.shape == (e, h, i // 2)
    m = 2
    x = (torch.randn(m, h, device=dev) / 4).to(torch.float16)
    tw = torch.full((m, topk), 1.0 / topk, device=dev)
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=dev)
    out = op.forward(torch.empty_like(x), x, tw, ids)
    # reference: dequant + silu-mul MoE on CPU
    w13d = grid_dequant(op.w13.cpu(), op.w13_scales.cpu(), g).float()
    w2d = grid_dequant(op.w2.cpu(), op.w2_scales.cpu(), g).float()
    ref = torch.zeros_like(x.float().cpu())
    xf = x.float().cpu()
    for r in range(m):
        for kk in range(topk):
            ex = int(ids[r, kk])
            hh = (w13d[ex].to(x.dtype).float() @ xf[r]).to(x.dtype).float()
            act = torch.nn.functional.silu(hh[:i]) * hh[i:]
            ref[r] += tw[r, kk].float().cpu() * (
                w2d[ex].to(x.dtype).float() @ act.to(x.dtype).float())
    torch.testing.assert_close(out.float().cpu(), ref, rtol=1e-2, atol=1e-2)
