# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import vllm_xpu_kernels.moe_shared_fused_int4_interface as interface
from tests.utils import seed_everything
from vllm_xpu_kernels.moe_shared_fused_int4_interface import (
    XpuMoESharedFusedDecodeInt4, is_available,
    moe_shared_fused_int4_workspace_bytes)

DEVICE = "xpu"
HIDDEN = 2048
NUM_EXPERTS = 256
TOP_K = 8
GROUP_SIZE = 128

requires_moe_kernel = pytest.mark.skipif(
    not is_available(), reason="moe_shared_fused_decode_int4 op not built")


def quant_int4_per_group(w: torch.Tensor, group_size: int, dtype):
    """Symmetric per-group int4 quant; w is [E, N, K], packs K.

    Returns int8-packed [E, N, K/2] (low nibble = first K element) and
    scales [E, N, K/group_size]. Packing mirrors implement_zp, which
    produces plain 4-bit two's-complement nibbles.
    """
    assert w.dim() == 3
    e, n, k = w.shape
    assert k % group_size == 0
    ng = k // group_size
    wf = w.float().reshape(e, n, ng, group_size)
    amax = wf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / 7).to(dtype)
    q = (wf / scale).round().clamp(-8, 7).to(torch.int8)
    # Low 4 bits of the int8 byte = the value's 4-bit two's-complement
    # encoding (for -8..7, bit 3 of the byte already is the sign).
    u = (q.to(torch.uint8) & 0xF).to(torch.uint8)
    packed = (u[..., 0::2] | (u[..., 1::2] << 4)).to(torch.uint8).reshape(
        e, n, ng * group_size // 2).to(torch.int8)
    assert packed.shape == (e, n, k // 2)
    return (packed.contiguous(), scale.reshape(e, n, ng).contiguous())


def dequant(wp, s, dtype):
    """wp [E, N, K/2] int8, s [E, N, ng] -> [E, N, K] in dtype."""
    e, n, kh = wp.shape
    ng = s.shape[2]
    gs = kh * 2 // ng
    u = wp.to(torch.uint8)  # [E, N, K/2]
    lo = u & 0xF
    hi = (u >> 4) & 0xF
    # Nibbles are plain 4-bit two's complement (0x0->0 .. 0x7->7,
    # 0x8->-8 .. 0xF->-1); implement_zp produces exactly this encoding,
    # so pack/dequant round-trip.
    def tc(v):
        return (v & 0x7).to(torch.float32) - ((v >> 3) & 0x1).to(torch.float32) * 8.0
    q = torch.stack([tc(lo), tc(hi)], dim=-1).reshape(e, n, kh * 2)
    return (q.float() * s.repeat_interleave(gs, dim=2)).to(dtype)


def make_weights(inter: int, dtype=torch.float16):
    # [E, N, K] storage order, K packed (same convention as the XPU path).
    w13f = torch.randn(NUM_EXPERTS, 2 * inter, HIDDEN, device=DEVICE) / 32
    w2f = torch.randn(NUM_EXPERTS, HIDDEN, inter, device=DEVICE) / 32
    sw13f = torch.randn(2 * inter, HIDDEN, device=DEVICE) / 32
    sw2f = torch.randn(HIDDEN, inter, device=DEVICE) / 32
    gate = (torch.randn(1, HIDDEN, device=DEVICE) / 32).to(dtype)
    (w13, s13) = quant_int4_per_group(w13f, GROUP_SIZE, dtype)
    (w2, s2) = quant_int4_per_group(w2f, GROUP_SIZE, dtype)
    (sw13, ss13) = quant_int4_per_group(sw13f.unsqueeze(0), GROUP_SIZE, dtype)
    (sw2, ss2) = quant_int4_per_group(sw2f.unsqueeze(0), GROUP_SIZE, dtype)
    deq = [dequant(w, s, dtype) for w, s in
           ((w13, s13), (w2, s2), (sw13, ss13), (sw2, ss2))]
    return (w13, s13, w2, s2, sw13[0], ss13, sw2[0], ss2, gate, deq)


def route(m: int):
    logits = torch.randn(m, NUM_EXPERTS, device=DEVICE)
    probs = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(probs, TOP_K, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.contiguous(), topk_ids.to(torch.int32).contiguous()


def ref_moe_shared(x, deq, gate, topk_weights, topk_ids, inter, with_shared=True):
    # Reference runs on CPU: oneDNN has no fp32 XPU primitive for some
    # shapes, so an XPU fp32 reference dies in torch dispatch instead of
    # testing anything. Numerics are unaffected (fp32 either way).
    dev = x.device
    w13, w2, sw13, sw2 = [t.float().cpu() if t is not None else None for t in deq]
    if with_shared:
        sw13, sw2 = sw13[0], sw2[0]
    gate = gate.float().cpu() if gate is not None else None
    tw = topk_weights.float().cpu()
    xf = x.float().cpu()
    out = torch.zeros_like(xf)
    for m in range(x.shape[0]):
        for k in range(TOP_K):
            e = int(topk_ids[m, k])
            if e < 0:
                continue  # Padding ID: contributes zero to routed experts.
            h = (w13[e].to(x.dtype).float() @ xf[m]).to(x.dtype).float()
            act = torch.nn.functional.silu(h[:inter]) * h[inter:]
            projection = (
                w2[e].to(x.dtype).float() @ act.to(x.dtype).float()
            ).to(x.dtype).float()
            out[m] += tw[m, k] * projection
    if with_shared:
        h = (sw13.to(x.dtype).float() @ xf.t()).t().to(x.dtype).float()
        act = torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]
        projection = (
            sw2.to(x.dtype).float() @ act.to(x.dtype).float().t()
        ).t().to(x.dtype).float()
        out += torch.sigmoid((gate @ xf.t()).t()) * projection
    return out.to(dev)


def make_op(weights, tile_m=0):
    (w13, s13, w2, s2, sw13, ss13, sw2, ss2, gate, _) = weights
    return XpuMoESharedFusedDecodeInt4(w13, s13, w2, s2, sw13, ss13, sw2,
                                       ss2, gate, TOP_K, tile_m=tile_m)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("inter", [256, 512])
@pytest.mark.parametrize("tile_m", [0, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@requires_moe_kernel
def test_moe_shared_fused_decode_int4(m, inter, tile_m, dtype):
    seed_everything(7)
    weights = make_weights(inter, dtype=dtype)
    op = make_op(weights, tile_m)
    x = (torch.randn(m, HIDDEN, device=DEVICE) / 4).to(dtype)
    topk_weights, topk_ids = route(m)
    out = op.forward(torch.empty_like(x), x, topk_weights, topk_ids)
    gate = weights[-2]
    ref = ref_moe_shared(x, weights[-1], gate, topk_weights, topk_ids, inter)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@requires_moe_kernel
def test_moe_shared_fused_decode_int4_workspace_size(dtype):
    seed_everything(7)
    inter = 256
    weights = make_weights(inter, dtype=dtype)
    (w13, s13, w2, s2, sw13, ss13, sw2, ss2, gate, _) = weights
    m = 3
    x = (torch.randn(m, HIDDEN, device=DEVICE) / 4).to(dtype)
    topk_weights, topk_ids = route(m)
    nbytes = moe_shared_fused_int4_workspace_bytes(
        m, TOP_K, HIDDEN, 2 * inter, inter)

    def run(ws):
        torch.ops._xpu_C.moe_shared_fused_decode_int4_interface(
            torch.empty_like(x), x, topk_ids, topk_weights, w13, s13, w2,
            s2, sw13, ss13, sw2, ss2, gate.reshape(-1), ws, 0)

    run(torch.empty(nbytes, dtype=torch.uint8, device=DEVICE))
    with pytest.raises(RuntimeError, match="workspace too small"):
        run(torch.empty(nbytes - 1, dtype=torch.uint8, device=DEVICE))


@requires_moe_kernel
def test_moe_shared_fused_decode_int4_rejects_fp8_weights():
    seed_everything(7)
    weights = make_weights(256)
    (w13, s13, w2, s2, sw13, ss13, sw2, ss2, gate, _) = weights
    op = make_op(weights)
    x = (torch.randn(2, HIDDEN, device=DEVICE) / 4).to(torch.float16)
    topk_weights, topk_ids = route(2)
    bad_w13 = torch.empty_like(w13, dtype=torch.float8_e4m3fn)
    with pytest.raises(RuntimeError, match="int8-packed"):
        torch.ops._xpu_C.moe_shared_fused_decode_int4_interface(
            torch.empty_like(x), x, topk_ids, topk_weights, bad_w13, s13,
            w2, s2, sw13, ss13, sw2, ss2, gate.reshape(-1),
            op.workspace(2), 0)


@pytest.mark.parametrize("m", [1, 2, 4, 8])
@pytest.mark.parametrize("tile_m", [0, 8, 16])
@requires_moe_kernel
def test_moe_shared_fused_decode_int4_routed_only(m, tile_m):
    # Routed-only via shared=None; the dense shared expert then runs as
    # a separate unfused MLP.
    seed_everything(7)
    dtype = torch.bfloat16
    inter = 512
    weights = make_weights(inter, dtype=dtype)
    (w13, s13, w2, s2, _, _, _, _, _, deq) = weights
    op = XpuMoESharedFusedDecodeInt4(
        w13, s13, w2, s2, None, None, None, None, None, TOP_K, tile_m=tile_m)
    x = (torch.randn(m, HIDDEN, device=DEVICE) / 4).to(dtype)
    topk_weights, topk_ids = route(m)
    out = op.forward(torch.empty_like(x), x, topk_weights, topk_ids)
    w13d, w2d, _, _ = deq
    ref = ref_moe_shared(x, (w13d, w2d, None, None), None, topk_weights,
                         topk_ids, inter, with_shared=False)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype,supported", [(torch.float16, True),
                                             (torch.bfloat16, True),
                                             (torch.float32, False)])
def test_moe_int4_activation_dtype_contract(dtype, supported):
    assert interface.supports(dtype, 8, 2048, 256, 128) is supported
    assert interface.supports(dtype, 8, 2048, 256, 7) is False


def test_moe_int4_supports_rejects_non_xpu_device():
    assert interface.supports(torch.float16, 8, 2048, 256, 128,
                              device=torch.device("cpu")) is False


@pytest.mark.parametrize("shape", [
    (0, 2048, 512, 256),
    (513, 2048, 512, 256),
    (8, 2048, 256, 7),
])
def test_moe_int4_workspace_rejects_invalid_dimensions(shape):
    with pytest.raises(ValueError):
        moe_shared_fused_int4_workspace_bytes(1, *shape)


@requires_moe_kernel
@pytest.mark.parametrize("m", [1, 2, 6])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_shared_fused_decode_int4_forward_routed(m, dtype):
    torch.manual_seed(0)
    inter = 512
    weights = make_weights(inter, dtype=dtype)
    (w13, s13, w2, s2, _, _, _, _, _, deq) = weights
    gatew = (torch.randn(NUM_EXPERTS, HIDDEN, device=DEVICE) * 0.02).to(dtype)
    # renormalize=False to match the plain softmax+topk reference
    # (vLLM passes its own renormalize flag at serve time).
    op = XpuMoESharedFusedDecodeInt4(
        w13, s13, w2, s2, None, None, None, None, None, TOP_K,
        router_weight=gatew, renormalize=False)
    x = (torch.randn(m, HIDDEN, device=DEVICE) / 4).to(dtype)
    # Reference routing on CPU: oneDNN has no fp32 XPU primitive for some
    # shapes, same reason ref_moe_shared runs on CPU below.
    ref_logits = torch.nn.functional.linear(x.float().cpu(),
                                            gatew.float().cpu())
    ref_w = torch.softmax(ref_logits, -1)
    ref_v, ref_i = torch.topk(ref_w, TOP_K, dim=-1)
    topk_weights, topk_ids = op.route(x)
    assert topk_weights.dtype == torch.float32
    assert topk_ids.dtype == torch.int32
    assert torch.equal(topk_ids.long().sort(-1)[0],
                       ref_i.to(topk_ids.device).sort(-1)[0])
    out, routed_ids = op.forward_routed(torch.empty_like(x), x)
    assert torch.equal(routed_ids, topk_ids)
    w13d, w2d, _, _ = deq
    ref = ref_moe_shared(x, (w13d, w2d, None, None), None, ref_v,
                         ref_i.to(torch.int32), inter, with_shared=False)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("m", [2, 8])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("all_padding", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@requires_moe_kernel
def test_moe_shared_fused_decode_int4_padding_routes(m, ids_dtype,
                                                     all_padding, dtype):
    """Padding IDs skip routed experts while preserving the shared expert."""
    seed_everything(7)
    inter = 256
    weights = make_weights(inter, dtype=dtype)
    op = make_op(weights)
    x = (torch.randn(m, HIDDEN, device=DEVICE) / 4).to(dtype)
    tw, ids = route(m)
    ids = ids.to(ids_dtype)
    # Reuse a populated workspace to catch stale permutation indices.
    op.forward(torch.empty_like(x), x, tw, ids)
    if all_padding:
        ids.fill_(-1)
    else:
        ids[0].fill_(-1)
        ids[:, 0] = -1
    out = op.forward(torch.empty_like(x), x, tw, ids)
    gate = weights[-2]
    ref = ref_moe_shared(x, weights[-1], gate, tw, ids, inter)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


@requires_moe_kernel
def test_moe_shared_fused_decode_int4_all_padding_routed_only():
    """All-padding with shared=None must read no expert and return zeros."""
    seed_everything(7)
    dtype = torch.bfloat16
    inter = 512
    weights = make_weights(inter, dtype=dtype)
    (w13, s13, w2, s2, _, _, _, _, _, _) = weights
    op = XpuMoESharedFusedDecodeInt4(
        w13, s13, w2, s2, None, None, None, None, None, TOP_K)
    m = 2
    x = (torch.randn(m, HIDDEN, device=DEVICE) / 4).to(dtype)
    tw, _ = route(m)
    ids = torch.full((m, TOP_K), -1, dtype=torch.int32, device=DEVICE)
    out = op.forward(torch.empty_like(x), x, tw, ids)
    torch.testing.assert_close(
        out.float(), torch.zeros_like(out.float()), rtol=1e-2, atol=1e-2)


@requires_moe_kernel
def test_moe_shared_fused_decode_int4_rejects_mixed_shared():
    seed_everything(7)
    weights = make_weights(256)
    (w13, s13, w2, s2, sw13, ss13, sw2, ss2, gate, _) = weights
    with pytest.raises(ValueError, match="all-or-none"):
        XpuMoESharedFusedDecodeInt4(
            w13, s13, w2, s2, sw13, ss13, sw2, ss2, None, TOP_K)
