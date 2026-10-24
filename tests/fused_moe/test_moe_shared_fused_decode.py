# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import vllm_xpu_kernels.moe_shared_fused_interface as interface
from tests.utils import seed_everything
from vllm_xpu_kernels.moe_shared_fused_interface import (
    XpuMoESharedFusedDecode, is_available, moe_shared_fused_workspace_bytes)

DEVICE = "xpu"
HIDDEN = 2048
NUM_EXPERTS = 256
TOP_K = 8

requires_moe_kernel = pytest.mark.skipif(
    not is_available(), reason="moe_shared_fused_decode op not built")


def quant_fp8_per_tensor(w: torch.Tensor):
    """Per-expert (3D) or per-tensor (2D) e4m3 quantization."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    if w.dim() == 3:
        scale = w.abs().amax(dim=(1, 2)) / fp8_max
        q = (w / scale[:, None, None]).to(torch.float8_e4m3fn)
        return q, scale.float(), q.float() * scale[:, None, None]
    scale = (w.abs().max() / fp8_max).reshape(1)
    q = (w / scale).to(torch.float8_e4m3fn)
    return q, scale.float(), q.float() * scale


def make_weights(inter: int, dtype=torch.float16):
    w13f = torch.randn(NUM_EXPERTS, HIDDEN, 2 * inter, device=DEVICE) / 32
    w2f = torch.randn(NUM_EXPERTS, inter, HIDDEN, device=DEVICE) / 32
    sw13f = torch.randn(HIDDEN, 2 * inter, device=DEVICE) / 32
    sw2f = torch.randn(inter, HIDDEN, device=DEVICE) / 32
    gate = (torch.randn(1, HIDDEN, device=DEVICE) / 32).to(dtype)
    return [quant_fp8_per_tensor(w) for w in (w13f, w2f, sw13f, sw2f)] + [gate]


def route(m: int):
    logits = torch.randn(m, NUM_EXPERTS, device=DEVICE)
    probs = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(probs, TOP_K, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.contiguous(), topk_ids.to(torch.int32).contiguous()


def ref_moe_shared(x, weights, topk_weights, topk_ids, inter):
    """FP32 routed MoE + sigmoid-gated shared expert on dequantized weights.

    Intermediate projections and activations are rounded to the input dtype.
    """
    (_, _, w13), (_, _, w2), (_, _, sw13), (_, _, sw2), gate = weights
    xf = x.float()
    out = torch.zeros_like(xf)
    for m in range(x.shape[0]):
        for k in range(TOP_K):
            e = int(topk_ids[m, k])
            h = (xf[m] @ w13[e]).to(x.dtype).float()
            act = torch.nn.functional.silu(h[:inter]) * h[inter:]
            projection = (act.to(x.dtype).float() @ w2[e]).to(x.dtype).float()
            out[m] += topk_weights[m, k] * projection
    h = (xf @ sw13).to(x.dtype).float()
    act = torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]
    projection = (act.to(x.dtype).float() @ sw2).to(x.dtype).float()
    out += torch.sigmoid(xf @ gate.float().t()) * projection
    return out


def make_op(weights, tile_m=0):
    (w13, s13, _), (w2, s2, _), (sw13, ss13, _), (sw2, ss2, _), gate = weights
    return XpuMoESharedFusedDecode(w13, s13, w2, s2, sw13, ss13, sw2, ss2,
                                   gate, TOP_K, tile_m=tile_m)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("inter", [256, 512])
@pytest.mark.parametrize("tile_m", [0, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@requires_moe_kernel
def test_moe_shared_fused_decode(m, inter, tile_m, dtype):
    seed_everything(7)
    weights = make_weights(inter, dtype=dtype)
    op = make_op(weights, tile_m)
    x = (torch.randn(m, HIDDEN, device=DEVICE) / 4).to(dtype)
    topk_weights, topk_ids = route(m)

    out = op.forward(torch.empty_like(x), x, topk_weights, topk_ids)
    ref = ref_moe_shared(x, weights, topk_weights, topk_ids, inter)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@requires_moe_kernel
def test_moe_shared_fused_decode_int64_ids(dtype):
    seed_everything(7)
    weights = make_weights(256, dtype=dtype)
    op = make_op(weights)
    x = (torch.randn(4, HIDDEN, device=DEVICE) / 4).to(dtype)
    topk_weights, topk_ids = route(4)
    out32 = op.forward(torch.empty_like(x), x, topk_weights, topk_ids)
    out64 = op.forward(torch.empty_like(x), x, topk_weights,
                       topk_ids.long().contiguous())
    torch.testing.assert_close(out32, out64, rtol=0, atol=0)


@requires_moe_kernel
def test_moe_shared_fused_decode_rejects_float32():
    seed_everything(7)
    weights = make_weights(256)
    op = make_op(weights)
    x = torch.randn(2, HIDDEN, device=DEVICE, dtype=torch.float32)
    topk_weights, topk_ids = route(2)
    with pytest.raises(RuntimeError, match="fp16/bf16 activations"):
        op.forward(torch.empty_like(x), x, topk_weights, topk_ids)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@requires_moe_kernel
def test_moe_shared_fused_decode_workspace_size(dtype):
    seed_everything(7)
    inter = 256
    weights = make_weights(inter, dtype=dtype)
    (w13, s13, _), (w2, s2, _), (sw13, ss13, _), (sw2, ss2, _), gate = weights
    m = 3
    x = (torch.randn(m, HIDDEN, device=DEVICE) / 4).to(dtype)
    topk_weights, topk_ids = route(m)
    nbytes = moe_shared_fused_workspace_bytes(m, TOP_K, NUM_EXPERTS, HIDDEN,
                                              2 * inter, inter)

    def run(ws):
        torch.ops._xpu_C.moe_shared_fused_decode_interface(
            torch.empty_like(x), x, topk_ids, topk_weights, w13, s13, w2, s2,
            sw13, ss13, sw2, ss2, gate.reshape(-1), ws, 0)

    run(torch.empty(nbytes, dtype=torch.uint8, device=DEVICE))
    with pytest.raises(RuntimeError, match="workspace too small"):
        run(torch.empty(nbytes - 1, dtype=torch.uint8, device=DEVICE))


@requires_moe_kernel
def test_moe_shared_fused_decode_rejects_mixed_shared_gate_dtype():
    seed_everything(7)
    weights = make_weights(256, dtype=torch.float16)
    op = make_op(weights)
    x = torch.randn(1, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
    topk_weights, topk_ids = route(1)
    with pytest.raises(RuntimeError, match="shared_gate.*input dtype"):
        op.forward(torch.empty_like(x), x, topk_weights, topk_ids)


@pytest.mark.parametrize("dtype,supported", [(torch.float16, True),
                                             (torch.bfloat16, True),
                                             (torch.float32, False),
                                             (torch.float64, False)])
def test_moe_activation_dtype_contract(monkeypatch, dtype, supported):
    monkeypatch.setattr(interface, "is_available", lambda: True)
    assert interface.supports(dtype, torch.float8_e4m3fn, 1, 8, 2048,
                              256) is supported
