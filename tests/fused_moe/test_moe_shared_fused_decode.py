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


def make_weights(inter: int, dtype=torch.float16, hidden=HIDDEN,
                 num_experts=NUM_EXPERTS):
    w13f = torch.randn(num_experts, hidden, 2 * inter, device=DEVICE) / 32
    w2f = torch.randn(num_experts, inter, hidden, device=DEVICE) / 32
    sw13f = torch.randn(hidden, 2 * inter, device=DEVICE) / 32
    sw2f = torch.randn(inter, hidden, device=DEVICE) / 32
    gate = (torch.randn(1, hidden, device=DEVICE) / 32).to(dtype)
    return [quant_fp8_per_tensor(w) for w in (w13f, w2f, sw13f, sw2f)] + [gate]


def route(m: int, num_experts: int = NUM_EXPERTS):
    logits = torch.randn(m, num_experts, device=DEVICE)
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
            if e < 0 or e >= w13.shape[0]:
                continue
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


@pytest.mark.parametrize("m", [1, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@requires_moe_kernel
def test_moe_shared_fused_decode_graph_replay_on_current_stream(m, dtype):
    """Replay on a non-default stream must consume new inputs and routing."""
    seed_everything(7)
    weights = make_weights(256, dtype=dtype)
    op = make_op(weights)
    stream = torch.xpu.Stream()
    stream.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(stream):
        x = (torch.randn(m, HIDDEN, device=DEVICE) / 4).to(dtype)
        topk_weights, topk_ids = route(m)
        out = torch.empty_like(x)
        # Allocate persistent workspace and initialize kernels before capture.
        op.forward(out, x, topk_weights, topk_ids)
    stream.synchronize()

    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph, stream=stream):
        op.forward(out, x, topk_weights, topk_ids)

    for _ in range(3):
        with torch.xpu.stream(stream):
            x.copy_((torch.randn_like(x.float()) / 4).to(dtype))
            new_weights, new_ids = route(m)
            topk_weights.copy_(new_weights)
            topk_ids.copy_(new_ids)
            out.fill_(float("nan"))
            graph.replay()
            ref = ref_moe_shared(x, weights, topk_weights, topk_ids, 256)
        stream.synchronize()
        torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@requires_moe_kernel
def test_moe_shared_fused_decode_gather_tail(dtype):
    """A legal H%16 shape need not fill a gather work-group."""
    seed_everything(7)
    hidden, inter, experts = 2064, 256, 8
    weights = make_weights(inter, dtype, hidden, experts)
    op = make_op(weights)
    x = (torch.randn(2, hidden, device=DEVICE) / 4).to(dtype)
    tw, ids = route(2, experts)
    out = op.forward(torch.empty_like(x), x, tw, ids)
    ref = ref_moe_shared(x, weights, tw, ids, inter)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tile_m", [0, 8, 16])
@requires_moe_kernel
def test_moe_shared_fused_decode_skewed_multi_tile_routes(dtype, tile_m):
    """Valid routes can require more tiles than the number of experts."""
    seed_everything(7)
    m, inter = 64, 256
    weights = make_weights(inter, dtype)
    op = make_op(weights, tile_m)
    x = (torch.randn(m, HIDDEN, device=DEVICE) / 4).to(dtype)
    row = torch.arange(m, device=DEVICE)[:, None]
    rest = 1 + (row * 7 + torch.arange(7, device=DEVICE)) % 255
    ids = torch.cat((torch.zeros(m, 1, device=DEVICE, dtype=torch.int64),
                     rest), dim=1).to(torch.int32).contiguous()
    tw = torch.full((m, TOP_K), 1 / TOP_K, device=DEVICE, dtype=torch.float32)
    out = op.forward(torch.empty_like(x), x, tw, ids)
    ref = ref_moe_shared(x, weights, tw, ids, inter)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


_NATIVE_ARGS = ("output", "x", "topk_ids", "topk_weights", "w13",
                "w13_scale", "w2", "w2_scale", "shared_w13",
                "shared_w13_scale", "shared_w2", "shared_w2_scale",
                "shared_gate", "ws")


@pytest.mark.parametrize("argument", _NATIVE_ARGS)
@pytest.mark.parametrize("destination", ["cpu", "xpu:1"])
@requires_moe_kernel
def test_moe_shared_fused_decode_rejects_wrong_device(argument, destination):
    """Reject every foreign pointer before any SYCL submission."""
    if destination == "xpu:1" and torch.xpu.device_count() < 2:
        pytest.skip("requires two XPUs")
    seed_everything(7)
    hidden, inter, experts = 256, 8, 8
    weights = make_weights(inter, hidden=hidden, num_experts=experts)
    (w13, s13, _), (w2, s2, _), (sw13, ss13, _), (sw2, ss2, _), gate = weights
    x = torch.randn(1, hidden, device=DEVICE, dtype=torch.float16)
    output = torch.full_like(x, float("nan"))
    tw, ids = route(1, experts)
    nbytes = moe_shared_fused_workspace_bytes(1, TOP_K, experts, hidden,
                                              2 * inter, inter)
    args = [output, x, ids, tw, w13, s13, w2, s2, sw13, ss13,
            sw2, ss2, gate.reshape(-1),
            torch.empty(nbytes, device=DEVICE, dtype=torch.uint8)]
    args[_NATIVE_ARGS.index(argument)] = args[
        _NATIVE_ARGS.index(argument)].to(destination)
    with pytest.raises(RuntimeError, match="XPU"):
        torch.ops._xpu_C.moe_shared_fused_decode_interface(*args, 0)
    assert torch.isnan(output).all()


@pytest.mark.parametrize("tile_m", [-1, 1, 32])
def test_moe_shared_fused_decode_invalid_tile_raises_value_error(tile_m):
    with pytest.raises(ValueError, match="tile_m"):
        dummy = torch.empty(0, device="cpu")
        XpuMoESharedFusedDecode(*([dummy] * 9), top_k=TOP_K, tile_m=tile_m)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [8, 32])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("all_padding", [False, True])
@requires_moe_kernel
def test_moe_shared_fused_decode_padding_routes(dtype, m, ids_dtype,
                                              all_padding):
    """Padding IDs skip routed experts while preserving the shared expert."""
    seed_everything(7)
    inter, experts = 256, 8
    weights = make_weights(inter, dtype, num_experts=experts)
    op = make_op(weights)
    x = (torch.randn(m, HIDDEN, device=DEVICE) / 4).to(dtype)
    tw, ids = route(m, experts)
    ids = ids.to(ids_dtype)
    # Reuse a populated workspace to catch stale permutation indices.
    op.forward(torch.empty_like(x), x, tw, ids)
    if all_padding:
        ids.fill_(-1)
    else:
        ids[0].fill_(-1)
        ids[:, 0] = -1
    out = op.forward(torch.empty_like(x), x, tw, ids)
    ref = ref_moe_shared(x, weights, tw, ids, inter)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
