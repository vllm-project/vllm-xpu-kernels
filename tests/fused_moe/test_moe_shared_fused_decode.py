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
    top_k = topk_ids.shape[1]
    ids = topk_ids.reshape(-1).long()
    valid = (ids >= 0) & (ids < w13.shape[0])
    safe_ids = ids.clamp(0, w13.shape[0] - 1)
    routed_x = xf.repeat_interleave(top_k, dim=0).unsqueeze(1)
    projections = []
    # Bound gathered weights' memory while batching routes. No per-route
    # scalar reads or tiny expert projections are submitted to XPU.
    for start in range(0, ids.numel(), 32):
        end = start + 32
        experts = safe_ids[start:end]
        h = torch.bmm(routed_x[start:end], w13[experts]).squeeze(1)
        h = h.to(x.dtype).float()
        act = torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]
        projection = torch.bmm(act.to(x.dtype).float().unsqueeze(1),
                               w2[experts]).squeeze(1)
        projections.append(projection.to(x.dtype).float())
    routed = torch.cat(projections) * topk_weights.reshape(-1, 1)
    routed = torch.where(valid[:, None], routed, 0)
    out = routed.reshape(x.shape[0], top_k, x.shape[1]).sum(dim=1)
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
    nbytes = moe_shared_fused_workspace_bytes(m, TOP_K, HIDDEN, 2 * inter,
                                              inter)

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


@pytest.mark.parametrize("shape", [(0, 8, 2048, 256), (-1, 8, 2048, 256),
                                   (1, 0, 2048, 256), (1, -8, 2048, 256),
                                   (1, 8, 0, 256), (1, 8, -16, 256),
                                   (1, 8, 2048, 0), (1, 8, 2048, -8)])
def test_moe_supports_rejects_nonpositive_dimensions(monkeypatch, shape):
    monkeypatch.setattr(interface, "is_available", lambda: True)
    assert not interface.supports(torch.float16, torch.float8_e4m3fn, *shape)


@pytest.mark.parametrize("argument", ["w13", "w13_scales", "w2", "w2_scales",
                                      "shared_w13", "shared_w13_scale",
                                      "shared_w2", "shared_w2_scale",
                                      "shared_gate"])
@pytest.mark.parametrize("destination", ["cpu", "xpu:1"])
@requires_moe_kernel
def test_moe_plan_rejects_weights_on_wrong_device(argument, destination):
    if destination == "xpu:1" and torch.xpu.device_count() < 2:
        pytest.skip("requires two XPUs")
    weights = make_weights(32, hidden=256, num_experts=8)
    (w13, s13, _), (w2, s2, _), (sw13, ss13, _), (sw2, ss2, _), gate = weights
    args = dict(w13=w13, w13_scales=s13, w2=w2, w2_scales=s2,
                shared_w13=sw13, shared_w13_scale=ss13, shared_w2=sw2,
                shared_w2_scale=ss2, shared_gate=gate, top_k=TOP_K)
    args[argument] = args[argument].to(destination)
    with pytest.raises(ValueError, match="XPU"):
        XpuMoESharedFusedDecode(**args)


@pytest.mark.parametrize("destination", ["cpu", "xpu:1"])
@requires_moe_kernel
def test_moe_plan_rejects_input_on_wrong_device(destination):
    if destination == "xpu:1" and torch.xpu.device_count() < 2:
        pytest.skip("requires two XPUs")
    op = make_op(make_weights(32, hidden=256, num_experts=8))
    ws = op.workspace(1)
    x = torch.randn(1, 256, device=destination, dtype=torch.float16)
    output = torch.full_like(x, float("nan"))
    tw, ids = route(1, 8)
    with pytest.raises(ValueError, match="same XPU device as w13"):
        op.forward(output, x, tw.to(destination), ids.to(destination))
    assert op.workspace(1) is ws
    assert torch.isnan(output).all()


@requires_moe_kernel
def test_moe_plan_workspace_is_bound_to_weights_device():
    if torch.xpu.device_count() < 2:
        pytest.skip("requires two XPUs")
    op = make_op(make_weights(32, hidden=256, num_experts=8))
    ws = op.workspace(1)
    with torch.xpu.device(1):
        assert op.workspace(1) is ws
        assert op.workspace(2).device == op.w13.device
    assert ws.device == op.w13.device


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@requires_moe_kernel
def test_moe_shared_fused_decode_gather_tail(dtype):
    """A legal H%32 shape need not fill a gather work-group."""
    seed_everything(7)
    hidden, inter, experts = 2080, 256, 8
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
    hidden, inter, experts = 256, 32, 8
    weights = make_weights(inter, hidden=hidden, num_experts=experts)
    (w13, s13, _), (w2, s2, _), (sw13, ss13, _), (sw2, ss2, _), gate = weights
    x = torch.randn(1, hidden, device=DEVICE, dtype=torch.float16)
    output = torch.full_like(x, float("nan"))
    tw, ids = route(1, experts)
    nbytes = moe_shared_fused_workspace_bytes(1, TOP_K, hidden, 2 * inter,
                                              inter)
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


@pytest.mark.parametrize("shape", [(1, 8, 16, 256), (1, 8, 2064, 256),
                                   (1, 8, 2048, 8), (1, 8, 2048, 16),
                                   (65, 8, 2048, 256),
                                   (1, 8, 1 << 24, 256),
                                   (1, 8, 2048, 1 << 24)])
def test_moe_supports_rejects_unsupported_layout(monkeypatch, shape):
    monkeypatch.setattr(interface, "is_available", lambda: True)
    assert not interface.supports(torch.float16, torch.float8_e4m3fn, *shape)


@pytest.mark.parametrize("shape", [(0, 8, 256, 64, 32),
                                   (-1, 8, 256, 64, 32),
                                   (1, 0, 256, 64, 32),
                                   (65, 8, 256, 64, 32),
                                   (1, 8, 0, 64, 32),
                                   (1, 8, 16, 64, 32),
                                   (1, 8, 256, 32, 16),
                                   (1, 8, 256, 96, 32),
                                   (1, 8, 1 << 24, 64, 32)])
def test_moe_workspace_rejects_invalid_dimensions(shape):
    with pytest.raises(ValueError):
        moe_shared_fused_workspace_bytes(*shape)


def make_native_args(m=1, hidden=256, inter=32, num_experts=8,
                     dtype=torch.float16):
    weights = make_weights(inter, dtype, hidden, num_experts)
    (w13, s13, _), (w2, s2, _), (sw13, ss13, _), (sw2, ss2, _), gate = weights
    x = (torch.randn(m, hidden, device=DEVICE) / 4).to(dtype)
    tw, ids = route(m, num_experts)
    ws = torch.empty(moe_shared_fused_workspace_bytes(
        m, TOP_K, hidden, 2 * inter, inter), dtype=torch.uint8, device=DEVICE)
    return [torch.full_like(x, float("nan")), x, ids, tw, w13, s13, w2, s2,
            sw13, ss13, sw2, ss2, gate.reshape(-1), ws], weights


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [8, 32])
@requires_moe_kernel
def test_moe_shared_fused_decode_wide_invalid_ids(dtype, m):
    seed_everything(7)
    args, weights = make_native_args(m, dtype=dtype)
    # Values with valid low 32 bits must not become valid experts on cast.
    args[2] = args[2].long()
    args[2][:, 0] += 1 << 32
    if m == 32:
        args[2][0, 1] = torch.iinfo(torch.int64).min
    invalid = (args[2] < 0) | (args[2] >= args[4].shape[0])
    masked_ids = torch.where(invalid, -1, args[2])
    masked_args = list(args)
    masked_args[2] = masked_ids
    torch.ops._xpu_C.moe_shared_fused_decode_interface(*masked_args, 0)
    expected = args[0].clone()
    torch.ops._xpu_C.moe_shared_fused_decode_interface(*args, 0)
    # A loose numerical tolerance alone can hide a missing low-weight route.
    torch.testing.assert_close(args[0], expected, rtol=0, atol=0)
    ref = ref_moe_shared(args[1], weights, args[3], args[2], 32)
    torch.testing.assert_close(args[0].float(), ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("argument", ["x", "w13", "w2", "shared_w13",
                                      "shared_w2", "shared_gate", "ws"])
@requires_moe_kernel
def test_moe_shared_fused_decode_rejects_misaligned_storage(argument):
    args, _ = make_native_args()
    idx = _NATIVE_ARGS.index(argument)
    tensor = args[idx]
    storage = torch.empty(tensor.numel() + 1, device=DEVICE,
                          dtype=tensor.dtype)
    shifted = storage[1:].reshape(tensor.shape)
    shifted.copy_(tensor)
    assert shifted.is_contiguous()
    args[idx] = shifted
    with pytest.raises(RuntimeError, match="aligned base address"):
        torch.ops._xpu_C.moe_shared_fused_decode_interface(*args, 0)
    assert torch.isnan(args[0]).all()


@pytest.mark.parametrize("argument", _NATIVE_ARGS[:-1])
@requires_moe_kernel
def test_moe_shared_fused_decode_rejects_workspace_overlap(argument):
    args, _ = make_native_args()
    idx = _NATIVE_ARGS.index(argument)
    tensor = args[idx]
    tensor_bytes = tensor.numel() * tensor.element_size()
    ws = torch.empty(max(args[-1].numel(), tensor_bytes), dtype=torch.uint8,
                     device=DEVICE)
    alias = ws[:tensor_bytes].view(tensor.dtype).reshape(tensor.shape)
    alias.copy_(tensor)
    args[idx] = alias
    args[-1] = ws
    with pytest.raises(RuntimeError, match="single memory location|overlap"):
        torch.ops._xpu_C.moe_shared_fused_decode_interface(*args, 0)
    assert torch.isnan(args[0]).all()


@pytest.mark.parametrize("shape", ["zero_hidden", "zero_inter",
                                   "unaligned_hidden", "unaligned_inter"])
@requires_moe_kernel
def test_moe_shared_fused_decode_rejects_unsupported_dimensions(shape):
    args, _ = make_native_args()
    hidden = 0 if shape == "zero_hidden" else (
        16 if shape == "unaligned_hidden" else 256)
    inter = 0 if shape == "zero_inter" else (
        8 if shape == "unaligned_inter" else 32)
    args[0] = torch.empty(1, hidden, dtype=torch.float16, device=DEVICE)
    args[1] = torch.empty_like(args[0])
    args[4] = torch.empty(8, hidden, 2 * inter,
                          dtype=torch.float8_e4m3fn, device=DEVICE)
    args[6] = torch.empty(8, inter, hidden,
                          dtype=torch.float8_e4m3fn, device=DEVICE)
    args[8] = torch.empty(hidden, 2 * inter,
                          dtype=torch.float8_e4m3fn, device=DEVICE)
    args[10] = torch.empty(inter, hidden,
                           dtype=torch.float8_e4m3fn, device=DEVICE)
    args[12] = torch.empty(hidden, dtype=torch.float16, device=DEVICE)
    with pytest.raises(RuntimeError, match="positive multiples of 32"):
        torch.ops._xpu_C.moe_shared_fused_decode_interface(*args, 0)


@pytest.mark.parametrize("argument", ["x", "topk_ids"])
@requires_moe_kernel
def test_moe_shared_fused_decode_rejects_wrong_input_rank(argument):
    args, _ = make_native_args()
    idx = _NATIVE_ARGS.index(argument)
    args[idx] = args[idx].reshape(-1)
    with pytest.raises(RuntimeError, match="rank two"):
        torch.ops._xpu_C.moe_shared_fused_decode_interface(*args, 0)
    assert torch.isnan(args[0]).all()


@pytest.mark.parametrize("tile_m", [-1, 1, 32])
@requires_moe_kernel
def test_moe_shared_fused_decode_rejects_raw_invalid_tile(tile_m):
    args, _ = make_native_args()
    with pytest.raises(RuntimeError, match="tile_m"):
        torch.ops._xpu_C.moe_shared_fused_decode_interface(*args, tile_m)
    assert torch.isnan(args[0]).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@requires_moe_kernel
def test_moe_shared_fused_decode_small_aligned_shape_at_row_limit(dtype):
    seed_everything(7)
    args, weights = make_native_args(64, hidden=32, inter=32, dtype=dtype)
    torch.ops._xpu_C.moe_shared_fused_decode_interface(*args, 0)
    ref = ref_moe_shared(args[1], weights, args[3], args[2], 32)
    torch.testing.assert_close(args[0].float(), ref, rtol=1e-2, atol=1e-2)


@requires_moe_kernel
def test_moe_shared_fused_decode_on_noncurrent_device():
    if torch.xpu.device_count() < 2:
        pytest.skip("requires two XPUs")
    args, weights = make_native_args()
    # All tensors are on xpu:0 while the thread's current device is xpu:1.
    with torch.xpu.device(1):
        torch.ops._xpu_C.moe_shared_fused_decode_interface(*args, 0)
        assert torch.xpu.current_device() == 1
    ref = ref_moe_shared(args[1], weights, args[3], args[2], 32)
    torch.testing.assert_close(args[0].float(), ref, rtol=1e-2, atol=1e-2)
