# SPDX-License-Identifier: Apache-2.0
# Tests for router_gemv_topk_softmax / router_resadd_norm_gemv_topk_softmax.
import pytest
import torch

import vllm_xpu_kernels._moe_C  # noqa: F401

OPS = torch.ops._moe_C

DEVICE = "xpu"
EPS = 1e-6
SHAPES = [(256, 2048, 8), (128, 2816, 8), (512, 1024, 10), (64, 4096, 4),
          (200, 384, 16)]


def _reference(x, w, top_k, renormalize):
    logits = torch.nn.functional.linear(x, w)
    m = x.shape[0]
    topk_weights = torch.empty(m, top_k, dtype=torch.float32, device=DEVICE)
    topk_ids = torch.empty(m, top_k, dtype=torch.int32, device=DEVICE)
    token_expert_indices = torch.empty_like(topk_ids)
    torch.ops._moe_C.topk_softmax(topk_weights, topk_ids, token_expert_indices,
                                  logits, renormalize, None, None)
    return logits, topk_weights, topk_ids


def _norm_reference(x, residual, norm_weight):
    s = x.float() + residual.float()
    normed = (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + EPS)) * (
        norm_weight.float() + 1.0)
    return normed.to(x.dtype), s.to(x.dtype)


def _check_topk(logits, topk_weights, topk_ids, ref):
    ref_logits, ref_w, ref_ids = ref
    torch.testing.assert_close(logits,
                               ref_logits.float(),
                               atol=2e-3,
                               rtol=2e-3)
    # Compare as sets per token (order of equal-probability picks may
    # differ), then weights by expert id.
    ids, order = topk_ids.long().sort(dim=-1)
    rids, rorder = ref_ids.long().sort(dim=-1)
    assert torch.equal(ids, rids)
    torch.testing.assert_close(topk_weights.gather(-1, order),
                               ref_w.gather(-1, rorder),
                               atol=1e-3,
                               rtol=1e-3)


def _outputs(m, e, top_k):
    return (torch.empty(m, e, device=DEVICE, dtype=torch.float32),
            torch.empty(m, top_k, device=DEVICE, dtype=torch.float32),
            torch.empty(m, top_k, device=DEVICE, dtype=torch.int32))


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16])
@pytest.mark.parametrize("e,k,top_k", SHAPES)
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_router_gemv_topk_softmax(m, e, k, top_k, renormalize, dtype):
    torch.manual_seed(0)
    x = torch.randn(m, k, device=DEVICE, dtype=dtype)
    w = torch.randn(e, k, device=DEVICE, dtype=dtype) * 0.02
    logits, topk_weights, topk_ids = _outputs(m, e, top_k)
    OPS.router_gemv_topk_softmax(x, w, logits, topk_weights, topk_ids,
                                 renormalize)
    _check_topk(logits, topk_weights, topk_ids,
                _reference(x, w, top_k, renormalize))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_router_gemv_topk_softmax_strided_input(dtype):
    torch.manual_seed(0)
    base = torch.randn(4, 4096, device=DEVICE, dtype=dtype)
    x = base[:, :2048]
    w = torch.randn(256, 2048, device=DEVICE, dtype=dtype) * 0.02
    logits, topk_weights, topk_ids = _outputs(4, 256, 8)
    OPS.router_gemv_topk_softmax(x, w, logits, topk_weights, topk_ids, True)
    _check_topk(logits, topk_weights, topk_ids,
                _reference(x.contiguous(), w, 8, True))


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16])
@pytest.mark.parametrize("e,k,top_k", SHAPES + [(256, 6144, 8)])
@pytest.mark.parametrize("inplace", ["none", "residual", "both"])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_router_resadd_norm_gemv_topk_softmax(m, e, k, top_k, inplace,
                                              renormalize, dtype):
    torch.manual_seed(0)
    x = torch.randn(m, k, device=DEVICE, dtype=dtype)
    residual = torch.randn(m, k, device=DEVICE, dtype=dtype)
    norm_weight = torch.randn(k, device=DEVICE, dtype=dtype) * 0.1
    w = torch.randn(e, k, device=DEVICE, dtype=dtype) * 0.02
    ref_normed, ref_residual = _norm_reference(x, residual, norm_weight)

    x_in, res_in = x.clone(), residual.clone()
    normed_out = x_in if inplace == "both" else torch.empty_like(x)
    residual_out = res_in if inplace != "none" else torch.empty_like(x)
    logits, topk_weights, topk_ids = _outputs(m, e, top_k)
    OPS.router_resadd_norm_gemv_topk_softmax(x_in, res_in, norm_weight, EPS,
                                             w, logits, topk_weights,
                                             topk_ids, normed_out,
                                             residual_out, renormalize)
    torch.testing.assert_close(residual_out, ref_residual, atol=0, rtol=0)
    # Reduction/rsqrt can round to adjacent activation values.
    torch.testing.assert_close(normed_out,
                               ref_normed,
                               atol=1e-3,
                               rtol=max(1e-3,
                                        torch.finfo(dtype).eps))
    # Check routing against the validated, rounded norm output rather than
    # propagating a permitted norm ULP into the projection reference.
    _check_topk(logits, topk_weights, topk_ids,
                _reference(normed_out, w, top_k, renormalize))
    if inplace == "none":  # inputs untouched
        assert torch.equal(x_in, x) and torch.equal(res_in, residual)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_router_resadd_norm_strided(dtype):
    torch.manual_seed(0)
    m, k = 6, 2048
    xb = torch.randn(m, 2 * k, device=DEVICE, dtype=dtype)
    rb = torch.randn(m, 3 * k, device=DEVICE, dtype=dtype)
    x, residual = xb[:, k:], rb[:, :k]
    norm_weight = torch.randn(k, device=DEVICE, dtype=dtype) * 0.1
    w = torch.randn(256, k, device=DEVICE, dtype=dtype) * 0.02
    ref_normed, ref_residual = _norm_reference(x, residual, norm_weight)
    ob = torch.empty(m, 2 * k, device=DEVICE, dtype=dtype)
    normed_out = ob[:, :k]
    logits, topk_weights, topk_ids = _outputs(m, 256, 8)
    # residual updated in place through a strided view
    OPS.router_resadd_norm_gemv_topk_softmax(x, residual, norm_weight, EPS, w,
                                             logits, topk_weights, topk_ids,
                                             normed_out, residual, True)
    torch.testing.assert_close(residual, ref_residual, atol=0, rtol=0)
    torch.testing.assert_close(normed_out,
                               ref_normed,
                               atol=1e-3,
                               rtol=max(1e-3,
                                        torch.finfo(dtype).eps))
    _check_topk(logits, topk_weights, topk_ids,
                _reference(normed_out, w, 8, True))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_router_back_to_back_and_streams(dtype):
    # The ops keep self-resetting device counters per queue: interleave both
    # ops and all M on two streams, without synchronizing in between.
    torch.manual_seed(0)
    e, k, top_k = 256, 2048, 8
    w = torch.randn(e, k, device=DEVICE, dtype=dtype) * 0.02
    g = torch.randn(k, device=DEVICE, dtype=dtype) * 0.1
    cases = []
    for it in range(4):
        for m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16]:
            x = torch.randn(m, k, device=DEVICE, dtype=dtype)
            r = torch.randn(m, k, device=DEVICE, dtype=dtype)
            cases.append((x, r, _outputs(m, e, top_k), _outputs(m, e, top_k),
                          torch.empty_like(x), torch.empty_like(x)))
    torch.xpu.synchronize()
    streams = [torch.xpu.current_stream(), torch.xpu.Stream()]
    for i, (x, r, outs, nouts, normed_out, residual_out) in enumerate(cases):
        with torch.xpu.stream(streams[i % 2]):
            OPS.router_gemv_topk_softmax(x, w, *outs, True)
            OPS.router_resadd_norm_gemv_topk_softmax(x, r, g, EPS, w, *nouts,
                                                     normed_out, residual_out,
                                                     True)
    torch.xpu.synchronize()
    for x, r, outs, nouts, normed_out, residual_out in cases:
        _check_topk(*outs, _reference(x, w, top_k, True))
        ref_normed, ref_residual = _norm_reference(x, r, g)
        torch.testing.assert_close(residual_out, ref_residual, atol=0, rtol=0)
        torch.testing.assert_close(normed_out,
                                   ref_normed,
                                   atol=1e-3,
                                   rtol=max(1e-3,
                                            torch.finfo(dtype).eps))
        _check_topk(*nouts, _reference(normed_out, w, top_k, True))


def test_bf16_router_does_not_round_logits_through_fp16():
    x = torch.zeros(1, 128, device=DEVICE, dtype=torch.bfloat16)
    x[0, 0] = 2.0**17
    w = torch.zeros(64, 128, device=DEVICE, dtype=x.dtype)
    w[:, 0] = torch.arange(1, 65, device=DEVICE).to(x.dtype)
    logits, weights, ids = _outputs(1, 64, 8)
    OPS.router_gemv_topk_softmax(x, w, logits, weights, ids, True)
    _check_topk(logits, weights, ids, _reference(x, w, 8, True))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_router_rejects_mixed_weight_dtype(dtype):
    x = torch.zeros(1, 128, device=DEVICE, dtype=dtype)
    other = torch.bfloat16 if dtype == torch.float16 else torch.float16
    w = torch.zeros(64, 128, device=DEVICE, dtype=other)
    logits, weights, ids = _outputs(1, 64, 8)
    with pytest.raises(RuntimeError, match="router_weight"):
        OPS.router_gemv_topk_softmax(x, w, logits, weights, ids, True)


def test_router_rejects_more_than_sixteen_rows():
    x = torch.zeros(17, 128, device=DEVICE, dtype=torch.bfloat16)
    w = torch.zeros(64, 128, device=DEVICE, dtype=x.dtype)
    logits, weights, ids = _outputs(17, 64, 8)
    with pytest.raises(RuntimeError, match="at most 16 tokens"):
        OPS.router_gemv_topk_softmax(x, w, logits, weights, ids, True)
