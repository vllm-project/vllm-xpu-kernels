# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.ops.topk_op import (fused_topk_sigmoid, fused_topk_softmax,
                               topk_sigmoid, topk_softmax)
from tests.utils import format_tc, seed_everything

#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "n_token": [1],
        "n_expert": [16, 512],
    },
}


@pytest.mark.parametrize("n_token", [1, 33, 40, 64, 1031])
@pytest.mark.parametrize("n_hidden", [1024])
@pytest.mark.parametrize("n_expert", [16, 192, 256, 384, 512, 1024])
@pytest.mark.parametrize("topk", [2, 3, 4, 5, 8, 10])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32],
                         ids=format_tc)
def test_fused_topk_softmax(n_token: int, n_hidden: int, n_expert: int,
                            topk: int, renormalize: bool, has_bias: bool,
                            dtype: torch.dtype):
    seed_everything(0)
    hidden_states = torch.randn((n_token, n_hidden), dtype=dtype, device="xpu")
    gating_output = torch.randn((n_token, n_expert), dtype=dtype, device="xpu")
    bias = None
    if has_bias:
        bias = torch.randn((n_expert, ), dtype=torch.float32, device="xpu")

    baseline_topk_weights, baseline_topk_ids = topk_softmax(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        bias=bias)

    test_topk_weights, test_topk_ids = fused_topk_softmax(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        bias=bias)

    torch.testing.assert_close(baseline_topk_weights,
                               test_topk_weights,
                               atol=2e-2,
                               rtol=0)

    torch.testing.assert_close(baseline_topk_ids,
                               test_topk_ids,
                               atol=0,
                               rtol=0)


@pytest.mark.parametrize("n_token", [1, 33, 512, 513, 2048, 2049])
@pytest.mark.parametrize("topk", [3, 10])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                         ids=format_tc)
def test_fused_topk_softmax_packed(n_token, topk, renormalize, dtype):
    seed_everything(2026)
    logits = torch.randn(n_token, 512, dtype=dtype)
    logits[0, :16] = 0
    logits[0, 1:16:2] = -0.0
    logits[0, 16:] = -8
    padding = torch.arange(n_token) % 17 == 2
    ref_weights, ref_ids = topk_softmax(torch.empty(n_token, 1),
                                        logits.float(),
                                        topk,
                                        renormalize,
                                        is_padding=padding)
    weights = torch.empty(n_token, topk, device="xpu", dtype=torch.float32)
    indices = torch.empty(n_token, topk, device="xpu", dtype=torch.int32)
    source_rows = torch.empty_like(indices)
    torch.ops._moe_C.topk_softmax(weights, indices, source_rows,
                                  logits.to("xpu"), renormalize, None,
                                  padding.to("xpu"))
    torch.testing.assert_close(indices.cpu(), ref_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights.cpu(),
                               ref_weights,
                               rtol=2e-4,
                               atol=2e-6)
    expected_source = (torch.arange(topk)[None, :] * n_token +
                       torch.arange(n_token)[:, None]).to(torch.int32)
    torch.testing.assert_close(source_rows.cpu(), expected_source)


@pytest.mark.parametrize("distribution",
                         ["normal", "all_equal", "quantized", "single_peak",
                          "ties_32", "ties_33"])
@pytest.mark.parametrize("n_token", [4096, 4097])
@pytest.mark.parametrize("renormalize", [True, False])
def test_fused_topk_softmax_large_compact(distribution, n_token, renormalize):
    seed_everything(2026)
    n_expert, topk = 512, 10
    if distribution == "normal":
        logits = torch.randn(n_token, n_expert)
    elif distribution == "all_equal":
        logits = torch.zeros(n_token, n_expert)
    elif distribution == "quantized":
        logits = torch.randint(-2, 3, (n_token, n_expert)).float() / 8
    else:
        logits = torch.full((n_token, n_expert), -20.0)
        if distribution == "single_peak":
            logits[:, -1] = 20
        else:
            permutation = torch.randperm(n_expert)
            candidates = 32 if distribution == "ties_32" else 33
            logits[:, permutation[:candidates]] = 0
    padding = torch.arange(n_token) % 17 == 2
    ref_weights, ref_ids = topk_softmax(torch.empty(n_token, 1), logits, topk,
                                        renormalize, is_padding=padding)
    weights = torch.empty(n_token, topk, device="xpu", dtype=torch.float32)
    indices = torch.empty(n_token, topk, device="xpu", dtype=torch.int32)
    source_rows = torch.empty_like(indices)
    torch.ops._moe_C.topk_softmax(weights, indices, source_rows,
                                  logits.to("xpu"), renormalize, None,
                                  padding.to("xpu"))
    torch.testing.assert_close(indices.cpu(), ref_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights.cpu(), ref_weights, rtol=2e-4,
                               atol=2e-6)
    expected_source = (torch.arange(topk)[None, :] * n_token +
                       torch.arange(n_token)[:, None]).to(torch.int32)
    torch.testing.assert_close(source_rows.cpu(), expected_source)


@pytest.mark.parametrize("n_token,n_expert", [(8, 128), (1031, 512)])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32],
                         ids=format_tc)
def test_fused_topk_padding(n_token: int, n_expert: int, scoring_func: str,
                            has_bias: bool, dtype: torch.dtype):
    seed_everything(0)
    n_hidden = 1024
    topk = 4
    hidden_states = torch.randn((n_token, n_hidden), dtype=dtype, device="xpu")
    gating_output = torch.randn((n_token, n_expert), dtype=dtype, device="xpu")
    bias = None
    if has_bias:
        bias = torch.randn((n_expert, ), dtype=torch.float32, device="xpu")
    is_padding = torch.zeros(n_token, dtype=torch.bool, device="xpu")
    is_padding[1::2] = True

    if scoring_func == "softmax":
        baseline_topk_weights, baseline_topk_ids = topk_softmax(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=True,
            bias=bias,
            is_padding=is_padding)
        test_topk_weights, test_topk_ids = fused_topk_softmax(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=True,
            bias=bias,
            is_padding=is_padding)
    else:
        baseline_topk_weights, baseline_topk_ids = topk_sigmoid(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=True,
            bias=bias,
            is_padding=is_padding)
        test_topk_weights, test_topk_ids = fused_topk_sigmoid(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=True,
            bias=bias,
            is_padding=is_padding)

    torch.testing.assert_close(test_topk_ids[is_padding],
                               torch.full_like(test_topk_ids[is_padding], -1),
                               atol=0,
                               rtol=0)
    torch.testing.assert_close(test_topk_weights[is_padding],
                               torch.zeros_like(test_topk_weights[is_padding]),
                               atol=0,
                               rtol=0)
    torch.testing.assert_close(baseline_topk_weights[~is_padding],
                               test_topk_weights[~is_padding],
                               atol=2e-2,
                               rtol=0)
    torch.testing.assert_close(baseline_topk_ids,
                               test_topk_ids,
                               atol=0,
                               rtol=0)


@pytest.mark.parametrize("n_token", [1, 33, 64])
@pytest.mark.parametrize("n_hidden", [1024])
@pytest.mark.parametrize("n_expert", [16, 192, 512, 1024])
@pytest.mark.parametrize("topk", [2, 4, 5, 8])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32],
                         ids=format_tc)
def test_fused_topk_sigmoid(n_token: int, n_hidden: int, n_expert: int,
                            topk: int, renormalize: bool, has_bias: bool,
                            routed_scaling_factor: float, dtype: torch.dtype):
    seed_everything(0)
    hidden_states = torch.randn((n_token, n_hidden), dtype=dtype, device="xpu")
    gating_output = torch.randn((n_token, n_expert), dtype=dtype, device="xpu")
    bias = None
    if has_bias:
        bias = torch.randn((n_expert, ), dtype=torch.float32, device="xpu")

    baseline_topk_weights, baseline_topk_ids = topk_sigmoid(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        bias=bias,
        routed_scaling_factor=routed_scaling_factor)

    test_topk_weights, test_topk_ids = fused_topk_sigmoid(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        bias=bias,
        routed_scaling_factor=routed_scaling_factor)

    torch.testing.assert_close(baseline_topk_weights,
                               test_topk_weights,
                               atol=2e-2,
                               rtol=0)

    torch.testing.assert_close(baseline_topk_ids,
                               test_topk_ids,
                               atol=0,
                               rtol=0)
