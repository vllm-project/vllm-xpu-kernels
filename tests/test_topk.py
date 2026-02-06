# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.ops.topk_op import (fused_topk_softmax, fused_topk_sigmoid,
                              topk_softmax, topk_sigmoid)
from tests.utils import seed_everything

#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "n_token": [1],
        "n_expert": [16, 512],
    },
}


@pytest.mark.parametrize("n_token", [1, 33, 64])
@pytest.mark.parametrize("n_hidden", [1024])
@pytest.mark.parametrize("n_expert", [16, 192, 512, 1024])
@pytest.mark.parametrize("topk", [2, 4, 8])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
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

    torch.testing.assert_close(baseline_topk_ids, test_topk_ids, atol=0, rtol=0)


@pytest.mark.parametrize("n_token", [1, 33, 64])
@pytest.mark.parametrize("n_hidden", [1024])
@pytest.mark.parametrize("n_expert", [16, 192, 512, 1024])
@pytest.mark.parametrize("topk", [2, 4, 8])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
def test_fused_topk_sigmoid(n_token: int, n_hidden: int, n_expert: int,
                            topk: int, renormalize: bool, has_bias: bool,
                            dtype: torch.dtype):
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
        bias=bias)

    test_topk_weights, test_topk_ids = fused_topk_sigmoid(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        bias=bias)

    torch.testing.assert_close(baseline_topk_weights,
                               test_topk_weights,
                               atol=2e-2,
                               rtol=0)

    torch.testing.assert_close(baseline_topk_ids, test_topk_ids, atol=0, rtol=0)
