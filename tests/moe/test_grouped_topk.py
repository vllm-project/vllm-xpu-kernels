# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.moe.ops.grouped_topk import grouped_topk, grouped_topk_native
from tests.utils import opcheck

dpcpp_device = torch.device("xpu")


class TestTorchMethod:

    @pytest.mark.parametrize("seed", [123, 356, 478])
    @pytest.mark.parametrize("dtype",
                             [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("n_token", [1, 2, 64, 256])
    @pytest.mark.parametrize("n_expert", [64, 128, 256])
    @pytest.mark.parametrize("n_topk", [4, 6, 8])
    @pytest.mark.parametrize("n_topk_group", [4, 6, 8])
    @pytest.mark.parametrize("n_expert_group", [8])
    @pytest.mark.parametrize("renormalize", [True, False])
    @pytest.mark.parametrize("scoring_func", ["sigmoid", "softmax"])
    @pytest.mark.parametrize("has_bias", [True, False])
    def test_grouped_topk(
        self,
        seed,
        dtype,
        n_token,
        n_expert,
        n_topk,
        n_expert_group,
        n_topk_group,
        renormalize,
        scoring_func,
        has_bias,
    ):

        torch.manual_seed(seed)
        torch.set_default_device("xpu")
        gating_output = torch.randn(n_token, n_expert,
                                    device=dpcpp_device).to(dtype)
        hidden_states = torch.zeros(n_token, n_expert,
                                    device=dpcpp_device).to(dtype)
        bias = None
        if has_bias:
            if has_bias and scoring_func == "sigmoid" \
                and dtype is not torch.float32:
                # using a bias of bigger number to avoid Low-precision
                bias = torch.arange(1, n_expert + 1).to(dpcpp_device).to(dtype)
            else:
                bias = torch.randn(n_expert, device=dpcpp_device).to(dtype)

        ref_topk_weights, ref_topk_indices = grouped_topk_native(
            hidden_states,
            gating_output,
            n_topk,
            renormalize,
            n_expert_group,
            n_topk_group,
            scoring_func=scoring_func,
            e_score_correction_bias=bias,
        )

        topk_weights, topk_indices = grouped_topk(
            hidden_states,
            gating_output,
            n_topk,
            renormalize,
            n_expert_group,
            n_topk_group,
            scoring_func=scoring_func,
            e_score_correction_bias=bias,
        )

        # Compare the results
        torch.testing.assert_close(ref_topk_weights,
                                   topk_weights,
                                   atol=2e-2,
                                   rtol=1e-2)
        assert torch.equal(ref_topk_indices, topk_indices)

        opcheck(
            torch.ops._moe_C.grouped_topk,
            (hidden_states, gating_output, n_topk, renormalize, n_expert_group,
             n_topk_group, scoring_func, bias),
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("renormalize", [True])
    @pytest.mark.parametrize("full_nan", [True, False])
    def test_grouped_topk_sigmoid_nan(
        self,
        dtype,
        renormalize,
        full_nan,
    ):
        n_token = 512
        n_expert = 256
        n_topk = 8
        n_expert_group = 8
        n_topk_group = 4

        gating_output = torch.randn(n_token, n_expert,
                                    device=dpcpp_device).to(dtype)
        hidden_states = torch.zeros(n_token, n_expert,
                                    device=dpcpp_device).to(dtype)
        bias = torch.randn(n_expert, device=dpcpp_device).to(dtype)

        if full_nan:
            gating_output = torch.full(gating_output.size(),
                                       float("nan"),
                                       device=dpcpp_device,
                                       dtype=dtype).contiguous()
        else:
            gating_output[0][0] = float("nan")

        topk_weights, topk_indices = grouped_topk(
            hidden_states,
            gating_output,
            n_topk,
            renormalize,
            n_expert_group,
            n_topk_group,
            e_score_correction_bias=bias,
        )

        assert torch.all(topk_indices < n_expert)
