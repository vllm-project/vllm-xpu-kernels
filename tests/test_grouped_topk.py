# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.ops.grouped_topk_op import fused_grouped_topk, grouped_topk
from tests.utils import REF_DEVICE, seed_everything

DEVICE = REF_DEVICE
KERNEL_DEVICE = "xpu"


def _to_kernel(x):
    return None if x is None else x.to(KERNEL_DEVICE)


#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "n_hidden": [4],
        "n_token": [1],
        "routed_scaling_factor": [1.0],
        "scoring_func": ["softmax"],
        "renormalize": [True],
    },
}


@pytest.mark.parametrize("n_token", [1, 33, 64])
@pytest.mark.parametrize("n_hidden", [1024, 2048])
@pytest.mark.parametrize("n_expert", [16])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("num_expert_group", [8])
@pytest.mark.parametrize("topk_group", [2])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
def test_grouped_topk(n_token: int, n_hidden: int, n_expert: int, topk: int,
                      renormalize: bool, num_expert_group: int,
                      topk_group: int, scoring_func: str,
                      routed_scaling_factor: float, dtype: torch.dtype):
    seed_everything(0)
    hidden_states = torch.randn((n_token, n_hidden), dtype=dtype, device="xpu")
    gating_output = torch.randn((n_token, n_expert), dtype=dtype, device="xpu")
    # The correction bias is always float32 (matches real-world callers and
    # the kernel, which only instantiates the float32 bias specialization).
    e_score_correction_bias = torch.randn((n_expert, ),
                                          dtype=torch.float32,
                                          device="xpu")

    baseline_topk_weights, baseline_topk_ids = grouped_topk(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias)

    test_topk_weights, test_topk_ids = fused_grouped_topk(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias)

    if renormalize:
        torch.testing.assert_close(baseline_topk_weights,
                                   test_topk_weights,
                                   atol=2e-2,
                                   rtol=0)

    torch.testing.assert_close(baseline_topk_ids,
                               test_topk_ids,
                               atol=0,
                               rtol=0)


def _run_and_check(n_token: int, n_expert: int, topk: int, renormalize: bool,
                   num_expert_group: int, topk_group: int, scoring_func: str,
                   routed_scaling_factor: float, dtype: torch.dtype):
    """Shared body: compare fused XPU kernel against the python reference."""
    seed_everything(0)
    hidden_states = torch.randn((n_token, n_expert), dtype=dtype, device=DEVICE)
    gating_output = torch.randn((n_token, n_expert), dtype=dtype, device=DEVICE)
    # The correction bias is always float32 (matches real-world callers and
    # the kernel, which only instantiates the float32 bias specialization).
    e_score_correction_bias = torch.randn((n_expert, ),
                                          dtype=torch.float32,
                                          device=DEVICE)

    baseline_topk_weights, baseline_topk_ids = grouped_topk(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias)

    test_topk_weights, test_topk_ids = fused_grouped_topk(
        hidden_states=_to_kernel(hidden_states),
        gating_output=_to_kernel(gating_output),
        topk=topk,
        renormalize=renormalize,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=_to_kernel(e_score_correction_bias))
    test_topk_weights = test_topk_weights.to(DEVICE)
    test_topk_ids = test_topk_ids.to(DEVICE)

    # Sort by expert id so that any difference in intra-token ordering does
    # not cause spurious failures (routing is order-insensitive).
    baseline_topk_ids, baseline_order = torch.sort(baseline_topk_ids, dim=-1)
    test_topk_ids, test_order = torch.sort(test_topk_ids, dim=-1)
    baseline_topk_weights = torch.gather(baseline_topk_weights, 1,
                                         baseline_order)
    test_topk_weights = torch.gather(test_topk_weights, 1, test_order)

    torch.testing.assert_close(baseline_topk_ids,
                               test_topk_ids,
                               atol=0,
                               rtol=0)
    torch.testing.assert_close(baseline_topk_weights,
                               test_topk_weights,
                               atol=2e-2,
                               rtol=0)


# ---------------------------------------------------------------------------
# Path 1: single_group_topk fast path (n_group == 1 && topk_group == 1).
# Exercises the tiered flashinfer-style block / warp kernels. The expert
# counts below straddle the dispatch tiers (128 / 256 / 384 / 512).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_token", [1, 64])
@pytest.mark.parametrize("n_expert", [128, 256, 288, 512])
@pytest.mark.parametrize("topk", [2, 8])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("routed_scaling_factor", [1.0])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
def test_grouped_topk_single_group(n_token: int, n_expert: int, topk: int,
                                   renormalize: bool, scoring_func: str,
                                   routed_scaling_factor: float,
                                   dtype: torch.dtype):
    _run_and_check(n_token=n_token,
                   n_expert=n_expert,
                   topk=topk,
                   renormalize=renormalize,
                   num_expert_group=1,
                   topk_group=1,
                   scoring_func=scoring_func,
                   routed_scaling_factor=routed_scaling_factor,
                   dtype=dtype)


# ---------------------------------------------------------------------------
# Path 2: grouped_topk_fused_small_expert_count (register reduceTopK).
# DeepSeek-style multi-group routing: experts_per_group == 32,
# experts_per_group * topk_group == 128 (== MaxNumExpertsUnit), topk <= 8.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_token", [1, 64])
@pytest.mark.parametrize("n_expert", [256])
@pytest.mark.parametrize("num_expert_group", [8])
@pytest.mark.parametrize("topk_group", [4])
@pytest.mark.parametrize("topk", [6, 8])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
def test_grouped_topk_multi_group(n_token: int, n_expert: int,
                                  num_expert_group: int, topk_group: int,
                                  topk: int, renormalize: bool,
                                  scoring_func: str, dtype: torch.dtype):
    _run_and_check(n_token=n_token,
                   n_expert=n_expert,
                   topk=topk,
                   renormalize=renormalize,
                   num_expert_group=num_expert_group,
                   topk_group=topk_group,
                   scoring_func=scoring_func,
                   routed_scaling_factor=1.0,
                   dtype=dtype)


# ---------------------------------------------------------------------------
# Path 3: WarpSelect fallback. experts_per_group == 64 (> WARP_SIZE) so the
# multi-group fast path is skipped and the generic WarpSelect kernel runs.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_token", [1, 64])
@pytest.mark.parametrize("n_expert", [512])
@pytest.mark.parametrize("num_expert_group", [8])
@pytest.mark.parametrize("topk_group", [2])
@pytest.mark.parametrize("topk", [4, 8])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
def test_grouped_topk_warpselect_fallback(n_token: int, n_expert: int,
                                          num_expert_group: int,
                                          topk_group: int, topk: int,
                                          renormalize: bool, scoring_func: str,
                                          dtype: torch.dtype):
    _run_and_check(n_token=n_token,
                   n_expert=n_expert,
                   topk=topk,
                   renormalize=renormalize,
                   num_expert_group=num_expert_group,
                   topk_group=topk_group,
                   scoring_func=scoring_func,
                   routed_scaling_factor=1.0,
                   dtype=dtype)


# ---------------------------------------------------------------------------
# Regression: masked / empty-selection handling must match across dispatch
# paths. With all-(-inf) logits and sigmoid scoring every candidate scores 0
# and no finite expert can be selected. Path 1 emits uniform 1/topk weights
# when renormalize is set; the WarpSelect fallback (reached when the matching
# Path 1 tier is compiled out, e.g. 512 experts / topk 8) must do the same
# instead of returning all-zero weights.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_token", [1, 8])
@pytest.mark.parametrize("n_expert", [128, 256, 512])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
def test_grouped_topk_all_masked_uniform(n_token: int, n_expert: int,
                                         topk: int, renormalize: bool,
                                         dtype: torch.dtype):
    seed_everything(0)
    neg_inf = float("-inf")
    gating_output = torch.full((n_token, n_expert),
                               neg_inf,
                               dtype=dtype,
                               device=DEVICE)
    e_score_correction_bias = torch.zeros((n_expert, ),
                                          dtype=torch.float32,
                                          device=DEVICE)

    test_topk_weights, test_topk_ids = fused_grouped_topk(
        hidden_states=_to_kernel(gating_output),
        gating_output=_to_kernel(gating_output),
        topk=topk,
        renormalize=renormalize,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        routed_scaling_factor=1.0,
        e_score_correction_bias=_to_kernel(e_score_correction_bias))
    test_topk_weights = test_topk_weights.to(DEVICE)

    assert not torch.isnan(test_topk_weights).any(), "weights must not be NaN"
    if renormalize:
        expected = torch.full_like(test_topk_weights, 1.0 / topk)
        torch.testing.assert_close(test_topk_weights,
                                   expected,
                                   atol=2e-2,
                                   rtol=0)
    else:
        torch.testing.assert_close(test_topk_weights,
                                   torch.zeros_like(test_topk_weights),
                                   atol=0,
                                   rtol=0)


# ---------------------------------------------------------------------------
# Regression: single-group topk == 22 with num_experts < 512 must not reach a
# Path 2 tier whose register capacity is only 8 (which would overrun
# topScores/topExperts). It must fall through to WarpSelect and produce correct
# results. The topk == 22 fast path is reserved for the 512-expert Nemotron
# specialization only.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_token", [1, 33])
@pytest.mark.parametrize("n_expert", [256, 384])
@pytest.mark.parametrize("topk", [22])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
def test_grouped_topk_single_group_topk22(n_token: int, n_expert: int,
                                          topk: int, renormalize: bool,
                                          scoring_func: str,
                                          dtype: torch.dtype):
    _run_and_check(n_token=n_token,
                   n_expert=n_expert,
                   topk=topk,
                   renormalize=renormalize,
                   num_expert_group=1,
                   topk_group=1,
                   scoring_func=scoring_func,
                   routed_scaling_factor=1.0,
                   dtype=dtype)
