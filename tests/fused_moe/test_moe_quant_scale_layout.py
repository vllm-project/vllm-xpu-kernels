# SPDX-License-Identifier: Apache-2.0
"""The MoE activation quant kernels can emit the per-expert MN-major,
M-padded-to-4 scale surface directly, which removes the reorder pass between
the two grouped GEMMs. Check that the fused output matches quant + reorder.
"""
import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401
import vllm_xpu_kernels._moe_C  # noqa: F401
from tests.utils import seed_everything
from vllm_xpu_kernels.moe_utils import (mxfp_scale_padded_rows, quant_act_xpu,
                                        reorder_mxfp_scales)

KERNEL_DEVICE = "xpu"

ROWS_PER_EXPERT = [
    [0, 1, 2, 3, 4, 5, 7, 8],
    [16, 0, 33, 100],
    [1],
    [0, 0, 0],
    [64, 32, 96, 33],
]
HIDDEN_SIZES = [128, 1024]
RECIPES = ["mxfp8", "mxfp4", "mxfp4_fp8"]
DTYPES = [torch.bfloat16]

MINI_PYTEST_PARAMS = {
    "rows_per_expert_list": [[16, 0, 33, 100]],
    "hidden": [128],
}


def _desc(rows_list, device):
    """The int32 [2, E + 1] descriptor remap_hidden_states publishes."""
    rows = torch.tensor(rows_list, dtype=torch.int32, device=device)
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    src = torch.cat([zero, torch.cumsum(rows, 0, dtype=torch.int32)])
    padded = torch.cat(
        [zero, torch.cumsum((rows + 3) & ~3, 0, dtype=torch.int32)])
    return rows, torch.stack([src, padded])


@pytest.mark.parametrize("rows_per_expert_list", ROWS_PER_EXPERT)
@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_quant_act_moe_scale_layout(rows_per_expert_list, hidden, recipe,
                                    dtype):
    seed_everything(0)
    num_experts = len(rows_per_expert_list)
    num_rows = sum(rows_per_expert_list)
    # The activation buffer between the GEMMs is sized for the worst case, so
    # the tail past the routed rows is unrouted garbage the kernel must skip.
    buf_rows = num_rows + 5

    x = torch.randn((buf_rows, hidden), dtype=dtype).to(KERNEL_DEVICE)

    rows, desc = _desc(rows_per_expert_list, x.device)
    total_padded = mxfp_scale_padded_rows(buf_rows, num_experts)

    fused_q, fused_s = quant_act_xpu(x, recipe, desc, total_padded)
    ref_q, ref_s = quant_act_xpu(x, recipe)

    assert fused_s.dtype == torch.float8_e8m0fnu
    assert fused_s.shape == (total_padded, hidden // 32)
    torch.testing.assert_close(fused_q.view(torch.uint8),
                               ref_q.view(torch.uint8),
                               atol=0,
                               rtol=0)

    expected = reorder_mxfp_scales(ref_s[:num_rows], rows) \
        if num_rows > 0 else fused_s[:0]
    got = fused_s[:expected.shape[0]]
    torch.testing.assert_close(got.view(torch.uint8),
                               expected.view(torch.uint8),
                               atol=0,
                               rtol=0)
    # Nothing past the reordered region may be touched.
    tail = fused_s[expected.shape[0]:].view(torch.uint8)
    assert torch.all(tail == 0)
