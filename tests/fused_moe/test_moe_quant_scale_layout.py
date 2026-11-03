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
from vllm_xpu_kernels.moe_utils import (canonical_activation,
                                        fused_act_quant_xpu,
                                        fused_silu_mul_quant_act_xpu,
                                        mxfp_scale_padded_rows, quant_act_xpu,
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


@pytest.mark.parametrize("rows_per_expert_list", ROWS_PER_EXPERT)
@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("recipe", ["mxfp8", "mxfp4_fp8"])
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_fused_silu_mul_quant_act_moe_scale_layout(rows_per_expert_list,
                                                   hidden, recipe, dtype):
    seed_everything(0)
    num_experts = len(rows_per_expert_list)
    num_rows = sum(rows_per_expert_list)
    buf_rows = num_rows + 5

    gate_up = torch.randn((buf_rows, hidden * 2), dtype=dtype).to(
        KERNEL_DEVICE)

    rows, desc = _desc(rows_per_expert_list, gate_up.device)
    total_padded = mxfp_scale_padded_rows(buf_rows, num_experts)

    fused_q, fused_s = fused_silu_mul_quant_act_xpu(gate_up, recipe, desc,
                                                    total_padded)
    ref_q, ref_s = fused_silu_mul_quant_act_xpu(gate_up, recipe)

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
    tail = fused_s[expected.shape[0]:].view(torch.uint8)
    assert torch.all(tail == 0)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@torch.inference_mode()
def test_fused_silu_mul_quant_act_matches_unfused(hidden):
    seed_everything(0)
    from vllm_xpu_kernels.moe_utils import dequant_mxfp8

    gate_up = torch.randn((128, hidden * 2), dtype=torch.bfloat16).to(
        KERNEL_DEVICE) * 3
    fused_q, fused_s = fused_silu_mul_quant_act_xpu(gate_up, "mxfp8")

    act = torch.empty((128, hidden),
                      dtype=gate_up.dtype,
                      device=gate_up.device)
    torch.ops._C.silu_and_mul(act, gate_up)
    ref_q, ref_s = quant_act_xpu(act, "mxfp8")

    gate = gate_up[:, :hidden].float()
    up = gate_up[:, hidden:].float()
    golden = gate * torch.sigmoid(gate) * up

    fused_err = (dequant_mxfp8(fused_q, fused_s).float() -
                 golden).norm() / golden.norm()
    ref_err = (dequant_mxfp8(ref_q, ref_s).float() -
               golden).norm() / golden.norm()
    assert fused_err <= ref_err
    assert fused_err < 0.05


@torch.inference_mode()
def test_fused_silu_mul_quant_act_declines_unfusable_recipes():
    """Recipes without a fused kernel must fall back, not raise."""
    gate_up = torch.randn((16, 256), dtype=torch.bfloat16).to(KERNEL_DEVICE)
    assert fused_silu_mul_quant_act_xpu(gate_up, "int4") is None
    assert fused_silu_mul_quant_act_xpu(gate_up, "bf16") is None
    assert fused_silu_mul_quant_act_xpu(gate_up, "fp8") is None
    desc = torch.zeros((2, 3), dtype=torch.int32, device=KERNEL_DEVICE)
    assert fused_silu_mul_quant_act_xpu(gate_up, "mxfp4", desc, 16) is None


@torch.inference_mode()
def test_fused_silu_mul_quant_act_mxfp4_matches_unfused():
    seed_everything(0)
    from vllm_xpu_kernels.moe_utils import dequant_mxfp4

    hidden = 256
    gate_up = torch.randn((128, hidden * 2), dtype=torch.bfloat16).to(
        KERNEL_DEVICE) * 3
    fused_q, fused_s = fused_silu_mul_quant_act_xpu(gate_up, "mxfp4")

    act = torch.empty((128, hidden),
                      dtype=gate_up.dtype,
                      device=gate_up.device)
    torch.ops._C.silu_and_mul(act, gate_up)
    ref_q, ref_s = quant_act_xpu(act, "mxfp4")

    assert (fused_q.shape, fused_q.dtype) == (ref_q.shape, ref_q.dtype)
    assert (fused_s.shape, fused_s.dtype) == (ref_s.shape, ref_s.dtype)

    gate = gate_up[:, :hidden].float()
    up = gate_up[:, hidden:].float()
    golden = gate * torch.sigmoid(gate) * up
    fused_err = (dequant_mxfp4(fused_q, fused_s).float() -
                 golden).norm() / golden.norm()
    ref_err = (dequant_mxfp4(ref_q, ref_s).float() -
               golden).norm() / golden.norm()
    assert fused_err < ref_err * 1.02
    assert fused_err < 0.2


@torch.inference_mode()
def test_fused_silu_mul_quant_act_fp8_static_matches_unfused():
    seed_everything(0)
    hidden = 256
    gate_up = torch.randn((128, hidden * 2),
                          dtype=torch.bfloat16).to(KERNEL_DEVICE) * 3
    static_scale = torch.tensor([0.05],
                                dtype=torch.float32,
                                device=KERNEL_DEVICE)
    fused_q, fused_s = fused_silu_mul_quant_act_xpu(gate_up,
                                                    "fp8",
                                                    static_scale=static_scale)

    act = torch.empty((128, hidden),
                      dtype=gate_up.dtype,
                      device=gate_up.device)
    torch.ops._C.silu_and_mul(act, gate_up)
    ref_q, ref_s = quant_act_xpu(act, "fp8", static_scale=static_scale)

    assert fused_q.dtype == ref_q.dtype == torch.float8_e4m3fn
    assert fused_q.shape == ref_q.shape
    torch.testing.assert_close(fused_s, ref_s, atol=0, rtol=0)
    torch.testing.assert_close(fused_q.view(torch.uint8),
                               ref_q.view(torch.uint8),
                               atol=0,
                               rtol=0)


def test_canonical_activation_normalizes_aliases():
    assert canonical_activation("silu") == "silu"
    assert canonical_activation("silu_and_mul") == "silu"
    assert canonical_activation("SWIGLUOAI") == "swigluoai"
    assert canonical_activation("FusedMoEActivationFormat.SWIGLUOAI") \
        == "swigluoai"
    assert canonical_activation("relu2_no_mul") == "relu2_no_mul"
    assert canonical_activation("not_an_activation") is None


def test_fused_act_quant_declines_unregistered_combinations():
    gate_up = torch.randn((16, 256), dtype=torch.bfloat16).to(KERNEL_DEVICE)
    for activation in ("gelu", "gelu_tanh", "swigluoai", "relu2_no_mul",
                       "swiglustep", "not_an_activation"):
        for recipe in ("mxfp8", "mxfp4_fp8", "fp8block", "mxfp4", "fp8"):
            assert fused_act_quant_xpu(activation, gate_up, recipe) is None
