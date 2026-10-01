# SPDX-License-Identifier: Apache-2.0
"""Fused silu_and_mul + MXFP4 per-group quant must match the two-step path.

The fused op replaces ``silu_and_mul`` followed by
``per_token_group_quant_mxfp4`` on the MoE gemm1 -> gemm2 path, so the
contract that matters is bit-exact equivalence with running those two kernels
back to back, for both scale surfaces the quantiser can write (plain float32
scales and the padded MN-major e8m0 MoE surface).
"""

import pytest
import torch

import tests.register_ops as ops
from tests.utils import format_tc
from vllm_xpu_kernels.moe_utils import mxfp_scale_padded_rows

DEVICE = "xpu"
MXFP4_GROUP_SIZE = 32

DTYPES = [torch.bfloat16, torch.half, torch.float]
# inter_size must be a multiple of the vector width (16 / sizeof(scalar_t))
# and of the MX group size.
SHAPES = [(1, 32), (4, 64), (37, 256), (128, 512)]

MINI_PYTEST_PARAMS = {
    "default": {
        "shape": [(4, 64)],
        "dtype": [torch.bfloat16],
    },
}


def _two_step(x, out_s):
    num_tokens, inter_size = x.shape[0], x.shape[1] // 2
    act = torch.empty(num_tokens,
                      inter_size,
                      dtype=x.dtype,
                      device=x.device)
    torch.ops._C.silu_and_mul(act, x)
    out_q = torch.empty(num_tokens,
                        inter_size // 2,
                        dtype=torch.uint8,
                        device=x.device)
    torch.ops._C.per_token_group_quant_mxfp4(act, out_q, out_s,
                                             MXFP4_GROUP_SIZE, 1e-10)
    return out_q


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES, ids=format_tc)
@pytest.mark.parametrize("column_major_scale", [False, True])
@torch.inference_mode()
def test_matches_separate_act_and_quant(shape, dtype, column_major_scale):
    torch.manual_seed(0)
    num_tokens, inter_size = shape
    # Scale by 4 so the silu saturation region is exercised on both signs.
    x = torch.randn(num_tokens,
                    2 * inter_size,
                    dtype=dtype,
                    device=DEVICE) * 4.0

    scale_k = inter_size // MXFP4_GROUP_SIZE

    def _alloc_scale():
        if column_major_scale:
            return torch.empty((scale_k, num_tokens),
                               dtype=torch.float32,
                               device=DEVICE).permute(-1, -2)
        return torch.empty((num_tokens, scale_k),
                           dtype=torch.float32,
                           device=DEVICE)

    ref_s = _alloc_scale()
    ref_q = _two_step(x, ref_s)

    out_s = _alloc_scale()
    out_q = torch.empty(num_tokens,
                        inter_size // 2,
                        dtype=torch.uint8,
                        device=DEVICE)
    ops.silu_and_mul_per_token_group_quant_mxfp4(x, out_q, out_s,
                                                 MXFP4_GROUP_SIZE)

    torch.testing.assert_close(out_s, ref_s, atol=0, rtol=0)
    torch.testing.assert_close(out_q, ref_q, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES, ids=format_tc)
@torch.inference_mode()
def test_matches_separate_act_and_quant_moe_layout(dtype):
    """The e8m0 MoE surface must be written identically by both paths."""
    torch.manual_seed(0)
    num_experts = 3
    rows_per_expert = [5, 0, 11]
    num_tokens = sum(rows_per_expert)
    inter_size = 128
    scale_k = inter_size // MXFP4_GROUP_SIZE

    x = torch.randn(num_tokens,
                    2 * inter_size,
                    dtype=dtype,
                    device=DEVICE) * 4.0

    src, dst, s_acc, d_acc = [0], [0], 0, 0
    for rows in rows_per_expert:
        s_acc += rows
        d_acc += (rows + 3) // 4 * 4
        src.append(s_acc)
        dst.append(d_acc)
    desc = torch.tensor([src, dst], dtype=torch.int32, device=DEVICE)

    def _alloc_scale():
        # The kernel validates the surface against the worst-case padding
        # (every expert rounded up to a multiple of 4 rows), so size it with
        # mxfp_scale_padded_rows rather than the exact d_acc.
        return torch.zeros((mxfp_scale_padded_rows(num_tokens, num_experts),
                            scale_k),
                           dtype=torch.float8_e8m0fnu,
                           device=DEVICE)

    ref_s = _alloc_scale()
    act = torch.empty(num_tokens, inter_size, dtype=dtype, device=DEVICE)
    torch.ops._C.silu_and_mul(act, x)
    ref_q = torch.empty(num_tokens,
                        inter_size // 2,
                        dtype=torch.uint8,
                        device=DEVICE)
    torch.ops._C.per_token_group_quant_mxfp4(act, ref_q, ref_s,
                                             MXFP4_GROUP_SIZE, 1e-10, desc)

    out_s = _alloc_scale()
    out_q = torch.empty(num_tokens,
                        inter_size // 2,
                        dtype=torch.uint8,
                        device=DEVICE)
    ops.silu_and_mul_per_token_group_quant_mxfp4(x, out_q, out_s,
                                                 MXFP4_GROUP_SIZE, 1e-10, desc)

    torch.testing.assert_close(out_s.view(torch.uint8),
                               ref_s.view(torch.uint8),
                               atol=0,
                               rtol=0)
    torch.testing.assert_close(out_q, ref_q, atol=0, rtol=0)
