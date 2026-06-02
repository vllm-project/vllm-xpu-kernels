# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401 – registers torch.ops._C.*
from tests.ops.mx_utils import _floatx_unpacked_to_f32, to_mxfp, unpack_uint4
from tests.utils import seed_everything

DTYPES = [torch.float16, torch.bfloat16]
SEEDS = [0]
GROUP_SIZE = 32
NUM_TOKENS = [1, 7, 83, 2048]
HIDDEN_SIZES = [128, 1024, 5120]
XPU_DEVICES = [
    f"xpu:{i}" for i in range(min(torch.xpu.device_count(), 2))
] or ["xpu:0"]
EPS = 1e-10


def _ref_silu_and_mul_mxfp4_quant(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (packed_q_uint8, scales_f32, silu_out_f32) for reference."""
    hidden = x.shape[-1] // 2
    gate, up = x.split(hidden, dim=-1)
    gate_f = gate.float()
    up_f = up.float()
    silu_out = (gate_f * torch.sigmoid(gate_f) * up_f).contiguous()

    scale_e8m0, q_packed = to_mxfp(silu_out.contiguous(),
                                   block_size=GROUP_SIZE,
                                   format="mxfp4")
    exp_u8 = scale_e8m0.view(torch.uint8).to(torch.int32)
    scales_f32 = torch.where(
        exp_u8 == 0,
        torch.ones_like(exp_u8, dtype=torch.float32),
        torch.exp2((exp_u8 - 127).to(torch.float32)),
    )
    return q_packed.view(torch.uint8), scales_f32, silu_out.float()


def _ops_silu_and_mul_mxfp4_quant(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = x.shape[0]
    hidden = x.shape[-1] // 2
    num_groups = hidden // GROUP_SIZE
    out = torch.empty(num_tokens, hidden // 2, dtype=torch.uint8,
                      device=x.device)
    scales = torch.empty(num_tokens, num_groups, dtype=torch.float32,
                         device=x.device)
    torch.ops._C.silu_and_mul_mxfp4_quant(out, x.contiguous(), scales,
                                          GROUP_SIZE, EPS)
    return out, scales


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_mxfp4_quant(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    if hidden_size % GROUP_SIZE != 0:
        pytest.skip("hidden_size not divisible by 32")

    seed_everything(seed)
    torch.set_default_device(device)

    scale = 1.0 / hidden_size
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype) * scale

    # Reference
    ref_q_u8, ref_scales_f32, _ = _ref_silu_and_mul_mxfp4_quant(x)

    # Kernel
    ops_q, ops_scales = _ops_silu_and_mul_mxfp4_quant(x)

    assert ops_q.dtype == torch.uint8
    assert ops_q.shape == (num_tokens, hidden_size // 2)
    assert ops_scales.dtype == torch.float32
    assert ops_scales.shape == (num_tokens, hidden_size // GROUP_SIZE)
    assert not torch.isnan(ops_scales).any()
    assert not torch.isinf(ops_scales).any()

    log2_s = torch.log2(ops_scales.float())
    torch.testing.assert_close(log2_s, log2_s.round(), atol=1e-5, rtol=0)

    torch.testing.assert_close(ref_scales_f32, ops_scales,
                               atol=1e-5, rtol=1e-5)

    if not torch.equal(ref_q_u8, ops_q):
        ref_unpacked = unpack_uint4(ref_q_u8.cpu())
        ops_unpacked = unpack_uint4(ops_q.cpu())
        ref_f = _floatx_unpacked_to_f32(ref_unpacked, 2, 1).to(x.device)
        ops_f = _floatx_unpacked_to_f32(ops_unpacked, 2, 1).to(x.device)
        ref_scales_e = ref_scales_f32.unsqueeze(-1).repeat(
            1, 1, GROUP_SIZE).reshape_as(ref_f)
        ops_scales_e = ops_scales.unsqueeze(-1).repeat(
            1, 1, GROUP_SIZE).reshape_as(ops_f)
        ref_deq = ref_f * ref_scales_e
        ops_deq = ops_f * ops_scales_e
        torch.testing.assert_close(ref_deq, ops_deq, atol=0.5, rtol=0.5)
