# SPDX-License-Identifier: Apache-2.0

import os
import random
from typing import Optional, Union

import numpy as np
import pytest
import torch

from tests.ops.fp8_quant_op import per_token_group_quant_fp8, scaled_fp8_quant
from tests.ops.mx_utils import to_mxfp

SKIP_TEST_FOR_MINI_SCOPE = os.getenv("XPU_KERNEL_PYTEST_PROFILER") == "MINI"


def as_float32_tensor(x: Union[float, torch.tensor]) -> torch.tensor:
    return torch.as_tensor(x, dtype=torch.float32, device="xpu")


def ref_dynamic_per_tensor_fp8_quant(x, fp8_dtype=torch.float8_e5m2):

    fp8_traits = torch.finfo(fp8_dtype)
    fp8_traits_max = fp8_traits.max
    fp8_traits_min = fp8_traits.min
    fp8_max = as_float32_tensor(fp8_traits_max)
    one = as_float32_tensor(1.0)

    # For fp8, in order to match the xpu kernel output, we have to do exactly
    # the same operations as in the corresponding fp8 kernel to prevent
    # rounding errors.

    x_max = as_float32_tensor(x.abs().max())
    ref_scale = x_max / fp8_max
    ref_iscale = one / ref_scale
    ref_out = ((as_float32_tensor(x) * ref_iscale).clamp(
        fp8_traits_min, fp8_traits_max).to(fp8_dtype))
    return ref_out, ref_scale.view((1, ))


def ref_dynamic_per_token_quant(
    x: torch.tensor,
    quant_dtype: torch.dtype,
    scale_ub: Optional[torch.tensor] = None
) -> tuple[torch.tensor, torch.tensor]:

    assert quant_dtype in [torch.float8_e5m2, torch.float8_e4m3fn]
    # if scale_ub is not None:
    #     assert quant_dtype == FP8_DTYPE

    qtype_traits = torch.finfo(quant_dtype)
    qtype_traits_max = qtype_traits.max
    qtype_traits_min = qtype_traits.min
    qtype_max = as_float32_tensor(qtype_traits_max)
    s_1 = as_float32_tensor(1.0)
    s_512 = as_float32_tensor(512.0)

    # For fp8, in order to match the cuda kernel output, we have to do exactly
    # the same operations as in the corresponding fp8 kernel to prevent
    # rounding errors.

    # Compute scales
    x_token_max, _ = x.abs().max(dim=-1)
    x_token_max = as_float32_tensor(x_token_max)
    if scale_ub is not None:
        x_token_max = x_token_max.clamp(max=scale_ub)
    scales = (x_token_max / qtype_max)[:, None]

    # Quant
    min_scaling_factor = s_1 / (qtype_max * s_512)
    scales = scales.clamp(min=min_scaling_factor)
    torch_out = as_float32_tensor(x) / scales
    torch_out = torch_out.clamp(qtype_traits_min,
                                qtype_traits_max).to(quant_dtype)

    return torch_out, scales


def ref_per_block_quant(
    x: torch.Tensor,
    block_m: int = 1,
    block_n: int = 128,
    fp8_dtype=torch.float8_e4m3fn,
    eps: float = 1e-10,
):
    """
    Reference FP8 2D block quantization

    Args:
        x: [M, N] float tensor (fp16/fp32)
        block_m: block rows
        block_n: block cols
        fp8_dtype: torch.float8_e4m3fn or e5m2
    Returns:
        q: FP8 tensor [M, N]
        scales: FP32 tensor [ceil(M/BM), ceil(N/BN)]
    """
    assert fp8_dtype == torch.float8_e4m3fn
    assert x.dim() == 2
    M, N = x.shape
    device = x.device

    assert (
        block_m <= M and block_n <= N and M % block_m == 0 and N % block_n == 0
    ), f"Invalid block size: block_m={block_m}, block_n={block_n}, M={M}, N={N}"
    BM, BN = block_m, block_n
    grid_m = (M + BM - 1) // BM
    grid_n = (N + BN - 1) // BN

    scales = torch.empty((grid_m, grid_n), device=device, dtype=torch.float32)
    q = torch.empty_like(x, dtype=fp8_dtype)

    FP8_MAX = 448.0

    for gm in range(grid_m):
        for gn in range(grid_n):
            m0 = gm * BM
            n0 = gn * BN
            m1 = min(m0 + BM, M)
            n1 = min(n0 + BN, N)

            block = x[m0:m1, n0:n1]

            # absmax
            amax = block.abs().max()
            scale = amax / FP8_MAX
            scale = torch.clamp(scale, min=eps)

            scales[gm, gn] = scale

            # quantize
            q_block = (block / scale).to(fp8_dtype)
            q[m0:m1, n0:n1] = q_block

    return q, scales


def assert_close_percentage(a: torch.Tensor,
                            b: torch.Tensor,
                            mismatch_threshold: float = 0.01):
    """
    Assert that two tensors are close within a mismatch percentage.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        mismatch_threshold (float):
            Allowed mismatch ratio (0.01 = 1% mismatch allowed).

    Raises:
        AssertionError: If mismatch percentage exceeds the threshold.
    """
    if a.shape != b.shape:
        raise AssertionError(f"Shape mismatch: {a.shape} vs {b.shape}")

    mismatch_mask = a != b
    mismatch_count = mismatch_mask.sum().item()
    total_count = a.numel()
    mismatch_ratio = mismatch_count / total_count

    if mismatch_ratio > mismatch_threshold:
        raise AssertionError(
            f"Tensors differ in {mismatch_ratio * 100:.2f}% of elements "
            f"(allowed {mismatch_threshold * 100:.2f}%)")


def seed_everything(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


DTYPES = [torch.half, torch.bfloat16, torch.float]
HIDDEN_SIZES = [
    1,
    2,
    3,
    4,
    16,
    67,
    768,
    2048,
    5120,
    5137,
    8192,
    8193,
]  # Arbitrary values for testing
HIDDEN_SIZES += list(range(1024, 1033))  # vectorized conversion edge cases
NUM_TOKENS = [1, 7, 83, 4096]  # Arbitrary values for testing
SCALE_UBS = [True, False]
SEEDS = [0]
FP8_DTYPES = [torch.float8_e5m2, torch.float8_e4m3fn]

# block quant parameters
MXFP8_HP_DTYPES = [torch.float, torch.bfloat16]
NUM_TOKENS_BLOCK_QUANT = [1, 2, 4, 8]
HIDDEN_SIZES_BLOCK_QUANT = [256]
GROUP_SIZE = [32, 64, 128]
COLUMN_MAJOR_SCALE = [True, False]

# override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [1],
        "hidden_size": [1],
    },
}


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_dynamic_per_tensor_fp8_quant(
    num_tokens: int,
    hidden_size: int,
    fp8_dtype: torch.dtype,
    dtype: torch.dtype,
    seed: int,
) -> None:
    seed_everything(seed)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="xpu")

    ref_out, ref_scale = ref_dynamic_per_tensor_fp8_quant(x, fp8_dtype)

    ops_out, ops_scale = scaled_fp8_quant(x, fp8_dtype=fp8_dtype)

    torch.testing.assert_close(ref_scale, ops_scale)
    torch.testing.assert_close(ref_out.to(dtype=torch.float32),
                               ops_out.to(dtype=torch.float32))


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale_ub", SCALE_UBS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@torch.inference_mode()
def test_dynamic_per_token_fp8_quant(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    scale_ub: bool,
    seed: int,
    fp8_dtype: torch.dtype,
) -> None:
    seed_everything(seed)

    x = (torch.rand(num_tokens, hidden_size, dtype=dtype, device="xpu") + 1e-6
         )  # avoid nans

    scale_ub = torch.mean(x).to(dtype=torch.float32,
                                device="xpu") if scale_ub else None
    ref_out, ref_scales = ref_dynamic_per_token_quant(x, fp8_dtype, scale_ub)

    ops_out, ops_scales = scaled_fp8_quant(x,
                                           scale_ub=scale_ub,
                                           use_per_token_if_dynamic=True,
                                           fp8_dtype=fp8_dtype)

    torch.testing.assert_close(ref_scales, ops_scales)
    assert_close_percentage(
        ref_out.to(dtype=torch.float32),
        ops_out.to(dtype=torch.float32),
        mismatch_threshold=0.005,
    )  # 0.5% mismatch allowed


@pytest.mark.parametrize("num_tokens_block_quant", NUM_TOKENS_BLOCK_QUANT)
@pytest.mark.parametrize("hidden_size_block_quant", HIDDEN_SIZES_BLOCK_QUANT)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("group_size", GROUP_SIZE)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("column_major_scale", COLUMN_MAJOR_SCALE)
@torch.inference_mode()
def test_per_block_fp8_quant(
    num_tokens_block_quant: int,
    hidden_size_block_quant: int,
    dtype: torch.dtype,
    group_size: int,
    seed: int,
    column_major_scale: bool,
) -> None:
    seed_everything(seed)

    x = (torch.rand(num_tokens_block_quant,
                    hidden_size_block_quant,
                    dtype=dtype,
                    device="xpu") + 1e-6)  # avoid nans

    ref_out, ref_scales = ref_per_block_quant(x, 1, group_size)

    ops_out, ops_scales = per_token_group_quant_fp8(
        x,
        group_size=group_size,
        dtype=torch.float8_e4m3fn,
        use_ue8m0=False,
        column_major_scales=column_major_scale)

    assert torch.allclose(ref_out.float(),
                          ops_out.float(),
                          atol=0.15,
                          rtol=0.15)
    assert torch.allclose(ref_scales.float(),
                          ops_scales.float(),
                          atol=0.01,
                          rtol=0.01)


@pytest.mark.parametrize("num_tokens_block_quant", NUM_TOKENS_BLOCK_QUANT)
@pytest.mark.parametrize("hidden_size_block_quant", HIDDEN_SIZES_BLOCK_QUANT)
@pytest.mark.parametrize("mxfp8_hp_dtypes", MXFP8_HP_DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("column_major_scale", COLUMN_MAJOR_SCALE)
@torch.inference_mode()
def test_per_block_mxfp8_quant(
    num_tokens_block_quant: int,
    hidden_size_block_quant: int,
    mxfp8_hp_dtypes: torch.dtype,
    seed: int,
    column_major_scale: bool,
) -> None:
    seed_everything(seed)

    x = (torch.rand(num_tokens_block_quant,
                    hidden_size_block_quant,
                    dtype=mxfp8_hp_dtypes,
                    device="xpu") + 1e-6)  # avoid nans

    ref_scales, ref_out = to_mxfp(x)

    ops_out, ops_scales = per_token_group_quant_fp8(
        x,
        group_size=32,
        dtype=torch.float8_e4m3fn,
        use_ue8m0=True,
        column_major_scales=column_major_scale)

    assert torch.allclose(ref_out.float(),
                          ops_out.float(),
                          atol=0.15,
                          rtol=0.15)
    assert torch.allclose(ref_scales.float(),
                          ops_scales.float(),
                          atol=0.01,
                          rtol=0.01)


# Regression test for a case with large activations where an int32 index cannot
# represent the number of elements.
@torch.inference_mode()
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@pytest.mark.skipif(
    SKIP_TEST_FOR_MINI_SCOPE,
    reason="skip fp8 quant large shape test for the mini pytest profiler.")
def test_fp8_quant_large(seed: int, fp8_dtype: torch.dtype) -> None:
    seed_everything(seed)

    num_tokens = 1024000  # Mistral-Nemo's max_position_embeddings
    hidden_size = 1152  # Smallest hidden_size to reproduce the error
    dtype = torch.bfloat16

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="xpu")
    ref_out, scale = ref_dynamic_per_tensor_fp8_quant(x, fp8_dtype)

    ops_out, _ = scaled_fp8_quant(x, scale, fp8_dtype=fp8_dtype)

    # Minimize memory footprint in this test by freeing x and upconverting
    # the outputs in place. (torch.allclose does not support fp8)
    del x
    ref_out = ref_out.to(dtype=dtype)
    ops_out = ops_out.to(dtype=dtype)

    torch.testing.assert_close(ref_out, ops_out)


if __name__ == "__main__":
    test_dynamic_per_tensor_fp8_quant(1024, 1024, torch.float8_e5m2,
                                      torch.float16, 0)
