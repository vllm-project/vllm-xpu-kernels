# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the batched-weight RMS norm kernel (torch.ops._C.rms_norm).

``rms_norm`` can use the outermost input dimension index to select the
corresponding weight row when ``weight`` is 2D (``[num_rows, hidden_size]``)
instead of the usual 1D (``[hidden_size]``). This is used by e.g. vLLM's
DFlash/DFlash2 speculative-decoding draft path to normalize all draft layers'
K in a single kernel launch, with one weight row per layer. See
https://github.com/vllm-project/vllm-xpu-kernels/issues/573.

The batched-weight result must exactly match looping ``rms_norm`` over the
outermost dimension with the corresponding weight row.
"""

import pytest
import torch

import tests.register_ops as ops
from tests.utils import opcheck

DTYPES = [torch.half, torch.bfloat16]
SEEDS = [0]
XPU_DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

# override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "shape": [(6, 3, 4, 128)],
    },
}


@pytest.mark.parametrize(
    "shape",
    [
        (28, 17, 128),  # 3D: [num_rows, tokens, hidden]
        (1, 5, 2, 128),  # 4D: single outer row (edge case)
        (28, 13, 8, 128),  # 4D: [L, num_ctx, nkv, hd] (DFlash K-norm shape);
        # small hidden_size triggers the multi-row fast-path kernel.
        (6, 3, 4, 5120),  # 4D: large hidden size; generic vectorized kernel.
        (6, 3, 4, 769),  # 4D: non-power-of-two hidden size; scalar fallback.
    ],
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_rms_norm_batched_weight_matches_loop(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    num_rows, hidden = shape[0], shape[-1]
    eps = 1e-6

    x = torch.randn(*shape, dtype=dtype) * 0.1
    # Distinct weight per row so that a wrong row index would be caught.
    weight = torch.randn(num_rows, hidden, dtype=dtype) * 0.1 + 1.0

    # Reference: normalize each outer row with its own weight row.
    out_ref = torch.empty_like(x)
    for i in range(x.shape[0]):
        ops.rms_norm(out_ref[i], x[i], weight[i], eps)

    # Batched call: single kernel launch, weight selected per outer row.
    out = torch.empty_like(x)
    ops.rms_norm(out, x, weight, eps)

    torch.testing.assert_close(out, out_ref, atol=0, rtol=0)

    opcheck(torch.ops._C.rms_norm, (torch.empty_like(x), x, weight, eps))


@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_rms_norm_batched_weight_zero_row_repro(device: str) -> None:
    """Direct regression test for the issue #573 reproduction: a zeroed
    weight row must zero out exactly that row's output, not some other
    row's."""
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    dtype = torch.float16
    num_layers, num_ctx, nkv, hd = 2, 4, 8, 128
    eps = 1e-6

    x = torch.randn(num_layers, num_ctx, nkv, hd, dtype=dtype)
    weight = torch.ones(num_layers, hd, dtype=dtype)
    weight[1] = 0.0  # layer 1's weight is all zeros

    out = torch.empty_like(x)
    ops.rms_norm(out, x, weight, eps)

    ref_layer0 = x[0].float() * torch.rsqrt(
        x[0].float().pow(2).mean(-1, keepdim=True) + eps)

    torch.testing.assert_close(out[0].float(), ref_layer0, atol=1e-2,
                                rtol=1e-2)
    # Layer 1 used weight row 1 (all zeros), so its output must be exactly 0.
    assert out[1].float().abs().max().item() == 0.0


@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_rms_norm_batched_weight_validates_shapes(device: str) -> None:
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    x = torch.randn(4, 8, 128, dtype=torch.float)
    out = torch.empty_like(x)
    # Row-count mismatch: weight's outer dim must match input's outer dim.
    with pytest.raises(RuntimeError):
        torch.ops._C.rms_norm(out, x, torch.randn(3, 128), 1e-6)
    # Hidden-size mismatch.
    with pytest.raises(RuntimeError):
        torch.ops._C.rms_norm(out, x, torch.randn(4, 64), 1e-6)
    # weight must be 1D or 2D.
    with pytest.raises(RuntimeError):
        torch.ops._C.rms_norm(out, x, torch.randn(4, 8, 128), 1e-6)
