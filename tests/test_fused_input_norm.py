# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from tests.register_ops import fused_input_norm
from tests.utils import opcheck

DTYPES = [torch.half, torch.bfloat16, torch.float32]
NUM_PATCHES = [1, 7, 256, 20000]  # Qwen2.5-VL pixel_values-like sizes
# patch_size per channel; row_size = channel * patch_size.
PATCH_SIZES = [1, 392, 588]  # 392 -> 1176/3 (Qwen2.5-VL)
CHANNELS = [3]
SEEDS = [0]
XPU_DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

# override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_patches": [7],
        "patch_size": [392],
    },
}


def _ref_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    channel: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    # Mirror FusedInputNorm.forward: reshape to [patches, channel,
    # patch_size], run batch_norm with running_mean=0, running_var=1,
    # eps=0 (== per-channel affine), cast to out_dtype.
    patches, size = x.shape
    patch_size = size // channel
    xf = x.view(patches, channel, patch_size).to(torch.float32)
    running_mean = torch.zeros_like(weight)
    running_var = torch.ones_like(weight)
    y = F.batch_norm(
        xf,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        training=False,
        eps=0.0,
    )
    return y.view(patches, size).to(out_dtype)


@pytest.mark.parametrize("num_patches", NUM_PATCHES)
@pytest.mark.parametrize("patch_size", PATCH_SIZES)
@pytest.mark.parametrize("channel", CHANNELS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_fused_input_norm(
    num_patches: int,
    patch_size: int,
    channel: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    size = channel * patch_size
    x = torch.randint(0, 256, (num_patches, size), dtype=torch.uint8)

    # Emulate image_std / image_mean folded into weight/bias, like
    # FusedInputNorm: weight = 1/std', bias = -mean'/std'.
    image_mean = torch.rand(channel, dtype=torch.float32) * 0.5
    image_std = torch.rand(channel, dtype=torch.float32) * 0.5 + 0.5
    rescale_factor = 1.0 / 255.0
    mean_t = image_mean * (1.0 / rescale_factor)
    std_t = image_std * (1.0 / rescale_factor)
    weight = (1.0 / std_t).contiguous()
    bias = (-mean_t / std_t).contiguous()

    ref_out = _ref_forward(x, weight, bias, channel, dtype)

    out = torch.empty((num_patches, size), dtype=dtype)
    fused_input_norm(out, x, weight, bias)

    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-5
    else:
        atol, rtol = 1e-2, 1e-2
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)

    opcheck(torch.ops._C.fused_input_norm, (out, x, weight, bias))
