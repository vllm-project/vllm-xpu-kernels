# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

import tests.register_ops as ops


def ggml_dequantize(
    weight: torch.Tensor,
    quant_type: int,
    m: int,
    n: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    return ops.ggml_dequantize(weight, quant_type, m, n, dtype)