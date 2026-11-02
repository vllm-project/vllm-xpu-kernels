# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import tests.register_ops as ops  # noqa: F401


def scatter_row_native(
    dst: torch.Tensor,
    src: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    """Reference: dst[index[i], :] = src[i, :] for each row i."""
    out = dst.clone()
    out[index] = src
    return out


def scatter_row(
    dst: torch.Tensor,
    src: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    torch.ops._xpu_C.scatter_row(dst, src, index)
    return dst
