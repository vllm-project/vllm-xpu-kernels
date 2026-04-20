# SPDX-License-Identifier: Apache-2.0
import torch

from vllm_xpu_kernels import _xpu_C  # noqa: F401


def cutlass_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """CUTLASS bf16 GEMM on Intel XE2 GPUs.

    Computes D = A @ B (follows torch.mm convention)

    B layout is detected from strides:
      - Row-major [K, N] (contiguous): uses RR kernel
      - Column-major [K, N] (i.e. B.T is contiguous): uses RC kernel (faster)

    Args:
        A: [M, K] bf16 tensor, must be contiguous
        B: [K, N] bf16 tensor, row-major or column-major

    Returns:
        D: [M, N] bf16 tensor
    """
    return torch.ops._xpu_C.cutlass_gemm(A, B)
