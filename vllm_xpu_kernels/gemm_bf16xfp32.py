# SPDX-License-Identifier: Apache-2.0
"""BF16xFP32 GEMM: emulated FP32-precision GEMM using two BF16 DPAS operations.

  result = A @ B_high + (A @ B_low) * scale

where B_high, B_low are BF16 decompositions of an FP32 weight matrix B.
The kernel expects B_high/B_low in transposed [K, N] layout (see
:func:`split_fp32_weight`):
  B_high = bf16(B)
  B_low  = bf16((B - fp32(B_high)) / scale)
"""

import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401  (registers torch.ops._xpu_C.*)


def split_fp32_weight(
    weight: torch.Tensor, scale: float = 1.0 / 256.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split FP32 weight into two BF16 components for DualGemm.

    The kernel computes ``Y = X @ W.T`` (F.linear semantics) as
    ``Y[m,n] = sum_k A[m,k] * B[k,n]``, so the returned components are
    transposed to ``[K, N]`` (N-contiguous) as the kernel expects.

    Args:
        weight: [N, K] FP32 weight tensor (nn.Linear layout: out x in)
        scale: Fixed scale factor (default 1/256 = 2^-8, matches CUDA hpc-ops;
            W_low stores the amplified residual, reconstructed as W_low * scale)

    Returns:
        (W_high, W_low): both [K, N] BF16 tensors (transposed & contiguous)
        Reconstruction: weight ≈ fp32(W_high.T) + fp32(W_low.T) * scale
    """
    if weight.dtype != torch.float32:
        raise ValueError(f"weight must be FP32 (got {weight.dtype})")
    if scale == 0.0:
        raise ValueError("scale must be non-zero")
    W_high = weight.to(torch.bfloat16)
    residual = weight - W_high.float()
    W_low = (residual / scale).to(torch.bfloat16)
    # Transpose to [K, N] contiguous for the kernel's B layout.
    return W_high.t().contiguous(), W_low.t().contiguous()


def gemm_bf16xfp32(
    A: torch.Tensor,
    B_high: torch.Tensor,
    B_low: torch.Tensor,
    scale: float = 1.0 / 256.0,
) -> torch.Tensor:
    """BF16xFP32 GEMM via DualGemm.

    Computes ``Y = A @ (W_high + W_low * scale)`` where ``B_high``/``B_low``
    are the [K, N] transposed BF16 decompositions of an FP32 weight
    (see :func:`split_fp32_weight`). This equals ``F.linear(A, W)`` for the
    original [N, K] weight ``W``.

    Args:
        A: [M, K] BF16 activation tensor
        B_high: [K, N] BF16 (high bits of FP32 weight, transposed)
        B_low: [K, N] BF16 (low bits of FP32 weight, normalized by scale)
        scale: The scale factor used during weight splitting

    Returns:
        [M, N] FP32 output tensor
    """
    return torch.ops._xpu_C.gemm_bf16xfp32(A, B_high, B_low, scale)
