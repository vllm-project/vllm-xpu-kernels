# SPDX-License-Identifier: Apache-2.0

import torch

import tests.register_ops as ops  # noqa: F401


def per_token_group_quant_mxfp4(
    x: torch.Tensor,
    group_size: int = 32,
    eps: float = 1e-10,
    column_major_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x:                    Input tensor of shape [M, N].
        group_size:           Block size; must be 32 for the MX specification.
        eps:                  Preventing log2(0).
        column_major_scales:  When True, allocate the scale tensor in
                              column-major order so that scales for consecutive
                              tokens of the same group column are contiguous.
                              Required by certain GEMM back-ends.

    Returns:
        Tuple of:
          * output_q [M, N/2]         – float4_e2m1fn_x2 packed FP4 tensor
          * output_s [M, N/group_size] – float8_e8m0fnu UE8M0-rounded scale
    """
    assert x.ndim == 2, "input must be 2-D"
    assert x.shape[-1] % group_size == 0, (
        f"last dimension {x.shape[-1]} must be divisible by group_size "
        f"{group_size}")
    assert x.stride(-1) == 1, "input groups must be contiguous"

    M, N = x.shape

    # Packed FP4 output: two nibbles per byte
    out_q = torch.empty(M, N // 2, device=x.device, dtype=torch.uint8)

    # Scale tensor – row-major or column-major
    if column_major_scales:
        # Allocate (N//group_size, M), then permute to (M, N//group_size).
        # The underlying storage is column-major: stride(0)=1, stride(1)=M.
        shape = (N // group_size, M)
        out_s = torch.empty(shape, device=x.device,
                            dtype=torch.float32).permute(-1, -2)
    else:
        out_s = torch.empty(M,
                            N // group_size,
                            device=x.device,
                            dtype=torch.float32)

    torch.ops._C.per_token_group_quant_mxfp4(x.contiguous(), out_q, out_s,
                                             group_size, eps)

    out_q = out_q.view(torch.float4_e2m1fn_x2)
    out_s = out_s.to(dtype=torch.float8_e8m0fnu,
                     memory_format=torch.preserve_format)
    if column_major_scales:
        # Verify that the column-major-like strides are preserved after casting.
        assert out_s.stride(0) == 1 and out_s.stride(1) == M, (
            "scale tensor must retain column-major-like strides")
    return out_q, out_s
