# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

import torch

import tests.register_ops as ops

# Add parent directory to Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #noqa: E501


def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    num_token_padding: Optional[int] = None,
    scale_ub: Optional[torch.Tensor] = None,
    use_per_token_if_dynamic: bool = False,
    output: Optional[torch.Tensor] = None,
    fp8_dtype: torch.dtype = torch.float8_e5m2,
    group_shape: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensors for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8
        scale: Optional scaling factor for the FP8 quantization
        scale_ub: Optional upper bound for scaling factor in dynamic
            per token case
        num_token_padding: If specified, pad the first dimension
            of the output to at least this value.
        use_per_token_if_dynamic: Whether to do per_tensor or per_token
            in the dynamic quantization case.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    shape: Union[tuple[int, int], torch.Size] = input.shape
    out_dtype: torch.dtype = fp8_dtype
    if num_token_padding:
        shape = (max(num_token_padding, input.shape[0]), shape[1])
    if output is None:
        output = torch.empty(shape, device=input.device, dtype=out_dtype)
    else:
        assert num_token_padding is None, \
            "padding not supported if output passed in"
        assert output.dtype == out_dtype

    if scale is None:
        if use_per_token_if_dynamic:
            scale = torch.zeros((shape[0], 1),
                                device=input.device,
                                dtype=torch.float32)
            ops.dynamic_per_token_scaled_fp8_quant(output, input.contiguous(),
                                                   scale, scale_ub)
        else:
            scale = torch.empty(1, device=input.device, dtype=torch.float32)
            ops.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        ops.static_scaled_fp8_quant(output, input, scale, group_shape)

    return output, scale


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.float8_e4m3fn,
    out_q: torch.Tensor | None = None,
    column_major_scales: bool = False,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.
    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dtype of output tensor. Note that only `torch.float8_e4m3fn`
        is supported for now.
        out_q: Optional output tensor. If not provided, function will create.
        column_major_scales: Outputs scales in column major.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
        scaling factor.
    """

    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}")
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    assert out_q is None or out_q.shape == x.shape
    x_q = out_q
    if x_q is None:
        x_q = torch.empty_like(x, device=x.device, dtype=dtype)

    if column_major_scales:
        shape = (x.shape[-1] // group_size, ) + x.shape[:-1]
        x_s = torch.empty(shape, device=x.device,
                          dtype=torch.float32).permute(-1, -2)
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size, )
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    # TODO(bnell): this causes some fp8 moe test to fail.
    torch.ops._C.per_token_group_fp8_quant(x, x_q, x_s, group_size, eps,
                                           fp8_min, fp8_max, use_ue8m0)

    if use_ue8m0:
        x_s = x_s.to(torch.float8_e8m0fnu)

    return x_q, x_s


class GroupShape:
    row: int
    col: int


# Normalize the group_shape to the full extent for any dims that are -1
def _normalize_quant_group_shape(x: torch.Tensor, group_shape: GroupShape):
    # -1 means full extent
    return (
        group_shape[0] if group_shape[0] > 0 else x.shape[-2],
        group_shape[1] if group_shape[1] > 0 else x.shape[-1],
    )


# Quantize assuming once scale per group of elements with shape group_shape,
# example group shapes:
#  * (-1, -1)   for per-tensor quantization
#  * (1, -1)    for per-row quantization
#  * (-1, 1)    for per-column quantization
#  * (128, 128) for 128x128 deepseek style block quantization
#  * (1, 128)   for deepseek style activation quantization
#               (i.e. per-token-per-group)
def scaled_quantize(
    x: torch.Tensor,
    group_shape: GroupShape,
    quant_dtype: torch.dtype,
    compute_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x: Input tensor to quantize
        group_shape: Shape of quantization groups
        quant_dtype: Target quantized dtype (e.g., torch.float8_e4m3fn)
        compute_dtype: Optional dtype for intermediate computations.
            If None, uses input dtype. Use torch.float32 for higher precision.
    """
    group_shape = _normalize_quant_group_shape(x, group_shape)
    assert quant_dtype.is_floating_point, (
        "currently `scaled_quantize` only supports floating point dtypes "
        "but could be extended to support other dtypes")

    finfo = torch.finfo(quant_dtype)

    # Convert to compute dtype if specified
    x_compute = x if compute_dtype is None else x.to(compute_dtype)

    # Reshape (M, N) into (BLK_M, BLOCK_SIZE_M, BLK_N, BLOCK_SIZE_N)
    assert x.ndim == 2
    assert x.shape[0] % group_shape[0] == 0 and x.shape[1] % group_shape[1] == 0
    blk_m, blk_n = x.shape[0] // group_shape[0], x.shape[1] // group_shape[1]
    x_blkd = x_compute.reshape(blk_m, group_shape[0], blk_n, group_shape[1])

    # Permute to (BLK_M, BLK_N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    x_blkd_permd = x_blkd.permute(0, 2, 1, 3)
    # Flatten to (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N)
    x_blkd_permd = x_blkd_permd.flatten(start_dim=2)

    # Compute scales
    min_val, max_val = x_blkd_permd.aminmax(dim=-1)
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax

    # Apply scale and convert from:
    # (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N) to (M, N)
    x_scl_sat = ((x_blkd_permd * scale.unsqueeze(-1)).clamp(
        min=finfo.min,
        max=finfo.max).reshape(blk_m, blk_n, group_shape[0],
                               group_shape[1]).permute(0, 2, 1,
                                                       3).reshape(x.shape))
    return x_scl_sat.to(quant_dtype).contiguous(), scale.float().reciprocal()
