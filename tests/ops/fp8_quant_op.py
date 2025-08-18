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
            scale = torch.empty((shape[0], 1),
                                device=input.device,
                                dtype=torch.float32)
            ops.dynamic_per_token_scaled_fp8_quant(output, input.contiguous(),
                                                   scale, scale_ub)
        else:
            scale = torch.zeros(1, device=input.device, dtype=torch.float32)
            ops.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        assert scale.numel() == 1, f"{scale.shape}"
        ops.static_scaled_fp8_quant(output, input, scale)

    return output, scale
