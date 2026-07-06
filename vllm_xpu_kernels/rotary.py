# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Python wrapper for apply_rotary_emb SYCL kernel.

The underlying _C.apply_rotary_emb kernel expects 3D tensors:
  input/output: [num_tokens, num_heads, head_size]
  cos/sin:      [num_tokens, rot_dim/2]

This wrapper handles arbitrary input shapes [..., num_heads, head_size]
and cos/sin broadcasting from [seq_len, rot_dim/2] to [num_tokens, rot_dim/2].
"""

import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401 — registers torch.ops._xpu_C ops


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox: bool = False,
) -> torch.Tensor:
    """Apply rotary embedding with automatic shape handling.

    Args:
        x: [..., num_heads, head_size] input tensor
        cos: [seq_len, rot_dim/2] or [..., rot_dim/2]
        sin: [seq_len, rot_dim/2] or [..., rot_dim/2]
        is_neox: True for GPT-NeoX style, False for GPT-J (interleaved)

    Returns:
        Tensor same shape as x with rotary embedding applied.
    """
    head_size = x.shape[-1]
    num_heads = x.shape[-2]
    num_tokens = x.numel() // (num_heads * head_size)

    # Flatten to [num_tokens, num_heads, head_size]
    input_3d = x.reshape(num_tokens, num_heads, head_size).contiguous()

    # Flatten cos/sin to [cos_tokens, rot_dim/2]
    half_rot = cos.shape[-1]
    cos_tokens = cos.numel() // half_rot
    cos_2d = cos.reshape(cos_tokens, half_rot).contiguous()
    sin_2d = sin.reshape(cos_tokens, half_rot).contiguous()

    # Expand cos/sin if fewer tokens than input (batch broadcasting)
    if cos_tokens < num_tokens:
        assert num_tokens % cos_tokens == 0, (
            f"num_tokens ({num_tokens}) must be divisible"
            f" by cos_tokens ({cos_tokens})"
        )
        repeats = num_tokens // cos_tokens
        cos_2d = cos_2d.repeat(repeats, 1)
        sin_2d = sin_2d.repeat(repeats, 1)

    output = torch.empty_like(input_3d)
    torch.ops._xpu_C.apply_rotary_emb(output, input_3d, cos_2d, sin_2d, is_neox)
    return output.reshape(x.shape)
