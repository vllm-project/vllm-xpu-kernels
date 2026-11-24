# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for miscellaneous utilities
"""

from typing import Optional

import pytest
import torch

from tests.ops.rotary_embedding_op import RotaryEmbedding
from tests.utils import opcheck


def rotary_embedding_opcheck(rot,
                             positions: torch.Tensor,
                             query: torch.Tensor,
                             key: Optional[torch.Tensor] = None,
                             offsets: Optional[torch.Tensor] = None):
    cos_sin_cache = rot.cos_sin_cache.to(query.device, dtype=query.dtype)

    # ops.rotary_embedding()/batched_rotary_embedding()
    # are in-place operations that update the query and key tensors.
    if offsets is not None:
        raise NotImplementedError(
            "batched_rotary_embedding is not implemented yet.")
    else:
        opcheck(torch.ops._C.rotary_embedding,
                (positions, query, key, rot.head_size, cos_sin_cache,
                 rot.is_neox_style))


#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "max_position": [11],
        "head_size": [32],
        "seq_len": [11],
    },
}


@pytest.mark.parametrize("device", ["xpu"])
@pytest.mark.parametrize("max_position", [11, 4096, 32768])
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("rotary_dim", [32])
@pytest.mark.parametrize("head_size", [32, 108])
@pytest.mark.parametrize("seq_len", [11, 1024])
@pytest.mark.parametrize("use_key", [True, False])
@pytest.mark.parametrize("head_stride_is_contiguous", [True, False])
def test_rotary_embedding_opcheck(device, max_position, is_neox_style,
                                  rotary_dim, head_size, seq_len, use_key,
                                  head_stride_is_contiguous):
    batch_size = 1
    base = 10000
    num_heads = 7
    rot = RotaryEmbedding(head_size, rotary_dim, max_position, base,
                          is_neox_style, torch.float32)

    positions = torch.randint(0,
                              max_position, (batch_size, seq_len),
                              device=device)
    head_stride = head_size + (64 if head_stride_is_contiguous else 0)

    query = torch.randn(batch_size,
                        seq_len,
                        num_heads,
                        head_stride,
                        dtype=torch.float32,
                        device=device)
    key = torch.randn_like(query) if use_key else None
    query = query[..., :head_size]
    key = key[..., :head_size] if use_key else None

    rotary_embedding_opcheck(rot, positions, query, key)

    # if we have a contiguous head stride, test the alternate
    # [..., num_heads * head_dim] shape/layout
    if head_stride_is_contiguous:
        rotary_embedding_opcheck(
            rot, positions, query.flatten(start_dim=-2),
            key.flatten(start_dim=-2) if use_key else None)


@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("head_size,rotary_dim", [(64, 64), (128, 64)])
@pytest.mark.parametrize("head_pad", [0, 1])
@pytest.mark.parametrize("num_tokens", [7, 1024])
@torch.inference_mode()
def test_rotary_embedding_accuracy(dtype, is_neox_style, head_size,
                                   rotary_dim, head_pad, num_tokens):
    # head_pad=1 gives odd head/token strides, so the vectorized launch is
    # rejected and the element-wise fallback runs with odd addresses.
    torch.manual_seed(0)
    device = "xpu"
    num_heads, num_kv_heads, max_position = 8, 2, 4096
    rot = RotaryEmbedding(head_size, rotary_dim, max_position, 10000,
                          is_neox_style, torch.float32)
    positions = torch.randint(0, max_position, (num_tokens, ), device=device)
    query = torch.randn(num_tokens, num_heads, head_size + head_pad,
                        dtype=dtype, device=device)[..., :head_size]
    key = torch.randn(num_tokens, num_kv_heads, head_size + head_pad,
                      dtype=dtype, device=device)[..., :head_size]

    # The kernel reads cos/sin in dtype and computes in float, so the float32
    # reference uses the same rounded cache.
    cos_sin_cache = rot.cos_sin_cache.to(device, dtype=dtype)
    rot.cos_sin_cache = cos_sin_cache.float()
    ref_q, ref_k = rot.forward_native(positions,
                                      query.float().contiguous(),
                                      key.float().contiguous())

    # Keep the (possibly odd) strides of query and key in the kernel inputs.
    q = torch.empty_strided(query.shape, query.stride(), dtype=dtype,
                            device=device)
    k = torch.empty_strided(key.shape, key.stride(), dtype=dtype,
                            device=device)
    q.copy_(query)
    k.copy_(key)
    torch.ops._C.rotary_embedding(positions, q, k, head_size, cos_sin_cache,
                                  is_neox_style)

    tol = {torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 1e-2}
    atol = rtol = tol[dtype]
    torch.testing.assert_close(q.float(), ref_q, atol=atol, rtol=rtol)
    torch.testing.assert_close(k.float(), ref_k, atol=atol, rtol=rtol)
