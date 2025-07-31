# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

DTYPES = [torch.half, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES)
def test_flash_attn_varlen_func(dtype):
    torch.set_default_device("xpu")
    batch_size = 1
    seq_len = 4
    num_heads = 8
    head_dim = 16

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype)

    max_seqlen_q = seq_len
    cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32)
    max_seqlen_k = seq_len
    cu_seqlens_k = cu_seqlens_q

    # Call the flash attention function
    output = flash_attn_varlen_func(q, k, v, max_seqlen_q, cu_seqlens_q,
                                    max_seqlen_k, cu_seqlens_k)

    assert output is not None
    assert output.dtype == dtype
    assert output.shape == (batch_size, seq_len, num_heads, head_dim)
