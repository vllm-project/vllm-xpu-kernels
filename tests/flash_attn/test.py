import pytest
import torch

from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

DTYPES = [torch.half, torch.bfloat16]
dtype = torch.half

torch.set_default_device("xpu")
batch_size = 1
seq_len = 512
num_heads = 8
head_dim = 128

max_seqlen_q = seq_len
cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32)
max_seqlen_k = seq_len
cu_seqlens_k = cu_seqlens_q

block_size = 128
num_blocks = max_seqlen_q // block_size
max_num_blocks_per_seq = seq_len // block_size

block_tables = torch.randint(0, num_blocks, (batch_size, max_num_blocks_per_seq), dtype=torch.int32)

print(block_tables)
print(cu_seqlens_q)

q = torch.randn(sum(cu_seqlens_q), num_heads, head_dim, dtype=dtype)
k = torch.randn(num_blocks, block_size, num_heads, head_dim, dtype=dtype)
v = torch.randn(num_blocks, block_size, num_heads, head_dim, dtype=dtype)

# Call the flash attention function
output= flash_attn_varlen_func(q, k, v, max_seqlen_q, cu_seqlens_q,
                                max_seqlen_k, cu_seqlens_k, block_table=block_tables)

assert output is not None
assert output.dtype == dtype
