import pytest
import torch
from typing import List, Optional, Tuple
from einops import rearrange
import torch.nn.functional as F

from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

DTYPES = [torch.half, torch.bfloat16]
dtype = torch.half

def assert_close_verbose(a, b, rtol=1e-2, atol=1e-1):
    if a.shape != b.shape:
        raise AssertionError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # Compute absolute and relative differences
    diff = torch.abs(a - b)
    tol = atol + rtol * torch.abs(b)

    # Find mismatches
    mismatch_mask = diff > tol
    if mismatch_mask.any():
        idx = mismatch_mask.nonzero(as_tuple=False)
        for i in idx:
            coord = tuple(i.tolist())
            val_a = a[coord].item()
            val_b = b[coord].item()
            d = diff[coord].item()
            t = tol[coord].item()
            print(f"Mismatch at {coord}: a={val_a}, b={val_b}, diff={d}, tol={t}")
        # raise AssertionError(f"Tensors are not close. {mismatch_mask.sum().item()} elements differ.")
    else:
        print("Tensors are close within tolerance.")

def ref_chunked_prefill(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    casual: Optional[bool] = False,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx:start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = torch.triu(empty_mask,
                                             diagonal=kv_len -
                                             (query_len + sliding_window) +
                                             1).bool().logical_not()
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        if casual:
            attn.masked_fill_(mask, float("-inf"))
        # print(attn[:, :16, -1])
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)

torch.set_default_device("xpu")
torch.manual_seed(0)
batch_size = 3
seq_len = 512 - 1
seq_len_2 = 64 - 1
q_num_heads = 16
kv_num_heads = q_num_heads // 8
head_dim = 64

# seq_lens_q = torch.randint(1, max_seqlen_q, (batch_size,), dtype=torch.int32)
seq_lens_q = torch.tensor([1, 1, 1], dtype=torch.int32)
seq_lens_k = torch.tensor([523, 37, 2011], dtype=torch.int32)
cu_seq_lens_q = torch.cat([torch.tensor([0], dtype=torch.int32), seq_lens_q])
cu_seqlens_q = torch.cumsum(cu_seq_lens_q, 0).to(torch.int32)
max_seqlen_q = max(seq_lens_q)

cu_seq_lens_k = torch.cat([torch.tensor([0], dtype=torch.int32), seq_lens_k])
cu_seqlens_k = torch.cumsum(cu_seq_lens_k, 0).to(torch.int32)
max_seqlen_k = max(seq_lens_k)

block_size = 64
num_blocks = (max_seqlen_k + block_size - 1) // block_size
max_num_blocks_per_seq = (max_seqlen_k + block_size - 1) // block_size

block_tables = torch.randint(0, num_blocks, (batch_size, max_num_blocks_per_seq), dtype=torch.int32)

print(block_tables)
print(cu_seqlens_q)
print(cu_seqlens_k)
print(seq_lens_q)
print(seq_lens_k)

q = torch.randn(cu_seqlens_q[-1], q_num_heads, head_dim, dtype=dtype)
k = torch.randn(num_blocks, block_size, kv_num_heads, head_dim, dtype=dtype)
v = torch.randn(num_blocks, block_size, kv_num_heads, head_dim, dtype=dtype)

# Call the flash attention function
output= flash_attn_varlen_func(q, k, v, max_seqlen_q, cu_seqlens_q,
                                max_seqlen_k, cu_seqlens_k, block_table=block_tables)
output_ref = ref_chunked_prefill(q, k, v, seq_lens_q, seq_lens_k, block_tables, head_dim**(-0.5))

assert output is not None

# print(output[315, 7, 48].cpu())
# print(output_ref[315, 7, 48].cpu())

torch.testing.assert_close(output.float().cpu(), output_ref.float().cpu(), atol=1e-2, rtol=1e-2)