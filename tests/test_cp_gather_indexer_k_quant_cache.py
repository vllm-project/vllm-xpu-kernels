# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401
from tests import register_ops as ops

eps = 1e-4

DEVICE = "xpu"
BATCH_SEQ_LENS = [[1], [3], [6], [3, 4], [5, 4], [4, 3], [4, 4], [7, 8, 5]]
HEAD_DIMS = [128, 256, 512]
QUANT_BLOCK_SIZES = [128]
BLOCK_SIZES = [4, 16]
SCALE_FMTS = ["ue8m0", "fp8e4m3"]
# TODO: will add back torch.float16
# after fp8_e4m3 acc is verified
DTYPES = [torch.float32, torch.bfloat16]

# override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "batch_seq_lens": [[1]],
        "head_dim": [128],
        "quant_block_size": [128],
        "block_size": [16],
        "scale_fmt": ["ue8m0"],
        "dtype": [torch.float32],
    },
}


def ref_cp_gather_indexer_k_quant_cache(
        kv_cache: torch.Tensor,  # [num_cache_blocks, block_size, cache_stride]
        dst_k: torch.Tensor,  # [num_tokens, head_dim]
        dst_scale: torch.Tensor,  # [num_tokens, num_groups * 4]
        block_table: torch.Tensor,  # [batch_size, max_blocks_per_seq]
        cu_seq_lens: torch.Tensor,  # [batch_size + 1] 
) -> None:
    head_dim = dst_k.shape[1]
    num_groups = dst_scale.shape[
        1] // 4  # dst_scale stores float32 as uint8 bytes
    quant_block_size = head_dim // num_groups
    block_size = kv_cache.shape[1]
    cache_stride = kv_cache.shape[2]
    batch_size = block_table.shape[0]

    kv_flat = kv_cache.view(-1)  # flat uint8 view
    dst_k_flat = dst_k.view(-1)  # flat uint8 view
    dst_s_f32 = dst_scale.view(-1).view(
        torch.float32)  # [num_tokens * num_groups]

    block_stride = block_size * cache_stride  # in uint8 elements

    for batch_id in range(batch_size):
        seq_start = cu_seq_lens[batch_id].item()
        seq_end = cu_seq_lens[batch_id + 1].item()

        for inbatch_pos in range(seq_end - seq_start):
            token_idx = seq_start + inbatch_pos
            block_slot = inbatch_pos // block_size
            block_offset = inbatch_pos % block_size
            block_idx = block_table[batch_id, block_slot].item()

            # FP8 bytes: kv_cache[block_idx, block_offset, :head_dim]
            src_fp8_start = block_idx * block_stride + block_offset * head_dim
            dst_fp8_start = token_idx * head_dim
            dst_k_flat[dst_fp8_start: dst_fp8_start + head_dim] = \
                kv_flat[src_fp8_start: src_fp8_start + head_dim]

            # Scales: one float32 per quant group
            for g in range(num_groups):
                cache_inblock_offset = block_offset * head_dim + \
                                        g * quant_block_size
                src_scale_byte = (block_idx * block_stride +
                                  block_size * head_dim +
                                  cache_inblock_offset * 4 // quant_block_size)
                dst_scale_float_idx = token_idx * num_groups + g
                kv_f32 = kv_flat.view(torch.float32)
                dst_s_f32[dst_scale_float_idx] = kv_f32[src_scale_byte // 4]


@pytest.mark.parametrize("batch_seq_lens", BATCH_SEQ_LENS)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("quant_block_size", QUANT_BLOCK_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("scale_fmt", SCALE_FMTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_cp_gather_indexer_k_quant_cache_correctness(batch_seq_lens,
                                                     block_size, head_dim,
                                                     quant_block_size,
                                                     scale_fmt, dtype):

    num_tokens = sum(batch_seq_lens)
    num_groups = head_dim // quant_block_size
    # FP8 bytes + float32 scales, in uint8 units
    cache_stride = head_dim + num_groups * 4

    max_blocks_per_seq = max(
        (seq_len + block_size - 1) // block_size for seq_len in batch_seq_lens)
    total_blocks = sum(
        (seq_len + block_size - 1) // block_size for seq_len in batch_seq_lens)

    torch.manual_seed(0)
    k = torch.randn((num_tokens, head_dim), device=DEVICE, dtype=dtype)

    # Build block_table and slot_mapping
    block_table_list = []
    slot_lists = []
    next_block = 0

    for seq_len in batch_seq_lens:
        num_seq_blocks = (seq_len + block_size - 1) // block_size
        seq_blocks = list(range(next_block, next_block + num_seq_blocks))
        next_block += num_seq_blocks

        # Pad to max_blocks_per_seq with −1 (unused)
        padded = seq_blocks + [-1] * (max_blocks_per_seq - num_seq_blocks)
        block_table_list.append(padded)

        for pos in range(seq_len):
            block_idx_for_pos = seq_blocks[pos // block_size]
            block_offset_for_pos = pos % block_size
            slot_lists.append(block_idx_for_pos * block_size +
                              block_offset_for_pos)

    # Clamp negative padding to 0 for kv_cache allocation purposes only
    block_table_cpu = torch.tensor(
        [[max(0, b) for b in row] for row in block_table_list],
        dtype=torch.int32,
    )
    block_table = block_table_cpu.to(DEVICE)

    slot_mapping = torch.tensor(slot_lists, dtype=torch.int64, device=DEVICE)

    # Cumulative sequence lengths
    cu_seq_lens_list = [0]
    for seq_len in batch_seq_lens:
        cu_seq_lens_list.append(cu_seq_lens_list[-1] + seq_len)
    cu_seq_lens = torch.tensor(cu_seq_lens_list,
                               dtype=torch.int32,
                               device=DEVICE)

    kv_cache = torch.zeros((total_blocks, block_size, cache_stride),
                           dtype=torch.uint8,
                           device=DEVICE)
    ops.indexer_k_quant_and_cache(k, kv_cache, slot_mapping, quant_block_size,
                                  scale_fmt)

    dst_k_ref = torch.zeros((num_tokens, head_dim),
                            dtype=torch.uint8,
                            device=DEVICE)
    dst_s_ref = torch.zeros((num_tokens, num_groups * 4),
                            dtype=torch.uint8,
                            device=DEVICE)
    dst_k_xpu = torch.zeros_like(dst_k_ref)
    dst_s_xpu = torch.zeros_like(dst_s_ref)

    # Run reference
    ref_cp_gather_indexer_k_quant_cache(kv_cache, dst_k_ref, dst_s_ref,
                                        block_table, cu_seq_lens)

    # Run XPU kernel
    ops.cp_gather_indexer_k_quant_cache(kv_cache, dst_k_xpu, dst_s_xpu,
                                        block_table, cu_seq_lens)

    # Compare token by token for clear diagnostics
    all_fp8_match = True
    all_scale_match = True

    for tok in range(num_tokens):
        ref_fp8 = dst_k_ref[tok]
        out_fp8 = dst_k_xpu[tok]
        ref_sc = dst_s_ref[tok].view(torch.float32)
        out_sc = dst_s_xpu[tok].view(torch.float32)

        fp8_ok = torch.equal(ref_fp8, out_fp8)
        scale_ok = torch.allclose(ref_sc, out_sc, atol=1e-5)

        if not fp8_ok:
            all_fp8_match = False
            print(f"  [token {tok}] FP8 MISMATCH")
            diff = (ref_fp8.view(torch.float8_e4m3fn).float() -
                    out_fp8.view(torch.float8_e4m3fn).float()).abs()
            print(f"    max FP8 abs-diff: {diff.max().item()}")

        if not scale_ok:
            all_scale_match = False
            print(f"  [token {tok}] Scale MISMATCH")
            print(f"    ref: {ref_sc}")
            print(f"    out: {out_sc}")

    tag = (f"batch_seq_lens={batch_seq_lens} block_size={block_size} "
           f"head_dim={head_dim} quant_block_size={quant_block_size} "
           f"scale_fmt={scale_fmt}")
    print(f"\n[{tag}]")
    print(f"  FP8  all-match : {all_fp8_match}")
    print(f"  Scale all-match: {all_scale_match}")

    assert all_fp8_match, f"FP8 data mismatch [{tag}]"
    assert all_scale_match, f"Scale mismatch [{tag}]"
