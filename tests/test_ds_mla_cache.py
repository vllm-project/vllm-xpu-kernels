# SPDX-License-Identifier: Apache-2.0
"""Tests for the DSV3.2 packed ("fp8_ds_mla") MLA KV cache ops."""

import pytest
import torch

from tests.register_ops import (concat_and_cache_mla,
                                cp_gather_and_upconvert_fp8_kv_cache)
from tests.utils import seed_everything

KERNEL_DEVICE = "xpu"

NOPE_DIM = 512
PE_DIM = 64
TILE = 128
NUM_TILES = NOPE_DIM // TILE
SCALES_OFFSET = NOPE_DIM
ROPE_OFFSET = SCALES_OFFSET + NUM_TILES * 4
ENTRY_BYTES = ROPE_OFFSET + PE_DIM * 2
SCALE_DIVISOR = 448.0

DTYPES = [torch.bfloat16, torch.half]
BLOCK_SIZES = [16, 64]


def _ref_pack(kv_c: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
    """Reference 656-byte packing of one batch of tokens."""
    num_tokens = kv_c.shape[0]
    nope = kv_c.float().view(num_tokens, NUM_TILES, TILE)
    scales = (nope.abs().amax(dim=-1) / SCALE_DIVISOR).clamp(
        min=torch.finfo(torch.float32).tiny)
    quantized = (nope / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)

    packed = torch.empty((num_tokens, ENTRY_BYTES),
                         dtype=torch.uint8,
                         device=kv_c.device)
    packed[:, :SCALES_OFFSET] = quantized.reshape(num_tokens,
                                                  NOPE_DIM).view(torch.uint8)
    packed[:, SCALES_OFFSET:ROPE_OFFSET] = scales.contiguous().view(torch.uint8)
    # The RoPE tail is always stored as BF16.
    packed[:, ROPE_OFFSET:] = k_pe.to(torch.bfloat16).contiguous().view(
        torch.uint8)
    return packed


def _ref_unpack(packed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    num_tokens = packed.shape[0]
    nope = (packed[:, :SCALES_OFFSET].view(torch.float8_e4m3fn).float().view(
        num_tokens, NUM_TILES, TILE))
    scales = packed[:, SCALES_OFFSET:ROPE_OFFSET].contiguous().view(
        torch.float32)
    out = torch.empty((num_tokens, NOPE_DIM + PE_DIM),
                      dtype=dtype,
                      device=packed.device)
    out[:, :NOPE_DIM] = (nope * scales.unsqueeze(-1)).reshape(
        num_tokens, NOPE_DIM).to(dtype)
    out[:, NOPE_DIM:] = packed[:, ROPE_OFFSET:].contiguous().view(
        torch.bfloat16).to(dtype)
    return out


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_tokens", [1, 42])
@torch.inference_mode()
def test_concat_and_cache_ds_mla(dtype: torch.dtype, block_size: int,
                                 num_tokens: int) -> None:
    seed_everything(0)
    num_blocks = 8
    kv_c = torch.randn(num_tokens,
                       NOPE_DIM,
                       dtype=dtype,
                       device=KERNEL_DEVICE)
    k_pe = torch.randn(num_tokens, PE_DIM, dtype=dtype, device=KERNEL_DEVICE)
    kv_cache = torch.zeros(num_blocks,
                           block_size,
                           ENTRY_BYTES,
                           dtype=torch.uint8,
                           device=KERNEL_DEVICE)

    slots = torch.randperm(num_blocks * block_size,
                           device=KERNEL_DEVICE)[:num_tokens].to(torch.int64)
    # Mark the last token as padding; it must be left untouched.
    if num_tokens > 1:
        slots[-1] = -1

    scale = torch.ones(1, dtype=torch.float32, device=KERNEL_DEVICE)
    concat_and_cache_mla(kv_c, k_pe, kv_cache, slots, "fp8_ds_mla", scale)

    entries = kv_cache.view(-1, ENTRY_BYTES)
    written = slots >= 0
    ref = _ref_pack(kv_c[written], k_pe[written])
    torch.testing.assert_close(entries[slots[written]].cpu(), ref.cpu())
    if num_tokens > 1:
        untouched = torch.ones(num_blocks * block_size,
                               dtype=torch.bool,
                               device=KERNEL_DEVICE)
        untouched[slots[written]] = False
        assert entries[untouched].sum().item() == 0


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("use_seq_starts", [False, True])
@torch.inference_mode()
def test_cp_gather_and_upconvert_fp8_kv_cache(block_size: int,
                                              use_seq_starts: bool) -> None:
    seed_everything(0)
    num_blocks = 16
    seq_lens = [1, block_size, block_size + 3, 2 * block_size]
    num_reqs = len(seq_lens)
    max_blocks = (max(seq_lens) + 2 * block_size) // block_size + 1

    kv_cache = torch.zeros(num_blocks,
                           block_size,
                           ENTRY_BYTES,
                           dtype=torch.uint8,
                           device=KERNEL_DEVICE)
    entries = kv_cache.view(-1, ENTRY_BYTES)
    total_slots = num_blocks * block_size
    kv_c = torch.randn(total_slots,
                       NOPE_DIM,
                       dtype=torch.bfloat16,
                       device=KERNEL_DEVICE)
    k_pe = torch.randn(total_slots,
                       PE_DIM,
                       dtype=torch.bfloat16,
                       device=KERNEL_DEVICE)
    entries.copy_(_ref_pack(kv_c, k_pe))

    block_table = torch.randint(0,
                                num_blocks, (num_reqs, max_blocks),
                                dtype=torch.int32,
                                device=KERNEL_DEVICE)
    workspace_starts = torch.tensor(
        [sum(seq_lens[:i]) for i in range(num_reqs)],
        dtype=torch.int32,
        device=KERNEL_DEVICE)
    seq_starts = None
    if use_seq_starts:
        seq_starts = torch.tensor([0, block_size, 1, 2 * block_size],
                                  dtype=torch.int32,
                                  device=KERNEL_DEVICE)

    total_tokens = sum(seq_lens)
    dst = torch.zeros(total_tokens,
                      NOPE_DIM + PE_DIM,
                      dtype=torch.bfloat16,
                      device=KERNEL_DEVICE)
    cp_gather_and_upconvert_fp8_kv_cache(kv_cache, dst, block_table,
                                         workspace_starts, num_reqs,
                                         seq_starts)

    src_slots = []
    for req in range(num_reqs):
        begin = 0 if seq_starts is None else int(seq_starts[req])
        for i in range(seq_lens[req]):
            token = begin + i
            block = int(block_table[req, token // block_size])
            src_slots.append(block * block_size + token % block_size)
    src_slots_t = torch.tensor(src_slots,
                               dtype=torch.int64,
                               device=KERNEL_DEVICE)

    ref = _ref_unpack(entries[src_slots_t], torch.bfloat16)
    torch.testing.assert_close(dst.cpu(), ref.cpu())


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_ds_mla_pack_then_gather(dtype: torch.dtype) -> None:
    """Pack with concat_and_cache_mla and read back with the gather op."""
    seed_everything(0)
    block_size = 16
    num_tokens = 2 * block_size
    kv_c = torch.randn(num_tokens, NOPE_DIM, dtype=dtype, device=KERNEL_DEVICE)
    k_pe = torch.randn(num_tokens, PE_DIM, dtype=dtype, device=KERNEL_DEVICE)
    k_pe[0, :4] = torch.tensor([1.0, -1.0, 0.5, 2.0], dtype=dtype)
    kv_cache = torch.zeros(2,
                           block_size,
                           ENTRY_BYTES,
                           dtype=torch.uint8,
                           device=KERNEL_DEVICE)
    slots = torch.arange(num_tokens, dtype=torch.int64, device=KERNEL_DEVICE)
    scale = torch.ones(1, dtype=torch.float32, device=KERNEL_DEVICE)
    concat_and_cache_mla(kv_c, k_pe, kv_cache, slots, "fp8_ds_mla", scale)

    dst = torch.zeros(num_tokens,
                      NOPE_DIM + PE_DIM,
                      dtype=torch.bfloat16,
                      device=KERNEL_DEVICE)
    block_table = torch.tensor([[0, 1]],
                               dtype=torch.int32,
                               device=KERNEL_DEVICE)
    workspace_starts = torch.zeros(1, dtype=torch.int32, device=KERNEL_DEVICE)
    cp_gather_and_upconvert_fp8_kv_cache(kv_cache, dst, block_table,
                                         workspace_starts, 1, None)

    torch.testing.assert_close(dst[:, NOPE_DIM:].cpu(),
                               k_pe.to(torch.bfloat16).cpu(),
                               atol=0,
                               rtol=0)
    torch.testing.assert_close(dst[:, :NOPE_DIM].float().cpu(),
                               kv_c.float().cpu(),
                               atol=0.1,
                               rtol=0.1)


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_concat_and_cache_ds_mla_non_byte_cache(
        cache_dtype: torch.dtype) -> None:
    """A cache with wider elements is still laid out as 656-byte entries."""
    seed_everything(0)
    num_blocks, block_size, num_tokens = 4, 16, 42
    elem = torch.empty(0, dtype=cache_dtype).element_size()
    kv_c = torch.randn(num_tokens,
                       NOPE_DIM,
                       dtype=torch.bfloat16,
                       device=KERNEL_DEVICE)
    k_pe = torch.randn(num_tokens,
                       PE_DIM,
                       dtype=torch.bfloat16,
                       device=KERNEL_DEVICE)
    kv_cache = torch.zeros(num_blocks,
                           block_size,
                           ENTRY_BYTES // elem,
                           dtype=cache_dtype,
                           device=KERNEL_DEVICE)
    slots = torch.randperm(num_blocks * block_size,
                           device=KERNEL_DEVICE)[:num_tokens].to(torch.int64)
    scale = torch.ones(1, dtype=torch.float32, device=KERNEL_DEVICE)
    concat_and_cache_mla(kv_c, k_pe, kv_cache, slots, "fp8_ds_mla", scale)

    entries = kv_cache.view(torch.uint8).view(-1, ENTRY_BYTES)
    torch.testing.assert_close(entries[slots].cpu(),
                               _ref_pack(kv_c, k_pe).cpu())
    untouched = torch.ones(num_blocks * block_size,
                           dtype=torch.bool,
                           device=KERNEL_DEVICE)
    untouched[slots] = False
    assert entries[untouched].sum().item() == 0


@torch.inference_mode()
def test_cp_gather_and_upconvert_rejects_host_seq_starts() -> None:
    kv_cache = torch.zeros(1,
                           16,
                           ENTRY_BYTES,
                           dtype=torch.uint8,
                           device=KERNEL_DEVICE)
    dst = torch.zeros(1,
                      NOPE_DIM + PE_DIM,
                      dtype=torch.bfloat16,
                      device=KERNEL_DEVICE)
    block_table = torch.zeros(1, 1, dtype=torch.int32, device=KERNEL_DEVICE)
    workspace_starts = torch.zeros(1, dtype=torch.int32, device=KERNEL_DEVICE)
    seq_starts = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="same device"):
        cp_gather_and_upconvert_fp8_kv_cache(kv_cache, dst, block_table,
                                             workspace_starts, 1, seq_starts)
