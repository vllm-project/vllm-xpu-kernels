# SPDX-License-Identifier: Apache-2.0

import random

import pytest
import torch

from tests import register_ops as ops
from tests.register_ops import reshape_and_cache, reshape_and_cache_flash
from tests.utils import (_convert_from_fp8, create_kv_caches_with_random,
                         create_kv_caches_with_random_flash, opcheck,
                         seed_everything)

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [42]  # Arbitrary values for testing
NUM_LAYERS = [1]  # Arbitrary values for testing
NUM_HEADS = [8]  # Arbitrary values for testing
HEAD_SIZES = [64, 80, 120, 256]
BLOCK_SIZES = [8, 16, 32]

# Parameters for MLA tests.
KV_LORA_RANKS = [512]
QK_ROPE_HEAD_DIMS = [64]
NUM_TOKENS_MLA = [42]
BLOCK_SIZES_MLA = [16]
NUM_BLOCKS_MLA = [8]

# Arbitrary values for testing
# don't make it too large. e.g. [1024, 36000] will OOM
NUM_BLOCKS = [1024]

SEEDS = [0]
DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

KV_CACHE_DTYPE = ["auto"]  # FIXME: will add "fp8" when accuracy is improved


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@torch.inference_mode()
def test_reshape_and_cache(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    kv_cache_dtype: str,
) -> None:
    if kv_cache_dtype == "fp8" and head_size % 16:
        pytest.skip()

    # Note: torch.set_default_device("xpu:1") not works.
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)
    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping_lst = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long)

    qkv = torch.randn(num_tokens, 3, num_heads, head_size, dtype=dtype)
    _, key, value = qkv.unbind(dim=1)

    # Create the KV caches.
    key_caches, value_caches = create_kv_caches_with_random(
        num_blocks,
        block_size,
        1,
        num_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        seed=seed,
        device=device,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Using default kv_scale
    k_scale = (key.amax() / 64.0).to(torch.float32)
    v_scale = (value.amax() / 64.0).to(torch.float32)

    def permute_and_compact(x):
        return x.contiguous()

    key_cache_compact = permute_and_compact(key_cache)
    value_cache_compact = permute_and_compact(value_cache)

    if kv_cache_dtype in ["fp8", "fp8_e4m3", "fp8_e5m2"]:
        cloned_key_cache = _convert_from_fp8(key_cache_compact, k_scale.item())
        cloned_value_cache = _convert_from_fp8(value_cache_compact,
                                               v_scale.item())
    else:
        cloned_key_cache = key_cache_compact.clone()
        cloned_value_cache = value_cache_compact.clone()

    # Call the reshape_and_cache kernel.
    opcheck(
        torch.ops._C_cache_ops.reshape_and_cache,
        (
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        ),
        cond=(head_size == HEAD_SIZES[0]),
    )
    reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
    key_cache_compact = permute_and_compact(key_cache)
    value_cache_compact = permute_and_compact(value_cache)

    reshaped_key = key.reshape(num_tokens, *key_cache[0, :, :, 0, :].shape)
    block_indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indices_lst = block_indices.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets_lst = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indices_lst[i]
        block_offset = block_offsets_lst[i]
        cloned_key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        cloned_value_cache[block_idx, :, :, block_offset] = value[i]

    if kv_cache_dtype in ["fp8", "fp8_e4m3", "fp8_e5m2"]:
        result_key_cache = _convert_from_fp8(key_cache_compact, k_scale.item())
        result_value_cache = _convert_from_fp8(value_cache_compact,
                                               v_scale.item())
        torch.testing.assert_close(result_key_cache,
                                   cloned_key_cache,
                                   atol=0.1,
                                   rtol=0.1)
        torch.testing.assert_close(result_value_cache,
                                   cloned_value_cache,
                                   atol=0.1,
                                   rtol=0.1)
    else:
        torch.testing.assert_close(key_cache_compact, cloned_key_cache)
        torch.testing.assert_close(value_cache_compact, cloned_value_cache)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@torch.inference_mode()
def test_reshape_and_cache_flash(
    num_tokens: int,
    num_heads: int,
    num_layers: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
) -> None:
    # Note: torch.set_default_device("xpu:1") not works.
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping_lst = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst,
                                dtype=torch.long,
                                device=device)
    qkv = torch.randn(num_tokens,
                      3,
                      num_heads,
                      head_size,
                      dtype=dtype,
                      device=device)
    _, key, value = qkv.unbind(dim=1)

    # Create the KV caches.
    key_caches, value_caches = create_kv_caches_with_random_flash(
        num_blocks,
        block_size,
        1,  # num_layers
        num_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        seed=seed,
        device=device,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]

    k_scale = (key.amax() / 64.0).to(torch.float32)
    v_scale = (value.amax() / 64.0).to(torch.float32)

    def permute_and_compact(x):
        return x.contiguous()

    key_cache_compact = permute_and_compact(key_cache)
    value_cache_compact = permute_and_compact(value_cache)

    # Clone the KV caches.
    if kv_cache_dtype in ["fp8", "fp8_e4m3", "fp8_e5m2"]:
        cloned_key_cache = _convert_from_fp8(key_cache_compact, k_scale.item())
        cloned_value_cache = _convert_from_fp8(value_cache_compact,
                                               v_scale.item())
    else:
        cloned_key_cache = key_cache_compact.clone()
        cloned_value_cache = value_cache_compact.clone()

    # Call the reshape_and_cache kernel.
    opcheck(
        torch.ops._C_cache_ops.reshape_and_cache_flash,
        (
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        ),
    )
    reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
    key_cache_compact = permute_and_compact(key_cache)
    value_cache_compact = permute_and_compact(value_cache)

    # Run the reference implementation.
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies_lst = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets_lst = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indicies_lst[i]
        block_offset = block_offsets_lst[i]
        cloned_key_cache[block_idx, block_offset, :, :] = key[i]
        cloned_value_cache[block_idx, block_offset, :, :] = value[i]
    if kv_cache_dtype in ["fp8", "fp8_e4m3", "fp8_e5m2"]:
        result_key_cache = _convert_from_fp8(key_cache_compact, k_scale.item())
        result_value_cache = _convert_from_fp8(value_cache_compact,
                                               v_scale.item())
        torch.testing.assert_close(result_key_cache,
                                   cloned_key_cache,
                                   atol=0.1,
                                   rtol=0.1)
        torch.testing.assert_close(result_value_cache,
                                   cloned_value_cache,
                                   atol=0.1,
                                   rtol=0.1)
    else:
        torch.testing.assert_close(key_cache_compact, cloned_key_cache)
        torch.testing.assert_close(value_cache_compact, cloned_value_cache)


def _create_mla_cache(
    num_blocks: int,
    block_size: int,
    entry_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
) -> torch.Tensor:
    cache_dtype = torch.uint8 if kv_cache_dtype == "fp8" else dtype
    return torch.zeros(num_blocks,
                       block_size,
                       entry_size,
                       dtype=cache_dtype,
                       device=device)


def _fill_mla_cache(cache: torch.Tensor, kv_cache_dtype: str):
    rand_dtype = torch.float16 if kv_cache_dtype == "fp8" else cache.dtype

    vals = torch.randn(*cache.shape, device=cache.device, dtype=rand_dtype)
    if kv_cache_dtype == "fp8":
        temp = torch.zeros_like(cache)
        ops.convert_fp8(temp, vals, 1.0, kv_dtype=kv_cache_dtype)
        vals = temp
    cache.copy_(vals)


@pytest.mark.parametrize("kv_lora_rank", KV_LORA_RANKS)
@pytest.mark.parametrize("qk_rope_head_dim", QK_ROPE_HEAD_DIMS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS_MLA)
@pytest.mark.parametrize("block_size", BLOCK_SIZES_MLA)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS_MLA)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@torch.inference_mode()
def test_concat_and_cache_mla(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_tokens: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
) -> None:
    seed_everything(seed)
    torch.set_default_device(device)

    total_slots = num_blocks * block_size
    slot_mapping_lst = random.sample(range(total_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst,
                                dtype=torch.long,
                                device=device)

    kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype, device=device)
    k_pe = torch.randn(num_tokens,
                       qk_rope_head_dim,
                       dtype=dtype,
                       device=device)
    entry_size = kv_lora_rank + qk_rope_head_dim

    scale = torch.tensor(0.1, dtype=torch.float32, device=device)
    kv_cache = _create_mla_cache(num_blocks, block_size, entry_size, dtype,
                                 kv_cache_dtype, device)
    ref_temp = torch.zeros(*kv_cache.shape, dtype=dtype, device=device)

    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        ref_temp[block_idx, block_offset, :kv_lora_rank] = kv_c[i]
        ref_temp[block_idx, block_offset, kv_lora_rank:] = k_pe[i]

    if kv_cache_dtype == "fp8":
        ref_kv_cache = torch.empty_like(ref_temp, dtype=kv_cache.dtype)
        ops.convert_fp8(ref_kv_cache,
                        ref_temp,
                        scale.item(),
                        kv_dtype=kv_cache_dtype)
    else:
        ref_kv_cache = ref_temp

    opcheck(
        torch.ops._C_cache_ops.concat_and_cache_mla,
        (kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale),
        # test_utils=DEFAULT_OPCHECK_TEST_UTILS,
    )

    ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping,
                             kv_cache_dtype, scale)

    if kv_cache_dtype == "fp8":
        result_temp = torch.empty_like(kv_cache, dtype=torch.float16)
        ops.convert_fp8(result_temp,
                        kv_cache.contiguous(),
                        scale.item(),
                        kv_dtype=kv_cache_dtype)
        expected_temp = torch.empty_like(ref_kv_cache, dtype=torch.float16)
        ops.convert_fp8(expected_temp,
                        ref_kv_cache,
                        scale.item(),
                        kv_dtype=kv_cache_dtype)
        torch.testing.assert_close(result_temp,
                                   expected_temp,
                                   atol=0.001,
                                   rtol=0.1)
    else:
        torch.testing.assert_close(kv_cache, ref_kv_cache)


@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_blocks", [1024])
@pytest.mark.parametrize("max_seq_len", [512])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("kv_cache_dtype",
                         ["auto"])  # You can also test "fp8" if needed.
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_gather_cache_mla(kv_lora_rank, qk_rope_head_dim, block_size,
                          num_blocks, max_seq_len, batch_size, dtype,
                          kv_cache_dtype, device):
    entry_size = kv_lora_rank + qk_rope_head_dim
    src_cache = _create_mla_cache(num_blocks, block_size, entry_size, dtype,
                                  kv_cache_dtype, device)
    _fill_mla_cache(src_cache, kv_cache_dtype=kv_cache_dtype)

    seq_len_tensor = torch.randint(0,
                                   max_seq_len + 1, (batch_size, ),
                                   device=device)

    total_tokens = seq_len_tensor.sum()
    cu_seq_lens = torch.empty((batch_size + 1),
                              dtype=torch.int32,
                              device=device)
    cu_seq_lens[0] = 0
    cu_seq_lens[1:] = seq_len_tensor.cumsum(dim=0).to(dtype=torch.int32)
    print("seq_len_tensor", seq_len_tensor)

    tot_blocks_tensor = (seq_len_tensor + block_size - 1) // block_size
    block_table = torch.empty((batch_size, num_blocks),
                              dtype=torch.int32,
                              device=device)

    for b in range(batch_size):
        perm = torch.randperm(num_blocks, device=device)
        block_table[b, :] = perm

    dst = torch.zeros((total_tokens, entry_size),
                      dtype=src_cache.dtype,
                      device=device)

    expected_batches = []
    for b in range(batch_size):
        s = seq_len_tensor[b]
        if s == 0:
            continue
        tot = tot_blocks_tensor[b]
        blocks = block_table[b, :tot].tolist()

        gathered_rows = []
        for i in range(tot - 1):
            gathered_rows.append(src_cache[blocks[i]])
        remaining = s - (tot - 1) * block_size
        gathered_rows.append(src_cache[blocks[-1], :remaining, :])

        batch_expected = torch.cat(gathered_rows, dim=0)
        expected_batches.append(batch_expected)
    expected = torch.cat(expected_batches, dim=0)

    opcheck(
        torch.ops._C_cache_ops.gather_cache,
        (src_cache, dst, block_table, cu_seq_lens, batch_size, None),
        # test_utils=DEFAULT_OPCHECK_TEST_UTILS,
    )

    ops.gather_cache(src_cache, dst, block_table, cu_seq_lens, batch_size)
    torch.testing.assert_close(dst, expected)
