# SPDX-License-Identifier: Apache-2.0

import random

import pytest
import torch

from tests.ops.cache_op import reshape_and_cache, reshape_and_cache_flash
from tests.utils import (_convert_from_fp8, create_kv_caches_with_random,
                         create_kv_caches_with_random_flash, opcheck)

DTYPES = [torch.half, torch.bfloat16]
NUM_TOKENS = [42]  # Arbitrary values for testing
NUM_HEADS = [8]  # Arbitrary values for testing
NUM_LAYERS = [1]
HEAD_SIZES = [64, 80, 120, 256]
BLOCK_SIZES = [8, 16, 32]
NUM_BLOCKS = [1024]  # don't make it too large. e.g. [36000] will OOM
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
