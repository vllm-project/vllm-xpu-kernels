# SPDX-License-Identifier: Apache-2.0

import random

import pytest
import torch
from typing import Optional, Union

from tests.ops.cache_op import reshape_and_cache_flash
from tests.utils import (
    opcheck,
    STR_DTYPE_TO_TORCH_DTYPE,
    create_kv_caches_with_random_flash,
)

DTYPES = [torch.half, torch.bfloat16]
NUM_TOKENS = [42]  # Arbitrary values for testing
NUM_HEADS = [8]  # Arbitrary values for testing
NUM_LAYERS = [1]
HEAD_SIZES = [64]
# HEAD_SIZES = [64, 80, 120, 256]
BLOCK_SIZES = [8]
# BLOCK_SIZES = [8, 16, 32]
CACHE_LAYOUTS = ["NHD"]
# CACHE_LAYOUTS = ["NHD", "HND"]

# Parameters for MLA tests.
KV_LORA_RANKS = [512]
QK_ROPE_HEAD_DIMS = [64]
NUM_TOKENS_MLA = [42]
BLOCK_SIZES_MLA = [16]
NUM_BLOCKS_MLA = [8]

# Arbitrary values for testing
# don't make it too large. e.g. [1024, 36000] will OOM
NUM_BLOCKS = [1024]
# NUM_BLOCKS = [1024, 10000]

NUM_MAPPINGS = [256]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)]

# We assume fp8 is always enabled for testing.
KV_CACHE_DTYPE = ["auto"]
# KV_CACHE_DTYPE = ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]

# STR_DTYPE_TO_TORCH_DTYPE = {
#     "half": torch.half,
#     "bfloat16": torch.bfloat16,
#     "fp8": torch.float8_e4m3fn,
#     "fp8_e4m3": torch.float8_e4m3fn,
#     "fp8_e5m2": torch.float8_e5m2,
#     "int8": torch.int8,
# }


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@pytest.mark.parametrize("kv_cache_layout", CACHE_LAYOUTS)
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
    kv_cache_layout: str,
) -> None:
    torch.set_default_device(device)

    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping_lst = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)
    qkv = torch.randn(num_tokens, 3, num_heads, head_size, dtype=dtype, device=device)
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
        device=device,
        cache_layout=kv_cache_layout,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]

    k_scale = (key.amax() / 64.0).to(torch.float32)
    v_scale = (value.amax() / 64.0).to(torch.float32)

    def permute_and_compact(x):
        y = x if kv_cache_layout == "NHD" else x.permute(0, 2, 1, 3)
        return y.contiguous()

    key_cache_compact = permute_and_compact(key_cache)
    value_cache_compact = permute_and_compact(value_cache)

    # Clone the KV caches.
    if kv_cache_dtype == "fp8":
        cloned_key_cache = torch.empty_like(key_cache_compact, dtype=torch.float16)
        ops.convert_fp8(
            cloned_key_cache, key_cache_compact, k_scale.item(), kv_cache_dtype
        )
        cloned_value_cache = torch.empty_like(value_cache_compact, dtype=torch.float16)
        ops.convert_fp8(
            cloned_value_cache, value_cache_compact, v_scale.item(), kv_cache_dtype
        )
    else:
        cloned_key_cache = key_cache_compact.clone()
        cloned_value_cache = value_cache_compact.clone()
    # Call the reshape_and_cache kernel.
    opcheck(
        reshape_and_cache_flash,
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

    if kv_cache_dtype == "fp8":
        result_key_cache = torch.empty_like(key_cache_compact, dtype=torch.float16)
        ops.convert_fp8(
            result_key_cache, key_cache_compact, k_scale.item(), kv_dtype=kv_cache_dtype
        )
        result_value_cache = torch.empty_like(value_cache_compact, dtype=torch.float16)
        ops.convert_fp8(
            result_value_cache,
            value_cache_compact,
            v_scale.item(),
            kv_dtype=kv_cache_dtype,
        )

    # Run the reference implementation.
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies_lst = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets_lst = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indicies_lst[i]
        block_offset = block_offsets_lst[i]
        if kv_cache_layout == "NHD":
            cloned_key_cache[block_idx, block_offset, :, :] = key[i]
            cloned_value_cache[block_idx, block_offset, :, :] = value[i]
        else:
            cloned_key_cache[block_idx, :, block_offset, :] = key[i]
            cloned_value_cache[block_idx, :, block_offset, :] = value[i]

    if kv_cache_dtype == "fp8":
        torch.testing.assert_close(
            result_key_cache, cloned_key_cache, atol=0.001, rtol=0.1
        )
        torch.testing.assert_close(
            result_value_cache, cloned_value_cache, atol=0.001, rtol=0.1
        )
    else:
        torch.testing.assert_close(key_cache_compact, cloned_key_cache)
        torch.testing.assert_close(value_cache_compact, cloned_value_cache)
