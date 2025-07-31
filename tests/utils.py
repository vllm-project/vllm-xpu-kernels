# SPDX-License-Identifier: Apache-2.0
import unittest
from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import random
import torch
from torch._prims_common import TensorLikeType

ALL_OPCHECK_TEST_UTILS: tuple[str, ...] = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
)


# Copied/modified from torch._refs.__init__.py
def fp8_allclose(
    a: TensorLikeType,
    b: TensorLikeType,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    """
    Reference implementation of torch.allclose
    """
    torch._refs._check_close_args(name="torch.allclose",
                                  a=a,
                                  b=b,
                                  rtol=rtol,
                                  atol=atol)

    return bool(
        torch.all(
            torch.isclose(a.double(),
                          b.double(),
                          rtol=rtol,
                          atol=atol,
                          equal_nan=equal_nan)).item())


# A special version of op check that has a restricted default set of test_utils
# and a patched version of allclose that supports fp8 types.
def opcheck(op: Union[torch._ops.OpOverload, torch._ops.OpOverloadPacket,
                      torch._library.custom_ops.CustomOpDef],
            args: tuple[Any, ...],
            kwargs: Optional[dict[str, Any]] = None,
            *,
            test_utils: Union[str, Sequence[str]] = ALL_OPCHECK_TEST_UTILS,
            raise_exception: bool = True,
            cond: bool = True) -> dict[str, str]:
    with unittest.mock.patch('torch.allclose', new=fp8_allclose):
        return torch.library.opcheck(
            op,
            args,
            kwargs,
            test_utils=test_utils,
            raise_exception=raise_exception) if cond else {}


STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
    "int8": torch.int8,
    "fp8_inc": torch.float8_e4m3fn,
}

def _generate_random_fp8(
    cache_dtype: Optional[Union[str, torch.dtype]],
    tensor: torch.Tensor,
    low: float,
    high: float,
) -> None:
    if cache_dtype == "fp8" or cache_dtype == "fp8_e4m3":
        dst_dtype = torch.float8_e4m3fn
    elif cache_dtype == "fp8_e5m2":
        dst_dtype = torch.float8_e5m2
    else:
        assert False
    tensor_tmp = torch.empty_like(tensor, dtype=torch.float16)
    tensor_tmp.uniform_(low, high)
    tensor = tensor_tmp.to(dst_dtype)
    del tensor_tmp

def get_kv_cache_torch_dtype(
        cache_dtype: Optional[Union[str, torch.dtype]],
        model_dtype: Optional[Union[str, torch.dtype]] = None) -> torch.dtype:
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype,
                          str) and model_dtype in STR_DTYPE_TO_TORCH_DTYPE:
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f"Invalid model dtype: {model_dtype}")
        elif cache_dtype in STR_DTYPE_TO_TORCH_DTYPE:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        else:
            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    return torch_dtype

def create_kv_caches_with_random_flash(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: Optional[int] = None,
    device: Optional[str] = "xpu",
    cache_layout: Optional[str] = "NHD",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if not seed is None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
    generic_kv_cache_shape = (num_blocks, 2, block_size, num_heads, head_size)
    assert cache_layout in ("NHD", "HND")
    stride_order = (0, 1, 2, 3, 4) if cache_layout == "NHD" else (0, 1, 3, 2,
                                                                  4)

    kv_cache_allocation_shape = tuple(generic_kv_cache_shape[i]
                                      for i in stride_order)
    scale = head_size**-0.5

    key_caches: list[torch.Tensor] = []
    value_caches: list[torch.Tensor] = []

    for _ in range(num_layers):
        key_value_cache = torch.empty(size=kv_cache_allocation_shape,
                                      dtype=torch_dtype,
                                      device=device).permute(*stride_order)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_value_cache.uniform_(-scale, scale)
        elif cache_dtype in ["fp8", "fp8_e4m3", "fp8_e5m2"]:
            _generate_random_fp8(cache_dtype, key_value_cache, -scale, scale)
        else:
            raise ValueError(
                f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_value_cache[:, 0])
        value_caches.append(key_value_cache[:, 1])
    return key_caches, value_caches
