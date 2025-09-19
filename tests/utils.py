# SPDX-License-Identifier: Apache-2.0
import random
import unittest
from collections.abc import Sequence
from typing import Any, Optional, Union

import argparse
import numpy as np
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
def opcheck(
    op: Union[
        torch._ops.OpOverload,
        torch._ops.OpOverloadPacket,
        torch._library.custom_ops.CustomOpDef,
    ],
    args: tuple[Any, ...],
    kwargs: Optional[dict[str, Any]] = None,
    *,
    test_utils: Union[str, Sequence[str]] = ALL_OPCHECK_TEST_UTILS,
    raise_exception: bool = True,
    cond: bool = True,
) -> dict[str, str]:
    with unittest.mock.patch("torch.allclose", new=fp8_allclose):
        return (torch.library.opcheck(op,
                                      args,
                                      kwargs,
                                      test_utils=test_utils,
                                      raise_exception=raise_exception)
                if cond else {})


STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.float8_e4m3fn,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "int8": torch.int8,
    "fp8_inc": torch.float8_e4m3fn,
}


def seed_everything(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _convert_from_fp8(
    tensor: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    return (tensor.to(torch.float32) * scale)


def _generate_random_fp8(tensor: torch.Tensor, low: float, high: float,
                         torch_dtype: torch.dtype) -> torch.Tensor:
    assert torch_dtype in [torch.float8_e4m3fn, torch.float8_e5m2], \
        f"Unsupported torch dtype for fp8: {torch_dtype}"
    tensor_tmp = torch.empty_like(tensor, dtype=torch.float16)
    tensor_tmp.uniform_(low, high)
    return tensor_tmp.to(torch_dtype)


def get_kv_cache_torch_dtype(
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.dtype:
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


def create_kv_caches_with_random(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: Optional[int] = None,
    device: Optional[str] = "xpu",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=torch_dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches: list[torch.Tensor] = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape,
                                dtype=torch_dtype,
                                device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(-scale, scale)
        elif cache_dtype in ["fp8", "fp8_e4m3", "fp8_e5m2"]:
            _generate_random_fp8(key_cache, -scale, scale, torch_dtype)
        else:
            raise ValueError(
                f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches: list[torch.Tensor] = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape,
                                  dtype=torch_dtype,
                                  device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(-scale, scale)
        elif cache_dtype in ["fp8", "fp8_e4m3", "fp8_e5m2"]:
            _generate_random_fp8(value_cache, -scale, scale, torch_dtype)
        else:
            raise ValueError(
                f"Does not support value cache of type {cache_dtype}")
        value_caches.append(value_cache)
    return key_caches, value_caches


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
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
    generic_kv_cache_shape = (num_blocks, 2, block_size, num_heads, head_size)
    stride_order = (0, 1, 2, 3, 4)

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
            key_value_cache = _generate_random_fp8(key_value_cache, -scale,
                                                   scale, torch_dtype)
        else:
            raise ValueError(
                f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_value_cache[:, 0])
        value_caches.append(key_value_cache[:, 1])
    return key_caches, value_caches


def get_model_config(model_name: str, tp_size: int = 1):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_arch = config.architectures[0] if config.architectures else "Unknown"

    if model_arch == "DbrxForCausalLM":
        original_num_groups = config.ffn_config.moe_num_experts
        original_intermediate_size = config.ffn_config.ffn_hidden_size
    elif model_arch == "JambaForCausalLM":
        original_num_groups = config.num_experts
        original_intermediate_size = config.intermediate_size
    elif model_arch in ["Qwen2MoeForCausalLM", "Qwen3MoeForCausalLM"]:
        original_num_groups = config.num_experts
        original_intermediate_size = config.moe_intermediate_size
    elif model_arch in ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"]:
        original_num_groups = config.n_routed_experts
        original_intermediate_size = config.moe_intermediate_size
    elif model_arch == "LlamaForCausalLM":
        original_num_groups = 1
        original_intermediate_size = config.intermediate_size
    else:
        original_num_groups = getattr(config, 'num_local_experts', 1)
        original_intermediate_size = getattr(config, 'intermediate_size',
                                             config.hidden_size * 4)

    effective_hidden_size = config.hidden_size
    effective_intermediate_size = original_intermediate_size
    effective_num_groups = original_num_groups

    if tp_size > 1:
        effective_hidden_size = config.hidden_size
        effective_intermediate_size = original_intermediate_size // tp_size
        if original_num_groups > 1:
            effective_num_groups = original_num_groups

    moe_config = {}
    if original_num_groups > 1:
        moe_config = {
            "moe_top_k":
            getattr(config, 'num_experts_per_tok', getattr(config, 'top_k',
                                                           2)),
            "moe_min_capacity":
            getattr(config, 'moe_min_capacity', 4),
            "moe_capacity_factor":
            getattr(config, 'moe_capacity_factor', 1.0),
            "moe_aux_loss_coef":
            getattr(config, 'moe_aux_loss_coef', 0.01),
        }

    shape_configs = {
        "num_groups":
        effective_num_groups,
        "hidden_size":
        effective_hidden_size,
        "intermediate_size":
        effective_intermediate_size,
        "dtype":
        config.torch_dtype,
        "model_arch":
        model_arch,
        "vocab_size":
        getattr(config, 'vocab_size', 32000),
        "num_layers":
        getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 32)),
        "num_attention_heads":
        getattr(config, 'num_attention_heads', getattr(config, 'n_head', 32)),
        "num_key_value_heads":
        getattr(config, 'num_key_value_heads',
                getattr(config, 'num_attention_heads', 32)),
        "head_dim":
        getattr(
            config, 'head_dim',
            config.hidden_size // getattr(config, 'num_attention_heads', 32)),
        "hidden_act":
        getattr(config, 'hidden_act', 'silu'),
        "max_position_embeddings":
        getattr(config, 'max_position_embeddings', 2048),
        "rope_theta":
        getattr(config, 'rope_theta', 10000.0),
        "rope_scaling":
        getattr(config, 'rope_scaling', None),
        "rms_norm_eps":
        getattr(config, 'rms_norm_eps', 1e-6),
        "layer_norm_eps":
        getattr(config, 'layer_norm_eps', 1e-5),
        "attention_dropout":
        getattr(config, 'attention_dropout', 0.0),
        "hidden_dropout":
        getattr(config, 'hidden_dropout', 0.0),
        "moe_config":
        moe_config,
        "is_moe":
        original_num_groups > 1,
        "original_config": {
            "hidden_size": config.hidden_size,
            "intermediate_size": original_intermediate_size,
            "num_groups": original_num_groups,
        },
        "tp_size":
        tp_size,
        "model_type":
        getattr(config, 'model_type', 'unknown'),
        "torch_dtype":
        str(config.torch_dtype)
        if hasattr(config, 'torch_dtype') else 'float32',
    }

    print(f"Full model config with TP={tp_size}:")
    for key, value in shape_configs.items():
        if key != "original_config":
            print(f"  {key}: {value}")

    return shape_configs


def check_ipex_availability():
    """
    Check if Intel Extension for PyTorch (IPEX) is available.
    
    Returns:
        bool: True if IPEX is available, False otherwise
    """
    import importlib.util
    if importlib.util.find_spec("intel_extension_for_pytorch") is not None:
        return True
    else:
        print("Warning: IPEX not available, skipping IPEX benchmarks")
        return False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=4096,
        help="Hidden size (2nd dimension) of the sequence",
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=None,
        help="Intermediate size for FFN layers",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=None,
        help="Number of expert groups for MoE models",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=torch.bfloat16,
        help="Data type from model config",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name to load configuration from",
    )
    parser.add_argument(
        "--head-size",
        type=int,
        choices=[64, 80, 96, 112, 120, 128, 192, 256],
        default=128,
    )
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
        default="auto",
    )
    parser.add_argument("--block-size",
                        type=int,
                        choices=[16, 32, 64],
                        default=64)
    parser.add_argument("--head-num-range",
                        type=int,
                        nargs='+',
                        default=[12, 32, 40, 48, 64, 96, 128],
                        help=("Range of attention head numbers to test/use. "
                              "Default: 12 32 40 48 64 96 128"))
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallelism size",
    )
    parser.add_argument("--use-residual",
                        action="store_true",
                        help="Whether to use residual connection")
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/rmsnorm/",
        help="Path to save rmsnorm benchmark results",
    )

    args = parser.parse_args()

    if args.model_name:
        model_config = get_model_config(args.model_name, args.tp_size)

        if args.hidden_size == 4096:
            args.hidden_size = model_config["hidden_size"]

        if args.intermediate_size is None:
            args.intermediate_size = model_config["intermediate_size"]

        if args.num_groups is None:
            args.num_groups = model_config["num_groups"]

        if args.dtype is None:
            args.dtype = model_config["dtype"]

        if args.head_size is None:
            args.head_size = model_config["head_dim"]

        if args.head_num_range == [12, 32, 40, 48, 64, 96, 128]:
            model_heads = model_config.get("num_attention_heads", 32)
            if model_heads not in args.head_num_range:
                args.head_num_range.append(model_heads)
                args.head_num_range.sort()
                print(
                    f"Added model's head number {model_heads} to head_num_range"
                )

        print(f"Using model configuration from: {args.model_name}")
        print(f"Updated hidden_size: {args.hidden_size}")
        print(f"Updated intermediate_size: {args.intermediate_size}")
        print(f"Updated num_groups: {args.num_groups}")
        print(f"Updated head_num_range: {args.head_num_range}")
        print(f"Updated dtype: {args.dtype}")

    return args
