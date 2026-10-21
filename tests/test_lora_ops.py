# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from threading import Lock

import pytest
import torch

from tests.lora import torch_ops, xpu_ops
from tests.lora.utils import (PunicaTensors, assert_close,
                              generate_data_for_nslices)
from tests.utils import format_tc, seed_everything

MINI_PYTEST_PARAMS = {
    "test_kernels": {
        "batches": [1],
        "num_loras": [1],
        "rank": [1],
        "hidden_size": [128],
        "nslices": [1],
        "dtype": [torch.float16],
        "device": ["xpu:0"],
        "seed": [0],
        "op_type": ["shrink", "expand"],
    },
    "test_kernels_hidden_size": {
        "batches": [4],
        "num_loras": [4],
        "rank": [32],
        "hidden_size": [128],
        "nslices": [1],
        "dtype": [torch.float16],
        "device": ["xpu:0"],
        "seed": [0],
        "op_type": ["shrink", "expand"],
    },
    "test_kernels_mixed_precision": {
        "batches": [1],
        "num_loras": [1],
        "rank": [1],
        "hidden_size": [128],
        "nslices": [1],
        "weight_dtype": [torch.float16],
        "device": ["xpu:0"],
        "seed": [0],
        "op_type": ["shrink", "expand"],
    },
}


@pytest.fixture(autouse=True)
def reset_device(reset_default_device):
    pass


# Utility shrink and expand operations used as reference implementations.
def sgmv_shrink_for_nslices(
    nslices: int,
    inputs_tensor: torch.Tensor,
    lora_weights_lst: list[torch.Tensor],
    out_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    prompt_lora_mapping: torch.Tensor,
    batches: int,
    max_seq_length: int,
    num_tokens: int,
    scaling: float,
):
    """
    Wrapper around torch_ops.sgmv_shrink that handles any nslices.
    """
    for index in range(nslices):
        torch_ops.sgmv_shrink(
            inputs_tensor,
            lora_weights_lst[index],
            out_tensor[index],
            b_seq_start_loc,
            seq_len_tensor,
            prompt_lora_mapping,
            batches,
            max_seq_length,
            num_tokens,
            scaling,
        )


def sgmv_expand_for_nslices(
    nslices: int,
    hidden_size: int,
    inputs_tensor: torch.Tensor,
    lora_weights_lst: list[torch.Tensor],
    out_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    prompt_lora_mapping: torch.Tensor,
    batches: int,
    max_seq_length: int,
    num_tokens: int,
    add_inputs: bool,
) -> None:
    """
    Wrapper around torch_ops.sgmv_expand that handles any nslices.
    """
    if nslices == 1:
        # Verify the torch's sgmv_expand op
        torch_ops.sgmv_expand(
            inputs_tensor[0],
            lora_weights_lst[0],
            out_tensor,
            b_seq_start_loc,
            seq_len_tensor,
            prompt_lora_mapping,
            batches,
            max_seq_length,
            num_tokens,
            add_inputs=add_inputs,
        )
    else:
        slice_offset = 0
        for index in range(nslices):
            lora_weights = lora_weights_lst[index]
            torch_ops.sgmv_expand_slice(
                inputs_tensor[index],
                lora_weights,
                out_tensor,
                b_seq_start_loc,
                seq_len_tensor,
                prompt_lora_mapping,
                batches,
                max_seq_length,
                num_tokens,
                slice_offset,
                hidden_size,
                add_inputs=add_inputs,
            )
            slice_offset += hidden_size


_dict_lock = Lock()


def check_lora_shrink_kernel(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    seq_length: int,
    scaling: float,
    input_dtype: torch.dtype = None,
):
    """
    Compare outputs of torch_ops.sgmv_shrink and xpu_ops.lora_shrink
    kernels.
    """
    data: PunicaTensors = generate_data_for_nslices(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        nslices,
        dtype,
        "shrink",
        device,
        input_dtype=input_dtype,
    )
    max_seq_length, token_nums = data.meta()

    # Setup metadata information for SGMV and reference kernels
    sgmv_meta_args = (
        data.b_seq_start_loc,
        data.seq_len_tensor,
        data.prompt_lora_mapping,
        batches,
        max_seq_length,
        token_nums,
    )

    ref_out_tensor = data.ref_out_tensor
    out_tensor = data.our_out_tensor.clone()

    # Preventing cache error pointer.
    with _dict_lock:
        for index in range(nslices):
            xpu_ops.bgmv_shrink(
                data.inputs_tensor,
                data.lora_weights[index],
                out_tensor[index],
                data.token_lora_mapping,
                scaling,
            )
    # Reference
    sgmv_shrink_for_nslices(
        nslices,
        data.inputs_tensor,
        data.lora_weights,
        ref_out_tensor,
        *sgmv_meta_args,
        scaling,
    )

    assert_close(out_tensor, ref_out_tensor)


def check_lora_expand_kernel(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    seq_length: int,
    add_inputs: bool,
    input_dtype: torch.dtype = None,
):
    """
    Compare outputs of torch_ops.sgmv_expand and xpu_ops.lora_expand
    kernels.
    """
    data: PunicaTensors = generate_data_for_nslices(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        nslices,
        dtype,
        "expand",
        device,
        input_dtype=input_dtype,
    )

    max_seq_length, token_nums = data.meta()

    # Setup metadata information for SGMV and reference kernels
    sgmv_meta_args = (
        data.b_seq_start_loc,
        data.seq_len_tensor,
        data.prompt_lora_mapping,
        batches,
        max_seq_length,
        token_nums,
    )

    # Setup metadata information for the LoRA kernel.

    # Setup output tensors
    ref_out_tensor = data.ref_out_tensor
    out_tensor = data.our_out_tensor.clone()

    with _dict_lock:
        # lora_expand kernel
        slice_offset = 0
        for index in range(nslices):
            xpu_ops.bgmv_expand_slice(
                data.inputs_tensor[index],
                data.lora_weights[index],
                out_tensor,
                data.token_lora_mapping,
                slice_offset,
                slice_size=hidden_size,
                add_inputs=add_inputs,
            )
            slice_offset += hidden_size

    # Reference
    sgmv_expand_for_nslices(
        nslices,
        hidden_size,
        data.inputs_tensor,
        data.lora_weights,
        ref_out_tensor,
        *sgmv_meta_args,
        add_inputs=add_inputs,
    )

    assert_close(out_tensor, ref_out_tensor)


# Tests
# We test the punica kernels along 2 verticals mainly.
# 1. Variations in hidden_dim size
# 2. Variations in all other parameters like (batch_size, max_rank, num_loras
#  etc.)

# We have collected the hidden_sizes included in the LoRA models
# currently supported by vLLM. It tests whether the corresponding Triton
# kernel can run normally when tensor parallelism is set to
# [1, 2, 4, 8, 16, 32, 64].
HIDDEN_SIZES = [
    128,
    256,
    512,
    896,
    1024,
    1152,
    1216,
    1280,
    1536,
    1664,
    2048,
    2240,
    2304,
    2368,
    2432,
    2560,
    2752,
    3072,
    3328,
    3456,
    3584,
    3712,
    4096,
    4480,
    4608,
    4736,
    4864,
    5120,
    5504,
    5632,
    5888,
    6144,
    6400,
    6848,
    6912,
    7168,
    7424,
    8192,
    8960,
    9216,
    9472,
    10240,
    11008,
    11264,
    13824,
    14336,
    14784,
    14848,
    15360,
    18944,
    22016,
    22528,
    24576,
    27392,
    27648,
    29568,
    29696,
    32000,
    32256,
    32512,
    32768,
    33024,
    36864,
    43264,
    49152,
    49408,
    60544,
    60672,
    64000,
    64256,
    102400,
    102656,
    128000,
    128256,
]
# The size of TP
divisibility = [1, 2, 8, 16, 64]

all_hidden_size = []
for div in divisibility:
    for hidden_size in HIDDEN_SIZES:
        all_hidden_size.append(hidden_size // div)

HIDDEN_SIZES = list(set(all_hidden_size))

# Test params that focuses on hidden_size variation.
hs_test_params = {
    "hidden_sizes": HIDDEN_SIZES,
    "batches": [4],
    "num_loras": [4],
    "max_ranks": [32],
}

# General tests params that tests for variations in all dimensions
# except hidden_size.
test_params = {
    "hidden_sizes": [2049],
    "batches": [1, 4, 16, 32],
    "num_loras": [1, 8, 32, 128],
    "max_ranks": [1, 4, 8, 16, 32, 64, 128, 256],
}

DTYPES = [torch.float16, torch.bfloat16]
DEVICES = [f"xpu:{0}"]
SEED = [0]


@pytest.mark.parametrize("batches", test_params["batches"])
@pytest.mark.parametrize("num_loras", test_params["num_loras"])
@pytest.mark.parametrize("rank", test_params["max_ranks"])
@pytest.mark.parametrize("hidden_size", test_params["hidden_sizes"])
@pytest.mark.parametrize("nslices", [1, 2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
def test_kernels(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    op_type: str,
):
    """
    Tests LoRA kernels.
    """
    torch.set_default_device(device)
    torch.xpu.set_device(device)
    seed_everything(seed)

    if op_type == "shrink":
        check_lora_shrink_kernel(
            batches=batches,
            num_loras=num_loras,
            rank=rank,
            hidden_size=hidden_size,
            nslices=nslices,
            dtype=dtype,
            device=device,
            seq_length=128,
            scaling=0.5,
        )
    else:
        check_lora_expand_kernel(
            batches=batches,
            num_loras=num_loras,
            rank=rank,
            hidden_size=hidden_size,
            nslices=nslices,
            dtype=dtype,
            device=device,
            seq_length=128,
            add_inputs=True,
        )


@pytest.mark.parametrize("batches", hs_test_params["batches"])
@pytest.mark.parametrize("num_loras", hs_test_params["num_loras"])
@pytest.mark.parametrize("rank", hs_test_params["max_ranks"])
@pytest.mark.parametrize("hidden_size", hs_test_params["hidden_sizes"])
@pytest.mark.parametrize("nslices", [1, 2, 3])
@pytest.mark.parametrize("dtype", DTYPES, ids=format_tc)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
def test_kernels_hidden_size(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    op_type: str,
):
    """
    Tests SGMV and LoRA kernels.
    """
    torch.set_default_device(device)
    torch.xpu.set_device(device)
    seed_everything(seed)

    if op_type == "shrink":
        check_lora_shrink_kernel(
            batches=batches,
            num_loras=num_loras,
            rank=rank,
            hidden_size=hidden_size,
            nslices=nslices,
            dtype=dtype,
            device=device,
            seq_length=128,
            scaling=0.5,
        )
    else:
        check_lora_expand_kernel(
            batches=batches,
            num_loras=num_loras,
            rank=rank,
            hidden_size=hidden_size,
            nslices=nslices,
            dtype=dtype,
            device=device,
            seq_length=128,
            add_inputs=True,
        )


@pytest.mark.parametrize("batches", [1, 4, 16])
@pytest.mark.parametrize("num_loras", [1, 8])
@pytest.mark.parametrize("rank", [1, 16, 64])
@pytest.mark.parametrize("hidden_size", [128, 2049])
@pytest.mark.parametrize("nslices", [1, 2, 3])
@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
def test_kernels_mixed_precision(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    weight_dtype: torch.dtype,
    device: str,
    seed: int,
    op_type: str,
):
    """
    Tests LoRA kernels with mixed precision:
    input=float32, weight=float16/bfloat16.
    """
    torch.set_default_device(device)
    torch.xpu.set_device(device)
    seed_everything(seed)

    if op_type == "shrink":
        check_lora_shrink_kernel(
            batches=batches,
            num_loras=num_loras,
            rank=rank,
            hidden_size=hidden_size,
            nslices=nslices,
            dtype=weight_dtype,
            device=device,
            seq_length=128,
            scaling=0.5,
            input_dtype=torch.float32,
        )
    else:
        check_lora_expand_kernel(
            batches=batches,
            num_loras=num_loras,
            rank=rank,
            hidden_size=hidden_size,
            nslices=nslices,
            dtype=weight_dtype,
            device=device,
            seq_length=128,
            add_inputs=True,
            input_dtype=torch.float32,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "active_rank,stored_rank", [(8, 32), (7, 19), (32, 32)]
)
@pytest.mark.parametrize("nslices", [1, 3, 8])
@pytest.mark.parametrize("add_inputs", [False, True])
def test_active_rank_storage(
    dtype, active_rank, stored_rank, nslices, add_inputs
):
    """Active rank must not change the strides between stored rows/adapters."""
    torch.manual_seed(42)
    device = "xpu:0"
    batch, hidden, num_loras = 5, 128, 2
    indices = torch.tensor([1, 0, -1, 2, 0], device=device, dtype=torch.int32)
    x = torch.randn(batch, hidden, device=device, dtype=dtype) * 0.1
    a = [
        torch.randn(
            num_loras, 1, stored_rank + 3 * s, hidden,
            device=device, dtype=dtype
        )
        * 0.1
        for s in range(nslices)
    ]
    b = [
        torch.randn(
            num_loras, 1, 17 + s, stored_rank + 3 * s,
            device=device, dtype=dtype
        )
        * 0.1
        for s in range(nslices)
    ]
    intermediate = torch.full(
        (nslices, batch, active_rank),
        float("nan"),
        device=device,
        dtype=torch.float32,
    )
    torch.ops._xpu_C.lora_shrink(x, a, intermediate, indices, 0.5)
    expected = torch.zeros_like(intermediate)
    for s in range(nslices):
        for row, adapter in enumerate([1, 0, -1, 2, 0]):
            if 0 <= adapter < num_loras:
                expected[s, row] = (
                    x[row].float() @ a[s][adapter, 0, :active_rank].float().T
                ) * 0.5
    torch.testing.assert_close(intermediate, expected, atol=1e-5, rtol=1e-4)
    # Nonzero offset and trailing columns must be left untouched.
    output = torch.randn(
        batch, 5 + sum(w.size(-2) for w in b), device=device, dtype=dtype
    )
    reference = output.clone()
    offset = 2
    for s, w in enumerate(b):
        width = w.size(-2)
        for row, adapter in enumerate([1, 0, -1, 2, 0]):
            if 0 <= adapter < num_loras:
                delta = (
                    expected[s, row] @ w[adapter, 0, :, :active_rank].float().T
                ).to(dtype)
                if add_inputs:
                    reference[row, offset : offset + width] += delta
                else:
                    reference[row, offset : offset + width] = delta
        offset += width
    torch.ops._xpu_C.lora_expand(
        intermediate, b, output, indices, 2, add_inputs
    )
    torch.testing.assert_close(output, reference, atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("op", ["shrink", "expand"])
@pytest.mark.parametrize("active_rank", [0, 9])
def test_active_rank_out_of_bounds(op, active_rank):
    weights = [torch.zeros(2, 1, 8, 8, device="xpu", dtype=torch.float16)]
    indices = torch.zeros(1, device="xpu", dtype=torch.int32)
    intermediate = torch.zeros(1, 1, active_rank, device="xpu")
    x = torch.zeros(1, 8, device="xpu", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="rank"):
        if op == "shrink":
            torch.ops._xpu_C.lora_shrink(x, weights, intermediate, indices, 1.0)
        else:
            torch.ops._xpu_C.lora_expand(
                intermediate, weights, x, indices, 0, True
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("add_inputs", [False, True])
@pytest.mark.parametrize("nslices", [1, 3, 8])
def test_active_rank_linear(dtype, add_inputs, nslices):
    torch.manual_seed(123)
    x = torch.randn(4, 128, device="xpu", dtype=dtype) * 0.1
    a = [
        torch.randn(2, 1, 32 + s * 3, 128, device="xpu", dtype=dtype) * 0.1
        for s in range(nslices)
    ]
    b = [
        torch.randn(2, 1, 17 + s, 32 + s * 3, device="xpu", dtype=dtype) * 0.1
        for s in range(nslices)
    ]
    indices = torch.tensor([1, -1, 2, 0], device="xpu", dtype=torch.int32)
    output = torch.randn(
        4, 3 + sum(w.size(-2) for w in b), device="xpu", dtype=dtype
    )
    expected = output.clone()
    offset = 0
    for aw, bw in zip(a, b):
        width = bw.shape[-2]
        for row, adapter in enumerate([1, -1, 2, 0]):
            if 0 <= adapter < 2:
                delta = (
                    (x[row].float() @ aw[adapter, 0, :8].float().T) * 0.5
                ).to(dtype).float() @ bw[adapter, 0, :, :8].float().T
                if add_inputs:
                    expected[row, offset : offset + width] += delta.to(dtype)
                else:
                    expected[row, offset : offset + width] = delta.to(dtype)
        offset += width
    torch.ops._xpu_C.lora_linear(x, a, b, output, indices, 0.5, 8, add_inputs)
    torch.testing.assert_close(output, expected, atol=2e-3, rtol=2e-2)
