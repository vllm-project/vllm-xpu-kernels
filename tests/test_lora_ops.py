# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.lora.lora_ops import bgmv_expand, bgmv_expand_slice, bgmv_shrink
from tests.lora.utils import (PunicaTensors, assert_close, generate_data,
                              generate_data_for_expand_nslices,
                              seed_everything)


def ref_torch_groupgemm(
    out_tensor,
    inputs,
    lora_weights,
    lora_indices_tensor,
    seq_len_tensor,
    batches,
    scaling,
    op_type,
) -> None:
    out_list = []
    current_offset = 0
    for lora_index, b_length in zip(range(batches), seq_len_tensor):
        input_weight = inputs[current_offset:b_length + current_offset, :]
        current_offset += b_length
        lora_weight = lora_weights[lora_indices_tensor[lora_index]]
        result = torch.nn.functional.linear(input_weight, lora_weight)
        result *= scaling
        out_list.append(result)
    cat_result = torch.cat(out_list, dim=0)
    if op_type == "expand":
        out_tensor += cat_result
    else:
        out_tensor.copy_(cat_result)


def torch_bgmv_expand_slice(inputs: torch.Tensor,
                            lora_b_weights: torch.Tensor,
                            output_tensor: torch.Tensor,
                            lora_indices_tensor: torch.Tensor,
                            slice_offset: int,
                            slice_size: int,
                            add_inputs: bool = True):
    selected_loras = lora_b_weights[lora_indices_tensor].to(
        dtype=output_tensor.dtype)
    inputs = inputs.to(dtype=output_tensor.dtype)
    if len(selected_loras.shape) == 4:
        selected_loras = selected_loras.squeeze(dim=1)
    outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)

    if add_inputs:
        output_tensor[:, slice_offset:slice_offset + slice_size] += outputs[:]
    else:
        output_tensor[:, slice_offset:slice_offset + slice_size] = outputs[:]


def check_bgmv_shrink(batches: int, num_loras: int, rank: int,
                      hidden_size: int, dtype: torch.dtype, device: str,
                      scaling: float):
    """
    Compare vllm.bgmv_shrink against a reference implementation.
    """
    seq_length = 1
    data: PunicaTensors = generate_data(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        dtype,
        "shrink",
        device,
    )

    bgmv_shrink(
        data.inputs_tensor,
        data.lora_weights,
        data.our_out_tensor,
        data.token_lora_mapping,
        scaling,
    )

    ref_torch_groupgemm(
        data.ref_out_tensor,
        data.inputs_tensor,
        data.lora_weights,
        data.token_lora_mapping,
        data.seq_len_tensor,
        batches,
        scaling,
        "shrink",
    )

    data.ref_out_tensor = data.ref_out_tensor.to(torch.float32)
    assert_close(data.our_out_tensor, data.ref_out_tensor)


def check_bgmv_expand(batches: int, num_loras: int, rank: int,
                      hidden_size: int, dtype: torch.dtype, device: str,
                      add_inputs: bool):
    """
    Compare vllm.bgmv_expand against a reference implementation.
    """
    seq_length = 1
    data: PunicaTensors = generate_data(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        dtype,
        "expand",
        device,
    )

    bgmv_expand(
        data.inputs_tensor,
        data.lora_weights,
        data.our_out_tensor,
        data.token_lora_mapping,
        add_inputs=add_inputs,
    )

    ref_torch_groupgemm(
        data.ref_out_tensor,
        data.inputs_tensor,
        data.lora_weights,
        data.token_lora_mapping,
        data.seq_len_tensor,
        batches,
        1.0,
        "expand",
    )
    # torch_bgmv_expand(
    #     data.inputs_tensor,
    #     data.lora_weights,
    #     data.ref_out_tensor,
    #     data.token_lora_mapping,
    #     add_inputs=add_inputs,
    # )
    assert_close(data.our_out_tensor, data.ref_out_tensor)


def check_bgmv_expand_slice(batches: int, num_loras: int, rank: int,
                            hidden_size: int, nslices: int, dtype: torch.dtype,
                            device: str, add_inputs: bool):
    """
    Compare vllm.bgmv_expand_slice against a reference implementation.
    """
    seq_length = 1
    data: PunicaTensors = generate_data_for_expand_nslices(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        dtype,
        nslices,
        device,
    )

    slice_offset = 0
    for index in range(nslices):
        bgmv_expand_slice(
            data.inputs_tensor,
            data.lora_weights[index],
            data.our_out_tensor,
            data.token_lora_mapping,
            slice_offset,
            slice_size=hidden_size,
            add_inputs=add_inputs,
        )
        ref_torch_groupgemm(
            # data.ref_out_tensor,
            data.ref_out_tensor[:, slice_offset:slice_offset + hidden_size],
            data.inputs_tensor,
            data.lora_weights[index],
            data.token_lora_mapping,
            data.seq_len_tensor,
            batches,
            1.0,
            op_type="expand",
        )

        slice_offset += hidden_size
    assert_close(data.our_out_tensor, data.ref_out_tensor)


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
#The size of TP
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
    "batches": [4],
    "num_loras": [4],
    "max_ranks": [32],
}

DTYPES = [torch.float16, torch.bfloat16]
DEVICES = [f"xpu:{0}"]
SEED = [0]


@pytest.mark.parametrize("batches", test_params['batches'])
@pytest.mark.parametrize("num_loras", test_params['num_loras'])
@pytest.mark.parametrize("rank", test_params['max_ranks'])
@pytest.mark.parametrize("hidden_size", test_params['hidden_sizes'])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
def test_bgmv(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    op_type: str,
):
    torch.set_default_device(device)
    seed_everything(seed)

    if op_type == "shrink":
        check_bgmv_shrink(batches=batches,
                          num_loras=num_loras,
                          rank=rank,
                          hidden_size=hidden_size,
                          dtype=dtype,
                          device=device,
                          scaling=0.5)
    else:
        check_bgmv_expand(batches=batches,
                          num_loras=num_loras,
                          rank=rank,
                          hidden_size=hidden_size,
                          dtype=dtype,
                          device=device,
                          add_inputs=True)


@pytest.mark.parametrize("batches", hs_test_params['batches'])
@pytest.mark.parametrize("num_loras", hs_test_params['num_loras'])
@pytest.mark.parametrize("rank", hs_test_params['max_ranks'])
@pytest.mark.parametrize("hidden_size", hs_test_params['hidden_sizes'])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
def test_bgmv_hidden_size(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    op_type: str,
):
    torch.set_default_device(device)
    seed_everything(seed)

    if op_type == "shrink":
        check_bgmv_shrink(batches=batches,
                          num_loras=num_loras,
                          rank=rank,
                          hidden_size=hidden_size,
                          dtype=dtype,
                          device=device,
                          scaling=0.5)
    else:
        check_bgmv_expand(batches=batches,
                          num_loras=num_loras,
                          rank=rank,
                          hidden_size=hidden_size,
                          dtype=dtype,
                          device=device,
                          add_inputs=True)


@pytest.mark.parametrize("batches", test_params['batches'])
@pytest.mark.parametrize("num_loras", test_params['num_loras'])
@pytest.mark.parametrize("rank", test_params['max_ranks'])
@pytest.mark.parametrize("hidden_size", test_params['hidden_sizes'])
@pytest.mark.parametrize("nslices", [2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_bgmv_expand_nslices(batches: int, num_loras: int, rank: int,
                             hidden_size: int, nslices: int,
                             dtype: torch.dtype, device: str, seed: int):

    torch.set_default_device(device)
    seed_everything(seed)

    check_bgmv_expand_slice(batches=batches,
                            num_loras=num_loras,
                            rank=rank,
                            hidden_size=hidden_size,
                            nslices=nslices,
                            dtype=dtype,
                            device=device,
                            add_inputs=True)


@pytest.mark.parametrize("batches", hs_test_params['batches'])
@pytest.mark.parametrize("num_loras", hs_test_params['num_loras'])
@pytest.mark.parametrize("rank", hs_test_params['max_ranks'])
@pytest.mark.parametrize("hidden_size", hs_test_params['hidden_sizes'])
@pytest.mark.parametrize("nslices", [2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_bgmv_expand_nslices_hidden_size(batches: int, num_loras: int,
                                         rank: int, hidden_size: int,
                                         nslices: int, dtype: torch.dtype,
                                         device: str, seed: int):

    torch.set_default_device(device)
    seed_everything(seed)

    check_bgmv_expand_slice(batches=batches,
                            num_loras=num_loras,
                            rank=rank,
                            hidden_size=hidden_size,
                            nslices=nslices,
                            dtype=dtype,
                            device=device,
                            add_inputs=True)
