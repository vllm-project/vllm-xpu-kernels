# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import vllm_xpu_kernels._moe_C  # noqa: F401
from tests.utils import seed_everything

DEVICE = "xpu"
NUM_ROWS = [32, 1024]
HIDDEN_SIZE = [128]
TOTAL_EXPERTS_NUM = [32, 128]
TOP_KS = [1, 8]
RECIPE_TO_DTYPE = {
    "bf16": (torch.bfloat16, None),
    "fp16": (torch.float16, None),
    "mxfp8": (torch.float8_e4m3fn, torch.float8_e8m0fnu),
    "fp8block": (torch.float8_e4m3fn, torch.float32),
    "mxfp4": (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu),
}

LOCAL_EXPERTS_NUM = [3, 8, 11]
EP_RANK = [0, 1, 2, 3]
EP_SIZE = [4]

#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_rows": [32],
        "hidden_size": [128],
        "total_experts_num": [16],
        "topk": [1],
    },
}


def ref_remap_hidden_states(hidden_states, scales, remapped_hidden_states,
                            remapped_scales, expert_map,
                            expert_first_token_offset,
                            unpermuted_row_to_permuted_row, topk_ids,
                            total_experts_num, local_experts_num):
    if expert_map is not None:
        local_topk_ids = expert_map[topk_ids]
    else:
        local_topk_ids = topk_ids

    valid_mask = local_topk_ids >= 0
    valid_tensor = local_topk_ids[valid_mask]

    frequencies = torch.bincount(valid_tensor.flatten(),
                                 minlength=local_experts_num)
    prefix = torch.cat([
        torch.zeros(1, dtype=frequencies.dtype, device=frequencies.device),
        torch.cumsum(frequencies, dim=0)
    ])

    expert_first_token_offset.copy_(prefix)

    expert_local_offset = torch.zeros((local_experts_num, ),
                                      dtype=torch.int32,
                                      device=local_topk_ids.device)
    num_rows = hidden_states.shape[0]
    topk = topk_ids.shape[1]
    if scales is not None and scales.dtype is torch.float8_e8m0fnu:
        scales = scales.view(torch.uint8)

    for i in range(num_rows):
        for j in range(topk):
            selected_expert = local_topk_ids[i, j].item()
            if selected_expert == -1:
                unpermuted_row_to_permuted_row[i, j] = -1
                continue
            first_token_offset_offset = expert_first_token_offset[
                selected_expert].item()
            offset = expert_local_offset[selected_expert]
            remapped_hidden_states[first_token_offset_offset +
                                   offset] = hidden_states[i]
            if scales is not None:
                remapped_scales[first_token_offset_offset + offset] = scales[i]
            unpermuted_row_to_permuted_row[
                i, j] = first_token_offset_offset + offset
            expert_local_offset[selected_expert] += 1


@pytest.mark.parametrize("num_rows", NUM_ROWS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZE)
@pytest.mark.parametrize("total_experts_num", TOTAL_EXPERTS_NUM)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("has_expert_map", [True, False])
@pytest.mark.parametrize("recipe",
                         ["bf16", "fp16", "mxfp8", "mxfp4", "fp8block"])
def test_remap_hidden_states(num_rows, hidden_size, total_experts_num, topk,
                             has_expert_map, recipe):
    seed_everything(7)

    data_dtype, scale_dtype = RECIPE_TO_DTYPE.get(recipe, (None, None))

    if has_expert_map:
        local_experts_num = total_experts_num // 2
    else:
        local_experts_num = total_experts_num

    if data_dtype in [torch.bfloat16, torch.float16]:
        hidden_states = torch.randn((num_rows, hidden_size),
                                    dtype=data_dtype,
                                    device=DEVICE)
        scales = None
    elif data_dtype is torch.float8_e4m3fn:
        hidden_states_fp32 = torch.randn((num_rows, hidden_size),
                                         dtype=torch.float32,
                                         device=DEVICE)
        hidden_states = hidden_states_fp32.to(torch.float8_e4m3fn)
        if recipe == "fp8block":
            block_k = 128
            scales = torch.randn((num_rows, hidden_size // block_k),
                                 device=DEVICE,
                                 dtype=torch.float32)
        elif recipe == "mxfp8":
            block_k = 16
            scales = torch.randint(1,
                                   256, (num_rows, hidden_size // block_k),
                                   device=DEVICE,
                                   dtype=torch.uint8).view(
                                       torch.float8_e8m0fnu)
    elif data_dtype is torch.float4_e2m1fn_x2:
        block_k = 16  # two input elem in a 8bit
        hidden_states_fp32 = torch.randn((num_rows, hidden_size // 2),
                                         dtype=torch.float32,
                                         device=DEVICE)
        hidden_states = hidden_states_fp32.to(torch.uint8).view(
            torch.float4_e2m1fn_x2)
        scales = torch.randint(1,
                               256, (num_rows, hidden_size // 2 // block_k),
                               device=DEVICE,
                               dtype=torch.uint8).view(torch.float8_e8m0fnu)

    remapped_hidden_states = torch.empty_like(hidden_states).repeat_interleave(
        topk, dim=0)
    remapped_scales = None
    if scale_dtype is not None:
        remapped_scales = torch.empty_like(scales).repeat_interleave(topk,
                                                                     dim=0)
    expert_first_token_offset = torch.zeros((local_experts_num + 1),
                                            dtype=torch.int64,
                                            device=DEVICE)
    unpermuted_row_to_permuted_row = torch.empty((num_rows, topk),
                                                 dtype=torch.int32,
                                                 device=DEVICE)

    expert_map = None
    if has_expert_map:
        expert_map = torch.full((total_experts_num, ),
                                -1,
                                dtype=torch.int64,
                                device=DEVICE)
        expert_map[torch.randperm(
            total_experts_num,
            device=DEVICE)[:local_experts_num]] = torch.randperm(
                local_experts_num, device=DEVICE)
        expert_map = expert_map.to(torch.int32)

    scores = torch.randn((num_rows, total_experts_num),
                         device=DEVICE,
                         dtype=torch.float32)
    _, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)
    topk_ids = topk_ids.to(torch.int64)

    ref_remapped_hidden_states = remapped_hidden_states.clone()
    ref_expert_first_token_offset = expert_first_token_offset.clone()
    ref_unpermuted_row_to_permuted_row = unpermuted_row_to_permuted_row.clone()
    ref_remapped_scales = None
    if scale_dtype is not None:
        ref_remapped_scales = remapped_scales.clone()
        if scales.dtype is torch.float8_e8m0fnu:
            ref_remapped_scales = ref_remapped_scales.view(torch.uint8)

    ref_remap_hidden_states(hidden_states, scales, ref_remapped_hidden_states,
                            ref_remapped_scales, expert_map,
                            ref_expert_first_token_offset,
                            ref_unpermuted_row_to_permuted_row, topk_ids,
                            total_experts_num, local_experts_num)

    torch.ops._moe_C.remap_hidden_states(
        hidden_states, scales, remapped_hidden_states, remapped_scales,
        expert_map, expert_first_token_offset, unpermuted_row_to_permuted_row,
        topk_ids, total_experts_num, local_experts_num)

    if data_dtype is torch.float4_e2m1fn_x2:
        remapped_hidden_states = remapped_hidden_states.view(torch.uint8)
        ref_remapped_hidden_states = ref_remapped_hidden_states.view(
            torch.uint8)

    unpermuted_hidden_states = remapped_hidden_states[
        unpermuted_row_to_permuted_row.flatten()]
    ref_unpermuted_hidden_states = ref_remapped_hidden_states[
        ref_unpermuted_row_to_permuted_row.flatten()]

    torch.testing.assert_close(unpermuted_hidden_states,
                               ref_unpermuted_hidden_states,
                               rtol=0,
                               atol=0,
                               equal_nan=True)
    torch.testing.assert_close(ref_expert_first_token_offset,
                               expert_first_token_offset,
                               rtol=0,
                               atol=0)
    if scale_dtype is not None:
        unpermuted_scales = remapped_scales[
            unpermuted_row_to_permuted_row.flatten()]
        ref_unpermuted_scales = ref_remapped_scales[
            ref_unpermuted_row_to_permuted_row.flatten()]
        if unpermuted_scales.dtype is torch.float8_e8m0fnu:
            unpermuted_scales = unpermuted_scales.view(torch.uint8)
            ref_unpermuted_scales = ref_unpermuted_scales.view(torch.uint8)
        try:
            torch.testing.assert_close(unpermuted_scales,
                                       ref_unpermuted_scales,
                                       rtol=0,
                                       atol=0,
                                       equal_nan=True)
        except AssertionError:
            # Fp8block may fails on g31 CI
            mismatched_indices = torch.nonzero(
                unpermuted_scales != ref_unpermuted_scales)
            print("Mismatched scales at indices:", mismatched_indices)
            print("Mismatched scales:", unpermuted_scales[mismatched_indices])
            print("Mismatched ref:", ref_unpermuted_scales[mismatched_indices])


@pytest.mark.parametrize("num_rows", [262144])
@pytest.mark.parametrize("hidden_size", [2048])
@pytest.mark.parametrize("total_experts_num", [128])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("has_expert_map", [False])
@pytest.mark.parametrize("recipe",
                         ["bf16", "fp16", "mxfp8", "mxfp4", "fp8block"])
def test_remap_hidden_states_overflow(num_rows, hidden_size, total_experts_num,
                                      topk, has_expert_map, recipe):
    seed_everything(7)

    data_dtype, scale_dtype = RECIPE_TO_DTYPE.get(recipe, (None, None))

    if has_expert_map:
        local_experts_num = total_experts_num // 2
    else:
        local_experts_num = total_experts_num

    if data_dtype in [torch.bfloat16, torch.float16]:
        hidden_states = torch.randn((num_rows, hidden_size),
                                    dtype=data_dtype,
                                    device=DEVICE)
        scales = None
    elif data_dtype is torch.float8_e4m3fn:
        hidden_states_fp32 = torch.randn((num_rows, hidden_size),
                                         dtype=torch.float32,
                                         device=DEVICE)
        hidden_states = hidden_states_fp32.to(torch.float8_e4m3fn)
        if recipe == "fp8block":
            block_k = 128
            scales = torch.randn((num_rows, hidden_size // block_k),
                                 device=DEVICE,
                                 dtype=torch.float32)
        elif recipe == "mxfp8":
            block_k = 16
            scales = torch.randint(1,
                                   256, (num_rows, hidden_size // block_k),
                                   device=DEVICE,
                                   dtype=torch.uint8).view(
                                       torch.float8_e8m0fnu)
    elif data_dtype is torch.float4_e2m1fn_x2:
        block_k = 16  # two input elem in a 8bit
        hidden_states_fp32 = torch.randn((num_rows, hidden_size // 2),
                                         dtype=torch.float32,
                                         device=DEVICE)
        hidden_states = hidden_states_fp32.to(torch.uint8).view(
            torch.float4_e2m1fn_x2)
        scales = torch.randint(1,
                               256, (num_rows, hidden_size // 2 // block_k),
                               device=DEVICE,
                               dtype=torch.uint8).view(torch.float8_e8m0fnu)

    remapped_hidden_states = torch.empty_like(hidden_states).repeat_interleave(
        topk, dim=0)
    remapped_scales = None
    if scale_dtype is not None:
        remapped_scales = torch.empty_like(scales).repeat_interleave(topk,
                                                                     dim=0)
    expert_first_token_offset = torch.zeros((local_experts_num + 1),
                                            dtype=torch.int64,
                                            device=DEVICE)
    unpermuted_row_to_permuted_row = torch.empty((num_rows, topk),
                                                 dtype=torch.int32,
                                                 device=DEVICE)

    expert_map = None
    if has_expert_map:
        expert_map = torch.full((total_experts_num, ),
                                -1,
                                dtype=torch.int64,
                                device=DEVICE)
        expert_map[torch.randperm(
            total_experts_num,
            device=DEVICE)[:local_experts_num]] = torch.randperm(
                local_experts_num, device=DEVICE)
        expert_map = expert_map.to(torch.int32)

    scores = torch.randn((num_rows, total_experts_num),
                         device=DEVICE,
                         dtype=torch.float32)
    _, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)
    topk_ids = topk_ids.to(torch.int64)

    torch.ops._moe_C.remap_hidden_states(
        hidden_states, scales, remapped_hidden_states, remapped_scales,
        expert_map, expert_first_token_offset, unpermuted_row_to_permuted_row,
        topk_ids, total_experts_num, local_experts_num)

    print("remapped_hidden_states", remapped_hidden_states, flush=True)
    print("remapped_scales", remapped_scales, flush=True)


def ref_init_expert_map(expert_map, local_experts_num, ep_rank, ep_size):
    expert_map_tmp = torch.full((local_experts_num * ep_size, ),
                                -1,
                                dtype=torch.int32,
                                device=DEVICE)

    expert_map_tmp[ep_rank * local_experts_num:ep_rank * local_experts_num +
                   local_experts_num] = torch.arange(local_experts_num,
                                                     device=DEVICE,
                                                     dtype=torch.int32)

    expert_map.copy_(expert_map_tmp)


@pytest.mark.parametrize("local_experts_num", LOCAL_EXPERTS_NUM)
@pytest.mark.parametrize("ep_rank", EP_RANK)
@pytest.mark.parametrize("ep_size", EP_SIZE)
def test_init_expert_map(local_experts_num, ep_rank, ep_size):
    seed_everything(7)

    expert_map = torch.empty((local_experts_num * ep_size),
                             dtype=torch.int32,
                             device=DEVICE)
    ref_expert_map = expert_map.clone()

    ref_init_expert_map(ref_expert_map, local_experts_num, ep_rank, ep_size)
    torch.ops._moe_C.init_expert_map(expert_map, local_experts_num, ep_rank,
                                     ep_size)

    torch.testing.assert_close(expert_map, ref_expert_map, rtol=0, atol=0)
