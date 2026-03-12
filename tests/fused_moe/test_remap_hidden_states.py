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
DTYPE = [torch.bfloat16, torch.float16, torch.float32]

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


def ref_remap_hidden_states(hidden_states, remapped_hidden_states, expert_map,
                            expert_first_token_offset,
                            unpermuted_row_to_permuted_row, topk_ids, num_rows,
                            hidden_size, topk, total_experts_num,
                            local_experts_num):
    local_topk_ids = expert_map[topk_ids]

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
            unpermuted_row_to_permuted_row[
                i, j] = first_token_offset_offset + offset
            expert_local_offset[selected_expert] += 1


@pytest.mark.parametrize("num_rows", NUM_ROWS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZE)
@pytest.mark.parametrize("total_experts_num", TOTAL_EXPERTS_NUM)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", DTYPE)
def test_remap_hidden_states(num_rows, hidden_size, total_experts_num, topk,
                             dtype):
    seed_everything(7)

    local_experts_num = total_experts_num // 2

    hidden_states = torch.randn((num_rows, hidden_size),
                                dtype=dtype,
                                device=DEVICE)

    remapped_hidden_states = torch.empty((num_rows * topk, hidden_size),
                                         dtype=dtype,
                                         device=DEVICE)
    expert_first_token_offset = torch.zeros((local_experts_num + 1),
                                            dtype=torch.int64,
                                            device=DEVICE)
    unpermuted_row_to_permuted_row = torch.empty((num_rows, topk),
                                                 dtype=torch.int32,
                                                 device=DEVICE)

    expert_map = torch.full((total_experts_num, ),
                            -1,
                            dtype=torch.int64,
                            device=DEVICE)
    expert_map[torch.randperm(
        total_experts_num,
        device=DEVICE)[:local_experts_num]] = torch.randperm(local_experts_num,
                                                             device=DEVICE)
    expert_map = expert_map.to(torch.int32)

    scores = torch.randn((num_rows, total_experts_num),
                         device=DEVICE,
                         dtype=torch.float32)
    _, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)
    topk_ids = topk_ids.to(torch.int64)

    ref_remapped_hidden_states = remapped_hidden_states.clone()
    ref_expert_first_token_offset = expert_first_token_offset.clone()
    ref_unpermuted_row_to_permuted_row = unpermuted_row_to_permuted_row.clone()

    ref_remap_hidden_states(hidden_states, ref_remapped_hidden_states,
                            expert_map, ref_expert_first_token_offset,
                            ref_unpermuted_row_to_permuted_row, topk_ids,
                            num_rows, hidden_size, topk, total_experts_num,
                            local_experts_num)

    torch.ops._moe_C.remap_hidden_states(hidden_states, remapped_hidden_states,
                                         expert_map, expert_first_token_offset,
                                         unpermuted_row_to_permuted_row,
                                         topk_ids, num_rows, hidden_size, topk,
                                         total_experts_num, local_experts_num)

    unpermuted_hidden_states = remapped_hidden_states[
        unpermuted_row_to_permuted_row.flatten()]
    ref_unpermuted_hidden_states = ref_remapped_hidden_states[
        ref_unpermuted_row_to_permuted_row.flatten()]

    torch.testing.assert_close(unpermuted_hidden_states,
                               ref_unpermuted_hidden_states,
                               rtol=1e-2,
                               atol=1e-2,
                               equal_nan=True)
    torch.testing.assert_close(ref_expert_first_token_offset,
                               expert_first_token_offset,
                               rtol=1e-2,
                               atol=1e-2)


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
