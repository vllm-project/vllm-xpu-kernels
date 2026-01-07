# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import vllm_xpu_kernels._moe_C  # noqa: F401
from tests.utils import seed_everything

DEVICE = "xpu"
INPUT_LENGTHS = [1, 8, 1024, 8192]
HIDDEN_SIZE = [128, 1024, 8192]
NUM_EXPERTS = [16, 32, 128]
TOP_KS = [1, 4, 6, 8]
EP_RANK = [0, 1, 2, 3]
EP_SIZE = [4]

#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "input_len": [1],
        "hidden_size": [128],
        "num_experts": [16],
        "topk": [2],
    },
}


def ref_moe_gather(output, moe_output, topk_weights,
                   permuted_row_to_unpermuted_row,
                   unpermuted_row_to_permuted_row, expert_first_token_offset,
                   num_experts):
    input_len = output.shape[0]
    hidden_size = output.shape[1]
    topk = topk_weights.shape[1]
    moe_indices = unpermuted_row_to_permuted_row.view(topk,
                                                      input_len).transpose(
                                                          0, 1)

    if expert_first_token_offset[-1] == 0:
        selected_outputs = torch.zeros_like(moe_output)
        selected_outputs = selected_outputs.view(
            [input_len, topk, hidden_size])
    else:
        selected_outputs = moe_output[
            moe_indices]  # (input_len, topk, hidden_size)
        mask = (moe_indices != 0).unsqueeze(-1)
        selected_outputs = selected_outputs * mask
        selected_outputs = selected_outputs.contiguous()

        unpermuted_row_to_permuted_row_0 = permuted_row_to_unpermuted_row[0]
        topk_idx = unpermuted_row_to_permuted_row_0 // input_len
        input_len_idx = unpermuted_row_to_permuted_row_0 % input_len
        transposed_idx = input_len_idx * topk + topk_idx
        selected_outputs.view([input_len * topk,
                               hidden_size])[transposed_idx] = moe_output[0]

    selected_outputs_fp32 = selected_outputs.float()
    topk_weights_fp32 = topk_weights.float()
    output_fp32 = torch.einsum('ijk,ij->ik', selected_outputs_fp32,
                               topk_weights_fp32)
    output.copy_(output_fp32.to(output.dtype))


@pytest.mark.parametrize("input_len", INPUT_LENGTHS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZE)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("ep_rank", EP_RANK)
@pytest.mark.parametrize("ep_size", EP_SIZE)
@pytest.mark.parametrize("dtype",
                         [torch.bfloat16, torch.float16, torch.float32])
def test_moe_gather(input_len, hidden_size, num_experts, topk, ep_rank,
                    ep_size, dtype):
    seed_everything(7)

    num_experts_per_node = num_experts // ep_size
    expert_start_id = num_experts_per_node * ep_rank
    expert_end_id = expert_start_id + num_experts_per_node

    num_moe_inputs = input_len * topk

    moe_output = torch.randn((num_moe_inputs, hidden_size),
                             dtype=dtype,
                             device=DEVICE)

    scores = torch.randn((input_len, num_experts),
                         device=DEVICE,
                         dtype=torch.float32)
    expert_scores, expert_indices = torch.topk(scores,
                                               k=topk,
                                               dim=-1,
                                               sorted=False)

    expert_row_counts = torch.bincount(expert_indices.flatten(),
                                       minlength=num_experts)
    expert_row_counts = expert_row_counts[expert_start_id:expert_end_id]
    expert_cumsum = torch.cumsum(expert_row_counts, dim=0)
    expert_first_token_offset = torch.cat(
        [torch.tensor([0], device=DEVICE), expert_cumsum])

    permuted_row_to_unpermuted_row = torch.randperm((input_len * topk),
                                                    dtype=torch.int32,
                                                    device=DEVICE)
    unpermuted_row_to_permuted_row = torch.argsort(
        permuted_row_to_unpermuted_row).to(torch.int32)
    permuted_row_to_unpermuted_row[expert_cumsum[-1]:] = 0
    unpermuted_row_to_permuted_row[unpermuted_row_to_permuted_row >=
                                   expert_cumsum[-1]] = 0

    output = torch.empty((input_len, hidden_size), device=DEVICE, dtype=dtype)
    torch.ops._moe_C.moe_gather(output, moe_output, expert_scores,
                                permuted_row_to_unpermuted_row,
                                unpermuted_row_to_permuted_row,
                                expert_first_token_offset,
                                num_experts_per_node)

    ref_output = torch.empty((input_len, hidden_size),
                             device=DEVICE,
                             dtype=dtype)
    ref_moe_gather(ref_output, moe_output, expert_scores,
                   permuted_row_to_unpermuted_row,
                   unpermuted_row_to_permuted_row, expert_first_token_offset,
                   num_experts_per_node)
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
