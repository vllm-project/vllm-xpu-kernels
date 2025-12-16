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
                   unpermuted_row_to_permuted_row, num_experts):
    input_len = output.shape[0]
    topk = topk_weights.shape[1]
    moe_indices = unpermuted_row_to_permuted_row.view(input_len, topk)

    selected_outputs = moe_output[
        moe_indices]  # (input_len, topk, hidden_size)
    selected_outputs_fp32 = selected_outputs.float()
    topk_weights_fp32 = topk_weights.float()
    output_fp32 = torch.einsum('ijk,ij->ik', selected_outputs_fp32,
                               topk_weights_fp32)
    output.copy_(output_fp32.to(output.dtype))


@pytest.mark.parametrize("input_len", INPUT_LENGTHS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZE)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype",
                         [torch.bfloat16, torch.float16, torch.float32])
def test_moe_gather(input_len, hidden_size, num_experts, topk, dtype):
    seed_everything(7)

    num_moe_inputs = input_len * topk

    moe_output = torch.randn((num_moe_inputs, hidden_size),
                             dtype=dtype,
                             device=DEVICE)

    scores = torch.randn((input_len, num_experts), device=DEVICE, dtype=dtype)
    expert_scores, _ = torch.topk(scores, k=topk, dim=-1, sorted=False)

    unpermuted_row_to_permuted_row = torch.randperm((input_len * topk),
                                                    dtype=torch.int32,
                                                    device=DEVICE)

    output = torch.empty((input_len, hidden_size), device=DEVICE, dtype=dtype)
    torch.ops._moe_C.moe_gather(output, moe_output, expert_scores,
                                unpermuted_row_to_permuted_row, num_experts)

    ref_output = torch.empty((input_len, hidden_size),
                             device=DEVICE,
                             dtype=dtype)
    ref_moe_gather(ref_output, moe_output, expert_scores,
                   unpermuted_row_to_permuted_row, num_experts)

    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
