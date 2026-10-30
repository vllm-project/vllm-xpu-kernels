# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import vllm_xpu_kernels._moe_C  # noqa: F401
from tests.utils import KERNEL_DEVICE, format_tc, seed_everything

DEVICE = "xpu"
INPUT_LENGTHS = [1, 8, 1024, 8192]
HIDDEN_SIZE = [128, 1024, 8192]
NUM_EXPERTS = [16, 32, 128]
TOP_KS = [1, 3, 4, 6, 8, 12, 16]
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


def _to_kernel(x):
    return x.to(KERNEL_DEVICE)


def ref_moe_gather(output, moe_output, topk_weights,
                   unpermuted_row_to_permuted_row,
                   num_experts):
    input_len = output.shape[0]
    topk = topk_weights.shape[1]
    moe_indices = unpermuted_row_to_permuted_row.view(input_len, topk)

    output_fp32 = torch.zeros_like(output, dtype=torch.float32)
    for topk_idx in range(topk):
        indices = moe_indices[:, topk_idx]
        valid_weights = topk_weights[:, topk_idx] * (indices >= 0)
        selected_output = moe_output[indices.clamp_min(0)].float()
        selected_output.mul_(valid_weights[:, None])
        output_fp32.add_(selected_output)

    output.copy_(output_fp32.to(output.dtype))


def _build_case(input_len, hidden_size, num_experts, topk, ep_rank,
                ep_size, dtype):
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

    permuted_row_to_unpermuted_row = torch.randperm((input_len * topk),
                                                    dtype=torch.int32,
                                                    device=DEVICE)
    unpermuted_row_to_permuted_row = torch.argsort(
        permuted_row_to_unpermuted_row).to(torch.int32)
    unpermuted_row_to_permuted_row[unpermuted_row_to_permuted_row >=
                                   expert_cumsum[-1]] = -1

    return (moe_output, expert_scores, unpermuted_row_to_permuted_row,
            num_experts_per_node)


@pytest.mark.parametrize("input_len", INPUT_LENGTHS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZE)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("ep_rank", EP_RANK)
@pytest.mark.parametrize("ep_size", EP_SIZE)
@pytest.mark.parametrize("dtype",
                         [torch.bfloat16, torch.float16, torch.float32],
                         ids=format_tc)
def test_moe_gather(input_len, hidden_size, num_experts, topk, ep_rank,
                    ep_size, dtype):
    seed_everything(7)

    (moe_output, expert_scores, unpermuted_row_to_permuted_row,
     num_experts_per_node) = _build_case(input_len, hidden_size, num_experts,
                                         topk, ep_rank, ep_size, dtype)

    output = torch.empty((input_len, hidden_size), device=DEVICE, dtype=dtype)
    torch.ops._moe_C.moe_gather(output, moe_output, expert_scores,
                                unpermuted_row_to_permuted_row,
                                num_experts_per_node)

    ref_output = torch.empty((input_len, hidden_size),
                             device=DEVICE,
                             dtype=dtype)
    ref_moe_gather(ref_output, moe_output, expert_scores,
                   unpermuted_row_to_permuted_row,
                   num_experts_per_node)
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)


PAD_ELEMS = [1, 8, 64]


@pytest.mark.parametrize("pad", PAD_ELEMS)
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("dtype",
                         [torch.bfloat16, torch.float16, torch.float32],
                         ids=format_tc)
def test_moe_gather_padded_src_stride(pad, hidden_size, dtype):
    seed_everything(7)

    input_len, num_experts, topk, ep_rank, ep_size = 64, 32, 4, 1, 4
    (moe_output, expert_scores, unpermuted_row_to_permuted_row,
     num_experts_per_node) = _build_case(input_len, hidden_size, num_experts,
                                         topk, ep_rank, ep_size, dtype)

    # The pitch has to be built on the device: transferring a padded CPU view
    # would silently repack it into a contiguous tensor and test nothing.
    pitch = hidden_size + pad
    storage = torch.empty((moe_output.shape[0], pitch),
                          dtype=dtype,
                          device=KERNEL_DEVICE)
    storage.fill_(float("nan"))
    padded = storage[:, :hidden_size]
    padded.copy_(moe_output)
    assert padded.stride() == (pitch, 1)

    output_kernel = torch.empty((input_len, hidden_size),
                                device=KERNEL_DEVICE,
                                dtype=dtype)
    torch.ops._moe_C.moe_gather(output_kernel, padded,
                                _to_kernel(expert_scores),
                                _to_kernel(unpermuted_row_to_permuted_row),
                                num_experts_per_node)

    ref_output = torch.empty((input_len, hidden_size),
                             device=DEVICE,
                             dtype=dtype)
    ref_moe_gather(ref_output, moe_output, expert_scores,
                   unpermuted_row_to_permuted_row, num_experts_per_node)
    torch.testing.assert_close(output_kernel.to(ref_output.device),
                               ref_output,
                               rtol=1e-2,
                               atol=1e-2)


def _valid_kernel_args(input_len=8, hidden_size=128, topk=4,
                       dtype=torch.bfloat16):
    num_moe_inputs = input_len * topk
    return {
        "output":
        torch.empty((input_len, hidden_size),
                    dtype=dtype,
                    device=KERNEL_DEVICE),
        "moe_output":
        torch.randn((num_moe_inputs, hidden_size),
                    dtype=dtype,
                    device=KERNEL_DEVICE),
        "topk_weights":
        torch.randn((input_len, topk),
                    dtype=torch.float32,
                    device=KERNEL_DEVICE),
        "row_map":
        torch.arange(num_moe_inputs,
                     dtype=torch.int32,
                     device=KERNEL_DEVICE),
        "num_experts_per_node":
        8,
    }


def _call(args):
    torch.ops._moe_C.moe_gather(args["output"], args["moe_output"],
                                args["topk_weights"], args["row_map"],
                                args["num_experts_per_node"])


def test_moe_gather_accepts_valid_layout():
    seed_everything(7)
    _call(_valid_kernel_args())


def test_moe_gather_rejects_non_contiguous_output():
    seed_everything(7)
    args = _valid_kernel_args()
    args["output"] = torch.empty((8, 256),
                                 dtype=torch.bfloat16,
                                 device=KERNEL_DEVICE)[:, :128]
    assert not args["output"].is_contiguous()
    with pytest.raises(RuntimeError, match="output must be contiguous"):
        _call(args)


def test_moe_gather_rejects_strided_hidden_dim():
    seed_everything(7)
    args = _valid_kernel_args()
    args["moe_output"] = torch.randn((128, 32),
                                     dtype=torch.bfloat16,
                                     device=KERNEL_DEVICE).t()
    assert args["moe_output"].stride(-1) != 1
    with pytest.raises(RuntimeError,
                       match="unit-stride along the hidden dimension"):
        _call(args)


def test_moe_gather_rejects_row_stride_below_hidden_size():
    seed_everything(7)
    args = _valid_kernel_args()
    storage = torch.randn((33, 128), dtype=torch.bfloat16,
                          device=KERNEL_DEVICE)
    args["moe_output"] = storage.as_strided((32, 128), (127, 1))
    assert args["moe_output"].stride(-2) < args["moe_output"].shape[-1]
    with pytest.raises(RuntimeError, match="smaller than hidden_size"):
        _call(args)
