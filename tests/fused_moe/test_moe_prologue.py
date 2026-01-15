# SPDX-License-Identifier: Apache-2.0
import gc

import pytest
import torch

from tests.ops.fp8_quant_op import scaled_fp8_quant
from tests.utils import seed_everything
from vllm_xpu_kernels.fused_moe_interface import xpu_fused_moe

DEVICE = "xpu"

# shape for Llama-4-scout
FUSED_MOE_MNK_FACTORS = [
    (1, 5120, 8192),
    (4, 5120, 8192),
    (16, 5120, 8192),
    (8192, 5120, 8192),
]
NUM_EXPERTS = [16]
TOP_KS = [1]

MINI_PYTEST_PARAMS = {
    "default": {
        "m,n,k": [(1, 256, 128)],
        "e": [2],
        "topk": [1],
        "dtype": [torch.bfloat16],
        "has_bias": [True]
    }
}

RECIPE_TO_DTYPE = {
    "bf16": (torch.bfloat16, None),
    "fp16": (torch.float16, None),
    "mxfp8": (torch.float8_e4m3fn, torch.float8_e8m0fnu),
    "fp8block": (torch.float8_e4m3fn, torch.float32),
    "mxfp4": (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu),
}


def ceilDiv(a, b):
    return (a + b - 1) // b

def compute_num_tokens_per_block(num_tokens, num_experts_per_node):
    for num_tokens_per_block in [32, 64, 128, 256, 512, 1024]:
        num_blocks_per_seq = ceilDiv(num_tokens, num_tokens_per_block)
        if num_blocks_per_seq * num_experts_per_node <= num_tokens_per_block:
            return num_tokens_per_block
    return 1024


def ref_prologue(x,
                 scales,
                 flat_expert_indices,
                 num_per_tok,
                 num_experts,
                 recipe,
                 ep_rank=0,
                 ep_size=1):
    expert_start_id = num_experts * ep_rank
    expert_end_id = expert_start_id + num_experts

    idxs = flat_expert_indices.argsort()
    counts = flat_expert_indices.bincount().cpu().numpy()
    tokens_per_expert = counts.cumsum()
    token_idxs = idxs // num_per_tok

    data_dtype, scale_dtype = RECIPE_TO_DTYPE.get(recipe, (None, None))
    if recipe in ["mxfp8", "mxfp4"]:
        scales = scales.view(torch.uint8)

    expand_input = []
    expand_scales = []
    for expert_id, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        if (start_idx == end_idx) or (expert_id
                                      < expert_start_id) or (expert_id
                                                             >= expert_end_id):
            continue
        exp_token_idxs = token_idxs[start_idx:end_idx]
        expert_tokens = x[exp_token_idxs]
        expand_input.append(expert_tokens)
        if scales is not None:
            expert_scales = scales[exp_token_idxs]
            expand_scales.append(expert_scales)

    expert_first_token_offset = torch.Tensor(tokens_per_expert).to(torch.int64)

    expand_input = torch.cat(expand_input, dim=0)
    if recipe == "mxfp4":
        expand_input = expand_input.to(torch.uint8).view(torch.float4_e2m1fn_x2)
    else:
        expand_input = expand_input.to(data_dtype)
    expand_scales = None if scales is None else torch.cat(expand_scales, dim=0)

    return expert_first_token_offset, expand_input, expand_scales


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("recipe", ["bf16", "fp16", "mxfp8", "mxfp4", "fp8block"])
def test_prologue(m, n, k, e, topk, recipe):
    seed_everything(7)
    data_dtype, scale_dtype = RECIPE_TO_DTYPE.get(recipe, (None, None))

    input_len = m
    hidden_size = k
    inter_size = n
    num_experts = e
    block_k = 0

    if data_dtype in [torch.bfloat16, torch.float16]:
        hidden_states = torch.randn((input_len, hidden_size), device=DEVICE, dtype=data_dtype) / 16
        ref_a = hidden_states.clone()
        scales = None
    elif data_dtype is torch.float8_e4m3fn:
        hidden_states_fp32 = torch.randn((input_len, hidden_size), device=DEVICE, dtype=torch.float32)
        hidden_states = hidden_states_fp32.to(torch.float8_e4m3fn)
        ref_a = hidden_states_fp32
        if recipe == "fp8block":
            block_k = 128
            scales = torch.randn((input_len, hidden_size // block_k), device=DEVICE, dtype=torch.float32)
        elif recipe == "mxfp8":
            block_k = 32
            scales = torch.randint(1,256,(input_len, hidden_size // block_k), device=DEVICE, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    elif data_dtype is torch.float4_e2m1fn_x2:
        block_k = 16  # two input elem in a 8bit
        hidden_states_fp32 = torch.randn((input_len, hidden_size // 2), device=DEVICE, dtype=torch.bfloat16)
        hidden_states = hidden_states_fp32.to(torch.uint8).view(torch.float4_e2m1fn_x2)
        ref_a = hidden_states_fp32
        scales = torch.randint(1,256,(input_len, hidden_size // block_k), device=DEVICE, dtype=torch.uint8).view(torch.float8_e8m0fnu)

    # moe gate
    scores = torch.randn((input_len, num_experts),
                         device=DEVICE,
                         dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(scores,
                                               k=topk,
                                               dim=-1,
                                               sorted=False)

    flat_expert_indices = topk_ids.view(-1)
    flat_expert_weights = topk_weights.view(-1, 1)

    ref_expert_offset, ref_expand_input, ref_expand_scales = ref_prologue(ref_a, scales, flat_expert_indices, topk, e, recipe)

    n_experts_per_token=topk
    num_experts=e
    num_rows, hidden_size = list(hidden_states.shape)
    num_experts_per_node = num_experts
    experts_per_token = n_experts_per_token
    num_moe_inputs = n_experts_per_token * num_rows
    permuted_elems = num_moe_inputs * hidden_size
    # interbuf_elems = num_moe_inputs * inter_size
    permuted_row_to_unpermuted_row_size = num_moe_inputs * 4
    permuted_token_selected_experts_size = num_moe_inputs * 4
    src_to_dest_map_size = experts_per_token * num_rows * 4
    expert_first_token_offset_size = (num_experts_per_node + 1) * 8
    num_tokens_per_block = compute_num_tokens_per_block(
        num_rows, num_experts_per_node)
    num_blocks_per_seq = ceilDiv(num_rows, num_tokens_per_block)
    blocked_expert_counts_size = num_experts_per_node * num_blocks_per_seq * 4
    blocked_expert_counts_cumsum_size = blocked_expert_counts_size
    blocked_row_to_unpermuted_row_size = num_experts_per_node * num_rows * 4
    permuted_data_size = permuted_elems * data_dtype.itemsize if scales is not None else 0
    permuted_act_scales_size = (permuted_elems // block_k) * scale_dtype.itemsize if scales is not None else 0
    permuted_token_final_scales_size = num_moe_inputs * 4

    ws_map = {}
    map_offset = 0

    def config_ws(name, size):
        nonlocal map_offset
        if size % 256 != 0:
            size += 256 - size % 256
        ws_map[name] = (size, map_offset)
        map_offset += size

    config_ws("permuted_row_to_unpermuted_row",
              permuted_row_to_unpermuted_row_size)
    config_ws("permuted_token_selected_experts",
              permuted_token_selected_experts_size)
    config_ws("unpermuted_row_to_permuted_row", src_to_dest_map_size)
    config_ws("blocked_expert_counts", blocked_expert_counts_size)
    config_ws("blocked_expert_counts_cumsum",
              blocked_expert_counts_cumsum_size)
    config_ws("blocked_row_to_unpermuted_row",
              blocked_row_to_unpermuted_row_size)
    config_ws("expert_first_token_offset", expert_first_token_offset_size)
    config_ws("permuted_token_final_scales", permuted_token_final_scales_size)
    config_ws("overlapped_gemm1_gemm2_inputs", permuted_data_size)
    config_ws("permuted_act_scales", permuted_act_scales_size)

    workspace = torch.zeros(map_offset,
                            dtype=torch.uint8,
                            device=hidden_states.device)
    if topk_ids.dtype == torch.int32:
        topk_ids = topk_ids.to(torch.int64)
    torch.ops._moe_C.fused_moe_prologue(
        input=hidden_states,
        input_scales=scales,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        workspace=workspace,
        hidden_size=hidden_size,
        inter_size=inter_size,
        block_k=block_k,
        ep_rank=0,
        ep_size=1,
        num_experts_on_rank=num_experts_per_node)

    expert_first_token_offset = workspace[
        ws_map["expert_first_token_offset"][1]:
        ws_map["expert_first_token_offset"][1] +
        expert_first_token_offset_size].view(torch.int64)
    permuted_row_to_unpermuted_row = workspace[
        ws_map["permuted_row_to_unpermuted_row"][1]:
        ws_map["permuted_row_to_unpermuted_row"][1] +
        permuted_row_to_unpermuted_row_size].view(torch.int32)
    unpermuted_row_to_permuted_row = workspace[
        ws_map["unpermuted_row_to_permuted_row"][1]:
        ws_map["unpermuted_row_to_permuted_row"][1] +
        src_to_dest_map_size].view(torch.int32)
    expand_input = workspace[ws_map["overlapped_gemm1_gemm2_inputs"][1]:
                            ws_map["overlapped_gemm1_gemm2_inputs"][1] +
                            permuted_data_size].view(hidden_states.dtype).view(num_moe_inputs, hidden_size)
    expand_scales = workspace[ws_map["permuted_act_scales"][1]: ws_map["permuted_act_scales"][1] + permuted_act_scales_size]
    if scale_dtype == torch.float32:
        expand_scales = expand_scales.view(torch.float32)
    expand_scales = expand_scales.view(num_moe_inputs, hidden_size//block_k)

    torch.testing.assert_close(ref_expert_offset.cpu(), expert_first_token_offset[1:1+ref_expert_offset.numel()].cpu(), rtol=0, atol=0)
    if data_dtype is not torch.float4_e2m1fn_x2:
        torch.testing.assert_close(ref_expand_input, expand_input, rtol=0, atol=0)
    else:
        torch.testing.assert_close(ref_expand_input.view(torch.uint8), expand_input.view(torch.uint8), rtol=0, atol=0)
    if scales is not None:
        torch.testing.assert_close(ref_expand_scales, expand_scales, rtol=0, atol=0)

if __name__ == "__main__":
    test_prologue(m=100, n=512, k=1024, e=16, topk=2, recipe="fp8block")
