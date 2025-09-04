import torch
from torch.nn.modules.utils import _pair
from torch import nn, Tensor
from typing import List
import numpy
import vllm_xpu_kernels._xpu_C

def cutlass_grouped_gemm(input_A, input_B, output, offset, n, k, num_experts):
    torch.ops._xpu_C.cutlass_grouped_gemm(input_A, input_B, output, offset, n, k, num_experts)

def cutlass_fused_moe(hidden_states, w13, w2, topk_weights, topk_ids, n_experts_per_token, activation, num_experts):
    token_cnt, hidden_size = list(hidden_states.shape)
    intermediate_size = list(w2.shape)[-1]
    expert_cache = torch.empty((token_cnt, hidden_size),
                           dtype=torch.float32,
                           device=hidden_states.device)

    # map token to experts
    idxs = topk_ids.argsort()
    counts = topk_ids.to(torch.long).bincount().cpu().numpy()
    tokens_per_expert = counts.cumsum()
    num_per_tok = n_experts_per_token
    token_idxs = idxs // num_per_tok
    grouped_input_A = []
    offset = []
    for expert_id, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        offset.append(end_idx - start_idx)
        if start_idx == end_idx:
            continue
        exp_token_idxs = token_idxs[start_idx:end_idx]
        expert_tokens = hidden_states[exp_token_idxs]
        grouped_input_A.append(expert_tokens)

    total_input_size = token_cnt * num_per_tok

    # gemm1
    input_A = torch.cat(grouped_input_A, dim=0).contiguous()
    input_B = w13.transpose(-1, -2).contiguous().transpose(-1, -2)
    assert(list(input_A.shape)[0] == total_input_size)
    gemm1_output = torch.empty((total_input_size, 2*intermediate_size), dtype=torch.float32,device=hidden_states.device)
    offset = torch.tensor(offset, dtype=torch.int64, device='cpu')
    cutlass_grouped_gemm(input_A=input_A,
                         input_B=input_B,
                         output=gemm1_output,
                         offset=offset,
                         n=2*intermediate_size,
                         k=hidden_size,
                         num_experts=num_experts)
    # act
    gate, up = torch.split(gemm1_output, intermediate_size, dim=1)
    act = torch.nn.SiLU()
    act_output = act(gate) * up


    # gemm 2
    group_A = act_output.to(torch.bfloat16).contiguous()
    output = torch.empty((list(group_A.shape)[0], hidden_size), dtype=torch.float32, device=hidden_states.device)
    w2 = w2.transpose(-1, -2).contiguous().transpose(-1, -2)
    cutlass_grouped_gemm(input_A=group_A,
                         input_B=w2,
                         output=output,
                         offset=offset,
                         n=hidden_size,
                         k=intermediate_size,
                         num_experts=num_experts)

    # apply scores
    for expert_id, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        if start_idx == end_idx:
            continue

        exp_token_idxs = token_idxs[start_idx:end_idx]
        expert_tokens = hidden_states[exp_token_idxs]
        expert_out = output[start_idx:end_idx]
        expert_out.mul_(topk_weights[idxs[start_idx:end_idx]])
        expert_cache.scatter_reduce_(
            0,
            exp_token_idxs.view(-1, 1).repeat(1, hidden_size),
            expert_out,
            reduce='sum'
        )

    return expert_cache
