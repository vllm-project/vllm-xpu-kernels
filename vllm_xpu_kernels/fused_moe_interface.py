import torch
from torch.nn.modules.utils import _pair
from torch import nn, Tensor
from typing import List
import numpy

# from . import _vllm_fp8_C
def ref_gemm1(x, weight):
    w1, w3 = torch.split(w13, list(weight.shape)[0]/2, dim=0)
    act_fn = torch.nn.SiLU()
    gate = (x @ w1.T).silu()
    up = x @ w3.T
    return gate * up



def cutlass_fused_moe(hidden_states, w13, w2, topk_weights, topk_ids, n_experts_per_token, inplace, activation, num_experts):
    token_cnt, hidden_size = list(hidden_states.shape)
    intermediate_size = list(w2.shape)[-1]
    expert_cache = torch.empty_like(hidden_states, shape=(token_cnt, intermediate_size))
    idxs = topk_ids.argsort()
    counts = topk_ids.bincount().cpu().numpy()
    tokens_per_expert = counts.cumsum()
    num_per_tok = n_experts_per_token
    token_idxs = idxs // num_per_tok
    experts_intermeidate = []

    for expert_id, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        if start_idx == end_idx:
            continue

        expert_w13 = w13[expert_id, :, :]
        exp_token_idxs = token_idxs[start_idx:end_idx]
        expert_tokens = x[exp_token_idxs]
        expert_out    = ref_gemm1(expert_tokens, expert_w13)
        # expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
        # expert_cache.scatter_reduce_(
        #     0,
        #     exp_token_idxs.view(-1, 1).repeat(1, x.shape[-1]),
        #     expert_out,
        #     reduce='sum'
        # )
        experts_intermediate.append(expert_out)

    group_A = torch.stack(experts_intermediate, dim=0).contiguous()
    output = torch.emtpy_like(x)
    offset = tokens_per_expert
    print(group_A.shape)
    print(output.shape)
    print(tokens_per_expert)
    # gemm2(group_A, w2, output, offset)

