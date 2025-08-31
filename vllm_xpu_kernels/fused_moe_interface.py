import torch
from torch.nn.modules.utils import _pair
from torch import nn, Tensor
from typing import List
import numpy

from . import _vllm_fp8_C
def ref_gemm1(x, w13):
    w1, w3 = torch.split(w13, int(list(w13.shape)[0]/2), dim=0)
    act_fn = torch.nn.SiLU()
    gate = act_fn(x @ w1.T)
    up = x @ w3.T
    return gate * up



def cutlass_fused_moe(hidden_states, w13, w2, topk_weights, topk_ids, n_experts_per_token, inplace, activation, num_experts):
    token_cnt, hidden_size = list(hidden_states.shape)
    intermediate_size = list(w2.shape)[-1]
    expert_cache = torch.empty((token_cnt, intermediate_size),
                           dtype=hidden_states.dtype,
                           device=hidden_states.device)

    idxs = topk_ids.argsort()
    counts = topk_ids.to(torch.long).bincount().cpu().numpy()
    tokens_per_expert = counts.cumsum()
    num_per_tok = n_experts_per_token
    token_idxs = idxs // num_per_tok
    experts_intermediate = []
    print("tokens_per_expert", tokens_per_expert)

    for expert_id, end_idx in enumerate(tokens_per_expert):
        print("expert id: ", expert_id)
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        if start_idx == end_idx:
            continue

        expert_w13 = w13[expert_id, :, :]
        exp_token_idxs = token_idxs[start_idx:end_idx]
        expert_tokens = hidden_states[exp_token_idxs]
        expert_out    = ref_gemm1(expert_tokens, expert_w13)
        # expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
        # expert_cache.scatter_reduce_(
        #     0,
        #     exp_token_idxs.view(-1, 1).repeat(1, x.shape[-1]),
        #     expert_out,
        #     reduce='sum'
        # )
        experts_intermediate.append(expert_out)

    group_A = torch.cat(experts_intermediate, dim=0).contiguous()
    output = torch.empty((list(group_A.shape)[0], hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
    offset = torch.tensor(tokens_per_expert, dtype=torch.int64, device='xpu')
    print("groupA [num_tokens, inter]:", group_A.shape)
    print("weight2 [expert, hidden, inter(ld)]", w2.shape)
    print("output [num_tokens, hidden]", output.shape)
    print("offset [expert]", offset.shape)
    # cutlass_grouped_gemm(Tensor input, Tensor weight, Tensor res, Tensor offset, int64_t hidden_size, int64_t intermediate_size,             int64_t num_of_expert) -> Tensor
    print("enter kernel")


    offset = torch.tensor([6, 12, 8, 4, 3, 6, 12, 8, 4, 3, 0, 0, 0, 0, 0, 0] ,dtype=torch.int64, device='xpu' )

    hidden_states = torch.ops._vllm_fp8_C.cutlass_grouped_gemm(group_A, w2, output, offset, hidden_size, intermediate_size, num_experts)
    print(hidden_states)
