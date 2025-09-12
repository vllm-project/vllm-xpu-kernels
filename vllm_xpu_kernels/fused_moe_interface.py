import torch
from torch.nn.modules.utils import _pair
from torch import nn, Tensor
from typing import List
import numpy
import vllm_xpu_kernels._xpu_C

def prepare_gemm_args(n, k, offset, A, B, D, alpha, beta):
    gemm_args = {}
    device = A.device
    # problem_sizes = []
    ptr_A = []
    ptr_B = []
    ptr_D = []
    ptr_alpha = []
    ptr_beta = []
    total_elements_A = 0
    total_elements_B = 0
    total_elements_D = 0

    def process_data_ptr(tensor, offset, addr_list, dim):
        mul = 2
        if tensor.dtype == torch.float32:
            mul = 4
        # print("process data_ptr:", tensor.shape)
        # print(tensor.data_ptr())
        # print(offset*mul)
        # addr = tensor.data_ptr() + offset*mul
        if dim == 1:
            addr = tensor[offset].data_ptr()
        elif dim == 2:
            addr = tensor[offset, :].data_ptr()
        elif dim == 3:
            addr = tensor[offset, :, :].data_ptr()
        for i in range(8):                  # 64bit -> 8 bytes
            byte_val = (addr >> (i * 8)) & 0xFF
            addr_list.append(byte_val)

    groups = 0
    for m in offset:
        if m != 0:
            # problem_sizes.extend([m, n, k])
            process_data_ptr(A, total_elements_A, ptr_A, 2)
            process_data_ptr(B, total_elements_B, ptr_B, 3)
            process_data_ptr(D, total_elements_D, ptr_D, 2)
            process_data_ptr(alpha, groups, ptr_alpha, 1)
            process_data_ptr(beta, groups, ptr_beta, 1)
            total_elements_A += m;
            total_elements_D += m;
            groups += 1
        total_elements_B += 1;

    # problem_sizes = torch.tensor(problem_sizes, dtype=torch.int64, device='cpu').contiguous()
    ptr_A = torch.tensor(ptr_A, dtype=torch.uint8, device=device).contiguous()
    ptr_B = torch.tensor(ptr_B, dtype=torch.uint8, device=device).contiguous()
    ptr_D = torch.tensor(ptr_D, dtype=torch.uint8, device=device).contiguous()
    ptr_alpha = torch.tensor(ptr_alpha, dtype=torch.uint8, device=device).contiguous()
    ptr_beta = torch.tensor(ptr_beta, dtype=torch.uint8, device=device).contiguous()

    # gemm_args["problem_sizes"] = problem_sizes
    gemm_args["ptr_A"] = ptr_A
    gemm_args["ptr_B"] = ptr_B
    gemm_args["ptr_D"] = ptr_D
    gemm_args["ptr_alpha"] = ptr_alpha
    gemm_args["ptr_beta"] = ptr_beta
    gemm_args["groups"] = groups
    return gemm_args


def cutlass_grouped_gemm(input_A, input_B, output, offset, n, k, num_experts):
    device = "xpu"
    alpha = torch.ones(num_experts, dtype=torch.float32, device=input_A.device)
    beta = torch.zeros(num_experts, dtype=torch.float32, device=input_A.device)
    gemm_args = prepare_gemm_args(n, k, offset, input_A, input_B, output, alpha, beta)
    torch.ops._xpu_C.cutlass_grouped_gemm(offset=offset, N=n, K=k, **gemm_args)

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
    alpha = torch.ones(num_experts, dtype=torch.float32, device=input_A.device)
    beta = torch.zeros(num_experts, dtype=torch.float32, device=input_A.device)
    gemm_args = prepare_gemm_args(2*intermediate_size, hidden_size, offset, input_A, input_B, gemm1_output, alpha, beta)
    offset_t = torch.tensor(offset, dtype=torch.int64, device='cpu')
    torch.ops._xpu_C.cutlass_grouped_gemm(offset=offset_t, N=2*intermediate_size, K=hidden_size, **gemm_args)

    # act
    gate, up = torch.split(gemm1_output, intermediate_size, dim=1)
    act = torch.nn.SiLU()
    act_output = act(gate) * up


    # gemm 2
    input_A = act_output.to(torch.bfloat16).contiguous()
    output = torch.empty((list(input_A.shape)[0], hidden_size), dtype=torch.float32, device=hidden_states.device)
    input_B = w2.transpose(-1, -2).contiguous().transpose(-1, -2)
    gemm_args = prepare_gemm_args(hidden_size, intermediate_size, offset, input_A, input_B, output, alpha, beta)
    torch.ops._xpu_C.cutlass_grouped_gemm(offset=offset_t, N=hidden_size, K=intermediate_size, **gemm_args)

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
