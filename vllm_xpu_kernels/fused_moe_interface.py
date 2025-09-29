# SPDX-License-Identifier: Apache-2.0
import torch

try:
    from . import _xpu_C  # noqa: F401
    FUSEDMOE_UNAVAILABLE_REASON = None
    FUSEDMOE_AVAILABLE = True
except ImportError as e:
    FUSEDMOE_UNAVAILABLE_REASON = str(e)
    FUSEDMOE_AVAILABLE = False


def prepare_gemm_args(n, k, A, B, D, e):
    gemm_args = {}
    gemm_args["ptr_A"] = A
    gemm_args["ptr_B"] = B
    gemm_args["ptr_D"] = D
    gemm_args["groups"] = e  # FIXME: groups
    return gemm_args


def cutlass_grouped_gemm(input_A, input_B, output, tokens_per_expert, n, k, num_experts):
    gemm_args = prepare_gemm_args(n, k, input_A, input_B, output, num_experts)
    torch.ops._xpu_C.cutlass_grouped_gemm(tokens_per_expert=tokens_per_expert, N=n, K=k, **gemm_args)


def cutlass_fused_moe(hidden_states, w13, w2, topk_weights, topk_ids,
                      n_experts_per_token, activation, num_experts):

    token_cnt, hidden_size = list(hidden_states.shape)
    intermediate_size = list(w2.shape)[-1]
    total_input_size = token_cnt * n_experts_per_token
    if not hasattr(cutlass_fused_moe, "moe_buffer"):
        moe_buffer = {}
        moe_buffer["expert_cache"] = torch.empty((token_cnt * hidden_size),
                                                 dtype=hidden_states.dtype,
                                                 device=hidden_states.device)
        moe_buffer["gemm1_input"] = torch.empty(
            (total_input_size, hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device)
        moe_buffer["gemm1_output"] = torch.empty(
            (total_input_size, 2 * intermediate_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device)
        moe_buffer["gemm2_output"] = torch.empty(
            (total_input_size, hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device)

        cutlass_fused_moe.moe_buffer = moe_buffer

    expert_cache = cutlass_fused_moe.moe_buffer[
        "expert_cache"][:hidden_states.numel()].view_as(hidden_states).zero_()
    input_A = cutlass_fused_moe.moe_buffer["gemm1_input"][:total_input_size, :]
    gemm1_output = cutlass_fused_moe.moe_buffer[
        "gemm1_output"][:total_input_size, :]
    gemm2_output = cutlass_fused_moe.moe_buffer[
        "gemm2_output"][:total_input_size, :]


    # map token to experts
    idxs = topk_ids.argsort()
    counts = topk_ids.to(torch.int).bincount()
    tokens_per_expert = counts.cumsum()
    num_per_tok = n_experts_per_token
    token_idxs = idxs // num_per_tok

    ########### gemm1 ##################
    input_B = w13
    assert (list(input_A.shape)[0] == total_input_size)
    gemm_args = prepare_gemm_args(2 * intermediate_size, hidden_size,
                                  input_A, input_B, gemm1_output,
                                  num_experts)
    torch.ops._xpu_C.cutlass_grouped_gemm(tokens_per_expert=tokens_per_expert,
                                          N=2 * intermediate_size,
                                          K=hidden_size,
                                          **gemm_args)
    # act
    gate, up_ = torch.split(gemm1_output, intermediate_size, dim=1)
    act = torch.nn.SiLU()
    act_output = act(gate) * up_

    ########### gemm2 ##################
    input_A = act_output.contiguous()
    input_B = w2
    gemm_args = prepare_gemm_args(hidden_size, intermediate_size,
                                  input_A, input_B, gemm2_output,
                                  num_experts)
    torch.ops._xpu_C.cutlass_grouped_gemm(tokens_per_expert=tokens_per_expert,
                                          N=hidden_size,
                                          K=intermediate_size,
                                          **gemm_args)

    # apply scores
    for expert_id, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        if start_idx == end_idx:
            continue

        exp_token_idxs = token_idxs[start_idx:end_idx]
        expert_out = gemm2_output[start_idx:end_idx]
        expert_out.mul_(topk_weights[idxs[start_idx:end_idx]])
        expert_cache.scatter_reduce_(0,
                                     exp_token_idxs.view(-1, 1).repeat(
                                         1, hidden_size),
                                     expert_out,
                                     reduce='sum')
    return expert_cache
