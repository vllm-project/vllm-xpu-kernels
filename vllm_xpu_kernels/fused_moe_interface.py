# SPDX-License-Identifier: Apache-2.0
import torch

try:
    from . import _xpu_C  # noqa: F401
    FUSEDMOE_UNAVAILABLE_REASON = None
    FUSEDMOE_AVAILABLE = True
except ImportError as e:
    FUSEDMOE_UNAVAILABLE_REASON = str(e)
    FUSEDMOE_AVAILABLE = False


def cutlass_grouped_gemm(input_A, input_B, bias, output, expert_token_count, n,
                         k, num_experts):
    expert_token_count_ = torch.tensor(expert_token_count,
                                       dtype=torch.int64,
                                       device="cpu")

    def exclusive_prefix_sum(arr):
        prefix = [0]
        for i, x in enumerate(arr):
            prefix.append(prefix[-1] + x)
        return prefix

    if bias is not None:
        bias = bias.repeat_interleave(expert_token_count_.to(bias.device),
                                      dim=0).float()

    expert_offset = torch.tensor(exclusive_prefix_sum(expert_token_count),
                                 dtype=torch.int64,
                                 device="xpu")
    torch.ops._xpu_C.cutlass_grouped_gemm(
        ptr_A=input_A,
        ptr_B=input_B,
        bias=bias,
        ptr_D=output,
        expert_first_token_offset=expert_offset,
        expert_token_count=expert_token_count_,
        N=n,
        K=k,
        groups=num_experts)


def ceilDiv(a, b):
    return (a + b - 1) // b


def compute_num_tokens_per_block(num_tokens, num_experts_per_node):
    for num_tokens_per_block in [32, 64, 128, 256, 512, 1024]:
        num_blocks_per_seq = ceilDiv(num_tokens, num_tokens_per_block)
        if num_blocks_per_seq * num_experts_per_node <= num_tokens_per_block:
            return num_tokens_per_block
    return 1024


def xpu_fused_moe(hidden_states, w13, w13_bias, w2, w2_bias, topk_weights,
                  topk_ids, n_experts_per_token, activation, num_experts):

    output = torch.zeros_like(hidden_states)

    # TODO: will all integrated in Cpp func. Temporary expose before gemm fusion
    num_rows, hidden_size = list(hidden_states.shape)
    inter_size = list(w2.shape)[-1]
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
    permuted_data_size = permuted_elems * 2
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

    workspace = torch.zeros(map_offset,
                            dtype=torch.uint8,
                            device=hidden_states.device)
    torch.ops._xpu_C.fused_moe(output=output,
                               input=hidden_states,
                               token_selected_experts=topk_ids,
                               token_final_scales=topk_weights,
                               fc1_expert_weights=w13,
                               fc2_expert_weights=w2,
                               workspace=workspace)

    expert_first_token_offset = workspace[
        ws_map["expert_first_token_offset"][1]:
        ws_map["expert_first_token_offset"][1] +
        expert_first_token_offset_size].view(torch.int64)
    permuted_row_to_unpermuted_row = workspace[
        ws_map["permuted_row_to_unpermuted_row"][1]:
        ws_map["permuted_row_to_unpermuted_row"][1] +
        permuted_row_to_unpermuted_row_size].view(torch.int32)
    gemm1_input = workspace[ws_map["overlapped_gemm1_gemm2_inputs"][1]:
                            ws_map["overlapped_gemm1_gemm2_inputs"][1] +
                            permuted_data_size].view(hidden_states.dtype).view(
                                num_moe_inputs, hidden_size)
    # permuted_token_final_scales = workspace[
    #     ws_map["permuted_token_final_scales"][1]:
    #     ws_map["permuted_token_final_scales"][1] +
    #     permuted_token_final_scales_size].view(torch.float)
    expert_token_count = (expert_first_token_offset[1:] -
                          expert_first_token_offset[:-1]).to(torch.int64)
    if w13_bias is None:
        w13_bias = torch.zeros((num_experts, 2 * inter_size),
                               dtype=torch.float,
                               device=hidden_states.device)
        w2_bias = torch.zeros((num_experts, hidden_size),
                              dtype=torch.float,
                              device=hidden_states.device)
    if w13_bias.shape == (num_experts, 2 * inter_size):
        w13_bias = w13_bias.repeat_interleave(expert_token_count,
                                              dim=0).float()
    if w2_bias.shape == (num_experts, hidden_size):
        w2_bias = w2_bias.repeat_interleave(expert_token_count, dim=0).float()
    expert_token_count = expert_token_count.cpu()

    gemm1_output = torch.empty((num_moe_inputs, 2 * inter_size),
                               dtype=hidden_states.dtype,
                               device=hidden_states.device)

    ########### gemm1 ##################
    input_B = w13.transpose(-1, -2).contiguous().transpose(-1, -2)

    torch.ops._xpu_C.cutlass_grouped_gemm(
        ptr_A=gemm1_input,
        ptr_B=input_B,
        bias=w13_bias,
        ptr_D=gemm1_output,
        expert_first_token_offset=expert_first_token_offset,
        expert_token_count=expert_token_count,
        N=2 * inter_size,
        K=hidden_size,
        groups=num_experts_per_node)

    # act
    gate, up_ = torch.split(gemm1_output, inter_size, dim=1)
    act = torch.nn.SiLU()
    act_output = act(gate) * up_

    ########### gemm2 ##################
    input_A = act_output.contiguous()
    input_B = w2.transpose(-1, -2).contiguous().transpose(-1, -2)
    gemm2_output = torch.empty((num_moe_inputs, hidden_size),
                               dtype=hidden_states.dtype,
                               device=hidden_states.device)
    torch.ops._xpu_C.cutlass_grouped_gemm(
        ptr_A=input_A,
        ptr_B=input_B,
        bias=w2_bias,
        ptr_D=gemm2_output,
        expert_first_token_offset=expert_first_token_offset,
        expert_token_count=expert_token_count,
        N=hidden_size,
        K=inter_size,
        groups=num_experts_per_node)

    topk_weights = topk_weights.view(-1, 1)
    expert_cache = output

    for expert_id, end_idx in enumerate(expert_first_token_offset):
        start_idx = 0 if expert_id == 0 else expert_first_token_offset[
            expert_id - 1]
        if start_idx == end_idx:
            continue

        exp_token_idxs = permuted_row_to_unpermuted_row[
            start_idx:end_idx] % num_rows
        expert_out = gemm2_output[start_idx:end_idx]
        expert_out.mul_(
            topk_weights[permuted_row_to_unpermuted_row[start_idx:end_idx] %
                         num_rows])
        expert_cache.scatter_reduce_(0,
                                     exp_token_idxs.view(-1, 1).repeat(
                                         1, hidden_size),
                                     expert_out,
                                     reduce='sum')
    return output
