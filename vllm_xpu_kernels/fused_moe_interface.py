# SPDX-License-Identifier: Apache-2.0
import torch

try:
    from . import _xpu_C  # noqa: F401
    FUSEDMOE_UNAVAILABLE_REASON = None
    FUSEDMOE_AVAILABLE = True
except ImportError as e:
    FUSEDMOE_UNAVAILABLE_REASON = str(e)
    FUSEDMOE_AVAILABLE = False


def prepare_gemm_args(n, k, offset, A, B, D, alpha, beta, e):

    if not hasattr(prepare_gemm_args, "gemm_args"):
        gemm_args = {}
        device = A.device
        ptr_A = torch.empty(e * 8, dtype=torch.uint8,
                            device=device).contiguous()
        ptr_B = torch.empty(e * 8, dtype=torch.uint8,
                            device=device).contiguous()
        ptr_D = torch.empty(e * 8, dtype=torch.uint8,
                            device=device).contiguous()
        ptr_alpha = torch.empty(e * 8, dtype=torch.uint8,
                                device=device).contiguous()
        ptr_beta = torch.empty(e * 8, dtype=torch.uint8,
                               device=device).contiguous()
        # gemm_args["ptr_A"] = ptr_A
        # gemm_args["ptr_B"] = ptr_B
        # gemm_args["ptr_D"] = ptr_D
        gemm_args["ptr_alpha"] = ptr_alpha
        gemm_args["ptr_beta"] = ptr_beta
        prepare_gemm_args.gemm_args = gemm_args

    # ptr_A = prepare_gemm_args.gemm_args["ptr_A"]
    # ptr_B = prepare_gemm_args.gemm_args["ptr_B"]
    # ptr_D = prepare_gemm_args.gemm_args["ptr_D"]
    ptr_alpha = prepare_gemm_args.gemm_args["ptr_alpha"]
    ptr_beta = prepare_gemm_args.gemm_args["ptr_beta"]
    total_elements_A = 0
    total_elements_D = 0

    def process_data_ptr(tensor, offset, addr_tensor, dim, group):
        if dim == 1:
            addr = tensor[offset].data_ptr()
        elif dim == 2:
            addr = tensor[offset, :].data_ptr()
        elif dim == 3:
            addr = tensor[offset, :, :].data_ptr()
        for i in range(8):  # 64bit -> 8 bytes
            byte_val = (addr >> (i * 8)) & 0xFF
            addr_tensor[8 * group + i] = byte_val

    groups = 0
    for expert_i, m in enumerate(offset):
        if m != 0:
            # problem_sizes.extend([m, n, k])
            # process_data_ptr(A, total_elements_A, ptr_A, 2, groups)
            # process_data_ptr(B, expert_i, ptr_B, 3, groups)
            # process_data_ptr(D, total_elements_D, ptr_D, 2, groups)
            process_data_ptr(alpha, groups, ptr_alpha, 1, groups)
            process_data_ptr(beta, groups, ptr_beta, 1, groups)
            total_elements_A += m
            total_elements_D += m
            groups += 1

    prepare_gemm_args.gemm_args["groups"] = e  # FIXME: groups
    return prepare_gemm_args.gemm_args


def cutlass_grouped_gemm(input_A, input_B, output, expert_token_count, n, k, num_experts):
    expert_token_count_ = torch.tensor(expert_token_count, dtype=torch.int64, device="cpu")
    def exclusive_prefix_sum(arr):
        prefix = [0]
        for i, x in enumerate(arr):
            prefix.append(prefix[-1] + x)
        return prefix
    expert_offset = torch.tensor(exclusive_prefix_sum(expert_token_count), dtype=torch.int64, device="xpu")
    torch.ops._xpu_C.cutlass_grouped_gemm(ptr_A=input_A, ptr_B=input_B, ptr_D=output, expert_first_token_offset=expert_offset, expert_token_count=expert_token_count_, N=n, K=k, groups=num_experts)


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
        moe_buffer["alpha"] = torch.ones(num_experts,
                                         dtype=torch.float32,
                                         device=hidden_states.device)
        moe_buffer["beta"] = torch.zeros(num_experts,
                                         dtype=torch.float32,
                                         device=hidden_states.device)

        cutlass_fused_moe.moe_buffer = moe_buffer

    expert_cache = cutlass_fused_moe.moe_buffer[
        "expert_cache"][:hidden_states.numel()].view_as(hidden_states).zero_()
    input_A = cutlass_fused_moe.moe_buffer["gemm1_input"][:total_input_size, :]
    gemm1_output = cutlass_fused_moe.moe_buffer[
        "gemm1_output"][:total_input_size, :]
    gemm2_output = cutlass_fused_moe.moe_buffer[
        "gemm2_output"][:total_input_size, :]
    alpha = cutlass_fused_moe.moe_buffer["alpha"]
    beta = cutlass_fused_moe.moe_buffer["beta"]

    # map token to experts
    idxs = topk_ids.argsort()
    counts = topk_ids.to(torch.long).bincount().cpu().numpy()
    tokens_per_expert = counts.cumsum()
    num_per_tok = n_experts_per_token
    token_idxs = idxs // num_per_tok
    offset = []
    for expert_id, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        offset.append(end_idx - start_idx)
        if start_idx == end_idx:
            continue
        exp_token_idxs = token_idxs[start_idx:end_idx]
        # expert_tokens = hidden_states[exp_token_idxs]
        # grouped_input_A.append(expert_tokens)
        input_A[start_idx:end_idx, :].copy_(hidden_states[exp_token_idxs])

    while len(offset) < num_experts:
        offset.append(0)

    ########### gemm1 ##################
    input_B = w13.transpose(-1, -2).contiguous().transpose(-1, -2)
    assert (list(input_A.shape)[0] == total_input_size)
    gemm_args = prepare_gemm_args(2 * intermediate_size, hidden_size, offset,
                                  input_A, input_B, gemm1_output, alpha, beta,
                                  num_experts)
    offset_t = torch.tensor(offset, dtype=torch.int64, device='cpu')
    torch.ops._xpu_C.cutlass_grouped_gemm(offset=offset_t,
                                          N=2 * intermediate_size,
                                          K=hidden_size,
                                          **gemm_args)
    # act
    gate, up_ = torch.split(gemm1_output, intermediate_size, dim=1)
    act = torch.nn.SiLU()
    act_output = act(gate) * up_

    ########### gemm2 ##################
    input_A = act_output.contiguous()
    input_B = w2.transpose(-1, -2).contiguous().transpose(-1, -2)
    gemm_args = prepare_gemm_args(hidden_size, intermediate_size, offset,
                                  input_A, input_B, gemm2_output, alpha, beta,
                                  num_experts)
    torch.ops._xpu_C.cutlass_grouped_gemm(offset=offset_t,
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

def ceilDiv(a, b):
    return (a + b - 1) // b

def compute_num_tokens_per_block(num_tokens, num_experts_per_node):
    for num_tokens_per_block in [32, 64, 128, 256, 512, 1024]:
        num_blocks_per_seq = ceilDiv(num_tokens, num_tokens_per_block)
        if num_blocks_per_seq * num_experts_per_node <= num_tokens_per_block:
            return num_tokens_per_block
    return 1024

def xpu_fused_moe(hidden_states, w13, w2, topk_weights, topk_ids,
                      n_experts_per_token, activation, num_experts):

    output = torch.zeros_like(hidden_states)

    # TODO: will all integrated in Cpp func. Temporary expose before gemm fusion
    num_rows, hidden_size = list(hidden_states.shape)
    inter_size = list(w2.shape)[-1]
    num_experts_per_node = num_experts
    experts_per_token = n_experts_per_token
    num_moe_inputs = n_experts_per_token * num_rows
    permuted_elems = num_moe_inputs * hidden_size
    interbuf_elems = num_moe_inputs * inter_size
    permuted_row_to_unpermuted_row_size = num_moe_inputs * 4
    permuted_token_selected_experts_size = num_moe_inputs * 4
    src_to_dest_map_size = experts_per_token * num_rows * 4
    expert_first_token_offset_size = (num_experts_per_node + 1) * 8
    num_tokens_per_block = compute_num_tokens_per_block(num_rows, num_experts_per_node)
    num_blocks_per_seq = ceilDiv(num_rows, num_tokens_per_block)
    blocked_expert_counts_size = num_experts_per_node * num_blocks_per_seq * 4
    blocked_expert_counts_cumsum_size = blocked_expert_counts_size
    blocked_row_to_unpermuted_row_size = num_experts_per_node * num_rows * 4
    permuted_data_size = permuted_elems * 2
    permuted_token_final_scales_size = num_moe_inputs * 4

    ws_map = {}
    map_offset = 0
    ws_map["permuted_token_selected_experts"] = (permuted_token_selected_experts_size, map_offset)
    map_offset += permuted_token_selected_experts_size
    ws_map["permuted_row_to_unpermuted_row"] = (permuted_row_to_unpermuted_row_size, map_offset)
    map_offset += permuted_row_to_unpermuted_row_size
    ws_map["src_to_dest_map"] = (src_to_dest_map_size, map_offset)
    map_offset += src_to_dest_map_size
    ws_map["expert_first_token_offset"] = (expert_first_token_offset_size, map_offset)
    map_offset += expert_first_token_offset_size
    ws_map["blocked_expert_counts"] = (blocked_expert_counts_size, map_offset)
    map_offset += blocked_expert_counts_size
    ws_map["blocked_expert_counts_cumsum"] = (blocked_expert_counts_cumsum_size, map_offset)
    map_offset += blocked_expert_counts_cumsum_size
    ws_map["blocked_row_to_unpermuted_row"] = (blocked_row_to_unpermuted_row_size, map_offset)
    map_offset += blocked_row_to_unpermuted_row_size
    if map_offset % 256 != 0:
       map_offset += 256 - (map_offset % 256)


    ws_map["overlapped_gemm1_gemm2_inputs"] = (permuted_data_size, map_offset)
    map_offset += permuted_data_size
    ws_map["permuted_token_final_scales"] = (permuted_token_final_scales_size, map_offset)
    map_offset += permuted_token_final_scales_size

    workspace = torch.zeros(map_offset, dtype=torch.uint8, device=hidden_states.device)
    torch.ops._xpu_C.fused_moe(output=output,
                               input=hidden_states,
                               token_selected_experts=topk_ids,
                               token_final_scales=topk_weights,
                               fc1_expert_weights=w13,
                               fc2_expert_weights=w2,
                               workspace=workspace)


    expert_first_token_offset = workspace[ws_map["expert_first_token_offset"][1]: ws_map["expert_first_token_offset"][1] + expert_first_token_offset_size].view(torch.int64)
    permuted_row_to_unpermuted_row = workspace[ws_map["permuted_row_to_unpermuted_row"][1]: ws_map["permuted_row_to_unpermuted_row"][1] + permuted_row_to_unpermuted_row_size].view(torch.int32)
    gemm1_input = workspace[ws_map["overlapped_gemm1_gemm2_inputs"][1]:ws_map["overlapped_gemm1_gemm2_inputs"][1] + permuted_data_size].view(hidden_states.dtype).view(num_moe_inputs, hidden_size)
    permuted_token_final_scales = workspace[ws_map["permuted_token_final_scales"][1]:ws_map["permuted_token_final_scales"][1] + permuted_token_final_scales_size].view(torch.float)
    gemm1_output = torch.empty((num_moe_inputs, 2 * inter_size),
                                dtype=hidden_states.dtype,
                                device=hidden_states.device)

    ########### gemm1 ##################
    input_B = w13.transpose(-1, -2).contiguous().transpose(-1, -2)
    expert_token_count = (expert_first_token_offset[1:] - expert_first_token_offset[:-1]).to(torch.int64).cpu()

    torch.ops._xpu_C.cutlass_grouped_gemm(ptr_A=gemm1_input, ptr_B=input_B,
                                          ptr_D=gemm1_output,
                                          expert_first_token_offset=expert_first_token_offset,
                                          expert_token_count=expert_token_count,
                                          N=2*inter_size, K=hidden_size,
                                          groups=num_experts_per_node)

    # act
    gate, up_ = torch.split(gemm1_output, inter_size, dim=1)
    act = torch.nn.SiLU()
    act_output = act(gate) * up_

    # )))))))))))))))))))))))))))))))))))))))))))))
    # ref gg

    # n = 2*inter_size
    # k = hidden_size
    # num_experts = num_experts_per_node
    # token_per_group = list(expert_token_count)
    # assert(len(token_per_group) == num_experts)
    # # input
    # # import ipdb
    # # ipdb.set_trace()
    # # input_A = gemm1_input.clone()
    # input_A = gemm1_input
    # # print("input_A shape", input_A.shape)
    # # input_A = torch.randn((sum(token_per_group), k),
    # #                       dtype=hidden_states.dtype,
    # #                       device=hidden_states.device).contiguous()
    # ref_A = input_A
    # # weight
    # # input_B = torch.randn((num_experts, n, k), dtype=hidden_states.dtype, device=hidden_states.device)
    # # input_B = input_B.transpose(-1, -2).contiguous().transpose(-1, -2)
    # # output offset
    # output = torch.empty((sum(token_per_group), n), dtype=hidden_states.dtype, device=hidden_states.device)


    # ref = []
    # pre_token_sum = 0
    # for i in range(num_experts_per_node):
    #     cur_token_num = token_per_group[i]
    #     if cur_token_num == 0:
    #         continue
    #     input = ref_A[pre_token_sum:pre_token_sum + cur_token_num, :]
    #     weight = input_B[i, :, :]
    #     expert_output = input @ weight.T
    #     ref.append(expert_output)
    #     pre_token_sum += cur_token_num
    # ref = torch.cat(ref, dim=0)
    # # cutlass_grouped_gemm(input_A, input_B, output, token_per_group, n, k, num_experts_per_node)
    # expert_token_count_ = torch.tensor(token_per_group, dtype=torch.int64, device="cpu")
    # def exclusive_prefix_sum(arr):
    #     prefix = [0]
    #     for i, x in enumerate(arr):
    #         prefix.append(prefix[-1] + x)
    #     return prefix
    # expert_offset = torch.tensor(exclusive_prefix_sum(token_per_group), dtype=torch.int64, device="xpu")
    # torch.ops._xpu_C.cutlass_grouped_gemm(ptr_A=input_A, ptr_B=input_B, ptr_D=output, expert_first_token_offset=expert_offset, expert_token_count=expert_token_count_, N=n, K=k, groups=num_experts)




    print("ker gemm1 inputA: ", gemm1_input, gemm1_input.shape)
    # for off in expert_first_token_offset[:-1]:
    #     print("ker gemm1 inputA at ", off, ": ", gemm1_input[off,:])


    print("ker expert_first_token_offset: ", expert_first_token_offset)
    print("ker expert_token_count: ", expert_token_count)
    print("ker gemm1 output: ", gemm1_output, gemm1_output.shape)
    # print("2nd ref ouput: ", ref, ref.shape)
    # print("2nd ker ouput: ", output, output.shape)
    # try:
    #     torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
    #     print("a and b close enough")
    # except AssertionError as e:
    #     print("a and b diffs")
    #     print(e)
    # return
    ########### gemm2 ##################
    input_A = act_output.contiguous()
    input_B = w2.transpose(-1, -2).contiguous().transpose(-1, -2)
    gemm2_output = torch.empty((num_moe_inputs, hidden_size),
                                dtype=hidden_states.dtype,
                                device=hidden_states.device)
    torch.ops._xpu_C.cutlass_grouped_gemm(ptr_A=input_A, ptr_B=input_B,
                                          ptr_D=gemm2_output,
                                          expert_first_token_offset=expert_first_token_offset,
                                          expert_token_count=expert_token_count,
                                          N=hidden_size, K=inter_size,
                                          groups=num_experts_per_node)

    print("ker gemm2 output:", gemm2_output, gemm2_output.shape)

    # apply expert weights
    # for permuted_row, expanded_token in enumerate(permuted_row_to_unpermuted_row):
    #     origin_idx = expanded_token.item() % num_rows
    #     output[origin_idx, :] += gemm2_output[permuted_row, :] * permuted_token_final_scales[permuted_row]


    # map token to experts
    # topk_ids = topk_ids.view(-1)
    topk_weights = topk_weights.view(-1,1)
    # idxs = topk_ids.argsort()
    # counts = topk_ids.to(torch.long).bincount().cpu().numpy()
    # tokens_per_expert = counts.cumsum()
    # num_per_tok = n_experts_per_token
    # token_idxs = idxs // num_per_tok
    expert_cache = output

    # print("ref idxs:", idxs)
    # print("ref token_idxs:", token_idxs)
    # print("permuted_row_to_unpermuted_row:", permuted_row_to_unpermuted_row, permuted_row_to_unpermuted_row.shape)
    # print("permuted_token_final_scales:", permuted_token_final_scales, permuted_token_final_scales.shape)
    # apply scores
    for expert_id, end_idx in enumerate(expert_first_token_offset):
        start_idx = 0 if expert_id == 0 else expert_first_token_offset[expert_id - 1]
        if start_idx == end_idx:
            continue

        exp_token_idxs = permuted_row_to_unpermuted_row[start_idx:end_idx] % num_rows
        expert_out = gemm2_output[start_idx:end_idx]
        expert_out.mul_(topk_weights[permuted_row_to_unpermuted_row[start_idx:end_idx] % num_rows])
        expert_cache.scatter_reduce_(0,
                                     exp_token_idxs.view(-1, 1).repeat(
                                         1, hidden_size),
                                     expert_out,
                                     reduce='sum')
    return output


