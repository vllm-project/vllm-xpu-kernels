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


def cutlass_grouped_gemm(input_A, input_B, output, offset, n, k, num_experts):
    alpha = torch.ones(num_experts, dtype=torch.float32, device=input_A.device)
    beta = torch.zeros(num_experts, dtype=torch.float32, device=input_A.device)
    gemm_args = prepare_gemm_args(n, k, offset, input_A, input_B, output,
                                  alpha, beta, num_experts)
    offset_ = torch.tensor(offset, dtype=torch.int64, device="cpu")
    # print("input_A: ", input_A.shape, hex(input_A.data_ptr()))
    print("offset: ", offset_)
    def exclusive_prefix_sum(arr):
        prefix = [0]
        for i, x in enumerate(arr):
            # if x == 0:
            #     continue
            prefix.append(prefix[-1] + x)
        return prefix
    expert_offset = torch.tensor(exclusive_prefix_sum(offset), dtype=torch.int64, device="xpu")
    # print("offset cumsum:", expert_offset)
    # debug = True
    # if debug:
    #     for start in exclusive_prefix_sum(offset):
    #         if start < input_A.shape[0]:
    #             # print("offset: ", start, " 0-10:", input_A.view(-1)[start:start+10])
    #             print("offset: ", start, " addr:", hex(input_A.view(-1)[start*input_A.shape[1]].data_ptr()))
    # print("log: ", expert_offset, expert_offset.numel())
    # assert(expert_offset.numel() == num_experts + 1)
    torch.ops._xpu_C.cutlass_grouped_gemm(ptr_A=input_A, ptr_B=input_B, ptr_D=output, expert_first_token_offset=expert_offset, offset=offset_, N=n, K=k, **gemm_args)


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
    workspace = torch.zeros(6247760, dtype=torch.uint8, device=hidden_states.device)
    torch.ops._xpu_C.fused_moe(output=output,
                               input=hidden_states,
                               token_selected_experts=topk_ids,
                               token_final_scales=topk_weights,
                               fc1_expert_weights=w13,
                               fc2_expert_weights=w2,
                               workspace=workspace)

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
    ws_map["overlapped_gemm1_gemm2_inputs"] = (permuted_data_size, map_offset)
    map_offset += permuted_data_size

    expert_first_token_offset = workspace[ws_map["expert_first_token_offset"][1]:]
    gemm1_input = workspace[ws_map["overlapped_gemm1_gemm2_inputs"][1]:]
    gemm1_output = torch.empty((num_moe_inputs, 2 * inter_size),
                                dtype=hidden_states.dtype,
                                device=hidden_states.device)


    return output


