# SPDX-License-Identifier: Apache-2.0
import torch

try:
    from . import _C  # noqa: F401
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
                                       device=input_A.device)

    def exclusive_prefix_sum(arr):
        prefix = [0]
        for i, x in enumerate(arr):
            prefix.append(prefix[-1] + x)
        return prefix

    if bias is not None:
        bias = bias.repeat_interleave(expert_token_count_, dim=0).float()

    expert_offset = torch.tensor(exclusive_prefix_sum(expert_token_count),
                                 dtype=torch.int64,
                                 device="xpu")
    torch.ops._xpu_C.cutlass_grouped_gemm(
        ptr_A=input_A,
        ptr_B=input_B,
        ptr_bias=bias,
        ptr_D=output,
        expert_first_token_offset=expert_offset,
        N=n,
        K=k,
        groups=num_experts)


def cutlass_grouped_gemm_xe2(input_A, input_B, scales, bias, output,
                             num_rows_per_expert, n, k, num_experts, is_B_int4,
                             is_B_mxfp4):
    expert_first_token_offset = torch.cat([
        torch.tensor([0],
                     dtype=num_rows_per_expert.dtype,
                     device=num_rows_per_expert.device),
        torch.cumsum(num_rows_per_expert, dim=0)
    ]).to(torch.int64)
    torch.ops._xpu_C.cutlass_grouped_gemm_xe2(
        ptr_A=input_A,
        ptr_B=input_B,
        ptr_scales=scales,
        ptr_bias=bias,
        ptr_D=output,
        expert_first_token_offset=expert_first_token_offset,
        N=n,
        K=k,
        num_experts=num_experts,
        is_B_int4=is_B_int4,
        is_B_mxfp4=is_B_mxfp4)


def ceilDiv(a, b):
    return (a + b - 1) // b


def compute_num_tokens_per_block(num_tokens, num_experts_per_node):
    for num_tokens_per_block in [32, 64, 128, 256, 512, 1024]:
        num_blocks_per_seq = ceilDiv(num_tokens, num_tokens_per_block)
        if num_blocks_per_seq * num_experts_per_node <= num_tokens_per_block:
            return num_tokens_per_block
    return 1024


def implement_zp(qweight):
    # change u4 to s4 to avoid zero point in gemm kernel
    # only support default zero point now
    assert qweight.dtype == torch.uint8, "Input tensor must be uint8"

    high_u4 = (qweight >> 4) & 0x0F
    low_u4 = qweight & 0x0F

    high_s8 = high_u4.to(torch.int8)
    low_s8 = low_u4.to(torch.int8)

    high_s8 = high_s8 - 8
    low_s8 = low_s8 - 8

    def pack_compact(a, b):

        def process_number(x):
            sign = (x < 0).to(torch.uint8)
            abs_low3 = (x.view(torch.uint8) & 0x7).to(torch.uint8)
            return (sign << 3) | abs_low3

        packed_a = process_number(a)
        packed_b = process_number(b)

        return (packed_a << 4) | packed_b

    result = pack_compact(high_s8, low_s8)

    return result


def xpu_fused_moe(hidden_states,
                  w13,
                  w13_scales,
                  w13_bias,
                  w2,
                  w2_scales,
                  w2_bias,
                  topk_weights,
                  topk_ids,
                  n_experts_per_token,
                  activation,
                  num_experts,
                  is_fp8=False,
                  is_int4=False,
                  is_mxfp4=False):
    '''
    hidden_states: [num_rows, hidden_size]
    w13: [num_experts, 2*inter_size, hidden_size]
    w13_scales: 
        None for bf16/fp16 
        or [num_experts] for fp8 
        or [num_experts, 2*inter_size, hidden_size // group_size] for 4bits
    w13_bias: [num_experts, 2*inter_size] or None
    w2: [num_experts, hidden_size, inter_size]
    w2_scales:
        None for bf16/fp16 
        or [num_experts] for fp8 
        or [num_experts, hidden_size, inter_size // group_size] for 4bits
    w2_bias: [num_experts, hidden_size] or None
    topk_weights: [num_rows, topk]
    topk_ids: [num_rows, topk]
    n_experts_per_token: int
    activation: str
    num_experts: int
    is_int4: bool
    is_mxfp4: bool
    '''

    output = torch.zeros_like(hidden_states)
    inter_size = list(w13.shape)[-2] // 2

    assert w13.is_contiguous() and w2.is_contiguous()

    # 4bits support [E, N, K]
    # other types [E, K, N]
    if not is_int4 and not is_mxfp4:
        if not hasattr(w13, 'xpu_fused_moe'):
            w13.data = w13.transpose(-1, -2).contiguous()
            w2.data = w2.transpose(-1, -2).contiguous()
            w13.xpu_fused_moe = True
            w13.inter_size = inter_size
        else:
            inter_size = w13.inter_size

    if is_int4:
        for i in range(num_experts):
            w13[i] = implement_zp(w13[i])
            w2[i] = implement_zp(w2[i])
        w13.contiguous()
        w2.contiguous()

    # TODO: will all integrated in Cpp func. Temporary expose before gemm fusion
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
    if topk_ids.dtype == torch.int32:
        topk_ids = topk_ids.to(torch.int64)
    torch.ops._xpu_C.fused_moe(output=output,
                               input=hidden_states,
                               token_selected_experts=topk_ids,
                               token_final_scales=topk_weights,
                               workspace=workspace,
                               hidden_size=hidden_size,
                               inter_size=inter_size,
                               num_experts_on_rank=num_experts_per_node)

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
    if not is_fp8 and not is_int4 and not is_mxfp4:
        expert_token_count = (expert_first_token_offset[1:] -
                              expert_first_token_offset[:-1]).to(torch.int64)
        if w13_bias is None:
            w13_bias = None
            w2_bias = None
        else:
            if w13_bias.shape == (num_experts, 2 * inter_size):
                w13_bias = w13_bias.repeat_interleave(expert_token_count,
                                                      dim=0).float()
            if w2_bias.shape == (num_experts, hidden_size):
                w2_bias = w2_bias.repeat_interleave(expert_token_count,
                                                    dim=0).float()
    gemm1_output = torch.empty((num_moe_inputs, 2 * inter_size),
                               dtype=hidden_states.dtype,
                               device=hidden_states.device)

    ########### gemm1 ##################
    input_B = w13

    if not is_fp8 and not is_int4 and not is_mxfp4:
        torch.ops._xpu_C.cutlass_grouped_gemm(
            ptr_A=gemm1_input,
            ptr_B=input_B,
            ptr_bias=w13_bias,
            ptr_D=gemm1_output,
            expert_first_token_offset=expert_first_token_offset,
            N=2 * inter_size,
            K=hidden_size,
            groups=num_experts_per_node)
    else:
        torch.ops._xpu_C.cutlass_grouped_gemm_xe2(
            ptr_A=gemm1_input,
            ptr_B=input_B,
            ptr_scales=w13_scales,
            ptr_bias=w13_bias,
            ptr_D=gemm1_output,
            expert_first_token_offset=expert_first_token_offset,
            N=2 * inter_size,
            K=hidden_size,
            num_experts=num_experts_per_node,
            is_B_int4=is_int4,
            is_B_mxfp4=is_mxfp4)

    # act
    act_output = torch.empty((num_moe_inputs, inter_size),
                             dtype=gemm1_output.dtype,
                             device=gemm1_output.device)
    if activation == "silu":
        torch.ops._C.silu_and_mul(act_output, gemm1_output)
    elif activation == "gelu":
        torch.ops._C.gelu_and_mul(act_output, gemm1_output)
    elif activation == "swigluoai":
        torch.ops._C.swigluoai_and_mul(act_output, gemm1_output, 1.702, 7.0)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}.")

    ########### gemm2 ##################
    input_A = act_output.contiguous()
    input_B = w2
    gemm2_output = torch.empty((num_moe_inputs, hidden_size),
                               dtype=hidden_states.dtype,
                               device=hidden_states.device)
    if not is_fp8 and not is_int4 and not is_mxfp4:
        torch.ops._xpu_C.cutlass_grouped_gemm(
            ptr_A=input_A,
            ptr_B=input_B,
            ptr_bias=w2_bias,
            ptr_D=gemm2_output,
            expert_first_token_offset=expert_first_token_offset,
            N=hidden_size,
            K=inter_size,
            groups=num_experts_per_node)
    else:
        torch.ops._xpu_C.cutlass_grouped_gemm_xe2(
            ptr_A=input_A,
            ptr_B=input_B,
            ptr_scales=w2_scales,
            ptr_bias=w2_bias,
            ptr_D=gemm2_output,
            expert_first_token_offset=expert_first_token_offset,
            N=hidden_size,
            K=inter_size,
            num_experts=num_experts_per_node,
            is_B_int4=is_int4,
            is_B_mxfp4=is_mxfp4)

    expert_cache = output

    iter_for_weight_apply = expert_first_token_offset[1:]
    for expert_id, end_idx in enumerate(iter_for_weight_apply):
        start_idx = 0 if expert_id == 0 else iter_for_weight_apply[expert_id -
                                                                   1]
        if start_idx == end_idx:
            continue

        exp_token_idxs = permuted_row_to_unpermuted_row[start_idx:end_idx]
        scores_token_ids = exp_token_idxs % num_rows
        scores_k_slot = exp_token_idxs // num_rows
        scores = topk_weights[scores_token_ids, scores_k_slot]
        expert_out = gemm2_output[start_idx:end_idx]
        expert_out.mul_(scores.view(-1, 1))
        expert_cache.scatter_reduce_(0,
                                     scores_token_ids.view(-1, 1).repeat(
                                         1, hidden_size),
                                     expert_out,
                                     reduce='sum')
    return output
