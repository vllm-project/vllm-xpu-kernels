# SPDX-License-Identifier: Apache-2.0
import torch

from vllm_xpu_kernels.fused_moe_interface import (  # noqa: F401
    FUSEDMOE_AVAILABLE, FUSEDMOE_UNAVAILABLE_REASON, ceilDiv,
    compute_num_tokens_per_block, cutlass_grouped_gemm,
    cutlass_grouped_gemm_xe2, implement_zp)


# vllm_xpu_kernel,main,
#   https://github.com/vllm-project/vllm-xpu-kernels/tree/377e7eb
def xpu_fused_moe_CalKernelTime(hidden_states,
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
                                ep_rank=0,
                                ep_size=1,
                                expert_map=None,
                                output=None,
                                is_fp8=False,
                                is_int4=False,
                                is_mxfp4=False,
                                start_event_remap=None,
                                end_event_remap=None,
                                start_event_gemm1=None,
                                end_event_gemm1=None,
                                start_event_gemm2=None,
                                end_event_gemm2=None,
                                start_event_gather=None,
                                end_event_gather=None):
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
    if output is None:
        output = torch.empty_like(hidden_states)
    else:
        assert output.shape == hidden_states.shape, \
            "output shape must be the same as hidden_states shape"

    if hasattr(w13, 'xpu_fused_moe'):
        gemm1_n = w13.shape[2]
        gemm2_n = w2.shape[2]
    else:
        gemm1_n = w13.shape[1]
        gemm2_n = w2.shape[1]

    # 4bits support [E, N, K]
    # other types [E, K, N]
    if not is_int4 and not is_mxfp4:
        inter_size = list(w13.shape)[-1] // 2
    else:
        inter_size = list(w13.shape)[-2] // 2

    assert w13.is_contiguous() and w2.is_contiguous()

    # FIXME: move this to vllm
    if is_int4 and not hasattr(w13, 'xpu_fused_moe'):
        w13_tmp = torch.empty_like(w13)
        w2_tmp = torch.empty_like(w2)
        for i in range(num_experts):
            w13_tmp[i] = implement_zp(w13[i])
            w2_tmp[i] = implement_zp(w2[i])
        w13_tmp = w13_tmp.contiguous()
        w2_tmp = w2_tmp.contiguous()
        w13.data = w13_tmp
        w2.data = w2_tmp
        w13.xpu_fused_moe = True

    # TODO: will all integrated in Cpp func. Temporary expose before gemm fusion
    num_rows, hidden_size = list(hidden_states.shape)
    num_moe_inputs = n_experts_per_token * num_rows

    # Use actual GEMM N dimensions for correct FLOPS calculation
    gemm1_n = 2 * inter_size  # gemm1: N = 2 * inter_size
    gemm2_n = hidden_size      # gemm2: N = hidden_size
    
    if topk_ids.dtype == torch.int32:
        topk_ids = topk_ids.to(torch.int64)
    gemm1_output = torch.empty((num_moe_inputs, 2 * inter_size),
                               dtype=hidden_states.dtype,
                               device=hidden_states.device)

    if not is_fp8 and not is_int4 and not is_mxfp4:
        gemm1_scales = None
        gemm2_scales = None
    else:
        gemm1_scales = w13_scales
        gemm2_scales = w2_scales

    if expert_map is None and ep_size > 1:
        expert_map = torch.empty((num_experts * ep_size),
                                 dtype=torch.int32,
                                 device=hidden_states.device)
        torch.ops._moe_C.init_expert_map(expert_map, num_experts, ep_rank,
                                         ep_size)

    if expert_map is not None:
        total_experts_num = expert_map.shape[0]
    else:
        total_experts_num = num_experts * ep_size
    local_experts_num = num_experts

    remapped_hidden_states = torch.empty(
        (num_rows * n_experts_per_token, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device)
    rows_per_expert = torch.zeros((num_experts),
                                  dtype=torch.int32,
                                  device=hidden_states.device)
    unpermuted_row_to_permuted_row = torch.empty(
        (num_rows, n_experts_per_token),
        dtype=torch.int32,
        device=hidden_states.device)

    ########### remap ##################
    if start_event_remap is not None:
        start_event_remap.record()
    torch.ops._moe_C.remap_hidden_states(
        hidden_states=hidden_states,
        hidden_states_scales=None,
        remapped_hidden_states=remapped_hidden_states,
        remapped_hidden_states_scales=None,
        expert_map=expert_map,
        rows_per_expert=rows_per_expert,
        unpermuted_row_to_permuted_row=unpermuted_row_to_permuted_row,
        topk_ids=topk_ids,
        total_experts_num=total_experts_num,
        local_experts_num=local_experts_num)
    if end_event_remap is not None:
        end_event_remap.record()

    ########### gemm1 ##################
    input_B = w13

    if start_event_gemm1 is not None:
        start_event_gemm1.record()
    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=remapped_hidden_states,
        ptr_B=input_B,
        ptr_scales=gemm1_scales,
        ptr_bias=w13_bias,
        ptr_D=gemm1_output,
        rows_per_expert=rows_per_expert,
        N=2 * inter_size,
        K=hidden_size,
        num_experts=num_experts,
        is_B_int4=is_int4,
        is_B_mxfp4=is_mxfp4)
    if end_event_gemm1 is not None:
        end_event_gemm1.record()
    active_experts1 = (rows_per_expert > 0).sum().item()
    gemm1_m = remapped_hidden_states.shape[0]
    gemm1_k = remapped_hidden_states.shape[1]

    # act
    act_output = torch.empty((num_moe_inputs, inter_size),
                             dtype=gemm1_output.dtype,
                             device=gemm1_output.device)
    if activation == "silu":
        torch.ops._C.silu_and_mul(act_output, gemm1_output)
    elif activation == "gelu":
        torch.ops._C.gelu_and_mul(act_output, gemm1_output)
    elif activation == "swigluoai" or ("SWIGLUOAI" in str(activation)):
        torch.ops._C.swigluoai_and_mul(act_output, gemm1_output, 1.702, 7.0)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}.")

    ########### gemm2 ##################
    input_A = act_output.contiguous()
    input_B = w2
    gemm2_output = torch.empty((num_moe_inputs, hidden_size),
                               dtype=hidden_states.dtype,
                               device=hidden_states.device)

    if start_event_gemm2 is not None:
        start_event_gemm2.record()
    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=input_A,
        ptr_B=input_B,
        ptr_scales=gemm2_scales,
        ptr_bias=w2_bias,
        ptr_D=gemm2_output,
        rows_per_expert=rows_per_expert,
        N=hidden_size,
        K=inter_size,
        num_experts=num_experts,
        is_B_int4=is_int4,
        is_B_mxfp4=is_mxfp4)

    if end_event_gemm2 is not None:
        end_event_gemm2.record()

    if start_event_gather is not None:
        start_event_gather.record()
    torch.ops._moe_C.moe_gather(output, gemm2_output, topk_weights,
                                unpermuted_row_to_permuted_row,
                                num_experts)
    if end_event_gather is not None:
        end_event_gather.record()

    active_experts2 = (rows_per_expert > 0).sum().item()
    gemm2_m = input_A.shape[0]
    gemm2_k = input_A.shape[1]
    return ((gemm1_m, gemm1_n, gemm1_k, active_experts1),
            (gemm2_m, gemm2_n, gemm2_k, active_experts2))
