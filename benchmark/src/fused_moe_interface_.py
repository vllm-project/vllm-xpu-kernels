# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch

from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe, quant_act_xpu


class XpuFusedMoe_CalKernelTime(XpuFusedMoe):
    def __init__(
        self,
        w13,
        w13_scales,
        w13_bias,
        w2,
        w2_scales,
        w2_bias,
        n_experts_per_token,
        activation,
        num_experts,
        ep_rank=0,
        ep_size=1,
        expert_map=None,
        gemm1_clamp_limit: Optional[float]=None,
    ):
        super().__init__(
            w13,
            w13_scales,
            w13_bias,
            w2,
            w2_scales,
            w2_bias,
            n_experts_per_token,
            activation,
            num_experts,
            ep_rank,
            ep_size,
            expert_map,
            gemm1_clamp_limit)

    def apply(
        self,
        output,
        hidden_states,
        topk_weights,
        topk_ids,
        expert_map=None,
        a1q_scale=None,
        start_event_remap=None,
        end_event_remap=None,
        start_event_gemm1=None,
        end_event_gemm1=None,
        start_event_gemm2=None,
        end_event_gemm2=None,
        start_event_gather=None,
        end_event_gather=None
    ):
        num_rows, hidden_size = hidden_states.shape
        num_moe_inputs = self.n_experts_per_token * num_rows
        act_quant = a1q_scale is not None
        
        if expert_map is None and self.ep_size > 1:
            expert_map = self.expert_map

        if act_quant:
            remapped_scales = torch.empty(
                (num_rows * self.n_experts_per_token, a1q_scale.shape[1]),
                dtype=a1q_scale.dtype,
                device=a1q_scale.device)
        else:
            remapped_scales = None
        remapped_hidden_states = torch.empty(
            (num_rows * self.n_experts_per_token, hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device)
        rows_per_expert = torch.zeros((self.num_experts),
                                                dtype=torch.int32,
                                                device=hidden_states.device)
        unpermuted_row_to_permuted_row = torch.empty(
            (num_rows, self.n_experts_per_token),
            dtype=torch.int32,
            device=hidden_states.device)

        torch.xpu.synchronize()
        if start_event_remap is not None:
            start_event_remap.record()
        torch.ops._moe_C.remap_hidden_states(
            hidden_states=hidden_states,
            hidden_states_scales=a1q_scale,
            remapped_hidden_states=remapped_hidden_states,
            remapped_hidden_states_scales=remapped_scales,
            expert_map=expert_map,
            rows_per_expert=rows_per_expert,
            unpermuted_row_to_permuted_row=unpermuted_row_to_permuted_row,
            topk_ids=topk_ids,
            total_experts_num=self.total_experts_num,
            local_experts_num=self.local_experts_num)
        torch.xpu.synchronize()
        if end_event_remap is not None:
            end_event_remap.record()

        ########### gemm1 ##################
        gemm1_output = torch.empty((num_moe_inputs, 2 * self.inter_size),
                                dtype=output.dtype,
                                device=output.device)
        torch.xpu.synchronize()
        if start_event_gemm1 is not None:
            start_event_gemm1.record()
        torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=remapped_hidden_states,
            ptr_A_scale=remapped_scales,
            ptr_B=self.w13,
            ptr_B_scale=self.gemm1_wei_scales,
            ptr_bias=self.w13_bias,
            ptr_D=gemm1_output,
            rows_per_expert=rows_per_expert,
            N=2 * self.inter_size,
            K=hidden_size,
            num_experts=self.num_experts)
        if end_event_gemm1 is not None:
            end_event_gemm1.record()

        # Apply swiglu_limit clamping before activation
        if self.gemm1_clamp_limit is not None and self.gemm1_clamp_limit > 0:
            gate = gemm1_output[:, :self.inter_size]
            up = gemm1_output[:, self.inter_size:]
            gate.clamp_(max=self.gemm1_clamp_limit)
            up.clamp_(min=-self.gemm1_clamp_limit, max=self.gemm1_clamp_limit)

        # act
        act_output = torch.empty(
            (num_moe_inputs, self.inter_size * self.inter_size_scale),
            dtype=gemm1_output.dtype,
            device=gemm1_output.device)
        self.act_func(act_output, gemm1_output)

        ########### gemm2 ##################
        gemm2_output = torch.empty((num_moe_inputs, hidden_size),
                                dtype=output.dtype,
                                device=output.device)

        if act_quant:
            act_output, gemm2_act_scale = quant_act_xpu(act_output, self.recipe)
        torch.xpu.synchronize()
        if start_event_gemm2 is not None:
            start_event_gemm2.record()
        torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=act_output,
            ptr_A_scale=gemm2_act_scale if act_quant else None,
            ptr_B=self.w2,
            ptr_B_scale=self.gemm2_wei_scales,
            ptr_bias=self.w2_bias,
            ptr_D=gemm2_output,
            rows_per_expert=rows_per_expert,
            N=hidden_size,
            K=self.inter_size * self.inter_size_scale,
            num_experts=self.num_experts)
        if end_event_gemm2 is not None:
            end_event_gemm2.record()

        torch.xpu.synchronize()
        if start_event_gather is not None:
            start_event_gather.record()
        torch.ops._moe_C.moe_gather(output, gemm2_output, topk_weights,
                                    unpermuted_row_to_permuted_row,
                                    self.num_experts)
        if end_event_gather is not None:
            end_event_gather.record()

        gemm1_n = 2 * self.inter_size  # gemm1: N = 2 * inter_size
        gemm2_n = hidden_size      # gemm2: N = hidden_size

        active_experts1 = (rows_per_expert > 0).sum().item()
        gemm1_m = remapped_hidden_states.shape[0]
        gemm1_k = remapped_hidden_states.shape[1]

        active_experts2 = (rows_per_expert > 0).sum().item()
        gemm2_m = act_output.shape[0]
        gemm2_k = act_output.shape[1]
        return ((gemm1_m, gemm1_n, gemm1_k, active_experts1),
                (gemm2_m, gemm2_n, gemm2_k, active_experts2))
