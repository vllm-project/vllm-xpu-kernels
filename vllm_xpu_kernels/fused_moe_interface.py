# SPDX-License-Identifier: Apache-2.0
import os
from typing import Optional

import torch

try:
    from . import _C  # noqa: F401
    from . import _xpu_C  # noqa: F401
    FUSEDMOE_UNAVAILABLE_REASON = None
    FUSEDMOE_AVAILABLE = True
except ImportError as e:
    FUSEDMOE_UNAVAILABLE_REASON = str(e)
    FUSEDMOE_AVAILABLE = False

from .moe_utils import quant_act_xpu, ref_fused_moe

REF_FUSED_MOE_ENV = "VLLM_XPU_FUSED_MOE_USE_REF"
USE_MXFP4_FP8_ENV = "VLLM_XPU_FUSED_MOE_USE_MXFP4_FP8"
_VALID_TOKENS_CACHE = {}

def _is_env_enabled(env_name: str, default: str = "0") -> bool:
    value = os.environ.get(env_name, default).strip().upper()
    return value in ("1", "ON", "TRUE", "YES", "Y")


def _should_use_ref_fused_moe(is_mxfp8: bool, is_block_fp8: bool) -> bool:  
    if is_mxfp8 or is_block_fp8:
        return True
    return _is_env_enabled(REF_FUSED_MOE_ENV)


def _should_use_ep_valid_tokens(total_experts_num: int,
                                local_experts_num: int,
                                num_rows: int) -> bool:
    # Only enable this EP optimization for long-sequence prefill. Small-row
    # workloads, especially decode, do not show stable benefit and should keep
    # the existing path unchanged.
    return total_experts_num > local_experts_num and num_rows >= 1024


def _get_recipe(is_fp8, is_mxfp8, is_mxfp4, is_int4, is_block_fp8):
    if is_mxfp8:
        return "mxfp8"
    elif is_block_fp8:
        return "fp8block"
    elif is_mxfp4:
        return "mxfp4_fp8" if _is_env_enabled(USE_MXFP4_FP8_ENV) else "mxfp4"
    elif is_int4:
        return "int4"
    elif is_fp8:
        return "fp8"
    else:
        return "bf16"

def _get_weights_dtype(weight, scales):
    weight_dtype = weight.dtype
    is_fp8 = weight_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    is_int4 = weight_dtype == torch.uint8
    is_mxfp4 = weight_dtype == torch.float4_e2m1fn_x2
    is_mxfp8 = (is_fp8
                and scales is not None
                and scales.dtype == torch.uint8) 
    is_block_fp8 = (is_fp8 
                    and scales is not None
                    and scales.dtype == torch.float32
                    and scales.ndim == 3)
    is_fp8 = is_fp8 and not is_mxfp8 and not is_block_fp8

    return is_fp8, is_int4, is_mxfp4, is_mxfp8, is_block_fp8


def cutlass_grouped_gemm(input_A, input_A_scale, input_B, input_B_scale, bias,
                         output, expert_token_count, n, k, num_experts):
    num_rows_per_expert = torch.tensor(expert_token_count,
                                        dtype=torch.int32,
                                        device="xpu")
    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=input_A,
        ptr_A_scale=input_A_scale,
        ptr_B=input_B,
        ptr_B_scale=input_B_scale,
        ptr_bias=bias,
        ptr_D=output,
        rows_per_expert=num_rows_per_expert,
        N=n,
        K=k,
        num_experts=num_experts,
        is_B_int4=False,
        is_B_mxfp4=False)


def cutlass_grouped_gemm_xe2(input_A, input_B, scales, bias, output,
                             num_rows_per_expert, n, k, num_experts):
    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=input_A,
        ptr_A_scale=None,
        ptr_B=input_B,
        ptr_B_scale=scales,
        ptr_bias=bias,
        ptr_D=output,
        rows_per_expert=num_rows_per_expert,
        N=n,
        K=k,
        num_experts=num_experts)


def ceilDiv(a, b):
    return (a + b - 1) // b


def compute_num_tokens_per_block(num_tokens, num_experts_per_node):
    for num_tokens_per_block in [32, 64, 128, 256, 512, 1024]:
        num_blocks_per_seq = ceilDiv(num_tokens, num_tokens_per_block)
        if num_blocks_per_seq * num_experts_per_node <= num_tokens_per_block:
            return num_tokens_per_block
    return 1024


def fused_moe_activation(act_output,
                         gemm1_output,
                         activation,
                         valid_tokens=None):
    if activation == "silu":
        torch.ops._C.silu_and_mul(act_output, gemm1_output, valid_tokens)
    elif activation == "gelu":
        torch.ops._C.gelu_and_mul(act_output, gemm1_output, valid_tokens)
    elif activation == "gelu_tanh":
        torch.ops._C.gelu_tanh_and_mul(act_output, gemm1_output, valid_tokens)
    elif activation == "swigluoai" or ("SWIGLUOAI" in str(activation)):
        torch.ops._C.swigluoai_and_mul(act_output, gemm1_output,
                                       1.702, 7.0, valid_tokens)
    elif activation == "relu2_no_mul":
        torch.ops._C.relu2_no_mul(act_output, gemm1_output, valid_tokens)
    elif activation == "swiglustep":
        torch.ops._C.swiglustep_and_mul(act_output, gemm1_output,
                                        7.0, valid_tokens)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}.")

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

class XpuFusedMoe:
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
        assert w13.is_contiguous() and w2.is_contiguous()

        (is_fp8, 
         is_int4, 
         is_mxfp4, 
         is_mxfp8, 
         is_block_fp8) = _get_weights_dtype(w13, w13_scales)

        # 4bits support [E, N, K]
        # other types [E, K, N]
        if not is_int4 and not is_mxfp4:
            self.inter_size = w13.shape[-1] // 2
        else:
            self.inter_size = w13.shape[-2] // 2

        # FIXME: move this to vllm
        if is_int4 and not hasattr(w13, 'xpu_fused_moe'):
            w13_tmp = torch.empty_like(w13).to(torch.int8)
            w2_tmp = torch.empty_like(w2).to(torch.int8)
            for i in range(num_experts):
                w13_tmp[i] = implement_zp(w13[i])
                w2_tmp[i] = implement_zp(w2[i])
            w13_tmp = w13_tmp.contiguous()
            w2_tmp = w2_tmp.contiguous()
            w13.data = w13_tmp
            w2.data = w2_tmp
            w13.xpu_fused_moe = True

        self.w13 = w13
        self.w2 = w2

        if (not is_fp8 
            and not is_int4
            and not is_mxfp4
            and not is_block_fp8
            and not is_mxfp8):
            self.gemm1_wei_scales = None
            self.gemm2_wei_scales = None
        else:
            self.gemm1_wei_scales = w13_scales
            self.gemm2_wei_scales = w2_scales

        self.w13_bias = w13_bias
        self.w2_bias = w2_bias

        self.n_experts_per_token = n_experts_per_token
        self.activation = activation
        self.inter_size_scale = 2 if self.activation == "relu2_no_mul" else 1
        self.num_experts = num_experts
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.is_fp8 = is_fp8
        self.is_int4 = is_int4
        self.is_mxfp4 = is_mxfp4
        self.is_mxfp8 = is_mxfp8
        self.is_block_fp8 = is_block_fp8
        self.gemm1_clamp_limit = gemm1_clamp_limit
        self.recipe = _get_recipe(is_fp8, is_mxfp8, is_mxfp4, is_int4,
                                   is_block_fp8)
        self._use_ref = _should_use_ref_fused_moe(is_mxfp8, is_block_fp8)
        if self.activation == "silu":
            self.act_func = torch.ops._C.silu_and_mul
        elif self.activation == "gelu":
            self.act_func = torch.ops._C.gelu_and_mul
        elif self.activation == "gelu_tanh":
            self.act_func = torch.ops._C.gelu_tanh_and_mul
        elif self.activation == "swigluoai" \
            or ("SWIGLUOAI" in str(self.activation)):
            self.act_func = torch.ops._C.swigluoai_and_mul
        elif self.activation == "relu2_no_mul":
            self.act_func = torch.ops._C.relu2_no_mul
        elif self.activation == "swiglustep":
            self.act_func = torch.ops._C.swiglustep_and_mul
        else:
            raise ValueError(
                f"Unsupported FusedMoe activation: {self.activation}.")

        self.expert_map = expert_map
        if self.expert_map is None and self.ep_size > 1:
            self.expert_map = torch.empty((self.num_experts * self.ep_size),
                                    dtype=torch.int32,
                                    device=w13.device)
            torch.ops._moe_C.init_expert_map(
                self.expert_map,
                self.num_experts,
                self.ep_rank,
                self.ep_size)

        if self.expert_map is not None:
            self.total_experts_num = self.expert_map.shape[0]
        else:
            self.total_experts_num = self.num_experts * self.ep_size
        self.local_experts_num = self.num_experts
        # Cache a 1-element int64 tensor for valid_tokens to avoid
        # repeated device allocations per call which can increase latency.
        self._valid_tokens = None

    def apply(
        self,
        output,
        hidden_states,
        topk_weights,
        topk_ids,
        expert_map=None,
        a1q_scale=None,
    ):
        if self._use_ref:
            self._apply_ref(output, hidden_states,
                            topk_weights, topk_ids,
                            expert_map, a1q_scale)
        else:
            self._apply_kernel(output, hidden_states,
                               topk_weights, topk_ids,
                               expert_map, a1q_scale)

    def _apply_ref(
        self,
        output,
        hidden_states,
        topk_weights,
        topk_ids,
        expert_map=None,
        a1q_scale=None,
    ):
        return ref_fused_moe(recipe=self.recipe,
                            output=output,
                            hidden_states=hidden_states,
                            w13=self.w13,
                            w13_scales=self.gemm1_wei_scales,
                            w13_bias=self.w13_bias,
                            w2=self.w2,
                            w2_scales=self.gemm2_wei_scales,
                            w2_bias=self.w2_bias,
                            topk_weights=topk_weights,
                            topk_ids=topk_ids,
                            n_experts_per_token=self.n_experts_per_token,
                            activation=self.activation,
                            num_experts=self.num_experts,
                            ep_rank=self.ep_rank,
                            ep_size=self.ep_size,
                            expert_map=expert_map,
                            a1q_scale=a1q_scale)

    def _apply_kernel(
        self,
        output,
        hidden_states,
        topk_weights,
        topk_ids,
        expert_map=None,
        a1q_scale=None,
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

        valid_tokens = None
        if _should_use_ep_valid_tokens(self.total_experts_num,
                           self.local_experts_num,
                           num_rows):
            # Reuse cached tensor when possible to avoid allocation overhead.
            if (self._valid_tokens is None or
                    self._valid_tokens.device != hidden_states.device):
                self._valid_tokens = torch.empty((1,),
                                                dtype=torch.int64,
                                                device=hidden_states.device)
            valid_tokens = self._valid_tokens

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
            local_experts_num=self.local_experts_num,
            valid_tokens=valid_tokens)

        ########### gemm1 ##################
        gemm1_output = torch.empty((num_moe_inputs, 2 * self.inter_size),
                                dtype=output.dtype,
                                device=output.device)
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

        # Apply swiglu_limit clamping before activation
        if self.gemm1_clamp_limit is not None and self.gemm1_clamp_limit > 0:
            gate = gemm1_output[:, :self.inter_size]
            up = gemm1_output[:, self.inter_size:]
            gate.clamp_(max=self.gemm1_clamp_limit)
            up.clamp_(min=-self.gemm1_clamp_limit, max=self.gemm1_clamp_limit)

        # act: pass valid_tokens to the shared kernel to skip invalid rows.
        act_output = torch.empty(
            (num_moe_inputs, self.inter_size * self.inter_size_scale),
            dtype=gemm1_output.dtype,
            device=gemm1_output.device)
        fused_moe_activation(act_output,
                             gemm1_output,
                             self.activation,
                             valid_tokens)

        ########### gemm2 ##################
        gemm2_output = torch.empty((num_moe_inputs, hidden_size),
                                dtype=output.dtype,
                                device=output.device)

        if act_quant:
            act_output, gemm2_act_scale = quant_act_xpu(act_output, self.recipe)
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

        torch.ops._moe_C.moe_gather(output, gemm2_output, topk_weights,
                                    unpermuted_row_to_permuted_row,
                                    self.num_experts)

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
                  ep_rank=0,
                  ep_size=1,
                  expert_map=None,
                  output=None,
                  is_fp8=False,
                  is_int4=False,
                  is_mxfp4=False,
                  is_mxfp8=False,
                  is_block_fp8=False,
                  gemm1_clamp_limit: Optional[float]=None):
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
    is_mxfp8: bool
    is_block_fp8: bool
    '''
    if output is None:
        output = torch.empty_like(hidden_states)
    else:
        assert output.shape == hidden_states.shape, \
            "output shape must be the same as hidden_states shape"
    if _should_use_ref_fused_moe(is_mxfp8):
        recipe = _get_recipe(is_fp8, is_mxfp8, is_mxfp4, is_int4,
                             is_block_fp8)
        out = ref_fused_moe(recipe=recipe,
                            x=hidden_states,
                            w13=w13,
                            w13_scales=w13_scales,
                            w13_bias=w13_bias,
                            w2=w2,
                            w2_scales=w2_scales,
                            w2_bias=w2_bias,
                            expert_weights=topk_weights,
                            expert_indices=topk_ids,
                            num_per_tok=n_experts_per_token,
                            activation=activation,
                            num_experts=num_experts,
                            ep_rank=ep_rank,
                            ep_size=ep_size,
                            gemm1_clamp_limit=gemm1_clamp_limit)
        output.copy_(out)
        return output

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

    num_rows, hidden_size = list(hidden_states.shape)
    num_moe_inputs = n_experts_per_token * num_rows
    gemm1_output = torch.empty((num_moe_inputs, 2 * inter_size),
                               dtype=hidden_states.dtype,
                               device=hidden_states.device)

    if not is_fp8 and not is_int4 and not is_mxfp4 and not is_block_fp8:
        gemm1_wei_scales = None
        gemm2_wei_scales = None
    else:
        gemm1_wei_scales = w13_scales
        gemm2_wei_scales = w2_scales

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

    valid_tokens = None
    if _should_use_ep_valid_tokens(total_experts_num,
                                   local_experts_num,
                                   num_rows):
        # Reuse a module-level cache for functional API to avoid
        # allocating a new tensor on every call.
        dev_key = str(hidden_states.device)
        cached_tokens = _VALID_TOKENS_CACHE.get(dev_key)
        if (
            cached_tokens is None
            or cached_tokens.device != hidden_states.device
        ):
            cached_tokens = torch.empty(
                (1,),
                dtype=torch.int64,
                device=hidden_states.device,
            )
            _VALID_TOKENS_CACHE[dev_key] = cached_tokens
        valid_tokens = cached_tokens

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
        local_experts_num=local_experts_num,
        valid_tokens=valid_tokens)

    ########### gemm1 ##################
    input_B = w13

    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=remapped_hidden_states,
        ptr_A_scale=None,
        ptr_B=input_B,
        ptr_B_scale=gemm1_wei_scales,
        ptr_bias=w13_bias,
        ptr_D=gemm1_output,
        rows_per_expert=rows_per_expert,
        N=2 * inter_size,
        K=hidden_size,
        num_experts=num_experts,
        is_B_int4=is_int4,
        is_B_mxfp4=is_mxfp4)

    # Apply swiglu_limit clamping before activation
    if gemm1_clamp_limit is not None and gemm1_clamp_limit > 0:
        gate = gemm1_output[:, :inter_size]
        up = gemm1_output[:, inter_size:]
        gate.clamp_(max=gemm1_clamp_limit)
        up.clamp_(min=-gemm1_clamp_limit, max=gemm1_clamp_limit)

    inter_size_scale = 2 if activation == "relu2_no_mul" else 1
    # act: pass valid_tokens to the shared kernel to skip invalid rows.
    act_output = torch.empty((num_moe_inputs, inter_size * inter_size_scale),
                             dtype=gemm1_output.dtype,
                             device=gemm1_output.device)
    fused_moe_activation(act_output, gemm1_output, activation, valid_tokens)

    ########### gemm2 ##################
    input_A = act_output.contiguous()
    input_B = w2
    gemm2_output = torch.empty((num_moe_inputs, hidden_size),
                               dtype=hidden_states.dtype,
                               device=hidden_states.device)

    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=input_A,
        ptr_A_scale=None,
        ptr_B=input_B,
        ptr_B_scale=gemm2_wei_scales,
        ptr_bias=w2_bias,
        ptr_D=gemm2_output,
        rows_per_expert=rows_per_expert,
        N=hidden_size,
        K=inter_size * inter_size_scale,
        num_experts=num_experts,
        is_B_int4=is_int4,
        is_B_mxfp4=is_mxfp4)

    torch.ops._moe_C.moe_gather(output, gemm2_output, topk_weights,
                                unpermuted_row_to_permuted_row,
                                num_experts)

    return output