# SPDX-License-Identifier: Apache-2.0
import os

import torch

try:
    from . import _C  # noqa: F401
    from . import _xpu_C  # noqa: F401
    FUSEDMOE_UNAVAILABLE_REASON = None
    FUSEDMOE_AVAILABLE = True
except ImportError as e:
    FUSEDMOE_UNAVAILABLE_REASON = str(e)
    FUSEDMOE_AVAILABLE = False

from ._mx_utils import (fp4_e2m1fn_x2_to_float, hp_from_1x128, hp_from_128x128,
                        quant_fp8_act, quant_mxfp_act)

REF_FUSED_MOE_ENV = "VLLM_XPU_FUSED_MOE_USE_REF"


def _is_env_enabled(env_name: str, default: str = "0") -> bool:
    value = os.environ.get(env_name, default).strip().upper()
    return value in ("1", "ON", "TRUE", "YES", "Y")


def _should_use_ref_fused_moe(is_mxfp8: bool) -> bool:
    if is_mxfp8:
        return True
    return _is_env_enabled(REF_FUSED_MOE_ENV)


def _get_recipe(is_fp8, is_mxfp8, is_mxfp4, is_int4, is_block_fp8):
    if is_mxfp8:
        return "mxfp8"
    elif is_block_fp8:
        return "fp8block"
    elif is_mxfp4:
        return "mxfp4"
    elif is_int4:
        return "int4"
    elif is_fp8:
        return "fp8"
    else:
        return "bf16"


def cutlass_grouped_gemm(input_A, input_B, bias, output, expert_token_count, n,
                         k, num_experts):
    num_rows_per_expert = torch.tensor(expert_token_count,
                                        dtype=torch.int32,
                                        device="xpu")
    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=input_A,
        ptr_B=input_B,
        ptr_scales=None,
        ptr_bias=bias,
        ptr_D=output,
        rows_per_expert=num_rows_per_expert,
        N=n,
        K=k,
        num_experts=num_experts,
        is_B_int4=False,
        is_B_mxfp4=False)


def cutlass_grouped_gemm_xe2(input_A, input_B, scales, bias, output,
                             num_rows_per_expert, n, k, num_experts, is_B_int4,
                             is_B_mxfp4):
    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=input_A,
        ptr_B=input_B,
        ptr_scales=scales,
        ptr_bias=bias,
        ptr_D=output,
        rows_per_expert=num_rows_per_expert,
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


def fused_moe_activation(act_output, gemm1_output, activation):
    if activation == "silu":
        torch.ops._C.silu_and_mul(act_output, gemm1_output)
    elif activation == "gelu":
        torch.ops._C.gelu_and_mul(act_output, gemm1_output)
    elif activation == "gelu_tanh":
        torch.ops._C.gelu_tanh_and_mul(act_output, gemm1_output)
    elif activation == "swigluoai" or ("SWIGLUOAI" in str(activation)):
        torch.ops._C.swigluoai_and_mul(act_output, gemm1_output, 1.702, 7.0)
    elif activation == "relu2_no_mul":
        torch.ops._C.relu2_no_mul(act_output, gemm1_output)
    elif activation == "swiglustep":
        torch.ops._C.swiglustep_and_mul(act_output, gemm1_output, 7.0)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}.")

def _naive_fused_moe_activation(gemm1_output, activation):
    if activation == "silu":
        return torch.nn.functional.silu(gemm1_output)
    elif activation == "gelu":
        return torch.nn.functional.gelu(gemm1_output)
    elif activation == "gelu_tanh":
        return torch.nn.functional.gelu(gemm1_output, approximate="tanh")
    elif activation == "swigluoai" or ("SWIGLUOAI" in str(activation)):
        gate, up = gemm1_output.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up * 1.702 * 7.0
    elif activation == "relu2_no_mul":
        return torch.nn.functional.relu(gemm1_output).pow(2)
    elif activation == "swiglustep":
        gate, up = gemm1_output.chunk(2, dim=-1)
        return torch.nn.functional.relu(gate - 7.0).sign() * up * 7.0
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

def ref_fused_moe(recipe,
                  x,
                  w13,
                  w13_scales,
                  w13_bias,
                  w2,
                  w2_scales,
                  w2_bias,
                  expert_weights,
                  expert_indices,
                  num_per_tok,
                  activation,
                  num_experts,
                  ep_rank=0,
                  ep_size=1):

    activation_dtype = x.dtype 

    flat_expert_indices = expert_indices.view(-1)
    flat_expert_weights = expert_weights.view(-1, 1)

    expert_start_id = num_experts * ep_rank
    expert_end_id = expert_start_id + num_experts
    expert_cache = torch.zeros_like(x).to(activation_dtype)
    idxs = flat_expert_indices.argsort()
    counts = flat_expert_indices.bincount().cpu().numpy()
    tokens_per_expert = counts.cumsum()
    token_idxs = idxs // num_per_tok

    if recipe == "fp8block":
        x_f = x.to(torch.float32)
        _q, _scale = quant_fp8_act(x_f)
        x = hp_from_1x128(_q, _scale)
        w13 = w13.transpose(1, 2).contiguous()
        w2 = w2.transpose(1, 2).contiguous()
    elif recipe == "mxfp8":
        w13 = w13.transpose(1, 2).contiguous()
        w2 = w2.transpose(1, 2).contiguous()
        act_ori_shape = x.shape
        w13_ori_shape = w13.shape
        w2_ori_shape = w2.shape
        _q, _scale = quant_mxfp_act(x, "mxfp8")
        x = _q.float().reshape(-1, 32) * (_scale.reshape(-1, 1).float())
        x = x.reshape(act_ori_shape)
        w13_scales = w13_scales.view(torch.float8_e8m0fnu)
        w2_scales = w2_scales.view(torch.float8_e8m0fnu)
        w13 = (w13.to(activation_dtype).reshape(-1, 32)
               * w13_scales.reshape(-1, 1).to(activation_dtype))
        w2 = (w2.to(activation_dtype).reshape(-1, 32)
              * w2_scales.reshape(-1, 1).to(activation_dtype))
        w13 = w13.reshape(w13_ori_shape)
        w2 = w2.reshape(w2_ori_shape)
    elif recipe == "mxfp4":
        act_ori_shape = x.shape
        _q, _scale = quant_mxfp_act(x, "mxfp4")
        x = fp4_e2m1fn_x2_to_float(_q).reshape(-1, 32) * (_scale.reshape(
            -1, 1).to(activation_dtype))
        x = x.reshape(act_ori_shape)
        w13_ori_shape = w13.shape
        w2_ori_shape = w2.shape
        w13 = fp4_e2m1fn_x2_to_float(w13).reshape(
            -1, 32) * (w13_scales.reshape(-1, 1).to(activation_dtype))
        w2 = fp4_e2m1fn_x2_to_float(w2).reshape(-1, 32) * (w2_scales.reshape(
            -1, 1).to(activation_dtype))
        w13 = w13.reshape(w13_ori_shape[:-1] + (w13_ori_shape[-1] * 2, ))
        w2 = w2.reshape(w2_ori_shape[:-1] + (w2_ori_shape[-1] * 2, ))

    for expert_id, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        if (start_idx == end_idx) or (expert_id
                                      < expert_start_id) or (expert_id
                                                             >= expert_end_id):
            continue
        exp_token_idxs = token_idxs[start_idx:end_idx]
        expert_tokens = x[exp_token_idxs]

        ### dequant weight13
        expert_w13 = w13[expert_id, :, :]
        if recipe == "fp8block":
            expert_w13 = hp_from_128x128(w13[expert_id, :, :],
                                         w13_scales[expert_id, :, :])
        ###

        w1, w3 = torch.split(expert_w13,
                             int(list(expert_w13.shape)[0] / 2),
                             dim=0)
        if w13_bias is not None:
            w1_bias, w3_bias = w13_bias[expert_id, :].chunk(2)
        gemm1 = (expert_tokens.to(activation_dtype) @ w1.T.to(activation_dtype))
        if w13_bias is not None:
            gemm1 += w1_bias.to(activation_dtype)

        gate = _naive_fused_moe_activation(gemm1, activation)
        up = (expert_tokens.to(activation_dtype) @ w3.T.to(activation_dtype))
        if w13_bias is not None:
            up += w3_bias.to(activation_dtype)

        ### quant act for gemm2 and dequant weight 2
        gemm2_input = gate * up
        expert_w2 = w2[expert_id, :, :]
        if recipe == "fp8block":
            expert_w2 = hp_from_128x128(w2[expert_id, :, :],
                                        w2_scales[expert_id, :, :])
            gemm2_input_f = gemm2_input.to(torch.float32)
            _q, _scale = quant_fp8_act(gemm2_input_f)
            gemm2_input = hp_from_1x128(_q, _scale).to(activation_dtype)
        elif recipe == "mxfp8":
            _q, _scale = quant_mxfp_act(gemm2_input, "mxfp8")
            gemm2_input = (
                _q.to(activation_dtype).reshape(-1, 32)
                * _scale.reshape(-1, 1).to(activation_dtype))
            gemm2_input = gemm2_input.reshape(_q.shape)
        elif recipe == "mxfp4":
            _q, _scale = quant_mxfp_act(gemm2_input, "mxfp4")
            gemm2_input = fp4_e2m1fn_x2_to_float(_q).reshape(
                -1, 32) * (_scale.reshape(-1, 1).to(activation_dtype))
            gemm2_input = gemm2_input.reshape(_q.shape[:-1] +
                                              (_q.shape[-1] * 2, ))
        ###

        expert_out = (gemm2_input) @ expert_w2.T.to(activation_dtype)

        if w2_bias is not None:
            expert_out += w2_bias[expert_id, :].to(activation_dtype)

        expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
        expert_cache.scatter_reduce_(0,
                                     exp_token_idxs.view(-1, 1).repeat(
                                         1, x.shape[-1]),
                                     expert_out,
                                     reduce='sum')

    expert_cache = expert_cache.to(x.dtype)
    return expert_cache

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
        is_fp8=False,
        is_int4=False,
        is_mxfp4=False,
        is_mxfp8=False,
        is_block_fp8=False
    ):
        # 4bits support [E, N, K]
        # other types [E, K, N]
        if not is_int4 and not is_mxfp4:
            self.inter_size = w13.shape[-1] // 2
        else:
            self.inter_size = w13.shape[-2] // 2

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

        self.w13 = w13
        self.w2 = w2

        if not is_fp8 and not is_int4 and not is_mxfp4:
            self.gemm1_scales = None
            self.gemm2_scales = None
        else:
            self.gemm1_scales = w13_scales
            self.gemm2_scales = w2_scales

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
        self.recipe = _get_recipe(is_fp8, is_mxfp8, is_mxfp4, is_int4,
                                   is_block_fp8)
        self._use_ref = _should_use_ref_fused_moe(is_mxfp8)

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

    def apply(
        self,
        output,
        hidden_states,
        topk_weights,
        topk_ids,
        expert_map=None,
    ):
        if self._use_ref:
            self._apply_ref(output, hidden_states,
                            topk_weights, topk_ids,
                            expert_map)
        else:
            self._apply_kernel(output, hidden_states,
                               topk_weights, topk_ids,
                               expert_map)

    def _apply_ref(
        self,
        output,
        hidden_states,
        topk_weights,
        topk_ids,
        expert_map=None,
    ):
        out = ref_fused_moe(recipe=self.recipe,
                            x=hidden_states,
                            w13=self.w13,
                            w13_scales=self.gemm1_scales,
                            w13_bias=self.w13_bias,
                            w2=self.w2,
                            w2_scales=self.gemm2_scales,
                            w2_bias=self.w2_bias,
                            expert_weights=topk_weights,
                            expert_indices=topk_ids,
                            num_per_tok=self.n_experts_per_token,
                            activation=self.activation,
                            num_experts=self.num_experts,
                            ep_rank=self.ep_rank,
                            ep_size=self.ep_size)
        output.copy_(out)

    def _apply_kernel(
        self,
        output,
        hidden_states,
        topk_weights,
        topk_ids,
        expert_map=None,
    ):
        num_rows, hidden_size = hidden_states.shape
        num_moe_inputs = self.n_experts_per_token * num_rows
        
        if expert_map is None and self.ep_size > 1:
            expert_map = self.expert_map

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

        torch.ops._moe_C.remap_hidden_states(
            hidden_states=hidden_states,
            hidden_states_scales=None,
            remapped_hidden_states=remapped_hidden_states,
            remapped_hidden_states_scales=None,
            expert_map=expert_map,
            rows_per_expert=rows_per_expert,
            unpermuted_row_to_permuted_row=unpermuted_row_to_permuted_row,
            topk_ids=topk_ids,
            total_experts_num=self.total_experts_num,
            local_experts_num=self.local_experts_num)

        ########### gemm1 ##################
        gemm1_output = torch.empty((num_moe_inputs, 2 * self.inter_size),
                                dtype=hidden_states.dtype,
                                device=hidden_states.device)
        torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=remapped_hidden_states,
            ptr_B=self.w13,
            ptr_scales=self.gemm1_scales,
            ptr_bias=self.w13_bias,
            ptr_D=gemm1_output,
            rows_per_expert=rows_per_expert,
            N=2 * self.inter_size,
            K=hidden_size,
            num_experts=self.num_experts,
            is_B_int4=self.is_int4,
            is_B_mxfp4=self.is_mxfp4)

        # act
        act_output = torch.empty(
            (num_moe_inputs, self.inter_size * self.inter_size_scale),
            dtype=gemm1_output.dtype,
            device=gemm1_output.device)
        self.act_func(act_output, gemm1_output)

        ########### gemm2 ##################
        gemm2_output = torch.empty((num_moe_inputs, hidden_size),
                                dtype=hidden_states.dtype,
                                device=hidden_states.device)

        torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=act_output,
            ptr_B=self.w2,
            ptr_scales=self.gemm2_scales,
            ptr_bias=self.w2_bias,
            ptr_D=gemm2_output,
            rows_per_expert=rows_per_expert,
            N=hidden_size,
            K=self.inter_size * self.inter_size_scale,
            num_experts=self.num_experts,
            is_B_int4=self.is_int4,
            is_B_mxfp4=self.is_mxfp4)

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
                  is_block_fp8=False):
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
                            ep_size=ep_size)
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

    ########### gemm1 ##################
    input_B = w13

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

    inter_size_scale = 2 if activation == "relu2_no_mul" else 1
    # act
    act_output = torch.empty((num_moe_inputs, inter_size * inter_size_scale),
                             dtype=gemm1_output.dtype,
                             device=gemm1_output.device)
    fused_moe_activation(act_output, gemm1_output, activation)

    ########### gemm2 ##################
    input_A = act_output.contiguous()
    input_B = w2
    gemm2_output = torch.empty((num_moe_inputs, hidden_size),
                               dtype=hidden_states.dtype,
                               device=hidden_states.device)

    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=input_A,
        ptr_B=input_B,
        ptr_scales=gemm2_scales,
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
