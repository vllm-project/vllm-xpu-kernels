# SPDX-License-Identifier: Apache-2.0
import os
from typing import Optional

import torch

try:
    from . import _C  # noqa: F401
    from . import _moe_C  # noqa: F401
    from . import _xpu_C  # noqa: F401
    FUSEDMOE_UNAVAILABLE_REASON = None
    FUSEDMOE_AVAILABLE = True
except ImportError as e:
    FUSEDMOE_UNAVAILABLE_REASON = str(e)
    FUSEDMOE_AVAILABLE = False

from .moe_utils import (dequant_fp8_block_act, dequant_mxfp8,
                        mxfp_scale_padded_rows, quant_act_xpu, ref_fused_moe)

REF_FUSED_MOE_ENV = "VLLM_XPU_FUSED_MOE_USE_REF"
USE_MXFP4_FP8_ENV = "VLLM_XPU_FUSED_MOE_USE_MXFP4_FP8"
# MXFP8 / block-FP8 use the native grouped-GEMM path by default.
NATIVE_MXFP8_ENV = "VLLM_XPU_FUSED_MOE_NATIVE_MXFP8"
NATIVE_BLOCK_FP8_ENV = "VLLM_XPU_FUSED_MOE_NATIVE_BLOCK_FP8"


def _is_env_enabled(env_name: str, default: str = "0") -> bool:
    value = os.environ.get(env_name, default).strip().upper()
    return value in ("1", "ON", "TRUE", "YES", "Y")


def _env_native_default_on(env_name: str) -> bool:
    """True unless env is explicitly set to a falsy value (default native)."""
    return os.environ.get(env_name, "1").strip().upper() not in (
        "0", "OFF", "FALSE", "NO", "N")


def _should_use_ref_fused_moe(is_mxfp8: bool, is_block_fp8: bool) -> bool:
    """Return True when the slow Python ref path must be used.

    MXFP8 and block-FP8 run on the native grouped GEMM by default; the env
    toggles exist to fall back to the reference implementation when debugging.
    """
    if _is_env_enabled(REF_FUSED_MOE_ENV):
        return True
    if is_mxfp8 and not _env_native_default_on(NATIVE_MXFP8_ENV):
        return True
    return (is_block_fp8
            and not _env_native_default_on(NATIVE_BLOCK_FP8_ENV))


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
    """Infer the quantization recipe flags from the weight/scale dtypes."""
    weight_dtype = weight.dtype
    is_fp8 = weight_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    is_int4 = weight_dtype in (torch.uint8, torch.int8)
    is_mxfp4 = weight_dtype == torch.float4_e2m1fn_x2
    is_mxfp8 = (is_fp8 and scales is not None
                and scales.dtype in (torch.uint8, torch.float8_e8m0fnu))
    is_block_fp8 = (is_fp8 and scales is not None
                    and scales.dtype == torch.float32 and scales.ndim == 3)
    is_int4 = is_int4 and scales is not None
    is_fp8 = is_fp8 and not is_mxfp8 and not is_block_fp8

    return is_fp8, is_int4, is_mxfp4, is_mxfp8, is_block_fp8


def _is_packed_4bit(weight):
    return weight.dtype in (torch.uint8, torch.int8, torch.float4_e2m1fn_x2)


def _uses_xe2_grouped_gemm(weight):
    """Mirrors the routing in csrc/xpu/grouped_gemm/grouped_gemm_interface.cpp.

    BMG / PVC / LNL and the XE3 client parts run the xe_2 kernel; only CRI and
    NVL-P run the xe_3 one.
    """
    device_index = -1 if weight.device.index is None else weight.device.index
    return bool(torch.ops._xpu_C.is_xe2_arch(device_index)
                or torch.ops._xpu_C.is_xe3_arch(device_index))


def _to_n_major(scales):
    """Relayout an [E, N, G] scale surface so that N is the dense extent."""
    return scales.transpose(-1, -2).contiguous().transpose(-1, -2)


def _to_xe2_layout(weight, scales):
    """Checkpoint layout -> what csrc/xpu/grouped_gemm/xe_2 indexes.

    4-bit weights stay packed as [E, N, K // 2] alongside their [E, N,
    K // group] scales; every wider dtype is read as a contiguous [E, K, N]
    buffer. mx block scales are taken as loaded (the kernel normalizes either
    orientation on the host), but block-fp8 scales have to be K-major.
    """
    if _is_packed_4bit(weight):
        return weight.contiguous(), scales
    if scales is not None and scales.ndim == 3 \
            and scales.dtype == torch.float32:
        # Block-FP8: [E, ceil(N/128), ceil(K/128)] -> [E, K/128, N/128].
        scales = scales.transpose(-1, -2).contiguous()
    return weight.transpose(-1, -2).contiguous(), scales


def _to_xe3_layout(weight, scales):
    """Checkpoint layout -> what csrc/xpu/grouped_gemm/xe_3 indexes.

    Weights follow the same rule as XE2, but the mx block scales have to be
    e8m0 and dense along N. Block-fp8 ([E, ceil(N/128), ceil(K/128)] fp32) and
    per-tensor scales are already in the expected layout.
    """
    if _is_packed_4bit(weight):
        weight = weight.contiguous()
    else:
        weight = weight.transpose(-1, -2).contiguous()
    if scales is not None and scales.ndim == 3 \
            and scales.dtype in (torch.uint8, torch.float8_e8m0fnu):
        if scales.dtype == torch.uint8:
            scales = scales.view(torch.float8_e8m0fnu)
        scales = _to_n_major(scales)
    return weight, scales


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
        num_experts=num_experts)


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
    elif activation == "situ":
        torch.ops._C.situ_and_mul(
            act_output,
            gemm1_output,
            4.0,
            25.0,
        )
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
        is_fp8=None,
        is_int4=None,
        is_mxfp4=None,
        is_mxfp8=None,
        is_block_fp8=None,
        gemm1_clamp_limit: Optional[float] = None,
        activation_situ_beta: Optional[float] = None,
        activation_situ_linear_beta: Optional[float] = None,
    ):
        # Infer the quantization recipe from the weight/scale dtypes. Explicit
        # is_* arguments (used by the XE3 tests) still take precedence.
        (d_fp8, d_int4, d_mxfp4, d_mxfp8,
         d_block_fp8) = _get_weights_dtype(w13, w13_scales)
        is_fp8 = d_fp8 if is_fp8 is None else is_fp8
        is_int4 = d_int4 if is_int4 is None else is_int4
        is_mxfp4 = d_mxfp4 if is_mxfp4 is None else is_mxfp4
        is_mxfp8 = d_mxfp8 if is_mxfp8 is None else is_mxfp8
        is_block_fp8 = d_block_fp8 if is_block_fp8 is None else is_block_fp8

        self._use_ref = _should_use_ref_fused_moe(is_mxfp8, is_block_fp8)
        # Only the xe_3 grouped GEMM consumes block-scaled activations; xe_2
        # takes high-precision A, so its activation scales are folded in on
        # the Python side below.
        self._uses_xe2 = _uses_xe2_grouped_gemm(w13)

        # Weights and scales are taken in the layout vLLM loads them in --
        # [E, N, K] with K packed for 4-bit, plus [E, N, K // group] scales --
        # and are rewritten in place into whatever this device's grouped GEMM
        # indexes, so callers never have to know the kernel layouts. The mark
        # keeps rebuilding over the same parameters a no-op. ref_fused_moe
        # consumes the loaded layout directly, so it skips the rewrite.
        if not self._use_ref and not hasattr(w13, "xpu_fused_moe"):
            if is_int4:
                # change u4 to s4 to avoid zero point in gemm kernel
                # The buffer must be int8: the grouped GEMM detects packed
                # int4 weights via `B_dtype == at::kChar` (see
                # csrc/xpu/grouped_gemm/xe_2/grouped_gemm_xe2_interface.hpp),
                # and only then reads them as [E, N, K // 2].
                w13_tmp = torch.empty_like(w13, dtype=torch.int8)
                w2_tmp = torch.empty_like(w2, dtype=torch.int8)
                for i in range(num_experts):
                    w13_tmp[i] = implement_zp(w13[i])
                    w2_tmp[i] = implement_zp(w2[i])
                w13.data = w13_tmp.contiguous()
                w2.data = w2_tmp.contiguous()

            to_kernel_layout = _to_xe2_layout \
                if self._uses_xe2 else _to_xe3_layout
            w13_data, w13_scales_data = to_kernel_layout(w13, w13_scales)
            w2_data, w2_scales_data = to_kernel_layout(w2, w2_scales)
            w13.data = w13_data
            w2.data = w2_data
            if w13_scales is not None:
                w13_scales.data = w13_scales_data
            if w2_scales is not None:
                w2_scales.data = w2_scales_data
            w13.xpu_fused_moe = True

        # 4bits support [E, N, K]
        # other types [E, K, N]
        if self._use_ref or is_int4 or is_mxfp4:
            self.inter_size = w13.shape[-2] // 2
        else:
            self.inter_size = w13.shape[-1] // 2

        assert w13.is_contiguous() and w2.is_contiguous()

        self.w13 = w13
        self.w2 = w2

        if not is_fp8 \
            and not is_int4 \
            and not is_mxfp4 \
            and not is_block_fp8 \
            and not is_mxfp8:
            self.gemm1_wei_scales = None
            self.gemm2_wei_scales = None
        else:
            self.gemm1_wei_scales = w13_scales
            self.gemm2_wei_scales = w2_scales

        self.w13_bias = w13_bias
        self.w2_bias = w2_bias

        self.n_experts_per_token = n_experts_per_token
        self.activation = activation
        self.activation_situ_beta = activation_situ_beta
        self.activation_situ_linear_beta = activation_situ_linear_beta
        if self.activation == "situ":
            import math

            if self.activation_situ_beta is None:
                raise ValueError("SITU requires activation_situ_beta")
            if not math.isfinite(self.activation_situ_beta):
                raise ValueError("SITU activation_situ_beta must be finite")
            if self.activation_situ_beta <= 0:
                raise ValueError("SITU activation_situ_beta must be positive")
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
        elif self.activation == "situ":
            self.act_func = torch.ops._C.situ_and_mul
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
        a1q_scale=None,
        a2_scale=None,
    ):
        if self._use_ref:
            self._apply_ref(output, hidden_states,
                            topk_weights, topk_ids,
                            expert_map, a1q_scale,
                            a2_scale)
        else:
            self._apply_kernel(output, hidden_states,
                               topk_weights, topk_ids,
                               expert_map, a1q_scale,
                               a2_scale)

    def _apply_ref(
        self,
        output,
        hidden_states,
        topk_weights,
        topk_ids,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
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
                            a1q_scale=a1q_scale,
                            a2_scale=a2_scale,
                            gemm1_clamp_limit=self.gemm1_clamp_limit)

    def _apply_kernel(
        self,
        output,
        hidden_states,
        topk_weights,
        topk_ids,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
    ):
        num_rows, hidden_size = hidden_states.shape
        num_moe_inputs = self.n_experts_per_token * num_rows
        act_quant = a1q_scale is not None
        # Per-tensor fp8 uses a single global scalar that is invariant under
        # the row permutation/duplication, so it is passed straight through
        # instead of being remapped.
        per_tensor_scale = act_quant and a1q_scale.ndim <= 1

        # mxfp4 activations are packed two values per byte, so the stored
        # hidden dim is half of the logical contraction (gemm1 K) / output
        # (gemm2 N) dim expected by the grouped GEMM. mxfp4_fp8 (W4A8) keeps
        # the activation as unpacked mxfp8, so its hidden dim is already
        # logical and must NOT be doubled.
        # xe_2 dequantizes block-scaled activations below, so A stays
        # unpacked there and the logical hidden size is unchanged.
        native_act_scale = act_quant and not self._uses_xe2
        act_is_packed_mxfp4 = native_act_scale and self.recipe == "mxfp4"
        gemm_hidden_size = 2 * hidden_size \
            if act_is_packed_mxfp4 else hidden_size

        if expert_map is None and self.ep_size > 1:
            expert_map = self.expert_map

        # mx recipes need the activation scales in the per-expert MN-major,
        # M-padded-to-4 surface the mxfp grouped-GEMM mainloop indexes. Both
        # producers of that surface (the remap kernel for gemm1, the act quant
        # kernel for gemm2) write it directly, so no reorder pass is needed.
        mx_act_scale = (native_act_scale and not per_tensor_scale
                        and a1q_scale.dtype == torch.float8_e8m0fnu)
        padded_scale_rows = mxfp_scale_padded_rows(num_moe_inputs,
                                                   self.num_experts)
        if act_quant and not per_tensor_scale:
            alloc = torch.zeros if mx_act_scale else torch.empty
            remapped_scales = alloc(
                (padded_scale_rows if mx_act_scale else num_moe_inputs,
                 a1q_scale.shape[1]),
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
        # Per-expert row prefixes of the padded scale surface. remap computes
        # them anyway, and publishing them lets gemm2's quant kernel emit that
        # surface directly instead of running a reorder pass.
        expert_scale_desc = torch.empty(
            (2, self.num_experts + 1),
            dtype=torch.int32,
            device=hidden_states.device) if mx_act_scale else None

        torch.ops._moe_C.remap_hidden_states(
            hidden_states=hidden_states,
            hidden_states_scales=None if per_tensor_scale else a1q_scale,
            remapped_hidden_states=remapped_hidden_states,
            remapped_hidden_states_scales=remapped_scales,
            expert_map=expert_map,
            rows_per_expert=rows_per_expert,
            unpermuted_row_to_permuted_row=unpermuted_row_to_permuted_row,
            topk_ids=topk_ids,
            total_experts_num=self.total_experts_num,
            local_experts_num=self.local_experts_num,
            expert_scale_desc=expert_scale_desc)

        # xe_2 grouped GEMM runs W8A16 / W16A16: fold the activation scales
        # into A so the kernel sees high-precision activations.
        if self._uses_xe2 and remapped_scales is not None:
            if self.is_mxfp8:
                remapped_hidden_states = dequant_mxfp8(
                    remapped_hidden_states, remapped_scales).to(output.dtype)
                remapped_scales = None
            elif self.is_block_fp8:
                remapped_hidden_states = dequant_fp8_block_act(
                    remapped_hidden_states, remapped_scales).to(output.dtype)
                remapped_scales = None

        ########### gemm1 ##################
        # The grouped-GEMM writes ptr_D as its policy ElementOutput, so the
        # scratch buffers must match that dtype rather than the caller's
        # output dtype. Every policy emits bf16 except the fp16 one (see
        # csrc/xpu/grouped_gemm/xe_3/collective/moe_dtype_policy.hpp).
        gemm_output_dtype = (torch.float16 if output.dtype == torch.float16
                             else torch.bfloat16)
        gemm1_output = torch.empty((num_moe_inputs, 2 * self.inter_size),
                                dtype=gemm_output_dtype,
                                device=output.device)

        # remapped_scales is already in the layout the GEMM expects (mx
        # recipes get the padded MN-major surface straight out of remap).
        gemm1_act_scale = a1q_scale if per_tensor_scale else remapped_scales
        torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=remapped_hidden_states,
            ptr_A_scale=gemm1_act_scale,
            ptr_B=self.w13,
            ptr_B_scale=self.gemm1_wei_scales,
            ptr_bias=self.w13_bias,
            ptr_D=gemm1_output,
            rows_per_expert=rows_per_expert,
            N=2 * self.inter_size,
            K=gemm_hidden_size,
            num_experts=self.num_experts)

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
        if self.activation == "situ":
            self.act_func(
                act_output,
                gemm1_output,
                self.activation_situ_beta,
                -1.0 if self.activation_situ_linear_beta is None else
                self.activation_situ_linear_beta,
            )
        else:
            self.act_func(act_output, gemm1_output)

        ########### gemm2 ##################
        gemm2_output = torch.empty((num_moe_inputs, gemm_hidden_size),
                                dtype=gemm_output_dtype,
                                device=output.device)
        if act_quant:
            # mx recipes get the padded MN-major surface straight out of the
            # quant kernel: the rows are already grouped per expert here, so
            # the descriptor tells it where each scale belongs.
            act_output, gemm2_act_scale = quant_act_xpu(
                act_output, self.recipe, expert_scale_desc, padded_scale_rows,
                static_scale=a2_scale)
            if self._uses_xe2:
                if self.is_mxfp8:
                    act_output = dequant_mxfp8(
                        act_output, gemm2_act_scale).to(output.dtype)
                    gemm2_act_scale = None
                elif self.is_block_fp8:
                    act_output = dequant_fp8_block_act(
                        act_output, gemm2_act_scale).to(output.dtype)
                    gemm2_act_scale = None
            assert (expert_scale_desc is not None or gemm2_act_scale is None
                    or gemm2_act_scale.dtype != torch.float8_e8m0fnu), \
                "mx activation scales must come from the fused layout path"
        else:
            gemm2_act_scale = None
        torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=act_output,
            ptr_A_scale=gemm2_act_scale,
            ptr_B=self.w2,
            ptr_B_scale=self.gemm2_wei_scales,
            ptr_bias=self.w2_bias,
            ptr_D=gemm2_output,
            rows_per_expert=rows_per_expert,
            N=gemm_hidden_size,
            K=self.inter_size * self.inter_size_scale,
            num_experts=self.num_experts)

        if gemm2_output.dtype != output.dtype:
            gather_output = torch.empty_like(output,
                                             dtype=gemm2_output.dtype)
            torch.ops._moe_C.moe_gather(gather_output, gemm2_output,
                                        topk_weights,
                                        unpermuted_row_to_permuted_row,
                                        self.num_experts)
            output.copy_(gather_output)
        else:
            torch.ops._moe_C.moe_gather(output, gemm2_output, topk_weights,
                                        unpermuted_row_to_permuted_row,
                                        self.num_experts)

def quant_fp8_act(x: torch.Tensor):
    """
    Quantize FP32 tensor to block-wise scaled FP8 (1x128 blocks)
    Args:
        x: FP32 tensor, shape [..., K]
    Returns:
        q: FP8 tensor, same shape as x
        scales: FP32 tensor, shape [..., ceil(K / block_size)]
    """
    block_size = 128
    fp8_dtype = torch.float8_e4m3fn
    assert x.dtype == torch.float32
    assert x.shape[-1] % block_size == 0, \
        "Last dim must be divisible by block_size for simplicity"

    orig_shape = x.shape
    K = orig_shape[-1]
    num_blocks = K // block_size

    # reshape to [..., num_blocks, block_size]
    x_blocks = x.view(*orig_shape[:-1], num_blocks, block_size)

    # absmax per block
    absmax = x_blocks.abs().amax(dim=-1)  # [..., num_blocks]

    # avoid div-by-zero
    eps = torch.finfo(torch.float32).eps
    absmax = torch.clamp(absmax, min=eps)

    # fp8 max
    fp8_max = 448.0

    # scale
    scales = absmax / fp8_max  # FP32

    # scale + cast
    x_scaled = x_blocks / scales.unsqueeze(-1)
    q = x_scaled.to(fp8_dtype)

    # reshape back
    q = q.view(orig_shape)

    return q, scales


def quant_mxfp_act(x, recipe):
    from tests.fused_moe.test_grouped_gemm_xe3 import (
        bfloat16_to_fp4_e2m1fn_x2, data_to_mx_scale)

    # max value of `torch.float8_e4m3fn` (448)
    F8E4M3_MAX_VAL = torch.finfo(torch.float8_e4m3fn).max
    # exponent and mantissa bits of `torch.float4_e2m1fn_x2`
    FP4_MAX_VAL = 6.0

    if recipe == "mxfp8":
        max_val = F8E4M3_MAX_VAL
        min_val = -1 * max_val
    elif recipe == "mxfp4":
        max_val = FP4_MAX_VAL
        min_val = -1 * max_val

    ori_shape = x.shape
    x_scale = data_to_mx_scale(x.to(torch.bfloat16), 32,
                               recipe)  # (m, scale_k)
    if recipe == "mxfp8":
        x = (x.to(torch.bfloat16).reshape(-1, 32) /
             x_scale.reshape(-1, 1).float()).reshape(ori_shape)
        x = x.clamp(min=min_val, max=max_val).to(torch.float8_e4m3fn)
    elif recipe == "mxfp4":
        x = (x.to(torch.bfloat16).reshape(-1, 32) /
             x_scale.reshape(-1, 1).bfloat16()).reshape(ori_shape)
        x = x.clamp(min=min_val, max=max_val)
        x = bfloat16_to_fp4_e2m1fn_x2(x)
    return x, x_scale

