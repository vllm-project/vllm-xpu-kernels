# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from typing import Optional
import vllm_xpu_kernels._C  # noqa: F401
import vllm_xpu_kernels._moe_C  # noqa: F401


# layer norm ops
def rms_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
             epsilon: float) -> None:
    # TODO: Remove this contiguous call when the kernel is updated to support
    # non-contiguous input
    input_contiguous = input.contiguous()
    torch.ops._C.rms_norm(out, input_contiguous, weight, epsilon)


def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, epsilon: float) -> None:
    torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)


def silu_and_mul(out: torch.Tensor, input: torch.Tensor) -> None:
    torch.ops._C.silu_and_mul(out, input)


def gelu_fast(out: torch.Tensor, input: torch.Tensor) -> None:
    torch.ops._C.gelu_fast(out, input)


def gelu_new(out: torch.Tensor, input: torch.Tensor) -> None:
    torch.ops._C.gelu_new(out, input)


def gelu_quick(out: torch.Tensor, input: torch.Tensor) -> None:
    torch.ops._C.gelu_quick(out, input)


def mul_and_silu(out: torch.Tensor, input: torch.Tensor) -> None:
    torch.ops._C.mul_and_silu(out, input)


def gelu_and_mul(out: torch.Tensor, input: torch.Tensor) -> None:
    torch.ops._C.gelu_and_mul(out, input)


def gelu_tanh_and_mul(out: torch.Tensor, input: torch.Tensor) -> None:
    torch.ops._C.gelu_tanh_and_mul(out, input)


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    torch.ops._C.rotary_embedding(positions, query, key, head_size,
                                  cos_sin_cache, is_neox)


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


def concat_and_cache_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.concat_and_cache_mla(kv_c, k_pe, kv_cache,
                                                slot_mapping, kv_cache_dtype,
                                                scale)


def gather_cache(src_cache: torch.Tensor,
                 dst: torch.Tensor,
                 block_table: torch.Tensor,
                 cu_seq_lens: torch.Tensor,
                 batch_size: int,
                 seq_starts: Optional[torch.Tensor] = None) -> None:
    torch.ops._C_cache_ops.gather_cache(src_cache, dst, block_table,
                                        cu_seq_lens, batch_size, seq_starts)


def static_scaled_fp8_quant(out: torch.Tensor, input: torch.Tensor,
                            scale: torch.Tensor) -> None:
    torch.ops._C.static_scaled_fp8_quant(out, input, scale)


def dynamic_scaled_fp8_quant(out: torch.Tensor, input: torch.Tensor,
                             scale: torch.Tensor) -> None:
    torch.ops._C.dynamic_scaled_fp8_quant(out, input, scale)


def dynamic_per_token_scaled_fp8_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    scale_ub: Optional[torch.Tensor] = None,
) -> None:
    torch.ops._C.dynamic_per_token_scaled_fp8_quant(out, input, scales,
                                                    scale_ub)


def swigluoai_and_mul(
    out: torch.Tensor,
    input: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> None:
    """SwigluOAI and Mul activation function."""
    torch.ops._C.swigluoai_and_mul(out, input, alpha, limit)


# moe
def moe_sum(input: torch.Tensor, output: torch.Tensor) -> None:
    torch.ops._moe_C.moe_sum(input, output)


def grouped_topk(scores: torch.Tensor, scores_with_bias: torch.Tensor,
                 num_expert_group: int, topk_group: int, topk: int,
                 renormalize: bool, routed_scaling_factor: float):
    return torch.ops._moe_C.grouped_topk(scores, scores_with_bias,
                                         num_expert_group, topk_group, topk,
                                         renormalize, routed_scaling_factor)


def fused_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: Optional[torch.Tensor] = None,
):
    return torch.ops._moe_C.fused_grouped_topk(hidden_states, gating_output,
                                               topk, renormalize,
                                               num_expert_group, topk_group,
                                               scoring_func,
                                               routed_scaling_factor,
                                               e_score_correction_bias)
