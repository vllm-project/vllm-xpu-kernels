# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch

from . import _C  # noqa: F401
from . import _moe_C  # noqa: F401
from . import _xpu_C  # noqa: F401

finfo = torch.finfo(torch.float8_e4m3fn)
FP8_E4M3_MIN = finfo.min
FP8_E4M3_MAX = finfo.max
    
_FP4_E2M1_LUT = torch.tensor(
    [
         0.0,  0.5,  1.0,  1.5,
         2.0,  3.0,  4.0,  6.0,
        -0.0, -0.5, -1.0, -1.5,
        -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)

_FP4_E2M1_LUT_XPU = None

def _get_lut(device):
    global _FP4_E2M1_LUT_XPU
    if _FP4_E2M1_LUT_XPU is None:
        _FP4_E2M1_LUT_XPU = _FP4_E2M1_LUT.to(device)
    return _FP4_E2M1_LUT_XPU

def _fp4_e2m1fn_x2_to_float_lut(
    packed: torch.Tensor,
) -> torch.Tensor:
    """
    packed:
        float4_e2m1fn_x2 tensor
        shape [..., N]

    return:
        fp32 tensor
        shape [..., N*2]
    """

    lut = _get_lut(packed.device)

    u8 = packed.view(torch.uint8)

    lo = u8 & 0xF
    hi = u8 >> 4

    lo_fp = lut[lo.long()]
    hi_fp = lut[hi.long()]

    out = torch.empty(
        *u8.shape[:-1],
        u8.shape[-1] * 2,
        device=u8.device,
        dtype=torch.float32,
    )

    out[..., 0::2] = lo_fp
    out[..., 1::2] = hi_fp

    return out

def dequant_mxfp4(x_lp, x_scale):
    ori_shape = x_lp.shape
    x = _fp4_e2m1fn_x2_to_float_lut(x_lp).reshape(-1, 32) * (x_scale.reshape(
            -1, 1).to(torch.float32))
    return x.reshape(ori_shape[:-1] + (ori_shape[-1] * 2, ))

def dequant_mxfp8(x_lp, x_scale):
    ori_shape = x_lp.shape
    x = x_lp.to(torch.float32).reshape(-1, 32) * \
        (x_scale.reshape(-1, 1).to(torch.float32))
    return x.reshape(ori_shape)

def quant_mxfp_act_xpu(x, recipe, expert_scale_desc=None, scale_rows=0):
    assert recipe in ("mxfp8", "mxfp4")
    if recipe == "mxfp8":
        return _quant_mxfp8_act_xpu(x, expert_scale_desc, scale_rows)
    else:
        return _quant_mxfp4_act_xpu(x, expert_scale_desc, scale_rows)


def _alloc_mx_act_scale(x, scale_k, expert_scale_desc, scale_rows):
    # Without a MoE descriptor the kernel writes float32 scales that the
    # caller converts to e8m0. With one it writes the padded MN-major e8m0
    # surface the mxfp grouped-GEMM mainloop indexes directly, which removes
    # both the conversion pass and the separate reorder pass.
    if expert_scale_desc is None:
        return torch.empty(x.shape[:-1] + (scale_k, ),
                           device=x.device,
                           dtype=torch.float32)
    return torch.zeros((scale_rows, scale_k),
                       device=x.device,
                       dtype=torch.float8_e8m0fnu)


def _quant_mxfp8_act_xpu(x, expert_scale_desc=None, scale_rows=0):
    MXFP8_BLOCK_SIZE = 32
    assert x.shape[-1] % MXFP8_BLOCK_SIZE == 0

    eps = 1e-10
    x_q = torch.empty_like(x, device=x.device, dtype=torch.float8_e4m3fn)
    x_s = _alloc_mx_act_scale(x, x.shape[-1] // MXFP8_BLOCK_SIZE,
                              expert_scale_desc, scale_rows)
    torch.ops._C.per_token_group_fp8_quant(
        x,
        x_q,
        x_s,
        MXFP8_BLOCK_SIZE,
        eps,
        FP8_E4M3_MIN,
        FP8_E4M3_MAX,
        True,
        False,
        False,  # dummy_is_scale_transposed, dummy_is_tma_aligned
        expert_scale_desc,
    )
    if expert_scale_desc is None:
        x_s = x_s.to(torch.float8_e8m0fnu)
    return x_q, x_s

def _quant_mxfp4_act_xpu(x, expert_scale_desc=None, scale_rows=0):
    MXFP4_BLOCK_SIZE = 32
    eps = 1e-10
    M, N = x.shape
    # Packed FP4 output: two nibbles per byte
    x_q = torch.empty(M, N // 2, device=x.device, dtype=torch.uint8)
    x_s = _alloc_mx_act_scale(x, N // MXFP4_BLOCK_SIZE, expert_scale_desc,
                              scale_rows)

    torch.ops._C.per_token_group_quant_mxfp4(x, x_q, x_s, MXFP4_BLOCK_SIZE,
                                             eps, expert_scale_desc)

    x_q = x_q.view(torch.float4_e2m1fn_x2)
    if expert_scale_desc is None:
        x_s = x_s.to(dtype=torch.float8_e8m0fnu,
                     memory_format=torch.preserve_format)
    return x_q, x_s

def qdq_fp8_act(x):
    x_fp = x.to(torch.float32)
    scale = (x_fp.abs().max() / FP8_E4M3_MAX).clamp(
            min=torch.finfo(torch.float32).eps
        )
    return (x_fp / scale).clamp(
            -FP8_E4M3_MAX, FP8_E4M3_MAX
        ).to(torch.float8_e4m3fn).to(torch.float32) * scale

def dequant_fp8_block_wei(x_lp, x_scale):
    orig_shape = x_lp.shape
    M, K = orig_shape
    x_lp = x_lp.view(M // 128, 128, K // 128, 128)
    x_scale = x_scale.unsqueeze(1).unsqueeze(-1)
    x_hp = x_lp.to(torch.float32)
    x_hp = x_hp * x_scale
    return x_hp.reshape(orig_shape).to(torch.float32)

def dequant_fp8_block_act(x_lp, x_scale):
    orig_shape = x_lp.shape
    x_lp = x_lp.reshape(x_lp.shape[0], x_lp.shape[-1] // 128, 128)
    x_hp = x_lp.to(torch.float32)
    x_hp = x_hp * x_scale.unsqueeze(-1)
    return x_hp.reshape(orig_shape).to(torch.float32)

def quant_fp8_block_act(x: torch.Tensor):
    x_q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)    
    shape = x.shape[:-1] + (x.shape[-1] // 128,)
    x_s = torch.empty(shape, device=x.device, dtype=torch.float32)
    torch.ops._C.per_token_group_fp8_quant(
            x,
            x_q,
            x_s,
            128,
            1e-10,
            FP8_E4M3_MIN,
            FP8_E4M3_MAX,
            False,
            False,
            False,
        )
    return x_q, x_s

def _as_e8m0(s):
        """Reinterpret uint8 scale bits as float8_e8m0fnu for correct
        conversion to float32 (2^(e-127)).  float8_e8m0fnu tensors are
        returned unchanged."""
        if s.dtype == torch.uint8:
            return s.view(torch.float8_e8m0fnu)
        return s

def dequant_act(x, x_scale, recipe):
    if recipe == "fp8block":
        return dequant_fp8_block_act(x, x_scale)
    elif recipe == "fp8":
        return x.to(torch.float32) * _as_e8m0(x_scale).to(torch.float32)
    elif recipe == "mxfp4":
        return dequant_mxfp4(x, x_scale)
    elif recipe in ("mxfp8", "mxfp4_fp8"):
        # W4A8 (mxfp4_fp8): mxfp4 weights + mxfp8 activations. The activation
        # is e4m3 with an e8m0 per-32-block scale, i.e. dequantized exactly
        # like the mxfp8 recipe.
        return dequant_mxfp8(x, x_scale)
    else:
        # bf16: no quantization noise, return unchanged
        return x

def qdq_act(x, recipe):
    if recipe == "fp8block":
        _q, _s = quant_fp8_block_act(x)
        return dequant_fp8_block_act(_q, _s)
    elif recipe == "fp8":
        return qdq_fp8_act(x)
    elif recipe == "mxfp4":
        _aq, _as = quant_mxfp_act_xpu(x, "mxfp4")
        return dequant_mxfp4(_aq, _as)
    elif recipe in ("mxfp8", "mxfp4_fp8"):
        # W4A8 (mxfp4_fp8): the activation is quantized to mxfp8 (e4m3 + e8m0
        # per-32-block scale), matching the XE3 W4A8 grouped-GEMM kernel.
        _aq, _as = quant_mxfp_act_xpu(x, "mxfp8")
        return dequant_mxfp8(_aq, _as)
    else:
        # bf16: no quantization noise, return unchanged
        return x

def dequant_wei(wei, wei_scale, recipe):
    if recipe in ("mxfp4", "mxfp4_fp8"):
        return dequant_mxfp4(wei, _as_e8m0(wei_scale))
    elif recipe == "mxfp8":
        return dequant_mxfp8(wei, _as_e8m0(wei_scale))
    elif recipe == "fp8block":
        return dequant_fp8_block_wei(wei, wei_scale)
    elif recipe == "fp8":
        return wei.float() * wei_scale.float()
    else:
        # bf16: weights are already in compute dtype
        return wei


def ref_fused_moe_activation(act_output, gemm1_output, activation):
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


def ref_fused_moe(recipe,
                  output,
                  hidden_states,
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
                  a1q_scale=None,
                  a2_scale=None,
                  gemm1_clamp_limit: Optional[float] = None,
):
    """
    Reference fused MoE implementation with quantization simulation.

    Weights and scales are taken in the loaded [E, N, K] / [E, N, K // group]
    layout, not in any grouped-GEMM layout.

    Supported recipes:
        bf16          - no quantization (direct matmul)
        fp8block      - block-wise fp8 quant/dequant on activations and weights
        mxfp8         - mxfp8 (per-32-element group)
        mxfp4         - mxfp4 (per-32-element group)
        mxfp4_fp8     - mxfp4 weights + mxfp8 activations (W4A8; e4m3 activation
                        with e8m0 per-32-block scale, matching the XE3 W4A8
                        grouped-GEMM kernel)
        fp8           - per-tensor fp8 quant/dequant on activations

    NOT supported (raise NotImplementedError):
        int4

    Dimension constraints per recipe:
        fp8block: hidden_size % 128 == 0 (act quant group=128)
        mxfp8:    hidden_size % 32 == 0  (act quant block=32)
        mxfp4:    hidden_size % 32 == 0  (act quant block=32)
                  additionally, per-expert intermediate activations must satisfy
                  n_tokens * inter_per_card % 32 == 0 at runtime.
    """
    assert recipe in ("bf16", "fp8block", "mxfp8", "mxfp4", \
        "mxfp4_fp8", "fp8"), f"Unsupported recipe: {recipe}"
    
    num_rows, hidden_size = hidden_states.shape
    inter_size = w13.shape[-2] // 2
    num_moe_inputs = n_experts_per_token * num_rows
    compute_dtype = hidden_states.dtype if a1q_scale is None else torch.bfloat16

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

    

    # ---- remap hidden states (unchanged from _apply_kernel) ----
    per_tensor_scale = a1q_scale is not None and a1q_scale.numel() == 1
    if a1q_scale is not None and not per_tensor_scale:
        remapped_scales = torch.empty(
                (num_rows * n_experts_per_token, a1q_scale.shape[1]),
                dtype=a1q_scale.dtype,
                device=a1q_scale.device)
    else:
        remapped_scales = None
    remapped_hidden_states = torch.empty(
        (num_moe_inputs, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device)
    rows_per_expert = torch.zeros(num_experts,
                                  dtype=torch.int32,
                                  device=hidden_states.device)
    unpermuted_row_to_permuted_row = torch.empty(
        (num_rows, n_experts_per_token),
        dtype=torch.int32,
        device=hidden_states.device)

    torch.ops._moe_C.remap_hidden_states(
        hidden_states=hidden_states,
        hidden_states_scales=None if per_tensor_scale else a1q_scale,
        remapped_hidden_states=remapped_hidden_states,
        remapped_hidden_states_scales=remapped_scales,
        expert_map=expert_map,
        rows_per_expert=rows_per_expert,
        unpermuted_row_to_permuted_row=unpermuted_row_to_permuted_row,
        topk_ids=topk_ids,
        total_experts_num=total_experts_num,
        local_experts_num=local_experts_num)

    # mxfp4 packs two activation values per byte, so the stored hidden dim is
    # half the logical size and must be doubled here. mxfp4_fp8 (W4A8) keeps
    # the activation as unpacked mxfp8 (e4m3), so its hidden dim is already the
    # logical size and must NOT be doubled.
    if a1q_scale is not None and recipe == "mxfp4":
        hidden_size = 2 * hidden_size

    # ---- GEMM1: cutlass grouped GEMM replaced by torch matmul ----
    gemm1_output = torch.zeros((num_moe_inputs, 2 * inter_size),
                               dtype=compute_dtype,
                               device=hidden_states.device)
    offset = 0
    for i in range(num_experts):
        n_tokens = rows_per_expert[i].item()
        if n_tokens == 0:
            continue
        tokens_i = remapped_hidden_states[offset:offset + n_tokens]

        # activation: quant → dequant round-trip
        if per_tensor_scale:
            # Per-tensor: dequant with the single global scale
            # (keep the scale on-device)
            tokens_i_qdq = (tokens_i.to(torch.float32)
                            * a1q_scale.to(torch.float32)).to(compute_dtype)
        elif a1q_scale is not None:
            tokens_i_qdq = dequant_act(
                tokens_i,
                remapped_scales[offset:offset + n_tokens],
                recipe).to(compute_dtype)
        else:
            tokens_i_qdq = qdq_act(tokens_i, recipe).to(compute_dtype)
        # weight dequant
        w13_scales_i = None if w13_scales is None else w13_scales[i]
        w13_i = dequant_wei(w13[i], w13_scales_i, recipe).to(compute_dtype)
        out_i = tokens_i_qdq @ w13_i.T
        if w13_bias is not None:
            out_i = out_i + w13_bias[i].to(compute_dtype)
        gemm1_output[offset:offset + n_tokens] = out_i
        offset += n_tokens

    # Apply swiglu_limit clamping before activation
    if gemm1_clamp_limit is not None and gemm1_clamp_limit > 0:
        gemm1_output[:, :inter_size].clamp_(max=gemm1_clamp_limit)
        gemm1_output[:, inter_size:].clamp_(min=-gemm1_clamp_limit,
                                            max=gemm1_clamp_limit)

    # ---- activation (unchanged from _apply_kernel) ----
    inter_size_scale = 2 if activation == "relu2_no_mul" else 1
    act_output = torch.empty(
        (num_moe_inputs, inter_size * inter_size_scale),
        dtype=compute_dtype,
        device=hidden_states.device)
    ref_fused_moe_activation(act_output, gemm1_output, activation)

    # ---- GEMM2: cutlass grouped GEMM replaced by torch matmul ----
    gemm2_output = torch.zeros((num_moe_inputs, hidden_size),
                               dtype=compute_dtype,
                               device=hidden_states.device)
    offset = 0
    for i in range(num_experts):
        n_tokens = rows_per_expert[i].item()
        if n_tokens == 0:
            continue
        act_i = act_output[offset:offset + n_tokens]

        # activation: quant → dequant round-trip
        if a2_scale is not None:
            # Static per-tensor: quantize with static scale, then dequant
            act_i_qdq = ((act_i.float() / a2_scale.float()).clamp(
                FP8_E4M3_MIN, FP8_E4M3_MAX
            ).to(torch.float8_e4m3fn).float() * a2_scale.float()).to(
                compute_dtype)
        else:
            act_i_qdq = qdq_act(act_i, recipe).to(compute_dtype)

        # weight dequant
        w2_scales_i = None if w2_scales is None else w2_scales[i]
        w2_i = dequant_wei(w2[i], w2_scales_i, recipe).to(compute_dtype)
        out_i = act_i_qdq @ w2_i.T
        if w2_bias is not None:
            out_i = out_i + w2_bias[i].to(compute_dtype)
        gemm2_output[offset:offset + n_tokens] = out_i
        offset += n_tokens

    # ---- moe_gather (unchanged from _apply_kernel) ----
    torch.ops._moe_C.moe_gather(output, gemm2_output, topk_weights,
                                unpermuted_row_to_permuted_row,
                                num_experts)
    return output

def quant_fp8_pertensor_act(x: torch.Tensor):
    """Dynamic per-tensor FP8 quantization (single global scale)."""
    x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty(1, device=x.device, dtype=torch.float32)
    torch.ops._C.dynamic_scaled_fp8_quant(x_fp8, x, scale)
    return x_fp8, scale


def quant_fp8_static_pertensor_act(x: torch.Tensor,
                                    static_scale: torch.Tensor):
    """Static per-tensor FP8 quantization using pre-computed scale."""
    x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = static_scale.float().reshape(1)
    torch.ops._C.static_scaled_fp8_quant(x_fp8, x, scale, None)
    return x_fp8, scale


def quant_act_xpu(x,
                  recipe,
                  expert_scale_desc=None,
                  scale_rows=0,
                  static_scale=None):
    """Quantize MoE activations.

    ``expert_scale_desc`` is the int32 ``[2, num_experts + 1]`` tensor that
    ``remap_hidden_states`` publishes. When given, the mx kernels write the
    per-expert MN-major, M-padded-to-4 scale surface (``scale_rows`` tall)
    that the grouped GEMM indexes directly. The rows of ``x`` must already be
    grouped per expert, as they are between the two grouped GEMMs, which is
    what lets the quant kernel place each scale at its final position and
    makes the reorder pass unnecessary.
    """
    if recipe in ("mxfp4", "mxfp8"):
        return quant_mxfp_act_xpu(x, recipe, expert_scale_desc, scale_rows)
    elif recipe == "mxfp4_fp8":
        # W4A8: mxfp4 weights + mxfp8 activations. The activation is quantized
        # to mxfp8 (e4m3 + e8m0 per-32-block scale) to match the XE3 W4A8
        # grouped-GEMM kernel (see PR #165).
        return quant_mxfp_act_xpu(x, "mxfp8", expert_scale_desc, scale_rows)
    elif recipe == "fp8block":
        return quant_fp8_block_act(x)
    elif recipe == "fp8":
        if static_scale is not None:
            return quant_fp8_static_pertensor_act(x, static_scale)
        return quant_fp8_pertensor_act(x)
    else:
        raise NotImplementedError(f"Unsupported recipe for quant_act_xpu: {recipe}") # noqa: E501
    
def mxfp_scale_padded_rows(num_rows, num_experts):
    # After cutlass-sycl PR #570, the optimized mxfp mainloop requires the
    # per-expert scale-A surface width (M dim, since scale is MN-major) to be
    # a multiple of 4 (ScaleAlignElems for 8-bit scales). Each expert's M is
    # rounded up to a multiple of 4, so the worst case adds 3 rows per expert.
    return num_rows + 3 * num_experts


def reorder_mxfp_scales(A_scales, rows_per_expert):
    # Pad each expert's M up to a multiple of 4 with zeros so the cumulative
    # scale offsets used by the grouped-gemm kernel (sum of round_up_4(rows))
    # remain aligned.
    num_experts = rows_per_expert.shape[0]
    total_padded = mxfp_scale_padded_rows(A_scales.shape[0], num_experts)
    return torch.ops._moe_C.reorder_mxfp_scales(A_scales, rows_per_expert,
                                                total_padded)
